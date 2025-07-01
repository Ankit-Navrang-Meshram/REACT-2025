import torch
import torch.nn as nn
from vqvae import VQVAE
from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from cross_attention import CrossAttention , HistoricalCrossAttention
from resampling_method import LinearInterpolate, Conv1DResampling, AdaptiveAveragePooling, LinearProjectionResampling

class TransformerPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 256
        self.num_layers = args.num_layers if hasattr(args, 'num_layers') else 12
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 8
        self.intermediate_size = args.intermediate_size if hasattr(args, 'intermediate_size') else 384
        self.audio_dim = args.audio_dim if hasattr(args, 'audio_dim') else 1024
        self.conv_kernel_size = args.conv_kernel_size if hasattr(args, 'conv_kernel_size') else 5
        self.leaky_relu_slope = args.leaky_relu_slope if hasattr(args, 'leaky_relu_slope') else 0.2

        self.audio_feature_map = nn.Linear(self.audio_dim, self.latent_dim)

        layers = [nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 
                     self.conv_kernel_size, stride=1, 
                     padding=self.conv_kernel_size//2, 
                     padding_mode='replicate'),
            nn.LeakyReLU(self.leaky_relu_slope, True),
            nn.InstanceNorm1d(self.latent_dim, affine=False)
        )]

        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=self.latent_dim,
            hidden_size=self.latent_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size
        )
        self.encoder_pos_embedding = PositionalEncoding(self.latent_dim, batch_first=True)
        self.encoder_linear_embedding = LinearEmbedding(self.latent_dim, self.latent_dim)

    def forward(self, audio):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.audio_feature_map(audio)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
        return encoder_features

class VqvaePredict(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size if hasattr(args , 'window_size') else 2
        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.in_fps = args.in_fps if hasattr(args, 'in_fps') else 50
        self.out_fps = args.out_fps if hasattr(args, 'out_fps') else 30
        self.vqvae_path = args.vqvae_path if hasattr(args, 'vqvae_path') else None
        self.resampling_method = args.resampling_method if hasattr(args, 'resampling_method') else 'linear_interpolation'
        self.with_style = args.with_style if hasattr(args, 'with_style') else False
        self.audio_encoded_dim = args.audio_encoded_dim if hasattr(args, 'audio_encoded_dim') else 1024
        self.temperature = args.temperature if hasattr(args, 'temperature') else 0.2
        self.k = args.k if hasattr(args, 'k') else 1
        self.style_input_dim = args.style_input_dim if hasattr(args, 'style_input_dim') else 256
        self.style_output_dim = args.style_output_dim if hasattr(args, 'style_output_dim') else 1024
        self.w2v_model = args.w2v_model if hasattr(args, 'w2v_model') else 'facebook/w2v-bert-2.0'
        self.trainable_layers = args.trainable_layers if hasattr(args, 'trainable_layers') else ['encoder.layers.23', 'encoder.layers.22']
        self.device = args.device if hasattr(args, 'device') else 'cuda'
        self.sampling_rate = args.sampling_rate if hasattr(args, 'sampling_rate') else 16000
        self.online = args.online if hasattr(args , 'online') else False
        self.processor = AutoFeatureExtractor.from_pretrained(self.w2v_model)
        self.feature_extractor = Wav2Vec2BertModel.from_pretrained(self.w2v_model)

        for name, params in self.feature_extractor.named_parameters():
            params.requires_grad = any(layer in name for layer in self.trainable_layers)

        # Load motion prior
        self.motion_prior = VQVAE(args)

        if self.vqvae_path is not None:
            print(f"Loading VQVAE weights from {self.vqvae_path}")
            self.motion_prior.load_state_dict(
                torch.load(self.vqvae_path, map_location='cpu', weights_only=False)['model_state_dict'], 
                strict=True
            )
        else:
            print("No VQVAE weights provided, initializing with default weights.")

        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.motion_prior.eval()
        
        self.feature_predictor = TransformerPredictor(args)
 
        self.style_embedding = nn.Linear(self.style_input_dim, self.style_output_dim)
        self.style_ca = CrossAttention()
        self.past_ca = HistoricalCrossAttention()

        # Initialize resampling methods
        self.linear_interpolate_input = LinearInterpolate(args).linear_interpolate_input
        self.adaptive_avg_pooling_resampling = AdaptiveAveragePooling(args).adaptive_avg_pooling_resampling
        self.linear_projection_resampling = LinearProjectionResampling(args).forward
        self.conv1d_resampling = Conv1DResampling(args).forward

    def forward_offline(self, batch, style, speaker_3dmm=None, sample=False, training=True):
        batch = self.processor(batch, return_tensors='pt', sampling_rate=self.sampling_rate)
        batch['input_features'] = batch['input_features'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)

        # Audio feature extraction
        audio_feature = self.feature_extractor(**batch).last_hidden_state

        # Resampling of audio features
        if self.resampling_method == "linear_interpolation":
            audio_feature = self.linear_interpolate_input(audio_feature)
        elif self.resampling_method == "adaptive_avg_pooling_resampling":
            audio_feature = self.adaptive_avg_pooling_resampling(audio_feature)
        elif self.resampling_method == "linear_projection_resampling":
            audio_feature = self.linear_projection_resampling(audio_feature)
        elif self.resampling_method == "conv1d_resampling":
            audio_feature = self.conv1d_resampling(audio_feature)

        if self.with_style:
            style = self.style_embedding(style)
            audio_feature = self.style_ca(audio_feature, style)

        prediction = self.feature_predictor(audio_feature)

        motion_quant_pred, _, _ = self.motion_prior.quantize(
            prediction, 
            sample=sample,
            temperature=self.temperature,
            k=self.k
        )
        
        motion_out, emotion_out = self.motion_prior.motion_decoder(motion_quant_pred)

        if training:
            with torch.no_grad():
                motion_quant, _ = self.motion_prior.get_quant(speaker_3dmm)
            return motion_out, emotion_out, motion_quant_pred, motion_quant
        else:
            return motion_out, emotion_out
    
    def forward_online(self, batch, speaker_3dmm=None, sample=False, training=True, past_motion=None):
        #style = self.style_embedding(style)
        motion_out_list, emotion_out_list, motion_quant_pred_list, motion_quant_list = [], [], [], []
        audio_window = (self.window_size*16000)
        periods = batch[0].shape[0] // audio_window

        if past_motion is None:
            past_quants = torch.zeros(size=(len(batch), self.window_size*self.out_fps, 256)).cuda()
        else:
            past_quants, _ = self.motion_prior.get_quant(past_motion)
        
        for p in range(periods):
            windows_batch = [b[p*audio_window: (p+1)*audio_window] for b in batch]
            windows_batch = self.processor(windows_batch, return_tensors='pt', sampling_rate=16000)
            windows_batch['input_features'] = windows_batch['input_features'].cuda()

            windows_batch['attention_mask'] = windows_batch['attention_mask'].cuda()
            # audio feature extraction
            audio_feature = self.feature_extractor(**windows_batch).last_hidden_state  # list of [B, Ts, 1024]

            # Resampling of audio features
            if self.resampling_method == "linear_interpolation":
                audio_feature = self.linear_interpolate_input(audio_feature)
            elif self.resampling_method == "adaptive_avg_pooling_resampling":
                audio_feature = self.adaptive_avg_pooling_resampling(audio_feature)
            elif self.resampling_method == "linear_projection_resampling":
                audio_feature = self.linear_projection_resampling(audio_feature)
            elif self.resampling_method == "conv1d_resampling":
                audio_feature = self.conv1d_resampling(audio_feature)

            #audio_feature = self.style_ca(audio_feature, style)

            prediction = self.feature_predictor(audio_feature)  # [B, T, 256]


            motion_quant_pred, _, _ = self.motion_prior.quantize(prediction, sample=sample,
                                                                temperature=self.temperature,  # 0.2 by default,
                                                                k=self.k)      # 1 by default,  
            
            motion_quant_pred = self.past_ca(motion_quant_pred, past_quants)

            motion_out, emotion_out = self.motion_prior.motion_decoder(motion_quant_pred)    # [B, T, 53]

            if training:
                with torch.no_grad():
                    motion_quant, _ = self.motion_prior.get_quant(speaker_3dmm[:, p*self.window_size*self.out_fps:(p+1)*self.window_size*self.out_fps, :])

                motion_out_list.append(motion_out)
                emotion_out_list.append(emotion_out)
                motion_quant_pred_list.append(motion_quant_pred)
                motion_quant_list.append(motion_quant)
                # return motion_out, emotion_out, motion_quant_pred, motion_quant # motion_out = predicted_listeners_3dmm , emotion_out = predicted_listeners_emotion, motion_quant_pred = predicted_decoder_input, motion_quant = ground_truth_decoder_input
            else:
                motion_out_list.append(motion_out)
                emotion_out_list.append(emotion_out)
                
            past_quants = motion_quant_pred
        if training:
            return torch.concat(motion_out_list, dim=1), torch.concat(emotion_out_list, dim=1), torch.concat(motion_quant_pred_list, dim=1), torch.concat(motion_quant_list, dim=1)
        else:
            return torch.concat(motion_out_list, dim=1), torch.concat(emotion_out_list, dim=1)


    def forward(self, batch, style, speaker_3dmm=None, sample=False, training=True):
        if self.online:
            return self.forward_online(batch, speaker_3dmm=speaker_3dmm, sample=sample, training=training , past_motion=None)
        return self.forward_offline(batch, style, speaker_3dmm=speaker_3dmm, sample=False, training=True)

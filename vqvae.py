import torch
import torch.nn as nn
from framework.model.base import BaseModel
from framework.model.utils.vqvae_quantizer import VectorQuantizer
from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 256
        self.num_layers = args.num_layers if hasattr(args, 'num_layers') else 6
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 8
        self.intermediate_size = args.intermediate_size if hasattr(args, 'intermediate_size') else 384
        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.conv_kernel_size = args.conv_kernel_size if hasattr(args, 'conv_kernel_size') else 5
        self.leaky_relu_slope = args.leaky_relu_slope if hasattr(args, 'leaky_relu_slope') else 0.2

        self.feature_mapping = nn.Sequential(
            nn.Linear(self.nfeats, self.latent_dim),
            nn.LeakyReLU(self.leaky_relu_slope, True)
        )

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

    def forward(self, motion):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.feature_mapping(motion)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

        return encoder_features

class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 256
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 8
        self.num_layers = args.num_layers if hasattr(args, 'num_layers') else 6
        self.intermediate_size = args.intermediate_size if hasattr(args, 'intermediate_size') else 384
        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.conv_kernel_size = args.conv_kernel_size if hasattr(args, 'conv_kernel_size') else 5
        self.leaky_relu_slope = args.leaky_relu_slope if hasattr(args, 'leaky_relu_slope') else 0.2

        self.expander = nn.ModuleList()

        self.expander.append(nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 
                     self.conv_kernel_size, stride=1, 
                     padding=self.conv_kernel_size//2, 
                     padding_mode='replicate'),
            nn.LeakyReLU(self.leaky_relu_slope, False),
            nn.InstanceNorm1d(self.latent_dim, affine=False)
        ))

        self.decoder_transformer = Transformer(
            in_size=self.latent_dim,
            hidden_size=self.latent_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size
        )
        
        self.decoder_pos_embedding = PositionalEncoding(self.latent_dim)
        self.decoder_linear_embedding = LinearEmbedding(self.latent_dim, self.latent_dim)

        self.feature_mapping_reverse = nn.Linear(self.latent_dim, self.nfeats, bias=False)
        self.feature_mapping_reverse_emotion = nn.Linear(self.latent_dim, self.n_emotion, bias=False)

    def forward(self, inputs):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        for i, module in enumerate(self.expander):
            inputs = module(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            if i > 0:  # this is not used
                inputs = inputs.repeat_interleave(2, dim=1)

        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(decoder_features)
        decoder_features = self.decoder_transformer((decoder_features, dummy_mask))

        pred_recons = self.feature_mapping_reverse(decoder_features)
        pred_emotion = self.feature_mapping_reverse_emotion(decoder_features)

        return pred_recons, pred_emotion

class VQVAE(BaseModel):
    def __init__(self, args):
        super().__init__()

        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.n_embed = args.n_embed if hasattr(args, 'n_embed') else 256
        self.zquant_dim = args.zquant_dim if hasattr(args, 'zquant_dim') else 128
        self.quantizer_beta = args.quantizer_beta if hasattr(args, 'quantizer_beta') else 0.25

        self.motion_encoder = TransformerEncoder(args)
        self.motion_decoder = TransformerDecoder(args)
        self.quantize = VectorQuantizer(self.n_embed, self.zquant_dim, beta=self.quantizer_beta)

    def forward(self, batch):
        encoder_motion = self.motion_encoder(batch)
        quant, quant_loss, _ = self.quantize(encoder_motion)
        motion_pred, emotion_pred = self.motion_decoder(quant)
        return motion_pred, emotion_pred, quant_loss

    def get_quant(self, x):
        encoder_features = self.motion_encoder(x)
        quant_z, _, info = self.quantize(encoder_features)
        indices = info[2]
        return quant_z, indices
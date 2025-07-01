import torch
from utility import Logs
from predictor import VqvaePredict
from dataset import REACTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from losses import Predictor_training_loss
from utility import save_entire_models
import argparse
from glob import glob
class Train_Predictor:
    def __init__(self, args):
        self.args = args
        self.compute_loss = Predictor_training_loss().compute_loss
        self.early_stopping = args.early_stopping if hasattr(args, 'early_stopping') else True
        self.type = args.type if hasattr(args, 'type') else 'Predictor'
        self.online = args.online if hasattr(args, 'online') else False
        self.style = args.style if hasattr(args, 'style') else None
        self.resample_method = args.resample_method if hasattr(args, 'resample_method') else 'linear_interpolation'
        self.speaker_audio_paths_train = glob(args.speaker_audio_paths_train)
        self.speaker_audio_paths_val = glob(args.speaker_audio_paths_val)
        self.mean_path = args.mean_path if hasattr(args, 'mean_path') else '../mean_face.npy'
        self.std_path = args.std_path if hasattr(args, 'std_path') else '../std_face.npy'
        self.vqvae_path = args.vqvae_path if hasattr(args, 'vqvae_path') else None
        self.sr = args.sr if hasattr(args, 'sr') else 16000
        self.duration = args.duration if hasattr(args, 'duration') else 8
        self.fps = args.fps if hasattr(args, 'fps') else 30
        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.in_fps = args.in_fps if hasattr(args, 'in_fps') else 50
        self.out_fps = args.out_fps if hasattr(args, 'out_fps') else 30
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') else 8
        self.num_workers = args.num_workers if hasattr(args, 'num_workers') else 0
        self.lr = args.lr if hasattr(args, 'lr') else 1e-4
        self.betas = args.betas if hasattr(args, 'betas') else (0.9, 0.98)
        self.eps = args.eps if hasattr(args, 'eps') else 1e-9
        self.patience = args.patience if hasattr(args, 'patience') else 5
        self.lr_factor = args.lr_factor if hasattr(args, 'lr_factor') else 0.1
        self.lr_patience = args.lr_patience if hasattr(args, 'lr_patience') else 5
        self.num_epochs = args.num_epochs if hasattr(args, 'num_epochs') else 100
        self.save_dir = args.save_dir if hasattr(args, 'save_dir') else None
        

    def start_training(self):
        log = Logs(type=self.type, online=self.online, style=self.style, resampling_method=self.resample_method)
        writer = log.setup_logging()
        writer.add_text('Training', 'Starting Predictor Training', global_step=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        writer.add_text('Training', f"Using device: {device}", global_step=0)

        model = VqvaePredict(args=self).to(device)

        # Initialize datasets and dataloaders
        training_dataset = REACTDataset(
            speaker_audio_paths=self.speaker_audio_paths_train,args = self.args
        )

        train_dataloader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        validation_dataset = REACTDataset(
            speaker_audio_paths=self.speaker_audio_paths_val, args = self.args
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=self.betas,
            eps=self.eps,
            lr=self.lr
        )
        
        writer.add_text('Training', "Starting Training", global_step=0)
        
        best_val_loss = np.inf
        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience
        )

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Training"):
                optimizer.zero_grad()
                audio, speaker_3dmm, listener_3dmm, listener_attr, style = batch
                style = style.to(device)
                listener_3dmm = listener_3dmm.to(device).float()
                speaker_3dmm = speaker_3dmm.to(device).float()
                
                motion_out, emotion_out, motion_quant_pred, motion_quant = model(
                    batch=[a for a in audio.detach().cpu()],
                    style=style,
                    speaker_3dmm=speaker_3dmm
                )

                loss = self.compute_loss(listener_3dmm, motion_out, listener_attr, emotion_out, 
                                      motion_quant, motion_quant_pred)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)
            writer.add_scalar('Training loss', avg_train_loss, epoch)
            writer.add_text('TrainingSummary', f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}", 
                          global_step=epoch)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation"):
                    audio, speaker_3dmm, listener_3dmm, listener_attr, style = batch
                    style = style.to(device)
                    listener_3dmm = listener_3dmm.to(device).float()
                    speaker_3dmm = speaker_3dmm.to(device).float()
                    
                    motion_out, emotion_out, motion_quant_pred, motion_quant = model(
                        batch=[a for a in audio.detach().cpu()],
                        style=style,
                        speaker_3dmm=speaker_3dmm
                    )

                    loss = self.compute_loss(listener_3dmm, motion_out, listener_attr, emotion_out, 
                                          motion_quant, motion_quant_pred)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_dataloader)
            writer.add_scalar('Validation loss', avg_val_loss, epoch)
            writer.add_text('ValidationSummary', f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}", 
                          global_step=epoch)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                save_entire_models(
                    model, optimizer, scheduler, epoch, best_val_loss,
                    duration=self.duration,
                    resampling_method=self.resample_method,
                    save_dir=self.save_dir
                )
                writer.add_text('ValidationSummary', 
                              f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}", 
                              global_step=epoch)
            elif self.early_stopping:
                counter += 1
                if counter >= self.patience:
                    writer.add_text('ValidationSummary', 
                                  f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.", 
                                  global_step=epoch)
                    break

        writer.add_text('Training', "Training is complete", global_step=0)
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Predictor model")
    parser.add_argument('--speaker_audio_paths_train', type=str, required=True, help='Path to training speaker audio files')
    parser.add_argument('--speaker_audio_paths_val', type=str, required=True, help='Path to validation speaker audio files')
    parser.add_argument('--mean_path', type=str, default='../mean_face.npy', help='Path to mean face numpy file')
    parser.add_argument('--std_path', type=str, default='../std_face.npy', help='Path to std face numpy file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models')
    parser.add_argument('--vqvae_path', type=str, required=True, help='Path to pretrained VQ-VAE model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98), help='AdamW betas')
    parser.add_argument('--eps', type=float, default=1e-9, help='AdamW epsilon')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
    parser.add_argument('--nfeats', type=int, default=58, help='Number of features')
    parser.add_argument('--n_emotion', type=int, default=25, help='Number of emotion features')
    parser.add_argument('--sr', type=int, default=16000, help='Audio sampling rate')
    parser.add_argument('--duration', type=int, default=8, help='Audio duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--in_fps', type=int, default=50, help='Input frames per second')
    parser.add_argument('--out_fps', type=int, default=30, help='Output frames per second')
    parser.add_argument('--resample_method', type=str, default='linear_interpolation', 
                        choices=['linear_interpolation', 'adaptive_avg_pooling_resampling', 
                                 'linear_projection_resampling', 'conv1d_resampling'], 
                        help='Resampling method')
    parser.add_argument('--online', action='store_true', help='Enable online mode')
    parser.add_argument('--with_style', action='store_true', help='Include style features')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=384, help='Transformer intermediate size')
    parser.add_argument('--audio_dim', type=int, default=1024, help='Audio feature dimension')
    parser.add_argument('--conv_kernel_size', type=int, default=5, help='Convolution kernel size')
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2, help='Leaky ReLU slope')
    parser.add_argument('--audio_encoded_dim', type=int, default=1024, help='Audio encoded dimension')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--k', type=int, default=1, help='Top-k sampling parameter')
    parser.add_argument('--style_input_dim', type=int, default=256, help='Style input dimension')
    parser.add_argument('--style_output_dim', type=int, default=1024, help='Style output dimension')
    parser.add_argument('--w2v_model', type=str, default='facebook/w2v-bert-2.0', help='Wav2Vec model name')
    parser.add_argument('--trainable_layers', type=list, default=['encoder.layers.23', 'encoder.layers.22'], 
                        help='Trainable layers for Wav2Vec model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    # Initialize trainer
    trainer = Train_Predictor(args)
    
    # Start training
    trainer.start_training()

if __name__ == "__main__":
    main()




"""
python train_predictor.py \
    --speaker_audio_paths_train '/home/mudasir/REACT/data/train/audio/speaker/*/*.wav' \
    --speaker_audio_paths_val '/home/mudasir/REACT/data/val/audio/speaker/*/*.wav' \
    --save_dir '/home/mudasir/REACT/REACT_SUBMISSION/save_models' \
    --vqvae_path '/home/mudasir/ankit/ProbTalk3D/save_models/VQVAE/audio_duration8/entire_model.pt' \
    --mean_path '/home/mudasir/REACT/mean_face.npy' \
    --std_path '/home/mudasir/REACT/std_face.npy' \
    --batch_size 8 \
    --num_workers 0 \
    --num_epochs 100 \
    --sr 16000 \
    --duration 8 \
    --fps 30 \
    --resample_method 'linear_interpolation' \
    --online \
    --with_style \
    --early_stopping \
    --device cuda

"""
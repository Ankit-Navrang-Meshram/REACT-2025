import torch
from utility import stage_1_save_models, save_entire_models , stage_1_load_models
from utility import Logs
import numpy as np
from losses import VQVAE_training_loss
from vqvae import VQVAE
from dataset import REACTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from glob import glob
import torch.multiprocessing as mp
import os
class Train_VQVAE:
    def __init__(self, args):
        self.args = args
        self.stage_1_loss = VQVAE_training_loss.stage_1_loss
        self.stage_2_loss = VQVAE_training_loss.stage_2_loss
        self.speaker_audio_paths_train = glob(args.speaker_audio_paths_train)
        self.speaker_audio_paths_val = glob(args.speaker_audio_paths_val)
        self.online = args.online
        self.type = args.type if hasattr(args, 'type') else 'VQVAE'
        self.style = args.style
        self.mean_path = args.mean_path
        self.std_path = args.std_path
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') else 32
        self.num_workers = args.num_workers if hasattr(args, 'num_workers') else 4
        self.nfeats = args.nfeats if hasattr(args, 'nfeats') else 58
        self.n_emotion = args.n_emotion if hasattr(args, 'n_emotion') else 25
        self.n_embed = args.n_embed if hasattr(args, 'n_embed') else 256
        self.zquant_dim = args.zquant_dim if hasattr(args, 'zquant_dim') else 128
        self.sr = args.sr if hasattr(args, 'sr') else 16000
        self.duration = args.duration if hasattr(args, 'duration') else 8
        self.fps = args.fps if hasattr(args, 'fps') else 30
        self.num_epochs = args.num_epochs if hasattr(args, 'num_epochs') else 100
        self.lr = args.lr if hasattr(args, 'lr') else 1e-4
        self.betas = args.betas if hasattr(args, 'betas') else (0.9, 0.98)
        self.eps = args.eps if hasattr(args, 'eps') else 1e-9
        self.patience = args.patience if hasattr(args, 'patience') else 10
        self.lr_factor = args.lr_factor if hasattr(args, 'lr_factor') else 0.1
        self.lr_patience = args.lr_patience if hasattr(args, 'lr_patience') else 5
        self.motion_encoder_path = args.motion_encoder_path if hasattr(args, 'motion_encoder_path') else None
        self.motion_decoder_path = args.motion_decoder_path if hasattr(args, 'motion_decoder_path') else None
        self.quantizer_path = args.quantizer_path if hasattr(args, 'quantizer_path') else None
        self.save_dir = args.save_dir if hasattr(args, 'save_dir') else None
        self.early_stopping = args.early_stopping if hasattr(args, 'early_stopping') else True

    def train_vqvae_stage1(self, model, train_dataloader, validation_dataloader, device, writer):
        model = model.to(device)

        # Freeze emotion prediction layers
        for param in model.motion_decoder.feature_mapping_reverse_emotion.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=self.betas, 
            eps=self.eps, 
            lr=self.lr
        )
        
        writer.add_text('VQVAE Stage 1 Training', "Starting Stage 1 Training", global_step=0)
        
        best_val_loss = np.inf
        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.lr_factor, 
            patience=self.lr_patience
        )

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            for audio, speaker_3dmm , listener_3dmm, listener_emotion, style in tqdm(
                train_dataloader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"
            ):
                optimizer.zero_grad()
                speaker_3dmm = speaker_3dmm.to(device).float()
                listener_emotion = listener_emotion.to(device).float()

                motion_pred, emotion_pred, quant_loss = model(speaker_3dmm)

                loss = self.stage_1_loss(quant_loss, motion_pred, emotion_pred, listener_3dmm, listener_emotion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            writer.add_text('VQVAE Stage 1 Training', f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}", global_step=epoch)
            writer.add_scalar('VQVAE Stage 1 Training Loss', avg_loss, epoch)

            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for audio, speaker_3dmm , listener_3dmm, listener_emotion, style in tqdm(
                    validation_dataloader, desc=f"Validation Epoch {epoch+1}/{self.num_epochs}"
                ):
                    speaker_3dmm = speaker_3dmm.to(device).float()
                    listener_emotion = listener_emotion.to(device).float()
                    
                    motion_pred, emotion_pred, quant_loss = model(speaker_3dmm)
                    loss = self.stage_1_loss(quant_loss, motion_pred, emotion_pred, listener_3dmm, listener_emotion)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(validation_dataloader)
                writer.add_text('VQVAE Stage 1 Validation', f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {avg_val_loss:.4f}", global_step=epoch)
                writer.add_scalar('VQVAE Stage 1 Validation Loss', avg_val_loss, epoch)
                scheduler.step(avg_val_loss)

                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    stage_1_save_models(
                        model, optimizer, scheduler, epoch, best_val_loss,
                        duration=self.duration, save_dir=self.save_dir,
                         with_style=self.style, online=self.online
                    )
                    writer.add_text('VQVAE Stage 1 Validation', 
                                  f"New best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}", 
                                  global_step=epoch)
                elif self.early_stopping:
                    counter += 1
                    if counter >= self.patience:
                        writer.add_text('ValidationSummary', 
                                    f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.", 
                                    global_step=epoch)
                        break

        writer.add_text('VQVAE Stage 1 Training', "Stage 1 Training Completed", global_step=self.num_epochs)

    def train_vqvae_stage2(self, model, train_dataloader, validation_dataloader, device, writer):
        model = model.to(device)

        # Freeze all components except emotion prediction layer
        model_components = [
            model.motion_encoder,
            model.motion_decoder.feature_mapping_reverse,
            model.quantize
        ]
        for component in model_components:
            for param in component.parameters():
                param.requires_grad = False
        for param in model.motion_decoder.feature_mapping_reverse_emotion.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.motion_decoder.feature_mapping_reverse_emotion.parameters(),
            betas=self.betas,
            eps=self.eps,
            lr=self.lr
        )
        
        writer.add_text('VQVAE Stage 2 Training', "Starting Stage 2 Training", global_step=0)
        best_val_loss = np.inf
        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience
        )
        
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            for audio, speaker_3dmm , listener_3dmm, listener_emotion, style in tqdm(
                train_dataloader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"
            ):
                optimizer.zero_grad()
                speaker_3dmm = speaker_3dmm.to(device).float()
                listener_emotion = listener_emotion.to(device).float()

                motion_pred, emotion_pred, quant_loss = model(speaker_3dmm)
                loss = self.stage_2_loss(emotion_pred, listener_emotion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            writer.add_text('VQVAE Stage 2 Training', f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}", global_step=epoch)
            writer.add_scalar('VQVAE Stage 2 Training Loss', avg_loss, epoch)

            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for audio, speaker_3dmm , listener_3dmm, listener_emotion, style  in tqdm(
                    validation_dataloader, desc=f"Validation Epoch {epoch+1}/{self.num_epochs}"
                ):
                    speaker_3dmm = speaker_3dmm.to(device).float()
                    listener_emotion = listener_emotion.to(device).float()

                    motion_pred, emotion_pred, quant_loss = model(speaker_3dmm)
                    loss = self.stage_2_loss(emotion_pred, listener_emotion)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(validation_dataloader)
                writer.add_text('VQVAE Stage 2 Validation', f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {avg_val_loss:.4f}", global_step=epoch)
                writer.add_scalar('VQVAE Stage 2 Validation Loss', avg_val_loss, epoch)
                scheduler.step(avg_val_loss)

                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    save_entire_models(
                        model, optimizer, scheduler, epoch, best_val_loss,
                        duration=self.duration, save_dir=self.save_dir,
                        type=self.type, with_style=self.style, online=self.online
                    )
                    writer.add_text('VQVAE Stage 2 Validation', 
                                  f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}", 
                                  global_step=epoch)
                elif self.early_stopping:
                    counter += 1
                    if counter >= self.patience:
                        writer.add_text('ValidationSummary', 
                                    f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.", 
                                    global_step=epoch)
                        break

        writer.add_text('VQVAE Stage 2 Training', "Stage 2 Training Completed", global_step=self.num_epochs)

    def start_training(self):
        log = Logs(type=self.type, online=self.online, style=self.style)
        writer = log.setup_logging()
        writer.add_text('Training', 'Starting VQ-VAE Training', global_step=0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        writer.add_text('Training', f"Using device: {device}", global_step=0)

        model = VQVAE(self.args)
        writer.add_text('Training', f"Model initialized", global_step=0)
        
        # Initialize datasets and dataloaders
        training_dataset = REACTDataset(speaker_audio_paths=self.speaker_audio_paths_train, args = self.args)

        train_dataloader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        validation_dataset = REACTDataset(
            speaker_audio_paths=self.speaker_audio_paths_val,args = self.args)

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        # Stage 1 training
        writer.add_text('Training', "Starting Stage 1 Training", global_step=0)
        self.train_vqvae_stage1(
            model, train_dataloader, validation_dataloader, device, writer
        )

        ## Load best model from stage 1 if paths are provided
        #if all([self.motion_encoder_path, self.motion_decoder_path, self.quantizer_path]):
        #    model.motion_encoder.load_state_dict(
        #        torch.load(self.motion_encoder_path, map_location=device)['model_state_dict']
        #    )
        #    model.motion_decoder.load_state_dict(
        #        torch.load(self.motion_decoder_path, map_location=device)['model_state_dict']
        #    )
        #    model.quantize.load_state_dict(
        #        torch.load(self.quantizer_path, map_location=device)['model_state_dict']
        #    )
        model , _ ,_ = stage_1_load_models(model ,duration=self.duration, type = self.type, save_dir=self.save_dir,with_style=self.style, online=self.online)
        writer.add_text('Training', "Stage 1 Model Loaded", global_step=0)

        # Stage 2 training
        writer.add_text('Training', "Starting Stage 2 Training", global_step=0)
        self.train_vqvae_stage2(
            model, train_dataloader, validation_dataloader, device, writer
        )

        writer.add_text('Training', "Training is complete", global_step=0)
        writer.close()




def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE model")
    parser.add_argument('--speaker_audio_paths_train', type=str, required=True, help='Path to training speaker audio files')
    parser.add_argument('--speaker_audio_paths_val', type=str, required=True, help='Path to validation speaker audio files')
    parser.add_argument('--mean_path', type=str, default='../mean_face.npy', help='Path to mean face numpy file')
    parser.add_argument('--std_path', type=str, default='../std_face.npy', help='Path to std face numpy file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98), help='AdamW betas')
    parser.add_argument('--eps', type=float, default=1e-9, help='AdamW epsilon')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
    parser.add_argument('--nfeats', type=int, default=58, help='Number of features')
    parser.add_argument('--n_emotion', type=int, default=25, help='Number of emotion features')
    parser.add_argument('--n_embed', type=int, default=256, help='Number of embeddings')
    parser.add_argument('--zquant_dim', type=int, default=128, help='Quantizer dimension')
    parser.add_argument('--sr', type=int, default=16000, help='Audio sampling rate')
    parser.add_argument('--duration', type=int, default=8, help='Audio duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--online', action='store_true', help='Enable online mode')
    parser.add_argument('--style', action='store_true', help='Include style features')
    parser.add_argument('--motion_encoder_path', type=str, help='Path to pretrained motion encoder')
    parser.add_argument('--motion_decoder_path', type=str, help='Path to pretrained motion decoder')
    parser.add_argument('--quantizer_path', type=str, help='Path to pretrained quantizer')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='Enable early stopping')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize trainer
    trainer = Train_VQVAE(args)
    
    # Start training
    trainer.start_training()
    
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()



"""
command to run this script:

python train_vqvae_main.py \
    --speaker_audio_paths_train '/home/mudasir/REACT/data/train/audio/speaker/*/*.wav' \
    --speaker_audio_paths_val '/home/mudasir/REACT/data/val/audio/speaker/*/*.wav' \
    --save_dir '/home/mudasir/REACT/REACT_SUBMISSION/save models' \
    --mean_path '/home/mudasir/REACT/mean_face.npy' \
    --std_path '/home/mudasir/REACT/std_face.npy' \
    --batch_size 8 \
    --num_workers 4 \
    --num_epochs 100 \
    --lr 0.0001 \
    --patience 10 \
    --lr_factor 0.1 \
    --lr_patience 5 \
    --nfeats 58 \
    --n_emotion 25 \
    --n_embed 256 \
    --zquant_dim 128 \
    --sr 16000 \
    --duration 8 \
    --fps 30 \
    --online \
    --style \
    --motion_encoder_path '/home/mudasir/REACT/REACT_SUBMISSION/save_models/VQVAE/offline/with_style/audio' \
    --motion_decoder_path /path/to/pretrained/motion_decoder.pt \
    --quantizer_path /path/to/pretrained/quantizer.pt \
    --early_stopping


    

    add boolean flags if you want to set them true else leave them:

    python train_vqvae.py \
    --speaker_audio_paths_train '/home/mudasir/REACT/data/train/audio/speaker/*/*.wav' \
    --speaker_audio_paths_val '/home/mudasir/REACT/data/val/audio/speaker/*/*.wav' \
    --save_dir '/home/mudasir/REACT/REACT_SUBMISSION/save_models' \
    --mean_path '/home/mudasir/REACT/mean_face.npy' \
    --std_path '/home/mudasir/REACT/std_face.npy' \
    --batch_size 8 \
    --num_workers 0 \
    --num_epochs 100 \
    --sr 16000 \
    --duration 8 \
    --fps 30 \
    --online \
    --style 
"""
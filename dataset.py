import numpy as np
import librosa
from random import randint
from torch.utils.data import Dataset
import torch
import wespeaker
import soundfile as sf
import os

class REACTDataset(Dataset):
    def __init__(self, speaker_audio_paths ,args):
        super().__init__()

        # Initialize parameters from args with defaults
        self.speaker_audio_paths = speaker_audio_paths
        self.mean_path = args.mean_path if hasattr(args, 'mean_path') else '../mean_face.npy'
        self.std_path = args.std_path if hasattr(args, 'std_path') else '../std_face.npy'
        self.sr = args.sr if hasattr(args, 'sr') else 16000
        self.duration = args.duration if hasattr(args, 'duration') else 25
        self.fps = args.fps if hasattr(args, 'fps') else 30
        self.wespeaker_model = args.wespeaker_model if hasattr(args, 'wespeaker_model') else 'english'
        self.device = args.device if hasattr(args, 'device') else 'cuda'

        # Speaker and listener file paths
        self.speaker_attr_paths = [fname.replace('audio', 'facial-attributes').replace('.wav', '.npy') 
                                 for fname in self.speaker_audio_paths]
        self.speaker_3dmm_paths = [fname.replace('audio', 'coefficients').replace('.wav', '.npy') 
                                  for fname in self.speaker_audio_paths]
        self.listener_3dmm_paths = [fname.replace('audio', 'coefficients').replace('speaker', 'listener').replace('.wav', '.npy') 
                                  for fname in self.speaker_audio_paths]
        self.listener_attr_paths = [fname.replace('speaker', 'listener') 
                                  for fname in self.speaker_attr_paths]

        # Load mean and std
        self.mean = np.load(self.mean_path)
        self.std = np.load(self.std_path)

        # Calculate max frames and audio length
        self.max_frames = self.duration * self.fps
        self.max_audio_len = self.sr * self.duration

        # Initialize wespeaker
        self.wespeaker = wespeaker.load_model(self.wespeaker_model)
        self.wespeaker.set_device(self.device)

    def __len__(self):
        return len(self.speaker_audio_paths)
    
    def transform(self, x):
        # x = (x - self.mean) / self.std
        x = x - self.mean
        return x
    
    def inverse_transform(self, x):
        # x = (x * self.std) + self.mean
        x = x + self.mean
        return x
    
    def pad_truncate_audio(self, x):
        if x.shape[0] > self.max_audio_len:
            x = x[:self.max_audio_len]
        else:
            pad_len = self.max_audio_len - x.shape[0]
            x = np.pad(x, (0, pad_len), 'constant', constant_values=0)
        return x
    
    def pad_truncate_frames(self, x):
        if x.shape[0] > self.max_frames:
            x = x[:self.max_frames, :]
        else:
            pad_len = self.max_frames - x.shape[0]
            x = np.pad(x, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        return x
    
    def __getitem__(self, idx):
        # Load full audio and calculate duration
        audio, _ = librosa.load(self.speaker_audio_paths[idx], sr=self.sr)
        audio_duration = librosa.get_duration(y=audio, sr=self.sr)

        # Select a random chunk based on duration
        if audio_duration > self.duration:
            max_start_time = audio_duration - self.duration
            start_time = np.random.uniform(0, max_start_time)
        else:
            start_time = 0

        # Convert start time to sample and frame indices
        start_sample = int(start_time * self.sr)
        audio_window_len = int(self.duration * self.sr)
        start_frame = int(start_time * self.fps)
        frame_window_len = int(self.duration * self.fps)

        # Extract audio chunk
        audio = audio[start_sample:start_sample + audio_window_len]
        audio = self.pad_truncate_audio(audio)

        # Creating style features
        temp_audio_file = f"temp_{idx}.wav"
        sf.write(temp_audio_file, audio, self.sr)
        style = self.wespeaker.extract_embedding(temp_audio_file)
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

        # Loading Listener facial attributes
        listener_attr = np.load(self.listener_attr_paths[idx])
        listener_attr = listener_attr[start_frame:start_frame + frame_window_len, :]
        listener_attr = self.pad_truncate_frames(listener_attr)
    
        # Loading listener 3dmm
        listener_3dmm = np.load(self.listener_3dmm_paths[idx]).squeeze(axis=1)
        listener_3dmm = listener_3dmm[start_frame:start_frame + frame_window_len, :]
        listener_3dmm = self.pad_truncate_frames(listener_3dmm)
        listener_3dmm = self.transform(listener_3dmm)

        # Loading speaker 3dmm
        speaker_3dmm = np.load(self.speaker_3dmm_paths[idx]).squeeze(axis=1)
        speaker_3dmm = speaker_3dmm[start_frame:start_frame + frame_window_len, :]
        speaker_3dmm = self.pad_truncate_frames(speaker_3dmm)
        speaker_3dmm = self.transform(speaker_3dmm)

        return audio, speaker_3dmm , listener_3dmm, listener_attr, style
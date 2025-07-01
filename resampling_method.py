import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAveragePooling:
    def __init__(self, args):
        self.in_fps = args.in_fps if hasattr(args, 'in_fps') else 30
        self.out_fps = args.out_fps if hasattr(args, 'out_fps') else 30

    def adaptive_avg_pooling_resampling(self,features):
        features = features.transpose(1, 2)  # [B, C, T]

        input_len = features.shape[2]
        seq_len_sec = input_len / float(self.in_fps)
        output_len = int(seq_len_sec * self.out_fps) + 1

        if output_len >= input_len:
            # Upsampling: use interpolation
            features = F.interpolate(features, size=output_len, mode='linear', align_corners=True)
        else:
            # Downsampling: use adaptive average pooling
            pool = nn.AdaptiveAvgPool1d(output_len)
            features = pool(features)

        return features.transpose(1, 2)  # [B, T, C]

class LinearInterpolate:
    def __init__(self, args):
        self.in_fps = args.in_fps if hasattr(args, 'in_fps') else 30
        self.out_fps = args.out_fps if hasattr(args, 'out_fps') else 30

    def linear_interpolate_input(self, features):
        features = features.transpose(1, 2)  # [B, C, T]
        seq_len = features.shape[2] / float(self.in_fps)

        output_len = int(seq_len * self.out_fps) + 1
        output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
        return output_features.transpose(1, 2)  # [B, T, C]

class LinearProjectionResampling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_len = args.in_len if hasattr(args, 'in_len') else 240
        self.out_len = args.out_len if hasattr(args, 'out_len') else 240
        self.proj = nn.Linear(self.in_len, self.out_len)

    def forward(self, x):
        # x: [B, T, C] → transpose to [B, C, T]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.proj(x)       # [B, C, out_T]
        x = x.transpose(1, 2)  # [B, out_T, C]
        return x

class Conv1DResampling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.in_channels if hasattr(args, 'in_channels') else 128
        self.mode = args.mode if hasattr(args, 'mode') else 'down'
        self.scale = args.scale if hasattr(args, 'scale') else 2
        assert self.mode in ['down', 'up'], "mode must be 'down' or 'up'"

        if self.mode == 'down':
            self.conv = nn.Conv1d(self.in_channels, self.in_channels, 
                                kernel_size=self.scale, stride=self.scale)
        else:  # 'up'
            self.conv = nn.ConvTranspose1d(self.in_channels, self.in_channels, 
                                         kernel_size=self.scale, stride=self.scale)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossAttention(nn.Module):
    def __init__(self, feat_dim=1024, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_heads

        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

        # Final linear layer after attention
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x_content, s_style):
        """
        x_content: (B, T, F) - query input (Wav2Vec-BERT embeddings)
        s_style: (B, F)      - key/value input (WeSpeaker embeddings)
        """
        B, T, dim = x_content.shape

        # Expand speaker embedding to match the sequence length
        s_style_exp = s_style.unsqueeze(1).expand(B, T, dim)  # (B, T, F)

        # Linear projections
        Q = self.q_proj(x_content)      # (B, T, F)
        K = self.k_proj(s_style_exp)    # (B, T, F)
        V = self.v_proj(s_style_exp)    # (B, T, F)

        # Reshape for multi-head: (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        attn = F.softmax(scores, dim=-1)                                        # (B, H, T, T)

        out = torch.matmul(attn, V)  # (B, H, T, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, dim)  # (B, T, F)

        return self.out_proj(out)  # (B, T, F)


class HistoricalCrossAttention(nn.Module):
    def __init__(self, feat_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        self.head_dim = feat_dim // num_heads

        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

        # Final linear layer after attention
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x_current, x_past):

        B, T, dim = x_current.shape


        # Linear projections
        Q = self.q_proj(x_current)      # (B, T, F)
        K = self.k_proj(x_past)    # (B, T, F)
        V = self.v_proj(x_past)    # (B, T, F)

        # Reshape for multi-head: (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        attn = F.softmax(scores, dim=-1)                                        # (B, H, T, T)

        out = torch.matmul(attn, V)  # (B, H, T, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, dim)  # (B, T, F)

        return self.out_proj(out)  # (B, T, F)

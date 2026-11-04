"""
src/week2_conformer_pipeline.py
===============================
Shared Conformer Architecture Building Blocks.
Synchronized between Member 2 (Counting) and Member 3 (Tagging).
Provides:
  - Conv2dSubsampling: 4x 2D Spectrogram Subsampling ([B, 1, 128, 251] -> [B, 63, d_model])
  - ConformerBlock: Macaron-style Convolution-augmented Transformer Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model=256, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.ln(x)
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return res + 0.5 * x


class ConformerConvModule(nn.Module):
    def __init__(self, d_model=256, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        res = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return res + x


class ConformerBlock(nn.Module):
    """
    Macaron-style Conformer Block:
      1. Feed-Forward 1 (half-step)
      2. Multi-Head Self-Attention
      3. Convolution Module
      4. Feed-Forward 2 (half-step)
      5. LayerNorm
    """
    def __init__(self, d_model=256, n_heads=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_model * 4, dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, d_model * 4, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ff1(x)
        res = x
        x_norm = self.ln_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln_out(x)


class Conv2dSubsampling(nn.Module):
    """
    4x Subsampling layer converting [B, 1, 128, 251] Log-Mel -> [B, 63, d_model].
    Exact match with Member 2's front-end representation.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # 128 freq bins -> 64 -> 32 freq bins
        self.out = nn.Linear(d_model * 32, d_model)

    def forward(self, x):
        # Input x: [B, 1, 128, 251]
        x = self.conv(x)  # [B, d_model, 32, 63]
        b, c, f, t = x.shape
        # Permute to [B, t=63, c*f=d_model*32]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return self.out(x)  # [B, 63, d_model]

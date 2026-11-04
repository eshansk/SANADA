"""
models/tagging/conformer_tagger.py
==================================
Conformer Multi-Label Tagging Architecture with Temporal Attention Pooling.
Member 3: Acoustic Label Tagging & Sound Event Detection Lead
Paper Reference: Gulati et al., Interspeech 2020; SANADA Base Paper (Section 5.2, Table 5)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tagging.config import TaggingConfig as CFG


class FeedForward(nn.Module):
    """Macaron-style half-step feed-forward module."""
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
    """Depthwise separable convolution module for local acoustic context capture."""
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
        self.ff1 = FeedForward(d_model, d_model * CFG.FFN_EXPANSION, dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, d_model * CFG.FFN_EXPANSION, dropout)
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
    4x 2D Spectrogram Subsampling Layer converting [B, 1, 128, 251] -> [B, 63, d_model].
    Matches exact front-end representation with Member 2 (Shravan).
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.out = nn.Linear(d_model * 32, d_model)

    def forward(self, x):
        # x: [B, 1, 128, 251]
        x = self.conv(x)  # [B, d_model, 32, 63]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return self.out(x)  # [B, 63, d_model]


class TemporalAttentionPooling(nn.Module):
    """Weights active acoustic frames higher than silent frames for transient sound detection."""
    def __init__(self, d_model=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [B, 63, d_model]
        weights = F.softmax(self.attn(x), dim=1)  # [B, 63, 1]
        return torch.sum(x * weights, dim=1)      # [B, d_model]


class SANADAConformerTagger(nn.Module):
    """
    4-Block Conformer Acoustic Tagger with Temporal Attention Pooling.
    Outputs raw class logits [B, C] for multi-label sound event tagging.
    """
    def __init__(
        self,
        num_classes: int = CFG.NUM_CLASSES,
        d_model: int = CFG.D_MODEL,
        n_blocks: int = CFG.N_BLOCKS,
        n_heads: int = CFG.N_HEADS,
        conv_kernel: int = CFG.CONV_KERNEL,
        dropout: float = CFG.DROPOUT
    ):
        super().__init__()
        self.num_classes = num_classes
        self.subsampling = Conv2dSubsampling(d_model=d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, CFG.T_SUBSAMPLED, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, n_heads=n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.attn_pool = TemporalAttentionPooling(d_model=d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input spectrogram tensor of shape [B, 1, 128, 251]
        Returns:
            Logits of shape [B, num_classes]
        """
        x = self.subsampling(x)  # [B, 63, 256]
        x = x + self.pos_emb[:, :x.size(1), :]
        for block in self.blocks:
            x = block(x)
        pooled = self.attn_pool(x)        # [B, 256]
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits

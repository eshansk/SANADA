r"""
models/counting/conformer_counter.py
====================================
4-Block Macaron Conformer Regressor for High-Density Source Counting
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead

Architecture Overview:
  Input [B, 1, 128, 251]
    ├── Conv2dSubsampling (stride=2, 4x time/freq reduction) -> [B, 63, 256]
    ├── Add Learnable Positional Embeddings
    ├── 4x Macaron Conformer Blocks:
    │     ├── FFN 1 (0.5x residual)
    │     ├── Multi-Head Self-Attention (MHSA, 4 heads)
    │     ├── Depthwise Separable Conv (kernel=31, GLU + Swish + BatchNorm)
    │     ├── FFN 2 (0.5x residual)
    │     └── LayerNorm
    ├── Global Average Pooling (GAP across 63 time steps) -> [B, 256]
    └── Linear Regression Head -> Continuous Scalar Prediction \hat{y} [B]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CountingConfig as CFG


class Conv2dSubsampling(nn.Module):
    """Reduces time & frequency by 4x: [B, 1, 128, 251] -> [B, 63, d_model]."""
    def __init__(self, d_model: int = CFG.D_MODEL):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 128 // 4 = 32 freq bins; 251 -> 126 -> 63 time frames
        self.proj = nn.Linear(64 * 32, d_model)
        self.dropout = nn.Dropout(CFG.DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # [B, 64, 32, 63]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, 63, 2048]
        return self.dropout(self.proj(x))  # [B, 63, d_model]


class FeedForwardModule(nn.Module):
    """Macaron-style half-step Feed-Forward Network."""
    def __init__(self, d_model: int = CFG.D_MODEL, expansion_factor: int = CFG.FFN_EXPANSION, dropout: float = CFG.DROPOUT):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Depthwise Separable Convolution (kernel=31) with GLU & Swish."""
    def __init__(self, d_model: int = CFG.D_MODEL, kernel_size: int = CFG.CONV_KERNEL, dropout: float = CFG.DROPOUT):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x).transpose(1, 2)  # [B, D, T]
        x = F.glu(self.pointwise_conv1(x), dim=1)  # [B, D, T]
        x = self.depthwise_conv(x)
        x = self.act(self.batch_norm(x))
        x = self.dropout(self.pointwise_conv2(x))
        return x.transpose(1, 2)  # [B, T, D]


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block with MHSA and Depthwise Separable Conv."""
    def __init__(self, d_model: int = CFG.D_MODEL, n_heads: int = CFG.N_HEADS,
                 conv_kernel: int = CFG.CONV_KERNEL, dropout: float = CFG.DROPOUT):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion_factor=CFG.FFN_EXPANSION, dropout=dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.mhsa_drop = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion_factor=CFG.FFN_EXPANSION, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        norm_x = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.mhsa_drop(attn_out)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class SANADAConformerCounter(nn.Module):
    """4-Block Macaron Conformer Regressor with Global Average Pooling (GAP)."""
    def __init__(self, d_model: int = CFG.D_MODEL, n_blocks: int = CFG.N_BLOCKS,
                 n_heads: int = CFG.N_HEADS, conv_kernel: int = CFG.CONV_KERNEL, dropout: float = CFG.DROPOUT):
        super().__init__()
        self.subsampling = Conv2dSubsampling(d_model=d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, CFG.T_SUBSAMPLED, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, n_heads=n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 251]
        x = self.subsampling(x)  # [B, 63, 256]
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        # Global Average Pooling across time dimension (T'=63)
        pooled = x.mean(dim=1)  # [B, 256]
        return self.regression_head(pooled).squeeze(-1)  # [B]

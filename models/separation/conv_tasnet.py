"""
models/separation/conv_tasnet.py
================================
Convolutional Time-Domain Audio Separation Network (Conv-TasNet)
with Dynamic Count-Conditioned Stem Gating (CC-Conv-TasNet)
Member 4: Monaural Audio Source Separation Lead
Samslesha-Nada (SANADA) Capstone Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import SeparationConfig as CFG


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN) across [Channel, Time] dimensions.
    Normalizes over all frames and channels simultaneously, making the representation
    invariant to utterance-level scaling differences.
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, Time]
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return ((x - mean) / torch.sqrt(var + self.eps)) * self.gamma + self.beta


class DepthwiseSeparableTCNBlock(nn.Module):
    """
    1D Dilated Depthwise Separable Convolution Block with Residual and Skip Paths.
    Exponential dilation expands the temporal receptive field across frames with minimal parameters.
    """
    def __init__(self, in_channels: int = 64, hidden_channels: int = 128, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1x1_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(hidden_channels)

        self.dconv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(hidden_channels)

        self.res_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.skip_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B, in_channels, T]
        h = self.norm1(self.prelu1(self.conv1x1_in(x)))
        h = self.norm2(self.prelu2(self.dconv(h)))
        res = x + self.res_out(h)
        skip = self.skip_out(h)
        return res, skip


class SANADAConvTasNet(nn.Module):
    """
    Time-Domain Audio Separation Network for Monaural Overlapping Audio.
    Includes Count-Conditioned Dynamic Stem Gating synced with Member 2's
    Conformer Counter (\\hat{K}) to eliminate the SANADA Paper's zero-padding ambiguity.
    """
    def __init__(
        self,
        max_sources: int = CFG.MAX_SOURCES,
        enc_dim: int = CFG.N_FILTERS,
        bottleneck_dim: int = CFG.BOTTLENECK_DIM,
        hidden_dim: int = CFG.HIDDEN_DIM,
        kernel_len: int = CFG.KERNEL_LEN,
        stride: int = CFG.STRIDE,
        num_blocks: int = CFG.NUM_BLOCKS,
        num_repeats: int = CFG.NUM_REPEATS
    ):
        super().__init__()
        self.max_sources = max_sources
        self.enc_dim = enc_dim
        self.kernel_len = kernel_len
        self.stride = stride

        # 1. Linear 1D-Conv Encoder (Learned Time-Domain Basis)
        self.encoder = nn.Conv1d(1, enc_dim, kernel_size=kernel_len, stride=stride, bias=False)
        self.enc_norm = GlobalLayerNorm(enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck_dim, kernel_size=1)

        # 2. Stacked Dilated Depthwise-Separable TCN Masker Network
        tcn_blocks = []
        for _ in range(num_repeats):
            for x in range(num_blocks):
                tcn_blocks.append(
                    DepthwiseSeparableTCNBlock(
                        in_channels=bottleneck_dim,
                        hidden_channels=hidden_dim,
                        kernel_size=3,
                        dilation=2 ** x
                    )
                )
        self.tcn = nn.ModuleList(tcn_blocks)

        # 3. Mask Generation Head (K_max * enc_dim multiplicative masks)
        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(bottleneck_dim, max_sources * enc_dim, kernel_size=1)

        # 4. Transposed 1D-Conv Decoder (Waveform Synthesis)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size=kernel_len, stride=stride, bias=False)

    def forward(self, mixture_wav: torch.Tensor, active_k: int = None) -> torch.Tensor:
        """
        Forward pass for source separation.
        Args:
            mixture_wav: Tensor of shape [B, 1, N] or [B, N]
            active_k   : Optional integer \\hat{K} from Member 2's Conformer Counter.
                         Dynamically slices top \\hat{K} highest-energy stems.
        Returns:
            est_wavs   : Separated stems tensor of shape [B, K_out, N]
        """
        if mixture_wav.ndim == 2:
            mixture_wav = mixture_wav.unsqueeze(1)

        B, _, N = mixture_wav.size()

        # Exact stride alignment padding
        rest = (N - self.kernel_len) % self.stride
        pad_len = (self.stride - rest) % self.stride
        if pad_len > 0:
            mixture_wav = F.pad(mixture_wav, (0, pad_len))

        # 1. Time-Domain Encoding: [B, enc_dim, T_frames]
        w = F.relu(self.encoder(mixture_wav))
        x = self.bottleneck(self.enc_norm(w))

        # 2. Receptive Field Mask Estimation via TCN Skip Sum
        skip_sum = 0.0
        for block in self.tcn:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # 3. Multi-Source Masking: [B, max_sources, enc_dim, T_frames]
        masks = torch.sigmoid(self.mask_conv(self.mask_prelu(skip_sum)))
        masks = masks.view(B, self.max_sources, self.enc_dim, -1)

        # 4. Latent Modulation & Waveform Decoding: [B * max_sources, 1, N]
        masked_w = w.unsqueeze(1) * masks
        masked_w_flat = masked_w.view(B * self.max_sources, self.enc_dim, -1)
        est_wavs = self.decoder(masked_w_flat)
        est_wavs = est_wavs.view(B, self.max_sources, -1)[:, :, :N]

        # 5. Member 2 Dynamic Stem Gating: Eliminate zero-padding ambiguity
        if active_k is not None and 1 <= active_k < self.max_sources:
            energies = torch.mean(est_wavs ** 2, dim=-1)  # [B, max_sources]
            top_idx = torch.topk(energies, k=active_k, dim=-1).indices  # [B, active_k]
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, N)
            est_wavs = torch.gather(est_wavs, dim=1, index=top_idx_exp)

        return est_wavs

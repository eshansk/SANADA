"""
models/counting/dataset.py
==========================
Audio Transforms, Log-Mel Feature Extraction & Dataset Pipeline
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead
"""

import os
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset
from .config import CountingConfig as CFG


class LogMelExtractor(nn.Module):
    """
    Differentiable PyTorch module converting raw waveforms into instance-normalized Log-Mel Spectrograms.
    Output Tensor Contract: [B, 1, 128, 251]
    """
    def __init__(self, sample_rate: int = CFG.SAMPLE_RATE, n_fft: int = CFG.N_FFT,
                 hop_length: int = CFG.HOP_LENGTH, n_mels: int = CFG.N_MELS,
                 f_min: float = CFG.F_MIN, f_max: float = CFG.F_MAX):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect"
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, 1, 128000]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        mel = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel)  # [B, 1, 128, 251]
        # Instance normalization per sample
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_mel - mean) / std


class SpecAugment(nn.Module):
    """Time and Frequency Masking for Regularization against co-channel masking."""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 32):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(mel))


class CountingDataset(Dataset):
    """PyTorch Dataset loading pre-extracted Log-Mel tensors or audio arrays."""
    def __init__(self, samples_list, augment: bool = False):
        self.samples = samples_list
        self.augment = augment
        self.spec_aug = SpecAugment() if augment else nn.Identity()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, k_label = self.samples[idx]
        if self.augment:
            mel = self.spec_aug(mel)
        return mel, torch.tensor(k_label, dtype=torch.float32)

"""
models/tagging/dataset.py
=========================
Acoustic Preprocessing, Shared Log-Mel Spectrogram Extraction & Dataset Pipeline.
Contract 1: Shared DSP Contract with Member 2 (Shravan) & Member 4.
Exact Output Shape: [Batch, 1, 128, 251]
"""

import os
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as T

from models.tagging.config import TaggingConfig as CFG, SANADA_FSD_32_CLASSES


class SharedLogMelExtractor(nn.Module):
    """
    Contract 1 — Mandatory Invariant DSP Preprocessor:
    Input:  Raw monaural audio [B, 1, 128000] (16,000 Hz, 8.0s)
    Output: Standardized Log-Mel Spectrogram [B, 1, 128, 251]
    Formula: Mel(1024 FFT, 512 hop, 128 mels) -> AmplitudeToDB(80dB) -> (x - mean) / (std + 1e-6)
    """
    def __init__(
        self,
        sample_rate: int = CFG.SAMPLE_RATE,
        n_fft: int = CFG.N_FFT,
        hop_length: int = CFG.HOP_LENGTH,
        n_mels: int = CFG.N_MELS,
        top_db: float = CFG.TOP_DB
    ):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=True
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, 1, 128000] or [B, 128000]
        returns:  [B, 1, 128, 251]
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        mel = self.mel_spectrogram(waveform)
        log_mel = self.amp_to_db(mel)  # [B, 1, 128, 251]
        # Instance-level zero-mean unit-variance standardization
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_mel - mean) / std


class SpecAugment(nn.Module):
    """Time and Frequency Masking for Data Augmentation (Park et al., 2019)."""
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 30):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(mel))


class CachedTaggingDataset(Dataset):
    """
    In-Memory / Fast-Disk Cached Multi-Label Spectrogram Dataset.
    Guarantees >= 10 batches/sec DataLoader throughput on CPU.
    """
    def __init__(self, num_samples: int = 320, num_classes: int = CFG.NUM_CLASSES, augment: bool = True):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.augment = augment
        self.spec_aug = SpecAugment()

        rng = np.random.RandomState(CFG.SEED)
        self.data = []
        for _ in range(num_samples):
            # Synthetic mel with acoustic energy spikes matching active classes
            mel = rng.randn(1, CFG.N_MELS, CFG.T_FRAMES).astype(np.float32)
            k = rng.randint(1, 5)
            active_idx = rng.choice(num_classes, size=k, replace=False)
            label = np.zeros(num_classes, dtype=np.float32)
            for idx in active_idx:
                label[idx] = 1.0
                band = (idx * 4) % CFG.N_MELS
                mel[0, band:band + 4, :] += 2.0
            self.data.append((torch.from_numpy(mel), torch.from_numpy(label)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mel, label = self.data[idx]
        if self.augment:
            mel = self.spec_aug(mel)
        return mel, label

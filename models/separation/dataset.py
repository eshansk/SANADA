"""
models/separation/dataset.py
============================
PyTorch Dataset, Utterance-Level PIT Loss Wrapper & Time-Domain Augmentation
Member 4: Monaural Audio Source Separation Lead
Samslesha-Nada (SANADA) Capstone Project
"""

import os
import itertools
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .config import SeparationConfig as CFG


def compute_si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).
    Formula from SANADA Paper Section 5.3:
        s_target = (<est, ref> / ||ref||^2) * ref
        e_noise  = est - s_target
        SI-SDR   = 10 * log10(||s_target||^2 / (||e_noise||^2 + eps))
    Invariant to scalar volume scaling: SI-SDR(s, est) == SI-SDR(s, alpha * est).
    """
    ref = reference - torch.mean(reference, dim=-1, keepdim=True)
    est = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    dot = torch.sum(ref * est, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot / ref_energy) * ref
    e_noise = est - s_target

    target_pow = torch.sum(s_target ** 2, dim=-1) + eps
    noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps
    return 10.0 * torch.log10(target_pow / noise_pow)


class PITLossWrapper(nn.Module):
    """
    Utterance-Level Permutation Invariant Training (uPIT) Loss Wrapper.
    Evaluates all K! permutations between estimated stems and target stems,
    backpropagating through the minimum negative SI-SDR assignment.
    Applies an inactive-channel silence penalty to eliminate zero-padding bleed.
    """
    def __init__(self, silence_penalty_weight: float = CFG.SILENCE_WEIGHT):
        super().__init__()
        self.silence_weight = silence_penalty_weight

    def forward(self, est_sources: torch.Tensor, target_sources: torch.Tensor, valid_k: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            est_sources   : [B, K_max, N]
            target_sources: [B, K_max, N]
            valid_k       : Optional [B] tensor indicating active source counts (e.g. 2, 3, 4)
        """
        B, K_max, N = est_sources.size()
        batch_losses = []

        for b in range(B):
            k_active = int(valid_k[b].item()) if valid_k is not None else K_max
            est_b = est_sources[b, :k_active, :]     # [k_active, N]
            tgt_b = target_sources[b, :k_active, :]  # [k_active, N]

            active_perms = list(itertools.permutations(range(k_active)))
            perm_losses = []
            for p in active_perms:
                est_perm = est_b[list(p), :]
                neg_sisdr = -compute_si_sdr(tgt_b, est_perm).mean()
                perm_losses.append(neg_sisdr)

            min_loss = torch.stack(perm_losses).min()

            # Penalize leakage energy on inactive channels (k_active .. K_max)
            if k_active < K_max:
                inactive_est = est_sources[b, k_active:, :]
                silence_loss = torch.mean(inactive_est ** 2) * 10.0
                min_loss = min_loss + self.silence_weight * silence_loss

            batch_losses.append(min_loss)

        return torch.stack(batch_losses).mean()


class SeparationDataset(Dataset):
    """
    Dataset loader for paired audio mixtures and isolated ground-truth stems.
    Applies synchronized polarity inversion and uniform gain augmentations.
    """
    def __init__(self, manifest_records, segment_samples: int = 32000, augment: bool = True):
        self.records = manifest_records
        self.segment_samples = segment_samples
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        mix_path = item["mixture_path"]
        stem_paths = item["source_files"].split(";") if isinstance(item["source_files"], str) else item["source_files"]
        k_val = int(item["num_sources"])

        mix, sr = sf.read(mix_path, dtype="float32")
        if mix.ndim > 1:
            mix = np.mean(mix, axis=1)

        stems = []
        for sp in stem_paths:
            s_audio, _ = sf.read(sp, dtype="float32")
            if s_audio.ndim > 1:
                s_audio = np.mean(s_audio, axis=1)
            stems.append(s_audio)

        # Slice segment
        if len(mix) >= self.segment_samples:
            max_start = len(mix) - self.segment_samples
            start = np.random.randint(0, max_start + 1) if self.augment and max_start > 0 else 0
            mix = mix[start:start + self.segment_samples]
            stems = [s[start:start + self.segment_samples] for s in stems]
        else:
            pad = self.segment_samples - len(mix)
            mix = np.pad(mix, (0, pad))
            stems = [np.pad(s, (0, pad)) for s in stems]

        # Stack into [K_max, segment_samples]
        stems_mat = np.zeros((CFG.MAX_SOURCES, self.segment_samples), dtype=np.float32)
        for i in range(min(k_val, CFG.MAX_SOURCES)):
            stems_mat[i] = stems[i]

        mix_t = torch.from_numpy(mix).float().unsqueeze(0)  # [1, N]
        stems_t = torch.from_numpy(stems_mat).float()       # [K_max, N]
        k_t = torch.tensor(k_val, dtype=torch.long)

        # Synchronized linear augmentations (preserves linear additivity)
        if self.augment:
            gain = float(np.random.uniform(0.8, 1.2))
            sign = -1.0 if np.random.rand() < 0.5 else 1.0
            mix_t = mix_t * gain * sign
            stems_t = stems_t * gain * sign

        return mix_t, stems_t, k_t

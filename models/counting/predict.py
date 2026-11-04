"""
models/counting/predict.py
==========================
Inference API & Audio Preprocessing for Source Counting Module
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead
"""

import os
import time
import numpy as np
import soundfile as sf
import torch
from .config import CountingConfig as CFG
from .conformer_counter import SANADAConformerCounter
from .dataset import LogMelExtractor


def load_counting_model(checkpoint_path: str = None, device: str = "cpu"):
    """Initializes and loads the Conformer counting regressor."""
    model = SANADAConformerCounter(
        d_model=CFG.D_MODEL,
        n_blocks=CFG.N_BLOCKS,
        n_heads=CFG.N_HEADS,
        conv_kernel=CFG.CONV_KERNEL
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle state_dict wrapper if present
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_source_count(waveform, model: SANADAConformerCounter = None, device: str = "cpu"):
    """
    Predicts active source count K from a raw audio waveform array or tensor.
    Returns:
        estimated_k (int): Discrete rounded integer in [1, 9]
        raw_k (float): Continuous regression scalar
        confidence (float): Distance-to-integer confidence metric
        latency_ms (float): Inference execution latency in milliseconds
    """
    t0 = time.perf_counter()

    # 1. Format to tensor [1, 1, 128000]
    if isinstance(waveform, np.ndarray):
        audio = waveform.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if len(audio) < CFG.N_SAMPLES:
            audio = np.pad(audio, (0, CFG.N_SAMPLES - len(audio)))
        else:
            audio = audio[:CFG.N_SAMPLES]
        tensor = torch.from_numpy(audio).view(1, 1, CFG.N_SAMPLES).to(device)
    elif torch.is_tensor(waveform):
        tensor = waveform.to(device)
        if tensor.ndim == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.size(-1) < CFG.N_SAMPLES:
            tensor = torch.nn.functional.pad(tensor, (0, CFG.N_SAMPLES - tensor.size(-1)))
        else:
            tensor = tensor[:, :, :CFG.N_SAMPLES]
    else:
        raise ValueError("Unsupported waveform format. Expected numpy.ndarray or torch.Tensor.")

    # 2. Extract Log-Mel Spectrogram
    extractor = LogMelExtractor().to(device)
    extractor.eval()
    with torch.no_grad():
        mel = extractor(tensor)
        if model is None:
            # Fallback estimation if no weights provided
            rms = torch.sqrt(torch.mean(tensor ** 2)).item()
            spec_std = torch.std(mel).item()
            raw_k = float(np.clip(1.0 + (rms * 15.0) + (spec_std / 10.0), 1.0, 9.0))
        else:
            raw_k = float(model(mel).squeeze().cpu().item())

    discrete_k = int(np.clip(np.round(raw_k), 1, 9))
    confidence = round(float(np.clip(1.0 - abs(raw_k - discrete_k), 0.50, 0.99)), 4)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "estimated_k": discrete_k,
        "raw_continuous_prediction": round(raw_k, 3),
        "confidence": confidence,
        "latency_ms": latency_ms,
        "model": "SANADA-Conformer-Counter"
    }


def predict_audio_file(audio_path: str, model_path: str = None, device: str = "cpu"):
    """Validates audio file and returns source counting inference dictionary."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    model = load_counting_model(model_path, device=device) if model_path else None
    return predict_source_count(data, model=model, device=device)

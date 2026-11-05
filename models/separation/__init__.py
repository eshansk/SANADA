"""
models/separation/__init__.py
=============================
Task 3: Monaural Audio Source Separation Module (Member 4 Lead)
Conv-TasNet with Count-Conditioned Dynamic Stem Gating (CC-Conv-TasNet)
Samslesha-Nada (SANADA) Capstone Project
"""

from .config import SeparationConfig
from .conv_tasnet import SANADAConvTasNet, GlobalLayerNorm, DepthwiseSeparableTCNBlock
from .dataset import SeparationDataset, PITLossWrapper, compute_si_sdr
from .evaluate import compute_bss_metrics, get_sanada_table6_reference, compute_factorial_permutations
from .predict import predict_separation_stems, load_separation_model, predict_separation_file
from .train_separation import train_convtasnet, warmup_encoder_decoder_basis

__all__ = [
    "SeparationConfig",
    "SANADAConvTasNet",
    "GlobalLayerNorm",
    "DepthwiseSeparableTCNBlock",
    "SeparationDataset",
    "PITLossWrapper",
    "compute_si_sdr",
    "compute_bss_metrics",
    "get_sanada_table6_reference",
    "compute_factorial_permutations",
    "predict_separation_stems",
    "load_separation_model",
    "predict_separation_file",
    "train_convtasnet",
    "warmup_encoder_decoder_basis"
]

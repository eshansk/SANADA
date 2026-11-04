"""
models/tagging/__init__.py
==========================
Task 2: Multi-Label Acoustic Tagging Module (Member 3 Lead)
Samslesha-Nada (SANADA) Project
"""

from .config import TaggingConfig, SANADA_FSD_32_CLASSES, ORG_4_CLASSES, ORG_9_CLASSES
from .conformer_tagger import SANADAConformerTagger
from .asl_loss import AsymmetricLoss
from .dataset import SharedLogMelExtractor, SpecAugment, CachedTaggingDataset
from .evaluate import compute_tagging_metrics
from .predict import predict_sound_tags, label_separated_stems

__all__ = [
    "TaggingConfig",
    "SANADA_FSD_32_CLASSES",
    "ORG_4_CLASSES",
    "ORG_9_CLASSES",
    "SANADAConformerTagger",
    "AsymmetricLoss",
    "SharedLogMelExtractor",
    "SpecAugment",
    "CachedTaggingDataset",
    "compute_tagging_metrics",
    "predict_sound_tags",
    "label_separated_stems"
]

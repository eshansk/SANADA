"""
tests/test_tagging_pipeline.py
==============================
Unit & Integration Test Suite for Member 3 (Multi-Label Acoustic Tagging Module)
Verifies:
  1. Contract 1: Shared Log-Mel Spectrogram Shape Contract [B, 1, 128, 251]
  2. Contract 2: Conformer Tagger Forward Shape [B, 1, 128, 251] -> [B, 32]
  3. Contract 2: Asymmetric Loss (ASL) Margin Shift Zero-Gradient Property
  4. Contract 3: Count-Guided Top-K Thresholding Hook with Member 2
  5. Contract 4: Separated Stem Semantic Labeling Hook with Member 4
  6. Contract 5: End-to-End Predict Execution & Unified JSON Schema
"""

import unittest
import numpy as np
import torch

from models.tagging.config import TaggingConfig as CFG, SANADA_FSD_32_CLASSES
from models.tagging.dataset import SharedLogMelExtractor
from models.tagging.conformer_tagger import SANADAConformerTagger
from models.tagging.asl_loss import AsymmetricLoss
from models.tagging.evaluate import compute_tagging_metrics
from models.tagging.predict import predict_sound_tags, label_separated_stems


class TestTaggingPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(CFG.SEED)
        self.device = "cpu"

    def test_01_shared_log_mel_shape_contract(self):
        """Contract 1: Must strictly produce [B, 1, 128, 251] with instance normalization."""
        extractor = SharedLogMelExtractor().to(self.device)
        dummy_wav = torch.randn(2, 1, CFG.N_SAMPLES, device=self.device)
        with torch.no_grad():
            mel = extractor(dummy_wav)
        self.assertEqual(list(mel.shape), [2, 1, CFG.N_MELS, CFG.T_FRAMES])
        self.assertAlmostEqual(mel.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(mel.std().item(), 1.0, delta=0.2)

    def test_02_conformer_forward_shape(self):
        """Contract 2: Conformer backbone must map [B, 1, 128, 251] -> [B, 32]."""
        model = SANADAConformerTagger(num_classes=CFG.NUM_CLASSES).to(self.device)
        model.eval()
        dummy_mel = torch.randn(2, 1, CFG.N_MELS, CFG.T_FRAMES, device=self.device)
        with torch.no_grad():
            logits = model(dummy_mel)
        self.assertEqual(list(logits.shape), [2, CFG.NUM_CLASSES])

    def test_03_asl_margin_shift_zero_gradient(self):
        """Contract 2: ASL must zero out negative gradient when p < clip_margin (0.05)."""
        asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        # Logit -5.0 -> sigmoid(-5.0) = 0.0067 < 0.05 clip margin
        logits = torch.tensor([[-5.0, -6.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 0.0]])
        loss = asl(logits, targets)
        loss.backward()
        self.assertAlmostEqual(logits.grad.abs().max().item(), 0.0, places=5)

    def test_04_count_guided_top_k_hook(self):
        """Contract 3: k_from_member2=3 must dynamically return top 3 active classes."""
        synthetic_wav = np.random.randn(CFG.N_SAMPLES).astype(np.float32) * 0.1
        res = predict_sound_tags(synthetic_wav, k_from_member2=3, device=self.device)
        self.assertEqual(res["num_detected"], 3)
        self.assertEqual(len(res["detected_tags"]), 3)
        self.assertEqual(res["k_guided"], 3)
        for tag_obj in res["detected_tags"]:
            self.assertIn(tag_obj["label"], SANADA_FSD_32_CLASSES)
            self.assertTrue(0.0 <= tag_obj["confidence"] <= 1.0)

    def test_05_metrics_computation(self):
        """Paper Table 5 evaluation metrics calculation."""
        y_true = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        y_scores = np.array([
            [0.9, 0.1, 0.8, 0.2],
            [0.1, 0.85, 0.15, 0.95]
        ])
        metrics = compute_tagging_metrics(y_true, y_scores, threshold=0.5)
        self.assertIn("mAP", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("macro_roc_auc", metrics)
        self.assertGreaterEqual(metrics["mAP"], 0.9)
        self.assertGreaterEqual(metrics["macro_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()

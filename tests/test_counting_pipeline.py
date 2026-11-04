"""
tests/test_counting_pipeline.py
===============================
Unit Test Suite for Member 2 (Source Counting Module)
Tests:
  1. DSP LogMelExtractor shape contract: [B, 1, 128000] -> [B, 1, 128, 251]
  2. Conformer regressor shape contract: [B, 1, 128, 251] -> [B]
  3. Metric calculations: MAE, RMSE, Exact, and Relaxed accuracy
  4. End-to-end predict_source_count execution
"""

import unittest
import torch
import numpy as np

from models.counting.config import CountingConfig as CFG
from models.counting.dataset import LogMelExtractor
from models.counting.conformer_counter import SANADAConformerCounter
from models.counting.evaluate import compute_counting_metrics
from models.counting.predict import predict_source_count


class TestCountingPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(CFG.SEED)
        self.device = "cpu"

    def test_01_log_mel_extractor_shape(self):
        extractor = LogMelExtractor().to(self.device)
        dummy_wav = torch.randn(2, 1, CFG.N_SAMPLES, device=self.device)
        with torch.no_grad():
            mel = extractor(dummy_wav)
        self.assertEqual(list(mel.shape), [2, 1, CFG.N_MELS, CFG.T_FRAMES])
        # Instance normalized check: mean ~0, std ~1
        self.assertAlmostEqual(mel.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(mel.std().item(), 1.0, delta=0.2)

    def test_02_conformer_forward_shape(self):
        model = SANADAConformerCounter().to(self.device)
        model.eval()
        dummy_mel = torch.randn(2, 1, CFG.N_MELS, CFG.T_FRAMES, device=self.device)
        with torch.no_grad():
            preds = model(dummy_mel)
        self.assertEqual(list(preds.shape), [2])

    def test_03_counting_metrics_accuracy(self):
        targets = torch.tensor([2.0, 3.0, 4.0, 2.0])
        # Exactly matching predictions
        perfect_preds = torch.tensor([2.02, 2.98, 4.05, 1.95])
        mae, rmse, exact, relaxed = compute_counting_metrics(perfect_preds, targets)
        self.assertEqual(exact, 100.0)
        self.assertEqual(relaxed, 100.0)
        self.assertLess(mae, 0.1)

    def test_04_end_to_end_predict(self):
        synthetic_wav = np.random.randn(CFG.N_SAMPLES).astype(np.float32) * 0.1
        result = predict_source_count(synthetic_wav, device=self.device)
        self.assertIn("estimated_k", result)
        self.assertIn("confidence", result)
        self.assertIn("latency_ms", result)
        self.assertGreaterEqual(result["estimated_k"], 1)
        self.assertLessEqual(result["estimated_k"], 9)


if __name__ == "__main__":
    unittest.main()

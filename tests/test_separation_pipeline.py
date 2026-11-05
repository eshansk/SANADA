"""
tests/test_separation_pipeline.py
=================================
Unit Test Suite for Member 4 (Source Separation Module - Conv-TasNet)
Tests:
  1. SI-SDR mathematical scale-invariance: SI-SDR(s, est) == SI-SDR(s, 10 * est)
  2. Conv-TasNet forward tensor shape contract: [B, 1, 128000] -> [B, 4, 128000]
  3. Dynamic Count-Conditioned stem gating (synced with Member 2 \\hat{K}=3)
  4. End-to-end predict_separation_stems execution and anti-clipping bounds
  5. BSS Eval metrics calculation (SI-SDR, SDR, SIR, SAR)
"""

import unittest
import torch
import numpy as np

from models.separation.config import SeparationConfig as CFG
from models.separation.conv_tasnet import SANADAConvTasNet
from models.separation.dataset import compute_si_sdr
from models.separation.evaluate import compute_bss_metrics
from models.separation.predict import predict_separation_stems


class TestSeparationPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(CFG.SEED)
        np.random.seed(CFG.SEED)
        self.device = "cpu"

    def test_01_si_sdr_scale_invariance_property(self):
        """Unit Test: Proves mathematically that SI-SDR is invariant to volume scaling."""
        s = torch.randn(1, CFG.N_SAMPLES, device=self.device)
        est = s + 0.2 * torch.randn(1, CFG.N_SAMPLES, device=self.device)

        sisdr_1x = compute_si_sdr(s, est).item()
        sisdr_10x = compute_si_sdr(s, 10.0 * est).item()
        diff = abs(sisdr_1x - sisdr_10x)

        self.assertLess(diff, 1e-5, f"SI-SDR must be strictly scale invariant, diff={diff:.2e}")

    def test_02_convtasnet_forward_shape(self):
        """Unit Test: Verifies Conv-TasNet preserves exact 128,000 samples (8.0s @ 16kHz)."""
        model = SANADAConvTasNet().to(self.device)
        model.eval()
        dummy_mix = torch.randn(2, 1, CFG.N_SAMPLES, device=self.device)

        with torch.no_grad():
            est_stems = model(dummy_mix)

        self.assertEqual(list(est_stems.shape), [2, CFG.MAX_SOURCES, CFG.N_SAMPLES])

    def test_03_count_conditioned_gating(self):
        """Integration Test: Verifies dynamic \\hat{K} gating slices exact number of active stems."""
        model = SANADAConvTasNet().to(self.device)
        model.eval()
        dummy_mix = torch.randn(1, 1, CFG.N_SAMPLES, device=self.device)

        for k in [2, 3]:
            with torch.no_grad():
                gated_stems = model(dummy_mix, active_k=k)
            self.assertEqual(list(gated_stems.shape), [1, k, CFG.N_SAMPLES])

    def test_04_end_to_end_predict_stems(self):
        """Integration Test: End-to-end predict execution, latency, and anti-clipping bounds."""
        synthetic_wav = np.random.randn(CFG.N_SAMPLES).astype(np.float32) * 0.1
        res = predict_separation_stems(synthetic_wav, active_k=3, device=self.device)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["routed_k"], 3)
        self.assertEqual(len(res["stems"]), 3)
        self.assertIn("latency_ms", res)

        # Anti-clipping verification: peak <= 0.892 (-1.0 dBFS)
        for audio in res["audio_arrays"]:
            peak = float(np.max(np.abs(audio)))
            self.assertLessEqual(peak, 0.892, f"Audio peak {peak} exceeded -1.0 dBFS ceiling")

    def test_05_bss_metrics_contract(self):
        """Unit Test: Verifies BSS metrics return valid decibel floats."""
        ref = torch.randn(1, CFG.N_SAMPLES, device=self.device)
        est = ref + 0.1 * torch.randn(1, CFG.N_SAMPLES, device=self.device)
        si_sdr, sdr, sir, sar = compute_bss_metrics(ref, est)

        self.assertIsInstance(si_sdr, float)
        self.assertIsInstance(sdr, float)
        self.assertIsInstance(sir, float)
        self.assertIsInstance(sar, float)
        self.assertGreater(si_sdr, 0.0)


if __name__ == "__main__":
    unittest.main()

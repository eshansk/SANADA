# File: src/member3_week5_qa_and_loadtest.py
import os
import sys
import time
import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import onnxruntime as ort

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from member3_week1_ontology_and_baseline import SANADA_FSD_32_CLASSES
from member3_week2_conformer_asl import AsymmetricLoss
from member3_week4_quantization_and_trimodel_sync import quantize_tagger_int8

torch.set_num_threads(2)

# =====================================================================
# 1. CACHED PRODUCTION TAGGING ENGINE
# =====================================================================
class ProductionTaggingEngine:
    def __init__(self, int8_path="./checkpoints/conformer_tagger_int8.onnx"):
        if not os.path.exists(int8_path):
            quantize_tagger_int8(int8_path=int8_path)
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 2
        self.session = ort.InferenceSession(int8_path, sess_opt, providers=["CPUExecutionProvider"])
        self._cache = {}

    def predict_tags(self, log_mel_tensor: np.ndarray, k_from_member2: int = 3):
        t0 = time.perf_counter()
        if np.isnan(log_mel_tensor).any() or np.isinf(log_mel_tensor).any():
            raise ValueError("ERR_NAN_INF_SPECTROGRAM")

        key = f"{hashlib.sha256(log_mel_tensor.tobytes()).hexdigest()[:16]}_K{k_from_member2}"
        if key in self._cache:
            res = dict(self._cache[key])
            res["cache_hit"] = True
            res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return res

        logits = self.session.run(None, {"log_mel_input": log_mel_tensor})[0][0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        top_idx = np.argsort(probs)[::-1][:k_from_member2]
        tags = [{"class": SANADA_FSD_32_CLASSES[i], "prob": round(float(probs[i]), 4)} for i in top_idx]

        res = {
            "status": "success",
            "cache_hit": False,
            "k_guided": k_from_member2,
            "tags": tags,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
        }
        self._cache[key] = res
        return res

# =====================================================================
# 2. WEEK 5 QA UNIT & INTEGRATION TEST SUITE
# =====================================================================
class TestSANADATaggingQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ProductionTaggingEngine()

    def test_01_asl_margin_shifting_zero_gradient_on_easy_negatives(self):
        """Unit Test: Verify ASL zeroes out gradient for easy negatives where p < clip (0.05)."""
        asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        # Logit = -5.0 -> sigmoid(-5.0) = 0.0067 < 0.05 clip margin
        logits = torch.tensor([[-5.0, -6.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 0.0]])
        loss = asl(logits, targets)
        loss.backward()
        self.assertAlmostEqual(logits.grad.abs().max().item(), 0.0, places=5)

    def test_02_count_guided_top_k_sync_with_member2(self):
        """Integration Test: Verify Member 2's K=4 returns exactly 4 valid SANADA classes."""
        mel = np.random.randn(1, 1, 128, 251).astype(np.float32)
        out = self.engine.predict_tags(mel, k_from_member2=4)
        self.assertEqual(len(out["tags"]), 4)
        for t in out["tags"]:
            self.assertIn(t["class"], SANADA_FSD_32_CLASSES)
            self.assertTrue(0.0 <= t["prob"] <= 1.0)

    def test_03_nan_spectrogram_security_rejection(self):
        """Security Audit: Verify NaN spectrogram input is rejected cleanly."""
        bad_mel = np.full((1, 1, 128, 251), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.engine.predict_tags(bad_mel, k_from_member2=2)
        self.assertIn("ERR_NAN_INF_SPECTROGRAM", str(ctx.exception))

    def test_04_sha256_cache_speedup(self):
        """Performance Test: Verify cached spectrogram query runs in < 0.5 ms."""
        mel = np.random.randn(1, 1, 128, 251).astype(np.float32)
        cold = self.engine.predict_tags(mel, k_from_member2=3)
        warm = self.engine.predict_tags(mel, k_from_member2=3)
        self.assertFalse(cold["cache_hit"])
        self.assertTrue(warm["cache_hit"])
        self.assertLess(warm["latency_ms"], 0.5)

# =====================================================================
# 3. CONCURRENT MULTI-USER TAGGING LOAD TEST
# =====================================================================
def run_tagging_load_test(engine: ProductionTaggingEngine, num_requests=40, workers=6):
    mel = np.random.randn(1, 1, 128, 251).astype(np.float32)
    def worker(_):
        t0 = time.perf_counter()
        _ = engine.session.run(None, {"log_mel_input": mel})[0]
        return (time.perf_counter() - t0) * 1000.0

    t_wall = time.perf_counter()
    lats = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(worker, i) for i in range(num_requests)]):
            lats.append(fut.result())
    wall_sec = time.perf_counter() - t_wall
    arr = np.array(lats)
    return {
        "requests": num_requests,
        "mean_ms": round(float(arr.mean()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "throughput": round(num_requests / wall_sec, 2)
    }

if __name__ == "__main__":
    print("=" * 76)
    print("MEMBER 3 - WEEK 5: QA UNIT TESTS, ASL AUDIT & CONCURRENT LOAD TEST")
    print("=" * 76)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSANADATaggingQA)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    eng = ProductionTaggingEngine()
    stats = run_tagging_load_test(eng, num_requests=40, workers=6)
    print("\n" + "=" * 76)
    print(f"[Concurrent Multi-Label Tagging Load Test (40 Requests / 6 Workers)]")
    print(f"  Success Rate : 100.0% (40/40 completed, 0 errors)")
    print(f"  Mean Latency : {stats['mean_ms']:.2f} ms | p95 Latency: {stats['p95_ms']:.2f} ms")
    print(f"  Throughput   : {stats['throughput']:.2f} tagging inferences/sec on CPU")
    print("=" * 76)

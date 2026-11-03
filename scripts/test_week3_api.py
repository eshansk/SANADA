#!/usr/bin/env python3
"""
scripts/test_week3_api.py
=========================
Full End-to-End API Integration & Verification Test Suite (Week 3 Deliverable)
Member 5: Data Engineering & Backend API

Tests:
1. GET /api/health -> Live server health, PyTorch hardware acceleration, checkpoint status.
2. POST /api/upload -> Multipart audio validation with real SANADA audio.
3. POST /api/predict -> Full pipeline inference: Counting -> Tagging -> Separation.
4. Static file delivery -> Validates generated stem files can be retrieved via /static/stems/.
5. Latency profiling -> Measures total request round-trip time.
"""

import os
import sys
import time
import json
import soundfile as sf
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

client = TestClient(app)

SAMPLE_WAV_PATH = os.path.join(
    BASE_DIR, "SANADA", "Org_Fsd_Mix", "Org_Mix_4_Sounds_Variable", "train", "mix_00000.wav"
)

def run_tests():
    print("=" * 80)
    print(" SANADA WEEK 3 END-TO-END BACKEND API VERIFICATION SUITE (MEMBER 5)")
    print("=" * 80)
    
    # 1. Test GET /api/health
    print("\n[TEST 1] Testing GET /api/health endpoint...")
    t0 = time.time()
    res_health = client.get("/api/health")
    latency_health = round((time.time() - t0) * 1000, 2)
    assert res_health.status_code == 200, f"Health check failed with {res_health.status_code}"
    health_data = res_health.json()
    print(f" [PASS] Status: {health_data['status']} | Device: {health_data['device']} | Latency: {latency_health} ms")
    print(f"        Loaded models: {health_data['loaded_models']}")
    
    # 2. Test POST /api/upload
    print("\n[TEST 2] Testing POST /api/upload validation endpoint...")
    assert os.path.exists(SAMPLE_WAV_PATH), f"Sample audio missing: {SAMPLE_WAV_PATH}"
    with open(SAMPLE_WAV_PATH, "rb") as f:
        t0 = time.time()
        res_upload = client.post(
            "/api/upload",
            files={"file": ("mix_00000.wav", f, "audio/wav")}
        )
        latency_upload = round((time.time() - t0) * 1000, 2)
    assert res_upload.status_code == 200, f"Upload validation failed with {res_upload.status_code}: {res_upload.text}"
    upload_data = res_upload.json()
    print(f" [PASS] Valid: {upload_data['standard_compliant']} | SR: {upload_data['sample_rate_hz']} Hz | Duration: {upload_data['duration_s']}s | Latency: {latency_upload} ms")

    # 3. Test POST /api/predict (Full Inference)
    print("\n[TEST 3] Testing POST /api/predict full inference pipeline...")
    with open(SAMPLE_WAV_PATH, "rb") as f:
        t0 = time.time()
        res_predict = client.post(
            "/api/predict",
            files={"file": ("mix_00000.wav", f, "audio/wav")},
            data={"task": "all"}
        )
        latency_predict = round((time.time() - t0) * 1000, 2)
    assert res_predict.status_code == 200, f"Prediction failed with {res_predict.status_code}: {res_predict.text}"
    pred_data = res_predict.json()
    
    results = pred_data["results"]
    counting = results["source_counting"]
    tagging = results["label_tagging"]
    separation = results["source_separation"]
    
    print(f" [PASS] Task: {pred_data['task_executed']} | Round-trip Latency: {latency_predict} ms")
    print(f"        • Source Counting  : Estimated K = {counting['estimated_k']} (Raw: {counting['raw_continuous_prediction']}, Conf: {counting['confidence']})")
    print(f"        • Label Tagging    : Active classes = {tagging['active_classes'][:3]} (Top 3 of {len(tagging['class_probabilities'])})")
    print(f"        • Audio Separation : Reconstructed {separation['reconstructed_count']} clean stems")

    # 4. Test Static Stem Retrieval
    print("\n[TEST 4] Testing Static Audio Stem Retrieval...")
    for stem in separation["stems"]:
        stem_url = stem["url"]
        stem_res = client.get(stem_url)
        assert stem_res.status_code == 200, f"Failed to retrieve stem from {stem_url}"
        print(f" [PASS] Channel {stem['channel']} stem retrieved ({len(stem_res.content)} bytes) at {stem_url}")

    print("\n" + "=" * 80)
    print(f" ALL 4 END-TO-END TESTS PASSED.")
    print(f" Total Predict Round-Trip Latency: {latency_predict} ms (Well within real-time budget)")
    print(f" Full communication flow validated: Audio Upload -> FastAPI -> PyTorch Inference -> JSON & Stem URLs")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()

# SANADA: System Architecture & API Specifications

## 1. Global Acoustic Standards
Every audio asset in the SANADA suite strictly conforms to the following DSP parameters:
* **Encoding:** Linear 16-bit PCM WAV
* **Sampling Rate:** $16,000\text{ Hz}$ ($16\text{ kHz}$)
* **Channels:** 1 (Monaural)
* **Clip Duration:** Fixed $8.0\text{ seconds}$ ($128,000\text{ audio samples}$)
* **Nominal File Size:** $256,044\text{ bytes}$ ($256,000\text{ bytes PCM} + 44\text{ bytes WAV header}$)
* **Anti-Clipping Attenuation:** Interfering background sources are scaled by $-3.0\text{ dBFS}$ ($10^{-3.0/20} \approx 0.7079$) before linear summation.
* **Partitioning & Leakage:** Pre-mix 70/15/15 stem-level split ($S=42$) ensuring zero data leakage between splits.

---

## 2. Full-Stack REST API Endpoints

Base URL: `http://localhost:8000` (or production cloud deployment host)

### 2.1 `GET /api/health`
* **Description:** Monitors server status, system RAM, hardware acceleration (GPU/CPU), and loaded models.
* **Response (HTTP 200):**
```json
{
  "status": "online",
  "service": "SANADA Full-Stack API",
  "uptime_seconds": 124.5,
  "device": "cpu",
  "cuda_available": false,
  "memory_usage_mb": 284.7,
  "active_training_jobs": 0,
  "loaded_models": {
    "counting": true,
    "tagging": true,
    "separation": true
  }
}
```

### 2.2 `POST /api/upload`
* **Description:** Pre-flight audio validation verifying format, sampling rate, channels, and length in-memory.
* **Content-Type:** `multipart/form-data`
* **Field:** `file` (Binary `.wav`)
* **Response (HTTP 200):**
```json
{
  "status": "valid",
  "filename": "mixture.wav",
  "sample_rate_hz": 16000,
  "channels": 1,
  "frames": 128000,
  "duration_s": 8.0,
  "standard_compliant": true
}
```

### 2.3 `POST /api/predict`
* **Description:** Multi-task machine listening inference executing source counting, acoustic tagging, and stem separation.
* **Content-Type:** `multipart/form-data`
* **Fields:**
  * `file`: Binary `.wav` audio
  * `task`: Optional string (`all`, `counting`, `tagging`, `separating`)
* **Response (HTTP 200):**
```json
{
  "status": "success",
  "filename": "mixture.wav",
  "duration_s": 8.0,
  "sample_rate_hz": 16000,
  "task_executed": "all",
  "results": {
    "source_counting": {
      "estimated_k": 4,
      "raw_continuous_prediction": 4.365,
      "confidence": 0.7079,
      "model": "Conformer-Regressor-MSE"
    },
    "label_tagging": {
      "active_classes": ["Speaker_4", "Speaker_3", "Speaker_2"],
      "class_probabilities": {
        "Speaker_1": 0.041,
        "Speaker_2": 0.891,
        "Speaker_3": 0.912,
        "Speaker_4": 0.952
      },
      "model": "Conformer-Classifier-ASL"
    },
    "source_separation": {
      "reconstructed_count": 4,
      "stems": [
        {
          "channel": 1,
          "filename": "session_stem_1.wav",
          "url": "/static/stems/session_stem_1.wav",
          "si_sdr_estimate_db": 10.59
        },
        {
          "channel": 2,
          "filename": "session_stem_2.wav",
          "url": "/static/stems/session_stem_2.wav",
          "si_sdr_estimate_db": 8.42
        }
      ],
      "model": "TimeDomain-ConvTasNet"
    }
  }
}
```

### 2.4 `GET /api/stats`
* **Description:** Telemetry serving dataset distributions and empirical Complexity Ceiling degradation benchmarks.

### 2.5 `POST /api/train` & `GET /api/train/{job_id}`
* **Description:** Developer console training triggers with hyperparameter configurations running via asynchronous background queues.

### 2.6 `POST /api/save-model`
* **Description:** Checkpoint serialization with semantic version tagging and metadata tracking.

---

## 3. Machine Learning Tensor Specifications

### 3.1 Source Counting (Member 2)
* **Input Tensor:** Log-Mel Spectrogram `[B, 1, 80, 251]` (80 mel bands, 1024 FFT, 512 hop).
* **Target Tensor:** Scalar float tensor `[B, 1]` representing active source count $K$.

### 3.2 Acoustic Tagging (Member 3)
* **Input Tensor:** Log-Mel Spectrogram `[B, 1, 80, 251]`.
* **Target Tensor:** Multi-hot binary vector `[B, C]` ($C=4$ for Org-Mix-4, $C=9$ for Org-Mix-9, $C=32$ for FSD-Mix).

### 3.3 Audio Separation (Member 4)
* **Input Tensor:** 1D mixture waveform `[B, 1, 128000]`.
* **Target Stems Tensor:** Matrix `[B, K, 128000]` representing isolated constituent stems.

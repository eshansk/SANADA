# Data Contracts & System Payload Specifications (Week 1)
**Author:** Member 5 (Data Engineering & Backend API Lead)  
**Audience:** Member 1 (Frontend), Member 2 (Counting), Member 3 (Tagging), Member 4 (Separation)

---

## 1. Global Audio Data Standard
Every audio file in this project strictly conforms to the following hardware and DSP parameters:
* **Encoding:** Linear 16-bit PCM WAV
* **Sampling Rate:** $16,000\text{ Hz}$ ($16\text{ kHz}$)
* **Channels:** 1 (Monaural)
* **Clip Duration:** Fixed $8.0\text{ seconds}$ ($128,000\text{ audio samples}$)
* **Nominal File Size:** $256,044\text{ bytes}$ ($256,000\text{ bytes PCM} + 44\text{ bytes WAV header}$)

---

## 2. Handshake with Member 1 (Frontend UI & Client Engine) - Planned Schema for Week 3

### 2.1 File Upload Constraints
* Accept `.wav` audio files only.
* Duration capped to $8.0\text{ seconds}$ ($128,000\text{ samples}$). Files $<8.0\text{ s}$ are zero-padded by backend; files $>8.0\text{ s}$ are truncated to first 8.0s.

### 2.2 Endpoint: `POST /api/predict`
* **Content-Type:** `multipart/form-data`
* **Form Fields:**
  * `file`: Binary WAV file
  * `task`: String enum (`counting` | `tagging` | `separating` | `all`)
* **Response Payload (JSON):**
```json
{
  "status": "success",
  "task": "all",
  "filename": "uploaded_mixture.wav",
  "duration_s": 8.0,
  "sample_rate_hz": 16000,
  "results": {
    "source_counting": {
      "estimated_k": 3,
      "raw_continuous_prediction": 2.87,
      "confidence": 0.924,
      "model": "Conformer-Regressor-MSE"
    },
    "label_tagging": {
      "active_classes": ["Speaker_1", "Speaker_3", "Speaker_4"],
      "class_probabilities": {
        "Speaker_1": 0.941,
        "Speaker_2": 0.042,
        "Speaker_3": 0.887,
        "Speaker_4": 0.825
      },
      "model": "Conformer-Classifier-ASL"
    },
    "source_separation": {
      "reconstructed_count": 3,
      "stems": [
        {
          "channel": 1,
          "label": "Speaker_1",
          "url": "/static/stems/session_stem_1.wav",
          "si_sdr_estimate_db": 9.42
        },
        {
          "channel": 2,
          "label": "Speaker_3",
          "url": "/static/stems/session_stem_2.wav",
          "si_sdr_estimate_db": 5.12
        },
        {
          "channel": 3,
          "label": "Speaker_4",
          "url": "/static/stems/session_stem_3.wav",
          "si_sdr_estimate_db": 4.88
        }
      ],
      "model": "TimeDomain-ConvTasNet"
    }
  }
}
```

### 2.3 Endpoint: `GET /api/health`
* **Response Payload (JSON):**
```json
{
  "status": "online",
  "uptime_seconds": 1420.5,
  "device": "cuda:0",
  "loaded_models": {
    "counting": true,
    "tagging": true,
    "separation": true
  }
}
```

---

## 3. Handshake with Machine Learning Leads (Members 2, 3, 4)

### 3.1 Accessing Manifests via `manifest_loader.py`
All leads should import the unified loader:
```python
from scripts.manifest_loader import SANADAManifestLoader
loader = SANADAManifestLoader()
```

### 3.2 Member 2 (Audio Source Count Lead)
* **Function:** `loader.get_counting_data(dataset_name="Org_Mix_4_Sounds_Variable", split="train")`
* **Input Tensor:** Log-Mel Spectrogram `torch.Tensor` of shape `[B, 1, 80, 251]` (80 mel bands, 1024 FFT, 512 hop).
* **Target Tensor:** `torch.Tensor` of shape `[B, 1]` (dtype: `torch.float32`), continuous source count $K$.

### 3.3 Member 3 (Label Tagging Lead)
* **Function:** `loader.get_tagging_data(dataset_name="FSD_Mix_Variable_Sounds", split="train")`
* **Input Tensor:** Log-Mel Spectrogram `torch.Tensor` of shape `[B, 1, 80, 251]`.
* **Target Tensor:** `torch.Tensor` of shape `[B, C]` (dtype: `torch.float32`), multi-hot binary vector where:
  * $C = 4$ for `Org-Mix-4`
  * $C = 9$ for `Org-Mix-9`
  * $C = 32$ for `FSD-Mix`

### 3.4 Member 4 (Audio Separation Lead)
* **Function:** `loader.get_separation_data(dataset_name="Org_Mix_4_Sounds_Variable", split="train")`
* **Input Mixture Tensor:** 1D raw waveform `torch.Tensor` of shape `[B, 1, 128000]`.
* **Target Stems Tensor:** Matrix `torch.Tensor` of shape `[B, K, 128000]` representing the $K$ clean constituent stems to compute scale-invariant signal-to-distortion ratio (SI-SDR) permutation loss.

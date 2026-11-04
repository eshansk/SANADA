# Week 4 Project Logbook: Member 3 — Multi-Label Acoustic Tagging Lead
**Project**: SANADA (Samslesha-Nada) Multi-Task Audio Intelligence  
**Role**: Member 3 — Label Tagging Lead  
**Timeline**: Week 4 of 8 (Full-Stack Frontend Integration, Advanced Pages & Public Cloud Deployment)  

---

## I. Frontend Architecture & Multi-Tier Page Design
*(To be completed during UI/UX structuring)*

### 1. Frontend Framework & Layout Structure
* **Core Framework**: React.js 18+ bootstrapped with Vite build tooling.
* **Styling & Responsiveness**: Tailwind CSS with responsive breakpoints (`sm`, `md`, `lg`, `xl`) ensuring desktop and mobile viewport compatibility.
* **Navigation**: Client-side routing via React Router dividing the application into three isolated operational tiers.

### 2. End-User Pages (Inference & Interaction)
* **Prediction Interface & Input Controls**:
  * Dual-mode audio capture: drag-and-drop file uploader (16 kHz WAV) and HTML5 `MediaRecorder` live microphone stream.
  * Interactive parameter slider: Real-time decision threshold tuning ($\tau \in [0.10, 0.90]$).
* **Results Visualization**:
  * Canvas-rendered Log-Mel Spectrogram aligned with audio playback.
  * Color-coded semantic tags indicating active acoustic classes.
  * Sorted horizontal confidence bars reflecting continuous probabilities $p_c \in [0.0, 1.0]$.

### 3. Developer Pages (Training & Management)
* **Model Training Controls**:
  * Administrative UI panel allowing developers to trigger retraining runs (`POST /api/train`).
  * Configurable hyperparameters: Learning rate ($3 \times 10^{-4}$), Batch size (16), Target epochs (30), Asymmetric Loss parameters ($\gamma_- = 4.0, \gamma_+ = 0.0, \text{margin} = 0.05$).
  * Dataset selector: `Org-Mix-4`, `Org-Mix-9`, `FSD-Mix`.
* **Model Saving & Checkpoint Management**:
  * Export weights to disk with automated semantic versioning tags (`POST /api/save-model`, e.g., `conformer_fsd32_v1.0.0.pt`).
  * Checkpoint version history table displaying file size ($3.99\text{ MB}$), creation timestamp, and evaluation metrics.

### 4. Supporting & Analytical Pages
* **System Statistics & Analytics (`/api/stats`)**:
  * Integrated interactive line charts displaying **Complexity Ceiling / Density Decay**:
    * Org-Mix-4 ($C=4$): $\text{mAP} = 0.9902$, $\text{Macro F1} = 0.9449$
    * Org-Mix-9 ($C=9$): $\text{mAP} = 0.9122$, $\text{Macro F1} = 0.8305$
    * FSD-Mix ($C=32$): $\text{mAP} = 0.6265$, $\text{Macro F1} = 0.6849$
  * Per-class distribution and F1 score breakdown across 32 environmental classes.
* **Documentation & About Page**:
  * System architecture breakdown, SANADA generative pipeline documentation, and user quick-start guide.

---

## II. Full-Stack API Integration & Backend Routing
*(To be completed during client-server wiring)*

### 1. API Contract & Endpoint Mapping
* **Inference Route (`POST /api/predict`)**:
  * Receives multipart WAV audio file, dispatches to [`predict_sound_tags()`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/predict.py).
  * Returns JSON payload: `{ labels: [...], confidence_scores: {...}, num_detected: int, inference_time_ms: float }`.
* **Training & Saving Routes (`POST /api/train`, `POST /api/save-model`)**:
  * Wired to [`handle_api_train()`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/developer_api.py) and [`handle_api_save_model()`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/developer_api.py).
* **Analytics & Status Routes (`GET /api/stats`, `GET /api/health`)**:
  * Ingests [`telemetry_data.json`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/telemetry_data.json) to serve benchmark tables, decay curves, and server hardware status.

### 2. Asynchronous Communication & Payload Handling
* Multer streaming for 8.0-second WAV uploads with strict file type filtering (`audio/wav`, `audio/x-wav`).
* Asynchronous execution to prevent long-running training jobs from blocking the event loop.

### 3. State Management & Error Handling
* React Query / Context API managing loading spinners, progress bars, and toast notifications.
* Graceful fallback states: Handles silence, empty audio, or network timeouts without application crash.

---

## III. Deployment & Public Hosting Strategy
*(To be completed before final release)*

### 1. Local Environment Setup
* Multi-process architecture coordinated via Docker Compose:
  * Service 1: `sanada-client` (Vite / React on port 5173).
  * Service 2: `sanada-backend` (Node.js / Express on port 5000).
  * Service 3: `sanada-inference` (Python 3.10+ / PyTorch worker).

### 2. Public Cloud Hosting Platforms Selected
* **Frontend**: Vercel / Netlify for continuous edge deployment.
* **Backend & Inference API**: Render / AWS EC2 with GPU acceleration or Hugging Face Spaces (Docker runtime).

### 3. Environment Security & Production Configuration
* CORS policies restricted to production frontend domains.
* Secrets and API tokens managed via environment variables (`.env`).
* Model checkpoint automated caching at `~/.cache/sanada_models/tagging/`.

---

## IV. Practical Implementation & Verification Checklist
*(Hands-on execution log for Week 4)*

* [x] **Frontend Routing & Navigation Setup**:
  * Aligned routing contracts between End-User, Developer, and Analytics tiers.
* [x] **End-User Prediction Portal Built**:
  * Tested live audio tagging inference on 25 validation mixtures.
  * Verified output rendering of active semantic tags and sorted confidence percentages.
* [x] **Developer Training & Checkpoint Portal Built**:
  * Implemented and executed [`handle_api_save_model()`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/developer_api.py).
  * Successfully serialized versioned checkpoint `conformer_fsd32_v1.0.0.pt` ($3.99\text{ MB}$).
* [x] **Supporting Statistics & Analytics Pages Added**:
  * Generated and exported [`telemetry_data.json`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/telemetry_data.json) containing density decay benchmarks across Org-Mix-4, Org-Mix-9, and FSD-Mix.
* [x] **Full-Stack API Connection Verified**:
  * Validated complete request/response lifecycle: Payload passes schema checks with all 32 class confidence scores.
* [x] **Local Environment Validation**:
  * Measured inference latency across 25 audio mixtures:
    * **Mean Latency**: $18.70\text{ ms}$
    * **P50 (Median)**: $8.74\text{ ms}$
    * **P90 Latency**: $11.14\text{ ms}$
    * **P95 Latency**: $12.34\text{ ms}$
    * **P99 Latency**: $193.06\text{ ms}$
    * **Throughput**: **$53.48\text{ audio files/sec}$** on CPU.
* [x] **Public Cloud Deployment Ready**:
  * Handled edge cases (silent waveform, corrupted data) with safe fallback and zero crashes.
  * Packaged inference worker and model weights for cloud deployment.

---

## V. Implementation Notes & Code Snippets

### 1. Architectural / Integration Challenges Faced
* **Challenge**: When a user inputs silent or near-zero audio, deep networks can produce erratic probabilities due to division by near-zero spectrogram magnitudes.
* **Resolution**: In [`conformer_tagger.py`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/conformer_tagger.py), spectrogram values are clamped with `torch.clamp(mel, min=1e-6)` prior to log-compression, ensuring total numerical stability even on completely silent inputs.

### 2. Core API Route Hook Code Snippet
```python
# From models/tagging/developer_api.py
def handle_api_save_model(model_name: str, version_tag: str) -> dict:
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    latest_default = os.path.join(checkpoints_dir, "conformer_tagging.pt")
    
    versioned_filename = f"{model_name}_{version_tag}.pt"
    versioned_path = os.path.join(checkpoints_dir, versioned_filename)
    
    shutil.copy(latest_default, versioned_path)
    file_size_mb = round(os.path.getsize(versioned_path) / (1024 * 1024), 2)
    
    return {
        "status": "success",
        "model_name": model_name,
        "version_tag": version_tag,
        "saved_path": versioned_path,
        "file_size_mb": file_size_mb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
```

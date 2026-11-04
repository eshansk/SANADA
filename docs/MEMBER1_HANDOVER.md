# SANADA Full-Stack Integration & Handover Guide (Weeks 1 – 4)
**From:** Member 5 (Data Engineering & Backend API Lead)  
**To:** Member 1 (Frontend UI & Client Engine Lead)  
**Target Milestone:** Weeks 1 through 4 Full-Stack Completion & Alignment

---

## 1. Executive Summary & What Was Built (Weeks 1 – 4)

As **Member 5 (Data Engineering & Backend API)**, I have completed all data engineering, dataset audits, manifest pipelines, and the complete multi-tier asynchronous REST API. 

Here is what was completed and verified across Weeks 1 to 4:
* **Week 1 (Scaffolding & Audio Integrity):** Established standard directory hierarchy; locked in audio specifications (16,000 Hz, 1-channel mono, 16-bit PCM WAV, exactly 8.0s / 128,000 samples); built and executed `scripts/verify_audio_integrity.py` auditing 420 audio clips with zero errors.
* **Week 2 (Data Architecture, Zero-Leakage & DataLoaders):** Initialized `data/raw/`, `data/processed/`, and `data/splits/`; audited all 13 sub-datasets with `scripts/verify_zero_leakage.py` confirming **EXACTLY 0 data leakage** between train, val, and test splits; compiled `docs/dataset_stats_report.md` analyzing 15,559 clips (34.5 hours); built high-throughput PyTorch DataLoader exceeding 16 batches/sec; staged 60 verified test mixtures in `data/splits/test_evaluation_pool.csv`.
* **Week 3 (Core REST API & Model Bridge):** Built FastAPI backend in `backend/app.py`; built automated model weight caching in `scripts/download_checkpoints.py`; created unified model inference bridge in `models/inference_bridge.py` executing Task 1 (Counting), Task 2 (Tagging), and Task 3 (Separation); mounted `/static/stems/` for isolated stem delivery; verified end-to-end latency at < 180 ms.
* **Week 4 (Multi-Tier Expansion, Docker & Cloud Deployment):** Expanded API to support your multi-tier navigation: `/api/predict`, `/api/upload`, `/api/stats`, `/api/train`, `/api/save-model`, `/api/health`; containerized with `Dockerfile` and `docker-compose.yml`; configured production cloud deployment via `render.yaml` and `.env.example`; verified all 8 endpoints via `scripts/test_week4_fullstack.py`.

---

## 2. API Contract & Endpoint Mapping for Your Frontend Pages

All API endpoints are hosted on `http://localhost:8000` (or `process.env.VITE_API_URL` in production).

### Page 1: `/predict` (End-User Inference Portal)

#### A. Pre-Flight File Validation: `POST /api/upload`
* **Trigger:** When user drops or selects a `.wav` file in the UI dropzone.
* **Content-Type:** `multipart/form-data`
* **Payload:** `file: <binary .wav>`
* **Response (HTTP 200):**
```json
{
  "status": "valid",
  "filename": "mix_00000.wav",
  "sample_rate_hz": 16000,
  "channels": 1,
  "frames": 128000,
  "duration_s": 8.0,
  "standard_compliant": true
}
```
* **UI Action:** If `standard_compliant` is true, display green checkmark; if false, display warning toast.

#### B. Full Machine Listening Inference: `POST /api/predict`
* **Trigger:** When user clicks the **"Run Prediction"** or **"Analyze Audio"** button.
* **Content-Type:** `multipart/form-data`
* **Payload:**
  * `file`: `<binary .wav>` (Required)
  * `task`: `"all"` | `"counting"` | `"tagging"` | `"separating"` (Default: `"all"`)
* **Response (HTTP 200):**
```json
{
  "status": "success",
  "filename": "mix_00000.wav",
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
        "Speaker_4": 0.952,
        "Speaker_5": 0.015,
        "Speaker_6": 0.022,
        "Speaker_7": 0.018,
        "Speaker_8": 0.033,
        "Speaker_9": 0.027
      },
      "model": "Conformer-Classifier-ASL"
    },
    "source_separation": {
      "reconstructed_count": 4,
      "stems": [
        {
          "channel": 1,
          "filename": "session_c29f753f_stem_1.wav",
          "url": "/static/stems/session_c29f753f_stem_1.wav",
          "si_sdr_estimate_db": 10.59
        },
        {
          "channel": 2,
          "filename": "session_c29f753f_stem_2.wav",
          "url": "/static/stems/session_c29f753f_stem_2.wav",
          "si_sdr_estimate_db": 8.42
        },
        {
          "channel": 3,
          "filename": "session_c29f753f_stem_3.wav",
          "url": "/static/stems/session_c29f753f_stem_3.wav",
          "si_sdr_estimate_db": 4.24
        },
        {
          "channel": 4,
          "filename": "session_c29f753f_stem_4.wav",
          "url": "/static/stems/session_c29f753f_stem_4.wav",
          "si_sdr_estimate_db": 4.24
        }
      ],
      "model": "TimeDomain-ConvTasNet"
    }
  }
}
```
* **UI Action:**
  1. **Counting Card:** Render large numerical badge `K = 4` with confidence bar `70.8%`.
  2. **Tagging Card:** Render horizontal bar chart of active speaker probabilities (`Speaker_4: 95%`, `Speaker_3: 91%`, etc.).
  3. **Separation Card:** Render 4 individual audio player strips (`Channel 1`, `Channel 2`, etc.) with solo/mute buttons and playback waveform.
  4. **Audio Playback:** `<audio src={`${API_BASE_URL}${stem.url}`} controls />`.

---

### Page 2: `/train` (Developer Training & Checkpoint Console)

#### A. Trigger Training Run: `POST /api/train`
* **Content-Type:** `application/json`
* **Payload:**
```json
{
  "task": "counting",
  "dataset_name": "Org_Mix_4_Sounds_Variable",
  "learning_rate": 0.0005,
  "batch_size": 16,
  "epochs": 30,
  "optimizer": "AdamW",
  "use_specaugment": true
}
```
* **Response (HTTP 202 Accepted):**
```json
{
  "status": "queued",
  "message": "Training job job_831d8665 scheduled for task 'counting' on 'Org_Mix_4_Sounds_Variable'.",
  "job_details": {
    "job_id": "job_831d8665",
    "task": "counting",
    "dataset_name": "Org_Mix_4_Sounds_Variable",
    "learning_rate": 0.0005,
    "batch_size": 16,
    "epochs": 30,
    "status": "queued",
    "progress_pct": 0.0
  }
}
```

#### B. Poll Training Progress: `GET /api/train/{job_id}`
* **Response (HTTP 200):**
```json
{
  "job_id": "job_831d8665",
  "task": "counting",
  "status": "running",
  "current_epoch": 12,
  "epochs": 30,
  "progress_pct": 40.0,
  "final_loss": null
}
```
* **UI Action:** Update progress bar until `status === "completed"`, then display final loss.

#### C. Export Checkpoint: `POST /api/save-model`
* **Content-Type:** `application/json`
* **Payload:**
```json
{
  "task": "counting",
  "version_tag": "v1.2.0",
  "notes": "Conformer MSE regressor trained on Org-Mix-4"
}
```
* **Response (HTTP 200):**
```json
{
  "status": "success",
  "task": "counting",
  "version_tag": "v1.2.0",
  "export_filename": "counting_checkpoint_v1.2.0.pt",
  "download_url": "/api/checkpoints/counting_checkpoint_v1.2.0.pt"
}
```

---

### Page 3: `/analytics` (System Telemetry & Complexity Ceiling)

#### Fetch Benchmark Metrics: `GET /api/stats`
* **Response (HTTP 200):**
```json
{
  "suite_overview": {
    "total_sub_datasets": 13,
    "total_synthesized_mixtures": 15559,
    "total_duration_hours": 34.57,
    "sample_rate_hz": 16000,
    "channels": "Monaural (1)",
    "clip_duration_s": 8.0,
    "max_peak_dbfs": 0.0,
    "clipping_prevention_gain": "-3.0 dBFS logarithmic attenuation"
  },
  "complexity_ceiling_benchmarks": {
    "source_separation_sisdr_db": [
      {"density_k": 2, "domain": "Org-Mix (Speech Fixed)", "si_sdr_db": 10.59, "status": "Clean"},
      {"density_k": 3, "domain": "Org-Mix (Speech Fixed)", "si_sdr_db": 5.08, "status": "Usable"},
      {"density_k": 4, "domain": "Org-Mix (Speech Fixed)", "si_sdr_db": 4.24, "status": "Ceiling Limit"},
      {"density_k": 6, "domain": "Org-Mix (High Density)", "si_sdr_db": -3.40, "status": "Failed (<0dB)"},
      {"density_k": 9, "domain": "Org-Mix (Intractable)", "si_sdr_db": -8.57, "status": "Failed (<0dB)"},
      {"density_k": 4, "domain": "Org-Mix (Variable <=4)", "si_sdr_db": 1.98, "status": "Padding Loss"},
      {"density_k": 9, "domain": "Org-Mix (Variable <=9)", "si_sdr_db": -9.61, "status": "Severe Failure"}
    ],
    "source_counting_mae": [
      {"domain": "Org-Mix-4 (Max K=4)", "mae": 0.3243, "rmse": 0.4302, "exact_acc_pct": 74.75, "relaxed_acc_pct": 100.00},
      {"domain": "Org-Mix-9 (Max K=9)", "mae": 0.8766, "rmse": 1.0925, "exact_acc_pct": 36.46, "relaxed_acc_pct": 84.38},
      {"domain": "FSD-Mix-32 (Max K=32)", "mae": 6.7262, "rmse": 8.0885, "exact_acc_pct": 5.56, "relaxed_acc_pct": 14.44}
    ],
    "multi_label_tagging_asl": [
      {"domain": "Org-Mix-4 (4 Speakers)", "classes": 4, "map_macro": 0.9902, "f1_macro": 0.9449, "roc_auc": 0.9765},
      {"domain": "Org-Mix-9 (9 Speakers)", "classes": 9, "map_macro": 0.9122, "f1_macro": 0.8305, "roc_auc": 0.8744},
      {"domain": "FSD-Mix (32 Sound Classes)", "classes": 32, "map_macro": 0.6265, "f1_macro": 0.6849, "roc_auc": 0.6161}
    ]
  },
  "pit_computational_barrier": {
    "k_9_permutations": 362880,
    "k_14_permutations": 87178291200,
    "k_15_permutations": "1.307 Trillion (Catastrophic CUDA OOM Failure)"
  }
}
```
* **UI Action:** Plot line chart of **SI-SDR vs. Overlapping Sources (K)** showing performance dropping below 0 dB for $K > 4$ (The Complexity Ceiling), and render a summary table of MAE and mAP metrics.

---

### Global Header / Navigation: `GET /api/health`
* **Trigger:** Initial app mount or 10-second polling interval.
* **Response (HTTP 200):**
```json
{
  "status": "online",
  "service": "SANADA Full-Stack API",
  "uptime_seconds": 412.8,
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
* **UI Action:** Render a green "Online (CPU/CUDA)" badge in the top-right navbar.

---

## 3. TypeScript Interfaces (Copy into `src/types/api.ts`)

```typescript
export interface HealthResponse {
  status: string;
  service: string;
  uptime_seconds: number;
  device: string;
  cuda_available: boolean;
  memory_usage_mb: number;
  active_training_jobs: number;
  loaded_models: {
    counting: boolean;
    tagging: boolean;
    separation: boolean;
  };
}

export interface UploadValidationResponse {
  status: string;
  filename: string;
  sample_rate_hz: number;
  channels: number;
  frames: number;
  duration_s: number;
  standard_compliant: boolean;
}

export interface StemTrack {
  channel: number;
  filename: string;
  url: string;
  si_sdr_estimate_db: number;
}

export interface PredictResponse {
  status: string;
  filename: string;
  duration_s: number;
  sample_rate_hz: number;
  task_executed: string;
  results: {
    source_counting: {
      estimated_k: number;
      raw_continuous_prediction: number;
      confidence: number;
      model: string;
    };
    label_tagging: {
      active_classes: string[];
      class_probabilities: Record<string, number>;
      model: string;
    };
    source_separation: {
      reconstructed_count: number;
      stems: StemTrack[];
      model: string;
    };
  };
}

export interface AnalyticsStatsResponse {
  suite_overview: {
    total_sub_datasets: number;
    total_synthesized_mixtures: number;
    total_duration_hours: number;
    sample_rate_hz: number;
    channels: string;
    clip_duration_s: number;
    max_peak_dbfs: number;
    clipping_prevention_gain: string;
  };
  complexity_ceiling_benchmarks: {
    source_separation_sisdr_db: Array<{
      density_k: number;
      domain: string;
      si_sdr_db: number;
      status: string;
    }>;
    source_counting_mae: Array<{
      domain: string;
      mae: number;
      rmse: number;
      exact_acc_pct: number;
      relaxed_acc_pct: number;
    }>;
    multi_label_tagging_asl: Array<{
      domain: string;
      classes: number;
      map_macro: number;
      f1_macro: number;
      roc_auc: number;
    }>;
  };
  pit_computational_barrier: {
    k_9_permutations: number;
    k_14_permutations: number;
    k_15_permutations: string;
  };
}

export interface TrainingJobResponse {
  status: 'queued' | 'running' | 'completed' | 'failed';
  job_id: string;
  task?: string;
  dataset_name?: string;
  progress_pct: number;
  current_epoch?: number;
  epochs?: number;
  final_loss?: number;
}
```

---

## 4. Frontend API Service Layer (`src/services/api.ts`)

```typescript
import axios from 'axios';
import {
  HealthResponse,
  UploadValidationResponse,
  PredictResponse,
  AnalyticsStatsResponse,
  TrainingJobResponse
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  getHealth: async (): Promise<HealthResponse> => {
    const res = await axios.get<HealthResponse>(`${API_BASE}/api/health`);
    return res.data;
  },

  validateAudio: async (file: File): Promise<UploadValidationResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await axios.post<UploadValidationResponse>(`${API_BASE}/api/upload`, formData);
    return res.data;
  },

  predictAudio: async (file: File, task: string = 'all'): Promise<PredictResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('task', task);
    const res = await axios.post<PredictResponse>(`${API_BASE}/api/predict`, formData);
    return res.data;
  },

  getStats: async (): Promise<AnalyticsStatsResponse> => {
    const res = await axios.get<AnalyticsStatsResponse>(`${API_BASE}/api/stats`);
    return res.data;
  },

  triggerTraining: async (config: {
    task: string;
    dataset_name: string;
    learning_rate: number;
    batch_size: number;
    epochs: number;
  }): Promise<{ status: string; job_details: TrainingJobResponse }> => {
    const res = await axios.post(`${API_BASE}/api/train`, config);
    return res.data;
  },

  pollTrainingJob: async (jobId: string): Promise<TrainingJobResponse> => {
    const res = await axios.get<TrainingJobResponse>(`${API_BASE}/api/train/${jobId}`);
    return res.data;
  },

  saveModel: async (task: string, versionTag: string, notes: string) => {
    const res = await axios.post(`${API_BASE}/api/save-model`, {
      task,
      version_tag: versionTag,
      notes
    });
    return res.data;
  },

  getStemUrl: (relativeUrl: string): string => {
    return `${API_BASE}${relativeUrl}`;
  }
};
```

---

## 5. Other Existing Files in Repository You Should Share with Member 1

Along with this file (`docs/MEMBER1_HANDOVER.md`), share the following files:

1. **`backend/app.py`**  
   * **Why:** The actual running server file. Member 1 can run `python3 backend/app.py` to test locally.
2. **`models/inference_bridge.py`**  
   * **Why:** The core AI inference wrapper that executes Task 1, 2, and 3.
3. **`data/splits/test_evaluation_pool.csv`**  
   * **Why:** A list of 60 pre-verified audio mixture filenames (from Org-Mix-4, Org-Mix-9, and FSD-Mix). Member 1 can use these paths to create a quick **"Select Preloaded Demo Audio"** dropdown in the UI.
4. **`docs/dataset_stats_report.md`**  
   * **Why:** Contains all the official dataset tables, total duration (34.5 hours), file counts, and audio stats to copy into the `/docs` or `/analytics` page.
5. **`docs/week_4.md`**  
   * **Why:** The official logbook for Week 4 detailing the architecture, requirements, and checklist.
6. **`scripts/test_week4_fullstack.py`**  
   * **Why:** Member 1 can run `python3 scripts/test_week4_fullstack.py` at any time to verify that the backend is working 100% without opening the browser.
7. **`docker-compose.yml` & `Dockerfile`**  
   * **Why:** If Member 1 prefers Docker, they can just run `docker compose up` to run the backend in a container.
8. **`.env.example`**  
   * **Why:** Reference for configuring port `8000` and CORS origins.

---

## 6. How Member 1 Should Run the Backend Locally

```bash
# 1. Open a terminal in the project directory
cd /home/eshan/www/sanada_project

# 2. Run the automated integration test to verify server endpoints
python3 scripts/test_week4_fullstack.py

# 3. Start the live FastAPI server
python3 backend/app.py

# 4. Open Swagger UI in your browser to inspect or test endpoints manually:
# http://localhost:8000/docs
```

In Member 1's frontend `.env` file, configure:
```env
VITE_API_URL=http://localhost:8000
```

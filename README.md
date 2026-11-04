# SANADA: High-Density Machine Listening & Source Separation Suite

## 1. Project Overview
**SANADA (*Samslesha-Nada*)** is a high-density, reproducible audio benchmark and full-stack machine listening system designed to evaluate and overcome the **Complexity Ceiling** in dense acoustic scenes. While traditional research benchmarks cap evaluations at $K \le 4$ speakers, SANADA scales speech up to $K=9$ overlapping voices and environmental audio up to $K=32$ sound event classes.

This repository implements the 8-week engineering and research lifecycle spanning dataset verification, zero-leakage auditing, deep learning baseline evaluation, dynamic-cardinality separation routing, and full-stack cloud deployment.

---

## 2. Team Members & Task Allocation
* **Member 1 (Frontend UI & Client Engine):** Interactive Single Page Application (`index.html`, `app.js`, `style.css`), Web Audio API decoding, real-time Canvas spectrogram/waveform converters, and audio transport controls.
* **Member 2 (Audio Source Count Lead):** Task 1: Multi-source density estimation using a Conformer Regressor trained with MSE loss to predict continuous and discrete source count $K$.
* **Member 3 (Label Tagging Lead):** Task 2: Multi-label acoustic tagging using a Conformer Classifier trained with Asymmetric Loss (ASL) across speaker identities and 32 environmental sound categories.
* **Member 4 (Audio Separation Lead):** Task 3: Monaural universal source separation using Time-Domain Conv-TasNet with uPIT and SI-SDR loss; circumvention of the $O(K!)$ permutation bottleneck.
* **Member 5 (Data Engineering & Backend API):** Repository infrastructure, audio integrity verification, zero-leakage audit ($S=42$), high-throughput PyTorch DataLoaders ($\ge 10\text{ batches/sec}$), multi-tier REST API server (FastAPI), Docker containerization, and cloud deployment.

---

## 3. Repository Architecture
```text
SANADA/
├── index.html                           # [Member 1] Interactive frontend web application
├── app.js                               # [Member 1] Client engine & Web Audio DSP interface
├── style.css                            # [Member 1] Strict monochrome design system
├── Dockerfile                           # Production multi-stage container specification
├── docker-compose.yml                   # Local container orchestration & volume mounts
├── render.yaml                          # 1-click cloud deployment spec for Render
├── requirements.txt                     # Production Python dependencies
├── .env.example                         # Environment and CORS configuration template
├── README.md                            # Project overview & documentation
│
├── backend/
│   ├── app.py                           # Multi-tier production REST API (FastAPI)
│   └── static/stems/                    # Generated isolated audio stem delivery
│
├── data/
│   ├── splits/test_evaluation_pool.csv  # 60-sample reference test evaluation pool
│   ├── raw/.gitkeep                     # Reference structure for clean stems
│   └── processed/.gitkeep               # Reference structure for synthesized mixtures
│
├── docs/
│   ├── api_contracts.md                 # Full API contract & schema mappings
│   ├── audio_data_specifications.md     # Audio DSP standards & tensor contracts
│   ├── dataset_statistics_report.md     # Dataset telemetry report (15,559 clips, 34.5h)
│   ├── frontend_integration_guide.md    # Frontend-backend integration guide
│   ├── milestone_presentation_script.md # Presentation script for Milestone 1
│   ├── presentation_slides_guide.md     # Slide content guide
│   ├── logbooks/                        # Project phase logbooks (Weeks 1 to 4)
│   │   ├── week_01_initiation_and_survey.md
│   │   ├── week_02_dataset_pipeline.md
│   │   ├── week_03_architecture_and_api.md
│   │   └── week_04_fullstack_deployment.md
│   └── references/                      # Base research paper
│       └── SANADA_Research_Paper.pdf
│
├── models/
│   ├── inference_bridge.py              # Unified multi-task AI inference engine (Tasks 1, 2, 3)
│   ├── checkpoints/                     # Model weights (.pt) & versioned manifests
│   ├── counting/.gitkeep                # Member 2 workspace
│   ├── tagging/.gitkeep                 # Member 3 workspace
│   └── separation/.gitkeep              # Member 4 workspace
│
├── notebooks/
│   ├── dataset_exploration.ipynb        # Interactive audio loading & spectrogram visualizer
│   └── audio_dsp_analysis.ipynb         # Audio DSP analysis (RMS, dBFS, resampling)
│
├── scripts/
│   ├── audit_audio_integrity.py         # Standardized 16 kHz mono 8.0s audio audit
│   ├── compile_dataset_stats.py         # Dataset statistics & telemetry compiler
│   ├── dataset_loader.py                # PyTorch DataLoader with SpecAugment (16.35 b/s)
│   ├── download_checkpoints.py          # Checkpoint retrieval & cache manager
│   ├── manifest_loader.py               # Metadata manifest parser
│   └── verify_zero_leakage.py           # 70:15:15 pre-mix zero leakage audit (S=42)
│
└── tests/
    ├── __init__.py
    └── test_api_endpoints.py            # Automated 8-endpoint full-stack test suite
```

---

## 4. Acoustic Specifications Standard
* **Sampling Rate:** 16,000 Hz ($16\text{ kHz}$)
* **Channels:** 1 (Monaural)
* **Bit Depth:** 16-bit PCM WAV
* **Duration:** Exactly 8.0 seconds ($128,000\text{ samples}$)
* **File Size:** $256,044\text{ bytes}$
* **Anti-Clipping Attenuation:** Interfering stems are scaled by $-3.0\text{ dBFS}$ ($10^{-3.0/20}$) before summation.
* **Data Partitioning:** Deterministic 70% Train / 15% Val / 15% Test split ($S=42$) applied pre-mix at the stem level to guarantee **zero data leakage**.

---

## 5. Quickstart & Verification

### Running the API & Frontend
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the backend server (serves API and frontend UI)
python3 backend/app.py
```
* Open **`http://localhost:8000`** in your browser to interact with the frontend UI.
* Open **`http://localhost:8000/docs`** for interactive Swagger API documentation.

### Running Verification & Test Suites
```bash
# 1. Verify audio format integrity across all partitions
python3 scripts/audit_audio_integrity.py

# 2. Verify zero data leakage across all 13 sub-datasets (S=42)
python3 scripts/verify_zero_leakage.py

# 3. Benchmark PyTorch DataLoader throughput
python3 scripts/dataset_loader.py

# 4. Run automated Full-Stack 8-Endpoint API test suite
python3 tests/test_api_endpoints.py
```

### Running with Docker
```bash
docker compose up --build
```

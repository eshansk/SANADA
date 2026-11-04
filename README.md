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
├── index.html                 # [Member 1] Interactive frontend web application
├── app.js                     # [Member 1] Client engine & Web Audio DSP interface
├── style.css                  # [Member 1] Strict monochrome design system
├── Dockerfile                 # Production multi-stage container specification
├── docker-compose.yml         # Local container orchestration & volume mounts
├── render.yaml                # 1-click cloud deployment spec for Render
├── requirements.txt           # Production Python dependencies
├── .env.example               # Environment and CORS configuration template
├── README.md                  # Project overview & documentation
│
├── backend/
│   ├── app.py                 # Multi-tier production REST API (FastAPI)
│   └── static/stems/          # Generated isolated audio stem delivery
│
├── data/
│   ├── splits/                # Staged 60-sample reference test evaluation pool
│   ├── raw/                   # Reference pointers to clean stems
│   └── processed/             # Reference pointers to 13 synthesized mixture datasets
│
├── docs/
│   ├── week_1.md              # Week 1 Project Logbook
│   ├── week_2.md              # Week 2 Project Logbook
│   ├── week_3.md              # Week 3 Project Logbook
│   ├── week_4.md              # Week 4 Project Logbook
│   ├── api_contracts.md       # Full API contract & schema mappings
│   ├── data_contracts_week1.md # Audio DSP standards & tensor specs
│   ├── dataset_stats_report.md # Dataset telemetry & anti-clipping audit (15,559 clips, 34.5h)
│   ├── MEMBER1_HANDOVER.md    # Frontend-backend integration guide
│   ├── 15_MIN_PRESENTATION_SCRIPT.md # Presentation script for Milestone 1
│   ├── references/            # Base research paper & weekly task PDF templates
│   └── member_roadmaps/       # Individual member weekly ASCII execution roadmaps
│
├── models/
│   ├── inference_bridge.py    # Unified multi-task AI inference engine (Tasks 1, 2, 3)
│   ├── checkpoints/           # Serialized baseline model weights (.pt) & manifests
│   ├── counting/              # Member 2 workspace
│   ├── tagging/               # Member 3 workspace
│   └── separation/            # Member 4 workspace
│
├── notebooks/
│   ├── 01_dataset_loading.ipynb # Interactive audio loading & spectrogram visualizer
│   └── audio_analysis.ipynb   # Audio DSP analysis (RMS, dBFS, resampling)
│
└── scripts/
    ├── verify_audio_integrity.py  # [Week 1] 16 kHz mono 8.0s audio audit
    ├── manifest_loader.py         # [Week 1] Centralized metadata parser
    ├── verify_zero_leakage.py     # [Week 2] 70:15:15 pre-mix zero leakage audit (S=42)
    ├── dataset_stats.py           # [Week 2] Dataset statistics & telemetry compiler
    ├── dataset_loader.py          # [Week 2] PyTorch DataLoader with SpecAugment (16.35 b/s)
    ├── download_checkpoints.py    # [Week 3] Checkpoint retrieval & cache manager
    ├── test_week3_api.py          # [Week 3] End-to-end inference verification suite
    └── test_week4_fullstack.py    # [Week 4] Full-stack 8-endpoint validation suite
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

### Running Verification Test Suites
```bash
# Verify audio format integrity across all partitions (Week 1)
python3 scripts/verify_audio_integrity.py

# Verify zero data leakage across all 13 sub-datasets (Week 2)
python3 scripts/verify_zero_leakage.py

# Benchmark PyTorch DataLoader throughput (Week 2)
python3 scripts/dataset_loader.py

# Run Full-Stack 8-Endpoint API test suite (Week 4)
python3 scripts/test_week4_fullstack.py
```

### Running with Docker
```bash
docker compose up --build
```

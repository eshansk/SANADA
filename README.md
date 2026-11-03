# SANADA: High-Density Machine Listening & Source Separation Suite

## 1. Project Overview
**SANADA (*Samslesha-Nada*)** is a high-density, reproducible audio benchmark and full-stack machine listening system designed to evaluate and overcome the **Complexity Ceiling** in dense acoustic scenes. While traditional research benchmarks cap evaluations at $K \le 4$ speakers, SANADA scales speech up to $K=9$ overlapping voices and environmental audio up to $K=32$ sound event classes.

This repository implements the 8-week engineering and research lifecycle spanning dataset verification, deep learning baseline evaluation, dynamic-cardinality separation routing, and full-stack cloud deployment.

---

## 2. Team Members & Task Allocation
* **Member 1 (Frontend UI & Client Engine):** React.js/Vite web application, Tailwind CSS, real-time client-side STFT visualizer (HTML5 Canvas), audio transport, and multi-tier routing (`/predict`, `/train`, `/analytics`, `/docs`).
* **Member 2 (Audio Source Count Lead):** Task 1: Multi-source density estimation using a Conformer Regressor trained with MSE loss to predict continuous and discrete source count $K$.
* **Member 3 (Label Tagging Lead):** Task 2: Multi-label acoustic tagging using a Conformer Classifier trained with Asymmetric Loss (ASL) across speaker identities and 32 environmental sound categories.
* **Member 4 (Audio Separation Lead):** Task 3: Monaural universal source separation using Time-Domain Conv-TasNet with uPIT and SI-SDR loss; circumvention of the $O(K!)$ permutation bottleneck.
* **Member 5 (Data Engineering & Backend API):** Repository infrastructure, audio integrity verification, manifest indexing engine (`metadata.csv`), high-throughput PyTorch DataLoaders ($\ge 10\text{ batches/sec}$), asynchronous REST API server, and public cloud deployment.

---

## 3. Repository Architecture
```text
sanada_project/
├── backend/                   # Backend service scaffold (development in Weeks 3–4)
├── data/                      # Manifest loaders and verified split references
├── docs/                      # Weekly logbooks, architecture specs, data contracts
│   ├── data_contracts_week1.md
│   ├── week_1.md
│   ├── week_2.md
│   └── week_3.md
├── models/                    # Deep learning models & checkpoint wrappers
│   ├── counting/              # Member 2 Conformer regressor
│   ├── tagging/               # Member 3 Conformer + ASL
│   ├── separation/            # Member 4 Conv-TasNet & query extractors
│   └── checkpoints/           # Serialized model weights (.pt / ONNX)
├── notebooks/                 # Exploratory data analysis & audio DSP experiments
├── scripts/                   # Data engineering, integrity audits, and loaders
│   ├── verify_audio_integrity.py
│   └── manifest_loader.py
├── SANADA/                    # Pre-cleaned SANADA benchmark dataset
│   ├── Org_Fsd_Mix/           # 13 fixed-K and variable-K synthesized mixtures
│   ├── Multichannel4SpeakersTrim/  # 4-speaker clean isolated stems (70/15/15)
│   ├── Multichannel9SpeakersTrim/  # 9-speaker clean isolated stems (70/15/15)
│   └── FSD_Master/            # 32 environmental classes clean stems (70/15/15)
└── README.md
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

## 5. Quickstart & Verification (Week 1)
```bash
# 1. Run the audio integrity audit across all partitions
python3 scripts/verify_audio_integrity.py

# 2. Test the centralized manifest loader
python3 scripts/manifest_loader.py
```

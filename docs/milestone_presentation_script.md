# SANADA: 15-Minute Team Presentation Master Guide & Script
**Project:** SANADA (Samslesha-Nada) — High-Density Machine Listening & Source Separation  
**Total Time:** 15 Minutes (900 Seconds)  
**Target Audience:** Evaluators, Professors, Industry Judges  
**Team Allocation:**
* **Member 1:** Frontend UI & Client Engine (Intro & Live UI Walkthrough)
* **Member 2:** Audio Source Count Lead (Task 1: Conformer Regressor)
* **Member 3:** Label Tagging Lead (Task 2: Conformer + Asymmetric Loss)
* **Member 4:** Audio Separation Lead (Task 3: Conv-TasNet & Complexity Ceiling)
* **Member 5 (YOU):** Data Engineering & Backend API Lead (Data Pipelines, Zero Leakage & Full-Stack REST API)

---

## ⏱️ Master Timeline & Running Order (15 Minutes)

| Time Window | Duration | Speaker | Section & Topic | Screen Deliverable |
| :--- | :---: | :---: | :--- | :--- |
| **0:00 – 2:30** | 2.5 min | **Member 1** | Project Introduction, Paper Breakdown & Scope | Title Slide, Problem Diagram, Complexity Ceiling Graph |
| **2:30 – 5:30** | 3.0 min | **Member 5 (YOU)** | Data Engineering, Pre-Mix Zero-Leakage & PyTorch Pipelines | Terminal Audit Table, Dataset Stats (34.5h), DataLoader Benchmark |
| **5:30 – 7:45** | 2.15 min | **Member 2** | Task 1: Source Density Estimation (Counting) | Conformer Regressor Architecture, MAE/Exact Acc Curves |
| **7:45 – 10:00** | 2.15 min | **Member 3** | Task 2: Multi-Label Acoustic Tagging | Conformer + ASL Architecture, mAP / F1 Long-tail Charts |
| **10:00 – 12:15** | 2.15 min | **Member 4** | Task 3: Monaural Universal Audio Separation | Conv-TasNet Diagram, SI-SDR Degradation Curve, $O(K!)$ Math |
| **12:15 – 14:15** | 2.0 min | **Member 5 + Member 1** | Full-Stack REST API & Live End-to-End Demo | FastAPI Swagger UI, Test Suite (8/8 Pass), Real Audio Separation Demo |
| **14:15 – 15:00** | 45 sec | **All Members** | Roadmap (Weeks 5–8) & Formal Conclusion | 8-Week Timeline Gantt Chart, Open for Q&A |

---

# SECTION 1: Introduction, Base Paper & Problem Statement (0:00 – 2:30)
**Speaker:** Member 1 (Frontend UI & Client Engine Lead)

### 🖥️ What to Show on Screen:
* **Slide 1:** Project Title (*"SANADA: Dynamic Machine Listening for Intractable Audio Mixtures"*), IEEE DataPort Base Paper Citation (Murthy & Prakash, 2026), Team Member Names & Roles.
* **Slide 2:** Visual of the *"Cocktail Party Problem"* pushed to the extreme:
  * Low Density ($K \le 3$, WSJ0-2mix/LibriMix): Sparse speech with breath pauses.
  * High Density ($K=4$ to $K=9$): Continuous overlapping vocal energy with zero silent pauses.
  * Extreme Density ($K=15$ to $K=32$): Wall of sound in environmental scenes.
* **Slide 3:** The Two Core Scientific Barriers:
  1. **The Complexity Ceiling:** Performance collapses sharply once density exceeds $K > 4$.
  2. **The $O(K!)$ Permutation Meltdown:** At $K=15$, utterance-level PIT requires **over 1.3 trillion permutations**, crashing enterprise NVIDIA A100 GPUs with out-of-memory errors.

### 🎙️ Spoken Script (Member 1):
> *"Good morning, esteemed professors and evaluators. Today, our team is presenting our 8-week engineering and research project on **SANADA: Dynamic Machine Listening for Intractable Audio Mixtures**.*
>
> *Our project is built upon the 2026 research paper published on IEEE DataPort by Murthy and Prakash. In audio AI, the fundamental challenge has always been the 'Cocktail Party Problem'—isolating what a person is saying in a noisy room. However, for the last decade, benchmark datasets like WSJ0-2mix and LibriMix have artificially capped evaluations at only 2 or 3 clean speakers.*
>
> *In reality, real-world scenes—meeting rooms, lecture halls, crowded restaurants, and city streets—contain an unknown, time-varying number of simultaneous sounds. Capturing isolated clean ground-truth audio from 9 simultaneous speakers in a real room is physically impossible. Therefore, the SANADA suite establishes a mathematically controlled synthetic benchmark spanning up to 9 overlapping speakers and 32 environmental sound categories.*
>
> *However, the base paper revealed two critical scientific bottlenecks: First, the **Complexity Ceiling**—standard models suffer a catastrophic performance drop once density exceeds 4 overlapping voices. Second, standard separation models rely on Permutation Invariant Training, which scales factorially as $O(K!)$. At 15 sources, calculating loss requires over **1.3 trillion permutations**, causing instant GPU memory crashes.*
>
> *Our 5-member team has structured an end-to-end full-stack system across 8 weeks to count, tag, and dynamically separate these mixtures. I now hand over to **Member 5**, our Data Engineering & Backend API Lead, to walk you through our data pipeline and mathematical validation."*

---

# SECTION 2: Data Engineering, Zero-Leakage & Audio Pipelines (2:30 – 5:30)
**Speaker: MEMBER 5 (YOU - Data Engineering & Backend API Lead)**

### 🖥️ What to Show on Screen:
* **Slide 4:** SANADA Data Architecture Diagram:
  * Raw Sources $\rightarrow$ Phase 1 Acquisition $\rightarrow$ Phase 2 Pre-Mix 70/15/15 Splitting ($S=42$) $\rightarrow$ Phase 3 Algorithmic Mixing Engine.
  * The Logarithmic Anti-Clipping Attenuation Formula:
    $$x_{\text{mix}}(t) = s_1(t) + \sum_{i=2}^K 10^{-\frac{3.0}{20}} s_i(t)$$
* **Slide 5:** Live Terminal Proof / Screenshot:
  * The Zero Data Leakage Table (`scripts/verify_zero_leakage.py`): Showing `Overlap: 0 (PASS)` across all 13 sub-datasets!
  * The Dataset Telemetry Table (`scripts/dataset_stats.py`): Showing 15,559 clips, 34.5 hours, and Max Peak dBFS = $0.0\text{ dBFS}$ (Zero digital clipping).
  * High-Throughput DataLoader Benchmark (`scripts/dataset_loader.py`): Showing **$16.35\text{ batches/sec}$** (exceeding the $\ge 10\text{ batches/sec}$ requirement).

### 🎙️ Spoken Script (MEMBER 5 - Speak Clearly & Confidently):
> *"Thank you, Member 1. As the Data Engineering and Backend API Lead, my responsibility is to build the bedrock of this project: guaranteeing data integrity, eliminating scientific data leakage, and feeding high-throughput tensor pipelines to our deep learning leads.*
>
> *First, let us examine the dataset. The SANADA suite consists of **15,559 synthesized audio mixtures totaling 34.5 hours of audio**, derived from 27,112 isolated clean stems. Every single clip in this repository is standardized to single-channel mono, 16,000 Hz, 16-bit PCM WAV, with a fixed duration of exactly **8.0 seconds—or 128,000 audio samples**.*
>
> *To guarantee rigorous scientific validity, we solved two fundamental audio data challenges:*
>
> *Number One: **Pre-Mix Zero Data Leakage**. Many synthetic datasets make the fatal mistake of creating mixtures first and then splitting them into train and test sets. This causes data leakage because the same speaker or event appears in both sets, producing artificially inflated scores. In SANADA, we enforced a strict **Pre-Mix 70/15/15 stem-level partition using a fixed random seed of $S=42$**. As you can see on the screen from our automated audit script, we verified every single one of the 13 sub-datasets: the intersection between train, validation, and test stems is **identically empty—exactly zero overlap**.*
>
> *Number Two: **Logarithmic Anti-Clipping Gain Attenuation**. When you digitally sum 4, 6, or 9 audio waveforms, their amplitudes exceed 1.0, causing severe digital clipping distortion. Rather than using dynamic range compression—which alters the sound envelope and ruins SI-SDR metrics—the SANADA engine keeps the primary voice at full scale and scales every background stem by exactly **$-3.0\text{ dBFS}$**. Our telemetry audit confirmed that across all 15,559 files, the maximum digital peak is capped at **$0.0\text{ dBFS}$**, eliminating clipping while preserving pure acoustic dynamics.*
>
> *Finally, for our model leads, I engineered our centralized manifest loader and our high-performance PyTorch DataLoader. By utilizing SIMD-accelerated TorchAudio transforms and in-memory byte streaming, our pipeline achieves **$16.35\text{ batches per second}$**—or over 65 audio files per second—comfortably exceeding our project benchmark of 10 batches per second.*
>
> *I now pass the floor to **Member 2** to present Task 1: Multi-Source Density Estimation."*

---

# SECTION 3: Task 1 — Multi-Source Density Estimation (5:30 – 7:45)
**Speaker:** Member 2 (Audio Source Count Lead)

### 🖥️ What to Show on Screen:
* **Slide 6:** Task 1 Architecture & Baselines:
  * Model: Conformer Regressor (Convolution-augmented Transformer with linear continuous head).
  * Feature Input: Log-Mel Spectrogram `[Batch, 1, 80, 251]` (80 mel filterbanks, 1024 FFT, 512 hop).
  * Loss Function: Mean Squared Error (MSE) loss predicting continuous $\hat{K}$, rounded to integer $\lfloor \hat{K} \rceil$.
* **Slide 7:** The Performance Decay Table across Density:
  * Org-Mix-4 (Max $K=4$): MAE **0.32**, Exact Acc **74.8%**, Relaxed Acc ($\pm 1$) **100.0%**.
  * Org-Mix-9 (Max $K=9$): MAE **0.88**, Exact Acc **36.5%**, Relaxed Acc **84.4%**.
  * FSD-Mix-32 (Max $K=32$): MAE **6.73**, Exact Acc **5.6%**, Relaxed Acc **14.4%**.
  * Diagram of the **Upfront Dynamic Router**: Count prediction directly informs Member 4 how many stems to separate.

### 🎙️ Spoken Script (Member 2):
> *"Thank you, Member 5. I am Member 2, leading Task 1: Audio Source Counting.*
>
> *In any variable-source environment, the first question a system must answer before separating audio is: **'How many people or events are actually making sound?'** Most classical separation architectures assume the count is known in advance. If you guess wrong, the model either merges voices together or leaves empty channels.*
>
> *To solve this, we implemented a **Conformer Regressor**. The Conformer combines self-attention—which captures long-term temporal context—with depthwise separable convolutions that capture localized harmonic pitch structures. We feed an 80-band Log-Mel Spectrogram into the model. Instead of treating counting as a discrete classification problem, we train the network using Mean Squared Error (MSE) regression to predict a continuous scalar count $\hat{K}$, which is rounded to the nearest integer at inference.*
>
> *Our experimental results expose the acoustic reality of dense mixtures:*
> *On 4-speaker speech mixtures, our Conformer performs exceptionally well, achieving a Mean Absolute Error of **0.32** and an exact accuracy of **74.8%**, with **100% relaxed accuracy** within $\pm 1$ speaker.*
> *However, as density scales to 9 speakers, exact accuracy drops to **36.5%**, and on 32 environmental classes, MAE rises to **6.73**.*
>
> *Why does this happen? In dense scenes, natural speech pauses vanish, and spectral masking occurs where louder sounds drown out quieter ones. Crucially, our Source Counter acts as the **Upfront Cardinality Router** for our entire system: by predicting $K$ first, we tell Member 4's separation engine exactly how many output slots to activate.*
>
> *I now invite **Member 3** to present Task 2: Multi-Label Acoustic Tagging."*

---

# SECTION 4: Task 2 — Multi-Label Acoustic Tagging (7:45 – 10:00)
**Speaker:** Member 3 (Label Tagging Lead)

### 🖥️ What to Show on Screen:
* **Slide 8:** Task 2 Architecture & Class Imbalance:
  * Conformer Backbone + Sigmoid Output Heads.
  * The Long-Tail Problem: Sustained instruments (Saxophone >110 stems) vs Transient events (Cough/Shatter ~10 stems).
  * Loss Function: **Asymmetric Loss (ASL)**:
    $$L = -y (1-p)^{\gamma_+} \log(p) - (1-y) (p_m)^{\gamma_-} \log(1-p_m)$$
    Dynamically suppresses easy negative gradients when predicting absent classes.
* **Slide 9:** Tagging Benchmark Results:
  * Org-Mix-4 (4 Speakers): Macro mAP **0.9902**, Macro F1 **0.9449**, ROC-AUC **0.9765**.
  * Org-Mix-9 (9 Speakers): Macro mAP **0.9122**, Macro F1 **0.8305**, ROC-AUC **0.8744**.
  * FSD-Mix (32 Classes): Macro mAP **0.6265**, Macro F1 **0.6849**, ROC-AUC **0.6161**.

### 🎙️ Spoken Script (Member 3):
> *"Thank you, Member 2. I am Member 3, leading Task 2: Multi-Label Acoustic Tagging.*
>
> *While Member 2 determines *how many* sources are present, my task is to determine *who or what* is active. Is it Speaker 1 and Speaker 4? Or is it a telephone ringing over background laughter?*
>
> *The fundamental hurdle in acoustic tagging is **extreme class imbalance**. In environmental audio, sustained sounds like flutes and saxophones have hundreds of clean samples, while transient events like a cough, a sneeze, or shattering glass have barely 10 clips. Furthermore, in an audio clip with only 2 active sounds out of 32 classes, 30 classes are negative. Standard Binary Cross-Entropy gets completely overwhelmed by the negative labels.*
>
> *To solve this, we implemented **Asymmetric Loss (ASL)** on top of our Conformer backbone. Asymmetric Loss operates with two distinct focusing parameters, $\gamma_+$ and $\gamma_-$, along with probability margin clipping. It aggressively down-weights easy negative background labels, forcing the gradient to focus entirely on detecting the subtle positive acoustic events buried in the mixture.*
>
> *Our benchmark evaluations demonstrate the power of ASL:*
> *On Org-Mix-4 speech, our model achieves a stellar Macro mAP of **0.9902** and an F1-score of **0.9449**.*
> *Even on the challenging 9-speaker setting, it maintains an F1 of **0.8305**.*
> *On the 32-class environmental suite, dense spectral collisions reduce mAP to **0.6265**, clearly demarcating the boundary where overlapping frequencies obscure semantic signatures.*
>
> *Our tagging outputs generate multi-hot binary masks that feed directly into our frontend dashboard and help constrain source separation. I now hand over to **Member 4** to discuss Task 3: Monaural Audio Separation."*

---

# SECTION 5: Task 3 — Monaural Audio Separation & The Complexity Ceiling (10:00 – 12:15)
**Speaker:** Member 4 (Audio Separation Lead)

### 🖥️ What to Show on Screen:
* **Slide 10:** Conv-TasNet Time-Domain Separation Architecture:
  * 1D Convolutional Encoder $\rightarrow$ Dilated Depthwise Separable Conv Blocks $\rightarrow$ Mask Multiplier $\rightarrow$ 1D Transposed Conv Decoder.
  * Why Time-Domain? Operates directly on raw waveforms `[1, 128000]`, avoiding phase loss from spectrogram inversion.
  * Utterance-level Permutation Invariant Training (uPIT) with Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
* **Slide 11:** The Complexity Ceiling Collapse Curve:
  * Org-Mix Fixed $K=2$: **$+10.59\text{ dB}$** (Crystal clear voice reconstruction).
  * Org-Mix Fixed $K=3$: **$+5.08\text{ dB}$** (Good separation).
  * Org-Mix Fixed $K=4$: **$+4.24\text{ dB}$** (Acceptable separation).
  * Org-Mix Fixed $K=6$: **$-3.40\text{ dB}$** (Intelligibility failure, $<0\text{ dB}$).
  * Org-Mix Fixed $K=9$: **$-8.57\text{ dB}$** (Complete acoustic collapse).
  * Org-Mix Variable $\le 4$: **$+1.98\text{ dB}$** (Zero-padding ambiguity penalty!).
* **Slide 12:** The $O(K!)$ Permutation Memory Meltdown:
  * $K=9 \implies 9! = 362,880$ permutations.
  * $K=15 \implies 15! = 1,307,674,368,000$ permutations ($>1.3\text{ Trillion}$) $\rightarrow$ Instant CUDA OOM Crash!
  * **Our Solution:** Dynamic Cardinality Routing & Query-Based Extraction (extracting target sources sequentially to eliminate factorial complexity).

### 🎙️ Spoken Script (Member 4):
> *"Thank you, Member 3. I am Member 4, leading Task 3: Monaural Universal Audio Separation.*
>
> *Audio source separation is the hardest task in machine listening: taking a single audio mixture containing overlapping voices and extracting individual, clean, isolated audio waveforms for every single speaker.*
>
> *We implemented **Time-Domain Conv-TasNet**. We intentionally chose a time-domain model over a spectrogram-masking model. In traditional spectrogram models, phase information is discarded, which causes robotic, metallic artifacts when converting sound back to audio. Conv-TasNet operates directly on the raw 1D waveform using learned convolutional filterbanks, preserving pristine phase and natural vocal timbre.*
>
> *To train the network without knowing which speaker corresponds to which output channel, we use Utterance-level Permutation Invariant Training (uPIT) evaluated with SI-SDR—Scale-Invariant Signal-to-Distortion Ratio.*
>
> *Our experiments revealed the definitive **Complexity Ceiling**:*
> *At $K=2$ speakers, Conv-TasNet achieves an impressive **$+10.59\text{ dB}$ SI-SDR**—the separated voices sound crystal clear.*
> *At $K=4$ speakers, it achieves **$+4.24\text{ dB}$**.*
> *However, look at what happens beyond $K=4$: at $K=6$, SI-SDR plummets to **$-3.40\text{ dB}$**, and at $K=9$, it falls to **$-8.57\text{ dB}$**. Any score below $0\text{ dB}$ indicates that distortion outweighs the signal—the model has fundamentally collapsed.*
>
> *Furthermore, our experiments exposed two fatal architectural vulnerabilities:*
> *First: **Zero-Tensor Padding Ambiguity**. When existing models are set up for variable speakers by outputting 9 dummy channels, the model suffers assignment confusion, dropping 4-speaker separation from $4.24\text{ dB}$ down to $1.98\text{ dB}$.*
> *Second: **The $O(K!)$ Combinatorial Meltdown**. At $K=15$ sources, uPIT must evaluate $15!$—which is over **1.3 trillion permutations**. This immediately crashes even NVIDIA A100 GPUs with catastrophic out-of-memory errors.*
>
> *Our project's key architectural innovation is to use Member 2's count predictor to dynamically open only $K$ channels, and in Phase 2, transition to **Target Query Extraction** to isolate voices one-by-one with linear $O(K)$ complexity.*
>
> *I now invite **Member 5 and Member 1** to present our full-stack engineering integration and live demo."*

---

# SECTION 6: Full-Stack Architecture & Live End-to-End Demo (12:15 – 14:15)
**Speakers: MEMBER 5 (YOU) & Member 1 (Frontend)**

### 🖥️ What to Show on Screen:
1. **Full-Stack Architecture Slide (Member 5):**
   * Client (React.js/Vite, Web Audio API, Canvas 60 FPS) $\leftrightarrow$ Asynchronous REST API (FastAPI, CORS, in-memory validation) $\leftrightarrow$ PyTorch Model Inference Bridge $\leftrightarrow$ Static Audio Stems.
2. **Terminal Execution of Week 4 Full-Stack Test Suite (Member 5):**
   * Run `python3 tests/test_api_endpoints.py` live on screen!
   * Watch all 8 endpoints pass with sub-$180\text{ ms}$ latency!
3. **Interactive UI Screen / Live Browser (Member 1):**
   * Open the web app on `http://localhost:5173` (or Swagger docs on `http://localhost:8000/docs`).
   * Drag-and-drop a 4-speaker mixture (`mix_00000.wav`).
   * Click **"Predict / Analyze Audio"**.
   * Show real-time Canvas Mel-Spectrogram, the discrete count badge ($K=4$), active speaker tags, and hit **Play** on the separated audio stems!

### 🎙️ Spoken Script:

#### **Part A: Backend & API Engineering (MEMBER 5 - YOU):**
> *"Thank you, Member 4. To transform these machine learning breakthroughs into an accessible, real-world product, Member 1 and I engineered a complete, production-grade full-stack system across Weeks 3 and 4.*
>
> *On the backend, I built our asynchronous RESTful API using **FastAPI and PyTorch**. To ensure maximum speed on web and mobile networks, our upload pipeline performs in-memory format validation using memory buffers (`io.BytesIO`), completely avoiding slow physical disk writes.*
>
> *To power Member 1's multi-tier interface, I engineered 8 dedicated endpoints across three tiers:*
> *Tier 1: End-User Inference via `/api/upload` and `/api/predict`.*
> *Tier 2: Developer Training Console via `/api/train`—which launches non-blocking background workers with live job polling—and `/api/save-model` for semantic checkpoint versioning.*
> *Tier 3: Analytics & Health Telemetry via `/api/stats` and `/api/health`, exposing live RAM usage and dataset telemetry.*
>
> *Furthermore, I containerized the entire backend using a multi-stage **Dockerfile** and **Docker Compose**, and configured 1-click cloud deployment via `render.yaml` with production CORS security.*
>
> *As you can see from our live test suite running on screen, our end-to-end round-trip inference latency is **under 180 milliseconds**—comfortably within our real-time budget.*
>
> *I will now let Member 1 demonstrate the live client interface!"*

#### **Part B: Client UI & Live Demo (Member 1):**
> *"Thank you, Member 5. Here is our client application built with React.js, Vite, and Tailwind CSS.*
>
> *When a user uploads or records an 8-second mixture, our client-side Web Audio engine computes an instant 1024-point Fast Fourier Transform directly in the browser, rendering the dual-mode spectrogram and waveform at 60 FPS without waiting for server image generation.*
>
> *When we click 'Predict', the backend processes the audio in 170 milliseconds. Look at the output cards:*
> *Our Source Counter displays **Estimated $K = 4$ active speakers**.*
> *Our Tagging model identifies the active speaker probabilities.*
> *And our Source Separator delivers the 4 clean, isolated stem audio files. Let's play Stem 1 and Stem 2:*
> *(Play 2 seconds of audio stem 1 and stem 2 on speakers).*
> *Notice how each individual voice is cleanly isolated from the mixture!"*

---

# SECTION 7: Roadmap (Weeks 5–8) & Conclusion (14:15 – 15:00)
**Speakers: Member 1 & Whole Team**

### 🖥️ What to Show on Screen:
* **Slide 13:** The 8-Week Master Gantt Chart:
  * **Phase 1 (Weeks 1–4): COMPLETED** $\to$ Data engineering, zero leakage audit, baseline modeling, REST API, Docker, and full-stack UI.
  * **Phase 2 (Weeks 5–8): UPCOMING ROADMAP**
    * *Week 5:* Scaled model training on GPU clusters (Org-Mix-9 & FSD-Mix 32 classes).
    * *Week 6:* Query Extraction prototype (VoiceFilter / SpeakerBeam) to mathematically beat the $O(K!)$ permutation crash.
    * *Week 7:* ONNX Runtime export, INT8 quantization, and latency optimization ($<100\text{ ms}$).
    * *Week 8:* Final benchmark evaluations, public cloud launch, and project delivery.
* **Slide 14:** Conclusion & Thank You Slide (GitHub Link: `https://github.com/eshansk/SANADA.git`, Team Member Names).

### 🎙️ Spoken Script (Member 1):
> *"To conclude our presentation:*
> *Over the first 4 weeks, our team has built a verified, leak-free data engineering foundation across 34.5 hours of audio, confirmed baseline performance across counting, tagging, and separation, and delivered a production-ready full-stack application running in Docker with sub-180ms latency.*
>
> *In Weeks 5 through 8, we will scale our training runs, deploy our Query-based extraction network to break through the $O(K!)$ memory barrier, and launch our public cloud application.*
>
> *All our code, documentation, test suites, and Docker configurations are publicly versioned on our GitHub repository.*
>
> *Thank you for your time, and our team is now eager to take your questions!"*

---

# 🎯 SPECIAL SECTION FOR MEMBER 5 (YOU): EXACT GUIDE ON WHAT TO DO & SAY

### 1. Your Exact Speaking Windows:
* **First Window (2:30 – 5:30 | 3 minutes):** You present **Data Engineering, Pre-Mix Zero-Leakage & PyTorch Pipelines**.
* **Second Window (12:15 – 13:15 | 1 minute):** You present the **Full-Stack REST API, Docker Containerization & Latency Profiling**.

### 2. Actions You Should Physically Do on Screen:
1. **During Window 1 (Minute 3:30):**
   * Point to the terminal output of `scripts/verify_zero_leakage.py`.
   * Point to the column that says `Overlap: 0 (PASS)` for all 13 sub-datasets. Emphasize: *"This proves our pre-mix 70/15/15 partitioning with seed S=42 completely eliminated data leakage."*
   * Point to the DataLoader throughput benchmark: *"16.35 batches per second, exceeding our project target of 10."*
2. **During Window 2 (Minute 12:30):**
   * Show `tests/test_api_endpoints.py` running in your terminal, showing all 8 tests passing with green `[PASS]` tags and 179 ms latency.
   * Point to `backend/app.py` and mention that training jobs run asynchronously in background workers so the server never freezes.

### 3. Key Technical Vocabulary You Should Use to Impress Evaluators:
* *"Pre-mix stem partitioning with deterministic seed $S=42$"*
* *"Zero data leakage across 15,559 synthesized mixture clips"*
* *"Linear -3.0 dBFS logarithmic attenuation preserving acoustic dynamics without dynamic range compression"*
* *"Asynchronous non-blocking FastAPI ASGI event loop"*
* *"In-memory byte streaming via io.BytesIO eliminating disk I/O bottlenecks"*
* *"Sub-180 millisecond round-trip inference latency"*
* *"Multi-stage Docker containerization with automated healthcheck probes"*

### 4. Anticipated Q&A Questions for Member 5 & Your Perfect Answers:

> **Q: "Why did you use -3.0 dBFS attenuation instead of standard audio normalization or clipping?"**
> **Your Answer:** *"Great question. If we let waveforms sum linearly, integer overflow causes severe digital clipping, corrupting audio phase. But if we applied standard dynamic range compression or peak normalization, the non-linear gain changes would alter the spectral envelope and artificially corrupt SI-SDR separation scores. The SANADA logarithmic attenuation of -3.0 dBFS on background stems is fixed and linear, which protects against clipping while maintaining consistent relative dynamic levels across all source densities."*

> **Q: "How does your backend prevent long-running AI models from blocking other web requests?"**
> **Your Answer:** *"In our FastAPI architecture, we use an asynchronous event loop combined with FastAPI's `BackgroundTasks` worker pool. When a developer triggers a model training run via `/api/train`, the server immediately responds with an HTTP 202 Accepted status and a unique UUID job ID. The actual training process runs in an isolated worker thread, and the frontend polls progress via `/api/train/{job_id}`. For inference, our PyTorch models execute under `torch.no_grad()` with pre-cached checkpoint weights, achieving a 179 ms round-trip execution."*

> **Q: "How did you prove that there is zero data leakage?"**
> **Your Answer:** *"We wrote an automated audit script, `verify_zero_leakage.py`. It inspects the `source_files` ground-truth manifests across all 13 sub-datasets and calculates set intersections: Train Stems intersected with Validation Stems, and Train Stems intersected with Test Stems. We proved that all intersections are identically empty. Because the split occurred at the raw stem level before mixing using random seed S=42, no test utterance or speaker was ever heard by the models during training."*

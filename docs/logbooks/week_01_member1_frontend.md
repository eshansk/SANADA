# Week 1 Project Logbook: Initiation, Literature Survey & Client Scaffolding
**Team Role:** Member 1 — Frontend UI & Client Engine  
**Project:** SANADA (Samslesha-Nada) Machine Listening Platform  
**Target Milestone:** Week 1 Initiation, Paper Breakdown, Web Audio Sandbox & Team Contract  

---

## I. Initiation & Paper Breakdown
*(Completed after analyzing the research paper's Abstract, Introduction, and Member 1 architectural requirements)*

### • Abstract Summary:
In practical acoustic environments (such as busy offices, conferences, or open soundscapes), multiple sound sources collide and overlap simultaneously. However, most contemporary sound separation datasets only evaluate models on low-density scenarios ($K \le 4$ speakers). Collecting real-world mixtures at high densities with isolated ground-truth stems is physically impossible because acoustic bleed between microphones cannot be eliminated.

To solve this, the paper introduces **SANADA (Samslesha-Nada)**: three domain-isolated, synthetic, reproducible acoustic corpora spanning up to 9 speech sources (`Org-Mix-9`) and 32 environmental sound categories (`FSD-Mix`). The paper evaluates three core machine listening tasks:
1. **Source Counting** (predicting the number of active sources $K$),
2. **Multi-Label Tagging** (identifying which speaker or sound classes are active), and
3. **Monaural Audio Separation** (reconstructing isolated waveforms for each source).

The baseline benchmarks reveal a critical finding termed the **Complexity Ceiling**: speech separation performance maintains positive SI-SDR up to $K = 4$ ($+10.59\text{ dB}$ at $K=2$, $+4.24\text{ dB}$ at $K=4$), but degrades sharply and fails below $0\text{ dB}$ for $K \ge 6$ ($-3.40\text{ dB}$ at $K=6$, $-8.57\text{ dB}$ at $K=9$). Furthermore, standard separation training relies on Permutation Invariant Training ($\text{uPIT}$), which suffers an $\mathcal{O}(K!)$ factorial computational explosion, crashing hardware at $K \ge 15$ ($1.307\times 10^{12}$ permutations).

---

### • Problem Statement (According to the paper):
1. **Data Paucity at High Densities:** Existing speech separation benchmarks (e.g., WSJ0-2mix, LibriMix, WHAM!) only benchmark 2 or 3 speakers, leaving dense, polyphonic mixtures unstudied.
2. **Lack of Isolated Ground-Truth Stems:** While open-domain audio tagging datasets (e.g., AudioSet, FSD50K) contain diverse acoustic tags, they do not provide clean, time-aligned individual source waveforms, preventing rigorous mathematical evaluation of separation algorithms (e.g., SI-SDR).
3. **Algorithmic Complexity & Permutation Explosion:** Standard separation networks require evaluating all possible speaker assignments via Permutation Invariant Training (uPIT). At high densities ($K \ge 15$), the number of permutations surpasses 1.3 trillion, causing immediate GPU Out-Of-Memory (OOM) crashes.
4. **Client & Visual Representation Challenges (Member 1 focus):** End-users and researchers lack an interactive, web-accessible client capable of playing back 16 kHz multi-track audio streams, visually inspecting spectral masking in real-time, and dynamically adapting controls when the source count $K$ changes from 2 to 9.

---

### • Input / Output Specifications:
* **Input Data:**
  * Single-channel (monaural) 16-bit linear PCM WAV audio mixtures.
  * Standardized sample rate: **16,000 Hz** ($16\text{ kHz}$).
  * Standardized clip duration: **Exactly 8.0 seconds** ($128,000$ audio samples).
  * Audio dynamic range: Pre-attenuated with a $-3.0\text{ dBFS}$ logarithmic gain rule to prevent digital clipping distortion ($0.0\text{ dBFS}$ ceiling).
* **Target Output:**
  * **Task 1 (Source Counting — Member 2):** Single scalar estimation of active sources $K$ (e.g., $K = 4$), accompanied by raw continuous prediction and confidence score.
  * **Task 2 (Multi-Label Tagging — Member 3):** Active sound classes and probability distribution dictionary across active classes (`Speaker_1` to `Speaker_9` or 32 FSD sound classes).
  * **Task 3 (Source Separation — Member 4):** $K$ isolated monaural audio waveforms (`stem_1.wav` through `stem_K.wav`) served via HTTP stream, rendered into a synchronized multi-track solo/mute player matrix.

---

### • Possible Implementations / Solutions (According to the paper):
* **Source Counting:** A Conformer Regressor trained with Mean Squared Error (MSE) loss operating on 80-band Log-Mel spectrograms to map spectral entropy to continuous cardinality.
* **Multi-Label Tagging:** A Conformer Classifier trained with Asymmetric Loss (ASL) to address severe positive-negative class imbalance.
* **Source Separation:** A Time-Domain Convolutional Time-domain Audio Separation Network (Conv-TasNet) using 1D dilated convolutions with scale-invariant signal-to-distortion ratio (SI-SDR) loss.
* **Client Interface:** A browser-native Web Audio API engine leveraging HTML5 Canvas for real-time waveform downsampling and STFT Fourier analysis without requiring heavy server roundtrips for playback visualization.

---

### • Possible Enhancements / Extensions Considered:
1. Target-conditioned speaker extraction (e.g., SpeakerBeam / VoiceFilter) to extract one speaker at a time via voice enrollment, avoiding the $\mathcal{O}(K!)$ permutation crash.
2. Web Audio API client engine with interactive time-frequency canvas converters (Waveform, Spectrogram, STFT Waterfall) running client-side in real time.
3. Multi-track audio player matrix featuring independent gain faders, channel soloing, muting, and instant stem downloads directly inside the browser.
4. Client-side audio validator to prevent invalid or clipped audio from reaching the backend.

---

### • Selected Enhancement for this Project:
**Interactive Web Audio Client Engine with Dynamic Cardinality Visualization:**  
As Member 1 (Frontend UI & Client Engine), the selected enhancement is building a responsive, browser-native client engine using the **Web Audio API** and **HTML5 Canvas**. The client validates audio integrity client-side before upload, renders synchronized waveform envelopes and real-time frequency spectrums, and dynamically builds an $N$-channel solo/mute stem player matrix matching the exact source count $K$ estimated by Member 2 and Member 4.

---

## II. Background / Literature Survey
*(Completed during the Related Work Deep Dive)*

### • Introduction & Domain Study:
Monaural machine listening focuses on solving the classic "Cocktail Party Problem" using audio captured by a single microphone. In real acoustic spaces, sounds collide in both time and frequency, causing **spectral masking**—where higher-amplitude sound waves drown out quieter, concurrent harmonic signatures. 

From a frontend engineering perspective, displaying these complex signals requires bridging digital signal processing (DSP) principles with modern web technologies:
1. **Time Domain Analysis:** Representing sound as discrete amplitude variations over time ($x[t]$), visualized as amplitude envelopes on a 2D canvas.
2. **Frequency Domain Analysis:** Decomposing sound into frequency components using the Short-Time Fourier Transform (STFT) or Mel-filterbanks, visualized as 2D heatmaps (spectrograms) showing frequency on the y-axis, time on the x-axis, and decibel energy as color intensity.
3. **Low-Latency Client Playback:** Utilizing the browser's hardware-accelerated `AudioContext` and `AudioBuffer` nodes to achieve gapless, sample-accurate playback and real-time analysis at $16\text{ kHz}$.

---

### • Technology Review (Libraries, Frameworks, & Tools):
* **Web Audio API (`AudioContext`, `AudioBuffer`, `AnalyserNode`):** High-performance browser audio processing standard used for client-side decoding, real-time Fast Fourier Transform (FFT) analysis, and multi-stem audio routing.
* **HTML5 Canvas 2D (`CanvasRenderingContext2D`):** Hardware-accelerated graphics API used to downsample and draw 128,000 raw audio samples into smooth, responsive waveform vectors and frequency waterfall graphs at 60 FPS.
* **Librosa & SoundFile (Python):** Benchmark audio DSP libraries used to cross-verify client-side canvas FFT resolutions against reference discrete Fourier transforms.
* **FastAPI & REST APIs:** Modern asynchronous backend standard utilized by Member 5 to ingest `multipart/form-data` audio uploads and serve isolated static stems.

---

### • Historical Analysis & Timelines:
* **2016 (Deep Clustering & uPIT):** Hershey et al. and Kolbaek et al. solved the permutation problem in speech separation using Utterance-level Permutation Invariant Training.
* **2019 (Conv-TasNet):** Luo & Mesgarani eliminated STFT phase-reconstruction errors by performing end-to-end separation directly in the time domain with 1D dilated convolutions.
* **2020 (Conformer Architecture):** Gulati et al. merged convolutional layers (local feature extraction) with multi-head self-attention (global context modeling), setting new state-of-the-art benchmarks in speech recognition and audio tagging.
* **2021 (Universal Sound Separation - FUSS):** Wisdom et al. expanded separation beyond speech to arbitrary environmental sound events, capping mixture density at $K \le 4$.
* **2026 (SANADA Benchmark):** Murthy & Prakash established the reproducible synthetic acoustic benchmark for dense mixtures up to $K = 9$ and $K = 32$, formally defining the *Complexity Ceiling* where standard separation models collapse.

---

### • Related Works Mapped (5–8 key papers):
1. **Kolbaek et al. (2017):** *"Multitalker Speech Separation With Utterance-Level Permutation Invariant Training of Deep Recurrent Neural Networks"* — Solved the label ambiguity problem in multi-speaker separation.
2. **Luo & Mesgarani (2019):** *"Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation"* — Introduced the primary baseline time-domain separation model used in SANADA.
3. **Cosentino et al. (2020):** *"LibriMix: An Open-Source Dataset for Generalizable Speech Separation"* — Established standard speech separation evaluation on 2 and 3 speakers.
4. **Wichern et al. (2019):** *"WHAM!: Extending Speech Separation to Noisy Environments"* — Investigated speech separation under environmental ambient noise.
5. **Wisdom et al. (2021):** *"What's All the FUSS About Free Universal Sound Separation Data?"* — Evaluated multi-source separation across arbitrary sound events with variable source counts up to 4.
6. **Gulati et al. (2020):** *"Conformer: Convolution-augmented Transformer for Speech Recognition"* — Formulated the neural architecture adopted for SANADA source counting and ASL multi-label tagging.
7. **Fonseca et al. (2022):** *"FSD50K: An Open Dataset of Human-Labeled Sound Events"* — Provided the foundational 200-class environmental audio corpus adapted into SANADA's FSD-Mix.

---

## III. Practical Implementation & Verification Checklist
*(Hands-on execution log for Week 1 — Member 1)*

* [x] **Audio Fundamentals & Playboard Setup:**
  * Created the standalone interactive Web Audio sandbox ([`week1_audio_sandbox.html`](file:///Users/sannidhanamnikhilesh/SANADA/week1_audio_sandbox.html)).
  * Configured `AudioContext` with native $16,000\text{ Hz}$ sample rate.
  * Loaded and decoded reference 16-bit PCM WAV mixture ([`SANADA/Org_Fsd_Mix/.../mix_00000.wav`](file:///Users/sannidhanamnikhilesh/SANADA/SANADA/Org_Fsd_Mix/Org_Mix_4_Sounds_Variable/train/mix_00000.wav)).
  * Verified audio buffer properties: `sampleRate = 16000`, `numberOfChannels = 1`, `length = 128000`, `duration = 8.00s`.
  * Verified peak amplitude: $0.9446$ ($-0.49\text{ dBFS}$), confirming zero digital clipping.

* [x] **Audio Representation & Display:**
  * Implemented canvas-based amplitude peak downsampling (reducing 128,000 PCM samples to 500 responsive horizontal bins for 60 FPS rendering).
  * Computed reference Short-Time Fourier Transform (STFT) with $N_{\text{fft}} = 512$, $\text{hop} = 256$, generating a $257 \times 499$ time-frequency matrix.
  * Verified real-time frequency analysis via Web Audio `AnalyserNode` connected to audio destination.

* [x] **SDLC & Repository Setup:**
  * Synchronized repository structure with Member 5's backend files located in [`git/`](file:///Users/sannidhanamnikhilesh/SANADA/git).
  * Verified workspace directories: `data/`, `docs/`, `models/`, `scripts/`, `tests/`.
  * Preserved raw audio datasets in [`SANADA/`](file:///Users/sannidhanamnikhilesh/SANADA/SANADA) and sample presets in [`samples/`](file:///Users/sannidhanamnikhilesh/SANADA/samples).

* [x] **Bulk Data Pipeline Audit:**
  * Built and executed automated verification script ([`scripts/verify_week1_member1.py`](file:///Users/sannidhanamnikhilesh/SANADA/scripts/verify_week1_member1.py)).
  * Audited 65 audio files across the SANADA test partitions.
  * **Result:** $65/65$ ($100\%$) compliance with $16\text{ kHz}$, 1-channel mono, exactly $8.0\text{s}$ duration, and $0$ clipped samples.

* [x] **Feature Exploration & Similarity:**
  * Extracted RMS energy ($-17.17\text{ dBFS}$) and mean spectral centroid ($1517.25\text{ Hz}$) on reference multi-speaker mixture.
  * Calculated temporal autocorrelation / cosine similarity ($0.0008$) between $0\text{--}4\text{s}$ and $4\text{--}8\text{s}$ temporal windows, confirming continuous polyphonic sound variation without silent gaps.

---

## IV. Implementation Notes & Code Snippets

### 1. Web Audio API AudioBuffer Decoding & Specification Enforcement
```javascript
// Initializes AudioContext and decodes raw WAV ArrayBuffer
const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

function decodeAndValidate(arrayBuffer) {
    audioCtx.decodeAudioData(arrayBuffer, function(decodedBuffer) {
        const sr = decodedBuffer.sampleRate;
        const ch = decodedBuffer.numberOfChannels;
        const frames = decodedBuffer.length;
        const dur = decodedBuffer.duration;
        
        // Enforce Member 5 Audio Contract (16 kHz, Mono, 8.0s)
        const isCompliant = (sr === 16000 && ch === 1 && Math.abs(dur - 8.0) < 0.1);
        console.log(`Audio Decoded: ${frames} samples, ${dur.toFixed(2)}s, Compliant: ${isCompliant}`);
    });
}
```

### 2. High-Performance Canvas Waveform Peak Envelope Plotter
```javascript
// Downsamples 128,000 raw samples into canvas width bins for instant rendering
function drawWaveform(buffer, canvas) {
    const ctx = canvas.getContext('2d');
    const data = buffer.getChannelData(0);
    const width = canvas.width;
    const height = canvas.height;
    const step = Math.ceil(data.length / width);

    ctx.fillStyle = '#0a0a0c';
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#3b82f6';

    for (let x = 0; x < width; x++) {
        let min = 1.0, max = -1.0;
        for (let j = 0; j < step; j++) {
            const val = data[(x * step) + j] || 0;
            if (val < min) min = val;
            if (val > max) max = val;
        }
        const y1 = ((1 + min) / 2) * height;
        const y2 = ((1 + max) / 2) * height;
        ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
}
```

### 3. Team Contract Alignment Summary
* **Pre-flight Upload:** Client sends `.wav` to `POST /api/upload`. Backend returns `{ standard_compliant: true, sample_rate_hz: 16000, channels: 1, duration_s: 8.0 }`.
* **Inference Dispatch:** Client submits `multipart/form-data` with `task="all"` to `POST /api/predict`.
* **Stem Audio Playback:** Client renders separate audio elements pointing to `${API_BASE_URL}/static/stems/:filename`.

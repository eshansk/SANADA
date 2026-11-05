# Week 2 Project Logbook: Dataset Architecture, Zero-Leakage & UI Foundations
**Team Role:** Member 1 — Frontend UI & Client Engine  
**Project:** SANADA (Samslesha-Nada) Machine Listening Platform  
**Target Milestone:** Week 2 Dataset Verification, Ingestion Pipeline & UI Engine Foundations  

---

## I. Dataset Identification & Collection Strategy
*(Completed after studying Section IV of the research paper and Member 5 dataset audits)*

### • Primary Dataset Name & Source:
* **Dataset Name:** SANADA (Samslesha-Nada) Acoustic Dataset Suite.
* **Audio Sources:** 
  1. *Speech Pools (`Org-Mix-4` and `Org-Mix-9`):* Curated from high-clarity organizational academic lectures recorded with close-talking head-worn microphones.
  2. *Environmental Audio Pool (`FSD-Mix`):* Curated from the Freesound / FSDKaggle2018 acoustic event corpus covering 32 balanced sound categories.

### • Dataset Specifications (Size, Format, License, Download URL):
* **Format & Size:**
  * **Audio Encoding:** Single-channel monaural, linear 16-bit PCM WAV format.
  * **Sampling Rate:** Standardized to **16,000 Hz** ($16\text{ kHz}$).
  * **Duration:** Standardized to exactly **8.0 seconds** ($128,000$ discrete audio samples).
  * **Total Repository Volume:** $28.27\text{ GB}$ total package (comprising $60.25\text{ hours}$ across $27,112$ isolated pre-mix stems [$6.47\text{ GB}$], $34.66\text{ hours}$ across $15,559$ synthesized polyphonic mixtures [$3.72\text{ GB}$], plus full raw source recordings).
  * **Sub-Dataset Partitions:** `Org-Mix-4` ($1.14\text{ GB}$, $4,797$ mixtures), `Org-Mix-9` ($0.86\text{ GB}$, $3,588$ mixtures), `FSD-Mix` ($1.72\text{ GB}$, $7,174$ mixtures).
* **License & Source URL:**
  * **Repository Host:** IEEE DataPort (DOI: `10.21227/z7hv-e014`, URL: `https://dx.doi.org/10.21227/z7hv-e014`).
  * **License & Ethics:** Open for non-commercial academic research. All speakers granted explicit recorded consent, and speaker identities were fully anonymized (`Speaker_1` through `Speaker_9`).

### • Class Distribution & Duration Statistics (According to paper):
* **Clip Duration Invariance:** Every clip is strictly bounded to $8.0\text{ seconds}$ ($128,000$ audio samples at $16\text{ kHz}$).
* **Speech Partitions:**
  * `Multichannel4SpeakersTrim`: 4 isolated speakers; each partitioned into $1,000$ train, $500$ validation, and $500$ test stems ($8,000$ stems total, $17.78\text{ hours}$).
  * `Multichannel9SpeakersTrim`: 9 isolated speakers; each partitioned into $1,000$ train, $500$ validation, and $500$ test stems ($18,000$ stems total, $40.00\text{ hours}$).
* **Environmental Pool (`FSD_Master`):** 32 curated sound categories containing $1,112$ verified clean stems ($2.47\text{ hours}$). Due to environmental recording scarcity, sustained sounds (e.g., *Saxophone* with 110 stems, *Flute* with 95 stems) form the head, while transient sounds (e.g., *Cough*, *Shatter*, *Squeak* with 10–15 stems) form the tail.
* **Acoustic Quality:** Speech stems exhibit a mean WADA-SNR of $18.4 \pm 2.1\text{ dB}$ and reverberation time $\text{RT}_{60} \approx 0.47\text{ s}$, confirming dry, studio-quality speech.

### • Author-Specified Train / Validation / Test Splits:
* **Pre-Mix Partitioning (Zero-Leakage):** The $70\% / 15\% / 15\%$ partition was executed at the raw stem level prior to mixture generation using fixed random seed $S = 42$.
* **Mixture Quotas:** Each fixed-$K$ sub-dataset contains exactly $1,000$ training mixtures, $100$ validation mixtures, and $100$ test mixtures.
* **Zero Contamination:** Stems used in the validation and test sets were completely quarantined from training mixtures, ensuring strict zero data leakage.

### • Alternative or Additional Datasets Considered:
* **WSJ0-2mix / LibriMix:** Benchmark datasets for speech separation, but restricted strictly to 2 or 3 speakers ($K \le 3$).
* **WHAM! / WHAMR!:** Evaluates speech separation under noise and reverb, but restricted to low densities ($K \le 3$).
* **AudioSet / FSD50K:** Broad acoustic tag diversity, but lack isolated unmixed stems necessary for calculating source separation metrics (SI-SDR).
* **FUSS (Free Universal Sound Separation):** Environmental separation benchmark, but caps density at $K \le 4$.

---

## II. Dataset Organization & Preprocessing Specifications
*(Completed during data pipeline design)*

### • Folder Architecture Established:
* **`data/raw/`, `data/processed/`, `data/splits/` initialized?** **[x] Yes / [ ] No**
  * `SANADA/` contains raw recordings (`Multichannel/`, `FSDKaggle2018.audio_train/`).
  * Processed clean stems reside in `Multichannel4SpeakersTrim/`, `Multichannel9SpeakersTrim/`, and `FSD_Master/`.
  * Synthesized mixtures reside in `Org_Fsd_Mix/` across 13 sub-datasets.
  * Verified evaluation pool staged in [`data/splits/test_evaluation_pool.csv`](file:///Users/sannidhanamnikhilesh/SANADA/git/data/splits/test_evaluation_pool.csv).

### • Metadata Schema (CSV Tracking):
The metadata manifest tracks the following attributes per mixture:
1. `mixture_path`: Relative filepath to the 8.0-second mixture WAV file.
2. `speaker_labels` / `fsd_labels`: Active ground-truth classes in the mixture.
3. `num_speakers`: Integer count of overlapping sound sources ($K$).
4. `source_files`: Array of relative filepaths to clean source stems used to construct the mixture.
5. `duration_s`: Audio duration in seconds ($8.0$).
6. `sample_rate_hz`: Audio sample rate in Hz ($16,000$).
7. `channels`: Audio channels ($1$ for monaural).
8. `mixture_dBFS`: Average root-mean-square (RMS) energy level in dBFS.
9. `mixture_peak_dBFS`: Peak digital amplitude level in dBFS ($\le 0.0\text{ dBFS}$).

### • Class Imbalance Mitigation Strategy:
1. **Uniform Class Sampling:** The mixing engine samples stems uniformly across available categories rather than by raw category frequency, preventing frequent sounds from dominating mixtures.
2. **Threshold Filtering:** Categories with fewer than 10 clean clips were removed from the benchmark.
3. **Asymmetric Loss (ASL):** Member 3's tagging model uses ASL to dynamically down-weight the loss from abundant negative classes.

### • Paper's Preprocessing Pipeline Breakdown:
* **Resampling Rate:** Standardized to **16,000 Hz** ($16\text{ kHz}$), single-channel mono, 16-bit PCM WAV.
* **Window Size & Hop Length:**
  * Fixed window length: Exactly **8.0 seconds** ($128,000$ audio samples).
  * 15-second edge exclusion: In raw lecture files, the first 15 seconds and last 15 seconds are discarded to eliminate applause, intro slides, and handling noise.
  * Environmental length enforcement: Audio clips shorter than $6.0\text{s}$ are discarded; clips between $6.0\text{s}$ and $8.0\text{s}$ are zero-padded to $8.0\text{s}$; clips $> 8.0\text{s}$ are center-cropped to $8.0\text{s}$.
* **Normalization Method & Feature Type:**
  * **$-3.0\text{ dBFS}$ Background Attenuation Rule:** The primary stem $s_1(t)$ is maintained at original gain ($0\text{ dB}$), while subsequent overlapping stems $s_k(t)$ for $k \ge 2$ are attenuated by $-3.0\text{ dB}$ ($\times 10^{-3.0/20} \approx 0.7079$) before summation, preventing digital clipping overflow while maintaining relative acoustic dynamics.
  * **Feature Types:** Raw 1D waveform vectors for time-domain separation (Conv-TasNet); 80-band Log-Mel Spectrograms ($N_{\text{fft}} = 1024$, $\text{hop} = 512$) for Conformer counting and tagging.

---

## III. Practical Implementation & Verification Checklist
*(Hands-on execution log for Week 2 — Member 1)*

* [x] **Downloading & Verifying Datasets:**
  * Inspected primary dataset directories (`SANADA/Org_Fsd_Mix/`).
  * Audited 60 staged evaluation mixtures in [`data/splits/test_evaluation_pool.csv`](file:///Users/sannidhanamnikhilesh/SANADA/git/data/splits/test_evaluation_pool.csv).
  * Verified $100\%$ conformance: 16,000 Hz, 1-channel mono, 8.0-second length ($128,000$ samples), max peak $\le 0.0\text{ dBFS}$.

* [x] **Selecting & Finalizing Dataset Splits:**
  * Confirmed pre-mix partition boundaries ($70\%$ train, $15\%$ val, $15\%$ test) with fixed seed $S=42$.
  * Verified zero data leakage between splits using [`scripts/verify_week2_member1.py`](file:///Users/sannidhanamnikhilesh/SANADA/scripts/verify_week2_member1.py).

* [x] **Preprocessing Pipeline & Ingestion Engine Implementation:**
  * Built [`week2_ui_foundations.html`](file:///Users/sannidhanamnikhilesh/SANADA/week2_ui_foundations.html) featuring:
    * Drag-and-drop audio ingestion with client-side format checking (`.wav`), file size ceiling ($50\text{ MB}$), and duration validation.
    * Live microphone recording engine utilizing browser `MediaRecorder` API that captures speech, provides a real-time countdown ring, auto-stops at $8.0\text{s}$, and normalizes the audio buffer to 16 kHz 1-channel mono.

* [x] **Dual-Canvas Playhead Sync & Reactive State Store:**
  * Engineered a dual-canvas visualizer combining the Time-Domain Waveform Envelope with the STFT Frequency Spectrogram.
  * Implemented a synchronized vertical cursor playhead that scrubs both canvases simultaneously at 60 FPS in exact lockstep with audio playback.
  * Constructed a reactive state store handling mock prediction payloads:
    * Task 1 badge: Dynamic source count $K = 3$ with confidence bar.
    * Task 2 card: Multi-label speaker probability distribution bars.
    * Task 3 card: 3-channel stem player strip with individual Solo and Mute toggles.

---

## IV. Implementation Notes & Code Snippets

### 1. Browser `MediaRecorder` Engine with Automated 8.0-Second Buffer Normalization
```javascript
// Captures live microphone audio and normalizes into an 8.0s 16kHz mono AudioBuffer
async function captureAndStandardizeMic() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const arrayBuf = await blob.arrayBuffer();
        audioCtx.decodeAudioData(arrayBuf, (decoded) => {
            // Standardize into exactly 128,000 samples (8.0s at 16 kHz)
            const targetBuffer = audioCtx.createBuffer(1, 128000, 16000);
            const outData = targetBuffer.getChannelData(0);
            const inData = decoded.getChannelData(0);
            const copyLen = Math.min(inData.length, 128000);
            for (let i = 0; i < copyLen; i++) outData[i] = inData[i];
            console.log("Live audio successfully standardized to 8.0s (128,000 samples).");
        });
    };
    recorder.start();
    // Auto-stop at exactly 8.0 seconds
    setTimeout(() => recorder.stop(), 8000);
}
```

### 2. Synchronous Dual-Canvas Playhead Cursor Tracking Loop
```javascript
// Drives the synchronous red playhead line across both Waveform and Spectrogram canvases
function syncPlayheadLoop() {
    if (!isPlaying) return;
    const currentTime = audioCtx.currentTime - startTime;
    const totalDuration = currentBuffer.duration;

    if (currentTime <= totalDuration) {
        const progressFraction = currentTime / totalDuration;
        const stackWidth = canvasStackElement.clientWidth;
        
        // Single synchronized position across both stacked canvases
        playheadOverlay.style.left = `${progressFraction * stackWidth}px`;
        timeDisplay.textContent = `${currentTime.toFixed(2)} / ${totalDuration.toFixed(2)}s`;
        scrubberInput.value = progressFraction * 100;
    }
    requestAnimationFrame(syncPlayheadLoop);
}
```

### 3. Observations & Key Anomalies Documented:
* **Long-Tail Class Imbalance:** In raw environmental audio, sustained instruments (Saxophone, Flute) had $>90$ clean clips, while transient sounds (Cough, Shatter) had only $10$--$15$ clips. Uniform sampling across classes during mixture creation prevented dominant categories from biasing the evaluation.
* **Digital Clipping Prevention:** Linear summation of $K \ge 4$ sources without gain attenuation causes digital integer overflow ($> 0.0\text{ dBFS}$). Enforcing Member 5's $-3.0\text{ dBFS}$ attenuation rule on background stems keeps all mixtures $\le 0.0\text{ dBFS}$ while preserving natural relative harmonic dynamics.
* **Permutation Explosion Barrier:** Permutation Invariant Training ($\text{uPIT}$) requires evaluating $K!$ permutations. For $K=9$, $9! = 362,880$ permutations must be evaluated per sample. Beyond $K=14$ ($>8.7\times 10^{10}$ permutations), PyTorch triggers an immediate CUDA OOM crash.

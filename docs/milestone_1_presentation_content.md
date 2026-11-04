# Milestone 1: Presentation & Video Guide
## Project: SANADA - High-Density Audio Source Counting, Tagging, and Separation

---

## Slide 1: Introduction (Time: 0:00 - 0:45)

### Slide Content:
- **Project Title:** SANADA: Dynamic Machine Listening for Intractable Audio Mixtures
- **Team Members:** [Student Name 1], [Student Name 2], [Student Name 3], [Student Name 4]
- **Base Research Paper:** 
  - *"SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures, Evaluated with Baseline Models in Separation, Counting, and Tagging"* (Murthy & Prakash, 2026, IEEE DataPort).
- **One-Liner Summary:** 
  - Building an intelligent single-microphone audio system that counts, identifies, and un-mixes overlapping voices and sounds in crowded acoustic environments.

### 🎙️ Spoken Script (What to say on camera):
> "Hello everyone. Today, our team is presenting Milestone 1 for our project: **SANADA - Dynamic Machine Listening for Intractable Audio Mixtures**. 
> Our team consists of four members, and we are working on solving one of the most challenging problems in audio AI: understanding sound when multiple people and events collide in a single microphone recording. 
> Our work is based on the 2026 research paper on the SANADA benchmark published on IEEE DataPort, which studies how modern deep learning models behave when pushed beyond simple two-speaker setups into high-density, chaotic acoustic scenes."

---

## Slide 2: Understanding of Base Paper (Time: 0:45 - 1:30)

### Slide Content:
- **Key Idea of the Paper:**
  - Real-world acoustic scenes contain unknown, changing numbers of simultaneous sounds.
  - Existing datasets artificially cap evaluation at only 2 to 4 speakers (e.g., WSJ0-2mix, LibriMix).
  - The paper introduces **SANADA** (Samslesha-Nada): a high-density, reproducible benchmark scaling speech up to 9 speakers and environmental audio up to 32 classes.
- **Problem Solved by the Paper (In Simple Words):**
  - In real life, recording 9 people talking in a room while capturing their isolated clean voices simultaneously is physically impossible.
  - SANADA solves this by synthesizing controlled mixtures from clean stems with:
    - **Zero Data Leakage:** Pre-mix 70/15/15 file partitioning so test utterances are never heard during training.
    - **No Digital Clipping:** Linear -3.0 dB attenuation on background stems to preserve natural dynamic range without distortion.

### 🎙️ Spoken Script (What to say on camera):
> "To understand our base paper, let's look at why it was created. Almost all existing speech separation research has been stuck in the lab, testing models on only two or three clean voices. But in real life, rooms are full of competing voices. 
> You cannot physically record 9 people talking in a crowded hall and simultaneously capture each person's isolated clean voice. 
> The SANADA paper solves this by creating a massive, controlled synthetic benchmark. It provides 27,000 isolated clean stems and 15,000 mixtures, strictly separating data before mixing to eliminate data leakage, and applying a minus 3 dB volume rule so overlapping sounds don't distort digitally."

---

## Slide 3: Problem Definition (Time: 1:30 - 2:30)

### Slide Content:
- **Technical & Scientific Problem:**
  - **The Complexity Ceiling:** Deep learning models suffer a steep performance collapse once mixture density exceeds 4 overlapping sources ($K > 4$).
  - **Severe Spectral Masking:** When 4 to 9 people speak, natural breath pauses and silent gaps disappear completely; louder sounds mask quieter frequencies across both time and frequency.
  - **The Memory Meltdown ($O(K!)$):** Standard separation training (uPIT) checks all possible pairings between predicted and true voices. At 15 sources, this requires **over 1.3 trillion combinations**, causing immediate GPU out-of-memory crashes even on enterprise NVIDIA A100 GPUs.
- **Business & Real-World Relevance:**
  - **Conferencing Tools (Zoom / Teams):** When multiple people speak simultaneously, transcription models fail and generate gibberish.
  - **Assistive Hearing Aids:** Needs to isolate one target conversation in noisy restaurants.
  - **Emergency Dispatch & Defense:** Isolating a caller's voice from background gunshots, sirens, or screaming.

### 🎙️ Spoken Script (What to say on camera):
> "The technical problem we are addressing is what researchers call the **Complexity Ceiling**. When only two people speak, there are natural pauses and frequency gaps where an AI can easily tell them apart. But when 4, 6, or 9 people speak, all pauses vanish. The sound becomes a solid wall of acoustic energy.
> Furthermore, current separation training algorithms use Permutation Invariant Training, which evaluates combinations with a factorial complexity of K-factorial. For 15 sounds, that is over 1.3 trillion combinations, causing computers to instantly crash with out-of-memory errors. 
> Solving this is critical for real-world applications: automated Zoom and Teams meeting transcriptions, smart hearing aids that allow people to hear a single friend in a noisy cafe, and emergency 911 dispatch systems that need to isolate a caller's voice from chaotic background noise."

---

## Slide 4: Scope of Project (Time: 2:30 - 3:15)

### Slide Content:
- **In-Scope (What We Are Implementing):**
  - **Three Interconnected Tasks:**
    1. *Source Counting:* Predict active source count $K$ (e.g., 2, 3, 4).
    2. *Multi-Label Tagging:* Detect which speakers or sound categories are present.
    3. *Audio Source Separation:* Reconstruct clean individual audio files from the mixture.
  - Focus on organizational speech scaling from $K = 2$ up to $K = 4$ (and extending to $K = 9$).
  - Implementing the **Dynamic-Cardinality Pipeline**: Counter informs Separator to eliminate empty output channels.
  - Building a web/mobile demonstration interface where users upload mixtures and receive separated tracks.
- **Out-of-Scope (What We Are Not Doing):**
  - Multi-microphone array hardware / spatial beamforming (strictly monaural single-channel).
  - Unconstrained separation beyond hardware feasibility ($K \ge 15$ with standard uPIT).
  - Multi-lingual conversational speech (dataset is English organizational lectures and FSD audio).

### 🎙️ Spoken Script (What to say on camera):
> "Here is our project scope. 
> In-scope: We are implementing all three core machine listening tasks on the SANADA benchmark. Specifically, we are building a Source Counter to estimate active speakers, an Acoustic Tagger to identify who is talking, and a Conv-TasNet Separator to extract the individual clean audio tracks. We are starting with variable mixtures up to 4 speakers and scaling toward 9 speakers. We are also building a responsive web demo for real-time inference.
> Out-of-scope: We are strictly focusing on single-channel monaural audio—we are not using multi-microphone hardware arrays, because our goal is an algorithmic stress test on standard single-mic recordings. We are also avoiding impossible factorial combinations above 15 sources."

---

## Slide 5: Proposed Solution & Our Contribution (Time: 3:15 - 4:15)

### Slide Content:
- **High-Level Idea (Our Core Contribution):**
  - **The Two-Stage Teamwork Pipeline (Count First, Then Separate):**
    - Traditional models hardcode output channels (e.g., 9 slots) and pad empty channels with zeros, causing massive output confusion and wasted memory.
    - Our solution uses an upfront **Source Counter** to determine the exact number of active sources ($K$).
    - It dynamically configures the separator to open *only* $K$ output channels, completely eliminating dummy zero-padding ambiguity.
  - **Target Query Extraction:**
    - To prevent the $O(K!)$ memory explosion, we transition toward extracting target sources one-by-one, replacing the 1.3-trillion combinatorial explosion with a linear, memory-safe process.
- **Expected Outcome:**
  - High counting accuracy ($> 75\%$ exact match).
  - Robust multi-label tagging ($> 0.90$ Macro F1).
  - Clean separation audio achieving an SI-SDR score $> 4.2\text{ dB}$ on 4-speaker mixtures without GPU memory failure.

### 🎙️ Spoken Script (What to say on camera):
> "Our proposed solution introduces a smart **Two-Stage Dynamic Pipeline**. 
> In traditional systems, models have hardcoded output channels. If you have a 9-channel model but only 2 people are talking, the other 7 channels are filled with empty zeros. The model gets completely confused about where to send the sound.
> Our contribution is to connect the Counter and the Separator as a team. First, our Conformer-based Counter listens to the mixture and determines: 'There are exactly 3 speakers here.' It then dynamically configures the separator to extract only 3 tracks. 
> Furthermore, by employing target query extraction—pulling out voices one-by-one—we bypass the 1.3-trillion permutation problem, ensuring high-quality voice separation with zero risk of GPU memory crashes."

---

## Slide 6: Basic Architecture Diagram & Workflow (Time: 4:15 - 5:15)

### Slide Content:
- **Workflow Diagram:**
  - **Input:** Single-channel 8.0s WAV audio (16 kHz mono, 128,000 samples).
  - **Dual Feature Stream:**
    - *Path A:* Log-Mel Spectrogram (2D frequency picture) $\rightarrow$ fed into Stage 1 (Counting & Tagging).
    - *Path B:* Raw 1D Waveform (time domain) $\rightarrow$ fed into Stage 2 (Conv-TasNet Separation to preserve true audio phase).
  - **Stage 1 (Analysis):** Conformer Regressor predicts $K$; Conformer Classifier with Asymmetric Loss (ASL) outputs multi-hot speaker vector.
  - **Stage 2 (Separation):** Conv-TasNet dynamically unmixes the audio into $K$ clean waveforms using SI-SDR loss.
  - **Output:** $K$ playable `.wav` audio files + metadata labels.
- **Libraries & Stack:** Python, PyTorch, Librosa, SoundFile, Pandas, Express.js, React.

### 🎙️ Spoken Script (What to say on camera):
> "Here is our system architecture. 
> We feed an 8-second audio mixture into our pipeline. We split it into two feature paths: 
> For Counting and Tagging, we convert the audio into a Log-Mel Spectrogram—a 2D sound picture that allows the model to clearly inspect pitch patterns.
> For Separation, we feed the raw 1D waveform directly into Conv-TasNet. This is crucial because spectrograms discard audio phase, leading to robotic, metallic voices, whereas raw waveforms preserve natural sound quality.
> Stage 1 analyzes the spectrogram, predicting the speaker count and active identities. This count dynamically routes into Stage 2, which separates the raw wave into clean, individual, playable WAV files. Our stack is built on PyTorch, Librosa, SoundFile, and a full-stack web interface."

---

## Slide 7: Closing & Next Milestone Roadmap (Time: 5:15 - 6:00)

### Slide Content:
- **Quick Recap:**
  - Solved data loading and verification on the 28 GB SANADA suite.
  - Built verified PyTorch data pipelines for Counting, Tagging, and Separation.
  - Formulated the dynamic count-then-separate architecture to beat the complexity ceiling and avoid memory crashes.
- **Confidence Statement:**
  - *"By next milestone, we plan to show fully trained Source Counting and Tagging models achieving over 75% accuracy on variable mixtures, along with our preliminary Conv-TasNet separation outputs and the interactive web demo prototype."*

### 🎙️ Spoken Script (What to say on camera):
> "To conclude: for Milestone 1, we have thoroughly analyzed the research paper, audited the local 28 GB SANADA suite, and implemented our end-to-end data loading and verification notebook. We have designed a dynamic architecture that solves both the complexity ceiling and the memory crash problem.
> **By next milestone, we plan to show our fully trained Source Counting and Multi-Label Tagging models achieving over 75% exact accuracy, alongside our preliminary Conv-TasNet separation audio demos and our interactive web interface.**
> Thank you, and we are ready for any questions!"

---

## 👥 Team Work Breakdown (For Slide Reference or Q&A)
- **Member 1 (Data Engineering & Pipeline Lead):** Dataset audit, pre-mix splitting (70/15/15), feature transformations (Waveforms & Mel-Spectrograms), PyTorch DataLoader.
- **Member 2 (Task 1: Source Counting Lead):** Conformer regression model, predicting continuous count $K$, evaluating MAE and exact accuracy.
- **Member 3 (Task 2: Multi-Label Tagging Lead):** Multi-label presence detection, handling class imbalance with Asymmetric Loss (ASL), mAP/F1 evaluation.
- **Member 4 (Task 3: Source Separation Lead):** Conv-TasNet time-domain model, uPIT & SI-SDR loss implementation, query extraction logic.

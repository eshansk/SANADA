# Week 1 Project Logbook: Member 3 — Multi-Label Acoustic Tagging Lead
**Project**: SANADA (Samslesha-Nada) Multi-Task Audio Intelligence  
**Role**: Member 3 — Label Tagging Lead  
**Timeline**: Week 1 of 8  

---


## I. Initiation & Paper Breakdown

### 1. Abstract Summary
Practical acoustic environments contain an unknown, time-varying number of simultaneous audio sources, yet conventional audio corpora cap evaluations at $K \le 4$ sources. The SANADA suite introduces three domain-isolated, reproducible mixture corpora: **Org-Mix-4** ($K=2$ to $4$), **Org-Mix-9** ($K=2$ to $9$), and **FSD-Mix** (up to 32 classes, $K \le 32$). The paper establishes baseline benchmarks across three core tasks: monaural separation (Conv-TasNet), source counting (Conformer Regressor), and multi-label tagging (Conformer + Asymmetric Loss). The empirical findings expose a sharp "Complexity Ceiling" beyond $K=4$, severe $O(K!)$ memory explosion in permutation training at $K \ge 15$, and performance drops under dense spectral overlap.

### 2. Problem Statement (According to the Paper)
In polyphonic mixtures where multiple environmental sound events or human speakers overlap continuously, existing single-label audio classifiers fail because multiple acoustic signatures collide in the same time-frequency regions. For Member 3, the problem is **Multi-Label Acoustic Tagging (Acoustic Presence Detection)**: given an unconstrained single-channel mixture $x(t)$, detect the presence or absence of each candidate sound class without requiring explicit spatial separation, while coping with extreme class imbalance where only 2 to 4 classes are active out of 32 possible categories.

### 3. Input / Output Specifications
* **Input Data**:
  * Single-channel (monaural) 16-bit PCM WAV audio.
  * Sample Rate: $16{,}000\text{ Hz}$.
  * Fixed Duration: $8.0\text{ seconds}$ ($N = 128{,}000$ discrete time-domain samples).
  * Feature Transform: 80-band Log-Mel Spectrogram ($n_{\text{fft}} = 1024$, $\text{hop\_length} = 512$, giving a tensor of shape $[B, 80, 251]$).
* **Target Output**:
  * Multi-hot binary vector $\mathbf{y} \in \{0, 1\}^C$ where $C$ is the class vocabulary size:
    * **Org-Mix-4**: $C = 4$ speaker identities (`Speaker_4_1` to `Speaker_4_4`).
    * **Org-Mix-9**: $C = 9$ speaker identities (`Speaker1` to `Speaker9`).
    * **FSD-Mix**: $C = 32$ environmental categories.
  * Continuous probability predictions $\mathbf{p} = \sigma(\mathbf{z}) \in [0.0, 1.0]^C$.

### 4. Possible Implementations / Solutions (According to the Paper)
* **Backbone**: Convolution-augmented Transformer (**Conformer**), combining multi-head self-attention (global sequence context) with depthwise separable convolutions (local fine-grained acoustic features).
* **Loss Function**: **Asymmetric Loss (ASL)** with focusing parameters $\gamma_- = 4.0$, $\gamma_+ = 0.0$, and probability margin clip $m = 0.05$. This prevents millions of easy negative classes from dominating gradient updates.
* **Alternative Approaches Considered in Literature**:
  * ResNet / CNN-14 (Audio Spectrogram CNNs): Good local receptive fields but limited long-range temporal attention across 8-second clips.
  * Standard Binary Cross-Entropy (BCE): Suffers from severe positive/negative label gradient imbalance.
  * Standard Focal Loss: Equal focusing on positive and negative samples, whereas tagging requires asymmetric dampening of negative gradients.

### 5. Possible Enhancements / Extensions Considered
* **Dynamic Thresholding**: Rather than a static $\tau = 0.5$ decision threshold across all 32 classes, tune class-specific thresholds on validation splits to compensate for transient vs sustained sounds.
* **Multi-Scale Feature Aggregation**: Combining shallow convolutional feature maps with deep transformer tokens to better detect high-frequency transient events (e.g., `Keys_jangling`, `Shatter`).
* **Tag-Guided Separation Conditioning**: Sharing predicted label embeddings with Member 4 (Separation Lead) to perform query-based source extraction.

### 6. Selected Enhancement for this Project
* **Conformer Backbone with Per-Class Calibrated Decision Thresholds and Asymmetric Loss (ASL)**: Preserves the benchmark parity while adding precision-calibrated thresholds to boost macro F1-score on rare transient environmental events.

---

## II. Background / Literature Survey

### 1. Introduction & Domain Study
Sound Event Detection (SED) and Multi-Label Acoustic Tagging are fundamental components of ambient intelligence, surveillance, and automated scene analysis. In high-density scenes (lecture halls, busy streets, industrial areas), multiple sounds occur simultaneously without pauses. Unlike single-label classification where classes are mutually exclusive via Softmax, multi-label tagging requires independent Bernoulli modeling per class via Sigmoid activation.

### 2. Technology Review (Libraries, Frameworks & Tools)
* **PyTorch 2.x & Torchaudio**: Core deep learning framework for tensor computation, dynamic autograd graphs, and GPU acceleration.
* **Torchaudio Models & Transforms**: Built-in `torchaudio.models.Conformer` backbone and GPU-accelerated `MelSpectrogram` extraction.
* **Librosa & Soundfile**: High-precision audio I/O and Constant-Q Transform (CQT) spectral exploration.
* **Scikit-learn**: Industry standard metrics suite for macro-averaged mAP, F1-score, and ROC-AUC computation.

### 3. Historical Analysis & Timelines
* **2016–2017**: Deep Clustering and Utterance-level Permutation Invariant Training (uPIT) introduce neural multi-source processing.
* **2018–2019**: CNN architectures (PANNs, CNN-14) dominate DCASE challenges for audio tagging.
* **2020**: Conformer (Gulati et al.) merges multi-head self-attention with depthwise convolution, setting state-of-the-art results in speech and audio modeling.
* **2021**: Asymmetric Loss (Ben-Baruch et al., ICCV 2021) solves extreme negative label imbalance in multi-label classification.
* **2026**: SANADA benchmark formally establishes the variable-$K$ intractable mixture paradigm and proves the breakdown of existing models beyond $K=4$.

### 4. Related Works Mapped
1. **Gulati et al. (2020)** — *Conformer: Convolution-augmented Transformer for Speech Recognition*: Introduced macaron-style feedforward modules and depthwise convolutions for sequence modeling.
2. **Ben-Baruch et al. (2021)** — *Asymmetric Loss for Multi-Label Classification (ICCV)*: Formulated negative probability shifting and asymmetric focusing to eliminate easy negative gradient noise.
3. **Fonseca et al. (2022)** — *FSD50K: An Open Dataset of Human-Labeled Sound Events*: Standardized Freesound audio taxonomy under the AudioSet ontology.
4. **Luo & Mesgarani (2019)** — *Conv-TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation*: Landmark time-domain separation model used as baseline in SANADA.
5. **Wisdom et al. (2021)** — *What's all the FUSS about free universal sound separation data?*: Benchmark for variable sound events, highlighting the need for controlled variable-$K$ datasets.

---

## III. Practical Implementation & Verification Checklist

All hands-on execution steps for Week 1 have been implemented and validated via `models/tagging/week1_tagging_playboard.py`:

* [x] **Audio Fundamentals & Playboard Setup**:
  * Verified audio loader via `soundfile` and `torchaudio`.
  * Verified sample file `mix_00000.wav`: **$16{,}000\text{ Hz}$**, **$8.00\text{ s}$**, **$128{,}000\text{ samples}$**, single-channel (mono), RMS energy $= 0.02115$, peak level $= -16.08\text{ dBFS}$.
* [x] **Audio Representation & Display**:
  * Computed **STFT magnitude**: shape `[513, 251]` ($n_{\text{fft}} = 1024$, $\text{hop\_length} = 512$).
  * Computed **80-band Mel-Spectrogram**: shape `[1, 80, 251]`.
  * Computed **Constant-Q Transform (CQT)**: shape `[84, 251]` spanning 7 octaves.
* [x] **SDLC & Repository Setup**:
  * Established `models/tagging/` scaffold containing:
    * `label_vocab.py`: Fixed vocabulary mappings and multi-hot vector encoders for 32 FSD and 9 speech classes.
    * `asl_loss.py`: Numerical PyTorch implementation of Asymmetric Loss.
    * `conformer_tagger.py`: Conformer backbone with Mel front-end and linear projection head.
    * `evaluate.py`: Multi-label metrics engine for mAP, F1, and ROC-AUC.
* [x] **Bulk Data Pipeline / Sample Inspection**:
  * Ingested active labels from `SANADA/Org_Fsd_Mix/FSD_Mix_Variable_Sounds/train/metadata.csv`.
  * Encoded ground truth (`['Meow', 'Oboe']`) into a 32-dimensional multi-hot vector and decoded it back with 100% precision.
* [x] **Feature Exploration & Model Smoke Verification**:
  * Initialized `ConformerTaggingModel` ($996{,}152$ trainable parameters).
  * Executed forward pass on batch size 2: output logits shape `[2, 32]`.
  * Computed Asymmetric Loss ($\text{loss} = 0.0614$).
  * Ran backpropagation: all gradients computed successfully without `NaN`s (average gradient norm $= 0.009095$).
  * Computed spectral cosine similarity between two mixtures ($S_C = 0.7284$).

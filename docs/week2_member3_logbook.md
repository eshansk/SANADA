
# Week 2 Project Logbook: Member 3 — Multi-Label Acoustic Tagging Lead
**Project**: SANADA (Samslesha-Nada) Multi-Task Audio Intelligence  
**Role**: Member 3 — Label Tagging Lead  
**Timeline**: Week 2 of 8  

---

## I. Dataset Identification & Collection Strategy
*(To be completed after studying Section IV of the research paper)*

### 1. Primary Dataset Name & Source
* **Primary Dataset Name**: SANADA Acoustic Suite (**FSD-Mix**, **Org-Mix-4**, **Org-Mix-9**).
* **Source**: IEEE DataPort (DOI: [10.21227/z7hv-e014](https://dx.doi.org/10.21227/z7hv-e014)).
* **Source Material**:
  * Freesound Dataset / FSDKaggle2018 audio corpus under AudioSet ontology (environmental sounds).
  * Web lecture audio recorded with close-talking studio microphones (organisational speech).

### 2. Dataset Specifications (Size, Format, License, Download URL)
* **Format & Size**:
  * Audio Standard: Single-channel (monaural), 16 kHz, 16-bit PCM WAV.
  * Clip Duration: Exactly 8.0 seconds ($128{,}000$ samples).
  * Pre-mix Stems: 27,112 files ($60.25\text{ hours}$, $6.47\text{ GB}$).
  * Synthesized Mixtures: 15,600 files ($34.66\text{ hours}$, $3.72\text{ GB}$).
  * Total Repository Size: $28.27\text{ GB}$ (including raw recordings and stems).
* **License & Source URL**:
  * Open Academic Research License / Creative Commons via IEEE DataPort.
  * URL: `https://dx.doi.org/10.21227/z7hv-e014`.

### 3. Class Distribution & Duration Statistics (According to Paper)
* **Environmental Domain (FSD-Mix)**:
  * 32 curated sound event categories.
  * Natural open-domain audio exhibits an intrinsic long-tail distribution: sustained classes (e.g., *Saxophone*, *Flute*, *Clarinet*) provide up to 110 stems, whereas transient classes (e.g., *Shatter*, *Keys_jangling*, *Cough*) contain fewer than 20 stems (mean $= 34.8$ stems/class).
  * In the synthetic mixing engine, stems are sampled **uniformly across classes** to decouple source-separation difficulty from label frequency.
* **Speech Domains (Org-Mix-4 / Org-Mix-9)**:
  * 4 and 9 speaker identities respectively, strictly balanced with 1,000 train, 500 val, and 500 test stems per speaker identity.

### 4. Author-Specified Train / Validation / Test Splits
* Fixed file-level **70 / 15 / 15** partition across all categories before any temporal windowing or mixing.
* Seed initialized with deterministic PRNG seed $S = 42$ to prevent cross-split data leakage.
* In our active `FSD_Mix_Variable_Sounds` partition:
  * **Train**: 990 mixtures (drawn from 763 disjoint stems).
  * **Val**: 90 mixtures (drawn from 148 disjoint stems).
  * **Test**: 90 mixtures (drawn from 194 disjoint stems).

### 5. Alternative or Additional Datasets Considered
* **FSD50K / FSDKaggle2018**: Large-scale single-channel clips with dense AudioSet labels, but lack isolated unmixed stems to study high-density overlapping mixtures.
* **AudioSet**: Extremely broad taxonomy but consists of noisy YouTube audio without ground-truth clean stems.
* **LibriMix / WHAM! / WHAMR!**: High-quality speech separation datasets, but strictly capped at low densities ($K \le 3$).
* **FUSS**: Universal sound separation benchmark, but lacks variable-$K$ systematic density scaling up to $K=32$.

---

## II. Dataset Organization & Preprocessing Specifications
*(To be completed during data pipeline design)*

### 1. Folder Architecture Established
* Pre-cleaned and partitioned structure initialized: `[x] Yes / [ ] No`
```text
SANADA/
├── FSD_Master/                       # Isolated 8.0s ground-truth environmental stems (train, val, test)
├── FSD_Master_Prepared/              # 32 curated class folders
├── Multichannel4SpeakersTrim/        # Isolated 4-speaker speech stems
├── Multichannel9SpeakersTrim/        # Isolated 9-speaker speech stems
└── Org_Fsd_Mix/                      # Algorithmic polyphonic mixture directories
    ├── FSD_Mix_Variable_Sounds/      # 32-class variable mixtures (train, val, test)
    ├── Org_Mix_4_Sounds_Variable/    # 4-speaker variable mixtures (train, val, test)
    └── Org_Mix_9_Variable/           # 9-speaker variable mixtures (train, val, test)
```

### 2. Metadata Schema (CSV Tracking)
Tracked via unified `metadata.csv` manifests inside each partition:
* `mixture_path`: Absolute / relative path to 8.0-second WAV file.
* `speaker_labels` / `fsd_labels`: Ground-truth list of active class strings (e.g., `["Meow", "Oboe"]`).
* `num_speakers`: Active source count cardinality ($K$).
* `source_files`: Pre-mix file paths of individual stems used to construct the mixture.
* `duration_s` & `sample_rate_hz` & `channels`: Format verification attributes ($8.0$, $16000$, $1$).
* `mixture_dBFS` & `mixture_peak_dBFS`: Energy level statistics.

### 3. Class Imbalance Mitigation Strategy
* **Approach Selected**: **Asymmetric Loss (ASL) with Margin Clipping + SpecAugment**.
* In 32-class environmental audio, mixtures with $K \le 3$ active sources contain $29$ negative classes and only $3$ positive classes.
* We configure ASL with:
  * Negative focusing: $\gamma_- = 4.0$ (strongly suppresses gradients from easy negatives).
  * Positive focusing: $\gamma_+ = 0.0$ (preserves full gradient signal from rare positive classes).
  * Margin threshold: $\text{clip} = 0.05$ (completely zeros out negative loss for $p < 0.05$).

### 4. Paper's Preprocessing Pipeline Breakdown
* **Resampling Rate**: $16{,}000\text{ Hz}$ (single-channel 16-bit PCM WAV).
* **Window Size & Hop Length**: Hann window of $N_{\text{fft}} = 1024$ samples ($64\text{ ms}$), Hop length $= 512$ samples ($32\text{ ms}$).
* **Normalization & Feature Type**:
  * 80-channel Log-Mel Filterbank energies (`n_mels=80`).
  * Numerical clamping: $\log(\max(\text{Mel}, 10^{-6}))$.
  * Input tensor shape to Conformer: $[B, 80, 251]$ (or transposed to $[B, 251, 80]$).

---

## III. Practical Implementation & Verification Checklist
*(Hands-on execution log for Week 2 based on your workflow plan)*

* [x] **Downloading & Verifying Datasets**:
  * Verified local availability of pre-cleaned SANADA dataset.
  * Verified 100% readability across sampled audio files (corrupted file count $= 0$).
  * Confirmed all files adhere to $16{,}000\text{ Hz}$, $8.00\text{ s}$ ($128{,}000$ samples), and mono channel configuration.
* [x] **Selecting & Finalizing Dataset Splits**:
  * Ingested `train/metadata.csv` (990 items), `val/metadata.csv` (90 items), `test/metadata.csv` (90 items).
  * Extracted pre-mix stems from `source_files` column and verified disjoint sets:
    * $\text{Train Stems} \cap \text{Val Stems} = 0$
    * $\text{Train Stems} \cap \text{Test Stems} = 0$
    * $\text{Val Stems} \cap \text{Test Stems} = 0$
  * Confirmed **zero data leakage** across all evaluation splits.
* [x] **Preprocessing Pipeline Implementation**:
  * Implemented [`SanadaTaggingDataset`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/dataset.py) with on-the-fly Mel-spectrogram extraction.
  * Confirmed tensor alignment: Features $[80, 251]$, Targets $[32]$ (binary multi-hot).
  * Profiled feature extraction throughput: **$421.11\text{ samples/second}$**.
* [x] **Augmentation Implementation & DataLoader Optimization**:
  * Implemented [`AudioAugmentationPipeline`](file:///Users/manaswini/Desktop/sanada_folder/models/tagging/transforms.py) combining:
    * **SpecAugment**: Frequency masking ($F=15$) and Time masking ($T=35$).
    * **Gaussian Noise**: Additive zero-mean Gaussian noise ($\sigma = 0.005$).
  * Benchmarked PyTorch DataLoader throughput with augmentations enabled: **$25.97\text{ batches/second}$** ($415.45\text{ samples/sec}$ at $\text{batch\_size}=16$), exceeding the $\ge 10\text{ batches/sec}$ requirement.
  * Executed batch forward and backward pass: Verified all 32 class heads received non-zero gradients without `NaN`s ($\text{loss} = 0.3701$).

---

## IV. Implementation Notes & Code Snippets

### 1. Dataset Verification Observations / Anomalies Found
* **Observation**: In `FSD_Mix_Variable_Sounds/train/metadata.csv`, the column storing labels is named `speaker_labels` rather than `fsd_labels`, but contains environmental sound event strings formatted as Python lists (e.g., `'["Meow", "Oboe"]'`).
* **Resolution**: The dataset parser uses `ast.literal_eval()` on `speaker_labels`, dynamically resolving the active classes across both environmental and speech corpora.

### 2. Core Preprocessing / Augmentation Code Snippet
```python
# From models/tagging/transforms.py and dataset.py
import torch
import torch.nn as nn
import torchaudio.transforms as T

class AudioAugmentationPipeline(nn.Module):
    def __init__(self, freq_mask=15, time_mask=35, noise_std=0.005):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask)
        self.noise_std = noise_std

    def forward(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        # 1. Add low-level Gaussian noise for regularisation
        if self.training and self.noise_std > 0:
            log_mel_spec = log_mel_spec + torch.randn_like(log_mel_spec) * self.noise_std
        # 2. SpecAugment: Frequency & Time Masking
        out = self.freq_masking(log_mel_spec)
        out = self.time_masking(out)
        return out
```

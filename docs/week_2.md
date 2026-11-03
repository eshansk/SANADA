Week – 2:

I. Dataset Identification & Collection Strategy
(To be completed after studying Section IV of the research paper)

- Primary Dataset Name & Source:
  - Dataset Name: SANADA (Samslesha-Nada) Dataset Suite.
  - Audio Sources: Public organizational lecture videos recorded with close-talking microphones (for speech pools Org-Mix-4 and Org-Mix-9) and the Freesound / FSDKaggle2018 audio corpus (for environmental pool FSD-Mix).

- Dataset Specifications (Size, Format, License, Download URL):
  - Format & Size:
    - Audio Format: Single-channel (monaural), 16,000 Hz sampling rate, 16-bit PCM WAV.
    - Duration: Fixed 8.0 seconds per clip (128,000 samples).
    - Repository Size: Full repository is 28.27 GB (60.25 hours of 27,112 pre-mix stems [6.47 GB] + 34.66 hours of ~15,600 synthesized mixtures [3.72 GB], plus raw lecture recordings).
    - Sub-dataset Sizes: Org-Mix-4 is 1.14 GB (~4,800 mixtures), Org-Mix-9 is 0.86 GB (~3,600 mixtures), FSD-Mix is 1.72 GB (~7,200 mixtures).
  - License & Source URL:
    - Repository Host: IEEE DataPort (https://dx.doi.org/10.21227/z7hv-e014, DOI: 10.21227/z7hv-e014).
    - License & Ethics: Open for non-commercial research use. All human speakers provided consent, and identities were anonymized (Speaker_1 to Speaker_9).

- Class Distribution & Duration Statistics (According to paper):
  - Fixed Duration: Every audio clip is trimmed, padded, or windowed to exactly 8.0 seconds.
  - Speech Corpus Pools:
    - Multichannel4SpeakersTrim: 4 speakers, each allocated 1,000 train, 500 validation, and 500 test stems (8,000 stems total, 17.78 hours).
    - Multichannel9SpeakersTrim: 9 speakers, each allocated 1,000 train, 500 validation, and 500 test stems (18,000 stems total, 40.00 hours).
  - Environmental Audio Pool (FSD_Master):
    - 32 curated sound classes containing 1,112 total stems (2.47 hours).
    - Long-tail class imbalance: continuous sustained sounds (such as Saxophone with >110 stems and Flute with ~95 stems) are abundant, whereas transient impulsive events (such as Cough, Shatter, and Squeak with 10-15 stems) are scarce (mean: 34.8 stems/class).
  - Acoustic Quality Statistics: Organizational speech stems exhibit a mean WADA-SNR of 18.4 +- 2.1 dB and an RT60 reverberation time of ~0.47 s, confirming clear, near-anechoic recordings.

- Author-Specified Train / Validation / Test Splits:
  - Pre-Mix Partitioning (Stem Level): 70% train / 15% validation / 15% test partitioning is applied directly to raw isolated files using random seed S = 42 before any trimming or mixing takes place.
  - Mixture Quotas: For each mixture configuration, the mixing engine generates exactly 1,000 training mixtures, 100 validation mixtures, and 100 test mixtures.
  - Zero Data Leakage: Stems appearing in test mixtures never appear in training or validation sets.

- Alternative or Additional Datasets Considered:
  - WSJ0-2mix / LibriMix: Clean speech separation benchmarks, restricted strictly to fixed 2-speaker and 3-speaker conditions (K <= 3).
  - WHAM! / WHAMR!: Speech separation under noise and reverberation, capped at K <= 3.
  - AudioSet / FSD50K / ESC-50 / DESED: Audio event tagging corpora providing broad class diversity, but lacking isolated ground-truth stems for separation metrics.
  - FUSS: Domain-agnostic universal separation benchmark, but restricted to low source density (K <= 4).
  - MUSDB18 / Slakh: Music separation datasets with multitrack stems, restricted to fixed instrument taxonomies (vocals, drums, bass) and K <= 4.


II. Dataset Organization & Preprocessing Specifications
(To be completed during data pipeline design)

- Folder Architecture Established:
  - data/raw/, data/processed/, data/splits/ initialized? [x] Yes / [ ] No
    - Raw: `SANADA/Multichannel/`, `SANADA/Multichannel9/`, `SANADA/FSDKaggle2018.audio_train/`.
    - Processed Clean Stems: `SANADA/Multichannel4SpeakersTrim/`, `SANADA/Multichannel9SpeakersTrim/`, `SANADA/FSD_Master/`.
    - Synthesized Mixtures: `SANADA/Org_Fsd_Mix/` (containing 13 fixed-K and variable-K sub-datasets with train, val, and test splits).

- Metadata Schema (CSV Tracking):
  - Attributes tracked (filename, label, duration, split, source):
    - `mixture_path`: Relative file path to the 8.0-second mixture WAV file.
    - `speaker_labels` / `fsd_labels`: Active speaker identities or sound classes in the clip.
    - `num_speakers`: Integer count of active overlapping sources (K).
    - `source_files`: File paths to clean isolated stems used to construct the mixture.
    - `duration_s`: Audio duration in seconds (8.0).
    - `sample_rate_hz`: Audio sample rate in Hz (16,000).
    - `channels`: Channel count (1 for mono).
    - `mixture_dBFS`: Average RMS energy level in dBFS.
    - `mixture_peak_dBFS`: Peak digital amplitude level in dBFS.

- Class Imbalance Mitigation Strategy:
  - Approach selected (oversampling / undersampling / re-weighting):
    - Uniform Class Sampling: The mixing algorithm draws stems uniformly across classes rather than by class frequency, preventing common categories from dominating mixtures and isolating density evaluation from label bias.
    - Threshold Filtering: Sound categories with fewer than 10 valid clips after length constraints are eliminated.
    - Asymmetric Loss (ASL): Used during multi-label classification to down-weight easy negative labels.

- Paper's Preprocessing Pipeline Breakdown:
  - Resampling rate:
    - Standardized to 16,000 Hz (16 kHz), single-channel mono, 16-bit PCM WAV.
  - Window size & Hop length:
    - Fixed clip duration: 8.0 seconds (128,000 audio samples).
    - Speech buffering: 15.0-second buffer skipped at both start and end of lecture files to eliminate intro music, applause, and handling noise; 8.0-second segments randomly sampled from the middle.
    - Environmental curation: clips < 6.0 s discarded, clips > 8.0 s truncated, and clips between 6.0 s and 8.0 s zero-padded with silence to exactly 8.0 s.
  - Normalization method & Feature type (mel / CQT / MFCC):
    - Acoustic Gain Attenuation: Primary stem kept at full scale (s_1(t)), while subsequent overlapping stems are scaled by -3.0 dB (10^(-3.0/20) ≈ 0.7079) before summation to prevent digital clipping while preserving relative dynamics.
    - Feature Types: Raw 1D waveforms for separation models; Log-Mel spectrograms (n_mels=80, n_fft=1024, hop_length=512) for counting and tagging models.


III. Practical Implementation & Verification Checklist
(Hands-on execution log for Week 2 based on your workflow plan)

- [x] Downloading & Verifying Datasets: Primary audio partitions inspected from SANADA repository; confirmed 16 kHz mono format, validated 128,000 sample count, verified no corrupted files.
- [x] Selecting & Finalizing Dataset Splits: Pre-mix file partitioning verified with 70% train, 15% val, and 15% test split using seed S = 42; verified zero speaker/clip leakage between sets.
- [x] Preprocessing Pipeline Implementation: Verified 15.0-second boundary buffer exclusion, validated 8.0-second window lengths, and confirmed audio extraction scripts.
- [x] Augmentation Implementation & DataLoader Optimization: Validated the -3.0 dB background attenuation rule; built unified PyTorch Dataset and DataLoader pipeline supporting batch loading for source counting, tagging, and separation.


IV. Implementation Notes & Code Snippets

- Dataset Verification Observations / Anomalies Found:
  - Long-Tail Class Imbalance: In raw environmental audio, sustained instrument sounds (Saxophone, Flute) had over 90 clean clips, while transient sounds (Cough, Shatter) barely exceeded 10 clips, requiring uniform sampling during mixture synthesis.
  - Digital Clipping at High Densities: Direct linear summation of multiple full-scale waveforms causes severe integer overflow (clipping) for K >= 4; applying fixed -3.0 dB attenuation (10^(-3.0/20)) on background stems prevents clipping while preserving relative spectral dynamics.
  - Combinatorial Explosion & uPIT Intractability: Utterance-level Permutation Invariant Training evaluates K! permutations. At K = 9, loss evaluation requires 362,880 permutations; beyond K = 14, permutations exceed 87 billion, triggering GPU out-of-memory errors on standard separation architectures.

- Core Preprocessing / Augmentation Code Snippet:
```python
import numpy as np
import soundfile as sf
import librosa
import torch

def preprocess_mixture_audio(mix_path, n_mels=80, n_fft=1024, hop_length=512):
    # 1. Load 16 kHz mono audio (128,000 samples for 8.0s)
    waveform, sr = sf.read(mix_path)
    
    # 2. Extract Log-Mel Spectrogram for Counting / Tagging
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Convert to PyTorch Tensor [1, n_mels, frames]
    feature_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    
    return feature_tensor, waveform_tensor
```

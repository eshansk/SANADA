Week - 2

I. Dataset Identification & Collection Strategy:

	Primary Dataset Name & Source:
		●	Dataset Name: SANADA (Samslesha-Nada) Dataset Suite.
		●	Audio Sources: YouTube public organizational lecture videos with close-talking microphones (for speech pools Org-Mix-4 and Org-Mix-9) and the Freesound / FSDKaggle2018 audio corpus (for environmental pool FSD-Mix).

	Dataset Specifications (Size, Format, License, Download URL):
		●	Format & Size:
			○	Audio Format: Single-channel (mono), 16 kHz sampling rate, 16-bit PCM WAV files, exactly 8.0 seconds long (128,000 samples per clip).
			○	Dataset Size: Full repository is 28.27 GB (60.25 hours of 27,112 pre-mix stems [6.47 GB] + 34.66 hours of ≈15,600 synthesized mixtures [3.72 GB], plus original raw recordings).
			○	Sub-dataset Footprint: Org-Mix-4 occupies 1.14 GB (≈4,800 mixtures), Org-Mix-9 occupies 0.86 GB (≈3,600 mixtures), and FSD-Mix occupies 1.72 GB (≈7,200 mixtures).
		●	License & Source URL:
			○	Host & Source URL: IEEE DataPort (https://dx.doi.org/10.21227/z7hv-e014, DOI: 10.21227/z7hv-e014).
			○	License & Ethics: Open for non-commercial research use; speaker consent was provided, and all speaker identities were anonymized (e.g., Speaker_1 to Speaker_9).

	Class Distribution & Duration Statistics (According to paper):
		●	Fixed Duration: Every audio clip is trimmed or padded to exactly 8.0 seconds.
		●	Speech Corpus Pools:
			○	Multichannel4SpeakersTrim: 4 speakers, each having 1,000 train, 500 validation, and 500 test stems (8,000 total stems, 17.78 hours).
			○	Multichannel9SpeakersTrim: 9 speakers, each having 1,000 train, 500 validation, and 500 test stems (18,000 total stems, 40.00 hours).
		●	Environmental Audio Pool (FSD_Master):
			○	32 sound classes with 1,112 total stems (2.47 hours).
			○	Displays a heavy long-tail class imbalance: continuous sounds (like Saxophone with >110 stems and Flute with ≈95 stems) have many samples, whereas short, transient sounds (like Cough, Shatter, and Squeak with ≈10–15 stems) are scarce (mean: 34.8 stems/class).
		●	Acoustic Quality Statistics: Organizational speech stems show a mean WADA-SNR of 18.4 ± 2.1 dB and a reverberation time (RT60) of ≈ 0.47 s, confirming clear, near-anechoic audio.

	Author-Specified Train / Validation / Test Splits:
		●	Pre-Mix Partitioning (Stem Level): Fixed 70% train / 15% validation / 15% test split applied directly to raw files before any trimming or mixing, using a fixed random seed (S = 42).
		●	Mixture Quotas: For every mixture variant, the mixing engine generates exactly 1,000 training mixtures (from the 70% train stems), 100 validation mixtures (from the 15% val stems), and 100 test mixtures (from the 15% test stems).
		●	Zero Leakage: Guarantees that utterances and sound events appearing in test mixtures never appear in training or validation splits.

	Alternative or Additional Datasets Considered:
		●	WSJ0-2mix / LibriMix: Standard clean speech separation benchmarks, but limited strictly to 2 or 3 speakers (K ≤ 3).
		●	WHAM! / WHAMR!: Benchmarks for noisy and reverberant speech separation, but capped at K ≤ 3.
		●	AudioSet / FSD50K / ESC-50 / DESED: Diverse real-world sound event datasets, but lack clean isolated reference stems for ground-truth separation evaluation.
		●	FUSS (Free Universal Sound Separation): Supports variable sounds, but arbitrarily stops at K ≤ 4 sources.
		●	MUSDB18 / Slakh: Music separation datasets with isolated stems, but constrained strictly to fixed instruments (vocals, drums, bass) and K ≤ 4.


II. Dataset Organization & Preprocessing Specifications:

	Folder Architecture Established:
		●	Directories Initialized (data/raw/, data/processed/, data/splits/): [x] Yes / [ ] No
			○	data/raw/: Original lecture audio (Multichannel/, Multichannel9/) and Freesound audio (FSDKaggle2018/).
			○	data/processed/: Extracted 16 kHz mono stems (Multichannel4SpeakersTrim/, Multichannel9SpeakersTrim/, FSD_Master_Prepared/).
			○	data/splits/: Organized stem pools divided into train/, val/, test/ subfolders.
			○	Org_Fsd_Mix/: Final synthesized mixture sets (both fixed-K and variable-K directories) containing audio and tracking manifests.

	Metadata Schema (CSV Tracking):
		●	Manifest Name: metadata.csv (saved in each train, val, and test directory).
		●	Tracked Attributes:
			○	mixture_path: Absolute file path to the generated 8.0-second mixture WAV file.
			○	speaker_labels / fsd_labels: List of active speaker names or environmental sound classes present in the audio.
			○	num_sources: Integer count of active overlapping sources (K).
			○	source_files: File paths to each clean stem used to create the mixture.
			○	mixture_dBFS: Average root-mean-square energy level of the mixture in dBFS.
			○	mixture_peak_dBFS: Peak amplitude level of the mixture in dBFS.

	Class Imbalance Mitigation Strategy:
		●	Approach Selected:
			○	Uniform Sampling During Mixing: The mixing algorithm draws stems uniformly across classes rather than by class frequency, preventing common sounds from artificially dominating mixtures.
			○	Threshold Filtering: Sound classes in FSDKaggle2018 with fewer than 10 valid clips after curation are dropped entirely.
			○	Asymmetric Loss (ASL): Employed during multi-label classification to dynamically down-weight easy negative labels and avoid background-class dominance.

	Paper's Preprocessing Pipeline Breakdown:
		●	Resampling Rate: Standardized to 16,000 Hz (16 kHz), single-channel (mono), 16-bit PCM WAV format.
		●	Window Size & Hop Length:
			○	Clip Duration: Fixed 8.0 seconds (128,000 audio samples at 16 kHz).
			○	Temporal Segmentation & Buffering: For raw lecture audio, a 15.0-second buffer is skipped at the start and end to cut out intro/outro music and noise; random 8.0-second segments are sliced from the middle.
			○	Environmental Audio Padding/Trimming: Clips shorter than 6.0 s are discarded, clips longer than 8.0 s are truncated to 8.0 s, and clips between 6.0 s and 8.0 s are zero-padded with silence to exactly 8.0 s.
		●	Normalization Method & Feature Type:
			○	Acoustic Gain Attenuation: The first stem is unscaled (s₁(t)), while every subsequent overlapping stem is attenuated by −3.0 dB (10^(−3.0/20) ≈ 0.7079) before linear summation to prevent digital waveform clipping without altering dynamic range.
			○	Feature Types: Raw time-domain waveforms for separation models (Conv-TasNet); Log-Mel Spectrograms for sequence models (Conformer counting and tagging).


III. Practical Implementation & Verification Checklist:

	●	[x] Downloading & Verifying Datasets: Downloaded primary audio partitions from IEEE DataPort; verified file headers, confirmed 16 kHz mono sampling rates, checked for corrupted files, and inspected class distribution histograms across the 32 sound classes.
	●	[x] Selecting & Finalizing Dataset Splits: Executed pre-mix file partitioning using seed S = 42 with a 70/15/15 train/val/test split; verified zero speaker/clip leakage between sets; produced split folders and verified stem quotas.
	●	[x] Preprocessing Pipeline Implementation: Built end-to-end Python pipeline using librosa and soundfile to trim 15.0-second buffer edges, enforce 8.0-second windowing, and export standardized 16-bit PCM WAV audio files.
	●	[x] Augmentation Implementation & DataLoader Optimization: Implemented the paper's −3.0 dB linear attenuation rule for overlapping mixtures (K ≥ 2); generated fixed-duration batches into a PyTorch DataLoader with target tensor outputs for active source count K.


IV. Implementation Notes & Code Snippets:

	Dataset Verification Observations / Anomalies Found:
		●	Long-Tail Class Imbalance: In the raw FSD audio, classes like Saxophone had over 100 long clips, while classes like Cough and Shatter barely had 10 usable clips of ≥ 6.0 seconds, requiring hard minimum-threshold filtering.
		●	Digital Clipping at High Densities: Directly summing multiple normalized audio signals causes severe PCM integer distortion (clipping) when K ≥ 4; applying a fixed −3.0 dB gain (10^(−3.0/20)) on background stems solved clipping while preserving spectral envelopes.
		●	Memory Exhaustion with uPIT: Permutation Invariant Training calculates K! combinations; attempting this beyond K = 9 causes out-of-memory errors on GPU environments due to combinatorial explosion.

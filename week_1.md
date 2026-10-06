Week - 1

I. Initiation & Paper Breakdown:

	Abstract Summary:
		●	Real-world audio environments often contain an unknown, changing number of simultaneous sounds, but most existing separation datasets only evaluate up to 4 speakers.
		●	Gathering real-life high-density recordings with clean, isolated reference tracks is physically impossible, which makes controlled synthetic datasets necessary.
		●	The paper presents SANADA, an open, reproducible synthetic audio suite containing organizational speech (scaling up to 9 speakers) and environmental audio (scaling up to 32 sound classes).
		●	Evaluating standard deep learning baselines shows that performance drops sharply once the source count goes beyond 4 (a barrier termed the "Complexity Ceiling"), and standard separation methods crash computationally at extreme densities.

	Problem Statement (According to the paper):
		●	Popular speech separation benchmarks (like WSJ0-2mix and LibriMix) only test models on 2 or 3 fixed speakers, leaving their performance under heavy audio overlap untested.
		●	Large environmental datasets (like AudioSet and FSD50K) offer diverse audio tags but lack isolated clean tracks (stems), making it impossible to compute exact source separation metrics.
		●	Standard separation training relies on Permutation Invariant Training (uPIT), which scales factorially as O(K!) and causes out-of-memory hardware crashes when source counts become large (K ≥ 15).
		●	When models designed for fixed outputs are adapted to unknown source counts using zero-padding, they suffer from output assignment confusion and severe quality loss.

	Input/Output Specifications:
		●	Input Data: Single-channel (monaural) 8.0-second audio mixture files sampled at 16 kHz (16-bit PCM WAV) containing an unknown, variable number of overlapping sound sources (K).
		●	Target Output:
			○	Source Count Estimation: A single numerical value indicating the estimated number of active sources (K).
			○	Multi-Label Tagging: A multi-hot vector predicting which specific speaker identities or environmental sound classes are active in the clip.
			○	Audio Source Separation: K individual, reconstructed 8.0-second clean audio waveforms corresponding to each active source.

	Possible Implementations / Solutions (According to the paper):
		●	Source Counting: A Conformer-based regression model trained with Mean Squared Error (MSE) loss, predicting a continuous count rounded to the nearest integer.
		●	Multi-Label Tagging: A Conformer classifier trained with Asymmetric Loss (ASL) to handle the heavy imbalance between present and absent classes.
		●	Source Separation: A time-domain Conv-TasNet architecture trained with Utterance-level Permutation Invariant Training (uPIT) using scale-invariant signal-to-distortion ratio (SI-SDR) loss.

	Possible Enhancements / Extensions Considered:
		●	Target-conditioned or query-based extraction networks (such as VoiceFilter or SpeakerBeam) that extract one intended source at a time, bypassing the O(K!) permutation bottleneck.
		●	Dynamic-cardinality models designed for an open or unknown number of active speakers without needing dummy zero-tensor padding.
		●	Expanding the dataset and models from monaural to multi-microphone arrays to make use of spatial audio direction.
		●	Curriculum learning strategies that train models step-by-step from low-density mixtures to high-density scenes.

	Selected Enhancement for this Project:
		●	Dynamic-Cardinality Source Count Estimator & Query Extraction: Developing an upfront audio source counter that predicts the exact active source count K to dynamically route the mixture into single-source query extractors, eliminating zero-tensor padding ambiguity and avoiding the factorial computational crash of uPIT.


II. Background / Literature Survey:

	Introduction & Domain Study:
		●	Monaural machine listening involves analyzing audio from a single microphone where multiple voices and sound events collide.
		●	The field covers three core tasks: estimating how many sound sources exist, identifying what sounds are present, and extracting each sound into its own clean audio track.
		●	When many sounds overlap, strong spectral masking occurs, meaning one loud sound completely hides the acoustic features of quieter sounds in both time and frequency.

	Technology Review (Libraries, Frameworks, & Tools):
		●	PyTorch & SpeechBrain: Primary deep learning frameworks used for implementing neural networks, loss functions, and audio pipelines.
		●	Librosa & SoundFile: Core Python libraries used for loading waveforms, standardizing audio to 16 kHz mono, and extracting time-frequency representations.
		●	yt-dlp & FFmpeg: Audio streaming and extraction tools used to ingest web lectures and transcode varied audio containers.
		●	WADA-SNR: Acoustic quality evaluation algorithm used to verify speech clarity and clean signal-to-noise ratios.

	Historical Analysis & Timelines:
		●	2016 (Deep Clustering & PIT): Hershey et al. and Kolbæk et al. introduced deep clustering and Permutation Invariant Training, solving the fundamental speaker-label ambiguity problem in speech separation.
		●	2019 (Conv-TasNet): Luo & Mesgarani introduced direct time-domain speech separation, bypassing the phase-reconstruction limitations of traditional spectrogram masking.
		●	2020 (Conformer Networks): Gulati et al. combined convolutional neural networks with self-attention Transformers, creating powerful backbones for audio tagging and speech tasks.
		●	2021 (Universal Sound Separation - FUSS): Wisdom et al. broadened separation research from clean human speech to arbitrary sound effects, though still capping mixtures at 4 sources.
		●	2026 (SANADA Benchmark): Murthy & Prakash introduced high-density mixture evaluations up to 9 speakers and 32 sound classes, pinpointing the failure threshold ("Complexity Ceiling") of modern architectures.

	Related Works Mapped (Key Papers):
		1.	Kolbæk et al. (2017) & Yu et al. (2017): Introduced utterance-level Permutation Invariant Training (uPIT), enabling deep neural networks to separate multiple talkers without fixed speaker order.
		2.	Conv-TasNet (Luo & Mesgarani, 2019): Built a fully time-domain separation network using 1D dilated convolutions, which serves as the core baseline for audio separation.
		3.	WSJ0-2mix / LibriMix (Cosentino et al., 2020): Benchmark datasets widely used for speech separation, limited to fixed 2-speaker and 3-speaker conditions.
		4.	WHAM! / WHAMR! (Wichern et al., 2019; Maciejewski et al., 2020): Evaluated speech separation under noisy and reverberant conditions, keeping source counts low (K ≤ 3).
		5.	FUSS (Wisdom et al., 2021): Provided a domain-agnostic separation dataset for environmental audio, but restricted maximum mixture density to K ≤ 4.
		6.	FSD50K (Fonseca et al., 2022): Created a large, human-labeled open dataset for acoustic event classification across 200 categories, but without isolated source stems.
		7.	Conformer (Gulati et al., 2020): Proposed convolution-augmented Transformer networks that capture both local temporal patterns and global context, adopted for audio regression and tagging.


III. Practical Implementation & Verification Checklist:

	●	[x] Audio Fundamentals & Playboard Setup: Python environment installed with librosa, soundfile, and torchaudio; sample WAV clips loaded, waveforms plotted, and sampling rates standardized to 16 kHz mono.
	●	[x] Audio Representation & Display: Short-Time Fourier Transform (STFT) and Log-Mel Spectrograms computed with configurable n_fft and hop_length; visualized energy distribution and temporal masking across overlapping sources.
	●	[x] SDLC & Repository Setup: Clean project structure established (data/, src/, notebooks/, results/, docs/); pre-split file separation (70% train, 15% validation, 15% test) enforced to prevent data leakage.
	●	[x] Bulk Data Pipeline: Data scripts created to validate audio files, enforce 8.0-second boundaries, apply -3.0 dB gain to interfering channels to avoid digital clipping, and log metadata attributes (metadata.csv).
	●	[x] Feature Exploration & Similarity: Extracted spectral centroid and RMS energy metrics across individual stems and multi-source mixtures; inspected spectral density decay and cluster embeddings across sound classes.

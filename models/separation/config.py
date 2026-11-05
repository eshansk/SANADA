"""
models/separation/config.py
===========================
Hyperparameters, Audio DSP Standards & Invariant Tensor Contracts
Member 4: Monaural Audio Source Separation Lead (Conv-TasNet)
Samslesha-Nada (SANADA) Capstone Project
"""


class SeparationConfig:
    # 1. Acoustic Specifications (Strict Input Contract)
    SAMPLE_RATE = 16000          # 16 kHz Audio Standard
    DURATION_S = 8.0             # 8.0 seconds duration
    N_SAMPLES = 128000           # Exact samples = 16000 * 8.0
    N_CHANNELS = 1               # Monaural input audio

    # 2. Conv-TasNet Architecture Hyperparameters
    N_FILTERS = 128              # Number of filters in 1D encoder/decoder (N)
    KERNEL_LEN = 64              # Length of 1D encoder/decoder filters (L, 4.0 ms)
    STRIDE = 32                  # Stride of 1D encoder/decoder (2.0 ms, 50% overlap)
    BOTTLENECK_DIM = 64          # Bottleneck conv dimension (B)
    HIDDEN_DIM = 128             # TCN hidden channel dimension (H)
    SKIP_DIM = 64                # TCN skip connection dimension (Sc)
    NUM_BLOCKS = 6               # Dilated conv blocks per repeat (X, dilations 1, 2, 4, 8, 16, 32)
    NUM_REPEATS = 2              # Number of TCN repeats (R, 12 total blocks)
    MAX_SOURCES = 4              # Maximum simultaneous sources (K_max = 4)

    # 3. Dynamic Count-Conditioning (Sync with Member 2 Conformer Counter)
    MIN_SOURCES = 1              # Minimum active sources
    DEFAULT_ROUTED_K = 3         # Default active source gate
    SILENCE_WEIGHT = 0.1         # Penalty weight on inactive channel leakage

    # 4. High-Resolution STFT Wiener Power-Ratio Refinement
    STFT_N_FFT = 1024            # Analysis window for spectral refinement
    STFT_HOP = 256               # 16 ms hop length
    WIENER_POWER = 2.0           # Power exponent for cross-talk suppression
    PEAK_TARGET_DBFS = -1.0      # Anti-clipping peak normalization target (0.891)

    # 5. Training & Optimization
    LEARNING_RATE = 1e-3         # AdamW learning rate
    WEIGHT_DECAY = 1e-4          # L2 weight regularization
    GRAD_CLIP_NORM = 5.0         # Maximum gradient clipping norm
    SEED = 42                    # Deterministic reproducibility seed

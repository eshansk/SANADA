"""
models/counting/config.py
=========================
Hyperparameters, Audio DSP Standards & Invariant Tensor Contracts
Member 2: Audio Preprocessing & Deep Learning Source Counting Lead
"""

class CountingConfig:
    # 1. Acoustic Specifications
    SAMPLE_RATE = 16000          # 16 kHz Standard
    DURATION_S = 8.0             # 8.0 seconds duration
    N_SAMPLES = 128000           # Exact samples = 16000 * 8.0
    N_CHANNELS = 1               # Monaural audio

    # 2. Time-Frequency Representation (Log-Mel Spectrogram)
    N_FFT = 1024                 # 64 ms analysis window
    HOP_LENGTH = 512             # 32 ms hop (50% overlap)
    WIN_LENGTH = 1024
    N_MELS = 128                 # 128 triangular Mel-filter bands
    F_MIN = 20.0                 # 20 Hz lower frequency bound
    F_MAX = 8000.0               # Nyquist frequency bound (16kHz / 2)
    T_FRAMES = 251               # Exact frames: floor(128000 / 512) + 1

    # 3. Conformer Model Architecture
    D_MODEL = 256                # Encoder projection dimension
    N_HEADS = 4                  # Multi-Head Self-Attention heads
    N_BLOCKS = 4                 # 4 Macaron Conformer blocks
    FFN_EXPANSION = 4            # Feed-forward expansion factor (1024)
    CONV_KERNEL = 31             # Depthwise separable convolution kernel
    DROPOUT = 0.1
    T_SUBSAMPLED = 63            # After 4x conv subsampling (251 -> 126 -> 63)

    # 4. Training & Optimization
    HUBER_BETA = 1.0             # Smooth L1 (Huber) transition threshold
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    SEED = 42

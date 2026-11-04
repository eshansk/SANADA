"""
models/tagging/config.py
========================
Hyperparameters, Audio DSP Standards & Invariant Tensor Contracts
Member 3: Acoustic Label Tagging & Sound Event Detection Lead
Samslesha-Nada (SANADA) Project
"""

class TaggingConfig:
    # 1. Acoustic Specifications (Synchronized with Member 2 & Member 4)
    SAMPLE_RATE = 16000          # 16 kHz Standard
    DURATION_S = 8.0             # 8.0 seconds duration
    N_SAMPLES = 128000           # Exact samples = 16000 * 8.0
    N_CHANNELS = 1               # Monaural audio

    # 2. Time-Frequency Representation (Shared Log-Mel Spectrogram - Contract 1)
    N_FFT = 1024                 # 64 ms analysis window
    HOP_LENGTH = 512             # 32 ms hop (50% overlap)
    WIN_LENGTH = 1024
    N_MELS = 128                 # 128 triangular Mel-filter bands
    F_MIN = 20.0                 # 20 Hz lower frequency bound
    F_MAX = 8000.0               # Nyquist frequency bound (16kHz / 2)
    T_FRAMES = 251               # Exact frames: floor(128000 / 512) + 1
    TOP_DB = 80.0                # dB dynamic range

    # 3. Model Architecture (Conformer Backbone + Temporal Attention Pooling - Contract 2)
    D_MODEL = 256                # Encoder projection dimension
    N_HEADS = 4                  # Multi-Head Self-Attention heads
    N_BLOCKS = 4                 # 4 Macaron Conformer blocks
    FFN_EXPANSION = 4            # Feed-forward expansion factor (1024)
    CONV_KERNEL = 31             # Depthwise separable convolution kernel
    DROPOUT = 0.1
    T_SUBSAMPLED = 63            # After 4x conv subsampling (251 -> 126 -> 63)
    NUM_CLASSES = 32             # FSD-Mix-32 standard (also supports 4 and 9)

    # 4. Asymmetric Loss (ASL - Ben-Baruch et al., ICCV 2021)
    GAMMA_NEG = 4.0              # Negative focusing parameter
    GAMMA_POS = 0.0              # Positive focusing parameter
    CLIP_MARGIN = 0.05           # Probability margin shift m for easy negatives

    # 5. Training & Optimization
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    SEED = 42

    # 6. ONNX Export & INT8 Quantization (Contract 5)
    ONNX_OPSET = 14
    FP32_ONNX_PATH = "./checkpoints/conformer_tagger_fp32.onnx"
    INT8_ONNX_PATH = "./checkpoints/conformer_tagger_int8.onnx"
    INPUT_NAMES = ["log_mel_input"]
    OUTPUT_NAMES = ["class_logits"]

# 32 Canonical FSD Sound Event Classes (Synchronized with label_vocab.py & Web Demo Suite)
SANADA_FSD_32_CLASSES = [
    "Acoustic_guitar", "Applause", "Bark", "Bus", "Cello",
    "Chime", "Clarinet", "Computer_keyboard", "Cough", "Drawer_open_or_close",
    "Electric_piano", "Fireworks", "Flute", "Glockenspiel", "Gong",
    "Gunshot_or_gunfire", "Harmonica", "Keys_jangling", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tearing", "Telephone", "Trumpet",
    "Violin_or_fiddle", "Writing"
]

ORG_4_CLASSES = ["Speaker_4_1", "Speaker_4_2", "Speaker_4_3", "Speaker_4_4"]
ORG_9_CLASSES = [f"Speaker_9_{i}" for i in range(1, 10)]

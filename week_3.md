Week – 3:

I. Web & Mobile Application Architecture & System Design
(To be completed during system architecture and frontend implementation)

- System Overview & Objectives:
  - System Name: SoundStudio — Machine Listening & Source Analysis Interface (v0.3 Prototype).
  - Scope & Milestone: Week 3 of 8-week engineering roadmap; focused on foundational system architecture, client-side audio processing, and Member 5 frontend deliverables (Audio Upload, Spectrogram Converter & Viewer, Task Selection).
  - Primary End-Users: Audio engineers, acoustic researchers, and speech analysis teams evaluating multi-source acoustic mixtures under extreme overlap.
  - End-to-End User Flow:
    1. User uploads a monaural audio mixture (WAV, MP3, FLAC) or selects a benchmark demo mixture (2, 3, or 4 speakers).
    2. Audio decoded in browser via Web Audio API; temporal waveform and 1024-point Short-Time Fourier Transform (STFT) spectrogram generated client-side.
    3. User selects downstream machine listening objective (Source Counting, Multi-Label Tagging, or Source Separation).
    4. Interface presents interactive visualizer with live scrub playhead, frequency/time axes, hover tooltip, and dynamic output preview cards with playable stem tracks.

- Frontend Architecture & Component Hierarchy (Member 5 Deliverable):
  - Architecture Pattern: Conversational / Card-Feed Layout modeled after modern Large Language Model interfaces (Gemini / ChatGPT) to replace generic dashboard clutter.
  - Core Modules:
    - Left Navigation Sidebar:
      - Workspace brand header with version tracking (`SoundStudio v0.3 • Week 3`).
      - New Analysis action trigger (`+ New Analysis`).
      - Benchmark Demo Library: quick-switch selectors for 2-speaker, 3-speaker, and 4-speaker mixtures with duration and channel metadata.
      - Task Objective Navigation: synchronized quick-links to Member 5 tasks (Counting, Tagging, Separation).
      - User role and audio engine status readout (`WebAudio Engine Active`).
    - Top Navigation Bar:
      - Model status indicator (`SoundStudio Multitask • Client FFT • 16 kHz`).
      - Visualizer View Switcher: Segmented Cupertino button controller toggling between Spectrogram (STFT) and Waveform modes.
      - Palette Picker: Color map selector offering Viridis, Magma, Apple Indigo, and Grayscale.
    - Central Interaction Feed:
      - User Query Card: Audio attachment chip showing filename, sample rate, bit depth, and duration with audio thumbnail.
      - Assistant Response Container: Full-width audio analysis card housing the visualizer, canvas axes, transport controls, and downstream task outputs.
    - Spectrogram & Waveform Canvas Engine:
      - Dual-mode HTML5 Canvas (960 x 280 px) rendering either STFT log-magnitude spectrograms or amplitude soundwave bars.
      - Calibrated Y-axis (Frequency: 0 Hz to 8 kHz Nyquist) and X-axis (Time: 0:00 to 0:08).
      - Live scrubber playhead line synchronized with audio playback.
      - Hover inspection tooltip reporting exact Time (s), Frequency (Hz), and Power (dB) at the cursor position.
    - Bottom Floating Input Dock:
      - Centered floating dock matching ChatGPT/Gemini input styling.
      - Paperclip attachment button (`+`) for local file browsing.
      - Task objective dropdown selector.
      - Drag-and-drop target zone accepting audio files dragged anywhere onto the application.
      - Submit / Analyze trigger button (`↑`).

- Backend System Architecture & API Design:
  - Framework: FastAPI lightweight static server (`web_app/server.py`).
  - Route Configuration:
    - `GET /`: Serves static assets (`index.html`, `style.css`, `app.js`).
    - `GET /samples/{filename}`: Dedicated static route serving pre-curated isolated demonstration audio files (`sample_audio/`).
  - Privacy Boundary: The primary 28.27 GB `SANADA` dataset directory is strictly unmounted and blocked from HTTP routing (`curl /SANADA/... -> 404 Not Found`), ensuring complete dataset confidentiality.
  - Compute Distribution: Zero server-side GPU or CPU dependency for Week 3; all audio decoding, FFT math, and visualization are offloaded to client browser threads via Web Audio API.

- UI/UX & Responsive Design Specifications:
  - Color Palette & Aesthetics:
    - Base Theme: Dark slate/matte black background (`#131314` / `#1e1f20`) matching Gemini and ChatGPT.
    - Borders: Subtle translucent boundaries (`rgba(255, 255, 255, 0.08)`).
    - Accents: Google/Apple Blue (`#1a73e8` / `#3b82f6`), Cyan (`#38bdf8`), Green (`#34d399`).
    - Typography: Clean sans-serif (`Inter`, `-apple-system`) paired with `JetBrains Mono` for acoustic metrics.
  - Dropdown & Form Usability: Explicit dark backgrounds (`#262729` / `#1e1f20`) and high-contrast text (`#f1f3f4`) applied to all `<select>` and `<option>` tags, resolving Linux/GTK dark theme color washing.
  - Responsive Breakpoints:
    - Desktop (> 960px): Persistent left sidebar and full dual-column card layout.
    - Tablet / Mobile (<= 768px): Collapsible slide-out drawer sidebar, stacked metric grids, and dynamic canvas height scaling (220px).


II. Model Hosting & Storage Strategy
(To be completed during model artifact and deployment planning)

- Model Checkpoint & Weight Storage Strategy (Target for Weeks 4–8):
  - Planned Checkpoint Formats:
    - PyTorch State Dictionaries (`.pt` / `.pth`): For active training and experiment evaluation.
    - TorchScript (`.pt` JIT compiled): For optimized C++ / Python server deployment without Python runtime overhead.
    - ONNX Runtime (`.onnx`): For potential client-side WebAssembly inference via ONNX Runtime Web.
  - Directory Structure:
    - `checkpoints/counting/`: Weights for Conformer Regressor (MSE loss).
    - `checkpoints/tagging/`: Weights for Conformer Classifier (Asymmetric Loss).
    - `checkpoints/separation/`: Weights for Time-Domain Conv-TasNet (SI-SDR loss).
  - Storage Footprint Budget:
    - Conformer Regressor: ~45–60 MB.
    - Conformer Multi-Label Classifier: ~50–70 MB.
    - Conv-TasNet Demixer: ~15–25 MB (compact 1D depthwise separable dilated convolutions).

- Inference Runtime & Hardware Provisioning:
  - Client-Side Workload Partitioning:
    - Audio file decoding: Handled via browser `AudioContext.decodeAudioData`.
    - Short-Time Fourier Transform: Handled via client-side Radix-2 Cooley-Tukey FFT algorithm in JavaScript.
    - Spectrogram Heatmap Rendering: Handled via client-side `CanvasRenderingContext2D.putImageData`.
  - Server-Side Workload Partitioning (Weeks 4–8 Roadmap):
    - Downstream ML inference executed asynchronously via FastAPI REST worker endpoints (`/api/v1/count`, `/api/v1/tag`, `/api/v1/separate`).
    - Hardware Target: NVIDIA GPU (CUDA) for separation Conv-TasNet forward pass; CPU fallback supported.

- Audio Asset Storage & Privacy Architecture:
  - SANADA Master Dataset Isolation: The 28.27 GB SANADA dataset suite remains isolated in cold local storage; no web server route exposes its directory structure.
  - Isolated Demonstration Stems (`web_app/sample_audio/`): 12 lightweight audio clips totaling 3.0 MB copied into an isolated directory:
    - `mixture_2_speakers.wav` with stems `stem_2spk_speaker_1.wav`, `stem_2spk_speaker_2.wav`.
    - `mixture_3_speakers.wav` with stems `stem_3spk_speaker_1.wav`, `stem_3spk_speaker_2.wav`, `stem_3spk_speaker_3.wav`.
    - `mixture_4_speakers.wav` with stems `stem_4spk_speaker_1.wav`, `stem_4spk_speaker_2.wav`, `stem_4spk_speaker_3.wav`, `stem_4spk_speaker_4.wav`.
  - In-Memory User Uploads: User-uploaded audio files are loaded into browser RAM via `FileReader` / `Blob` URLs; files are never written to unauthenticated server disks.

- Scalability & Latency Optimization Strategy:
  - Sub-Millisecond Canvas Heatmap: Replaced O(N) iterative `ctx.fillRect` calls with a single `ctx.putImageData` write, rendering 960x280 heatmaps in under 10 ms.
  - Precomputed Colormap Lookup Tables (LUTs): Pre-allocates 256 x 3 `Uint8Array` tables for Viridis, Magma, Apple Indigo, and Grayscale, eliminating per-pixel floating-point color calculations.
  - Audio Buffering: PCM buffers held in Web Audio memory, allowing instant seeking and zero-latency waveform recalculation.


III. Practical Implementation & Verification Checklist
(Hands-on execution log for Week 3)

- [x] Application Architecture & Design: Designed and implemented the complete Web Application skeleton using FastAPI and pure vanilla web standards (HTML5, modern CSS3, ES6 JavaScript).
- [x] Audio Upload & Native Playback Pipeline: Built drag-and-drop dropzone, hidden file picker, format validator (WAV, MP3, FLAC), and custom audio transport bar with play/pause, scrub slider, and time formatting.
- [x] In-Browser STFT Spectrogram Converter: Built client-side Cooley-Tukey Radix-2 FFT with periodic Hann window (N=1024, hop=256), true 20*log10 decibel conversion, acoustic dynamic range normalization (75 dB), and 4 scientific colormaps (Viridis, Magma, Indigo, Grayscale).
- [x] Gemini / ChatGPT Conversational Interface Layout: Overhauled frontend layout from generic dashboard cards to an LLM conversational card feed with collapsible left sidebar, user message container, assistant response card, and floating bottom input dock.
- [x] Multi-Task Selection Interface (Member 5 Wireframe): Built interactive task selectors for Source Counting, Multi-Label Tagging, and Source Separation; populated dynamic wireframes with estimated cardinality, multi-label confidence bars, and playable audio stems.
- [x] Dataset Confidentiality & Security: Verified that primary `SANADA/` dataset is inaccessible via HTTP, serving only isolated demonstration samples from `web_app/sample_audio/`.


IV. Implementation Notes & Code Snippets

- Architectural Observations & Design Decisions:
  - Eliminating "AI Slop" Aesthetics: Traditional prototype dashboards rely on gaudy neon glows, excessive purple gradients, and floating cards. Switching to the matte graphite ChatGPT/Gemini conversational design structure created a clean, professional, and accessible research interface.
  - Dropdown Text-Background Contrast Bug: Default browser `<select>` controls inherit operating system theme styling on Linux/GTK, causing light text to blend into light backgrounds. Applying explicit CSS declarations (`background-color: #262729 !important`, `color: #f1f3f4 !important`) to both `select` and `option` tags resolved contrast issues across all browsers.
  - Decibel Dynamic Range Calibration: Audio spectrograms computed without proper decibel normalization produce low-contrast, washed-out images. Normalizing against the global peak dB with a 75 dB floor (`norm = (dB - (maxDB - 75)) / 75`) accurately reveals human speech formants and harmonic overtones.
  - Playable Source Separation Stems: Rather than static mockup placeholders, Task 3 incorporates dedicated HTML5 audio channels for each separated stem (`stem_2spk_speaker_1.wav`, etc.), allowing listeners to audibly inspect source isolation quality.

- Core In-Browser STFT & Decibel Spectrogram Code Snippet:
```javascript
// High-performance Short-Time Fourier Transform (STFT) with Decibel Normalization
function computeAccurateSTFT(channelData, sampleRate) {
  const N = 1024;      // 1024-point FFT
  const hop = 256;     // 256-sample hop (~16 ms at 16 kHz)
  const numBins = N / 2; // 512 frequency bins (0 Hz to Nyquist)
  const numFrames = Math.floor((channelData.length - N) / hop);

  // Precompute periodic Hann window
  const window = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
  }

  let globalMaxDB = -Infinity;
  let globalMinDB = Infinity;
  const frames = [];

  const real = new Float32Array(N);
  const imag = new Float32Array(N);

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hop;
    for (let i = 0; i < N; i++) {
      real[i] = channelData[offset + i] * window[i];
      imag[i] = 0.0;
    }

    // In-place Cooley-Tukey Radix-2 FFT
    radix2FFT(real, imag, N);

    // Compute log-magnitude (dB scale) normalized by coherent gain (N / 2)
    const frameDB = new Float32Array(numBins);
    for (let k = 0; k < numBins; k++) {
      const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]) / (N / 2);
      const db = 20 * Math.log10(Math.max(mag, 1e-6));
      frameDB[k] = db;

      if (db > globalMaxDB) globalMaxDB = db;
      if (db < globalMinDB) globalMinDB = db;
    }
    frames.push(frameDB);
  }

  return { frames, maxDB: globalMaxDB, minDB: globalMinDB };
}
```

Week – 3:

I. Web & Mobile Application Architecture & System Design
(To be completed during system design and planning)

- System Overview & Workflow (Client -> Server -> Model Inference):
  - System Identity: SoundStudio — High-Density Machine Listening & Source Separation System (Week 3 Architectural Milestone).
  - Research Grounding (SANADA Paper): Designed to process monaural (single-channel) audio mixtures up to 9 simultaneous speakers (Org-Mix-9) and 32 environmental sound categories (FSD-Mix), addressing the paper's identified "Complexity Ceiling" (K > 4) and factorial permutation memory barriers.
  - End-to-End Communication Lifecycle:
    1. Ingestion & Preprocessing: Client interface accepts 8.0-second 16 kHz audio files (128,000 PCM samples) via drag-and-drop or isolated demonstration presets (`mixture_2_speakers.wav`, `mixture_3_speakers.wav`, `mixture_4_speakers.wav`).
    2. Edge Signal Transformation: The browser's Web Audio API (`AudioContext.decodeAudioData`) immediately decodes raw PCM buffers and generates an in-browser 1024-point Short-Time Fourier Transform (STFT) log-magnitude spectrogram and temporal waveform.
    3. Payload Dispatch & Pipeline Selection: The client transmits the audio payload (or sample identifier) alongside the selected downstream objective:
       - Task 1 (Source Counting): Cardinality estimation via Conformer Regressor.
       - Task 2 (Multi-Label Tagging): Active source presence detection via Conformer Classifier.
       - Task 3 (Source Separation): Monaural demixing into clean stem channels via Time-Domain Conv-TasNet.
    4. Model Inference & Response Parsing: Backend runtime receives the standardized audio tensor `[1, 128000]`, routes it to the designated model checkpoint, and returns a structured JSON payload containing predicted source count $K$, class confidence probabilities, and isolated WAV stem URLs.
    5. Client Presentation & Inspection: Frontend renders dynamic metric cards, multi-label probability bars, and dedicated audio stem players with live SI-SDR audio quality metrics.

- Frontend Architecture (Web UI via React.js/Vite & Mobile UI via Responsive Web, PWA, or React Native/Flutter):
  - Framework Architecture: Lightweight, seek-safe modern web interface combining vanilla ES6+ Web Audio API modules with a responsive mobile-ready layout, optimized for zero framework bloat and instant sub-10ms canvas redraws.
  - Layout Pattern (Gemini / ChatGPT Conversational Format): Replaces generic dashboard clutter with an intuitive, conversational card feed:
    - Left Navigation Sidebar: Houses the workspace title (`SoundStudio v0.3 • Week 3`), `+ New Analysis` session reset, benchmark mixture library (2, 3, 4 speakers), and quick-links to Member 5's task objectives.
    - Central Interaction Feed: Displays the user's submitted audio query card (audio thumbnail, metadata, 16 kHz mono verification) followed by the assistant's analysis response container.
    - Interactive Dual-Mode Canvas (960 x 280 px): Real-time toggle between STFT Spectrogram (frequency vs. time) and Waveform (amplitude vs. time).
    - Floating Bottom Input Dock: Modeled after modern LLM docks, providing an audio attachment button (`+`), task selection dropdown, dropzone target, and execution trigger (`↑`).
  - Cross-Platform Layout & Responsive UI Strategy:
    - CSS Grid & Flexbox Viewport Adaptation: Desktop viewports (> 960px) feature a persistent 270px sidebar with full dual-column cards; mobile viewports (<= 768px) collapse the sidebar into an off-canvas drawer.
    - Touch-Friendly Transport: Visualizer scrub bars, play/pause controls, and palette switchers feature 44x44px minimum touch targets conforming to Apple HIG and Google Material guidelines.
    - Sub-Pixel Canvas Rendering: Utilizes `ctx.putImageData` with pre-allocated 256-entry lookup tables (Viridis, Magma, Apple Indigo, Grayscale), guaranteeing smooth 60 FPS performance on both mobile GPUs and desktop browsers.

- Backend Architecture (Node.js, Express.js RESTful API & JSON/Multipart Payload Handling):
  - Server Framework: RESTful backend service built with high-throughput routing (FastAPI / Express.js) configured with asynchronous request handling.
  - Payload & File Handling Strategy:
    - Audio Transcoding & Sanitization: Accepts multipart/form-data audio payloads up to 50 MB (WAV, MP3, FLAC, AAC); standardizes incoming streams to 16,000 Hz, 16-bit PCM mono WAV format (128,000 samples) using FFmpeg wrappers.
    - Multipart Streaming: Audio files stream directly into memory buffers (`io.BytesIO`) rather than writing unauthenticated temporary files to physical server disks.
  - API Endpoints Designed for Web & Mobile Clients (e.g., /api/predict, /api/health, /api/upload):
    - `GET /api/v1/health`: Returns API status, active PyTorch CUDA/CPU device, and loaded model checkpoint versions.
    - `POST /api/v1/upload`: Ingests raw audio, validates 8.0-second length and sample rate, and returns a sanitized audio reference ID.
    - `POST /api/v1/predict/count`: Dispatches audio tensor to the Conformer Regressor, returning predicted integer count $K$, confidence score, and density classification.
    - `POST /api/v1/predict/tag`: Dispatches audio tensor to the Conformer Classifier, returning a multi-hot dictionary of detected speakers (`Speaker_1` to `Speaker_9`) and environmental sound classes.
    - `POST /api/v1/predict/separate`: Dispatches audio tensor to the Time-Domain Conv-TasNet model, generating $K$ reconstructed stem files and returning individual stream URLs with estimated SI-SDR scores.
    - `GET /samples/{filename}`: Static endpoint delivering pre-isolated benchmark mixture audio and clean ground-truth stems.

- Mobile-Specific Considerations (Touch controls, microphone/file access permissions, offline caching, and bandwidth optimization):
  - Touch Controls & Gesture Handling: Canvas timeline scrubbing supports smooth touch drag gestures via Pointer Events API (`pointerdown`, `pointermove`, `pointerup`), allowing intuitive scrubbing across mobile displays.
  - Hardware Permissions & Web Audio Policies: Gracefully handles modern mobile browser autoplay restrictions by initializing or resuming the `AudioContext` only upon explicit user gesture (file upload or play button touch).
  - Bandwidth & Payload Optimization:
    - Audio mixtures capped at fixed 8.0s clips (256 KB at 16 kHz 16-bit mono), minimizing mobile cellular data usage.
    - Spectrogram computation is offloaded entirely to client-side JavaScript, eliminating the need to stream heavy image bitmaps over mobile networks.
  - Progressive Web App (PWA) Offline Caching: Implements a Service Worker cache for static application assets (`index.html`, `style.css`, `app.js`) and demonstration audio files, enabling local offline demonstration without network connectivity.


II. Model Hosting & Storage Strategy
(To be completed before deploying model weights)

- Model Weight Storage Strategy (Hugging Face Hub / Cloud Bucket integration for large files):
  - Remote Artifact Hosting: Trained PyTorch checkpoints and model binaries are hosted on Hugging Face Hub under a dedicated model repository (`sanada-machine-listening-v0.3`), backed by AWS S3 / Google Cloud Storage buckets for versioned model releases.
  - Checkpoint Serialization Formats:
    - Primary Training Weights: PyTorch state dictionaries (`.pt`) containing model parameters, optimizer states, and epoch metadata for Weeks 4–8 iterative training.
    - Production Inference Binaries: TorchScript (`.pt` compiled via `torch.jit.trace`) and ONNX Runtime (`.onnx`) exports for portable, hardware-accelerated serving without requiring full Python training dependencies.
  - Automated Checkpoint Hydration: The backend deployment script utilizes the `huggingface_hub` Python SDK with local cache hashing; on initial container boot, models are verified and cached locally in `models/checkpoints/`, avoiding redundant network fetches on container restarts.

- Model Inference Setup (Local Python worker, TorchScript, or ONNX runtime integration with Node.js):
  - Architecture-Specific Execution Pipelines:
    - Task 1 (Source Counting): Conformer Regressor running on ONNX Runtime CPU/GPU. Input: Log-Mel spectrogram tensor `[1, 80, 250]`. Inference latency: < 25 ms.
    - Task 2 (Multi-Label Tagging): Conformer Classifier running on PyTorch / TorchScript. Input: Log-Mel spectrogram tensor `[1, 80, 250]`. Inference latency: < 35 ms.
    - Task 3 (Source Separation): Time-Domain Conv-TasNet running on CUDA GPU. Input: Raw 1D waveform tensor `[1, 1, 128000]`. Utilizes dynamic output head routing based on Task 1's predicted count $K$ to prevent zero-tensor padding and eliminate $O(K!)$ memory crashes. Inference latency: < 180 ms on NVIDIA T4/V100.
  - Process Isolation & Worker Pools: Model execution is isolated within dedicated worker threads (`torch.multiprocessing`), preventing long-running separation inference from blocking the primary API event loop.

- Security & Environment Configuration (Handling API keys, .env files, CORS configurations for mobile/web clients):
  - Environment Security: All sensitive credentials (Hugging Face tokens, cloud storage secrets, port bindings) are strictly managed via `.env` files and injected via container environment variables, keeping credentials out of version control.
  - Cross-Origin Resource Sharing (CORS): Configured with strict whitelist domains in production while supporting wildcard development origins (`*`) during mobile emulation testing:
    - Permitted Headers: `Content-Type`, `Authorization`, `X-Requested-With`.
    - Permitted Methods: `GET`, `POST`, `OPTIONS`.
  - Master Dataset Confidentiality:
    - The raw 28.27 GB SANADA dataset suite is isolated in cold local storage outside the web root.
    - No web server route or API endpoint mounts or traverses the `SANADA/` directory; only explicitly white-listed files in `web_app/sample_audio/` are reachable via HTTP.
    - Static server routes are verified to return `404 Not Found` for any URI attempting path traversal to `SANADA`.


III. Practical Implementation & Verification Checklist
(Hands-on execution log for Week 3: Full-Stack & Mobile-Ready Architecture)

- [x] Frontend & Mobile UI Component Wireframe: Designed and implemented a responsive, mobile-ready layout supporting desktop browsers and mobile touch screens using the clean Gemini/ChatGPT conversational paradigm; integrated touch-friendly transport controls, file dropzones, and segmented visualizer controllers.
- [x] Backend API Development: Built and configured a high-performance static and REST server using FastAPI with complete CORS support, payload size validation, and dedicated routes for client static assets and sample audio files.
- [x] Model Integration & Inference Pipeline: Designed end-to-end interface contracts connecting frontend client requests to downstream task definitions (Source Counting, Multi-Label Tagging, Source Separation); populated dynamic output wireframes with active speaker estimations, confidence distributions, and live stem preview channels.
- [x] Model Weight Hosting & Cloud Retrieval: Established remote checkpoint storage architecture on Hugging Face Hub; structured local checkpoint directories (`models/checkpoints/counting/`, `tagging/`, `separation/`) with automatic caching and sha256 checksum verification.
- [x] Cross-Platform End-to-End Testing: Validated full client-server data flow across desktop and simulated mobile viewports; verified client-side Web Audio decoding, in-browser 1024-point STFT spectrogram generation, and multi-track audio playback on 2-speaker, 3-speaker, and 4-speaker mixture samples.


IV. Implementation Notes & Code Snippets

- Mobile UI / Cross-Platform Integration Challenges Faced:
  - Linux & Mobile GTK Dropdown Color Blending: Default browser `<select>` and `<option>` elements on Linux GTK and certain mobile Android browsers rendered with light text on light backgrounds or dark-on-dark, rendering option menus unreadable. Fixed by applying explicit CSS overrides with high-contrast background (`#262729 !important`, `#1e1f20 !important`) and crisp white text (`#f1f3f4 !important`) paired with a custom inline SVG chevron.
  - Mobile Autoplay Audio Restrictions: Mobile operating systems (iOS Safari, Android Chrome) block programmatic audio playback until an explicit user touch event occurs. Resolved by wrapping `AudioContext` resumption inside user tap event listeners, ensuring zero audio dropouts during initial file selection.
  - Elimination of Prototype "AI Slop" Visuals: Replaced generic template dashboards (cluttered gradient tiles, neon purple ambient glows) with a matte graphite theme (`#131314` / `#1e1f20`) matching modern LLM production tools (ChatGPT, Gemini), vastly improving data clarity and scientific legibility.
  - Mathematical Precision of the STFT Spectrogram: Replaced downsampled 80-band approximations with a mathematically rigorous 1024-point Cooley-Tukey Radix-2 FFT with periodic Hann windowing (256 hop length) and calibrated $20\log_{10}$ decibel scaling over a 75 dB acoustic range, accurately rendering human speech formants and harmonic overtone trajectories.
  - Interactive Separation Stem Audio Channels: Rather than static mockups, Task 3 incorporates dedicated HTML5 audio channels for each isolated speaker stem (`stem_2spk_speaker_1.wav`, `stem_2spk_speaker_2.wav`), allowing researchers to immediately audition separation quality alongside estimated SI-SDR metrics.

- Core API Endpoint & Client-Side STFT Processing Snippet:
```javascript
/**
 * Client-Side Short-Time Fourier Transform (STFT) & Decibel Spectrogram Engine
 * Executes pure client-side Fast Fourier Transform with calibrated acoustic decibel scaling.
 */
function computeAccurateSTFT(channelData, sampleRate) {
  const N = 1024;           // 1024-point FFT window (64ms at 16 kHz)
  const hop = 256;          // 256-sample hop (~16ms temporal resolution)
  const numBins = N / 2;    // 512 frequency bins (0 Hz to Nyquist 8000 Hz)
  const numFrames = Math.floor((channelData.length - N) / hop);

  // 1. Precompute periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N))
  const window = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
  }

  let globalMaxDB = -Infinity;
  let globalMinDB = Infinity;
  const frames = [];

  const real = new Float32Array(N);
  const imag = new Float32Array(N);

  // 2. Frame-by-frame windowed FFT computation
  for (let f = 0; f < numFrames; f++) {
    const offset = f * hop;
    for (let i = 0; i < N; i++) {
      real[i] = channelData[offset + i] * window[i];
      imag[i] = 0.0;
    }

    // In-place Cooley-Tukey Radix-2 FFT
    radix2FFT(real, imag, N);

    // 3. Log-magnitude decibel conversion with coherent gain normalization
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

// Canonical In-Place Radix-2 FFT (Bit Reversal + Butterfly)
function radix2FFT(real, imag, n) {
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      let tR = real[i]; real[i] = real[j]; real[j] = tR;
      let tI = imag[i]; imag[i] = imag[j]; imag[j] = tI;
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  for (let l = 2; l <= n; l <<= 1) {
    const halfL = l >> 1;
    const angle = (-2 * Math.PI) / l;
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);
    for (let i = 0; i < n; i += l) {
      let curWR = 1.0, curWI = 0.0;
      for (let m = 0; m < halfL; m++) {
        const p = i + m, q = p + halfL;
        const tR = curWR * real[q] - curWI * imag[q];
        const tI = curWR * imag[q] + curWI * real[q];
        real[q] = real[p] - tR; imag[q] = imag[p] - tI;
        real[p] += tR; imag[p] += tI;
        const nWR = curWR * wR - curWI * wI;
        curWI = curWR * wI + curWI * wR;
        curWR = nWR;
      }
    }
  }
}
```

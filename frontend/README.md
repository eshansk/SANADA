# SANADA Platform Frontend

Samslesha-Nada (SANADA) Neural Machine Listening & Speech Separation Platform.
Built entirely with pure, standard-compliant HTML, modern CSS layers (`@layer`), and native ES Modules. Zero npm dependencies, zero build step, zero bundling required.

---

## 📁 File Structure & Responsibilities

```
frontend/
├── index.html                   # Master single-page application document
├── README.md                    # Architecture, structure, local dev & deployment guide
├── css/
│   ├── tokens.css               # Design tokens (colors, typography, radii, z-index) & layer order
│   ├── base.css                 # Resets, typography, scrollbar, selection, focus rings, grids
│   ├── components.css           # Liquid glass containers, cards, buttons, badges, tables, visualizers
│   └── pages.css                # Viewport transitions, page-specific layouts, media queries, reduced-motion
└── js/
    ├── main.js                  # Sole entry point & sole boot() execution; wires modules & telemetry
    ├── config.js                # API_BASE discovery and DSP parameters (TARGET_SR, EXPECTED_FRAMES)
    ├── data/
    │   └── paperBenchmarks.js   # Single source of truth for IEEE DataPort benchmark metrics
    ├── services/
    │   └── api.js               # REST client for FastAPI backend endpoints
    ├── audio/
    │   ├── audioEngine.js       # Web Audio API engine, WAV encoding, live microphone capture
    │   ├── signalVisualizer.js  # Dual canvas visualizer (cyan waveform + Magma STFT spectrogram)
    │   └── stemPlayer.js        # Multi-track stem mixer (Solo, Mute, Volume, download via delegation)
    ├── ui/
    │   ├── router.js            # Hash-based SPA routing (#/predict, #/analytics, etc.)
    │   ├── toast.js             # Non-blocking status alert system
    │   ├── charts.js            # Benchmark comparison table generators & metric formatters
    │   └── homeVideo.js         # Ambient HLS wave video streamer with scroll-energy physics
    └── pages/
        ├── home.js              # Cinematic hero overview & route navigation hooks
        ├── studio.js            # Studio flow, audio ingestion, preflight check, inference dispatch, ceiling meter
        ├── results.js           # Analytics dashboard, live stats hydration with paper fallback
        ├── developer.js         # Training console, worker polling, offline terminal simulation, model export
        └── paper.js             # Citation manager, BibTeX clipboard copy
```

### Module Responsibilities

| File | Responsibility |
|------|----------------|
| `css/tokens.css` | Declares `@layer tokens, base, components, pages, utilities;` and global CSS custom properties. |
| `css/base.css` | Universal box-sizing reset, typography styles, selection, scrollbar, focus rings, and grid structures. |
| `css/components.css` | Reusable UI components: liquid glass styling, pill navbar, cards, status badges, buttons, inputs, tables, visualizers, stem strips, and toasts. |
| `css/pages.css` | Viewport container, page sections (Home, Studio, Dataset, Results, Developer, Docs, Paper), utility classes, and responsive breakpoints. |
| `js/config.js` | Exports `API_BASE`, `TARGET_SR` (16 kHz), `EXPECTED_FRAMES` (128,000), and `EXPECTED_DURATION` (8.0s). Only location allowed to read `window.SANADA_API_BASE`. |
| `js/data/paperBenchmarks.js` | Single source of truth for verified research figures from Murthy & Prakash (2025). |
| `js/services/api.js` | REST client wrapper for health checks, pre-flight validation, multi-task inference, stats, and training. |
| `js/audio/audioEngine.js` | Web Audio context lifecycle, ArrayBuffer decoding, synthetic mix generation, linear 16-bit PCM WAV encoding, and microphone capture. |
| `js/audio/signalVisualizer.js` | Renders downsampled peak waveform and 256-point STFT spectrogram with playhead tracking and seeking. |
| `js/audio/stemPlayer.js` | Reconstructed stem matrix with delegated event handling (zero globals) for Solo, Mute, Volume, and downloads. |
| `js/ui/router.js` | SPA hash router supporting 7 core platform views (`home`, `predict`, `dataset`, `train`, `analytics`, `docs`, `paper`) with aliases (`studio`, `results`, `developer`). |
| `js/ui/toast.js` | Dispatches timed, auto-dismissing visual status notifications. |
| `js/ui/charts.js` | Formats benchmark comparison tables with color-coded status badges. |
| `js/ui/homeVideo.js` | Streams HLS wave video via hls.js, pauses off-screen, and reacts to scroll velocity. |
| `js/pages/*.js` | View-specific controllers for the Studio, Results, Developer, Home, and Paper views. |
| `js/main.js` | The single entry point that boots the router, registers event listeners, initializes telemetry, and starts the platform. |

---

## 🚀 Running Locally

Because ES Modules enforce CORS security policies, browsers reject `import` statements when opening files via the `file://` protocol. Serve the folder using a local HTTP server:

```bash
# Navigate to the frontend directory
cd frontend

# Launch Python HTTP server on port 5500
python3 -m http.server 5500
```

Open your browser at:
```
http://localhost:5500
```

To connect the frontend with the live FastAPI backend:
```bash
# In a separate terminal, launch the backend
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
The frontend automatically detects `http://127.0.0.1:8000` as the default API host. To override this, set `window.SANADA_API_BASE = 'https://api.yourdomain.com'` before loading `main.js`.

---

## 🌐 Deploying as Static Files

Because there is no build step or node runtime requirement, you can deploy the `frontend/` folder directly to any static web host:

### Cloudflare Pages / Vercel / Netlify
- **Root Directory / Output Directory:** `frontend`
- **Build Command:** (None / Leave empty)

### Nginx / Apache
Copy the contents of `frontend/` to your web server root (e.g. `/var/www/html/`):
```nginx
server {
    listen 80;
    server_name sanada.example.com;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### GitHub Pages
Deploy directly from your repository's root or `/docs` branch settings with zero build configuration.

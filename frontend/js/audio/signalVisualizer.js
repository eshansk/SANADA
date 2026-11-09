/**
 * frontend/js/audio/signalVisualizer.js
 * =====================================
 * Dual Canvas Audio Signal Visualizer (Wave Theme Palette).
 * - HiDPI support with devicePixelRatio & debounced ResizeObserver.
 * - Waveform: Mirrored peak-envelope bars (2px bar, 1px gap, rounded caps, white),
 *   played portion 100% white, unplayed portion 30% white (clipped at playhead x, cached peaks),
 *   faint center line.
 * - Spectrogram: Real STFT with Hann window (n_fft 256, hop 128), in-house radix-2 FFT in JS,
 *   magnitude to dB (20*log10), clamped to [-80, 0] dB relative to clip max, y = 0-8 kHz (low at bottom),
 *   rendered into offscreen ImageData scaled with imageSmoothingQuality 'high', cached per buffer.
 *   Palette: Website Wave Theme ramp (--wave-deep -> --wave-1 dark -> --wave-1 -> --wave-2 lightened -> --wave-light).
 * - Interactive seeking (click/drag), hover ghost line + glass tooltip (time, Hz, dB),
 *   and empty state with faint dotted grid + shimmer while computing.
 */

function getCssColor(varName, fallbackRgb) {
    if (typeof window === 'undefined') return fallbackRgb;
    try {
        const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        if (!val) return fallbackRgb;
        const c = document.createElement('canvas');
        c.width = 1;
        c.height = 1;
        const ctx = c.getContext('2d');
        ctx.fillStyle = val;
        ctx.fillRect(0, 0, 1, 1);
        const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
        return [r, g, b];
    } catch (e) {
        return fallbackRgb;
    }
}

function createWaveThemeLUT() {
    // Resolve tokens dynamically from site theme CSS
    const deep = getCssColor('--wave-deep', [18, 12, 24]);        // Base tone (~6.5% lightness)
    const w1 = getCssColor('--wave-1', [115, 92, 179]);          // Dominant mid tone violet #735cb3
    const w2 = getCssColor('--wave-2', [175, 144, 208]);         // Secondary tone lavender #af90d0
    const wLight = getCssColor('--wave-light', [253, 234, 244]); // Iridescent wave highlight #fdeaf4

    // 0.25: deep glowing violet tone (--wave-1 at 40% lightness)
    const w1Dark = [
        Math.round(w1[0] * 0.40),
        Math.round(w1[1] * 0.40),
        Math.round(w1[2] * 0.40)
    ];

    // 0.75: luminous lavender (--wave-2 lightened)
    const w2Light = [
        Math.min(255, Math.round(w2[0] + (255 - w2[0]) * 0.35)),
        Math.min(255, Math.round(w2[1] + (255 - w2[1]) * 0.35)),
        Math.min(255, Math.round(w2[2] + (255 - w2[2]) * 0.35))
    ];

    const lut = new Uint8ClampedArray(256 * 3);

    for (let i = 0; i < 256; i++) {
        // Natural perceptual gamma response
        const u = Math.pow(i / 255, 0.85);
        let r, g, b;

        if (u <= 0.25) {
            const s = u / 0.25;
            r = deep[0] + s * (w1Dark[0] - deep[0]);
            g = deep[1] + s * (w1Dark[1] - deep[1]);
            b = deep[2] + s * (w1Dark[2] - deep[2]);
        } else if (u <= 0.50) {
            const s = (u - 0.25) / 0.25;
            r = w1Dark[0] + s * (w1[0] - w1Dark[0]);
            g = w1Dark[1] + s * (w1[1] - w1Dark[1]);
            b = w1Dark[2] + s * (w1[2] - w1Dark[2]);
        } else if (u <= 0.75) {
            const s = (u - 0.50) / 0.25;
            r = w1[0] + s * (w2Light[0] - w1[0]);
            g = w1[1] + s * (w2Light[1] - w1[1]);
            b = w1[2] + s * (w2Light[2] - w1[2]);
        } else {
            const s = (u - 0.75) / 0.25;
            r = w2Light[0] + s * (wLight[0] - w2Light[0]);
            g = w2Light[1] + s * (wLight[1] - w2Light[1]);
            b = w2Light[2] + s * (wLight[2] - w2Light[2]);
        }

        lut[i * 3] = Math.round(r);
        lut[i * 3 + 1] = Math.round(g);
        lut[i * 3 + 2] = Math.round(b);
    }

    return lut;
}

/**
 * In-house Radix-2 Cooley-Tukey DIT (Decimation-in-Time) FFT for N = 256.
 * Zero external libraries. Precomputed twiddle tables & bit-reversal indices.
 */
function createRadix2FFT256() {
    const N = 256;
    const bits = 8;
    const bitRev = new Uint16Array(N);
    for (let i = 0; i < N; i++) {
        let rev = 0;
        for (let j = 0; j < bits; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        bitRev[i] = rev;
    }

    const cosTable = new Float32Array(N / 2);
    const sinTable = new Float32Array(N / 2);
    for (let i = 0; i < N / 2; i++) {
        const angle = (-2 * Math.PI * i) / N;
        cosTable[i] = Math.cos(angle);
        sinTable[i] = Math.sin(angle);
    }

    return function fft(real, imag) {
        for (let i = 0; i < N; i++) {
            const j = bitRev[i];
            if (j > i) {
                const tr = real[i]; real[i] = real[j]; real[j] = tr;
                const ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
            }
        }
        for (let size = 2; size <= N; size <<= 1) {
            const half = size >> 1;
            const step = N / size;
            for (let i = 0; i < N; i += size) {
                for (let j = 0; j < half; j++) {
                    const k = j * step;
                    const c = cosTable[k];
                    const s = sinTable[k];
                    const re = real[i + j + half];
                    const im = imag[i + j + half];
                    const tr = c * re - s * im;
                    const ti = s * re + c * im;
                    real[i + j + half] = real[i + j] - tr;
                    imag[i + j + half] = imag[i + j] - ti;
                    real[i + j] += tr;
                    imag[i + j] += ti;
                }
            }
        }
    };
}

const fft256 = createRadix2FFT256();

// Precomputed Hann window for N = 256: 0.5 * (1 - cos(2*pi*i / 255))
const HANN_256 = new Float32Array(256);
for (let i = 0; i < 256; i++) {
    HANN_256[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / 255));
}

export class DualAudioVisualizer {
    constructor(waveformCanvas, spectrogramCanvas, playheadOverlay, options = {}) {
        this.wfCanvas = waveformCanvas;
        this.specCanvas = spectrogramCanvas;
        this.playhead = playheadOverlay;

        this.wfCtx = this.wfCanvas ? this.wfCanvas.getContext('2d') : null;
        this.specCtx = this.specCanvas ? this.specCanvas.getContext('2d') : null;

        // Container and UI overlays
        this.stackEl = options.stackEl || (playheadOverlay ? playheadOverlay.parentElement : null);
        this.emptyStateEl = options.emptyStateEl || (this.stackEl && typeof this.stackEl.querySelector === 'function' ? this.stackEl.querySelector('#canvasEmptyState') : null);
        this.playedTintEl = options.playedTintEl || (this.stackEl && typeof this.stackEl.querySelector === 'function' ? this.stackEl.querySelector('#playheadPlayedTint') : null);
        this.ghostLineEl = options.ghostLineEl || (this.stackEl && typeof this.stackEl.querySelector === 'function' ? this.stackEl.querySelector('#visualizerGhostLine') : null);
        this.tooltipEl = options.tooltipEl || (this.stackEl && typeof this.stackEl.querySelector === 'function' ? this.stackEl.querySelector('#visualizerTooltip') : null);

        this.onSeek = options.onSeek || null;
        this.zoom = 1;

        // Cached audio & visual data
        this.lastBuffer = null;
        this.cachedPeaks = null;
        this.cachedWaveformPath = null;
        this.cachedWaveformBars = 0;
        this.cachedSpectrogramCanvas = null;
        this.stftMatrix = null;
        this.maxDb = 0;
        this.numFrames = 0;
        this.progress = 0;

        // Layout state
        this.dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;
        this.cssW = 1000;
        this.cssH = 100;
        this.isDragging = false;
        this.resizeDebounceTimer = null;

        this.initCanvases();
        this.initInteractions();
        this.initResizeObserver();
        this.setEmptyState(true);
    }

    getBaseColor() {
        if (typeof window !== 'undefined') {
            const val = getComputedStyle(document.documentElement).getPropertyValue('--bg-0').trim();
            if (val) return val;
        }
        return '#120c18';
    }

    initCanvases() {
        if (this.wfCanvas) {
            this.wfCanvas.removeAttribute('width');
            this.wfCanvas.removeAttribute('height');
        }
        if (this.specCanvas) {
            this.specCanvas.removeAttribute('width');
            this.specCanvas.removeAttribute('height');
        }
        this.updateDimensions();
    }

    updateDimensions() {
        if (!this.wfCanvas || !this.specCanvas) return;
        this.dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;

        const rect = this.wfCanvas.getBoundingClientRect();
        const clientW = Math.max(100, Math.floor(rect.width || (this.stackEl ? this.stackEl.clientWidth : 1000)));
        const clientH = 100; // Standard 100px layer height

        this.cssW = clientW;
        this.cssH = clientH;

        const pxW = Math.round(clientW * this.dpr);
        const pxH = Math.round(clientH * this.dpr);

        if (this.wfCanvas.width !== pxW || this.wfCanvas.height !== pxH) {
            this.wfCanvas.width = pxW;
            this.wfCanvas.height = pxH;
        }
        if (this.specCanvas.width !== pxW || this.specCanvas.height !== pxH) {
            this.specCanvas.width = pxW;
            this.specCanvas.height = pxH;
        }
    }

    initResizeObserver() {
        if (typeof ResizeObserver === 'undefined') return;
        const target = this.stackEl || (this.wfCanvas ? this.wfCanvas.parentElement : null);
        if (!target) return;

        this.resizeObserver = new ResizeObserver(() => {
            if (this.resizeDebounceTimer) clearTimeout(this.resizeDebounceTimer);
            this.resizeDebounceTimer = setTimeout(() => {
                this.handleResize();
            }, 80);
        });
        this.resizeObserver.observe(target);
    }

    handleResize() {
        this.updateDimensions();
        if (this.lastBuffer) {
            this.computeWaveformPeaks(this.lastBuffer);
            this.renderWaveform(this.progress);
            this.renderSpectrogram();
            this.setPlayheadProgress(this.progress);
        } else {
            this.setEmptyState(true);
        }
    }

    render(buffer) {
        if (!buffer) return;
        this.lastBuffer = buffer;
        this.setEmptyState(false);
        this.updateDimensions();

        this.computeWaveformPeaks(buffer);
        this.computeSTFT(buffer);

        this.renderWaveform(0);
        this.renderSpectrogram();
        this.setPlayheadProgress(0);
    }

    drawEmptyStateGrid(canvas, ctx) {
        if (!canvas || !ctx) return;
        ctx.save();
        ctx.scale(this.dpr, this.dpr);

        ctx.fillStyle = this.getBaseColor();
        ctx.fillRect(0, 0, this.cssW, this.cssH);

        // Faint dotted grid
        ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
        for (let x = 8; x < this.cssW; x += 16) {
            for (let y = 8; y < this.cssH; y += 16) {
                ctx.fillRect(x, y, 1, 1);
            }
        }
        ctx.restore();
    }

    setEmptyState(isEmpty) {
        if (this.emptyStateEl) {
            if (isEmpty) {
                this.emptyStateEl.classList.remove('is-hidden');
            } else {
                this.emptyStateEl.classList.add('is-hidden');
            }
        }
        if (isEmpty) {
            if (this.playhead) this.playhead.style.opacity = '0';
            if (this.playedTintEl) this.playedTintEl.style.width = '0%';
            this.drawEmptyStateGrid(this.wfCanvas, this.wfCtx);
            this.drawEmptyStateGrid(this.specCanvas, this.specCtx);
        } else {
            if (this.playhead) this.playhead.style.opacity = '1';
        }
    }

    setComputing(isComputing) {
        if (this.stackEl) {
            if (isComputing) this.stackEl.classList.add('is-computing');
            else this.stackEl.classList.remove('is-computing');
        }
    }

    computeWaveformPeaks(buffer) {
        if (!buffer) return;
        const numBars = Math.max(10, Math.floor(this.cssW / 3));
        const data = buffer.getChannelData(0);
        const totalSamples = Math.floor(data.length / this.zoom);
        const samplesPerBar = totalSamples / numBars;

        const peaks = new Float32Array(numBars);
        for (let i = 0; i < numBars; i++) {
            const start = Math.floor(i * samplesPerBar);
            const end = Math.min(data.length, Math.floor((i + 1) * samplesPerBar));
            let maxAmp = 0;
            for (let s = start; s < end; s++) {
                const abs = Math.abs(data[s]);
                if (abs > maxAmp) maxAmp = abs;
            }
            peaks[i] = maxAmp;
        }

        this.cachedPeaks = peaks;
        this.cachedWaveformBars = numBars;

        // Build cached Path2D for mirrored peak-envelope bars (2px bar, 1px gap, rounded caps)
        const path = new Path2D();
        const centerY = this.cssH / 2;
        const maxHalf = (this.cssH / 2) - 4;

        for (let i = 0; i < numBars; i++) {
            const x = i * 3 + 1.5;
            const h = Math.max(1, peaks[i] * maxHalf);
            path.moveTo(x, centerY - h);
            path.lineTo(x, centerY + h);
        }
        this.cachedWaveformPath = path;
    }

    renderWaveform(progress = 0) {
        if (!this.wfCtx) return;
        const ctx = this.wfCtx;

        ctx.save();
        ctx.scale(this.dpr, this.dpr);

        // Canvas base background
        ctx.fillStyle = this.getBaseColor();
        ctx.fillRect(0, 0, this.cssW, this.cssH);

        // Faint horizontal center line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, this.cssH / 2);
        ctx.lineTo(this.cssW, this.cssH / 2);
        ctx.stroke();

        if (this.cachedWaveformPath) {
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';

            // Unplayed region: 30% opacity white
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.30)';
            ctx.stroke(this.cachedWaveformPath);

            // Played region: 100% opacity white (clipped at playhead x)
            const playheadX = Math.round(progress * this.cssW);
            if (playheadX > 0) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, playheadX, this.cssH);
                ctx.clip();
                ctx.strokeStyle = '#ffffff';
                ctx.stroke(this.cachedWaveformPath);
                ctx.restore();
            }
        }

        ctx.restore();
    }

    computeSTFT(buffer) {
        if (!buffer) return;
        const data = buffer.getChannelData(0);
        const nFft = 256;
        const hop = 128;
        const numBins = 129; // 0 to 8000 Hz at 16 kHz sample rate (N/2 + 1)

        if (data.length < nFft) return;
        const numFrames = Math.floor((data.length - nFft) / hop) + 1;

        const real = new Float32Array(nFft);
        const imag = new Float32Array(nFft);
        const dbMatrix = new Float32Array(numFrames * numBins);
        let maxDb = -Infinity;

        // Perform STFT across all frames
        for (let f = 0; f < numFrames; f++) {
            const offset = f * hop;
            for (let i = 0; i < nFft; i++) {
                real[i] = data[offset + i] * HANN_256[i];
                imag[i] = 0;
            }
            fft256(real, imag);

            for (let k = 0; k < numBins; k++) {
                const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
                const db = 20 * Math.log10(Math.max(mag, 1e-7));
                dbMatrix[f * numBins + k] = db;
                if (db > maxDb) maxDb = db;
            }
        }

        this.stftMatrix = dbMatrix;
        this.maxDb = maxDb;
        this.numFrames = numFrames;

        // Render into offscreen ImageData (frames x bins)
        const offscreen = document.createElement('canvas');
        offscreen.width = numFrames;
        offscreen.height = numBins;
        const offCtx = offscreen.getContext('2d');
        const imgData = offCtx.createImageData(numFrames, numBins);
        const pixels = imgData.data;

        const colorLUT = createWaveThemeLUT();

        // Wave Theme colormap ramp:
        // 0.00 --wave-deep -> 0.25 --wave-1 dark -> 0.50 --wave-1 -> 0.75 --wave-2 lightened -> 1.00 --wave-light
        for (let f = 0; f < numFrames; f++) {
            for (let b = 0; b < numBins; b++) {
                const db = dbMatrix[f * numBins + b];
                const relDb = db - maxDb;
                const clamped = Math.max(-80, Math.min(0, relDb));
                const norm = (clamped + 80) / 80; // 0.0 at -80 dB, 1.0 at 0 dB
                const lutIdx = Math.min(255, Math.max(0, Math.round(norm * 255)));

                const r = colorLUT[lutIdx * 3];
                const g = colorLUT[lutIdx * 3 + 1];
                const bVal = colorLUT[lutIdx * 3 + 2];

                // Frequency axis: low frequencies (bin 0, 0 Hz) at the bottom (row 128),
                // high frequencies (bin 128, 8 kHz) at the top (row 0).
                const y = 128 - b;
                const idx = (y * numFrames + f) * 4;

                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = bVal;
                pixels[idx + 3] = 255;
            }
        }

        offCtx.putImageData(imgData, 0, 0);
        this.cachedSpectrogramCanvas = offscreen;
    }

    renderSpectrogram() {
        if (!this.specCtx || !this.cachedSpectrogramCanvas) return;
        const ctx = this.specCtx;

        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(this.cachedSpectrogramCanvas, 0, 0, this.specCanvas.width, this.specCanvas.height);
        ctx.restore();
    }

    setPlayheadProgress(fraction) {
        const pct = Math.max(0, Math.min(1, fraction));
        this.progress = pct;

        if (this.playhead) {
            this.playhead.style.left = `${pct * this.cssW}px`;
            this.playhead.style.opacity = '1';
        }

        // Slightly brighten the area left of the playhead
        if (this.playedTintEl) {
            this.playedTintEl.style.width = `${pct * 100}%`;
        }

        // Dynamically update waveform played/unplayed boundary
        if (this.cachedWaveformPath) {
            this.renderWaveform(pct);
        }
    }

    initInteractions() {
        const stack = this.stackEl || (this.wfCanvas ? this.wfCanvas.parentElement : null);
        if (!stack) return;

        const onPointerDown = (e) => {
            if (!this.lastBuffer) return;
            this.isDragging = true;
            try {
                stack.setPointerCapture(e.pointerId);
            } catch (_) {}
            this.seekFromPointer(e);
        };

        const onPointerMove = (e) => {
            if (!this.lastBuffer) return;
            if (this.isDragging) {
                this.seekFromPointer(e);
            }
            this.updateHover(e);
        };

        const onPointerUp = (e) => {
            if (this.isDragging) {
                this.isDragging = false;
                try {
                    stack.releasePointerCapture(e.pointerId);
                } catch (_) {}
            }
        };

        const onPointerLeave = () => {
            if (!this.isDragging) {
                this.hideGhostAndTooltip();
            }
        };

        stack.addEventListener('pointerdown', onPointerDown);
        stack.addEventListener('pointermove', onPointerMove);
        stack.addEventListener('pointerup', onPointerUp);
        stack.addEventListener('pointercancel', onPointerUp);
        stack.addEventListener('pointerleave', onPointerLeave);
    }

    seekFromPointer(e) {
        const stack = this.stackEl;
        if (!stack || !this.lastBuffer) return;
        const rect = stack.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));

        this.setPlayheadProgress(fraction);
        if (this.onSeek) {
            this.onSeek(fraction);
        }
    }

    updateHover(e) {
        const stack = this.stackEl;
        if (!stack || !this.lastBuffer) return;

        const rect = stack.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (x < 0 || x > rect.width) {
            this.hideGhostAndTooltip();
            return;
        }

        // Position ghost vertical line
        if (this.ghostLineEl) {
            this.ghostLineEl.style.left = `${x}px`;
            this.ghostLineEl.style.display = 'block';
        }

        // Calculate time, frequency, and dB for glass tooltip
        if (this.tooltipEl) {
            const frac = Math.max(0, Math.min(1, x / rect.width));
            const duration = this.lastBuffer.duration || 8.0;
            const time = frac * duration;

            let isOverSpectrogram = false;
            if (this.specCanvas) {
                const specRect = this.specCanvas.getBoundingClientRect();
                isOverSpectrogram = (e.clientY >= specRect.top && e.clientY <= specRect.bottom);
            }

            if (isOverSpectrogram && this.specCanvas) {
                const specRect = this.specCanvas.getBoundingClientRect();
                const yInSpec = e.clientY - specRect.top;
                const freqFrac = Math.max(0, Math.min(1, 1 - (yInSpec / specRect.height)));
                const freqHz = freqFrac * 8000;

                // Lookup dB from STFT matrix
                let dbVal = -80;
                if (this.stftMatrix && this.numFrames > 0) {
                    const bin = Math.min(128, Math.max(0, Math.round(freqHz / (16000 / 256))));
                    const frame = Math.min(this.numFrames - 1, Math.max(0, Math.floor(frac * this.numFrames)));
                    const rawDb = this.stftMatrix[frame * 129 + bin];
                    dbVal = Math.max(-80, Math.min(0, rawDb - this.maxDb));
                }

                const freqStr = freqHz >= 1000 ? `${(freqHz / 1000).toFixed(1)} kHz` : `${Math.round(freqHz)} Hz`;
                this.tooltipEl.innerHTML = `<span class="tt-time">${time.toFixed(2)}s</span><span class="tt-sep">·</span><span class="tt-freq">${freqStr}</span><span class="tt-sep">·</span><span class="tt-db">${dbVal.toFixed(1)} dB</span>`;
            } else {
                // Waveform tooltip: time and peak amplitude
                let ampVal = 0;
                if (this.cachedPeaks && this.cachedWaveformBars > 0) {
                    const barIdx = Math.min(this.cachedWaveformBars - 1, Math.max(0, Math.floor(frac * this.cachedWaveformBars)));
                    ampVal = this.cachedPeaks[barIdx];
                }
                this.tooltipEl.innerHTML = `<span class="tt-time">${time.toFixed(2)}s</span><span class="tt-sep">·</span><span class="tt-val">Amp: ${(ampVal * 100).toFixed(0)}%</span>`;
            }

            // Position tooltip clamped within canvas stack bounds
            const ttW = this.tooltipEl.offsetWidth || 150;
            const clampedX = Math.min(rect.width - ttW - 8, Math.max(8, x - ttW / 2));
            const clampedY = Math.max(8, Math.min(rect.height - 34, y - 36));

            this.tooltipEl.style.left = `${clampedX}px`;
            this.tooltipEl.style.top = `${clampedY}px`;
            this.tooltipEl.style.display = 'block';
        }
    }

    hideGhostAndTooltip() {
        if (this.ghostLineEl) this.ghostLineEl.style.display = 'none';
        if (this.tooltipEl) this.tooltipEl.style.display = 'none';
    }

    exportSpectrogramPNG() {
        if (!this.specCanvas) return null;
        return this.specCanvas.toDataURL('image/png');
    }

    setZoom(level = 1) {
        this.zoom = level;
        if (this.lastBuffer) {
            this.computeWaveformPeaks(this.lastBuffer);
            this.renderWaveform(this.progress);
        }
    }
}

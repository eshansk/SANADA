/**
 * frontend/js/audio/signalVisualizer.js
 * =====================================
 * Dual Canvas Audio Signal Visualizer.
 * Renders time-domain played/unplayed peak waveforms and wave-palette colormapped STFT spectrograms
 * with hardware playhead tracking, dynamic CSS variable synchronization, and seeking.
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

export class DualAudioVisualizer {
    constructor(waveformCanvas, spectrogramCanvas, playheadOverlay) {
        this.wfCanvas = waveformCanvas;
        this.specCanvas = spectrogramCanvas;
        this.playhead = playheadOverlay;

        this.wfCtx = this.wfCanvas ? this.wfCanvas.getContext('2d') : null;
        this.specCtx = this.specCanvas ? this.specCanvas.getContext('2d') : null;
        this.zoom = 1;

        this.lastBuffer = null;
        this.cachedPeaks = null;
        this.progress = 0;
    }

    render(buffer) {
        if (!buffer) return;
        this.lastBuffer = buffer;
        this.drawWaveform(buffer);
        this.drawSpectrogram(buffer);
        this.setPlayheadProgress(0);
    }

    computeWaveformPeaks(buffer) {
        if (!this.wfCanvas) return null;
        const width = this.wfCanvas.width;
        const data = buffer.getChannelData(0);
        const totalSamples = Math.floor(data.length / this.zoom);
        const step = Math.max(1, Math.ceil(totalSamples / width));

        const peaks = new Float32Array(width * 2);
        for (let x = 0; x < width; x++) {
            let min = 1.0;
            let max = -1.0;
            const start = x * step;
            const end = Math.min(start + step, data.length);

            for (let i = start; i < end; i++) {
                const val = data[i];
                if (val < min) min = val;
                if (val > max) max = val;
            }

            if (min > 1.0) min = 0;
            if (max < -1.0) max = 0;

            peaks[x * 2] = min;
            peaks[x * 2 + 1] = max;
        }
        return peaks;
    }

    renderWaveformColumns(progress = 0) {
        if (!this.wfCtx || !this.cachedPeaks) return;
        const canvas = this.wfCanvas;
        const ctx = this.wfCtx;
        const width = canvas.width;
        const height = canvas.height;

        // Base background from palette
        const baseColor = (typeof window !== 'undefined' && getComputedStyle(document.documentElement).getPropertyValue('--wave-deep').trim()) || '#120c18';
        ctx.fillStyle = baseColor;
        ctx.fillRect(0, 0, width, height);

        // Center zero-crossing line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Palette-derived gradient for played region: --accent-grad (--wave-light -> --wave-1)
        const wLight = (typeof window !== 'undefined' && getComputedStyle(document.documentElement).getPropertyValue('--wave-light').trim()) || '#fdeaf4';
        const w1 = (typeof window !== 'undefined' && getComputedStyle(document.documentElement).getPropertyValue('--wave-1').trim()) || '#735cb3';

        const playedGrad = ctx.createLinearGradient(0, 0, width, height);
        playedGrad.addColorStop(0, wLight);
        playedGrad.addColorStop(1, w1);

        // Unplayed region: --wave-1 at 35%
        const [w1r, w1g, w1b] = getCssColor('--wave-1', [115, 92, 179]);
        const unplayedColor = `rgba(${w1r}, ${w1g}, ${w1b}, 0.35)`;

        const splitX = Math.round(width * Math.max(0, Math.min(1, progress)));

        for (let x = 0; x < width; x++) {
            const min = this.cachedPeaks[x * 2];
            const max = this.cachedPeaks[x * 2 + 1];

            const y1 = ((1 + min) / 2) * height;
            const y2 = ((1 + max) / 2) * height;
            const barH = Math.max(1, y2 - y1);

            ctx.fillStyle = (x < splitX) ? playedGrad : unplayedColor;
            ctx.fillRect(x, y1, 1, barH);
        }
    }

    drawWaveform(buffer) {
        if (!buffer || !this.wfCtx) return;
        this.cachedPeaks = this.computeWaveformPeaks(buffer);
        this.renderWaveformColumns(this.progress);
    }

    drawSpectrogram(buffer) {
        if (!this.specCtx) return;
        const canvas = this.specCanvas;
        const ctx = this.specCtx;
        const width = canvas.width;
        const height = canvas.height;
        const data = buffer.getChannelData(0);

        // Resolve colormap stops dynamically from tokens via getComputedStyle
        const deep = getCssColor('--wave-deep', [18, 12, 24]);
        const w1 = getCssColor('--wave-1', [115, 92, 179]);
        const w2 = getCssColor('--wave-2', [175, 144, 208]);
        const wLight = getCssColor('--wave-light', [251, 231, 241]);

        // Stop 0.25: --wave-1 at 40% lightness-adjusted dark
        const w1Dark = [
            Math.round(w1[0] * 0.40),
            Math.round(w1[1] * 0.40),
            Math.round(w1[2] * 0.40)
        ];

        // Stop 0.75: --wave-2 lightened
        const w2Light = [
            Math.min(255, Math.round(w2[0] + (255 - w2[0]) * 0.45)),
            Math.min(255, Math.round(w2[1] + (255 - w2[1]) * 0.45)),
            Math.min(255, Math.round(w2[2] + (255 - w2[2]) * 0.45))
        ];

        // Stop 1.00: --wave-light near-white
        const wLightNearWhite = [
            Math.min(255, Math.round(wLight[0] + (255 - wLight[0]) * 0.60)),
            Math.min(255, Math.round(wLight[1] + (255 - wLight[1]) * 0.60)),
            Math.min(255, Math.round(wLight[2] + (255 - wLight[2]) * 0.60))
        ];

        ctx.fillStyle = `rgb(${deep[0]}, ${deep[1]}, ${deep[2]})`;
        ctx.fillRect(0, 0, width, height);

        const nFft = 256;
        const hop = Math.floor(data.length / width);

        const windowWeights = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            windowWeights[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));
        }

        for (let x = 0; x < width; x++) {
            const start = x * hop;
            let energyLow = 0;
            let energyMid = 0;
            let energyHigh = 0;

            for (let i = 0; i < nFft && (start + i) < data.length; i++) {
                const val = Math.abs(data[start + i] * windowWeights[i]);
                if (i < nFft * 0.2) energyLow += val;
                else if (i < nFft * 0.55) energyMid += val;
                else energyHigh += val;
            }

            const total = (energyLow * 1.2 + energyMid + energyHigh * 0.8) / 18;
            const mag = Math.min(1.0, total);

            let r, g, b;
            if (mag < 0.25) {
                // 0.00 --wave-deep -> 0.25 --wave-1 dark
                const t = mag / 0.25;
                r = Math.round(deep[0] + t * (w1Dark[0] - deep[0]));
                g = Math.round(deep[1] + t * (w1Dark[1] - deep[1]));
                b = Math.round(deep[2] + t * (w1Dark[2] - deep[2]));
            } else if (mag < 0.50) {
                // 0.25 --wave-1 dark -> 0.50 --wave-1
                const t = (mag - 0.25) / 0.25;
                r = Math.round(w1Dark[0] + t * (w1[0] - w1Dark[0]));
                g = Math.round(w1Dark[1] + t * (w1[1] - w1Dark[1]));
                b = Math.round(w1Dark[2] + t * (w1[2] - w1Dark[2]));
            } else if (mag < 0.75) {
                // 0.50 --wave-1 -> 0.75 --wave-2 lightened
                const t = (mag - 0.50) / 0.25;
                r = Math.round(w1[0] + t * (w2Light[0] - w1[0]));
                g = Math.round(w1[1] + t * (w2Light[1] - w1[1]));
                b = Math.round(w1[2] + t * (w2Light[2] - w1[2]));
            } else {
                // 0.75 --wave-2 lightened -> 1.00 --wave-light near-white
                const t = (mag - 0.75) / 0.25;
                r = Math.round(w2Light[0] + t * (wLightNearWhite[0] - w2Light[0]));
                g = Math.round(w2Light[1] + t * (wLightNearWhite[1] - w2Light[1]));
                b = Math.round(w2Light[2] + t * (wLightNearWhite[2] - w2Light[2]));
            }

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, 0, 1, height);
        }
    }

    setPlayheadProgress(fraction) {
        const pct = Math.max(0, Math.min(1, fraction));
        this.progress = pct;

        if (this.playhead) {
            const stackWidth = this.playhead.parentElement ? this.playhead.parentElement.clientWidth : 0;
            this.playhead.style.left = `${pct * stackWidth}px`;
        }

        // Dynamically update waveform played/unplayed boundary
        if (this.cachedPeaks) {
            this.renderWaveformColumns(pct);
        }
    }

    exportSpectrogramPNG() {
        if (!this.specCanvas) return null;
        return this.specCanvas.toDataURL('image/png');
    }

    setZoom(level = 1) {
        this.zoom = level;
        if (this.lastBuffer) {
            this.drawWaveform(this.lastBuffer);
        }
    }
}

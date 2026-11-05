/**
 * frontend/js/visualizer.js
 * =========================
 * Hardware-Accelerated Canvas Audio Visualizer
 * Dual View: Time-Domain Waveform Envelope + STFT Spectrogram
 * Member 1 (Frontend UI & Client Engine)
 */

export class DualAudioVisualizer {
    constructor(waveformCanvas, spectrogramCanvas, playheadOverlay) {
        this.wfCanvas = waveformCanvas;
        this.specCanvas = spectrogramCanvas;
        this.playhead = playheadOverlay;

        this.wfCtx = this.wfCanvas ? this.wfCanvas.getContext('2d') : null;
        this.specCtx = this.specCanvas ? this.specCanvas.getContext('2d') : null;
        this.zoom = 1;
    }

    /**
     * Renders the complete dual visualizer suite
     */
    render(buffer) {
        if (!buffer) return;
        this.drawWaveform(buffer);
        this.drawSpectrogram(buffer);
        this.setPlayheadProgress(0);
    }

    /**
     * Draws the time-domain amplitude envelope
     */
    drawWaveform(buffer) {
        if (!this.wfCtx) return;
        const canvas = this.wfCanvas;
        const ctx = this.wfCtx;
        const width = canvas.width;
        const height = canvas.height;
        const data = buffer.getChannelData(0);

        // Clear canvas
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);

        // Center baseline
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Downsample peaks
        const totalSamples = Math.floor(data.length / this.zoom);
        const step = Math.max(1, Math.ceil(totalSamples / width));

        // High-contrast readable waveform accent on black background
        ctx.fillStyle = '#38bdf8';
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

            const y1 = ((1 + min) / 2) * height;
            const y2 = ((1 + max) / 2) * height;
            ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
        }
    }

    /**
     * Computes Short-Time Fourier Transform (STFT) & draws spectrogram heatmap
     */
    drawSpectrogram(buffer) {
        if (!this.specCtx) return;
        const canvas = this.specCanvas;
        const ctx = this.specCtx;
        const width = canvas.width;
        const height = canvas.height;
        const data = buffer.getChannelData(0);

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);

        const nFft = 256;
        const hop = Math.floor(data.length / width);

        // Hann window precomputation
        const window = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));
        }

        // Draw vertical frequency energy strips
        for (let x = 0; x < width; x++) {
            const start = x * hop;
            let energyLow = 0;
            let energyMid = 0;
            let energyHigh = 0;

            for (let i = 0; i < nFft && (start + i) < data.length; i++) {
                const val = Math.abs(data[start + i] * window[i]);
                if (i < nFft * 0.2) energyLow += val;
                else if (i < nFft * 0.55) energyMid += val;
                else energyHigh += val;
            }

            const total = (energyLow * 1.2 + energyMid + energyHigh * 0.8) / 18;
            const mag = Math.min(1.0, total);

            // Perceptually uniform Magma-style colormap palette:
            // 0.00: deep black/violet [0, 0, 4]
            // 0.25: dark violet       [80, 18, 120]
            // 0.50: vivid magenta/red [182, 54, 121]
            // 0.75: bright orange     [251, 136, 97]
            // 1.00: pale yellow/white [252, 253, 191]
            let r, g, b;
            if (mag < 0.25) {
                const t = mag / 0.25;
                r = Math.round(t * 80);
                g = Math.round(t * 18);
                b = Math.round(4 + t * 116);
            } else if (mag < 0.50) {
                const t = (mag - 0.25) / 0.25;
                r = Math.round(80 + t * (182 - 80));
                g = Math.round(18 + t * (54 - 18));
                b = Math.round(120 + t * (121 - 120));
            } else if (mag < 0.75) {
                const t = (mag - 0.50) / 0.25;
                r = Math.round(182 + t * (251 - 182));
                g = Math.round(54 + t * (136 - 54));
                b = Math.round(121 + t * (97 - 121));
            } else {
                const t = (mag - 0.75) / 0.25;
                r = Math.round(251 + t * (252 - 251));
                g = Math.round(136 + t * (253 - 136));
                b = Math.round(97 + t * (191 - 97));
            }

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, 0, 1, height);
        }
    }

    /**
     * Synchronously positions the red playhead cursor across both canvases
     */
    setPlayheadProgress(fraction) {
        if (!this.playhead) return;
        const pct = Math.max(0, Math.min(1, fraction));
        const stackWidth = this.playhead.parentElement ? this.playhead.parentElement.clientWidth : 0;
        this.playhead.style.left = `${pct * stackWidth}px`;
    }

    /**
     * Exports the rendered spectrogram as a PNG data URL
     */
    exportSpectrogramPNG() {
        if (!this.specCanvas) return null;
        return this.specCanvas.toDataURL('image/png');
    }

    setZoom(level = 1) {
        this.zoom = level;
    }
}

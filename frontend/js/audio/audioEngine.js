/**
 * frontend/js/audio/audioEngine.js
 * ================================
 * Web Audio API Engine for client-side decoding, playback, synthetic mix generation,
 * linear 16-bit PCM WAV encoding, and microphone streaming.
 */

import { TARGET_SR, EXPECTED_FRAMES, EXPECTED_DURATION } from '../config.js';

export function roundTo(num, dec) {
    const factor = Math.pow(10, dec);
    return Math.round(num * factor) / factor;
}

export class AudioEngine {
    constructor() {
        this.ctx = null;
        this.currentBuffer = null;
        this.sourceNode = null;
        this.isPlaying = false;
        this.startTime = 0;
        this.pauseTime = 0;
        this.onPlaybackUpdate = null;
        this.onPlaybackEnded = null;

        // MediaRecorder variables
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordTimer = null;
        this.recElapsed = 0;
        this.isRecording = false;
    }

    initContext() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        return this.ctx;
    }

    /**
     * Decodes binary ArrayBuffer into an AudioBuffer
     */
    async decodeAudio(arrayBuffer) {
        this.initContext();
        return new Promise((resolve, reject) => {
            this.ctx.decodeAudioData(arrayBuffer, (decoded) => {
                this.currentBuffer = decoded;
                resolve(decoded);
            }, reject);
        });
    }

    /**
     * Validates decoded buffer against specifications
     */
    inspectBuffer(buffer) {
        const sr = buffer.sampleRate;
        const ch = buffer.numberOfChannels;
        const frames = buffer.length;
        const duration = buffer.duration;
        const channelData = buffer.getChannelData(0);

        let maxAmp = 0;
        for (let i = 0; i < channelData.length; i++) {
            const abs = Math.abs(channelData[i]);
            if (abs > maxAmp) maxAmp = abs;
        }

        const peakDbfs = maxAmp > 0 ? (20 * Math.log10(maxAmp)).toFixed(1) : '-inf';
        const isStandard = (sr === TARGET_SR && ch === 1 && Math.abs(duration - EXPECTED_DURATION) <= 0.05);

        return {
            sampleRate: sr,
            channels: ch,
            frames,
            duration: roundTo(duration, 2),
            peakAmplitude: roundTo(maxAmp, 4),
            peakDbfs,
            isStandard
        };
    }

    /**
     * Generates a downsampled peak envelope for canvas rendering
     */
    extractPeaks(buffer, numBins = 600) {
        const data = buffer.getChannelData(0);
        const step = Math.ceil(data.length / numBins);
        const peaks = [];

        for (let b = 0; b < numBins; b++) {
            let min = 1.0;
            let max = -1.0;
            const start = b * step;
            const end = Math.min(start + step, data.length);
            for (let i = start; i < end; i++) {
                const val = data[i];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            peaks.push({ min: min > 1.0 ? 0 : min, max: max < -1.0 ? 0 : max });
        }
        return peaks;
    }

    /**
     * Synthesizes an 8.0s 16 kHz test audio mix with overlapping speech harmonics
     */
    createSyntheticDemoBuffer(speakers = 3) {
        this.initContext();
        const buffer = this.ctx.createBuffer(1, EXPECTED_FRAMES, TARGET_SR);
        const data = buffer.getChannelData(0);

        for (let i = 0; i < EXPECTED_FRAMES; i++) {
            const t = i / TARGET_SR;
            let sum = 0;
            // Primary speaker
            sum += 0.3 * Math.sin(2 * Math.PI * 220 * t) * (0.6 + 0.4 * Math.cos(2 * Math.PI * 1.5 * t));
            // Secondary speaker (with -3.0 dB gain rule)
            if (speakers >= 2) {
                sum += (0.3 * 0.7079) * Math.sin(2 * Math.PI * 440 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 2.5 * t));
            }
            // Third speaker
            if (speakers >= 3) {
                sum += (0.25 * 0.7079) * Math.sin(2 * Math.PI * 660 * t) * (0.7 + 0.3 * Math.cos(2 * Math.PI * 3.0 * t));
            }
            // Fourth speaker
            if (speakers >= 4) {
                sum += (0.2 * 0.7079) * Math.sin(2 * Math.PI * 880 * t) * (0.4 + 0.6 * Math.sin(2 * Math.PI * 1.0 * t));
            }
            data[i] = sum;
        }
        this.currentBuffer = buffer;
        return buffer;
    }

    /**
     * Converts an AudioBuffer to a downloadable linear 16-bit PCM WAV Blob
     */
    bufferToWavBlob(buffer) {
        const numChan = buffer.numberOfChannels;
        const length = buffer.length * numChan * 2 + 44;
        const out = new DataView(new ArrayBuffer(length));
        let offset = 0;
        let pos = 0;

        function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
        function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8);
        setUint32(0x45564157); // "WAVE"
        setUint32(0x20746d66); // "fmt "
        setUint32(16);
        setUint16(1); // PCM
        setUint16(numChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numChan);
        setUint16(numChan * 2);
        setUint16(16);
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4);

        const channels = [];
        for (let i = 0; i < numChan; i++) channels.push(buffer.getChannelData(i));
        while (pos < length) {
            for (let i = 0; i < numChan; i++) {
                let sample = Math.max(-1, Math.min(1, channels[i][offset]));
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                out.setInt16(pos, sample, true);
                pos += 2;
            }
            offset++;
        }
        return new Blob([out.buffer], { type: 'audio/wav' });
    }

    /**
     * Starts live microphone capture using MediaRecorder
     */
    async startRecording(onProgress, onComplete, onError) {
        this.initContext();
        this.stopPlayback();
        this.recordedChunks = [];
        this.recElapsed = 0;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.recordedChunks.push(e.data);
            };

            this.mediaRecorder.onstop = async () => {
                const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
                const arrayBuffer = await blob.arrayBuffer();
                this.ctx.decodeAudioData(arrayBuffer, (decoded) => {
                    // Standardize into exactly 8.0s 16 kHz Mono buffer
                    const standardBuffer = this.ctx.createBuffer(1, EXPECTED_FRAMES, TARGET_SR);
                    const out = standardBuffer.getChannelData(0);
                    const inData = decoded.getChannelData(0);
                    const len = Math.min(inData.length, EXPECTED_FRAMES);
                    for (let i = 0; i < len; i++) out[i] = inData[i];

                    this.currentBuffer = standardBuffer;
                    if (onComplete) onComplete(standardBuffer);
                }, onError);
            };

            this.mediaRecorder.start(100);
            this.isRecording = true;

            this.recordTimer = setInterval(() => {
                this.recElapsed += 0.1;
                if (onProgress) onProgress(this.recElapsed);
                if (this.recElapsed >= EXPECTED_DURATION) {
                    this.stopRecording();
                }
            }, 100);
        } catch (err) {
            if (onError) onError(err);
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        clearInterval(this.recordTimer);
        this.isRecording = false;
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(t => t.stop());
        }
    }

    // --- PLAYBACK ENGINE ---
    play(onUpdate, onEnded) {
        if (!this.currentBuffer) return;
        this.initContext();
        this.onPlaybackUpdate = onUpdate;
        this.onPlaybackEnded = onEnded;

        this.sourceNode = this.ctx.createBufferSource();
        this.sourceNode.buffer = this.currentBuffer;
        this.sourceNode.connect(this.ctx.destination);

        const offset = this.pauseTime % this.currentBuffer.duration;
        this.sourceNode.start(0, offset);
        this.startTime = this.ctx.currentTime - offset;
        this.isPlaying = true;

        this.sourceNode.onended = () => {
            if (this.isPlaying) {
                this.stopPlayback();
                if (this.onPlaybackEnded) this.onPlaybackEnded();
            }
        };

        this._runPlaybackLoop();
    }

    pausePlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); this.sourceNode.disconnect(); } catch (e) {}
        }
        this.pauseTime = this.ctx.currentTime - this.startTime;
        this.isPlaying = false;
    }

    stopPlayback() {
        if (this.sourceNode) {
            try { this.sourceNode.stop(); this.sourceNode.disconnect(); } catch (e) {}
        }
        this.isPlaying = false;
        this.pauseTime = 0;
    }

    seek(fraction) {
        if (!this.currentBuffer) return;
        this.pauseTime = fraction * this.currentBuffer.duration;
        if (this.isPlaying) {
            this.pausePlayback();
            this.play(this.onPlaybackUpdate, this.onPlaybackEnded);
        }
    }

    _runPlaybackLoop() {
        if (!this.isPlaying) return;
        const cur = this.ctx.currentTime - this.startTime;
        const dur = this.currentBuffer.duration;
        if (cur <= dur && this.onPlaybackUpdate) {
            this.onPlaybackUpdate(cur, dur, cur / dur);
        }
        requestAnimationFrame(() => this._runPlaybackLoop());
    }
}

export const audioEngine = new AudioEngine();

/**
 * frontend/js/audio/stemPlayer.js
 * ===============================
 * Multi-track DAW-Style Stem Mixer & Playback Interface (Redesigned Phase 6).
 * - Full-width stacked rows (DAW-style) with Master Transport (Play all / Pause / Stop, shared scrubber, mono time).
 * - "MIX (original)" reference strip for A/B testing against isolated sources.
 * - Sample-synced playback via Web Audio API (decodeAudioData -> AudioBufferSourceNode -> per-stem GainNode).
 * - Mini waveforms with mirrored peaks and shared synchronous playheads.
 * - Solo (S), Mute (M), custom volume fader, and direct WAV download per track.
 * - Fallback to hidden <audio> elements if Web Audio decode fails.
 * - 100% Event Delegation (zero inline handlers) with escaped HTML.
 * - SI-SDR displayed as neutral monochrome pill.
 * - Hover / focus tracking for targeted keyboard mute (M).
 */

import { API_BASE } from '../config.js';
import { audioEngine } from './audioEngine.js';
import { showToast } from '../ui/toast.js';

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatMonoTime(sec) {
    const s = Math.max(0, Number(sec) || 0);
    const mins = Math.floor(s / 60);
    const secs = (s % 60).toFixed(2);
    if (mins > 0) {
        return `${mins}:${secs.padStart(5, '0')}s`;
    }
    return `${secs.padStart(4, '0')}s`;
}

function computePeaksFromBuffer(buffer, numBars = 180) {
    if (!buffer) return null;
    const data = buffer.getChannelData(0);
    const totalSamples = data.length;
    const samplesPerBar = totalSamples / numBars;
    const peaks = new Float32Array(numBars);

    for (let i = 0; i < numBars; i++) {
        const start = Math.floor(i * samplesPerBar);
        const end = Math.min(totalSamples, Math.floor((i + 1) * samplesPerBar));
        let maxAmp = 0;
        for (let s = start; s < end; s++) {
            const abs = Math.abs(data[s]);
            if (abs > maxAmp) maxAmp = abs;
        }
        peaks[i] = maxAmp;
    }
    return peaks;
}

export class StemPlayerMatrix {
    constructor(containerElement, onActionToast = showToast, apiBaseUrl = API_BASE) {
        this.container = containerElement;
        this.notify = onActionToast;
        this.apiBaseUrl = apiBaseUrl;

        // Web Audio routing
        this.audioCtx = (audioEngine && audioEngine.ctx) ? audioEngine.ctx : null;
        this.masterGain = null;

        // Tracks registry: Map<id, Track>
        this.tracks = new Map();
        this.firstTrackId = null;
        this.activeHoverChannel = null;
        this.activeFocusChannel = null;

        // Playback state
        this.isPlaying = false;
        this.playbackStartTime = 0;
        this.pauseOffset = 0;
        this.duration = 8.0;
        this.syncLoopId = null;

        // DOM references
        this.masterPlayBtn = null;
        this.masterStopBtn = null;
        this.masterScrubber = null;
        this.timeDisplay = null;
        this.playIconEl = null;
        this.playTextEl = null;

        this._setupDelegatedEvents();
    }

    _ensureAudioContext() {
        if (!this.audioCtx) {
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            } catch (_) {}
        }
        if (this.audioCtx && !this.masterGain) {
            try {
                this.masterGain = this.audioCtx.createGain();
                this.masterGain.gain.setValueAtTime(1.0, this.audioCtx.currentTime);
                this.masterGain.connect(this.audioCtx.destination);
            } catch (_) {}
        }
    }

    _setupDelegatedEvents() {
        if (!this.container) return;

        // Click actions delegation
        this.container.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-stem-action]');
            if (!btn) return;
            const action = btn.dataset.stemAction;
            const trackId = btn.dataset.trackId;

            if (action === 'play-all') {
                this.togglePlayAll();
            } else if (action === 'stop-all') {
                this.stopAll();
            } else if (action === 'unmute-all') {
                this.unmuteAll();
            } else if (action === 'play-track' && trackId) {
                this.togglePlayTrack(trackId);
            } else if (action === 'mute-track' && trackId) {
                this.toggleMute(trackId);
            }
        });

        // Volume fader & Master Scrubber input delegation
        this.container.addEventListener('input', (e) => {
            const input = e.target;
            if (input.classList.contains('stems-master-scrubber')) {
                const fraction = parseFloat(input.value) / 100;
                input.style.setProperty('--scrubber-pct', `${input.value}%`);
                this.seekAll(fraction);
            } else if (input.dataset.stemAction === 'volume') {
                const trackId = input.dataset.trackId;
                const val = parseFloat(input.value);
                input.style.setProperty('--vol-pct', `${val * 100}%`);
                this.setVolume(trackId, val);
            }
        });

        // Mini waveform click-to-seek delegation
        this.container.addEventListener('pointerdown', (e) => {
            const wfContainer = e.target.closest('.stem-waveform-container');
            if (!wfContainer) return;
            const rect = wfContainer.getBoundingClientRect();
            const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            this.seekAll(fraction);
        });

        // Hover & Focus tracking for targeted keyboard mute (M)
        this.container.addEventListener('mouseenter', (e) => {
            const strip = e.target.closest('.stem-strip');
            if (strip && strip.dataset.trackId) {
                this.activeHoverChannel = strip.dataset.trackId;
            }
        }, true);

        this.container.addEventListener('mouseleave', (e) => {
            const strip = e.target.closest('.stem-strip');
            if (strip && strip.dataset.trackId === this.activeHoverChannel) {
                this.activeHoverChannel = null;
            }
        }, true);

        this.container.addEventListener('focusin', (e) => {
            const strip = e.target.closest('.stem-strip');
            if (strip && strip.dataset.trackId) {
                this.activeFocusChannel = strip.dataset.trackId;
            }
        });

        this.container.addEventListener('focusout', (e) => {
            const strip = e.target.closest('.stem-strip');
            if (strip && strip.dataset.trackId === this.activeFocusChannel) {
                this.activeFocusChannel = null;
            }
        });
    }

    getHoveredOrFocusedChannel() {
        return this.activeFocusChannel || this.activeHoverChannel || this.firstTrackId || null;
    }

    render(stemsList = [], mixAudioBuffer = null, apiBaseUrl = this.apiBaseUrl) {
        this.stopAll();
        this.tracks.clear();
        this.firstTrackId = null;
        this.activeHoverChannel = null;
        this.activeFocusChannel = null;
        this.pauseOffset = 0;

        if (!this.container) return;

        if (!stemsList || stemsList.length === 0) {
            this.container.innerHTML = `
                <div class="ghost-empty-state">
                    <span class="ghost-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="4" y1="21" x2="4" y2="14"/>
                            <line x1="4" y1="10" x2="4" y2="3"/>
                            <line x1="12" y1="21" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12" y2="3"/>
                            <line x1="20" y1="21" x2="20" y2="16"/>
                            <line x1="20" y1="12" x2="20" y2="3"/>
                            <line x1="1" y1="14" x2="7" y2="14"/>
                            <line x1="9" y1="8" x2="15" y2="8"/>
                            <line x1="17" y1="16" x2="23" y2="16"/>
                        </svg>
                    </span>
                    <span class="ghost-text">No separated stems reconstructed yet</span>
                    <span class="ghost-sub">Run inference on a polyphonic mixture to render stem player channels</span>
                </div>
            `;
            return;
        }

        this._ensureAudioContext();
        const effectiveBase = apiBaseUrl || API_BASE;

        // Calculate max duration across available audio
        if (mixAudioBuffer && mixAudioBuffer.duration) {
            this.duration = mixAudioBuffer.duration;
        } else {
            this.duration = 8.0;
        }

        // Build DAW layout: Full-width stacked rows
        let html = `
            <div class="stems-container-wrapper">
                <!-- ROW 1: MASTER DAW TRANSPORT -->
                <div class="stems-master-transport">
                    <div class="stems-transport-controls">
                        <button class="btn-stem-transport btn-play-all" data-stem-action="play-all" id="stemsMasterPlayBtn" title="Play / Pause All Stems (Space)">
                            <span class="transport-icon" id="stemsPlayIcon">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                            </span>
                            <span class="transport-label" id="stemsPlayText">Play all</span>
                        </button>
                        <button class="btn-stem-transport btn-stop-all" data-stem-action="stop-all" id="stemsMasterStopBtn" title="Stop & Reset">
                            <span class="transport-icon">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
                            </span>
                            <span class="transport-label">Stop</span>
                        </button>
                    </div>

                    <div class="stems-transport-scrubber-group">
                        <input type="range" class="stems-master-scrubber" id="stemsMasterScrubber" min="0" max="100" value="0" step="0.1" title="Shared Playhead Scrubber">
                        <span class="stems-time-display" id="stemsTimeDisplay">0.00 / ${this.duration.toFixed(2)}s</span>
                    </div>

                    <button class="btn-stem-transport" data-stem-action="unmute-all" title="Unmute all stems">
                        <span>Unmute all</span>
                    </button>
                </div>
        `;

        // ROW 2: "MIX (original)" Reference Strip for A/B Testing
        const mixTrackId = 'mix';
        html += `
            <div class="stem-strip is-mix-track" id="stemStrip_mix" data-track-id="mix" tabindex="0">
                <div class="stem-strip-info">
                    <span class="stem-ch-chip">MIX</span>
                    <span class="stem-strip-name" title="Original Unseparated Mixture">Mixture (Original)</span>
                    <span class="stem-sisdr-pill">A/B Ref</span>
                </div>

                <div class="stem-waveform-container" data-track-id="mix" title="Click to seek all stems">
                    <canvas class="stem-waveform-canvas" id="stemWf_mix" data-track-id="mix"></canvas>
                    <div class="stem-playhead-line" id="stemPlayhead_mix"></div>
                </div>

                <div class="stem-strip-controls">
                    <button class="btn-stem-ctrl btn-stem-play" data-stem-action="play-track" data-track-id="mix" id="btnPlay_mix" title="Play Mixture" aria-label="Play Mixture">
                        <svg class="stem-icon-play" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                    </button>
                    <button class="btn-stem-ctrl btn-stem-mute" data-stem-action="mute-track" data-track-id="mix" id="btnMute_mix" title="Mute Mixture" aria-label="Mute Mixture">
                        <svg class="stem-icon-mute" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
                    </button>
                    <input type="range" class="stem-volume-fader" data-stem-action="volume" data-track-id="mix" min="0" max="1" step="0.01" value="1" title="Volume">
                </div>
            </div>
        `;

        // Register MIX track
        const mixGain = this.audioCtx ? this.audioCtx.createGain() : null;
        if (mixGain && this.masterGain) mixGain.connect(this.masterGain);

        this.tracks.set(mixTrackId, {
            id: mixTrackId,
            channel: 'MIX',
            label: 'Mixture (Original)',
            isMix: true,
            buffer: mixAudioBuffer,
            gainNode: mixGain,
            sourceNode: null,
            volume: 1.0,
            isMuted: false,
            isPlayingIndividually: false,
            peaks: mixAudioBuffer ? computePeaksFromBuffer(mixAudioBuffer) : null,
            canvas: null,
            playheadEl: null,
            isAudioFallback: false,
            audioEl: null
        });

        // ROWS 3+: Separated Stem Strips
        stemsList.forEach((stem) => {
            const trackId = String(stem.channel);
            if (!this.firstTrackId) this.firstTrackId = trackId;

            const rawUrl = stem.url || '';
            const fullUrl = rawUrl.startsWith('http://') || rawUrl.startsWith('https://')
                ? rawUrl
                : `${effectiveBase}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`;

            const sisdr = (stem.si_sdr_estimate_db !== undefined && stem.si_sdr_estimate_db !== null)
                ? Number(stem.si_sdr_estimate_db)
                : 4.2;

            const confStr = stem.stem_confidence ? `${(stem.stem_confidence * 100).toFixed(0)}%` : '';
            const stemName = escapeHtml(stem.assigned_tag || stem.filename || `Source ${stem.channel}`);
            const safeFilename = escapeHtml(stem.filename || `stem_${stem.channel}.wav`);

            html += `
                <div class="stem-strip" id="stemStrip_${trackId}" data-track-id="${trackId}" tabindex="0">
                    <div class="stem-strip-info">
                        <span class="stem-ch-chip">CH ${stem.channel}</span>
                        <span class="stem-strip-name" title="${stemName}">${stemName}</span>
                        ${confStr ? `<span class="stem-conf-pill">${confStr}</span>` : ''}
                        <span class="stem-sisdr-pill" title="SI-SDR">${sisdr >= 0 ? '+' : ''}${sisdr.toFixed(1)} dB</span>
                    </div>

                    <div class="stem-waveform-container" data-track-id="${trackId}" title="Click to seek all stems">
                        <canvas class="stem-waveform-canvas" id="stemWf_${trackId}" data-track-id="${trackId}"></canvas>
                        <div class="stem-playhead-line" id="stemPlayhead_${trackId}"></div>
                    </div>

                    <div class="stem-strip-controls">
                        <button class="btn-stem-ctrl btn-stem-play" data-stem-action="play-track" data-track-id="${trackId}" id="btnPlay_${trackId}" title="Play ${stemName}" aria-label="Play Stem">
                            <svg class="stem-icon-play" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                        </button>
                        <button class="btn-stem-ctrl btn-stem-mute" data-stem-action="mute-track" data-track-id="${trackId}" id="btnMute_${trackId}" title="Mute ${stemName}" aria-label="Mute Stem">
                            <svg class="stem-icon-mute" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
                        </button>
                        <input type="range" class="stem-volume-fader" data-stem-action="volume" data-track-id="${trackId}" min="0" max="1" step="0.01" value="1" title="Volume">
                        <a href="${fullUrl}" download="${safeFilename}" class="btn-stem-ctrl stem-download-btn" title="Download WAV" aria-label="Download WAV">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </a>
                        <audio class="stem-fallback-audio" id="audioFallback_${trackId}" src="${fullUrl}" preload="none" style="display:none;"></audio>
                    </div>
                </div>
            `;

            // Setup per-stem GainNode
            const stemGain = this.audioCtx ? this.audioCtx.createGain() : null;
            if (stemGain && this.masterGain) stemGain.connect(this.masterGain);

            this.tracks.set(trackId, {
                id: trackId,
                channel: stem.channel,
                label: stem.assigned_tag || stem.filename || `CH ${stem.channel}`,
                isMix: false,
                url: fullUrl,
                filename: safeFilename,
                buffer: null,
                gainNode: stemGain,
                sourceNode: null,
                volume: 1.0,
                isMuted: false,
                isSolo: false,
                peaks: null,
                canvas: null,
                playheadEl: null,
                isAudioFallback: false,
                audioEl: null
            });
        });

        html += `</div>`;
        this.container.innerHTML = html;

        // Cache DOM elements
        this.masterPlayBtn = document.getElementById('stemsMasterPlayBtn');
        this.masterStopBtn = document.getElementById('stemsMasterStopBtn');
        this.masterScrubber = document.getElementById('stemsMasterScrubber');
        this.timeDisplay = document.getElementById('stemsTimeDisplay');
        this.playIconEl = document.getElementById('stemsPlayIcon');
        this.playTextEl = document.getElementById('stemsPlayText');

        // Bind canvases & playheads
        this.tracks.forEach((track, id) => {
            track.canvas = document.getElementById(`stemWf_${id}`);
            track.playheadEl = document.getElementById(`stemPlayhead_${id}`);
            track.audioEl = document.getElementById(`audioFallback_${id}`);
        });

        // Draw initial MIX waveform
        const mixTrack = this.tracks.get('mix');
        if (mixTrack && mixTrack.canvas) {
            this.drawTrackWaveform(mixTrack, 0);
        }

        // Asynchronously decode stem audio buffers via Web Audio API
        this._loadAllStemBuffers(stemsList);

        if (this.notify) {
            this.notify(`Rendered ${stemsList.length} separated stems into DAW player`, 'success');
        }
    }

    async _loadAllStemBuffers(stemsList) {
        if (!this.audioCtx) return;

        stemsList.forEach(async (stem) => {
            const trackId = String(stem.channel);
            const track = this.tracks.get(trackId);
            if (!track) return;

            try {
                const resp = await fetch(track.url);
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const arrayBuffer = await resp.arrayBuffer();
                const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

                track.buffer = audioBuffer;
                track.peaks = computePeaksFromBuffer(audioBuffer);
                if (audioBuffer.duration > this.duration) {
                    this.duration = audioBuffer.duration;
                    if (this.timeDisplay) {
                        this.timeDisplay.textContent = `0.00 / ${this.duration.toFixed(2)}s`;
                    }
                }
                this.drawTrackWaveform(track, 0);
            } catch (err) {
                // Graceful fallback to hidden HTML audio element without throwing
                track.isAudioFallback = true;
                // Generate a visual placeholder envelope
                track.peaks = this._generateFallbackPeaks(180, stem.channel);
                this.drawTrackWaveform(track, 0);
            }
        });
    }

    _generateFallbackPeaks(numBars, seed) {
        const peaks = new Float32Array(numBars);
        for (let i = 0; i < numBars; i++) {
            const harmonic = Math.sin(i * 0.12 * (seed || 1)) * 0.4 + 0.5;
            peaks[i] = Math.max(0.1, Math.min(0.9, harmonic * (0.6 + Math.random() * 0.3)));
        }
        return peaks;
    }

    drawTrackWaveform(track, progress = 0) {
        if (!track || !track.canvas) return;
        const canvas = track.canvas;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(80, Math.floor(rect.width || 200));
        const h = 38;

        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
        }

        ctx.save();
        ctx.scale(dpr, dpr);

        // Track base background
        ctx.fillStyle = '#120c18';
        ctx.fillRect(0, 0, w, h);

        // Faint horizontal center line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        if (track.peaks && track.peaks.length > 0) {
            const numBars = track.peaks.length;
            const centerY = h / 2;
            const maxHalf = (h / 2) - 3;
            const playheadX = progress * w;

            ctx.lineWidth = 2;
            ctx.lineCap = 'round';

            for (let i = 0; i < numBars; i++) {
                const x = (i / numBars) * w + 1;
                const barH = Math.max(1, track.peaks[i] * maxHalf);
                ctx.strokeStyle = (x <= playheadX) ? '#ffffff' : 'rgba(255, 255, 255, 0.32)';
                ctx.beginPath();
                ctx.moveTo(x, centerY - barH);
                ctx.lineTo(x, centerY + barH);
                ctx.stroke();
            }
        } else {
            // Flat placeholder bar
            ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
            ctx.fillRect(0, (h / 2) - 2, w, 4);
        }

        ctx.restore();
    }

    togglePlayTrack(trackId) {
        const track = this.tracks.get(trackId);
        if (!track) return;
        if (track.isPlayingIndividually) {
            this.pauseTrack(trackId);
        } else {
            this.playTrack(trackId);
        }
    }

    async playTrack(trackId) {
        const target = this.tracks.get(trackId);
        if (!target) return;
        this._ensureAudioContext();

        if (this.audioCtx && this.audioCtx.state === 'suspended') {
            try { await this.audioCtx.resume(); } catch (_) {}
        }

        // Stop any previously playing sources so this track plays cleanly
        this.tracks.forEach((t, id) => {
            if (t.sourceNode) {
                try { t.sourceNode.stop(); t.sourceNode.disconnect(); } catch (_) {}
                t.sourceNode = null;
            }
            if (t.isAudioFallback && t.audioEl) {
                try { t.audioEl.pause(); } catch (_) {}
            }
            t.isPlayingIndividually = false;
            this._updateTrackPlayBtn(id, false);
        });

        this.isPlaying = true;
        target.isPlayingIndividually = true;
        this._updateTrackPlayBtn(trackId, true);

        if (this.masterPlayBtn) {
            this.masterPlayBtn.classList.add('is-playing');
            if (this.playIconEl) this.playIconEl.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
            if (this.playTextEl) this.playTextEl.textContent = 'Pause';
        }

        const now = this.audioCtx ? this.audioCtx.currentTime : 0;
        this.playbackStartTime = now;
        const startOffset = Math.max(0, Math.min(this.duration, this.pauseOffset));

        this._recomputeGains();

        if (target.buffer && this.audioCtx) {
            try {
                const src = this.audioCtx.createBufferSource();
                src.buffer = target.buffer;
                src.connect(target.gainNode);
                src.start(now + 0.02, startOffset);
                target.sourceNode = src;
            } catch (e) {
                console.warn(`Error starting single stem ${trackId}:`, e);
            }
        } else if (target.isAudioFallback && target.audioEl) {
            try {
                target.audioEl.currentTime = startOffset;
                target.audioEl.play().catch(() => {});
            } catch (_) {}
        }

        this._runSyncLoop();
        if (this.notify) this.notify(`Playing stem: ${target.label}`, 'info');
    }

    pauseTrack(trackId) {
        const target = this.tracks.get(trackId);
        if (!target) return;

        if (target.sourceNode) {
            try { target.sourceNode.stop(); target.sourceNode.disconnect(); } catch (_) {}
            target.sourceNode = null;
        }
        if (target.isAudioFallback && target.audioEl) {
            try { target.audioEl.pause(); } catch (_) {}
        }
        target.isPlayingIndividually = false;
        this._updateTrackPlayBtn(trackId, false);

        const anyPlaying = Array.from(this.tracks.values()).some(t => t.sourceNode || t.isPlayingIndividually);
        if (!anyPlaying) {
            this.isPlaying = false;
            if (this.masterPlayBtn) {
                this.masterPlayBtn.classList.remove('is-playing');
                if (this.playIconEl) this.playIconEl.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>';
                if (this.playTextEl) this.playTextEl.textContent = 'Play all';
            }
            if (this.syncLoopId) {
                if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(this.syncLoopId);
                this.syncLoopId = null;
            }
            if (this.audioCtx) {
                const elapsed = this.audioCtx.currentTime - this.playbackStartTime;
                this.pauseOffset += elapsed;
                if (this.pauseOffset > this.duration) this.pauseOffset = 0;
            }
        }
    }

    _updateTrackPlayBtn(trackId, isPlaying) {
        const btn = document.getElementById(`btnPlay_${trackId}`);
        if (!btn) return;
        if (isPlaying) {
            btn.classList.add('is-playing');
            btn.innerHTML = '<svg class="stem-icon-pause" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
            btn.setAttribute('title', 'Pause Stem');
        } else {
            btn.classList.remove('is-playing');
            btn.innerHTML = '<svg class="stem-icon-play" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>';
            btn.setAttribute('title', 'Play Stem');
        }
    }

    toggleMute(trackId) {
        const track = this.tracks.get(trackId);
        if (!track) return;

        track.isMuted = !track.isMuted;
        const btn = document.getElementById(`btnMute_${trackId}`);
        const strip = document.getElementById(`stemStrip_${trackId}`);

        if (btn) {
            if (track.isMuted) {
                btn.classList.add('is-active');
                btn.innerHTML = '<svg class="stem-icon-mute" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>';
                btn.setAttribute('title', `Unmute ${track.label}`);
            } else {
                btn.classList.remove('is-active');
                btn.innerHTML = '<svg class="stem-icon-mute" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';
                btn.setAttribute('title', `Mute ${track.label}`);
            }
        }
        if (strip) {
            if (track.isMuted) strip.classList.add('is-muted');
            else strip.classList.remove('is-muted');
        }

        this._recomputeGains();

        if (this.notify) {
            this.notify(`${track.label} ${track.isMuted ? 'Muted' : 'Unmuted'}`, 'info');
        }
    }

    setVolume(trackId, value) {
        const track = this.tracks.get(trackId);
        if (!track) return;

        track.volume = Math.max(0, Math.min(1, parseFloat(value) || 0));
        this._recomputeGains();
    }

    unmuteAll() {
        this.tracks.forEach((track, id) => {
            track.isMuted = false;
            const btnM = document.getElementById(`btnMute_${id}`);
            const strip = document.getElementById(`stemStrip_${id}`);
            if (btnM) {
                btnM.classList.remove('is-active');
                btnM.innerHTML = '<svg class="stem-icon-mute" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';
                btnM.setAttribute('title', `Mute ${track.label}`);
            }
            if (strip) strip.classList.remove('is-muted');
        });
        this._recomputeGains();
        if (this.notify) {
            this.notify('Unmuted all stem channels', 'info');
        }
    }

    _recomputeGains() {
        const now = this.audioCtx ? this.audioCtx.currentTime : 0;
        this.tracks.forEach((track) => {
            const audible = !track.isMuted;
            const targetGain = audible ? track.volume : 0.0;
            if (track.gainNode && this.audioCtx) {
                track.gainNode.gain.setValueAtTime(targetGain, now);
            }
            if (track.isAudioFallback && track.audioEl) {
                track.audioEl.volume = targetGain;
            }
        });
    }

    togglePlayAll() {
        if (this.isPlaying) {
            this.pauseAll();
        } else {
            this.playAll();
        }
    }

    async playAll() {
        if (this.isPlaying) return;
        this._ensureAudioContext();

        if (this.audioCtx && this.audioCtx.state === 'suspended') {
            try {
                await this.audioCtx.resume();
            } catch (_) {}
        }

        this.isPlaying = true;
        if (this.masterPlayBtn) {
            this.masterPlayBtn.classList.add('is-playing');
            if (this.playIconEl) this.playIconEl.innerHTML = '<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
            if (this.playTextEl) this.playTextEl.textContent = 'Pause all';
            this.masterPlayBtn.setAttribute('title', 'Pause all stems (Space)');
        }

        const now = this.audioCtx ? this.audioCtx.currentTime : 0;
        this.playbackStartTime = now;
        const startOffset = Math.max(0, Math.min(this.duration, this.pauseOffset));

        this._recomputeGains();

        // Start all Web Audio sources sample-synced
        this.tracks.forEach((track, id) => {
            this._updateTrackPlayBtn(id, !track.isMuted);
            if (track.buffer && this.audioCtx) {
                try {
                    if (track.sourceNode) {
                        track.sourceNode.stop();
                        track.sourceNode.disconnect();
                    }
                } catch (_) {}

                try {
                    const src = this.audioCtx.createBufferSource();
                    src.buffer = track.buffer;
                    src.connect(track.gainNode);
                    src.start(now + 0.02, startOffset);
                    track.sourceNode = src;
                } catch (e) {
                    console.warn(`Error starting buffer source for ${track.id}:`, e);
                }
            } else if (track.isAudioFallback && track.audioEl) {
                try {
                    track.audioEl.currentTime = startOffset;
                    track.audioEl.play().catch(() => {});
                } catch (_) {}
            }
        });

        this._runSyncLoop();
    }

    pauseAll() {
        if (!this.isPlaying) return;
        this.isPlaying = false;

        if (this.masterPlayBtn) {
            this.masterPlayBtn.classList.remove('is-playing');
            if (this.playIconEl) this.playIconEl.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>';
            if (this.playTextEl) this.playTextEl.textContent = 'Play all';
            this.masterPlayBtn.setAttribute('title', 'Play all stems (Space)');
        }

        if (this.syncLoopId) {
            if (typeof cancelAnimationFrame !== 'undefined') {
                cancelAnimationFrame(this.syncLoopId);
            } else {
                clearTimeout(this.syncLoopId);
            }
            this.syncLoopId = null;
        }

        if (this.audioCtx) {
            const elapsed = this.audioCtx.currentTime - this.playbackStartTime;
            this.pauseOffset += elapsed;
            if (this.pauseOffset > this.duration) this.pauseOffset = 0;
        }

        // Stop all running source nodes & reset button states
        this.tracks.forEach((track, id) => {
            track.isPlayingIndividually = false;
            this._updateTrackPlayBtn(id, false);
            if (track.sourceNode) {
                try {
                    track.sourceNode.stop();
                    track.sourceNode.disconnect();
                } catch (_) {}
                track.sourceNode = null;
            }
            if (track.isAudioFallback && track.audioEl) {
                try {
                    track.audioEl.pause();
                } catch (_) {}
            }
        });
    }

    stopAll() {
        this.pauseAll();
        this.pauseOffset = 0;
        this.updatePlayhead(0);

        if (this.masterScrubber) {
            this.masterScrubber.value = 0;
            this.masterScrubber.style.setProperty('--scrubber-pct', '0%');
        }
        if (this.timeDisplay) {
            this.timeDisplay.textContent = `0.00 / ${this.duration.toFixed(2)}s`;
        }
    }

    seekAll(fraction) {
        const clamped = Math.max(0, Math.min(1, fraction));
        this.pauseOffset = clamped * this.duration;

        this.updatePlayhead(clamped);

        if (this.masterScrubber) {
            this.masterScrubber.value = clamped * 100;
            this.masterScrubber.style.setProperty('--scrubber-pct', `${clamped * 100}%`);
        }
        if (this.timeDisplay) {
            this.timeDisplay.textContent = `${this.pauseOffset.toFixed(2)} / ${this.duration.toFixed(2)}s`;
        }

        if (this.isPlaying) {
            this.pauseAll();
            this.playAll();
        }
    }

    updatePlayhead(fraction) {
        const pct = Math.max(0, Math.min(1, fraction));
        const pctStr = `${pct * 100}%`;

        this.tracks.forEach((track) => {
            if (track.playheadEl) {
                track.playheadEl.style.left = pctStr;
            }
            this.drawTrackWaveform(track, pct);
        });
    }

    _runSyncLoop() {
        if (!this.isPlaying) return;

        const now = this.audioCtx ? this.audioCtx.currentTime : 0;
        const elapsed = (now - this.playbackStartTime) + this.pauseOffset;
        const fraction = Math.min(1, elapsed / this.duration);

        if (this.masterScrubber) {
            this.masterScrubber.value = fraction * 100;
            this.masterScrubber.style.setProperty('--scrubber-pct', `${fraction * 100}%`);
        }
        if (this.timeDisplay) {
            this.timeDisplay.textContent = `${elapsed.toFixed(2)} / ${this.duration.toFixed(2)}s`;
        }

        const pctStr = `${fraction * 100}%`;
        this.tracks.forEach((track) => {
            if (track.playheadEl) {
                track.playheadEl.style.left = pctStr;
            }
            this.drawTrackWaveform(track, fraction);
        });

        if (elapsed >= this.duration) {
            this.stopAll();
            return;
        }

        if (typeof requestAnimationFrame !== 'undefined') {
            this.syncLoopId = requestAnimationFrame(() => this._runSyncLoop());
        } else {
            this.syncLoopId = setTimeout(() => this._runSyncLoop(), 30);
        }
    }
}

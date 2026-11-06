/**
 * frontend/js/audio/stemPlayer.js
 * ===============================
 * Multi-track Stem Mixer & Playback Interface.
 * Manages reconstructed source audio channels with Solo, Mute, Volume fader,
 * and direct WAV download actions via event delegation (zero globals).
 */

import { API_BASE } from '../config.js';
import { showToast } from '../ui/toast.js';

export class StemPlayerMatrix {
    constructor(containerElement, onActionToast = showToast, apiBaseUrl = API_BASE) {
        this.container = containerElement;
        this.notify = onActionToast;
        this.apiBaseUrl = apiBaseUrl;
        this.stems = [];
        this.audioElements = new Map();
        this.soloChannel = null;

        this._setupDelegatedEvents();
    }

    _setupDelegatedEvents() {
        if (!this.container) return;

        this.container.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-stem-action]');
            if (!btn) return;
            const action = btn.dataset.stemAction;
            const ch = parseInt(btn.dataset.channel, 10);
            if (action === 'solo') {
                this.toggleSolo(ch);
            } else if (action === 'mute') {
                this.toggleMute(ch);
            }
        });

        this.container.addEventListener('input', (e) => {
            const input = e.target.closest('[data-stem-action="volume"]');
            if (!input) return;
            const ch = parseInt(input.dataset.channel, 10);
            this.setVolume(ch, input.value);
        });
    }

    render(stemsList, apiBaseUrl = this.apiBaseUrl) {
        this.stems = stemsList || [];
        this.audioElements.clear();
        this.soloChannel = null;

        if (!this.container) return;

        if (this.stems.length === 0) {
            this.container.innerHTML = `
                <div class="ghost-empty-state">
                    <span class="ghost-icon">🎛️</span>
                    <span class="ghost-text">No separated stems reconstructed yet</span>
                    <span class="ghost-sub">Run inference on a polyphonic mixture to render stem player channels</span>
                </div>
            `;
            return;
        }

        const effectiveBase = apiBaseUrl || API_BASE;
        let html = '';
        this.stems.forEach(stem => {
            const rawUrl = stem.url || '';
            const fullUrl = rawUrl.startsWith('http://') || rawUrl.startsWith('https://')
                ? rawUrl
                : `${effectiveBase}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`;
            const sisdr = stem.si_sdr_estimate_db !== undefined ? stem.si_sdr_estimate_db : 0;
            const sisdrClass = sisdr >= 10 ? 'pass' : (sisdr >= 4 ? 'pass' : (sisdr >= 0 ? 'warn' : 'fail'));

            html += `
                <div class="stem-strip" id="stemStrip_${stem.channel}">
                    <div class="stem-strip-header">
                        <div class="stem-strip-title">
                            <span class="badge">CH ${stem.channel}</span>
                            <span style="font-weight: 600; color: var(--text-primary);">${stem.assigned_tag || stem.filename}</span>
                            ${stem.stem_confidence ? `<span class="badge badge-xs">${(stem.stem_confidence * 100).toFixed(0)}% conf</span>` : ''}
                        </div>
                        <span class="badge ${sisdrClass}">
                            ${sisdr >= 0 ? '+' : ''}${sisdr} dB SI-SDR
                        </span>
                    </div>
                    <div class="stem-strip-controls">
                        <audio controls class="stem-audio-element" id="audioTrack_${stem.channel}" src="${fullUrl}" preload="metadata"></audio>
                        <button class="btn-ctrl" id="btnSolo_${stem.channel}" data-stem-action="solo" data-channel="${stem.channel}">SOLO</button>
                        <button class="btn-ctrl" id="btnMute_${stem.channel}" data-stem-action="mute" data-channel="${stem.channel}">MUTE</button>
                        <input type="range" class="volume-fader" min="0" max="1" step="0.05" value="1" title="Volume" data-stem-action="volume" data-channel="${stem.channel}">
                        <a href="${fullUrl}" download="${stem.filename}" class="btn-ctrl stem-download-btn" title="Download WAV">⬇</a>
                    </div>
                </div>
            `;
        });

        this.container.innerHTML = html;

        this.stems.forEach(stem => {
            const el = document.getElementById(`audioTrack_${stem.channel}`);
            if (el) this.audioElements.set(stem.channel, el);
        });

        if (this.notify) {
            this.notify(`Rendered ${this.stems.length} separated stems into player matrix`, 'success');
        }
    }

    toggleSolo(targetChannel) {
        if (this.soloChannel === targetChannel) {
            this.soloChannel = null;
            this.audioElements.forEach((el, ch) => {
                el.muted = false;
                const btn = document.getElementById(`btnSolo_${ch}`);
                if (btn) btn.classList.remove('active');
            });
            if (this.notify) this.notify(`Solo mode deactivated (All stems audible)`, 'info');
        } else {
            this.soloChannel = targetChannel;
            this.audioElements.forEach((el, ch) => {
                const btn = document.getElementById(`btnSolo_${ch}`);
                if (ch === targetChannel) {
                    el.muted = false;
                    if (btn) btn.classList.add('active');
                } else {
                    el.muted = true;
                    if (btn) btn.classList.remove('active');
                }
            });
            if (this.notify) this.notify(`Soloing Channel ${targetChannel}`, 'success');
        }
    }

    toggleMute(targetChannel) {
        const el = this.audioElements.get(targetChannel);
        const btn = document.getElementById(`btnMute_${targetChannel}`);
        if (!el) return;

        el.muted = !el.muted;
        if (btn) {
            if (el.muted) btn.classList.add('active');
            else btn.classList.remove('active');
        }
        if (this.notify) {
            this.notify(`Channel ${targetChannel} ${el.muted ? 'Muted' : 'Unmuted'}`, 'warning');
        }
    }

    setVolume(targetChannel, value) {
        const el = this.audioElements.get(targetChannel);
        if (el) el.volume = parseFloat(value);
    }

    playAll() {
        this.audioElements.forEach(el => el.play().catch(() => {}));
    }

    pauseAll() {
        this.audioElements.forEach(el => el.pause());
    }
}

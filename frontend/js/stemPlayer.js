/**
 * frontend/js/stemPlayer.js
 * =========================
 * Multi-Track Stem Player Matrix Controller
 * Manages concurrent playback, Solo/Mute logic, and Volume Faders
 * Member 1 (Frontend UI & Client Engine)
 */

export class StemPlayerMatrix {
    constructor(containerElement, onActionToast) {
        this.container = containerElement;
        this.notify = onActionToast;
        this.stems = [];
        this.audioElements = new Map();
        this.soloChannel = null;
    }

    /**
     * Renders separated stems into the UI matrix
     */
    render(stemsList, apiBaseUrl) {
        this.stems = stemsList || [];
        this.audioElements.clear();
        this.soloChannel = null;

        if (!this.container) return;

        if (this.stems.length === 0) {
            this.container.innerHTML = `
                <div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 32px 0;">
                    No separated audio stems available. Run inference on a polyphonic mixture.
                </div>
            `;
            return;
        }

        let html = '';
        this.stems.forEach(stem => {
            const rawUrl = stem.url || '';
            const fullUrl = rawUrl.startsWith('http') ? rawUrl : `${apiBaseUrl}${rawUrl}`;
            const sisdr = stem.si_sdr_estimate_db !== undefined ? stem.si_sdr_estimate_db : 0;
            const sisdrClass = sisdr >= 10 ? 'pass' : (sisdr >= 4 ? 'pass' : (sisdr >= 0 ? 'warn' : 'fail'));

            html += `
                <div class="stem-strip" id="stemStrip_${stem.channel}">
                    <div class="stem-strip-header">
                        <div class="stem-strip-title">
                            <span class="badge" style="background: rgba(59, 130, 246, 0.15); color: var(--accent);">CH ${stem.channel}</span>
                            <span style="font-weight: 600; color: var(--text-primary);">${stem.assigned_tag || stem.filename}</span>
                            ${stem.stem_confidence ? `<span class="badge" style="font-size: 10px; color: var(--text-secondary);">${(stem.stem_confidence * 100).toFixed(0)}% conf</span>` : ''}
                        </div>
                        <span class="badge ${sisdrClass}">
                            ${sisdr >= 0 ? '+' : ''}${sisdr} dB SI-SDR
                        </span>
                    </div>
                    <div class="stem-strip-controls">
                        <audio controls class="stem-audio-element" id="audioTrack_${stem.channel}" src="${fullUrl}" preload="metadata"></audio>
                        <button class="btn-ctrl" id="btnSolo_${stem.channel}" onclick="window.stemMatrix.toggleSolo(${stem.channel})">SOLO</button>
                        <button class="btn-ctrl" id="btnMute_${stem.channel}" onclick="window.stemMatrix.toggleMute(${stem.channel})">MUTE</button>
                        <input type="range" class="volume-fader" min="0" max="1" step="0.05" value="1" title="Volume" oninput="window.stemMatrix.setVolume(${stem.channel}, this.value)">
                        <a href="${fullUrl}" download="${stem.filename}" class="btn-ctrl" title="Download WAV" style="text-decoration:none;">⬇</a>
                    </div>
                </div>
            `;
        });

        this.container.innerHTML = html;

        // Register audio elements
        this.stems.forEach(stem => {
            const el = document.getElementById(`audioTrack_${stem.channel}`);
            if (el) this.audioElements.set(stem.channel, el);
        });

        if (this.notify) {
            this.notify(`Rendered ${this.stems.length} separated stems into player matrix`, 'success');
        }
    }

    /**
     * Toggles exclusive solo auditioning
     */
    toggleSolo(targetChannel) {
        if (this.soloChannel === targetChannel) {
            // Un-solo everything
            this.soloChannel = null;
            this.audioElements.forEach((el, ch) => {
                el.muted = false;
                const btn = document.getElementById(`btnSolo_${ch}`);
                if (btn) btn.classList.remove('active');
            });
            if (this.notify) this.notify(`Solo mode deactivated (All stems audible)`, 'info');
        } else {
            // Solo selected channel
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

    /**
     * Toggles mute state for a channel
     */
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

    /**
     * Adjusts playback gain for an individual stem
     */
    setVolume(targetChannel, value) {
        const el = this.audioElements.get(targetChannel);
        if (el) el.volume = parseFloat(value);
    }

    /**
     * Starts synchronous playback across all stems
     */
    playAll() {
        this.audioElements.forEach(el => el.play().catch(() => {}));
    }

    /**
     * Pauses all stem streams
     */
    pauseAll() {
        this.audioElements.forEach(el => el.pause());
    }
}

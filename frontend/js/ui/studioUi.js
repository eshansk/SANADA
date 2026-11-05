/**
 * frontend/js/ui/studioUi.js
 * ==========================
 * Presentation-Only Enhancements for SANADA Studio & Interactive Pages
 * Handles Complexity Ceiling gauge, Copy JSON, Keyboard shortcuts, BibTeX copy, and Mobile menu
 */

export function initStudioUi(toastHandler) {
    // 1. Complexity Ceiling Gauge Watcher
    const kHeroNumber = document.getElementById('kHeroNumber');
    const ceilingGauge = document.getElementById('ceilingGauge');
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugeLabel = document.getElementById('gaugeLabel');
    const gaugeDot = document.getElementById('gaugeDot');

    function updateCeilingGauge(kVal) {
        if (!gaugeFill || !gaugeLabel || !gaugeDot) return;
        const k = parseInt(kVal, 10);
        if (isNaN(k)) {
            gaugeFill.style.width = '0%';
            gaugeFill.style.background = 'rgba(255, 255, 255, 0.2)';
            gaugeDot.className = 'status-dot';
            gaugeLabel.textContent = 'Awaiting inference prediction to evaluate Complexity Ceiling';
            return;
        }

        if (k <= 3) {
            gaugeFill.style.width = `${Math.min(100, (k / 9) * 100)}%`;
            gaugeFill.style.background = 'var(--success)';
            gaugeDot.className = 'status-dot online';
            gaugeLabel.innerHTML = `<strong>K = ${k} (Safe Zone)</strong>: Clean separation expected (&gt; 8.0 dB SI-SDR).`;
        } else if (k === 4) {
            gaugeFill.style.width = '44%';
            gaugeFill.style.background = 'var(--warning)';
            gaugeDot.className = 'status-dot';
            gaugeDot.style.background = 'var(--warning)';
            gaugeLabel.innerHTML = `<strong>K = 4 (Ceiling Boundary)</strong>: Speech separation usable (~4.2 dB SI-SDR); near breakdown limit.`;
        } else {
            gaugeFill.style.width = `${Math.min(100, (k / 9) * 100)}%`;
            gaugeFill.style.background = 'var(--danger)';
            gaugeDot.className = 'status-dot';
            gaugeDot.style.background = 'var(--danger)';
            gaugeLabel.innerHTML = `<strong>K = ${k} (Ceiling Exceeded)</strong>: Separation is unreliable (&lt; 0 dB SI-SDR). High permutation ambiguity.`;
        }
    }

    // Observe changes to kHeroNumber
    if (kHeroNumber) {
        const observer = new MutationObserver(() => {
            updateCeilingGauge(kHeroNumber.textContent.trim());
        });
        observer.observe(kHeroNumber, { childList: true, characterData: true, subtree: true });
    }

    // 2. Copy JSON Results Button
    const copyJsonBtn = document.getElementById('copyJsonBtn');
    if (copyJsonBtn) {
        copyJsonBtn.addEventListener('click', () => {
            const kNumber = document.getElementById('kHeroNumber')?.textContent.trim() || '—';
            const kSub = document.getElementById('kHeroSub')?.textContent.trim() || '';
            const kContinuous = document.getElementById('kContinuousVal')?.textContent.trim() || '—';
            const filename = document.getElementById('metaFilename')?.textContent.trim() || 'unnamed.wav';

            // Extract visible active tags
            const tags = [];
            document.querySelectorAll('#probBarsContainer .prob-row').forEach(row => {
                const label = row.querySelector('.prob-meta span:first-child')?.textContent.trim();
                const pct = row.querySelector('.prob-meta span:last-child')?.textContent.trim();
                if (label && pct) tags.push({ class: label.replace('●', '').trim(), probability: pct });
            });

            // Extract visible stem channels
            const stems = [];
            document.querySelectorAll('#stemsContainer .stem-strip').forEach(strip => {
                const title = strip.querySelector('.stem-strip-title')?.textContent.trim();
                const sisdr = strip.querySelector('.stem-strip-header .badge')?.textContent.trim();
                stems.push({ title, sisdr });
            });

            const payload = {
                metadata: {
                    filename,
                    evaluated_at: new Date().toISOString(),
                    standard_compliance: "16kHz Mono 8.0s"
                },
                task_1_counting: {
                    estimated_k: kNumber,
                    confidence: kSub,
                    continuous_raw: kContinuous
                },
                task_2_tagging: tags,
                task_3_separation: stems
            };

            navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
                .then(() => {
                    if (toastHandler) toastHandler('Inference results copied to clipboard as JSON!', 'success');
                })
                .catch(() => {
                    if (toastHandler) toastHandler('Failed to copy JSON to clipboard', 'warning');
                });
        });
    }

    // 3. Keyboard Shortcuts (Space = Play/Pause, M = Mute)
    window.addEventListener('keydown', (e) => {
        // Ignore if user is typing into an input, select, or textarea
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (['input', 'textarea', 'select'].includes(activeTag)) return;

        if (e.code === 'Space') {
            e.preventDefault();
            const playMasterBtn = document.getElementById('playMasterBtn');
            if (playMasterBtn && !playMasterBtn.disabled) {
                playMasterBtn.click();
            }
        } else if (e.code === 'KeyM') {
            e.preventDefault();
            // Trigger mute on first active stem or solo
            const muteBtn = document.querySelector('.stem-strip .btn-ctrl[id^="btnMute_"]');
            if (muteBtn) {
                muteBtn.click();
            } else if (toastHandler) {
                toastHandler('Mute shortcut (M) active when stems are loaded', 'info');
            }
        }
    });

    // 4. BibTeX Citation Copy Button
    const copyBibtexBtn = document.getElementById('copyBibtexBtn');
    if (copyBibtexBtn) {
        copyBibtexBtn.addEventListener('click', () => {
            const bibtexEl = document.getElementById('bibtexCode');
            if (!bibtexEl) return;
            navigator.clipboard.writeText(bibtexEl.textContent.trim())
                .then(() => {
                    if (toastHandler) toastHandler('BibTeX citation copied to clipboard!', 'success');
                })
                .catch(() => {
                    if (toastHandler) toastHandler('Unable to copy citation', 'warning');
                });
        });
    }

    // 5. Mobile Navbar Collapsing Menu Toggle
    const navToggle = document.getElementById('navToggle');
    const navTabs = document.querySelector('.nav-tabs');
    if (navToggle && navTabs) {
        navToggle.addEventListener('click', () => {
            navTabs.classList.toggle('open');
            navToggle.classList.toggle('active');
        });

        // Close menu when a navigation item is clicked on mobile
        navTabs.querySelectorAll('.tab-link').forEach(link => {
            link.addEventListener('click', () => {
                navTabs.classList.remove('open');
                navToggle.classList.remove('active');
            });
        });
    }
}

// Auto-boot if loaded directly as script
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        initStudioUi(window.showToast);
    });
}

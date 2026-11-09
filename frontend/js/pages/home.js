/**
 * frontend/js/pages/home.js
 * =========================
 * Homepage view logic, interactive call-to-action bindings,
 * and the Complexity Ceiling interactive explorer.
 */

import { switchRoute } from '../ui/router.js';
import { PAPER_BENCHMARKS } from '../data/paperBenchmarks.js';

function initCeilingExplorer() {
    const kButtons = document.querySelectorAll('#explorerKToggle .seg-btn');
    const sisdrNumEl = document.getElementById('explorerSisdrNum');
    const sisdrStatusEl = document.getElementById('explorerSisdrStatus');
    const barFillEl = document.getElementById('explorerBarFill');
    const dotsRowEl = document.getElementById('explorerDotsRow');
    const dotsCountEl = document.getElementById('explorerDotsCount');

    if (!kButtons.length || !sisdrNumEl || !barFillEl || !dotsRowEl) return;

    const benchmarks = PAPER_BENCHMARKS.separationSisdr.orgMixFixed;

    function renderK(k) {
        const item = benchmarks.find(b => b.k === k) || benchmarks[0];
        const val = item.sisdrDb;

        // 1. Text Readout
        sisdrNumEl.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)} dB`;
        if (sisdrStatusEl) {
            sisdrStatusEl.textContent = item.status;
            sisdrStatusEl.className = `explorer-metric-status ${val >= 4.0 ? 'status-pass' : (val >= 0 ? 'status-warn' : 'status-fail')}`;
        }

        // 2. Horizontal bar crossing vertical 0 dB line (scale: -10 to +12 dB = span 22)
        const zeroPct = (10 / 22) * 100; // ~45.45%
        if (val >= 0) {
            const widthPct = Math.min(100 - zeroPct, (val / 22) * 100);
            barFillEl.style.left = `${zeroPct}%`;
            barFillEl.style.width = `${widthPct}%`;
            barFillEl.style.background = '#ffffff';
            barFillEl.style.boxShadow = '0 0 16px rgba(255, 255, 255, 0.4)';
        } else {
            const widthPct = Math.min(zeroPct, (Math.abs(val) / 22) * 100);
            barFillEl.style.left = `${zeroPct - widthPct}%`;
            barFillEl.style.width = `${widthPct}%`;
            barFillEl.style.background = 'rgba(255, 255, 255, 0.35)';
            barFillEl.style.boxShadow = 'none';
        }

        // 3. Row of K dots going from solid to dissolving
        let dotsHtml = '';
        for (let i = 1; i <= k; i++) {
            let dotClass = 'dot-solid';
            let title = `Stem ${i} (Clean monaural reconstruction)`;
            if (i === 4) {
                dotClass = 'dot-ceiling';
                title = `Stem 4 (Ceiling boundary ~4.2 dB)`;
            } else if (i === 5) {
                dotClass = 'dot-dissolve-1';
                title = `Stem 5 (Degraded < 0 dB)`;
            } else if (i === 6) {
                dotClass = 'dot-dissolve-2';
                title = `Stem 6 (High permutation noise)`;
            } else if (i >= 7) {
                dotClass = 'dot-dissolve-3';
                title = `Stem ${i} (Intractable factorial ambiguity)`;
            }
            dotsHtml += `<span class="explorer-k-dot ${dotClass}" title="${title}"></span>`;
        }
        dotsRowEl.innerHTML = dotsHtml;
        if (dotsCountEl) dotsCountEl.textContent = k;
    }

    kButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            kButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const kVal = parseInt(btn.dataset.k, 10) || 2;
            renderK(kVal);
        });
    });

    // Initial render at K=2
    renderK(2);
}

export function initHomePage() {
    // Intercept in-page hash links to ensure fast router switching
    const homeLinks = document.querySelectorAll('#pageHome a[href^="#/"]');
    homeLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const hash = link.getAttribute('href');
            if (hash) {
                e.preventDefault();
                switchRoute(hash.replace('#/', ''));
            }
        });
    });

    // Initialize interactive complexity ceiling explorer
    initCeilingExplorer();
}

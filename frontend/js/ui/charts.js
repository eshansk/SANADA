/**
 * frontend/js/ui/charts.js
 * ========================
 * Benchmark Table and Dynamic SVG Chart Generators.
 * Generates mathematically scaled, proportional K-axis charts from PAPER_BENCHMARKS
 * or /api/stats telemetry, with white-tier styling, 0 dB dashed boundary, and hover tooltips.
 */

import { PAPER_BENCHMARKS } from '../data/paperBenchmarks.js';

export const ChartHelper = {
    getBenchmarks() {
        return PAPER_BENCHMARKS;
    },

    formatNumber(num, decimals = 2) {
        if (num === null || num === undefined || isNaN(num)) return '—';
        return Number(num).toFixed(decimals);
    },

    renderSisdrRows(rows) {
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.separationSisdr.orgMixFixed;
        return data.map(row => {
            const val = row.si_sdr_db !== undefined ? row.si_sdr_db : row.sisdrDb;
            const statusBadge = val >= 4.0 ? 'pass' : (val >= 0 ? 'warn' : 'fail');
            return `
                <tr>
                    <td><strong>K = ${row.density_k || row.k || '—'}</strong></td>
                    <td>${row.domain || 'Org-Mix Fixed'}</td>
                    <td class="font-mono" style="font-weight: 600;">${val >= 0 ? '+' : ''}${Number(val).toFixed(2)} dB</td>
                    <td><span class="badge ${statusBadge}">${row.status || (val >= 0 ? 'Usable' : 'Degraded')}</span></td>
                </tr>
            `;
        }).join('');
    },

    renderCountingRows(rows) {
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.counting.results;
        return data.map(row => `
            <tr>
                <td><strong>${row.domain || row.corpus}</strong></td>
                <td class="font-mono">${Number(row.mae).toFixed(4)}</td>
                <td class="font-mono">${Number(row.rmse).toFixed(4)}</td>
                <td><span class="badge pass">${row.exact_acc_pct || row.exactAccuracyPct}%</span></td>
                <td><span class="badge pass">${row.relaxed_acc_pct || row.plusMinusOneAccuracyPct}%</span></td>
            </tr>
        `).join('');
    },

    renderTaggingRows(rows) {
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.tagging.results;
        return data.map(row => `
            <tr>
                <td><strong>${row.domain || row.corpus}</strong></td>
                <td>${row.classes || '32 Classes'}</td>
                <td class="font-mono">${Number(row.map_macro || row.mAP).toFixed(4)}</td>
                <td class="font-mono">${Number(row.f1_macro || row.macroF1).toFixed(4)}</td>
                <td class="font-mono">${Number(row.roc_auc || row.rocAuc).toFixed(4)}</td>
            </tr>
        `).join('');
    },

    /**
     * DYNAMIC CHART 1: Separation SI-SDR vs Density (K)
     * Mathematical proportional K-spacing on X-axis (K = 2 to 9, range = 7)
     * Y-scale: -10 dB to +12 dB (range = 22 dB)
     * Dashed 0 dB failure line labeled in white
     */
    renderSisdrChart(container, rows) {
        if (!container) return;
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.separationSisdr.orgMixFixed;

        const W = 360;
        const H = 220;
        const padL = 45;
        const padR = 24;
        const padT = 28;
        const padB = 34;

        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        // X scale: K min 2, K max 9 (proportional spacing)
        const kMin = 2;
        const kMax = 9;
        const getX = (k) => padL + ((k - kMin) / (kMax - kMin)) * plotW;

        // Y scale: -10 dB to +12 dB
        const yMin = -10;
        const yMax = 12;
        const getY = (db) => (H - padB) - ((db - yMin) / (yMax - yMin)) * plotH;

        const zeroY = getY(0);

        // Generate points for polyline
        const points = data.map(d => {
            const k = d.k || d.density_k;
            const db = d.sisdrDb !== undefined ? d.sisdrDb : d.si_sdr_db;
            return { k, db, x: getX(k), y: getY(db), status: d.status || '' };
        });

        const polylinePoints = points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

        // Data point markers
        const markersHtml = points.map(p => `
            <g class="chart-point-group" tabindex="0">
                <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5"
                        fill="${p.db >= 0 ? '#ffffff' : 'rgba(255,255,255,0.45)'}"
                        stroke="#000000" stroke-width="1.5" />
                <text x="${p.x.toFixed(1)}" y="${(p.y - 7).toFixed(1)}"
                      text-anchor="middle"
                      fill="#ffffff"
                      font-family="var(--font-mono, monospace)"
                      font-size="8.5"
                      font-weight="600">${p.db >= 0 ? '+' : ''}${p.db.toFixed(1)}</text>
                <title>K = ${p.k}: ${p.db >= 0 ? '+' : ''}${p.db.toFixed(2)} dB (${p.status || (p.db >= 0 ? 'Usable' : 'Degraded')})</title>
            </g>
        `).join('');

        // X-axis tick labels
        const xTicksHtml = points.map(p => `
            <text x="${p.x.toFixed(1)}" y="${H - 12}" text-anchor="middle"
                  fill="rgba(255,255,255,0.6)" font-family="var(--font-mono, monospace)" font-size="9">K=${p.k}</text>
        `).join('');

        container.innerHTML = `
            <svg class="chart-svg" viewBox="0 0 ${W} ${H}" fill="none" role="img" aria-label="Separation SI-SDR vs Density K line chart">
                <!-- Axes -->
                <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
                <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>

                <!-- Y Axis Labels -->
                <text x="${padL - 6}" y="${getY(12) + 3}" text-anchor="end" fill="rgba(255,255,255,0.45)" font-family="var(--font-mono, monospace)" font-size="8">+12 dB</text>
                <text x="${padL - 6}" y="${zeroY + 3}" text-anchor="end" fill="rgba(255,255,255,0.65)" font-family="var(--font-mono, monospace)" font-size="8" font-weight="600">0 dB</text>
                <text x="${padL - 6}" y="${getY(-10) + 3}" text-anchor="end" fill="rgba(255,255,255,0.45)" font-family="var(--font-mono, monospace)" font-size="8">-10 dB</text>

                <!-- Dashed 0 dB failure line labeled in white -->
                <line x1="${padL}" y1="${zeroY.toFixed(1)}" x2="${W - padR}" y2="${zeroY.toFixed(1)}"
                      stroke="#ffffff" stroke-width="1.2" stroke-dasharray="4 4" stroke-opacity="0.8"/>
                <text x="${W - padR - 4}" y="${(zeroY - 6).toFixed(1)}" text-anchor="end"
                      fill="#ffffff" font-family="var(--font-mono, monospace)" font-size="8.5" font-weight="600" opacity="0.9">0 dB Failure Line</text>

                <!-- Polyline connecting data points -->
                <polyline points="${polylinePoints}" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linejoin="round"/>

                <!-- Markers & Hover Tooltips -->
                ${markersHtml}

                <!-- X Axis Labels -->
                ${xTicksHtml}
            </svg>
        `;
    },

    /**
     * DYNAMIC CHART 2: Counting Accuracy Bars & MAE Callouts
     * Conformer MSE Exact accuracy (%) & MAE across corpora.
     */
    renderCountingChart(container, rows) {
        if (!container) return;
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.counting.results;

        const W = 360;
        const H = 220;
        const padL = 40;
        const padR = 20;
        const padT = 24;
        const padB = 34;

        const plotH = H - padT - padB;
        const barW = 44;

        const positions = [90, 190, 290];
        const whiteOpacities = ['#ffffff', 'rgba(255,255,255,0.65)', 'rgba(255,255,255,0.3)'];

        const barsHtml = data.map((d, i) => {
            const acc = d.exactAccuracyPct || d.exact_acc_pct || 0;
            const barH = (acc / 100) * plotH;
            const x = positions[i] - (barW / 2);
            const y = (H - padB) - barH;
            const color = whiteOpacities[i] || '#ffffff';
            const corpusName = d.corpus || d.domain;
            const maeVal = Number(d.mae).toFixed(2);

            return `
                <g class="chart-bar-group" tabindex="0">
                    <rect x="${x}" y="${y.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}"
                          rx="4" fill="${color}" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
                    <text x="${positions[i]}" y="${(y - 6).toFixed(1)}" text-anchor="middle"
                          fill="#ffffff" font-family="var(--font-mono, monospace)" font-size="9" font-weight="600">${acc.toFixed(1)}%</text>
                    <text x="${positions[i]}" y="${(y + 16 < H - padB - 4) ? (y + 16).toFixed(1) : (y - 18).toFixed(1)}"
                          text-anchor="middle"
                          fill="${i === 0 ? '#000000' : '#ffffff'}"
                          font-family="var(--font-mono, monospace)"
                          font-size="8" font-weight="600">MAE ${maeVal}</text>
                    <text x="${positions[i]}" y="${H - 12}" text-anchor="middle"
                          fill="rgba(255,255,255,0.6)" font-family="var(--font-body, sans-serif)" font-size="11">${corpusName}</text>
                    <title>${corpusName}: Exact Acc ${acc}% | MAE ${Number(d.mae).toFixed(4)} | RMSE ${Number(d.rmse).toFixed(4)}</title>
                </g>
            `;
        }).join('');

        container.innerHTML = `
            <svg class="chart-svg" viewBox="0 0 ${W} ${H}" fill="none" role="img" aria-label="Counting Performance Decay bar chart">
                <!-- Axes -->
                <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
                <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>

                <!-- Y-Ticks -->
                <text x="${padL - 6}" y="${padT + 4}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">100%</text>
                <text x="${padL - 6}" y="${padT + plotH / 2 + 3}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">50%</text>
                <text x="${padL - 6}" y="${H - padB + 3}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">0%</text>

                <!-- Bars & Tooltips -->
                ${barsHtml}
            </svg>
        `;
    },

    /**
     * DYNAMIC CHART 3: Multi-Label ASL Tagging Grouped Bars
     * mAP, Macro F1, and ROC-AUC by white tier with top legend.
     */
    renderTaggingChart(container, rows) {
        if (!container) return;
        const data = (rows && rows.length > 0) ? rows : PAPER_BENCHMARKS.tagging.results;

        const W = 360;
        const H = 220;
        const padL = 40;
        const padR = 20;
        const padT = 38;
        const padB = 34;

        const plotH = H - padT - padB;
        const barW = 14;
        const groupGap = 90;
        const startX = 64;

        const tiers = [
            { key: 'map', label: 'mAP', fill: '#ffffff' },
            { key: 'f1', label: 'Macro F1', fill: 'rgba(255,255,255,0.65)' },
            { key: 'auc', label: 'ROC-AUC', fill: 'rgba(255,255,255,0.30)' }
        ];

        // Legend at the top inside chart
        const legendHtml = `
            <g class="chart-legend-svg" aria-hidden="true">
                <rect x="70" y="10" width="10" height="10" rx="2" fill="#ffffff"/>
                <text x="84" y="18" fill="rgba(255,255,255,0.85)" font-family="var(--font-mono, monospace)" font-size="9">mAP</text>

                <rect x="145" y="10" width="10" height="10" rx="2" fill="rgba(255,255,255,0.65)"/>
                <text x="159" y="18" fill="rgba(255,255,255,0.85)" font-family="var(--font-mono, monospace)" font-size="9">Macro F1</text>

                <rect x="235" y="10" width="10" height="10" rx="2" fill="rgba(255,255,255,0.30)"/>
                <text x="249" y="18" fill="rgba(255,255,255,0.85)" font-family="var(--font-mono, monospace)" font-size="9">ROC-AUC</text>
            </g>
        `;

        const groupsHtml = data.map((d, gi) => {
            const center = startX + gi * groupGap;
            const mapVal = d.mAP !== undefined ? d.mAP : d.map_macro;
            const f1Val = d.macroF1 !== undefined ? d.macroF1 : d.f1_macro;
            const aucVal = d.rocAuc !== undefined ? d.rocAuc : d.roc_auc;

            const values = [mapVal, f1Val, aucVal];
            const corpusName = d.corpus || d.domain;

            const bars = values.map((val, bi) => {
                const barH = (val / 1.0) * plotH;
                const bx = center + (bi - 1) * (barW + 3);
                const by = (H - padB) - barH;
                const tier = tiers[bi];

                return `
                    <rect x="${bx}" y="${by.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}"
                          rx="2" fill="${tier.fill}" stroke="rgba(255,255,255,0.2)" stroke-width="0.8">
                        <title>${corpusName} - ${tier.label}: ${Number(val).toFixed(4)}</title>
                    </rect>
                `;
            }).join('');

            return `
                <g class="chart-group" tabindex="0">
                    ${bars}
                    <text x="${center + 7}" y="${H - 12}" text-anchor="middle"
                          fill="rgba(255,255,255,0.6)" font-family="var(--font-body, sans-serif)" font-size="10">${corpusName} (${Number(mapVal).toFixed(2)})</text>
                    <title>${corpusName} Tagging: mAP=${Number(mapVal).toFixed(4)}, F1=${Number(f1Val).toFixed(4)}, AUC=${Number(aucVal).toFixed(4)}</title>
                </g>
            `;
        }).join('');

        container.innerHTML = `
            <svg class="chart-svg" viewBox="0 0 ${W} ${H}" fill="none" role="img" aria-label="Multi-Label ASL Tagging Metrics grouped bar chart">
                <!-- Top Legend -->
                ${legendHtml}

                <!-- Axes -->
                <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
                <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>

                <!-- Y Axis Labels -->
                <text x="${padL - 6}" y="${padT + 4}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">1.0</text>
                <text x="${padL - 6}" y="${padT + plotH / 2 + 3}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">0.5</text>
                <text x="${padL - 6}" y="${H - padB + 3}" text-anchor="end" fill="rgba(255,255,255,0.4)" font-family="var(--font-mono, monospace)" font-size="8">0.0</text>

                <!-- Groups & Tooltips -->
                ${groupsHtml}
            </svg>
        `;
    }
};

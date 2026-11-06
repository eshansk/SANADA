/**
 * frontend/js/ui/charts.js
 * ========================
 * Benchmark Table and Metric Formatters.
 * Transforms raw paper metrics into accessible HTML comparison rows with status badges.
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
        return (rows || []).map(row => {
            const val = row.si_sdr_db !== undefined ? row.si_sdr_db : row.sisdrDb;
            const statusBadge = val >= 4.0 ? 'pass' : (val >= 0 ? 'warn' : 'fail');
            return `
                <tr>
                    <td><strong>K = ${row.density_k || row.k || '—'}</strong></td>
                    <td>${row.domain || 'Org-Mix Fixed'}</td>
                    <td class="font-mono" style="font-weight: 600;">${val >= 0 ? '+' : ''}${Number(val).toFixed(2)} dB</td>
                    <td><span class="badge ${statusBadge}">${row.status}</span></td>
                </tr>
            `;
        }).join('');
    },

    renderCountingRows(rows) {
        return (rows || []).map(row => `
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
        return (rows || []).map(row => `
            <tr>
                <td><strong>${row.domain || row.corpus}</strong></td>
                <td>${row.classes || '32 Classes'}</td>
                <td class="font-mono">${Number(row.map_macro || row.mAP).toFixed(4)}</td>
                <td class="font-mono">${Number(row.f1_macro || row.macroF1).toFixed(4)}</td>
                <td class="font-mono">${Number(row.roc_auc || row.rocAuc).toFixed(4)}</td>
            </tr>
        `).join('');
    }
};

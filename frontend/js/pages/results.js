/**
 * frontend/js/pages/results.js
 * ============================
 * Analytics & Benchmark Comparison Page Logic.
 * Hydrates empirical results tables and system statistics from /api/stats
 * with fallback to paper benchmarks.
 */

import { API } from '../services/api.js';
import { PAPER_BENCHMARKS } from '../data/paperBenchmarks.js';
import { ChartHelper } from '../ui/charts.js';

export async function loadAnalytics() {
    const DOM = {
        statTotalClips: document.getElementById('statTotalClips'),
        statTotalHours: document.getElementById('statTotalHours'),
        statSampleRate: document.getElementById('statSampleRate'),
        statClippingGain: document.getElementById('statClippingGain'),
        sisdrTableBody: document.getElementById('sisdrTableBody'),
        countingTableBody: document.getElementById('countingTableBody'),
        taggingTableBody: document.getElementById('taggingTableBody')
    };

    try {
        const data = await API.getStats();

        if (DOM.statTotalClips) DOM.statTotalClips.textContent = data.suite_overview.total_synthesized_mixtures.toLocaleString();
        if (DOM.statTotalHours) DOM.statTotalHours.textContent = `${data.suite_overview.total_duration_hours}h`;
        if (DOM.statSampleRate) DOM.statSampleRate.textContent = `${data.suite_overview.sample_rate_hz} Hz`;
        if (DOM.statClippingGain) DOM.statClippingGain.textContent = data.suite_overview.clipping_prevention_gain;

        if (DOM.sisdrTableBody) {
            DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(data.complexity_ceiling_benchmarks.source_separation_sisdr_db);
        }
        if (DOM.countingTableBody) {
            DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(data.complexity_ceiling_benchmarks.source_counting_mae);
        }
        if (DOM.taggingTableBody) {
            DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(data.complexity_ceiling_benchmarks.multi_label_tagging_asl);
        }
    } catch (err) {
        console.warn('[Analytics] Backend unreachable; rendering verified paper benchmarks fallback.');
        if (DOM.sisdrTableBody) {
            DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(PAPER_BENCHMARKS.separationSisdr.orgMixFixed);
        }
        if (DOM.countingTableBody) {
            DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(PAPER_BENCHMARKS.counting.results);
        }
        if (DOM.taggingTableBody) {
            DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(PAPER_BENCHMARKS.tagging.results);
        }
    }
}

export function initResultsPage() {
    // Initial fetch
    loadAnalytics();
}

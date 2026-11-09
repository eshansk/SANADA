/**
 * frontend/js/pages/results.js
 * ============================
 * Analytics & Benchmark Comparison Page Logic.
 * Hydrates empirical results tables, summary tiles, and mathematically scaled
 * dynamic SVG charts from /api/stats with robust fallback to PAPER_BENCHMARKS (never empty).
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
        taggingTableBody: document.getElementById('taggingTableBody'),
        sisdrChartContainer: document.getElementById('sisdrChartContainer'),
        countingChartContainer: document.getElementById('countingChartContainer'),
        taggingChartContainer: document.getElementById('taggingChartContainer')
    };

    // Render baseline paper benchmarks immediately so charts and tables are never blank
    if (DOM.sisdrTableBody) {
        DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(PAPER_BENCHMARKS.separationSisdr.orgMixFixed);
    }
    if (DOM.countingTableBody) {
        DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(PAPER_BENCHMARKS.counting.results);
    }
    if (DOM.taggingTableBody) {
        DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(PAPER_BENCHMARKS.tagging.results);
    }

    if (DOM.sisdrChartContainer) {
        ChartHelper.renderSisdrChart(DOM.sisdrChartContainer, PAPER_BENCHMARKS.separationSisdr.orgMixFixed);
    }
    if (DOM.countingChartContainer) {
        ChartHelper.renderCountingChart(DOM.countingChartContainer, PAPER_BENCHMARKS.counting.results);
    }
    if (DOM.taggingChartContainer) {
        ChartHelper.renderTaggingChart(DOM.taggingChartContainer, PAPER_BENCHMARKS.tagging.results);
    }

    // Try fetching live statistics from backend API
    try {
        const data = await API.getStats();

        if (DOM.statTotalClips) DOM.statTotalClips.textContent = data.suite_overview.total_synthesized_mixtures.toLocaleString();
        if (DOM.statTotalHours) DOM.statTotalHours.textContent = `${data.suite_overview.total_duration_hours}h`;
        if (DOM.statSampleRate) DOM.statSampleRate.textContent = `${data.suite_overview.sample_rate_hz} Hz`;
        if (DOM.statClippingGain) DOM.statClippingGain.textContent = '-3.0 dBFS';

        if (data.complexity_ceiling_benchmarks) {
            const b = data.complexity_ceiling_benchmarks;
            if (DOM.sisdrTableBody && b.source_separation_sisdr_db) {
                DOM.sisdrTableBody.innerHTML = ChartHelper.renderSisdrRows(b.source_separation_sisdr_db);
            }
            if (DOM.countingTableBody && b.source_counting_mae) {
                DOM.countingTableBody.innerHTML = ChartHelper.renderCountingRows(b.source_counting_mae);
            }
            if (DOM.taggingTableBody && b.multi_label_tagging_asl) {
                DOM.taggingTableBody.innerHTML = ChartHelper.renderTaggingRows(b.multi_label_tagging_asl);
            }

            if (DOM.sisdrChartContainer && b.source_separation_sisdr_db) {
                const fixedRows = b.source_separation_sisdr_db.filter(d => 
                    !d.domain || d.domain.includes('Fixed') || d.domain.includes('Speech')
                );
                ChartHelper.renderSisdrChart(DOM.sisdrChartContainer, fixedRows.length > 0 ? fixedRows : b.source_separation_sisdr_db);
            }
            if (DOM.countingChartContainer && b.source_counting_mae) {
                ChartHelper.renderCountingChart(DOM.countingChartContainer, b.source_counting_mae);
            }
            if (DOM.taggingChartContainer && b.multi_label_tagging_asl) {
                ChartHelper.renderTaggingChart(DOM.taggingChartContainer, b.multi_label_tagging_asl);
            }
        }
    } catch (err) {
        // Backend offline; already populated from verified PAPER_BENCHMARKS
    }
}

export function initResultsPage() {
    loadAnalytics();
}

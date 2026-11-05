/**
 * frontend/js/ui/paper-data.js
 * ============================
 * Official SANADA Paper Benchmark Constants & Fallback Metadata
 * Source: "SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures,
 * Evaluated with Baseline Models in Separation, Counting, and Tagging",
 * P. V. N. Balarama Murthy and R. Vijaya Prakash (SR University / NGIT).
 * DOI: 10.21227/z7hv-e014 (IEEE DataPort). SSRN preprint 7199188.
 */

export const PAPER_BENCHMARKS = {
    citation: {
        title: "SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures, Evaluated with Baseline Models in Separation, Counting, and Tagging",
        authors: "P. V. N. Balarama Murthy and R. Vijaya Prakash",
        institutions: "SR University / Neil Gogte Institute of Technology",
        doi: "10.21227/z7hv-e014",
        repository: "IEEE DataPort",
        ssrnPreprint: "SSRN preprint 7199188, not peer reviewed"
    },

    datasetSummary: {
        totalStems: 27112,
        totalStemsDurationHours: 60.25,
        totalStemsSizeGb: 6.47,
        totalMixtures: 15600,
        totalMixturesDurationHours: 34.66,
        totalMixturesSizeGb: 3.72,
        subDatasetsCount: 13,
        sampleRateHz: 16000,
        bitDepth: "16-bit PCM",
        channels: 1, // Monaural
        clipDurationSeconds: 8.0,
        clipSamples: 128000
    },

    counting: {
        model: "Conformer Regressor (MSE Loss)",
        results: [
            {
                corpus: "Org-Mix-4",
                kMax: 4,
                mae: 0.3243,
                rmse: 0.4302,
                exactAccuracyPct: 74.75,
                plusMinusOneAccuracyPct: 100.0
            },
            {
                corpus: "Org-Mix-9",
                kMax: 9,
                mae: 0.8766,
                rmse: 1.0925,
                exactAccuracyPct: 36.46,
                plusMinusOneAccuracyPct: 84.38
            },
            {
                corpus: "FSD-Mix-32",
                kMax: 32,
                mae: 6.7262,
                rmse: 8.0885,
                exactAccuracyPct: 5.56,
                plusMinusOneAccuracyPct: 14.44
            }
        ]
    },

    tagging: {
        model: "Conformer Classifier (Asymmetric Loss - ASL)",
        results: [
            {
                corpus: "Org-Mix-4",
                mAP: 0.9902,
                macroF1: 0.9449,
                rocAuc: 0.9765
            },
            {
                corpus: "Org-Mix-9",
                mAP: 0.9122,
                macroF1: 0.8305,
                rocAuc: 0.8744
            },
            {
                corpus: "FSD-Mix",
                mAP: 0.6265,
                macroF1: 0.6849,
                rocAuc: 0.6161
            }
        ]
    },

    separationSisdr: {
        model: "Conv-TasNet (uPIT Loss)",
        orgMixFixed: [
            { k: 2, sisdrDb: 10.59, status: "Clean (+10.6 dB)" },
            { k: 3, sisdrDb: 5.08, status: "Usable (+5.1 dB)" },
            { k: 4, sisdrDb: 4.24, status: "Usable Ceiling (+4.2 dB)" },
            { k: 6, sisdrDb: -3.40, status: "Degraded (-3.4 dB)" },
            { k: 9, sisdrDb: -8.57, status: "Intractable (-8.6 dB)" }
        ],
        fsdMixFixed: [
            { k: 2, sisdrDb: 5.05, status: "Usable (+5.1 dB)" },
            { k: 3, sisdrDb: 0.73, status: "Near 0 dB Threshold (+0.7 dB)" }
        ],
        orgMixVariable: [
            { condition: "K <= 4 (Org-Mix-4 Variable)", sisdrDb: 1.98, status: "Usable (+2.0 dB)" },
            { condition: "K <= 9 (Org-Mix-9 Variable)", sisdrDb: -9.61, status: "Degraded (< 0 dB)" }
        ]
    }
};

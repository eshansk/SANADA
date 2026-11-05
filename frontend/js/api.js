/**
 * frontend/js/api.js
 * ==================
 * Master REST API Client Service Layer
 * Member 1 (Frontend UI & Client Engine)
 * Aligned with docs/MEMBER1_HANDOVER.md contract
 */

const API_BASE = window.SANADA_API_BASE || (window.location.hostname === 'localhost' ? 'http://127.0.0.1:8000' : `${window.location.protocol}//${window.location.hostname || '127.0.0.1'}:8000`);

export const API = {
    /**
     * Check backend system vitality, memory, and model status
     */
    async getHealth() {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Pre-flight audio validation (POST /api/upload)
     * Validates 16 kHz, single-channel mono, 8.0s length
     */
    async validateAudio(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(`Upload validation failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Multi-task machine listening prediction (POST /api/predict)
     * Executes Task 1 (Counting), Task 2 (Tagging), and Task 3 (Separation)
     */
    async predictAudio(file, task = 'all') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('task', task);
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(`Inference request failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * System & Complexity Ceiling Telemetry (GET /api/stats)
     */
    async getStats() {
        const res = await fetch(`${API_BASE}/api/stats`);
        if (!res.ok) throw new Error(`Failed to load analytics: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Trigger asynchronous model training run (POST /api/train)
     */
    async triggerTraining(config) {
        const res = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error(`Training trigger failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Poll status of training job (GET /api/train/{job_id})
     */
    async pollTrainingJob(jobId) {
        const res = await fetch(`${API_BASE}/api/train/${jobId}`);
        if (!res.ok) throw new Error(`Job polling failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Export and version model checkpoint (POST /api/save-model)
     */
    async saveModel(task, versionTag, notes) {
        const res = await fetch(`${API_BASE}/api/save-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task,
                version_tag: versionTag,
                notes
            })
        });
        if (!res.ok) throw new Error(`Save model failed: HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Formats relative stem URL to absolute stream endpoint
     */
    getStemUrl(relativeUrl) {
        if (!relativeUrl) return '';
        if (relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://')) return relativeUrl;
        return `${API_BASE}${relativeUrl.startsWith('/') ? '' : '/'}${relativeUrl}`;
    }
};

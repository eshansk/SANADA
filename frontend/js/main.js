/**
 * frontend/js/main.js
 * ====================
 * Master Application Entry Point.
 * Sole orchestrator and single boot execution for the SANADA web platform.
 * Wires router, UI sub-systems, page handlers, audio engines, and telemetry.
 */

import { API_BASE } from './config.js';
import { API } from './services/api.js';
import { showToast } from './ui/toast.js';
import { initRouter, switchRoute } from './ui/router.js';
import { initHomeVideo, updateWaveVisibility } from './ui/homeVideo.js';
import { initHomePage } from './pages/home.js';
import { initStudioPage } from './pages/studio.js';
import { initResultsPage, loadAnalytics } from './pages/results.js';
import { initDeveloperPage } from './pages/developer.js';
import { initPaperPage } from './pages/paper.js';

async function checkHealth() {
    const backendDot = document.getElementById('backendDot');
    const backendText = document.getElementById('backendText');

    try {
        const data = await API.getHealth();
        if (backendDot) backendDot.className = 'status-dot online';
        if (backendText) backendText.textContent = `Engine online (${data.device.toUpperCase()})`;
        return true;
    } catch (e) {
        if (backendDot) backendDot.className = 'status-dot';
        if (backendText) backendText.textContent = 'Engine offline';
        return false;
    }
}

function initMobileNav() {
    const navToggle = document.getElementById('navToggle');
    const navTabs = document.querySelector('.nav-tabs');
    if (!navToggle || !navTabs) return;

    navToggle.addEventListener('click', () => {
        navTabs.classList.toggle('open');
        navToggle.classList.toggle('active');
    });

    navTabs.querySelectorAll('.tab-link').forEach(link => {
        link.addEventListener('click', () => {
            navTabs.classList.remove('open');
            navToggle.classList.remove('active');
        });
    });
}

function initPingButton() {
    const pingBtn = document.getElementById('pingBtn');
    if (!pingBtn) return;

    pingBtn.addEventListener('click', async () => {
        const ok = await checkHealth();
        if (ok) {
            showToast('Backend API is healthy and reachable!', 'success');
        } else {
            showToast(`Cannot connect to ${API_BASE}. Start backend/app.py.`, 'warning');
        }
    });
}

function initVisibilityControls() {
    if (typeof document === 'undefined') return;
    document.addEventListener('visibilitychange', () => {
        document.body.classList.toggle('doc-hidden', document.hidden);
    });
}

export function boot() {
    // 1. Initialize Single-Page Router & sync route changes
    initRouter((route) => {
        updateWaveVisibility();
        if (route === 'analytics') {
            loadAnalytics();
        }
    });

    // 2. Initialize Page Controllers
    initHomePage();
    initStudioPage();
    initResultsPage();
    initDeveloperPage();
    initPaperPage();

    // 3. Initialize Shared Navigation & Controls
    initMobileNav();
    initPingButton();
    initVisibilityControls();

    // 4. Initialize Background Visual Video
    initHomeVideo('ctaVideo');

    // 5. Initial Vitality Telemetry Check
    checkHealth();
}

// Single Entry Point Initialization
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
}

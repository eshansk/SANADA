/**
 * frontend/js/config.js
 * =====================
 * Global runtime configuration and audio processing constraints.
 * Centralizes API endpoint discovery and DSP parameters.
 */

export const API_BASE = (typeof window !== 'undefined' && window.SANADA_API_BASE)
    || (typeof window !== 'undefined' && window.location.hostname === 'localhost'
        ? 'http://127.0.0.1:8000'
        : `${(typeof window !== 'undefined' && window.location.protocol) || 'http:'}//${(typeof window !== 'undefined' && window.location.hostname) || '127.0.0.1'}:8000`);

export const TARGET_SR = 16000;
export const EXPECTED_FRAMES = 128000;
export const EXPECTED_DURATION = 8.0;

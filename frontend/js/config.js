/**
 * frontend/js/config.js
 * =====================
 * Global runtime configuration and audio processing constraints.
 * Centralizes API endpoint discovery and DSP parameters.
 */

export const API_BASE = (typeof window !== 'undefined' && window.SANADA_API_BASE)
    || (typeof window !== 'undefined' && (
        window.location.hostname.includes('trycloudflare.com')
        || window.location.hostname.includes('loca.lt')
        || window.location.hostname.includes('lhr.life')
        || window.location.port === '8000'
    )
        ? window.location.origin
        : (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
            ? 'http://127.0.0.1:8000'
            : `${(typeof window !== 'undefined' && window.location.protocol) || 'http:'}//${(typeof window !== 'undefined' && window.location.hostname) || '127.0.0.1'}:8000`));

export const TARGET_SR = 16000;
export const EXPECTED_FRAMES = 128000;
export const EXPECTED_DURATION = 8.0;

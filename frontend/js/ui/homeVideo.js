/**
 * frontend/js/ui/homeVideo.js
 * ===========================
 * Ambient HLS Background Stream Engine.
 * Dynamically streams Mux wave animation on the homepage view, pauses off-screen,
 * and reacts to user scroll energy with momentum physics.
 */

const HLS_CDN_URL = 'https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js';
const MUX_HLS_SRC = 'https://stream.mux.com/8wrHPCX2dC3msyYU9ObwqNdm00u3ViXvOSHUMRYSEe5Q.m3u8';

let hlsInstance = null;
let currentRate = 1.0;
let scrollEnergy = 0;
let rafId = null;
let lastScrollY = typeof window !== 'undefined' ? window.scrollY : 0;
let videoElement = null;
let waveContainer = null;

function loadHlsScript() {
    return new Promise((resolve, reject) => {
        if (typeof window !== 'undefined' && typeof window.Hls !== 'undefined') {
            resolve(window.Hls);
            return;
        }

        const existingScript = document.querySelector(`script[src="${HLS_CDN_URL}"]`);
        if (existingScript) {
            existingScript.addEventListener('load', () => resolve(window.Hls));
            existingScript.addEventListener('error', reject);
            return;
        }

        const script = document.createElement('script');
        script.src = HLS_CDN_URL;
        script.async = true;
        script.onload = () => resolve(window.Hls);
        script.onerror = () => reject(new Error('Failed to load hls.js from CDN'));
        document.head.appendChild(script);
    });
}

function isHomePageActive() {
    const hash = (window.location.hash || '').replace(/^#\/?/, '').split('/')[0].trim();
    if (hash && hash !== 'home') return false;

    const pageHome = document.getElementById('pageHome');
    if (!pageHome) return true;
    return pageHome.classList.contains('active') || hash === '' || hash === 'home';
}

export function updateWaveVisibility() {
    if (!waveContainer) waveContainer = document.getElementById('homeWaveBg');
    if (!videoElement) videoElement = document.getElementById('ctaVideo');
    if (!waveContainer || !videoElement) return;

    const onHome = isHomePageActive();

    if (onHome) {
        waveContainer.style.display = 'block';
        requestAnimationFrame(() => {
            waveContainer.style.opacity = '1';
        });
        if (videoElement.paused) {
            videoElement.play().catch(() => {});
        }
    } else {
        waveContainer.style.opacity = '0';
        setTimeout(() => {
            if (!isHomePageActive()) {
                waveContainer.style.display = 'none';
                videoElement.pause();
            }
        }, 300);
    }
}

function tickScrollVelocity() {
    if (!videoElement) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
        videoElement.playbackRate = 1.0;
        rafId = null;
        return;
    }

    const targetRate = 1.0 + scrollEnergy;
    currentRate += (targetRate - currentRate) * 0.28;

    scrollEnergy *= 0.91;
    if (scrollEnergy < 0.03) {
        scrollEnergy = 0;
    }

    const clampedRate = Math.max(1.0, Math.min(3.8, currentRate));
    try {
        videoElement.playbackRate = Number(clampedRate.toFixed(2));
    } catch (e) {}

    if (scrollEnergy > 0 || Math.abs(currentRate - 1.0) > 0.02) {
        rafId = requestAnimationFrame(tickScrollVelocity);
    } else {
        currentRate = 1.0;
        try { videoElement.playbackRate = 1.0; } catch (e) {}
        rafId = null;
    }
}

function triggerVelocity() {
    if (!isHomePageActive()) return;
    if (!rafId) {
        rafId = requestAnimationFrame(tickScrollVelocity);
    }
}

function handleScroll() {
    if (!isHomePageActive()) return;
    const currentY = window.scrollY;
    const deltaY = Math.abs(currentY - lastScrollY);
    lastScrollY = currentY;

    if (deltaY > 0) {
        scrollEnergy = Math.min(2.8, scrollEnergy + Math.min(1.6, deltaY * 0.06));
        triggerVelocity();
    }
}

function handleWheel(e) {
    if (!isHomePageActive()) return;
    const delta = Math.abs(e.deltaY || e.detail || 0);
    if (delta > 0) {
        scrollEnergy = Math.min(2.8, scrollEnergy + Math.min(1.8, delta * 0.05));
        triggerVelocity();
    }
}

export async function initHomeVideo(videoId = 'ctaVideo') {
    videoElement = document.getElementById(videoId);
    waveContainer = document.getElementById('homeWaveBg');
    if (!videoElement) return;

    videoElement.crossOrigin = 'anonymous';

    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const shouldAutoplay = !reducedMotionQuery.matches;

    if (!shouldAutoplay) {
        videoElement.autoplay = false;
        videoElement.pause();
        return;
    }

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('wheel', handleWheel, { passive: true });

    window.addEventListener('hashchange', updateWaveVisibility);
    document.querySelectorAll('.tab-link').forEach(tab => {
        tab.addEventListener('click', () => {
            setTimeout(updateWaveVisibility, 50);
        });
    });

    const pageHome = document.getElementById('pageHome');
    if (pageHome) {
        const observer = new MutationObserver(() => updateWaveVisibility());
        observer.observe(pageHome, { attributes: true, attributeFilter: ['class'] });
    }

    function handleVideoError(reason) {
        console.warn(`[Wave Background] Dark fallback: ${reason}`);
        if (videoElement) videoElement.style.display = 'none';
        if (waveContainer) waveContainer.style.background = '#000000';
    }

    videoElement.addEventListener('error', () => {
        handleVideoError('Video playback error');
    });

    try {
        const Hls = await loadHlsScript();

        if (Hls && Hls.isSupported()) {
            hlsInstance = new Hls({
                enableWorker: true,
                lowLatencyMode: false,
                backBufferLength: 30
            });

            hlsInstance.loadSource(MUX_HLS_SRC);
            hlsInstance.attachMedia(videoElement);

            hlsInstance.on(Hls.Events.MANIFEST_PARSED, () => {
                if (shouldAutoplay && isHomePageActive()) {
                    videoElement.play().catch(err => {
                        console.warn('[Wave Background] Autoplay policy prevented playback:', err);
                    });
                }
            });

            hlsInstance.on(Hls.Events.ERROR, (event, data) => {
                if (data.fatal) {
                    switch (data.type) {
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            hlsInstance.startLoad();
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            hlsInstance.recoverMediaError();
                            break;
                        default:
                            handleVideoError(`Fatal HLS error: ${data.details}`);
                            hlsInstance.destroy();
                            break;
                    }
                }
            });

            window.addEventListener('pagehide', () => {
                if (hlsInstance) hlsInstance.destroy();
            });

        } else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
            videoElement.src = MUX_HLS_SRC;
            videoElement.addEventListener('loadedmetadata', () => {
                if (shouldAutoplay && isHomePageActive()) {
                    videoElement.play().catch(() => {});
                }
            });
        } else {
            handleVideoError('HLS not supported in this browser');
        }
    } catch (err) {
        if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
            videoElement.src = MUX_HLS_SRC;
            if (shouldAutoplay && isHomePageActive()) {
                videoElement.play().catch(() => {});
            }
        } else {
            handleVideoError(err.message);
        }
    }

    updateWaveVisibility();
}

// Backward-compatible alias
export const initCtaVideo = initHomeVideo;

/**
 * frontend/js/ui/homeVideo.js
 * ===========================
 * Ambient Background Wave Video Engine.
 * Streams local wave animation on the homepage view, pauses off-screen,
 * and reacts to user scroll energy with momentum physics.
 */

const LOCAL_VIDEO_SRC = 'assets/wave_bg.mp4?v=4';
export const BASE_RATE = 0.80; // Ambient slow-motion baseline speed (calm, natural cinematic wave)

let currentRate = BASE_RATE;
let scrollEnergy = 0;
let rafId = null;
let lastScrollY = typeof window !== 'undefined' ? window.scrollY : 0;
let videoElement = null;
let waveContainer = null;

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
        if (scrollEnergy === 0) {
            videoElement.playbackRate = BASE_RATE;
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
        videoElement.playbackRate = BASE_RATE;
        rafId = null;
        return;
    }

    const targetRate = BASE_RATE + scrollEnergy;
    currentRate += (targetRate - currentRate) * 0.28;

    scrollEnergy *= 0.91;
    if (scrollEnergy < 0.03) {
        scrollEnergy = 0;
    }

    const clampedRate = Math.max(BASE_RATE, Math.min(3.2, currentRate));
    try {
        videoElement.playbackRate = Number(clampedRate.toFixed(2));
    } catch (e) {}

    if (scrollEnergy > 0 || Math.abs(currentRate - BASE_RATE) > 0.02) {
        rafId = requestAnimationFrame(tickScrollVelocity);
    } else {
        currentRate = BASE_RATE;
        try { videoElement.playbackRate = BASE_RATE; } catch (e) {}
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
    videoElement.loop = true;
    videoElement.muted = true;
    videoElement.playsInline = true;
    videoElement.defaultPlaybackRate = BASE_RATE;
    videoElement.playbackRate = BASE_RATE;

    videoElement.addEventListener('loadedmetadata', () => {
        videoElement.defaultPlaybackRate = BASE_RATE;
        videoElement.playbackRate = BASE_RATE;
    });

    videoElement.addEventListener('play', () => {
        if (scrollEnergy === 0) {
            videoElement.playbackRate = BASE_RATE;
        }
    });

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
        console.warn(`[Wave Background] Fallback: ${reason}`);
        if (videoElement) videoElement.style.display = 'none';
        if (waveContainer) waveContainer.style.background = '#000000';
    }

    videoElement.addEventListener('error', () => {
        handleVideoError('Video playback error');
    });

    try {
        if (!videoElement.currentSrc && !videoElement.src) {
            videoElement.src = LOCAL_VIDEO_SRC;
        }
        if (shouldAutoplay && isHomePageActive()) {
            const playPromise = videoElement.play();
            if (playPromise !== undefined) {
                playPromise.catch(err => {
                    console.warn('[Wave Background] Autoplay policy prevented playback:', err);
                });
            }
        }
    } catch (err) {
        handleVideoError(err.message);
    }

    updateWaveVisibility();
}

// Backward-compatible alias
export const initCtaVideo = initHomeVideo;

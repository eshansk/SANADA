/**
 * frontend/js/ui/router.js
 * ========================
 * Hash-based Single-Page Application Router.
 * Coordinates view transitions, tab indicator states, and URL synchronization
 * across all 7 platform sections without page reloads.
 */

const VALID_ROUTES = ['home', 'predict', 'dataset', 'train', 'analytics', 'docs', 'paper'];

const ROUTE_ALIASES = {
    studio: 'predict',
    results: 'analytics',
    developer: 'train'
};

let currentRoute = 'home';
let routeChangeCallbacks = [];

export function resolveRoute(inputRoute) {
    const cleaned = (inputRoute || '').replace(/^#\/?/, '').split('/')[0].trim();
    const normalized = ROUTE_ALIASES[cleaned] || cleaned;
    return VALID_ROUTES.includes(normalized) ? normalized : 'home';
}

export function getCurrentRoute() {
    return currentRoute;
}

export function updateNavIndicator() {
    if (typeof document === 'undefined') return;
    const indicator = document.getElementById('navIndicator');
    const navTabs = document.querySelector('.navbar .nav-tabs');
    if (!indicator || !navTabs) return;

    // On narrow screens (<= 1100px) or mobile hamburger menu, indicator is hidden
    if (window.innerWidth <= 1100 || navTabs.classList.contains('open')) {
        indicator.style.opacity = '0';
        return;
    }

    const activeTab = navTabs.querySelector('.tab-link.active');
    if (activeTab && activeTab.offsetParent !== null) {
        indicator.style.opacity = '1';
        indicator.style.transform = `translateX(${activeTab.offsetLeft}px)`;
        indicator.style.width = `${activeTab.offsetWidth}px`;
    } else {
        indicator.style.opacity = '0';
    }
}

export function switchRoute(targetRoute, updateHash = true) {
    const route = resolveRoute(targetRoute);
    currentRoute = route;

    // Scroll to top on route change
    if (typeof window !== 'undefined') {
        window.scrollTo(0, 0);
    }

    if (typeof document !== 'undefined' && document.body) {
        document.body.dataset.page = route;
    }

    if (updateHash && window.location.hash !== `#/${route}`) {
        window.location.hash = `#/${route}`;
    }

    // Update navbar tabs
    const tabs = document.querySelectorAll('.navbar .nav-tabs .tab-link');
    tabs.forEach(tab => {
        if (tab.dataset.route === route) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Update sliding pill indicator
    updateNavIndicator();
    if (typeof requestAnimationFrame !== 'undefined') {
        requestAnimationFrame(updateNavIndicator);
    }

    // Toggle view-page containers
    const pages = {
        home: document.getElementById('pageHome'),
        predict: document.getElementById('pagePredict'),
        dataset: document.getElementById('pageDataset'),
        train: document.getElementById('pageTrain'),
        analytics: document.getElementById('pageAnalytics'),
        docs: document.getElementById('pageDocs'),
        paper: document.getElementById('pagePaper')
    };

    Object.entries(pages).forEach(([key, el]) => {
        if (!el) return;
        if (key === route) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Notify registered listeners
    routeChangeCallbacks.forEach(cb => {
        try {
            cb(route);
        } catch (err) {
            console.error('Route listener error:', err);
        }
    });
}

export function onRouteChanged(callback) {
    if (typeof callback === 'function') {
        routeChangeCallbacks.push(callback);
    }
}

export function initRouter(onRouteChangeCallback) {
    if (onRouteChangeCallback) {
        onRouteChanged(onRouteChangeCallback);
    }

    // Navigation tab click delegation
    document.querySelectorAll('.navbar .nav-tabs .tab-link').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const route = tab.dataset.route;
            if (route) {
                switchRoute(route);
            }
        });
    });

    // Update sliding pill on resize
    window.addEventListener('resize', () => {
        updateNavIndicator();
    });

    // Listen to hash changes in window
    window.addEventListener('hashchange', () => {
        const hashRoute = (window.location.hash || '').replace(/^#\/?/, '').trim();
        switchRoute(hashRoute, false);
    });

    // Initial route sync
    const initialRoute = resolveRoute(window.location.hash);
    switchRoute(initialRoute, false);
}

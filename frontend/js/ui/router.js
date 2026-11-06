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

export function switchRoute(targetRoute, updateHash = true) {
    const route = resolveRoute(targetRoute);
    currentRoute = route;

    if (typeof document !== 'undefined' && document.body) {
        document.body.dataset.page = route;
    }

    if (updateHash && window.location.hash !== `#/${route}`) {
        window.location.hash = `#/${route}`;
    }

    // Update navbar tabs
    const tabs = document.querySelectorAll('.tab-link');
    tabs.forEach(tab => {
        if (tab.dataset.route === route) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

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
    document.querySelectorAll('.tab-link').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const route = tab.dataset.route;
            if (route) {
                switchRoute(route);
            }
        });
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

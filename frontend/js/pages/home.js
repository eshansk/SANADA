/**
 * frontend/js/pages/home.js
 * =========================
 * Homepage view logic and interactive call-to-action bindings.
 */

import { switchRoute } from '../ui/router.js';

export function initHomePage() {
    // Intercept in-page hash links to ensure fast router switching
    const homeLinks = document.querySelectorAll('#pageHome a[href^="#/"]');
    homeLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const hash = link.getAttribute('href');
            if (hash) {
                e.preventDefault();
                switchRoute(hash.replace('#/', ''));
            }
        });
    });
}

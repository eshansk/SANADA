/**
 * frontend/js/auth-guard.js
 * =========================
 * Non-blocking authentication and navigation helper.
 * Route restriction is disabled so users can freely explore all portals.
 * Manages session cleanup and navbar Logout action.
 */

function logout() {
    try {
        sessionStorage.removeItem('sanada_auth');
    } catch (e) {
        console.warn('Failed to clear session', e);
    }
    window.location.href = 'login.html';
}

// Make logout globally accessible
if (typeof window !== 'undefined') {
    window.logout = logout;

    function wireLogout() {
        const logoutElements = document.querySelectorAll('#logoutBtn, .btn-logout, .logout-link');
        logoutElements.forEach((el) => {
            el.addEventListener('click', (e) => {
                e.preventDefault();
                logout();
            });
        });
    }

    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', wireLogout);
    } else {
        wireLogout();
    }
}

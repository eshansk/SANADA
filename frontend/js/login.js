/**
 * frontend/js/login.js
 * ====================
 * Login page authentication controller.
 * Handles client-side credential verification and portal routing.
 */

// =============================================================================
// DEMO CREDENTIALS CONFIGURATION
// NOTE: Front-end demo authentication. Replace with backend auth in production.
// =============================================================================
export const DEMO_CONFIG = {
    developer: {
        username: 'developer',
        password: 'dev123',
        role: 'developer',
        redirect: 'index.html'
    },
    user: {
        username: 'user',
        password: 'user123',
        role: 'user',
        redirect: 'user.html'
    }
};

function showError(message) {
    const errorBox = document.getElementById('loginError');
    const errorText = document.getElementById('loginErrorText');
    if (errorBox && errorText) {
        errorText.textContent = message;
        errorBox.style.display = 'flex';
    }
}

function hideError() {
    const errorBox = document.getElementById('loginError');
    if (errorBox) {
        errorBox.style.display = 'none';
    }
}

export function initLoginPage() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('usernameInput');
    const passwordInput = document.getElementById('passwordInput');
    const newUserLink = document.getElementById('newUserLink');

    if (!loginForm || !usernameInput || !passwordInput) return;

    // Reset error when user types
    usernameInput.addEventListener('input', hideError);
    passwordInput.addEventListener('input', hideError);

    // New User helper link
    if (newUserLink) {
        newUserLink.addEventListener('click', (e) => {
            e.preventDefault();
            usernameInput.value = DEMO_CONFIG.user.username;
            passwordInput.value = DEMO_CONFIG.user.password;
            hideError();
            usernameInput.focus();
        });
    }

    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const username = usernameInput.value.trim();
        const password = passwordInput.value.trim();

        hideError();

        // Validation: Don't allow empty fields
        if (!username || !password) {
            showError('Please enter both User ID and password.');
            return;
        }

        // Check for DEVELOPER account match
        if (
            username.toLowerCase() === DEMO_CONFIG.developer.username.toLowerCase() &&
            password === DEMO_CONFIG.developer.password
        ) {
            sessionStorage.setItem('sanada_auth', JSON.stringify({
                role: DEMO_CONFIG.developer.role,
                username: DEMO_CONFIG.developer.username,
                timestamp: Date.now()
            }));
            window.location.replace(DEMO_CONFIG.developer.redirect);
            return;
        }

        // Check for USER account match
        if (
            username.toLowerCase() === DEMO_CONFIG.user.username.toLowerCase() &&
            password === DEMO_CONFIG.user.password
        ) {
            sessionStorage.setItem('sanada_auth', JSON.stringify({
                role: DEMO_CONFIG.user.role,
                username: DEMO_CONFIG.user.username,
                timestamp: Date.now()
            }));
            window.location.replace(DEMO_CONFIG.user.redirect);
            return;
        }

        // No match: Stay on login.html and show error in glass error area
        showError('Invalid User ID or password.');
    });
}

// Auto-initialize when DOM is ready
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', initLoginPage);
    } else {
        initLoginPage();
    }
}

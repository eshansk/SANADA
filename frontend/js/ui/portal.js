/**
 * frontend/js/ui/portal.js
 * ========================
 * Controller for SANADA Portal Authentication & Role Differentiation.
 * Used exclusively for standalone portal authentication (login.html).
 */

import { showToast } from './toast.js';

/**
 * Initializes the standalone auth page if running on login.html.
 */
export function initAuthPage() {
    let currentRole = 'user';
    let currentMode = 'signin';

    const roleUserBtn = document.getElementById('roleUserBtn');
    const roleAdminBtn = document.getElementById('roleAdminBtn');
    const destText = document.getElementById('destText');
    const tabSignIn = document.getElementById('tabSignIn');
    const tabSignUp = document.getElementById('tabSignUp');
    const signInFormContainer = document.getElementById('signInFormContainer');
    const signUpFormContainer = document.getElementById('signUpFormContainer');

    const signInForm = document.getElementById('signInForm');
    const signUpForm = document.getElementById('signUpForm');
    const signinSubmitBtn = document.getElementById('signinSubmitBtn');
    const signupSubmitBtn = document.getElementById('signupSubmitBtn');

    const googleBtns = document.querySelectorAll('.btn-google');
    const appleBtns = document.querySelectorAll('.btn-apple');
    const switchModeLinks = document.querySelectorAll('.switch-auth-mode');

    const urlParams = new URLSearchParams(window.location.search);
    const hash = window.location.hash.toLowerCase();

    if (urlParams.get('role') === 'admin' || hash.includes('admin')) {
        setRole('admin');
    } else {
        setRole('user');
    }

    if (urlParams.get('mode') === 'signup' || hash.includes('signup')) {
        setMode('signup');
    } else {
        setMode('signin');
    }

    function setRole(role) {
        currentRole = role;
        if (role === 'admin') {
            roleAdminBtn?.classList.add('active');
            roleUserBtn?.classList.remove('active');
            if (destText) destText.textContent = 'Destination: Admin / Developer Suite (index.html)';
            if (signinSubmitBtn) signinSubmitBtn.innerHTML = '<span>Sign In to Admin Portal</span>';
            if (signupSubmitBtn) signupSubmitBtn.innerHTML = '<span>Create Admin Account</span>';
        } else {
            roleUserBtn?.classList.add('active');
            roleAdminBtn?.classList.remove('active');
            if (destText) destText.textContent = 'Destination: User Machine Listening Studio (user.html)';
            if (signinSubmitBtn) signinSubmitBtn.innerHTML = '<span>Sign In to User Studio</span>';
            if (signupSubmitBtn) signupSubmitBtn.innerHTML = '<span>Create User Account</span>';
        }
    }

    function setMode(mode) {
        currentMode = mode;
        if (mode === 'signup') {
            tabSignUp?.classList.add('active');
            tabSignIn?.classList.remove('active');
            if (signInFormContainer) signInFormContainer.style.display = 'none';
            if (signUpFormContainer) {
                signUpFormContainer.style.display = 'block';
                signUpFormContainer.style.animation = 'authFadeIn 0.25s ease-out';
            }
        } else {
            tabSignIn?.classList.add('active');
            tabSignUp?.classList.remove('active');
            if (signUpFormContainer) signUpFormContainer.style.display = 'none';
            if (signInFormContainer) {
                signInFormContainer.style.display = 'block';
                signInFormContainer.style.animation = 'authFadeIn 0.25s ease-out';
            }
        }
    }

    function completeAuth(name, provider = 'credentials') {
        const destUrl = currentRole === 'admin' ? 'index.html#/home' : 'user.html#/predict';
        const portalLabel = currentRole === 'admin' ? 'Admin / Developer Suite' : 'User Machine Listening Studio';

        try {
            sessionStorage.setItem('sanada_auth', JSON.stringify({
                name: name || 'Authorized User',
                role: currentRole,
                provider: provider,
                timestamp: Date.now()
            }));
        } catch (e) {
            console.warn('Storage unavailable', e);
        }

        showToast(`Authenticated as ${currentRole.toUpperCase()}. Redirecting to ${portalLabel}...`, 'success');

        setTimeout(() => {
            window.location.href = destUrl;
        }, 900);
    }

    roleUserBtn?.addEventListener('click', () => setRole('user'));
    roleAdminBtn?.addEventListener('click', () => setRole('admin'));

    tabSignIn?.addEventListener('click', () => setMode('signin'));
    tabSignUp?.addEventListener('click', () => setMode('signup'));

    switchModeLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetMode = link.dataset.targetMode || 'signin';
            setMode(targetMode);
        });
    });

    googleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            showToast('Connecting to Google Identity Services...', 'info');
            setTimeout(() => completeAuth('Google User', 'google'), 600);
        });
    });

    appleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            showToast('Connecting to Apple ID...', 'info');
            setTimeout(() => completeAuth('Apple User', 'apple'), 600);
        });
    });

    signInForm?.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = document.getElementById('signinEmail')?.value.trim();
        const password = document.getElementById('signinPassword')?.value;
        if (!email || !password) {
            showToast('Please enter both email and password.', 'warning');
            return;
        }
        completeAuth(email.split('@')[0], 'email');
    });

    signUpForm?.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = document.getElementById('signupName')?.value.trim();
        const email = document.getElementById('signupEmail')?.value.trim();
        const password = document.getElementById('signupPassword')?.value;
        const confirm = document.getElementById('signupConfirm')?.value;

        if (!name || !email || !password) {
            showToast('Please fill out all required fields.', 'warning');
            return;
        }
        if (password.length < 6) {
            showToast('Password should be at least 6 characters.', 'warning');
            return;
        }
        if (password !== confirm) {
            showToast('Passwords do not match. Please re-enter.', 'warning');
            return;
        }
        completeAuth(name, 'registration');
    });
}

// Auto-boot handlers if running on login.html
if (typeof window !== 'undefined') {
    function autoInit() {
        if (document.getElementById('roleUserBtn')) {
            initAuthPage();
        }
    }
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', autoInit);
    } else {
        autoInit();
    }
}

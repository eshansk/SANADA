/**
 * frontend/js/auth-guard.js
 * =========================
 * SANADA Authentication & Role Display Controller.
 *
 * Provides:
 *  1. Early synchronous role detection (preventing layout shifts / flashing nav buttons).
 *  2. Unauthenticated landing mode (0 navigation buttons, hero shows Login button).
 *  3. Dual login selector & modal (Login as Developer vs Login as User).
 *  4. Supabase PostgreSQL authentication via public.authenticate_user(p_username, p_password) RPC.
 *  5. Role-based display switching (Developer suite vs User machine listening studio).
 *  6. Session persistence and clean logout handling.
 */

import {
    isSupabaseConfigured,
    loginWithSupabase,
    signUpWithSupabase,
    verifyOtpWithSupabase,
    resendOtpWithSupabase,
    getSupabaseUrl,
    getSupabaseKey
} from './services/supabase.js';

// Demo credentials configuration
export const DEMO_CONFIG = {
    developer: {
        username: 'developer',
        password: 'dev123',
        role: 'developer',
        label: 'Developer',
        badge: 'Full Suite',
        desc: 'Source counting, tagging, separation, datasets & checkpoints'
    },
    user: {
        username: 'user',
        password: 'user123',
        role: 'user',
        label: 'User',
        badge: 'Inference',
        desc: 'Neural source count estimation & speech separation studio'
    }
};

// 1. Synchronous phase: run immediately to tag document root before first paint
(function setInitialRoleEarly() {
    if (typeof window === 'undefined') return;
    try {
        const raw = sessionStorage.getItem('sanada_auth');
        if (raw) {
            const auth = JSON.parse(raw);
            if (auth && (auth.role === 'developer' || auth.role === 'user')) {
                document.documentElement.dataset.role = auth.role;
                return;
            }
        }
    } catch (e) {}
    document.documentElement.dataset.role = 'unauthenticated';
})();

// Session Accessors
export function getAuth() {
    try {
        const raw = sessionStorage.getItem('sanada_auth');
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed && (parsed.role === 'developer' || parsed.role === 'user')) {
            return parsed;
        }
    } catch (e) {}
    return null;
}

export function setAuth(role, username, extra = {}) {
    try {
        sessionStorage.setItem('sanada_auth', JSON.stringify({
            role: role,
            username: username || (role === 'developer' ? 'developer' : 'user'),
            id: extra.id || null,
            timestamp: Date.now()
        }));
    } catch (e) {
        console.warn('Storage write error', e);
    }
}

export function clearAuth() {
    try {
        sessionStorage.removeItem('sanada_auth');
    } catch (e) {
        console.warn('Storage remove error', e);
    }
}

// Global Toast helper fallback
function notify(message, type = 'info') {
    if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
        window.showToast(message, type);
    }
}

// 2. Role Display Synchronization
export function applyRoleDisplay(role) {
    const activeRole = (role === 'developer' || role === 'user') ? role : 'unauthenticated';

    if (typeof document !== 'undefined') {
        document.documentElement.dataset.role = activeRole;
        if (document.body) {
            document.body.dataset.role = activeRole;
        }

        // Trigger custom role event for SPA router
        window.dispatchEvent(new CustomEvent('sanada:roleChanged', { detail: { role: activeRole } }));
    }
}

// 3. Modal State & Controller (Sign In, Create Account, OTP Verification)
let currentModalRole = 'developer';
let currentModalView = 'signin'; // 'signin' | 'signup' | 'otp'
let pendingSignup = { username: '', email: '', password: '' };
let resendCooldown = 0;
let resendTimer = null;

export function switchModalView(viewName) {
    currentModalView = (viewName === 'signup' || viewName === 'otp') ? viewName : 'signin';

    const signinView = document.getElementById('modalSigninView');
    const signupView = document.getElementById('modalSignupView');
    const otpView = document.getElementById('modalOtpView');

    hideModalAlerts();

    if (signinView) signinView.style.display = (currentModalView === 'signin') ? 'block' : 'none';
    if (signupView) signupView.style.display = (currentModalView === 'signup') ? 'block' : 'none';
    if (otpView) otpView.style.display = (currentModalView === 'otp') ? 'block' : 'none';

    if (currentModalView === 'signin') {
        setTimeout(() => {
            document.getElementById('modalUsernameInput')?.focus();
        }, 50);
    } else if (currentModalView === 'signup') {
        setTimeout(() => {
            document.getElementById('modalSignupUsernameInput')?.focus();
        }, 50);
    } else if (currentModalView === 'otp') {
        const targetLabel = document.getElementById('modalOtpTargetEmail');
        if (targetLabel && pendingSignup.email) {
            targetLabel.textContent = pendingSignup.email;
        }
        setTimeout(() => {
            document.getElementById('modalOtpInput')?.focus();
        }, 50);
    }
}

export function setModalActiveRole(role) {
    currentModalRole = (role === 'user') ? 'user' : 'developer';

    const devCard = document.getElementById('modalRoleDevBtn');
    const userCard = document.getElementById('modalRoleUserBtn');
    const submitBtn = document.getElementById('modalSubmitBtn');
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    const usernameLabel = document.getElementById('modalUsernameLabel');
    hideModalAlerts();

    if (currentModalRole === 'developer') {
        devCard?.classList.add('active');
        devCard?.setAttribute('aria-selected', 'true');
        userCard?.classList.remove('active');
        userCard?.setAttribute('aria-selected', 'false');

        if (usernameLabel) usernameLabel.textContent = 'Developer Username';
        if (usernameInput) usernameInput.placeholder = 'Enter developer username';
        if (passwordInput) passwordInput.placeholder = 'Enter password';
        if (submitBtn) submitBtn.innerHTML = '<span>Log In as Developer</span>';
    } else {
        userCard?.classList.add('active');
        userCard?.setAttribute('aria-selected', 'true');
        devCard?.classList.remove('active');
        devCard?.setAttribute('aria-selected', 'false');

        if (usernameLabel) usernameLabel.textContent = 'Username';
        if (usernameInput) usernameInput.placeholder = 'Enter username';
        if (passwordInput) passwordInput.placeholder = 'Enter password';
        if (submitBtn) submitBtn.innerHTML = '<span>Log In as User</span>';
    }
}

export function openAuthModal(preferredRole) {
    const modal = document.getElementById('sanadaAuthModal');
    if (!modal) return;

    if (preferredRole) {
        setModalActiveRole(preferredRole);
    } else {
        setModalActiveRole(currentModalRole || 'developer');
    }

    // Default to sign-in view
    switchModalView('signin');

    // Keep fields empty when opening
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    if (usernameInput) usernameInput.value = '';
    if (passwordInput) passwordInput.value = '';

    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    document.body.style.overflow = 'hidden';

    // Focus input
    setTimeout(() => {
        usernameInput?.focus();
    }, 50);
}

export function closeAuthModal() {
    const modal = document.getElementById('sanadaAuthModal');
    if (!modal) return;

    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';

    // Clear fields on close
    const usernameInput = document.getElementById('modalUsernameInput');
    const passwordInput = document.getElementById('modalPasswordInput');
    const regUsernameInput = document.getElementById('modalSignupUsernameInput');
    const regEmailInput = document.getElementById('modalSignupEmailInput');
    const regPasswordInput = document.getElementById('modalSignupPasswordInput');
    const otpInput = document.getElementById('modalOtpInput');

    if (usernameInput) usernameInput.value = '';
    if (passwordInput) passwordInput.value = '';
    if (regUsernameInput) regUsernameInput.value = '';
    if (regEmailInput) regEmailInput.value = '';
    if (regPasswordInput) regPasswordInput.value = '';
    if (otpInput) otpInput.value = '';

    if (resendTimer) {
        clearInterval(resendTimer);
        resendTimer = null;
        resendCooldown = 0;
    }

    hideModalAlerts();
    switchModalView('signin');
}

function showModalError(message) {
    const errorBox = document.getElementById('modalAuthError');
    const errorText = document.getElementById('modalAuthErrorText');
    const successBox = document.getElementById('modalAuthSuccess');
    if (successBox) successBox.style.display = 'none';
    if (errorBox && errorText) {
        errorText.textContent = message;
        errorBox.style.display = 'flex';
    }
}

function showModalSuccess(message) {
    const successBox = document.getElementById('modalAuthSuccess');
    const successText = document.getElementById('modalAuthSuccessText');
    const errorBox = document.getElementById('modalAuthError');
    if (errorBox) errorBox.style.display = 'none';
    if (successBox && successText) {
        successText.textContent = message;
        successBox.style.display = 'flex';
    }
}

function hideModalAlerts() {
    const errorBox = document.getElementById('modalAuthError');
    const successBox = document.getElementById('modalAuthSuccess');
    if (errorBox) errorBox.style.display = 'none';
    if (successBox) successBox.style.display = 'none';
}

// 4. Authenticate & Role Routing
export async function authenticate(username, password, explicitRole) {
    const rawUser = (username || '').trim();
    const rawPass = (password || '').trim();

    const submitBtn = document.getElementById('modalSubmitBtn');
    const originalBtnContent = submitBtn ? submitBtn.innerHTML : '';
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Verifying...</span>';
    }

    const errorBox = document.getElementById('modalAuthError');
    if (errorBox) errorBox.style.display = 'none';

    try {
        // --- PATH A: Supabase Authentication ---
        if (isSupabaseConfigured()) {
            const res = await loginWithSupabase(rawUser, rawPass);
            if (!res.success) {
                showModalError(res.error || 'Invalid username or password.');
                return false;
            }

            const dbUser = res.user;
            const activeRole = (dbUser.role === 'developer') ? 'developer' : 'user';

            setAuth(activeRole, dbUser.username, { id: dbUser.id });
            applyRoleDisplay(activeRole);
            closeAuthModal();

            if (explicitRole && explicitRole !== activeRole) {
                notify(`Signed in as ${activeRole.toUpperCase()} (account is registered as ${activeRole})`, 'info');
            } else {
                notify(`Signed in as ${activeRole === 'developer' ? 'Developer (Full Suite)' : 'User (Inference Studio)'}`, 'success');
            }

            // If on login.html, route to index.html
            if (window.location.pathname.endsWith('login.html')) {
                window.location.href = 'index.html';
            }
            return true;
        }

        // --- PATH B: Local Demo Mode Fallback (when Supabase keys not set yet) ---
        const u = rawUser.toLowerCase();
        let matchedRole = explicitRole;
        if (!matchedRole) {
            if (u === DEMO_CONFIG.developer.username) matchedRole = 'developer';
            else if (u === DEMO_CONFIG.user.username) matchedRole = 'user';
            else matchedRole = currentModalRole;
        }

        const cfg = DEMO_CONFIG[matchedRole];
        if (cfg && u === cfg.username.toLowerCase() && rawPass === cfg.password) {
            setAuth(cfg.role, cfg.username);
            applyRoleDisplay(cfg.role);
            closeAuthModal();

            notify(`Signed in as ${cfg.label} (Demo fallback — configure Supabase in frontend/js/services/supabase.js)`, 'info');

            if (window.location.pathname.endsWith('login.html')) {
                window.location.href = 'index.html';
            }
            return true;
        }

        showModalError('Supabase is not configured yet. Please paste your SUPABASE_URL and SUPABASE_ANON_KEY into frontend/js/services/supabase.js (or use developer / dev123 for demo mode).');
        return false;
    } catch (err) {
        console.error('[SANADA Auth] Unexpected error during authentication:', err);
        showModalError(err.message || 'Authentication error. Please try again.');
        return false;
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnContent;
        }
    }
}

// 5. Logout Confirmation & Action
export function openLogoutModal() {
    const modal = document.getElementById('sanadaLogoutModal');
    if (!modal) {
        if (confirm('Are you sure you want to log out of SANADA?')) {
            logout();
        }
        return;
    }
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
}

export function closeLogoutModal() {
    const modal = document.getElementById('sanadaLogoutModal');
    if (!modal) return;
    modal.classList.remove('open');
    modal.setAttribute('aria-hidden', 'true');
    const authModal = document.getElementById('sanadaAuthModal');
    if (!authModal || !authModal.classList.contains('open')) {
        document.body.style.overflow = '';
    }
}

export function logout() {
    clearAuth();
    applyRoleDisplay('unauthenticated');
    closeAuthModal();
    closeLogoutModal();

    notify('Logged out. Returned to landing page.', 'info');

    // Route to home on SPA
    if (typeof window !== 'undefined') {
        if (window.location.pathname.endsWith('user.html') || window.location.pathname.endsWith('login.html')) {
            window.location.href = 'index.html#/home';
        } else {
            window.location.hash = '#/home';
            if (typeof window.switchRoute === 'function') {
                window.switchRoute('home');
            }
        }
    }
}

// 6. DOM Event Bindings
function initAuthGuard() {
    // Sync current session state with DOM
    const current = getAuth();
    applyRoleDisplay(current ? current.role : 'unauthenticated');

    // Hero Login button
    const heroLoginBtn = document.getElementById('heroLoginBtn');
    if (heroLoginBtn) {
        heroLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openAuthModal('developer');
        });
    }

    // Top Navbar Login button
    const navLoginBtn = document.getElementById('navLoginBtn');
    if (navLoginBtn) {
        navLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            openAuthModal('developer');
        });
    }

    // Logout buttons: ask confirmation before logging out
    const logoutBtns = document.querySelectorAll('#logoutBtn, .btn-logout, .logout-link, .mobile-menu-logout, #mobileMenuLogoutBtn');
    logoutBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            // Close mobile menu if open
            const navTabs = document.querySelector('.navbar .nav-tabs');
            const navToggle = document.getElementById('navToggle');
            if (navTabs) navTabs.classList.remove('open');
            if (navToggle) {
                navToggle.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
            }
            openLogoutModal();
        });
    });

    // Logout Confirmation Modal controls
    const logoutCancelBtn = document.getElementById('modalLogoutCancelBtn');
    logoutCancelBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
    });

    const logoutCloseBtn = document.getElementById('modalLogoutCloseBtn');
    logoutCloseBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
    });

    const logoutConfirmBtn = document.getElementById('modalLogoutConfirmBtn');
    logoutConfirmBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        closeLogoutModal();
        logout();
    });

    const logoutModal = document.getElementById('sanadaLogoutModal');
    if (logoutModal) {
        logoutModal.addEventListener('click', (e) => {
            if (e.target === logoutModal) {
                closeLogoutModal();
            }
        });
    }

    // Modal Close button
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            closeAuthModal();
        });
    }

    // Close modal if user clicks Home tab or brand in navbar
    const homeTab = document.querySelector('.navbar .nav-tabs .tab-link[data-route="home"]');
    homeTab?.addEventListener('click', () => closeAuthModal());
    const brandLink = document.querySelector('.navbar .brand');
    brandLink?.addEventListener('click', () => closeAuthModal());

    // Modal Backdrop click
    const modal = document.getElementById('sanadaAuthModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeAuthModal();
            }
        });
    }

    // Escape key listener for both modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (logoutModal && logoutModal.classList.contains('open')) {
                closeLogoutModal();
            } else if (modal && modal.classList.contains('open')) {
                closeAuthModal();
            }
        }
    });

    // Dual Role selector cards in modal
    const devCard = document.getElementById('modalRoleDevBtn');
    const userCard = document.getElementById('modalRoleUserBtn');
    if (devCard) {
        devCard.addEventListener('click', () => setModalActiveRole('developer'));
    }
    if (userCard) {
        userCard.addEventListener('click', () => setModalActiveRole('user'));
    }

    // Modal Login Form submission
    const modalForm = document.getElementById('modalLoginForm');
    const modalUserInput = document.getElementById('modalUsernameInput');
    const modalPassInput = document.getElementById('modalPasswordInput');

    // Auto-clear error when user types
    modalUserInput?.addEventListener('input', () => {
        const errorBox = document.getElementById('modalAuthError');
        if (errorBox) errorBox.style.display = 'none';
    });
    modalPassInput?.addEventListener('input', () => {
        const errorBox = document.getElementById('modalAuthError');
        if (errorBox) errorBox.style.display = 'none';
    });

    if (modalForm) {
        modalForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const u = modalUserInput?.value;
            const p = modalPassInput?.value;
            await authenticate(u, p, currentModalRole);
        });
    }

    // Password reveal toggles (guaranteed single-execution via helper)
    const passToggle = document.getElementById('modalPassToggle');
    if (passToggle && !passToggle.hasAttribute('onclick')) {
        passToggle.addEventListener('click', (e) => {
            e.preventDefault();
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility('modalPasswordInput', passToggle);
            }
        });
    }

    // Switch between Sign In and Create Account
    const switchToSignupBtn = document.getElementById('modalSwitchToSignupBtn');
    switchToSignupBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        switchModalView('signup');
    });

    const switchToSigninBtn = document.getElementById('modalSwitchToSigninBtn');
    switchToSigninBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        switchModalView('signin');
    });

    const otpBackBtn = document.getElementById('modalOtpBackBtn');
    otpBackBtn?.addEventListener('click', (e) => {
        e.preventDefault();
        switchModalView('signup');
    });

    // Signup password reveal toggle
    const regPassToggle = document.getElementById('modalSignupPassToggle');
    if (regPassToggle && !regPassToggle.hasAttribute('onclick')) {
        regPassToggle.addEventListener('click', (e) => {
            e.preventDefault();
            if (typeof window.togglePasswordVisibility === 'function') {
                window.togglePasswordVisibility('modalSignupPasswordInput', regPassToggle);
            }
        });
    }

    // Auto-clear alerts when user types in signup & OTP fields
    ['modalSignupUsernameInput', 'modalSignupEmailInput', 'modalSignupPasswordInput', 'modalOtpInput'].forEach(id => {
        const el = document.getElementById(id);
        el?.addEventListener('input', () => {
            hideModalAlerts();
        });
    });

    // Create Account Form submission (Sign Up -> send OTP)
    const signupForm = document.getElementById('modalSignupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideModalAlerts();

            const uInput = document.getElementById('modalSignupUsernameInput');
            const eInput = document.getElementById('modalSignupEmailInput');
            const pInput = document.getElementById('modalSignupPasswordInput');
            const submitBtn = document.getElementById('modalSignupSubmitBtn');

            const username = (uInput?.value || '').trim();
            const email = (eInput?.value || '').trim().toLowerCase();
            const password = (pInput?.value || '').trim();

            if (!username) {
                showModalError('Please choose a username.');
                uInput?.focus();
                return;
            }
            if (username.length < 3) {
                showModalError('Username must be at least 3 characters.');
                uInput?.focus();
                return;
            }
            if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
                showModalError('Username can only contain letters, numbers, underscores, and hyphens.');
                uInput?.focus();
                return;
            }
            if (!email || !email.includes('@') || !email.includes('.')) {
                showModalError('Please enter a valid email address.');
                eInput?.focus();
                return;
            }
            if (!password || password.length < 6) {
                showModalError('Password must be at least 6 characters long.');
                pInput?.focus();
                return;
            }

            const origContent = submitBtn ? submitBtn.innerHTML : '';
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span>Sending Code...</span>';
            }

            try {
                if (!isSupabaseConfigured()) {
                    showModalError('Supabase is not configured yet. Paste your credentials into frontend/js/services/supabase.js');
                    return;
                }

                const res = await signUpWithSupabase(username, email, password);
                if (!res.success) {
                    showModalError(res.error || 'Failed to send verification code. Please try again.');
                    return;
                }

                pendingSignup = { username, email, password };
                switchModalView('otp');
                showModalSuccess(`A 6-digit verification code was sent to ${email}.`);
                startResendCooldown(60);
            } catch (err) {
                console.error('[SANADA Auth] Signup error:', err);
                showModalError(err.message || 'An unexpected error occurred during account creation.');
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = origContent;
                }
            }
        });
    }

    // Resend OTP button
    const otpResendBtn = document.getElementById('modalOtpResendBtn');
    otpResendBtn?.addEventListener('click', async (e) => {
        e.preventDefault();
        if (resendCooldown > 0) return;
        hideModalAlerts();

        const btn = document.getElementById('modalOtpResendBtn');
        if (btn) btn.textContent = 'Sending...';

        try {
            const res = await resendOtpWithSupabase(pendingSignup.email);
            if (!res.success) {
                showModalError(res.error || 'Failed to resend code.');
                if (btn) btn.textContent = 'Resend Code';
            } else {
                showModalSuccess(`A new 6-digit code has been sent to ${pendingSignup.email}.`);
                startResendCooldown(60);
            }
        } catch (err) {
            showModalError(err.message || 'Failed to resend code.');
            if (btn) btn.textContent = 'Resend Code';
        }
    });

    // OTP Form submission (Verify OTP -> Activate -> Sign In)
    const otpForm = document.getElementById('modalOtpForm');
    if (otpForm) {
        otpForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideModalAlerts();

            const otpInput = document.getElementById('modalOtpInput');
            const submitBtn = document.getElementById('modalOtpSubmitBtn');
            const token = (otpInput?.value || '').trim();

            if (!token) {
                showModalError('Please enter the 6-digit verification code.');
                otpInput?.focus();
                return;
            }

            const origContent = submitBtn ? submitBtn.innerHTML : '';
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span>Verifying Code...</span>';
            }

            try {
                const res = await verifyOtpWithSupabase(
                    pendingSignup.email,
                    token,
                    pendingSignup.username,
                    pendingSignup.password
                );

                if (!res.success) {
                    showModalError(res.error || 'Invalid or expired verification code.');
                    return;
                }

                const user = res.user;
                const username = user?.username || pendingSignup.username || 'user';
                const role = 'user';

                setAuth(role, username, { id: user?.id, email: pendingSignup.email });
                applyRoleDisplay(role);
                closeAuthModal();

                notify(`Account verified successfully! Welcome to SANADA, ${username}.`, 'success');

                // Clear pending signup data
                pendingSignup = { username: '', email: '', password: '' };

                if (window.location.pathname.endsWith('login.html')) {
                    window.location.href = 'index.html';
                }
            } catch (err) {
                console.error('[SANADA Auth] OTP verification error:', err);
                showModalError(err.message || 'Verification failed. Please try again.');
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = origContent;
                }
            }
        });
    }
}

// Make functions accessible globally for plain script tags or inline calls
if (typeof window !== 'undefined') {
    window.SANADA_AUTH = {
        getAuth,
        setAuth,
        clearAuth,
        applyRoleDisplay,
        openAuthModal,
        closeAuthModal,
        switchModalView,
        openLogoutModal,
        closeLogoutModal,
        authenticate,
        logout,
        isSupabaseConfigured,
        DEMO_CONFIG
    };
    window.logout = logout;
    window.openAuthModal = openAuthModal;
    window.closeAuthModal = closeAuthModal;
    window.switchModalView = switchModalView;
    window.openLogoutModal = openLogoutModal;
    window.closeLogoutModal = closeLogoutModal;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAuthGuard);
    } else {
        initAuthGuard();
    }
}

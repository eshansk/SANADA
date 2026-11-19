/**
 * frontend/js/services/supabase.js
 * ================================
 * Supabase Client & Authentication Service for SANADA.
 *
 * Integrates directly with your Supabase PostgreSQL database:
 *  - Table: public.app_users (username, password, role)
 *  - Trigger: trg_auto_hash_password (automatic bcrypt password encryption)
 *  - RPC Procedure: public.authenticate_user(p_username, p_password)
 *
 * CONFIGURATION INSTRUCTIONS:
 * ---------------------------
 * In your Supabase Dashboard:
 * 1. Navigate to Project Settings -> API.
 * 2. Copy the "Project URL" and paste it into SUPABASE_URL below.
 * 3. Copy the "Project API keys" -> anon / public key and paste it into SUPABASE_ANON_KEY below.
 *
 * You can also set them dynamically in the browser console via:
 *   window.setSupabaseConfig('https://your-project.supabase.co', 'your-anon-key');
 */

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

// =============================================================================
// 1. CREDENTIALS CONFIGURATION
// =============================================================================
export const SUPABASE_URL = (typeof window !== 'undefined' && window.SANADA_SUPABASE_URL)
    || (typeof localStorage !== 'undefined' && localStorage.getItem('sanada_supabase_url'))
    || 'https://elgsaepnceawrkswdpnv.supabase.co';

export const SUPABASE_ANON_KEY = (typeof window !== 'undefined' && window.SANADA_SUPABASE_ANON_KEY)
    || (typeof localStorage !== 'undefined' && localStorage.getItem('sanada_supabase_anon_key'))
    || 'sb_publishable_O-6zrNb2byz6he6Lx2Wg2g_pZKTicN1';

/**
 * Returns the currently active Supabase URL (from code, window, or localStorage).
 */
export function getSupabaseUrl() {
    if (typeof window !== 'undefined' && window.SANADA_SUPABASE_URL) return window.SANADA_SUPABASE_URL;
    if (typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('sanada_supabase_url');
        if (stored) return stored.trim();
    }
    return SUPABASE_URL;
}

/**
 * Returns the currently active Supabase Anon Key (from code, window, or localStorage).
 */
export function getSupabaseKey() {
    if (typeof window !== 'undefined' && window.SANADA_SUPABASE_ANON_KEY) return window.SANADA_SUPABASE_ANON_KEY;
    if (typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('sanada_supabase_anon_key');
        if (stored) return stored.trim();
    }
    return SUPABASE_ANON_KEY;
}

/**
 * Checks if Supabase credentials have been configured with valid non-placeholder values.
 * @returns {boolean}
 */
export function isSupabaseConfigured() {
    const url = getSupabaseUrl();
    const key = getSupabaseKey();
    return Boolean(
        url
        && typeof url === 'string'
        && url.startsWith('https://')
        && !url.includes('YOUR_PROJECT_ID')
        && !url.includes('YOUR_SUPABASE')
        && key
        && typeof key === 'string'
        && !key.includes('YOUR_SUPABASE_ANON_KEY')
        && key.length > 20
    );
}

// =============================================================================
// 2. SUPABASE CLIENT SINGLETON
// =============================================================================
let _clientInstance = null;

/**
 * Lazily instantiates and returns the Supabase client.
 * Returns null if credentials are not configured.
 */
export function getSupabaseClient() {
    if (_clientInstance) return _clientInstance;

    if (!isSupabaseConfigured()) {
        return null;
    }

    try {
        const url = getSupabaseUrl();
        const key = getSupabaseKey();
        _clientInstance = createClient(url, key, {
            auth: {
                persistSession: false,
                autoRefreshToken: false,
                detectSessionInUrl: false
            }
        });
        return _clientInstance;
    } catch (err) {
        console.error('[SANADA Supabase] Client initialization failed:', err);
        return null;
    }
}

export const supabase = getSupabaseClient();

// =============================================================================
// 3. USER AUTHENTICATION & REGISTRATION RPC / AUTH
// =============================================================================
/**
 * Authenticates a user against public.authenticate_user(p_username, p_password),
 * with fallback to client.auth.signInWithPassword for Supabase Auth accounts.
 *
 * @param {string} usernameOrEmail - User account username or email
 * @param {string} password - Raw plain-text password
 * @returns {Promise<{
 *   success: boolean,
 *   user?: { id: string, username: string, role: string, email?: string },
 *   error?: string
 * }>}
 */
export async function loginWithSupabase(usernameOrEmail, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured yet. Paste your SUPABASE_URL and SUPABASE_ANON_KEY into frontend/js/services/supabase.js'
        };
    }

    const cleanUser = (usernameOrEmail || '').trim();
    const cleanPass = (password || '').trim();

    if (!cleanUser || !cleanPass) {
        return {
            success: false,
            error: 'Please enter both username and password.'
        };
    }

    try {
        // 1. First attempt: PostgreSQL authenticate_user RPC function
        const { data, error } = await client.rpc('authenticate_user', {
            p_username: cleanUser,
            p_password: cleanPass
        });

        if (!error && Array.isArray(data) && data.length > 0) {
            const match = data[0];
            return {
                success: true,
                user: {
                    id: match.user_id || match.id || null,
                    username: match.username,
                    role: (match.role || 'user').toLowerCase()
                }
            };
        }

        // 2. Second attempt: Check Supabase Auth (for accounts created with email & password)
        try {
            const authRes = await client.auth.signInWithPassword({
                email: cleanUser,
                password: cleanPass
            });

            if (!authRes.error && authRes.data && authRes.data.user) {
                const suUser = authRes.data.user;
                const metadata = suUser.user_metadata || {};
                const resolvedUsername = metadata.username || (suUser.email ? suUser.email.split('@')[0] : 'user');
                const resolvedRole = (metadata.role || 'user').toLowerCase();
                return {
                    success: true,
                    user: {
                        id: suUser.id,
                        username: resolvedUsername,
                        email: suUser.email,
                        role: resolvedRole
                    }
                };
            }
        } catch (authErr) {
            // Ignore auth fallback error, proceed to invalid credentials response
        }

        // If authenticate_user had an actual code error (e.g. function missing), report it
        if (error && error.code === '42883') {
            return {
                success: false,
                error: 'Database function authenticate_user() not found. Please run the SQL setup script in Supabase SQL Editor.'
            };
        }

        return {
            success: false,
            error: 'Invalid username or password.'
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected network error:', err);
        return {
            success: false,
            error: err.message || 'Network error while contacting Supabase. Please check your internet connection.'
        };
    }
}

/**
 * Initiates user registration by creating an account in Supabase Auth
 * and triggering an email with a 6-digit OTP confirmation token.
 *
 * @param {string} username - Chosen username
 * @param {string} email - Valid email address to receive OTP
 * @param {string} password - Plain-text password (min 6 chars)
 * @returns {Promise<{ success: boolean, user?: any, error?: string }>}
 */
export async function signUpWithSupabase(username, email, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured yet.'
        };
    }

    const cleanUser = (username || '').trim();
    const cleanEmail = (email || '').trim().toLowerCase();
    const cleanPass = (password || '').trim();

    if (!cleanUser) {
        return { success: false, error: 'Please enter a username.' };
    }
    if (cleanUser.length < 3) {
        return { success: false, error: 'Username must be at least 3 characters long.' };
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(cleanUser)) {
        return { success: false, error: 'Username can only contain letters, numbers, underscores, and hyphens.' };
    }

    if (!cleanEmail || !cleanEmail.includes('@') || !cleanEmail.includes('.')) {
        return { success: false, error: 'Please enter a valid email address.' };
    }

    if (!cleanPass) {
        return { success: false, error: 'Please enter a password.' };
    }
    if (cleanPass.length < 6) {
        return { success: false, error: 'Password must be at least 6 characters long.' };
    }

    try {
        const { data, error } = await client.auth.signUp({
            email: cleanEmail,
            password: cleanPass,
            options: {
                data: {
                    username: cleanUser,
                    role: 'user'
                }
            }
        });

        if (error) {
            console.error('[SANADA Supabase] Sign up error:', error);
            let friendlyMsg = error.message;
            const errStr = (error.message || '').toLowerCase();
            if (error.status === 429 || errStr.includes('rate limit')) {
                friendlyMsg = 'Email rate limit reached. Please wait a short moment before requesting another code.';
            } else if (errStr.includes('already registered')) {
                friendlyMsg = 'An account with this email already exists. Please sign in instead.';
            } else if (errStr.includes('invalid') && errStr.includes('email')) {
                friendlyMsg = 'Please enter a valid email address (disposable email domains are blocked by Supabase).';
            }
            return {
                success: false,
                error: friendlyMsg
            };
        }

        // If identities is an empty array, the user already exists in auth.users
        if (data.user && Array.isArray(data.user.identities) && data.user.identities.length === 0) {
            return {
                success: false,
                error: 'An account with this email already exists. Please sign in instead.'
            };
        }

        return {
            success: true,
            user: data.user,
            email: cleanEmail,
            username: cleanUser
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected sign up error:', err);
        return {
            success: false,
            error: err.message || 'Network error during account registration.'
        };
    }
}

/**
 * Verifies the 6-digit OTP code sent to the user's email during registration.
 *
 * @param {string} email - User email address
 * @param {string} token - 6-digit OTP code entered by the user
 * @param {string} username - Username associated with registration
 * @param {string} password - Password used during registration (to sync to app_users)
 * @returns {Promise<{
 *   success: boolean,
 *   user?: { id: string, username: string, email: string, role: string },
 *   session?: any,
 *   error?: string
 * }>}
 */
export async function verifyOtpWithSupabase(email, token, username, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured.'
        };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    const cleanToken = (token || '').trim();

    if (!cleanEmail) {
        return { success: false, error: 'Missing email address for verification.' };
    }
    if (!cleanToken) {
        return { success: false, error: 'Please enter the verification code sent to your email.' };
    }

    try {
        // Verify OTP for signup confirmation
        let res = await client.auth.verifyOtp({
            email: cleanEmail,
            token: cleanToken,
            type: 'signup'
        });

        // Fallback check with 'email' type if 'signup' type returns expired/invalid
        if (res.error && (res.error.message.includes('expired') || res.error.status === 403)) {
            const fallbackRes = await client.auth.verifyOtp({
                email: cleanEmail,
                token: cleanToken,
                type: 'email'
            });
            if (!fallbackRes.error && fallbackRes.data) {
                res = fallbackRes;
            }
        }

        if (res.error) {
            console.error('[SANADA Supabase] verifyOtp error:', res.error);
            let friendly = res.error.message;
            const errLower = (res.error.message || '').toLowerCase();
            if (errLower.includes('expired') || errLower.includes('invalid') || res.error.status === 403) {
                friendly = 'Invalid or expired 6-digit code. Please verify the code in your email or click Resend.';
            }
            return {
                success: false,
                error: friendly
            };
        }

        const user = res.data.user;
        const resolvedUsername = username || (user?.user_metadata?.username) || cleanEmail.split('@')[0];

        // Sync into public.app_users via RPC if available
        if (password) {
            try {
                await client.rpc('register_verified_user', {
                    p_username: resolvedUsername,
                    p_password: password,
                    p_email: cleanEmail
                });
            } catch (rpcErr) {
                console.info('[SANADA Supabase] register_verified_user notice:', rpcErr?.message || rpcErr);
            }
        }

        return {
            success: true,
            user: {
                id: user?.id || null,
                username: resolvedUsername,
                email: cleanEmail,
                role: 'user'
            },
            session: res.data.session
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected verifyOtp error:', err);
        return {
            success: false,
            error: err.message || 'Verification failed. Please try again.'
        };
    }
}

/**
 * Resends the signup OTP verification code to the specified email address.
 *
 * @param {string} email - Email address to resend OTP to
 * @returns {Promise<{ success: boolean, error?: string }>}
 */
export async function resendOtpWithSupabase(email) {
    const client = getSupabaseClient();
    if (!client) {
        return { success: false, error: 'Supabase credentials are not configured.' };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    if (!cleanEmail) {
        return { success: false, error: 'Missing email address.' };
    }

    try {
        const { data, error } = await client.auth.resend({
            type: 'signup',
            email: cleanEmail
        });

        if (error) {
            console.error('[SANADA Supabase] Resend error:', error);
            let msg = error.message;
            if (error.status === 429 || error.message.toLowerCase().includes('rate limit') || error.message.toLowerCase().includes('security purposes')) {
                msg = error.message.includes('after') ? error.message : 'Please wait a short moment before requesting another code.';
            }
            return { success: false, error: msg };
        }

        return { success: true };
    } catch (err) {
        return { success: false, error: err.message || 'Failed to resend verification code.' };
    }
}

// =============================================================================
// 4. CONSOLE CONFIGURATION HELPER
// =============================================================================
// Allows setting or updating credentials instantly in devtools without touching files:
// window.setSupabaseConfig('https://xyz.supabase.co', 'eyJh...')
if (typeof window !== 'undefined') {
    window.setSupabaseConfig = function(url, anonKey) {
        if (!url || !anonKey) {
            console.error('[SANADA] Usage: setSupabaseConfig("https://xyz.supabase.co", "eyJhbGciOi...")');
            return;
        }
        localStorage.setItem('sanada_supabase_url', url.trim());
        localStorage.setItem('sanada_supabase_anon_key', anonKey.trim());
        console.log('[SANADA] Supabase credentials saved to localStorage!');
        console.log('[SANADA] Reloading to apply credentials...');
        window.location.reload();
    };

    window.SANADA_SUPABASE = {
        isConfigured: isSupabaseConfigured,
        getClient: getSupabaseClient,
        login: loginWithSupabase,
        signUp: signUpWithSupabase,
        verifyOtp: verifyOtpWithSupabase,
        resendOtp: resendOtpWithSupabase,
        getSupabaseUrl,
        getSupabaseKey
    };
}

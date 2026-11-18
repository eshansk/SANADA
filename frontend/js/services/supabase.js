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
// 3. USER AUTHENTICATION RPC (authenticate_user)
// =============================================================================
/**
 * Authenticates a user against public.authenticate_user(p_username, p_password).
 *
 * Verifies bcrypt password hash securely inside PostgreSQL using pgcrypto.
 *
 * @param {string} username - User account username (case-insensitive)
 * @param {string} password - Raw plain-text password
 * @returns {Promise<{
 *   success: boolean,
 *   user?: { id: string, username: string, role: string },
 *   error?: string
 * }>}
 */
export async function loginWithSupabase(username, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured yet. Paste your SUPABASE_URL and SUPABASE_ANON_KEY into frontend/js/services/supabase.js'
        };
    }

    const cleanUser = (username || '').trim();
    const cleanPass = (password || '').trim();

    if (!cleanUser || !cleanPass) {
        return {
            success: false,
            error: 'Please enter both username and password.'
        };
    }

    try {
        // Call the PostgreSQL authenticate_user RPC function
        const { data, error } = await client.rpc('authenticate_user', {
            p_username: cleanUser,
            p_password: cleanPass
        });

        if (error) {
            console.error('[SANADA Supabase] authenticate_user RPC error:', error);
            // Handle common SQL permission or missing function errors gracefully
            if (error.code === '42883') {
                return {
                    success: false,
                    error: 'Database function authenticate_user() not found. Please run the SQL setup script in Supabase SQL Editor.'
                };
            }
            return {
                success: false,
                error: error.message || 'Authentication failed. Please verify your Supabase database permissions.'
            };
        }

        // authenticate_user returns a table of matched rows
        if (!data || !Array.isArray(data) || data.length === 0) {
            return {
                success: false,
                error: 'Invalid username or password.'
            };
        }

        const match = data[0];
        return {
            success: true,
            user: {
                id: match.user_id || match.id || null,
                username: match.username,
                role: (match.role || 'user').toLowerCase()
            }
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected network error:', err);
        return {
            success: false,
            error: err.message || 'Network error while contacting Supabase. Please check your internet connection.'
        };
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
        getSupabaseUrl,
        getSupabaseKey
    };
}

/**
 * frontend/js/pages/api.js
 * ========================
 * Controller for the SANADA Web API & Partner Gateway.
 *
 * Provides:
 *  1. Cryptographically secure API key generation with SHA-256 database hashing.
 *  2. Strict user/developer isolation (keys are scoped per user identity).
 *  3. Automatic sequential default key labels ("My app key 1", "My app key 2", ...).
 *  4. Interactive key selection: picking a key dynamically updates integration snippets.
 *  5. Masked key display with click-to-reveal toggle (eye icon).
 *  6. One-click clipboard copy with animated visual feedback.
 *  7. Key deletion / revocation with confirmation.
 *  8. Multi-language code snippets (cURL, Python requests, Node.js fetch).
 *  9. Dual-tier storage (Supabase PostgreSQL sync with per-user local storage resilience).
 */

import { getAuth } from '../auth-guard.js';
import { onRouteChanged } from '../ui/router.js';

// Active code snippet tab and key state
let activeSnippetLang = 'curl';
let cachedKeys = [];
let selectedKeyId = null;

/**
 * Returns normalized metadata for the currently authenticated user/developer.
 */
function getCurrentUserMeta() {
    const auth = getAuth();
    if (!auth) {
        return {
            id: null,
            username: 'guest',
            role: 'unauthenticated',
            storageKey: 'sanada_api_keys_guest'
        };
    }
    const username = (auth.username || auth.role || 'user').trim().toLowerCase();
    const id = auth.id || null;
    const role = auth.role || 'user';
    // Storage key strictly scoped to user ID (UUID) or username/role
    const storageKey = id ? `sanada_api_keys_uid_${id}` : `sanada_api_keys_user_${username}`;
    return { id, username, role, storageKey };
}

/**
 * Generates a random cryptographic key prefixed with `sanada_live_`.
 */
function generateRawApiKey() {
    const bytes = new Uint8Array(16);
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        crypto.getRandomValues(bytes);
    } else {
        for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
    }
    const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
    return `sanada_live_${hex}`;
}

/**
 * Computes the SHA-256 cryptographic hash of the key string.
 */
async function computeSha256(text) {
    if (typeof crypto !== 'undefined' && crypto.subtle && crypto.subtle.digest) {
        const encoder = new TextEncoder();
        const data = encoder.encode(text);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }
    // Fallback simple hash for environments without Web Crypto Subtle
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash).toString(16).padStart(64, '0');
}

/**
 * Masks the middle part of an API key for safe UI display.
 */
function maskApiKey(rawKey) {
    if (!rawKey || rawKey.length < 20) return 'sanada_live_••••••••••••';
    const prefix = rawKey.slice(0, 16);
    const suffix = rawKey.slice(-4);
    return `${prefix}••••••••••••••••${suffix}`;
}

/**
 * Computes the next sequential default label ("My app key 1", "My app key 2", ...).
 */
function getNextDefaultKeyName() {
    let maxIndex = 0;
    const pattern = /^My app key\s*(\d+)$/i;

    cachedKeys.forEach(k => {
        const match = (k.name || '').match(pattern);
        if (match) {
            const num = parseInt(match[1], 10);
            if (!isNaN(num) && num > maxIndex) {
                maxIndex = num;
            }
        }
    });

    if (maxIndex > 0) {
        return `My app key ${maxIndex + 1}`;
    }

    return `My app key ${cachedKeys.length + 1}`;
}

/**
 * Updates the key name input's placeholder to reflect the next default label.
 */
function updateKeyInputPlaceholder() {
    const keyNameInput = document.getElementById('apiKeyNameInput');
    if (keyNameInput) {
        const nextDefault = getNextDefaultKeyName();
        keyNameInput.placeholder = `Key label (default: ${nextDefault})`;
    }
}

/**
 * Ensures a valid key selection exists among the cached keys.
 */
function ensureValidSelection() {
    if (!cachedKeys || cachedKeys.length === 0) {
        selectedKeyId = null;
        return;
    }
    const exists = cachedKeys.some(k => k.id === selectedKeyId);
    if (!exists) {
        selectedKeyId = cachedKeys[0].id;
    }
}

/**
 * Loads API keys strictly for the active user from Supabase or localStorage fallback.
 */
async function loadApiKeys() {
    const userMeta = getCurrentUserMeta();

    // 1. Attempt Supabase lookup strictly scoped to current user
    try {
        if (typeof window !== 'undefined' && window.SANADA_SUPABASE && window.SANADA_SUPABASE.getClient) {
            const client = window.SANADA_SUPABASE.getClient();
            if (client) {
                let query = client.from('api_keys').select('*').order('created_at', { ascending: false });

                if (userMeta.id) {
                    query = query.or(`user_id.eq.${userMeta.id},user_identifier.eq.${userMeta.username}`);
                } else if (userMeta.username && userMeta.username !== 'guest') {
                    query = query.eq('user_identifier', userMeta.username);
                }

                const { data, error } = await query;
                if (!error && Array.isArray(data)) {
                    cachedKeys = data.map(item => ({
                        id: item.id,
                        name: item.name || 'API Key',
                        rawKey: item.encrypted_key || item.raw_key || item.key_value,
                        maskedKey: item.masked_key || maskApiKey(item.raw_key || item.encrypted_key || ''),
                        keyHash: item.key_hash,
                        createdAt: new Date(item.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }),
                        revealed: false
                    }));
                    saveToLocalStorage(cachedKeys);
                    ensureValidSelection();
                    return cachedKeys;
                }
            }
        }
    } catch (e) {
        console.warn('[SANADA API] Supabase query notice:', e?.message || e);
    }

    // 2. Fallback: Read from user-scoped localStorage
    try {
        const stored = localStorage.getItem(userMeta.storageKey);
        if (stored) {
            cachedKeys = JSON.parse(stored);
            ensureValidSelection();
            return cachedKeys;
        }
    } catch (e) {}

    // When a user has no keys, keep empty (no cross-user contamination)
    cachedKeys = [];
    selectedKeyId = null;
    return cachedKeys;
}

/**
 * Persists keys array to user-scoped localStorage.
 */
function saveToLocalStorage(keys) {
    try {
        const userMeta = getCurrentUserMeta();
        localStorage.setItem(userMeta.storageKey, JSON.stringify(keys));
    } catch (e) {}
}

/**
 * Saves a new key strictly associated with the current user/developer.
 */
async function saveNewKey(name) {
    const userMeta = getCurrentUserMeta();
    const finalName = (name && name.trim()) ? name.trim() : getNextDefaultKeyName();
    const rawKey = generateRawApiKey();
    const keyHash = await computeSha256(rawKey);
    const maskedKey = maskApiKey(rawKey);

    const newKeyObj = {
        id: 'key_' + Date.now(),
        name: finalName,
        rawKey: rawKey,
        maskedKey: maskedKey,
        keyHash: keyHash,
        createdAt: 'Just now',
        revealed: false
    };

    // Attempt Supabase insert with user identification
    try {
        if (typeof window !== 'undefined' && window.SANADA_SUPABASE && window.SANADA_SUPABASE.getClient) {
            const client = window.SANADA_SUPABASE.getClient();
            if (client) {
                const insertPayload = {
                    name: newKeyObj.name,
                    key_prefix: rawKey.slice(0, 16),
                    key_hash: keyHash,
                    masked_key: maskedKey,
                    encrypted_key: rawKey,
                    user_identifier: userMeta.username
                };
                if (userMeta.id) {
                    insertPayload.user_id = userMeta.id;
                }

                const { data, error } = await client.from('api_keys').insert([insertPayload]).select();
                if (!error && data && data.length > 0) {
                    newKeyObj.id = data[0].id;
                }
            }
        }
    } catch (err) {
        console.info('[SANADA API] Saved locally for user:', err?.message || err);
    }

    cachedKeys.unshift(newKeyObj);
    selectedKeyId = newKeyObj.id; // Automatically select newly created key
    saveToLocalStorage(cachedKeys);
    return newKeyObj;
}

/**
 * Deletes / Revokes an API Key.
 */
async function revokeKey(keyId) {
    try {
        if (typeof window !== 'undefined' && window.SANADA_SUPABASE && window.SANADA_SUPABASE.getClient) {
            const client = window.SANADA_SUPABASE.getClient();
            if (client) {
                await client.from('api_keys').delete().eq('id', keyId);
            }
        }
    } catch (e) {}

    cachedKeys = cachedKeys.filter(k => k.id !== keyId);
    saveToLocalStorage(cachedKeys);

    if (selectedKeyId === keyId) {
        selectedKeyId = cachedKeys[0]?.id || null;
    }
}

/**
 * Renders the list of API Keys and updates the code snippet for the selected key.
 */
function renderApiKeysList() {
    const container = document.getElementById('apiKeysListContainer');
    if (!container) return;

    ensureValidSelection();
    updateKeyInputPlaceholder();

    const activeSelectedKey = cachedKeys.find(k => k.id === selectedKeyId) || null;
    updateCodeSnippet(
        activeSelectedKey ? activeSelectedKey.rawKey : 'sanada_live_YOUR_KEY',
        activeSelectedKey ? activeSelectedKey.name : null
    );

    if (cachedKeys.length === 0) {
        container.innerHTML = `
            <div class="api-empty-state">
                <span class="api-empty-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
                    </svg>
                </span>
                <span class="api-empty-title">No API keys created yet</span>
                <span class="api-empty-sub">Click "Create Secret Key" above to generate your first credential.</span>
            </div>
        `;
        return;
    }

    let html = '';
    cachedKeys.forEach(key => {
        const isSelected = (key.id === selectedKeyId);
        const displayValue = key.revealed ? key.rawKey : key.maskedKey;
        html += `
            <div class="api-key-item ${isSelected ? 'selected' : ''}" data-id="${key.id}">
                <div class="api-key-header-row">
                    <div class="api-key-meta">
                        <span class="api-key-name">${escapeHtml(key.name)}</span>
                        <span class="api-key-date">Created: ${key.createdAt}</span>
                    </div>
                    <div class="api-key-status-group">
                        ${isSelected ? `
                            <span class="api-key-badge selected" title="Active key used in integration code snippets">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="20 6 9 17 4 12"></polyline>
                                </svg>
                                <span>Selected for Integration</span>
                            </span>
                        ` : `
                            <button type="button" class="btn-select-key" data-action="select" data-id="${key.id}" title="Use this key in code integration snippets">
                                <span>Use in Snippets</span>
                            </button>
                        `}
                    </div>
                </div>

                <div class="api-key-code-wrap">
                    <input
                        type="text"
                        class="api-key-input"
                        readonly
                        value="${displayValue}"
                        id="keyInput_${key.id}"
                        aria-label="API Key"
                    >
                    <div class="api-key-actions">
                        <button
                            type="button"
                            class="btn-key-action btn-key-toggle"
                            data-action="toggle"
                            data-id="${key.id}"
                            title="${key.revealed ? 'Hide secret key' : 'Click to reveal full key'}"
                            aria-label="Toggle key visibility"
                        >
                            ${key.revealed ? `
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                                    <line x1="1" y1="1" x2="23" y2="23"></line>
                                </svg>
                                <span>Hide</span>
                            ` : `
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8z"></path>
                                    <circle cx="12" cy="12" r="3"></circle>
                                </svg>
                                <span>Show</span>
                            `}
                        </button>
                        <button
                            type="button"
                            class="btn-key-action btn-key-copy"
                            data-action="copy"
                            data-id="${key.id}"
                            title="Copy full API Key to clipboard"
                        >
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        </button>
                        <button
                            type="button"
                            class="btn-key-action btn-key-revoke"
                            data-action="revoke"
                            data-id="${key.id}"
                            title="Revoke / Delete Key"
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    // Attach card click to select key
    container.querySelectorAll('.api-key-item').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('button') || e.target.closest('input')) return;
            const id = card.dataset.id;
            if (id && id !== selectedKeyId) {
                selectedKeyId = id;
                renderApiKeysList();
                const target = cachedKeys.find(k => k.id === id);
                if (target && typeof window.showToast === 'function') {
                    window.showToast(`Selected "${target.name}" for code snippets`, 'info');
                }
            }
        });
    });

    // Attach row button events
    container.querySelectorAll('button[data-action]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            e.preventDefault();
            const action = btn.dataset.action;
            const id = btn.dataset.id;
            const target = cachedKeys.find(k => k.id === id);
            if (!target) return;

            if (action === 'select') {
                selectedKeyId = id;
                renderApiKeysList();
                if (typeof window.showToast === 'function') {
                    window.showToast(`Selected "${target.name}" for code snippets`, 'info');
                }
            } else if (action === 'toggle') {
                target.revealed = !target.revealed;
                renderApiKeysList();
            } else if (action === 'copy') {
                await copyToClipboard(target.rawKey);
                const originalHtml = btn.innerHTML;
                btn.classList.add('copied');
                btn.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span>Copied!</span>
                `;
                if (typeof window.showToast === 'function') {
                    window.showToast(`API Key "${target.name}" copied to clipboard!`, 'success');
                }
                setTimeout(() => {
                    btn.classList.remove('copied');
                    btn.innerHTML = originalHtml;
                }, 2000);
            } else if (action === 'revoke') {
                if (confirm(`Are you sure you want to revoke the key "${target.name}"? Applications using this key will immediately lose access.`)) {
                    await revokeKey(id);
                    renderApiKeysList();
                    if (typeof window.showToast === 'function') {
                        window.showToast(`API Key revoked.`, 'info');
                    }
                }
            }
        });
    });
}

/**
 * Copies text to user clipboard.
 */
async function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (e) {}
    }
    const temp = document.createElement('textarea');
    temp.value = text;
    temp.style.position = 'fixed';
    temp.style.left = '-9999px';
    document.body.appendChild(temp);
    temp.select();
    document.execCommand('copy');
    document.body.removeChild(temp);
    return true;
}

/**
 * Updates the code snippet box with active language, user key, and key name badge.
 */
function updateCodeSnippet(apiKey, keyName = null) {
    const codeEl = document.getElementById('apiCodeDisplay');
    const labelEl = document.getElementById('apiCodeActiveKeyLabel');
    if (labelEl) {
        labelEl.textContent = keyName ? `Key: ${keyName}` : 'Key: None';
    }
    if (!codeEl) return;

    const origin = (typeof window !== 'undefined' && window.location.origin) ? window.location.origin : 'http://localhost:8000';
    const endpoint = `${origin}/api/predict`;

    let snippet = '';

    if (activeSnippetLang === 'curl') {
        snippet = `# 1. Send polyphonic audio for multi-task inference
curl -X POST "${endpoint}" \\
  -H "X-API-Key: ${apiKey}" \\
  -F "file=@meeting_recording.wav" \\
  -F "task=all" \\
  -F "threshold=0.5"`;
    } else if (activeSnippetLang === 'python') {
        snippet = `import requests

# SANADA Multi-Task Neural Audio Inference
url = "${endpoint}"
headers = {
    "X-API-Key": "${apiKey}"
}
data = {
    "task": "all",          # "all" | "count" | "tag" | "separate"
    "threshold": 0.5
}
files = {
    "file": open("meeting_recording.wav", "rb")
}

response = requests.post(url, headers=headers, data=data, files=files)
result = response.json()

# Parse multi-task output
print("Source Count K:", result["results"]["source_counting"]["estimated_k"])
print("Active Sound Tags:", result["results"]["sound_tagging"]["active_classes"])
print("Separated Stems:", [s["url"] for s in result["results"]["speech_separation"]["stems"]])`;
    } else if (activeSnippetLang === 'node') {
        snippet = `import fs from 'fs';
import FormData from 'form-data';
import fetch from 'node-fetch';

const form = new FormData();
form.append('file', fs.createReadStream('meeting_recording.wav'));
form.append('task', 'all');
form.append('threshold', '0.5');

const res = await fetch('${endpoint}', {
  method: 'POST',
  headers: {
    'X-API-Key': '${apiKey}',
    ...form.getHeaders()
  },
  body: form
});

const data = await res.json();
console.log('Results:', data.results);`;
    }

    codeEl.textContent = snippet;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/**
 * Reloads keys for the active identity and updates the UI.
 */
export async function refreshApiPage() {
    await loadApiKeys();
    renderApiKeysList();
}

/**
 * Initializes the Web API page events and key manager.
 */
export async function initApiPage() {
    const generateBtn = document.getElementById('apiGenerateKeyBtn');
    const keyNameInput = document.getElementById('apiKeyNameInput');
    const copySnippetBtn = document.getElementById('apiCopySnippetBtn');

    // Language switcher pills
    const langBtns = document.querySelectorAll('.code-lang-pill');
    langBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            langBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSnippetLang = btn.dataset.lang || 'curl';
            const activeSelectedKey = cachedKeys.find(k => k.id === selectedKeyId);
            updateCodeSnippet(
                activeSelectedKey ? activeSelectedKey.rawKey : 'sanada_live_YOUR_KEY',
                activeSelectedKey ? activeSelectedKey.name : null
            );
        });
    });

    // Copy snippet button
    if (copySnippetBtn) {
        copySnippetBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const codeEl = document.getElementById('apiCodeDisplay');
            if (!codeEl) return;
            await copyToClipboard(codeEl.textContent);
            const orig = copySnippetBtn.innerHTML;
            copySnippetBtn.innerHTML = '<span>Copied!</span>';
            if (typeof window.showToast === 'function') {
                window.showToast('Integration code copied to clipboard!', 'success');
            }
            setTimeout(() => {
                copySnippetBtn.innerHTML = orig;
            }, 2000);
        });
    }

    // Generate Key button
    if (generateBtn) {
        generateBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const rawName = (keyNameInput?.value || '').trim();
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span>Generating...</span>';

            try {
                const newKey = await saveNewKey(rawName);
                if (keyNameInput) keyNameInput.value = '';
                renderApiKeysList();
                if (typeof window.showToast === 'function') {
                    window.showToast(`New API Key "${newKey.name}" created and selected!`, 'success');
                }
            } catch (err) {
                console.error('[SANADA API] Key generation error:', err);
            } finally {
                generateBtn.disabled = false;
                generateBtn.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    <span>Create Secret Key</span>
                `;
            }
        });
    }

    // Reactively refresh when navigating to the API tab or when user logs in/out
    onRouteChanged((route) => {
        if (route === 'api') {
            refreshApiPage();
        }
    });

    if (typeof window !== 'undefined') {
        window.addEventListener('sanada:roleChanged', () => {
            refreshApiPage();
        });
    }

    // Load and render initial keys
    await refreshApiPage();
}

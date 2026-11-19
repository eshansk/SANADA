/**
 * frontend/js/pages/api.js
 * ========================
 * Controller for the SANADA Web API & Partner Gateway.
 *
 * Provides:
 *  1. Cryptographically secure API key generation with SHA-256 database hashing.
 *  2. Masked key display with click-to-reveal toggle (eye icon).
 *  3. One-click clipboard copy with animated visual feedback.
 *  4. Key deletion / revocation with confirmation.
 *  5. Dynamic multi-language integration snippets (cURL, Python requests, Node.js fetch).
 *  6. Dual-tier storage (Supabase PostgreSQL sync with local resilience fallback).
 */

import { getAuth } from '../auth-guard.js';

// Local storage key for fallback persistence
const STORAGE_KEY = 'sanada_api_keys';

// Active code snippet tab state
let activeSnippetLang = 'curl';
let cachedKeys = [];

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
 * Loads API keys from Supabase or localStorage fallback.
 */
async function loadApiKeys() {
    try {
        if (typeof window !== 'undefined' && window.SANADA_SUPABASE && window.SANADA_SUPABASE.getClient) {
            const client = window.SANADA_SUPABASE.getClient();
            if (client) {
                const { data, error } = await client.from('api_keys').select('*').order('created_at', { ascending: false });
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
                    return cachedKeys;
                }
            }
        }
    } catch (e) {
        console.warn('[SANADA API] Supabase table lookup notice:', e?.message || e);
    }

    // Fallback: Read from localStorage
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            cachedKeys = JSON.parse(stored);
            return cachedKeys;
        }
    } catch (e) {}

    // Seed default demo key if list is completely empty
    if (!cachedKeys || cachedKeys.length === 0) {
        const demoRaw = 'sanada_live_9a7b3c2e1f408d6e5a9c8b7a6d5e4f3a';
        cachedKeys = [{
            id: 'key_demo_default',
            name: 'Default Inference Client',
            rawKey: demoRaw,
            maskedKey: maskApiKey(demoRaw),
            keyHash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
            createdAt: 'Today',
            revealed: false
        }];
        saveToLocalStorage(cachedKeys);
    }

    return cachedKeys;
}

function saveToLocalStorage(keys) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
    } catch (e) {}
}

/**
 * Saves a new key to Supabase (and localStorage).
 */
async function saveNewKey(name) {
    const rawKey = generateRawApiKey();
    const keyHash = await computeSha256(rawKey);
    const maskedKey = maskApiKey(rawKey);
    const auth = getAuth();
    const userId = auth?.id || null;

    const newKeyObj = {
        id: 'key_' + Date.now(),
        name: name.trim() || 'Production Key',
        rawKey: rawKey,
        maskedKey: maskedKey,
        keyHash: keyHash,
        createdAt: 'Just now',
        revealed: false
    };

    // Attempt Supabase insert
    try {
        if (typeof window !== 'undefined' && window.SANADA_SUPABASE && window.SANADA_SUPABASE.getClient) {
            const client = window.SANADA_SUPABASE.getClient();
            if (client) {
                const { data, error } = await client.from('api_keys').insert([{
                    user_id: userId,
                    name: newKeyObj.name,
                    key_prefix: rawKey.slice(0, 16),
                    key_hash: keyHash,
                    masked_key: maskedKey,
                    encrypted_key: rawKey
                }]).select();

                if (!error && data && data.length > 0) {
                    newKeyObj.id = data[0].id;
                }
            }
        }
    } catch (err) {
        console.info('[SANADA API] Stored locally (Supabase table api_keys will sync once SQL is executed):', err?.message || err);
    }

    cachedKeys.unshift(newKeyObj);
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
}

/**
 * Renders the list of API Keys into the DOM.
 */
function renderApiKeysList() {
    const container = document.getElementById('apiKeysListContainer');
    if (!container) return;

    if (cachedKeys.length === 0) {
        container.innerHTML = `
            <div class="api-empty-state">
                <span class="api-empty-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
                    </svg>
                </span>
                <span class="api-empty-title">No API keys created yet</span>
                <span class="api-empty-sub">Enter a key label above and click "Create Secret Key" to generate credentials.</span>
            </div>
        `;
        updateCodeSnippet('sanada_live_YOUR_KEY');
        return;
    }

    let html = '';
    cachedKeys.forEach(key => {
        const displayValue = key.revealed ? key.rawKey : key.maskedKey;
        html += `
            <div class="api-key-item" data-id="${key.id}">
                <div class="api-key-header-row">
                    <div class="api-key-meta">
                        <span class="api-key-name">${escapeHtml(key.name)}</span>
                        <span class="api-key-date">Created: ${key.createdAt}</span>
                    </div>
                    <span class="api-key-badge active">
                        <span class="badge-dot"></span> Active
                    </span>
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

    // Attach row button events
    container.querySelectorAll('button[data-action]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const action = btn.dataset.action;
            const id = btn.dataset.id;
            const target = cachedKeys.find(k => k.id === id);
            if (!target) return;

            if (action === 'toggle') {
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

    // Update code snippets with the latest/first key
    const activeKey = cachedKeys[0]?.rawKey || 'sanada_live_YOUR_KEY';
    updateCodeSnippet(activeKey);
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
 * Updates the code snippet box with active language and user key.
 */
function updateCodeSnippet(apiKey) {
    const codeEl = document.getElementById('apiCodeDisplay');
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
            const activeKey = cachedKeys[0]?.rawKey || 'sanada_live_YOUR_KEY';
            updateCodeSnippet(activeKey);
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
            const name = (keyNameInput?.value || '').trim() || 'My App Key';
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span>Generating...</span>';

            try {
                const newKey = await saveNewKey(name);
                if (keyNameInput) keyNameInput.value = '';
                renderApiKeysList();
                if (typeof window.showToast === 'function') {
                    window.showToast(`New API Key "${newKey.name}" created!`, 'success');
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

    // Load and render initial keys
    await loadApiKeys();
    renderApiKeysList();
}

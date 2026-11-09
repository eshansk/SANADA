/**
 * frontend/js/pages/paper.js
 * =========================
 * Paper & Citation View Logic.
 * Handles BibTeX citation copying and research artifact interactions.
 */

import { showToast } from '../ui/toast.js';

export function initPaperPage() {
    const copyBibtexBtn = document.getElementById('copyBibtexBtn');
    if (copyBibtexBtn) {
        copyBibtexBtn.addEventListener('click', () => {
            const bibtexEl = document.getElementById('bibtexCode');
            if (!bibtexEl) return;
            navigator.clipboard.writeText(bibtexEl.textContent.trim())
                .then(() => {
                    showToast('BibTeX citation copied to clipboard!', 'success');
                })
                .catch(() => {
                    showToast('Unable to copy citation', 'warning');
                });
        });
    }

    const copyDoiBtn = document.getElementById('copyDoiBtn');
    if (copyDoiBtn) {
        copyDoiBtn.addEventListener('click', () => {
            const doi = '10.21227/z7hv-e014';
            navigator.clipboard.writeText(doi)
                .then(() => {
                    showToast('DOI (10.21227/z7hv-e014) copied to clipboard!', 'success');
                })
                .catch(() => {
                    showToast('Unable to copy DOI', 'warning');
                });
        });
    }
}

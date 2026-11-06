/**
 * frontend/js/ui/toast.js
 * =======================
 * Non-blocking visual status alerts and user action confirmations.
 * Native DOM implementation with timed auto-dismissal.
 */

export function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✔' : type === 'warning' ? '⚠' : '✖';
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 250);
    }, 4000);
}

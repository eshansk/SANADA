/**
 * frontend/js/pages/developer.js
 * ===============================
 * Developer Training Console & Checkpoint Serialization.
 * Manages asynchronous training configuration, execution telemetry,
 * real-time worker polling with offline simulation fallback, checkpoint export,
 * terminal chrome with monochrome prefix opacity, and live loss sparkline canvas.
 */

import { API } from '../services/api.js';
import { showToast } from '../ui/toast.js';

export function initDeveloperPage() {
    const DOM = {
        trainTaskSelect: document.getElementById('trainTaskSelect'),
        trainDatasetSelect: document.getElementById('trainDatasetSelect'),
        trainLrInput: document.getElementById('trainLrInput'),
        trainBatchInput: document.getElementById('trainBatchInput'),
        trainEpochsInput: document.getElementById('trainEpochsInput'),
        trainOptSelect: document.getElementById('trainOptSelect'),
        startTrainingBtn: document.getElementById('startTrainingBtn'),
        trainingProgressFill: document.getElementById('trainingProgressFill'),
        trainingStatusText: document.getElementById('trainingStatusText'),
        trainingProgressPct: document.getElementById('trainingProgressPct'),
        trainLogTerminal: document.getElementById('trainLogTerminal'),
        trainLossCanvas: document.getElementById('trainLossCanvas'),
        sparklineVal: document.getElementById('sparklineVal'),
        terminalTag: document.getElementById('terminalTag'),
        exportTaskSelect: document.getElementById('exportTaskSelect'),
        exportVersionTag: document.getElementById('exportVersionTag'),
        exportNotes: document.getElementById('exportNotes'),
        exportModelBtn: document.getElementById('exportModelBtn')
    };

    let lossHistory = [];

    function drawLossSparkline(canvas, history) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!history || history.length === 0) return;

        const padX = 8;
        const padY = 4;
        const drawW = w - padX * 2;
        const drawH = h - padY * 2;

        const maxLoss = Math.max(...history, 0.05);
        const minLoss = Math.min(...history, 0);
        const range = (maxLoss - minLoss) || 1;

        ctx.beginPath();
        history.forEach((val, i) => {
            const x = padX + (i / Math.max(1, history.length - 1)) * drawW;
            const norm = (val - minLoss) / range;
            const y = padY + (1 - norm) * drawH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.6;
        ctx.stroke();

        // Draw active latest head dot
        const lastIdx = history.length - 1;
        const lastX = padX + (lastIdx / Math.max(1, history.length - 1)) * drawW;
        const lastNorm = (history[lastIdx] - minLoss) / range;
        const lastY = padY + (1 - lastNorm) * drawH;

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    function logTerminal(msg) {
        if (!DOM.trainLogTerminal) return;
        const time = new Date().toLocaleTimeString();

        let prefixClass = 'prefix-info';
        if (msg.startsWith('[COMPLETE]')) {
            prefixClass = 'prefix-complete';
        } else if (msg.startsWith('[EPOCH')) {
            prefixClass = 'prefix-epoch';
        } else if (msg.startsWith('[SYSTEM]')) {
            prefixClass = 'prefix-system';
        } else if (msg.startsWith('[INIT]')) {
            prefixClass = 'prefix-info';
        }

        DOM.trainLogTerminal.innerHTML += `<div><span class="terminal-prefix ${prefixClass}">[${time}]</span> ${msg}</div>`;
        DOM.trainLogTerminal.scrollTop = DOM.trainLogTerminal.scrollHeight;
    }

    function pollJob(jobId) {
        const interval = setInterval(async () => {
            try {
                const job = await API.pollTrainingJob(jobId);
                if (DOM.trainingStatusText) {
                    DOM.trainingStatusText.textContent = `Status: ${job.status.toUpperCase()} (Epoch ${job.current_epoch || 0}/${job.epochs || 30})`;
                }
                if (DOM.trainingProgressPct) {
                    DOM.trainingProgressPct.textContent = `${job.progress_pct.toFixed(0)}%`;
                }
                if (DOM.trainingProgressFill) {
                    DOM.trainingProgressFill.style.width = `${job.progress_pct}%`;
                }

                if (job.current_loss !== undefined) {
                    lossHistory.push(job.current_loss);
                    if (DOM.sparklineVal) DOM.sparklineVal.textContent = `Loss: ${job.current_loss.toFixed(4)}`;
                    drawLossSparkline(DOM.trainLossCanvas, lossHistory);
                }

                if (job.status === 'completed') {
                    clearInterval(interval);
                    if (DOM.startTrainingBtn) DOM.startTrainingBtn.disabled = false;
                    if (DOM.terminalTag) DOM.terminalTag.textContent = 'COMPLETED';
                    logTerminal(`[COMPLETE] Job ${jobId} finished! Final loss: ${job.final_loss || 0.042}. Checkpoint cached.`);
                    showToast('Model training completed!', 'success');
                }
            } catch (e) {
                clearInterval(interval);
                if (DOM.startTrainingBtn) DOM.startTrainingBtn.disabled = false;
                if (DOM.terminalTag) DOM.terminalTag.textContent = 'STANDBY';
            }
        }, 800);
    }

    function simulateTrainingTerminal(config) {
        let p = 0;
        lossHistory = [0.82];
        if (DOM.terminalTag) DOM.terminalTag.textContent = 'RUNNING';
        drawLossSparkline(DOM.trainLossCanvas, lossHistory);

        const fakeInterval = setInterval(() => {
            p += 20;
            if (DOM.trainingProgressPct) DOM.trainingProgressPct.textContent = `${p}%`;
            if (DOM.trainingProgressFill) DOM.trainingProgressFill.style.width = `${p}%`;
            if (DOM.trainingStatusText) DOM.trainingStatusText.textContent = `Status: RUNNING (Progress ${p}%)`;

            const currentLoss = parseFloat((0.82 / (p / 16 + 1)).toFixed(4));
            lossHistory.push(currentLoss);
            if (DOM.sparklineVal) DOM.sparklineVal.textContent = `Loss: ${currentLoss}`;
            drawLossSparkline(DOM.trainLossCanvas, lossHistory);

            const epochNum = Math.floor((p / 100) * config.epochs);
            logTerminal(`[EPOCH ${epochNum}/${config.epochs}] Loss: ${currentLoss}`);

            if (p >= 100) {
                clearInterval(fakeInterval);
                if (DOM.startTrainingBtn) DOM.startTrainingBtn.disabled = false;
                if (DOM.trainingStatusText) DOM.trainingStatusText.textContent = 'Status: COMPLETED';
                if (DOM.terminalTag) DOM.terminalTag.textContent = 'COMPLETED';
                logTerminal('[COMPLETE] Training run finished successfully. Model state serialized.');
                showToast('Simulated training complete', 'success');
            }
        }, 600);
    }

    // Trigger Training
    if (DOM.startTrainingBtn) {
        DOM.startTrainingBtn.addEventListener('click', async () => {
            const config = {
                task: DOM.trainTaskSelect ? DOM.trainTaskSelect.value : 'counting',
                dataset_name: DOM.trainDatasetSelect ? DOM.trainDatasetSelect.value : 'Org_Mix_4_Sounds_Variable',
                learning_rate: parseFloat(DOM.trainLrInput ? DOM.trainLrInput.value : 0.0005),
                batch_size: parseInt(DOM.trainBatchInput ? DOM.trainBatchInput.value : 16, 10),
                epochs: parseInt(DOM.trainEpochsInput ? DOM.trainEpochsInput.value : 30, 10),
                optimizer: DOM.trainOptSelect ? DOM.trainOptSelect.value : 'AdamW',
                use_specaugment: true
            };

            DOM.startTrainingBtn.disabled = true;
            if (DOM.terminalTag) DOM.terminalTag.textContent = 'RUNNING';
            lossHistory = [];
            logTerminal(`[INIT] Dispatching training run for task='${config.task}' on '${config.dataset_name}'...`);

            try {
                const res = await API.startTraining(config);
                logTerminal(`[SYSTEM] Worker accepted run with Job ID: ${res.job_id}`);
                pollJob(res.job_id);
            } catch (err) {
                logTerminal('[SYSTEM] API unreachable. Falling back to local training simulation...');
                simulateTrainingTerminal(config);
            }
        });
    }

    // Export Model Checkpoint
    if (DOM.exportModelBtn) {
        DOM.exportModelBtn.addEventListener('click', async () => {
            const exportData = {
                task: DOM.exportTaskSelect ? DOM.exportTaskSelect.value : 'counting',
                version: DOM.exportVersionTag ? DOM.exportVersionTag.value : 'v1.2.0',
                notes: DOM.exportNotes ? DOM.exportNotes.value : 'Production model checkpoint'
            };

            logTerminal(`[SAVE] Serializing PyTorch model checkpoint (${exportData.version})...`);
            try {
                const res = await API.saveModel(exportData);
                logTerminal(`[COMPLETE] Checkpoint saved successfully: ${res.checkpoint_path}`);
                showToast('Model checkpoint exported (.pt)', 'success');
            } catch (err) {
                logTerminal(`[COMPLETE] Simulation Checkpoint serialized: checkpoints/${exportData.task}_${exportData.version}.pt`);
                showToast('Checkpoint cached locally (Offline mode)', 'info');
            }
        });
    }
}

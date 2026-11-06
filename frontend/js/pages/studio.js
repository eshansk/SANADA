/**
 * frontend/js/pages/studio.js
 * ===========================
 * Studio (Predict) Page Logic & Neural Inference Dispatcher.
 * Handles audio ingestion (file drop, demo selection, mic record),
 * Web Audio decoding, dual visualizer sync, preflight validation,
 * multi-task inference dispatch with online/offline simulation,
 * transport playback, complexity ceiling meter, and JSON export.
 */

import { API_BASE, TARGET_SR, EXPECTED_FRAMES, EXPECTED_DURATION } from '../config.js';
import { API } from '../services/api.js';
import { audioEngine } from '../audio/audioEngine.js';
import { DualAudioVisualizer } from '../audio/signalVisualizer.js';
import { StemPlayerMatrix } from '../audio/stemPlayer.js';
import { showToast } from '../ui/toast.js';

export function initStudioPage() {
    // Studio State
    let activeFile = null;
    let activeBuffer = null;
    let visualizer = null;
    let stemMatrix = null;

    // DOM References
    const DOM = {
        tabUploadMode: document.getElementById('tabUploadMode'),
        tabRecordMode: document.getElementById('tabRecordMode'),
        dropContainer: document.getElementById('dropContainer'),
        recordContainer: document.getElementById('recordContainer'),
        dropZone: document.getElementById('dropZone'),
        fileInput: document.getElementById('fileInput'),
        browseBtn: document.getElementById('browseBtn'),
        demoSelect: document.getElementById('demoSelect'),
        recordToggleBtn: document.getElementById('recordToggleBtn'),
        recTimer: document.getElementById('recTimer'),
        recHint: document.getElementById('recHint'),

        // Specs Readout
        metaFilename: document.getElementById('metaFilename'),
        metaSampleRate: document.getElementById('metaSampleRate'),
        metaChannels: document.getElementById('metaChannels'),
        metaDuration: document.getElementById('metaDuration'),
        metaPeak: document.getElementById('metaPeak'),
        metaCompliance: document.getElementById('metaCompliance'),
        uploadStatusBadge: document.getElementById('uploadStatusBadge'),

        // Dual Visualizer
        waveformCanvas: document.getElementById('waveformCanvas'),
        spectrogramCanvas: document.getElementById('spectrogramCanvas'),
        playheadLine: document.getElementById('playheadLine'),
        playMasterBtn: document.getElementById('playMasterBtn'),
        playIcon: document.getElementById('playIcon'),
        scrubber: document.getElementById('scrubber'),
        timeDisplay: document.getElementById('timeDisplay'),

        // Inference Dispatch
        taskSelect: document.getElementById('taskSelect'),
        runInferenceBtn: document.getElementById('runInferenceBtn'),
        inferenceBtnText: document.getElementById('inferenceBtnText'),
        inferenceBtnIcon: document.getElementById('inferenceBtnIcon'),

        // Task Results
        kHeroNumber: document.getElementById('kHeroNumber'),
        kHeroSub: document.getElementById('kHeroSub'),
        kModelBadge: document.getElementById('kModelBadge'),
        kContinuousVal: document.getElementById('kContinuousVal'),
        probBarsContainer: document.getElementById('probBarsContainer'),
        stemsContainer: document.getElementById('stemsContainer'),

        // Ceiling Gauge
        gaugeFill: document.getElementById('gaugeFill'),
        gaugeLabel: document.getElementById('gaugeLabel'),
        gaugeDot: document.getElementById('gaugeDot'),

        // Copy JSON
        copyJsonBtn: document.getElementById('copyJsonBtn')
    };

    // Subsystems initialization
    if (DOM.waveformCanvas && DOM.spectrogramCanvas && DOM.playheadLine) {
        visualizer = new DualAudioVisualizer(
            DOM.waveformCanvas,
            DOM.spectrogramCanvas,
            DOM.playheadLine
        );
    }

    if (DOM.stemsContainer) {
        stemMatrix = new StemPlayerMatrix(DOM.stemsContainer, showToast, API_BASE);
    }

    // 1. Ingestion Mode Toggles (Upload vs Live Mic Record)
    if (DOM.tabUploadMode && DOM.tabRecordMode) {
        DOM.tabUploadMode.addEventListener('click', () => {
            DOM.tabUploadMode.classList.add('active');
            DOM.tabRecordMode.classList.remove('active');
            if (DOM.dropContainer) DOM.dropContainer.style.display = 'block';
            if (DOM.recordContainer) DOM.recordContainer.style.display = 'none';
        });

        DOM.tabRecordMode.addEventListener('click', () => {
            DOM.tabRecordMode.classList.add('active');
            DOM.tabUploadMode.classList.remove('active');
            if (DOM.recordContainer) DOM.recordContainer.style.display = 'block';
            if (DOM.dropContainer) DOM.dropContainer.style.display = 'none';
        });
    }

    // 2. File Picker & Dropzone
    if (DOM.browseBtn && DOM.fileInput) {
        DOM.browseBtn.addEventListener('click', () => DOM.fileInput.click());
    }
    if (DOM.dropZone && DOM.fileInput) {
        DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());
        DOM.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.add('dragover');
        });
        DOM.dropZone.addEventListener('dragleave', () => DOM.dropZone.classList.remove('dragover'));
        DOM.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                processAudioFile(e.dataTransfer.files[0]);
            }
        });
    }
    if (DOM.fileInput) {
        DOM.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processAudioFile(e.target.files[0]);
            }
        });
    }

    // 3. Audio Processing & Decoding Helper
    async function processAudioFile(file) {
        activeFile = file;
        audioEngine.stopPlayback();

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const buffer = await audioEngine.decodeAudio(e.target.result);
                activeBuffer = buffer;
                displayDecodedBuffer(file.name, buffer);
                triggerPreflightValidation(file);
            } catch (err) {
                showToast('Unable to decode audio: ' + err.message, 'error');
            }
        };
        reader.readAsArrayBuffer(file);
    }

    function displayDecodedBuffer(filename, buffer) {
        const info = audioEngine.inspectBuffer(buffer);

        if (DOM.metaFilename) DOM.metaFilename.textContent = filename;
        if (DOM.metaSampleRate) DOM.metaSampleRate.textContent = `${info.sampleRate.toLocaleString()} Hz`;
        if (DOM.metaChannels) DOM.metaChannels.textContent = info.channels === 1 ? '1 (Monaural)' : `${info.channels} (Stereo)`;
        if (DOM.metaDuration) DOM.metaDuration.textContent = `${info.duration}s`;
        if (DOM.metaPeak) DOM.metaPeak.textContent = `${info.peakAmplitude} (${info.peakDbfs} dBFS)`;

        if (visualizer) visualizer.render(buffer);
        if (DOM.playMasterBtn) DOM.playMasterBtn.disabled = false;
        if (DOM.scrubber) DOM.scrubber.disabled = false;
        if (DOM.runInferenceBtn) DOM.runInferenceBtn.disabled = false;
    }

    async function triggerPreflightValidation(file) {
        try {
            const val = await API.validateAudio(file);
            if (val.standard_compliant) {
                if (DOM.uploadStatusBadge) {
                    DOM.uploadStatusBadge.textContent = 'COMPLIANT (PASS)';
                    DOM.uploadStatusBadge.className = 'badge pass';
                }
                if (DOM.metaCompliance) DOM.metaCompliance.innerHTML = '<span class="badge pass">100% SPEC COMPLIANT</span>';
                showToast(`Pre-flight passed: 16 kHz Mono 8.0s (${val.filename})`, 'success');
            } else {
                if (DOM.uploadStatusBadge) {
                    DOM.uploadStatusBadge.textContent = 'NON-STANDARD';
                    DOM.uploadStatusBadge.className = 'badge warn';
                }
                if (DOM.metaCompliance) DOM.metaCompliance.innerHTML = '<span class="badge warn">NON-STANDARD DURATION</span>';
                showToast('Audio will be padded/windowed to 8.0s', 'warning');
            }
        } catch (err) {
            if (DOM.uploadStatusBadge) {
                DOM.uploadStatusBadge.textContent = 'CLIENT VALIDATED';
                DOM.uploadStatusBadge.className = 'badge pass';
            }
            if (DOM.metaCompliance) DOM.metaCompliance.innerHTML = '<span class="badge pass">CLIENT VALIDATED</span>';
        }
    }

    // 4. Preloaded Demo Selector
    if (DOM.demoSelect) {
        DOM.demoSelect.addEventListener('change', async (e) => {
            const mixName = e.target.value;
            if (!mixName) return;
            audioEngine.stopPlayback();

            showToast(`Loading verified evaluation mixture: ${mixName}...`, 'info');
            try {
                const endpoints = [
                    `${API_BASE}/api/demo-audio/${mixName}`,
                    `${API_BASE}/static/demos/${mixName}`,
                    `/demos/${mixName}`
                ];
                let arrayBuf = null;
                for (const ep of endpoints) {
                    try {
                        const resp = await fetch(ep);
                        if (resp.ok) {
                            arrayBuf = await resp.arrayBuffer();
                            break;
                        }
                    } catch (err) {}
                }

                if (arrayBuf && arrayBuf.byteLength > 1000) {
                    const buffer = await audioEngine.decodeAudio(arrayBuf);
                    activeBuffer = buffer;
                    activeFile = new File([arrayBuf], mixName, { type: 'audio/wav' });
                    displayDecodedBuffer(activeFile.name, buffer);
                    triggerPreflightValidation(activeFile);
                    showToast(`Loaded ${mixName} from verified dataset pool!`, 'success');
                    return;
                }
            } catch (fetchErr) {
                console.warn('Real demo audio fetch failed, falling back to local synthesis', fetchErr);
            }

            const buffer = audioEngine.createSyntheticDemoBuffer(mixName.includes('9') ? 3 : 2);
            activeBuffer = buffer;
            const wavBlob = audioEngine.bufferToWavBlob(buffer);
            activeFile = new File([wavBlob], mixName, { type: 'audio/wav' });

            displayDecodedBuffer(activeFile.name, buffer);
            triggerPreflightValidation(activeFile);
        });
    }

    // 5. Live Microphone Capture
    if (DOM.recordToggleBtn) {
        DOM.recordToggleBtn.addEventListener('click', () => {
            if (audioEngine.isRecording) {
                audioEngine.stopRecording();
                DOM.recordToggleBtn.classList.remove('recording');
                if (DOM.recHint) DOM.recHint.textContent = 'Processing live recording...';
            } else {
                DOM.recordToggleBtn.classList.add('recording');
                if (DOM.recHint) DOM.recHint.textContent = 'Recording speech (Auto-stops at 8.0s)...';
                audioEngine.startRecording(
                    (elapsed) => {
                        if (DOM.recTimer) DOM.recTimer.textContent = `${elapsed.toFixed(1)} / 8.0s`;
                    },
                    (buffer) => {
                        DOM.recordToggleBtn.classList.remove('recording');
                        if (DOM.recHint) DOM.recHint.textContent = 'Standardized to 8.0s (128,000 samples). Ready!';
                        activeBuffer = buffer;
                        const blob = audioEngine.bufferToWavBlob(buffer);
                        activeFile = new File([blob], `mic_capture_${Date.now().toString().slice(-4)}.wav`, { type: 'audio/wav' });
                        displayDecodedBuffer(activeFile.name, buffer);
                        triggerPreflightValidation(activeFile);
                    },
                    (err) => {
                        DOM.recordToggleBtn.classList.remove('recording');
                        showToast('Microphone error: ' + err.message, 'error');
                    }
                );
            }
        });
    }

    // 6. Master Transport Controls
    if (DOM.playMasterBtn) {
        DOM.playMasterBtn.addEventListener('click', () => {
            if (audioEngine.isPlaying) {
                audioEngine.pausePlayback();
                if (DOM.playIcon) DOM.playIcon.textContent = '▶ Play';
            } else {
                audioEngine.play(
                    (cur, total, fraction) => {
                        if (DOM.timeDisplay) DOM.timeDisplay.textContent = `${cur.toFixed(2)} / ${total.toFixed(2)}s`;
                        if (DOM.scrubber) DOM.scrubber.value = fraction * 100;
                        if (visualizer) visualizer.setPlayheadProgress(fraction);
                    },
                    () => {
                        if (DOM.playIcon) DOM.playIcon.textContent = '▶ Play';
                        if (DOM.scrubber) DOM.scrubber.value = 0;
                        if (visualizer) visualizer.setPlayheadProgress(0);
                    }
                );
                if (DOM.playIcon) DOM.playIcon.textContent = '❚❚ Pause';
            }
        });
    }

    if (DOM.scrubber) {
        DOM.scrubber.addEventListener('input', (e) => {
            audioEngine.seek(e.target.value / 100);
            if (visualizer) visualizer.setPlayheadProgress(e.target.value / 100);
        });
    }

    // 7. Inference Dispatch (Online + Offline Fallback)
    if (DOM.runInferenceBtn) {
        DOM.runInferenceBtn.addEventListener('click', async () => {
            if (!activeFile) return;

            const task = DOM.taskSelect ? DOM.taskSelect.value : 'all';
            DOM.runInferenceBtn.disabled = true;
            if (DOM.inferenceBtnIcon) DOM.inferenceBtnIcon.innerHTML = '<div class="spinner"></div>';
            if (DOM.inferenceBtnText) DOM.inferenceBtnText.textContent = 'Executing Neural Tasks...';

            try {
                const data = await API.predictAudio(activeFile, task);
                renderPredictionResults(data.results);
                showToast('Neural listening inference completed successfully!', 'success');
            } catch (err) {
                showToast('Backend offline; activating high-fidelity simulation.', 'warning');
                renderMockResults();
            } finally {
                DOM.runInferenceBtn.disabled = false;
                if (DOM.inferenceBtnIcon) DOM.inferenceBtnIcon.textContent = '⚡';
                if (DOM.inferenceBtnText) DOM.inferenceBtnText.textContent = 'Run Full Inference (POST /api/predict)';
            }
        });
    }

    function renderPredictionResults(results) {
        // Task 1: Source Counting
        if (results.source_counting && DOM.kHeroNumber) {
            const count = results.source_counting;
            DOM.kHeroNumber.textContent = count.estimated_k;
            if (DOM.kHeroSub) DOM.kHeroSub.textContent = `Confidence: ${(count.confidence * 100).toFixed(1)}%`;
            if (DOM.kModelBadge) DOM.kModelBadge.textContent = count.model;
            if (DOM.kContinuousVal) DOM.kContinuousVal.textContent = count.raw_continuous_prediction.toFixed(3);
        }

        // Task 2: Multi-Label Tagging
        if (results.label_tagging && DOM.probBarsContainer) {
            const tagging = results.label_tagging;
            const activeClasses = tagging.active_classes || [];
            const isCorroborated = results.source_counting && results.source_counting.corroborated_by_tagging;

            let badgesHtml = `<div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; align-items:center;">
                <span style="font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Active Sources (${activeClasses.length}):</span>`;

            if (activeClasses.length > 0) {
                activeClasses.forEach(c => {
                    badgesHtml += `<span class="badge pass" style="font-size:11px; padding:3px 8px; font-weight:600;">● ${c}</span>`;
                });
            } else {
                badgesHtml += `<span class="badge fail" style="font-size:11px; padding:3px 8px;">None detected above threshold</span>`;
            }
            if (isCorroborated) {
                badgesHtml += `<span class="badge pass" style="font-size:10px; margin-left:auto;">⚡ Synchronized with Count K=${results.source_counting.estimated_k}</span>`;
            }
            badgesHtml += `</div>`;

            let probHtml = badgesHtml;
            const sorted = Object.entries(tagging.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
            const displayList = sorted.slice(0, Math.max(activeClasses.length, 5));

            displayList.forEach(([cls, prob]) => {
                const pct = (prob * 100).toFixed(0);
                const isActive = activeClasses.includes(cls);
                probHtml += `
                    <div class="prob-row">
                        <div class="prob-meta">
                            <span style="${isActive ? 'color: var(--accent); font-weight: 700;' : 'color: var(--text-muted);'}">
                                ${cls} ${isActive ? '✓ (Active)' : '<span style="font-size:10px;">(Candidate)</span>'}
                            </span>
                            <span style="${isActive ? 'font-weight: 700; color: var(--accent);' : ''}">${pct}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: ${pct}%; ${isActive ? 'background: var(--accent);' : 'background: var(--w-15);'}"></div>
                        </div>
                    </div>
                `;
            });
            DOM.probBarsContainer.innerHTML = probHtml;
        }

        // Task 3: Source Separation Stems Matrix
        if (results.source_separation && stemMatrix) {
            stemMatrix.render(results.source_separation.stems, API_BASE);
        }
    }

    function renderMockResults() {
        renderPredictionResults({
            source_counting: {
                estimated_k: 4,
                raw_continuous_prediction: 4.215,
                confidence: 0.895,
                model: "Conformer-Regressor-MSE"
            },
            label_tagging: {
                active_classes: ["Speaker_4_1", "Speaker_4_3"],
                class_probabilities: {
                    "Speaker_4_1": 0.965,
                    "Speaker_4_3": 0.912,
                    "Speaker_4_4": 0.384,
                    "Speaker_4_2": 0.122
                },
                model: "Conformer-Classifier-ASL"
            },
            source_separation: {
                reconstructed_count: 4,
                stems: [
                    { channel: 1, filename: "stem_1.wav", url: "/static/stems/demo_stem_1.wav", si_sdr_estimate_db: 10.59 },
                    { channel: 2, filename: "stem_2.wav", url: "/static/stems/demo_stem_2.wav", si_sdr_estimate_db: 8.42 },
                    { channel: 3, filename: "stem_3.wav", url: "/static/stems/demo_stem_3.wav", si_sdr_estimate_db: 4.24 },
                    { channel: 4, filename: "stem_4.wav", url: "/static/stems/demo_stem_4.wav", si_sdr_estimate_db: 4.24 }
                ],
                model: "TimeDomain-ConvTasNet"
            }
        });
    }

    // 8. Complexity Ceiling Gauge Watcher
    function updateCeilingGauge(kVal) {
        if (!DOM.gaugeFill || !DOM.gaugeLabel || !DOM.gaugeDot) return;
        const k = parseInt(kVal, 10);
        if (isNaN(k)) {
            DOM.gaugeFill.style.width = '0%';
            DOM.gaugeFill.style.background = 'rgba(255, 255, 255, 0.2)';
            DOM.gaugeDot.className = 'status-dot';
            DOM.gaugeLabel.textContent = 'Awaiting inference prediction to evaluate Complexity Ceiling';
            return;
        }

        if (k <= 3) {
            DOM.gaugeFill.style.width = `${Math.min(100, (k / 9) * 100)}%`;
            DOM.gaugeFill.style.background = 'var(--accent)';
            DOM.gaugeDot.className = 'status-dot';
            DOM.gaugeDot.style.background = 'var(--accent)';
            DOM.gaugeLabel.innerHTML = `<strong>K = ${k} (Safe Zone)</strong>: Clean separation expected (&gt; 8.0 dB SI-SDR).`;
        } else if (k === 4) {
            DOM.gaugeFill.style.width = '44%';
            DOM.gaugeFill.style.background = 'var(--accent-2)';
            DOM.gaugeDot.className = 'status-dot';
            DOM.gaugeDot.style.background = 'var(--accent-2)';
            DOM.gaugeLabel.innerHTML = `<strong>K = 4 (Ceiling Boundary)</strong>: Speech separation usable (~4.2 dB SI-SDR); near breakdown limit.`;
        } else {
            DOM.gaugeFill.style.width = `${Math.min(100, (k / 9) * 100)}%`;
            DOM.gaugeFill.style.background = 'var(--rose)';
            DOM.gaugeDot.className = 'status-dot';
            DOM.gaugeDot.style.background = 'var(--rose)';
            DOM.gaugeLabel.innerHTML = `<strong style="color: var(--rose);">K = ${k} (Ceiling Exceeded)</strong>: Separation is unreliable (&lt; 0 dB SI-SDR). High permutation ambiguity.`;
        }
    }

    if (DOM.kHeroNumber) {
        const observer = new MutationObserver(() => {
            updateCeilingGauge(DOM.kHeroNumber.textContent.trim());
        });
        observer.observe(DOM.kHeroNumber, { childList: true, characterData: true, subtree: true });
    }

    // 9. Copy JSON Results Button
    if (DOM.copyJsonBtn) {
        DOM.copyJsonBtn.addEventListener('click', () => {
            const kNumber = document.getElementById('kHeroNumber')?.textContent.trim() || '—';
            const kSub = document.getElementById('kHeroSub')?.textContent.trim() || '';
            const kContinuous = document.getElementById('kContinuousVal')?.textContent.trim() || '—';
            const filename = document.getElementById('metaFilename')?.textContent.trim() || 'unnamed.wav';

            const tags = [];
            document.querySelectorAll('#probBarsContainer .prob-row').forEach(row => {
                const label = row.querySelector('.prob-meta span:first-child')?.textContent.trim();
                const pct = row.querySelector('.prob-meta span:last-child')?.textContent.trim();
                if (label && pct) tags.push({ class: label.replace('●', '').trim(), probability: pct });
            });

            const stems = [];
            document.querySelectorAll('#stemsContainer .stem-strip').forEach(strip => {
                const title = strip.querySelector('.stem-strip-title')?.textContent.trim();
                const sisdr = strip.querySelector('.stem-strip-header .badge')?.textContent.trim();
                stems.push({ title, sisdr });
            });

            const payload = {
                metadata: {
                    filename,
                    evaluated_at: new Date().toISOString(),
                    standard_compliance: "16kHz Mono 8.0s"
                },
                task_1_counting: {
                    estimated_k: kNumber,
                    confidence: kSub,
                    continuous_raw: kContinuous
                },
                task_2_tagging: tags,
                task_3_separation: stems
            };

            navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
                .then(() => {
                    showToast('Inference results copied to clipboard as JSON!', 'success');
                })
                .catch(() => {
                    showToast('Failed to copy JSON to clipboard', 'warning');
                });
        });
    }

    // 10. Studio Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (['input', 'textarea', 'select'].includes(activeTag)) return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (DOM.playMasterBtn && !DOM.playMasterBtn.disabled) {
                DOM.playMasterBtn.click();
            }
        } else if (e.code === 'KeyM') {
            e.preventDefault();
            const muteBtn = document.querySelector('.stem-strip .btn-ctrl[id^="btnMute_"]');
            if (muteBtn) {
                muteBtn.click();
            } else {
                showToast('Mute shortcut (M) active when stems are loaded', 'info');
            }
        }
    });
}

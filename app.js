/**
 * SANADA — Audio Source Count Estimator, Tagging & Separator
 * Frontend Engine & Web Audio Interface
 * Strictly Monochrome & Professional
 */

(function () {
    'use strict';

    // ==========================================
    // CONSTANTS & AVAILABLE AI MODELS
    // ==========================================
    const MODELS = {
        'sanada-dualtask': {
            id: 'sanada-dualtask',
            name: 'SANADA-DualTask v3.2',
            spec: 'Tagging + Count Estimator + Demix',
            desc: 'Joint neural estimator that simultaneously tags acoustic categories, counts overlapping sources, and outputs high-fidelity demixed audio waveforms.',
            maxSources: 8,
            sampleRate: '16kHz / 44.1kHz',
            architecture: 'Dilated Conv + Cross-Attention'
        },
        'sanada-countnet': {
            id: 'sanada-countnet',
            name: 'SANADA-CountNet Ultra',
            spec: 'Dedicated Source Count Estimator',
            desc: 'Specialized convolutional recurrent model engineered strictly for audio source enumeration (1 to 10 sources).',
            maxSources: 10,
            sampleRate: '16kHz native',
            architecture: 'CRNN + Direction-of-Arrival'
        },
        'sanada-sepformer': {
            id: 'sanada-sepformer',
            name: 'SANADA-SepFormer HD',
            spec: 'Deep Polyphonic Sound Demixer',
            desc: 'Dual-path Transformer for deep masking of complex acoustic overlaps and dense polyphony.',
            maxSources: 6,
            sampleRate: '44.1kHz Hi-Res',
            architecture: 'Dual-Path Transformer'
        },
        'sanada-tasnet': {
            id: 'sanada-tasnet',
            name: 'SANADA-ConvTasNet Spatial',
            spec: 'Time-Domain Separation Network',
            desc: 'Time-domain convolutional separation network optimized for low latency multichannel sound isolation.',
            maxSources: 6,
            sampleRate: '16kHz Multi-channel',
            architecture: 'TCN (Temporal ConvNet)'
        },
        'sanada-edge': {
            id: 'sanada-edge',
            name: 'SANADA-Edge Fast 16k',
            spec: 'Low-Power Mobile / Real-time',
            desc: 'Quantized edge model for real-time processing and low-power audio source detection.',
            maxSources: 4,
            sampleRate: '16kHz Quantized',
            architecture: 'MobileNetV3-Audio'
        }
    };

    // Built-in presets based on FSD dataset mixtures
    const SAMPLE_PRESETS = {
        'mix3': {
            title: 'Acoustic Trio Mix (Cello + Phone + Scissors)',
            filename: 'mix_00000_3sources.wav',
            url: 'samples/mixture_3_sources.wav',
            estimatedCount: 3,
            confidence: '97.8%',
            polyphony: 'High Polyphony (Overlapping)',
            snr: '+14.6 dB',
            tags: [
                {
                    label: 'Telephone (Bell / Ring)',
                    confidence: '98.4%',
                    pct: 98,
                    category: 'Harmonic Formant',
                    freqRange: '1.2 kHz – 4.5 kHz',
                    timeSpan: '0.0s – 8.0s'
                },
                {
                    label: 'Scissors (Impulsive Transient)',
                    confidence: '95.1%',
                    pct: 95,
                    category: 'Percussive / Friction',
                    freqRange: '2.5 kHz – 8.0 kHz',
                    timeSpan: '1.2s – 6.8s'
                },
                {
                    label: 'Cello (Sustained Bowed String)',
                    confidence: '93.7%',
                    pct: 94,
                    category: 'Harmonic Series',
                    freqRange: '65 Hz – 2.0 kHz',
                    timeSpan: '0.0s – 8.0s'
                }
            ],
            distribution: [
                { count: '1 Source', pct: 1.2 },
                { count: '2 Sources', pct: 4.5 },
                { count: '3 Sources', pct: 92.4, isPredicted: true },
                { count: '4 Sources', pct: 1.6 },
                { count: '5+ Sources', pct: 0.3 }
            ],
            stems: [
                {
                    id: 1,
                    name: 'Source 1 — Scissors',
                    char: 'High-frequency impulsive transient • FSD Master test clip',
                    url: 'samples/stem_3_scissors.wav',
                    duration: '8.00s'
                },
                {
                    id: 2,
                    name: 'Source 2 — Telephone',
                    char: 'Pulsed bell resonance & harmonic formant • 1.2 kHz - 4.5 kHz',
                    url: 'samples/stem_3_telephone.wav',
                    duration: '8.00s'
                },
                {
                    id: 3,
                    name: 'Source 3 — Cello',
                    char: 'Low-frequency sustained bowed string • 65 Hz fundamental',
                    url: 'samples/stem_3_cello.wav',
                    duration: '8.00s'
                }
            ]
        },
        'mix2': {
            title: 'Violin & Applause Dual Sound',
            filename: 'mix_00000_2sources.wav',
            url: 'samples/mixture_2_sources.wav',
            estimatedCount: 2,
            confidence: '99.1%',
            polyphony: 'Moderate Overlap',
            snr: '+16.8 dB',
            tags: [
                {
                    label: 'Violin / Fiddle',
                    confidence: '99.0%',
                    pct: 99,
                    category: 'Harmonic String',
                    freqRange: '440 Hz – 3.2 kHz',
                    timeSpan: '0.0s – 8.0s'
                },
                {
                    label: 'Applause / Crowd Clapping',
                    confidence: '96.5%',
                    pct: 97,
                    category: 'Stochastic Wideband',
                    freqRange: '500 Hz – 7.5 kHz',
                    timeSpan: '0.5s – 8.0s'
                }
            ],
            distribution: [
                { count: '1 Source', pct: 0.5 },
                { count: '2 Sources', pct: 97.4, isPredicted: true },
                { count: '3 Sources', pct: 1.8 },
                { count: '4 Sources', pct: 0.2 },
                { count: '5+ Sources', pct: 0.1 }
            ],
            stems: [
                {
                    id: 1,
                    name: 'Source 1 — Violin / Fiddle',
                    char: 'Melodic string harmonic series • 440 Hz - 3.2 kHz',
                    url: 'samples/stem_2_violin.wav',
                    duration: '8.00s'
                },
                {
                    id: 2,
                    name: 'Source 2 — Applause',
                    char: 'Stochastic wideband acoustic claps • Ambient texture',
                    url: 'samples/stem_2_applause.wav',
                    duration: '8.00s'
                }
            ]
        }
    };

    // Initial seed sessions for Chat History
    const INITIAL_SESSIONS = [
        {
            id: 'sess-1',
            title: 'Acoustic Trio Mix (Cello + Phone + Scissors)',
            timestamp: 'Today, 15:30',
            dateCategory: 'Today',
            modelId: 'sanada-dualtask',
            filename: 'mix_00000_3sources.wav',
            fileSize: '250 KB',
            duration: '00:08',
            estimatedCount: 3,
            confidence: '97.8%',
            presetKey: 'mix3'
        },
        {
            id: 'sess-2',
            title: 'Violin & Applause Dual Sound',
            timestamp: 'Today, 14:15',
            dateCategory: 'Today',
            modelId: 'sanada-sepformer',
            filename: 'mix_00000_2sources.wav',
            fileSize: '250 KB',
            duration: '00:08',
            estimatedCount: 2,
            confidence: '99.1%',
            presetKey: 'mix2'
        },
        {
            id: 'sess-3',
            title: 'Conference Room 4-Speaker Demix',
            timestamp: 'Yesterday, 18:20',
            dateCategory: 'Yesterday',
            modelId: 'sanada-countnet',
            filename: 'multichannel_room_04.wav',
            fileSize: '1.2 MB',
            duration: '00:15',
            estimatedCount: 4,
            confidence: '96.2%',
            presetKey: null
        },
        {
            id: 'sess-4',
            title: 'Urban Soundscape 5-Source Separation',
            timestamp: 'Oct 24, 11:45',
            dateCategory: 'Previous 7 Days',
            modelId: 'sanada-tasnet',
            filename: 'city_ambient_mix.flac',
            fileSize: '4.8 MB',
            duration: '00:24',
            estimatedCount: 5,
            confidence: '94.5%',
            presetKey: null
        }
    ];

    // ==========================================
    // APPLICATION STATE
    // ==========================================
    const state = {
        currentModelId: 'sanada-dualtask',
        currentSession: null,
        sessions: [],
        searchQuery: '',
        activeView: 'waveform', // 'waveform', 'spectrogram', or 'dual'
        zoomFactor: 1,
        // The 3 options for the user:
        tasks: {
            tagging: true,
            counting: true,
            separating: true
        },
        spectrogram: {
            fftSize: 512,
            windowFunc: 'hann',
            computed: false,
            matrix: null, // 2D array [timeFrames][freqBins]
            timeFrames: 0,
            freqBins: 0
        },
        audio: {
            file: null,
            url: null,
            filename: '',
            sizeText: '',
            duration: 0,
            sampleRate: 16000,
            channels: 'Mono (1 Ch)',
            audioBuffer: null,
            isPlaying: false,
            isMuted: false,
            volume: 0.85,
            playbackRate: 1.0,
            isLooping: false,
            peakDbfs: -2.4,
            rmsDbfs: -18.2
        },
        inference: {
            isRunning: false,
            estimatedCount: 0,
            confidence: '0%',
            stems: []
        },
        stemPlayers: []
    };

    // ==========================================
    // DOM ELEMENTS
    // ==========================================
    const DOM = {
        // Layout & Sidebar
        sidebar: document.getElementById('chatSidebar'),
        toggleSidebarBtn: document.getElementById('toggleSidebarBtn'),
        headerNewChatBtn: document.getElementById('headerNewChatBtn'),
        sidebarNewBtn: document.getElementById('sidebarNewBtn'),
        themeToggleBtn: document.getElementById('themeToggleBtn'),
        findChatInput: document.getElementById('findChatInput'),
        clearSearchBtn: document.getElementById('clearSearchBtn'),
        searchStatusBar: document.getElementById('searchStatusBar'),
        searchResultCount: document.getElementById('searchResultCount'),
        chatSessionsList: document.getElementById('chatSessionsList'),

        // Session Meta
        currentSessionTitle: document.getElementById('currentSessionTitle'),
        currentSessionDate: document.getElementById('currentSessionDate'),
        activeModelTag: document.getElementById('activeModelTag'),
        audioStateBadge: document.getElementById('audioStateBadge'),

        // Model Selector
        modelSelectTrigger: document.getElementById('modelSelectTrigger'),
        currentModelName: document.getElementById('currentModelName'),
        currentModelSpec: document.getElementById('currentModelSpec'),
        modelModalBackdrop: document.getElementById('modelModalBackdrop'),
        closeModelModalBtn: document.getElementById('closeModelModalBtn'),
        cancelModelSelectBtn: document.getElementById('cancelModelSelectBtn'),
        confirmModelSelectBtn: document.getElementById('confirmModelSelectBtn'),
        modelOptionsList: document.getElementById('modelOptionsList'),

        // Upload
        uploadSection: document.getElementById('uploadSection'),
        dropZone: document.getElementById('dropZone'),
        audioFileInput: document.getElementById('audioFileInput'),
        loadSample3Btn: document.getElementById('loadSample3Btn'),
        loadSample2Btn: document.getElementById('loadSample2Btn'),

        // Player Section
        playerSection: document.getElementById('playerSection'),
        replaceAudioBtn: document.getElementById('replaceAudioBtn'),
        removeAudioBtn: document.getElementById('removeAudioBtn'),
        metaFileName: document.getElementById('metaFileName'),
        metaFileSize: document.getElementById('metaFileSize'),
        metaDuration: document.getElementById('metaDuration'),
        metaSampleRate: document.getElementById('metaSampleRate'),
        metaChannels: document.getElementById('metaChannels'),

        // View Tabs
        tabWaveform: document.getElementById('tabWaveform'),
        tabSpectrogram: document.getElementById('tabSpectrogram'),
        tabDual: document.getElementById('tabDual'),
        waveformPane: document.getElementById('waveformPane'),
        spectrogramPane: document.getElementById('spectrogramPane'),
        waveformControls: document.getElementById('waveformControls'),
        spectrogramControls: document.getElementById('spectrogramControls'),

        // Waveform Converter & Viewer
        btnConvertWaveform: document.getElementById('btnConvertWaveform'),
        waveformContainer: document.getElementById('waveformContainer'),
        waveformCanvas: document.getElementById('waveformCanvas'),
        waveformPlayhead: document.getElementById('waveformPlayhead'),
        waveformHoverTime: document.getElementById('waveformHoverTime'),
        waveformStatus: document.getElementById('waveformStatus'),
        peakReadout: document.getElementById('peakReadout'),

        // Spectrogram Converter & Viewer
        specFftSizeSelect: document.getElementById('specFftSizeSelect'),
        specWindowSelect: document.getElementById('specWindowSelect'),
        btnConvertSpectrogram: document.getElementById('btnConvertSpectrogram'),
        btnExportSpectrogram: document.getElementById('btnExportSpectrogram'),
        spectrogramStatus: document.getElementById('spectrogramStatus'),
        spectrogramContainer: document.getElementById('spectrogramContainer'),
        spectrogramCanvas: document.getElementById('spectrogramCanvas'),
        spectrogramPlayhead: document.getElementById('spectrogramPlayhead'),
        spectrogramCrosshair: document.getElementById('spectrogramCrosshair'),
        spectrogramTooltip: document.getElementById('spectrogramTooltip'),

        // Controls
        audioScrubber: document.getElementById('audioScrubber'),
        playPauseBtn: document.getElementById('playPauseBtn'),
        playIcon: document.getElementById('playIcon'),
        pauseIcon: document.getElementById('pauseIcon'),
        stopBtn: document.getElementById('stopBtn'),
        currentTimeDisplay: document.getElementById('currentTimeDisplay'),
        totalTimeDisplay: document.getElementById('totalTimeDisplay'),
        muteBtn: document.getElementById('muteBtn'),
        volumeSlider: document.getElementById('volumeSlider'),
        playbackSpeedSelect: document.getElementById('playbackSpeedSelect'),
        loopBtn: document.getElementById('loopBtn'),

        // 3 Options Elements: Tagging, Counting, Separating
        btnSelectAllTasks: document.getElementById('btnSelectAllTasks'),
        taskCardTagging: document.getElementById('taskCardTagging'),
        taskCardCounting: document.getElementById('taskCardCounting'),
        taskCardSeparating: document.getElementById('taskCardSeparating'),
        checkTaskTagging: document.getElementById('checkTaskTagging'),
        checkTaskCounting: document.getElementById('checkTaskCounting'),
        checkTaskSeparating: document.getElementById('checkTaskSeparating'),
        selectedTasksSummary: document.getElementById('selectedTasksSummary'),
        actionModelName: document.getElementById('actionModelName'),
        runInferenceBtn: document.getElementById('runInferenceBtn'),
        runBtnLabel: document.getElementById('runBtnLabel'),

        // Progress
        progressSection: document.getElementById('progressSection'),
        progressStepTitle: document.getElementById('progressStepTitle'),
        progressPercentText: document.getElementById('progressPercentText'),
        progressFill: document.getElementById('progressFill'),
        step1: document.getElementById('step1'),
        step2: document.getElementById('step2'),
        step3: document.getElementById('step3'),
        step4: document.getElementById('step4'),

        // Results Section & 3 Blocks
        resultsSection: document.getElementById('resultsSection'),
        saveSessionBtn: document.getElementById('saveSessionBtn'),
        resultBlockTagging: document.getElementById('resultBlockTagging'),
        taggingResultsGrid: document.getElementById('taggingResultsGrid'),
        taggingCountSummary: document.getElementById('taggingCountSummary'),

        resultBlockCounting: document.getElementById('resultBlockCounting'),
        estimatedCountNumber: document.getElementById('estimatedCountNumber'),
        estimatedConfidence: document.getElementById('estimatedConfidence'),
        resultModelName: document.getElementById('resultModelName'),
        compositionalityRating: document.getElementById('compositionalityRating'),
        snrEstimate: document.getElementById('snrEstimate'),
        distBarsContainer: document.getElementById('distBarsContainer'),

        resultBlockSeparating: document.getElementById('resultBlockSeparating'),
        stemsGrid: document.getElementById('stemsGrid'),
        playAllStemsBtn: document.getElementById('playAllStemsBtn'),
        stopAllStemsBtn: document.getElementById('stopAllStemsBtn'),

        // Master Audio Element
        masterAudio: document.getElementById('masterAudio')
    };

    // ==========================================
    // WEB AUDIO API CONTEXT & ANALYSER
    // ==========================================
    let audioCtx = null;
    let analyserNode = null;
    let audioSourceNode = null;
    let animFrameId = null;
    let waveformPeaks = [];

    function initAudioContext() {
        if (!audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContextClass();
            analyserNode = audioCtx.createAnalyser();
            analyserNode.fftSize = 256;

            try {
                audioSourceNode = audioCtx.createMediaElementSource(DOM.masterAudio);
                audioSourceNode.connect(analyserNode);
                analyserNode.connect(audioCtx.destination);
            } catch (err) {
                console.warn('Audio node connection restricted or existing:', err);
            }
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    }

    // ==========================================
    // INITIALIZATION
    // ==========================================
    function init() {
        loadTheme();
        loadSessionsFromStorage();
        renderSidebarSessions();
        bindEvents();
        setupCanvases();
        updateTaskOptionsUI();
        updateModelUI(state.currentModelId);
    }

    // ==========================================
    // THEME MANAGEMENT (MONOCHROME DARK / LIGHT)
    // ==========================================
    function loadTheme() {
        const savedTheme = localStorage.getItem('sanada_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('sanada_theme', next);
        drawWaveform();
        if (state.spectrogram.computed) {
            drawSpectrogram();
        }
    }

    // ==========================================
    // SESSIONS / PREVIOUS CHATS STORAGE
    // ==========================================
    function loadSessionsFromStorage() {
        try {
            const saved = localStorage.getItem('sanada_sessions');
            if (saved) {
                state.sessions = JSON.parse(saved);
            } else {
                state.sessions = [...INITIAL_SESSIONS];
                saveSessionsToStorage();
            }
        } catch (e) {
            state.sessions = [...INITIAL_SESSIONS];
        }
    }

    function saveSessionsToStorage() {
        try {
            localStorage.setItem('sanada_sessions', JSON.stringify(state.sessions));
        } catch (e) {
            console.error('Storage error:', e);
        }
    }

    // ==========================================
    // FIND CHAT / SEARCH FUNCTIONALITY
    // ==========================================
    function handleFindChat(query) {
        state.searchQuery = (query || '').trim().toLowerCase();
        
        if (state.searchQuery.length > 0) {
            DOM.clearSearchBtn.style.display = 'flex';
            DOM.searchStatusBar.style.display = 'flex';
        } else {
            DOM.clearSearchBtn.style.display = 'none';
            DOM.searchStatusBar.style.display = 'none';
        }

        renderSidebarSessions();
    }

    function clearSearch() {
        DOM.findChatInput.value = '';
        handleFindChat('');
        DOM.findChatInput.focus();
    }

    function renderSidebarSessions() {
        const query = state.searchQuery;
        let filtered = state.sessions;

        if (query) {
            filtered = state.sessions.filter(sess => {
                const titleMatch = (sess.title || '').toLowerCase().includes(query);
                const fileMatch = (sess.filename || '').toLowerCase().includes(query);
                const modelMatch = (MODELS[sess.modelId]?.name || '').toLowerCase().includes(query);
                const countMatch = `${sess.estimatedCount} source`.includes(query);
                return titleMatch || fileMatch || modelMatch || countMatch;
            });
            DOM.searchResultCount.textContent = `${filtered.length} found`;
        }

        DOM.chatSessionsList.innerHTML = '';

        if (filtered.length === 0) {
            const emptyEl = document.createElement('div');
            emptyEl.style.padding = '24px 12px';
            emptyEl.style.textAlign = 'center';
            emptyEl.style.color = 'var(--text-muted)';
            emptyEl.style.fontSize = '12px';
            emptyEl.textContent = query ? 'No matching chats found' : 'No previous chats yet';
            DOM.chatSessionsList.appendChild(emptyEl);
            return;
        }

        const groups = {};
        filtered.forEach(sess => {
            const cat = sess.dateCategory || 'Recent';
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(sess);
        });

        Object.keys(groups).forEach(cat => {
            const groupLabel = document.createElement('div');
            groupLabel.className = 'chat-group-label';
            groupLabel.textContent = cat;
            DOM.chatSessionsList.appendChild(groupLabel);

            groups[cat].forEach(sess => {
                const item = document.createElement('div');
                item.className = 'chat-session-item' + (state.currentSession?.id === sess.id ? ' active' : '');
                item.dataset.sessionId = sess.id;

                const modelObj = MODELS[sess.modelId] || MODELS['sanada-dualtask'];

                item.innerHTML = `
                    <div class="session-top">
                        <span class="session-title-text" title="${escapeHtml(sess.title)}">${escapeHtml(sess.title)}</span>
                        <button type="button" class="session-delete-btn" title="Delete chat" data-del-id="${sess.id}">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="session-sub-row">
                        <span class="session-model-pill">${escapeHtml(modelObj.name.split(' ')[0])}</span>
                        <span>${escapeHtml(sess.duration || '00:08')}</span>
                        <span class="session-sources-badge">${sess.estimatedCount} Sources</span>
                    </div>
                `;

                item.addEventListener('click', (e) => {
                    if (e.target.closest('.session-delete-btn')) {
                        deleteSession(sess.id);
                        return;
                    }
                    loadSession(sess);
                });

                DOM.chatSessionsList.appendChild(item);
            });
        });
    }

    function deleteSession(id) {
        state.sessions = state.sessions.filter(s => s.id !== id);
        saveSessionsToStorage();
        if (state.currentSession?.id === id) {
            resetToNewSession();
        } else {
            renderSidebarSessions();
        }
    }

    function loadSession(sess) {
        state.currentSession = sess;
        renderSidebarSessions();

        DOM.currentSessionTitle.textContent = sess.title;
        DOM.currentSessionDate.textContent = sess.timestamp;
        selectModel(sess.modelId || 'sanada-dualtask');

        if (sess.presetKey && SAMPLE_PRESETS[sess.presetKey]) {
            loadPreset(sess.presetKey, false);
            showResultsForPreset(sess.presetKey);
        } else {
            DOM.metaFileName.textContent = sess.filename;
            DOM.metaFileSize.textContent = sess.fileSize || 'N/A';
            DOM.metaDuration.textContent = sess.duration || '00:08';
            DOM.audioStateBadge.textContent = 'SESSION LOADED';

            DOM.estimatedCountNumber.textContent = sess.estimatedCount || 3;
            DOM.estimatedConfidence.textContent = `Model Confidence: ${sess.confidence || '96.5%'}`;
            DOM.playerSection.style.display = 'flex';
            DOM.resultsSection.style.display = 'flex';
        }
    }

    function resetToNewSession() {
        stopPlayback();
        stopAllStems();

        state.currentSession = null;
        state.audio.file = null;
        state.audio.url = null;
        state.audio.filename = '';
        state.audio.duration = 0;
        state.audio.audioBuffer = null;
        state.spectrogram.computed = false;
        state.spectrogram.matrix = null;
        state.inference.isRunning = false;

        DOM.masterAudio.src = '';
        DOM.audioFileInput.value = '';

        DOM.currentSessionTitle.textContent = 'Active Audio Analysis';
        DOM.currentSessionDate.textContent = 'Just now';
        DOM.audioStateBadge.textContent = 'NO AUDIO LOADED';

        DOM.playerSection.style.display = 'none';
        DOM.progressSection.style.display = 'none';
        DOM.resultsSection.style.display = 'none';
        DOM.uploadSection.style.display = 'flex';

        renderSidebarSessions();
    }

    // ==========================================
    // AI MODEL SELECTION
    // ==========================================
    let tempSelectedModelId = state.currentModelId;

    function openModelModal() {
        tempSelectedModelId = state.currentModelId;
        document.querySelectorAll('.model-option-card').forEach(card => {
            if (card.dataset.modelId === tempSelectedModelId) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
        DOM.modelModalBackdrop.style.display = 'flex';
    }

    function closeModelModal() {
        DOM.modelModalBackdrop.style.display = 'none';
    }

    function selectModel(modelId) {
        if (!MODELS[modelId]) return;
        state.currentModelId = modelId;
        updateModelUI(modelId);
    }

    function updateModelUI(modelId) {
        const model = MODELS[modelId];
        DOM.currentModelName.textContent = model.name;
        DOM.currentModelSpec.textContent = model.spec;
        DOM.activeModelTag.textContent = `MODEL: ${model.name}`;
        DOM.actionModelName.textContent = model.name;
        DOM.resultModelName.textContent = model.name;
    }

    // ==========================================
    // THE 3 OPTIONS: TAGGING, COUNTING, SEPARATING
    // ==========================================
    function updateTaskOptionsUI() {
        // Update active classes on cards
        DOM.taskCardTagging.classList.toggle('active', state.tasks.tagging);
        DOM.checkTaskTagging.checked = state.tasks.tagging;

        DOM.taskCardCounting.classList.toggle('active', state.tasks.counting);
        DOM.checkTaskCounting.checked = state.tasks.counting;

        DOM.taskCardSeparating.classList.toggle('active', state.tasks.separating);
        DOM.checkTaskSeparating.checked = state.tasks.separating;

        // Build active tasks list string
        const activeList = [];
        if (state.tasks.tagging) activeList.push('Tagging');
        if (state.tasks.counting) activeList.push('Counting');
        if (state.tasks.separating) activeList.push('Separating');

        if (activeList.length === 0) {
            DOM.selectedTasksSummary.textContent = 'None selected (Please select at least 1)';
            DOM.runBtnLabel.textContent = 'Select a Task to Run';
            DOM.runInferenceBtn.disabled = true;
        } else {
            DOM.selectedTasksSummary.textContent = activeList.join(' + ');
            DOM.runBtnLabel.textContent = `Run Selected Tasks (${activeList.length})`;
            DOM.runInferenceBtn.disabled = false;
        }
    }

    function toggleTask(taskName) {
        if (state.tasks[taskName] !== undefined) {
            state.tasks[taskName] = !state.tasks[taskName];
            updateTaskOptionsUI();
        }
    }

    function selectAllTasks() {
        state.tasks.tagging = true;
        state.tasks.counting = true;
        state.tasks.separating = true;
        updateTaskOptionsUI();
    }

    // ==========================================
    // AUDIO LOADING & DRAG-AND-DROP
    // ==========================================
    function handleFileSelect(file) {
        if (!file) return;

        initAudioContext();
        stopPlayback();
        stopAllStems();

        state.audio.file = file;
        state.audio.filename = file.name;
        state.audio.sizeText = formatFileSize(file.size);
        state.audio.url = URL.createObjectURL(file);
        state.spectrogram.computed = false;

        DOM.currentSessionTitle.textContent = file.name.replace(/\.[^/.]+$/, "");
        DOM.currentSessionDate.textContent = 'Today, ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        DOM.metaFileName.textContent = file.name;
        DOM.metaFileSize.textContent = state.audio.sizeText;
        DOM.metaDuration.textContent = 'Analyzing...';
        DOM.metaSampleRate.textContent = 'Detecting...';
        DOM.metaChannels.textContent = 'Detecting...';
        DOM.audioStateBadge.textContent = 'AUDIO LOADED';

        DOM.masterAudio.src = state.audio.url;
        DOM.masterAudio.load();

        const reader = new FileReader();
        reader.onload = function (e) {
            const arrayBuffer = e.target.result;
            audioCtx.decodeAudioData(arrayBuffer, function (decodedBuffer) {
                state.audio.audioBuffer = decodedBuffer;
                state.audio.duration = decodedBuffer.duration;
                state.audio.sampleRate = decodedBuffer.sampleRate;
                DOM.metaDuration.textContent = formatTime(decodedBuffer.duration);
                DOM.totalTimeDisplay.textContent = formatTime(decodedBuffer.duration);
                DOM.metaSampleRate.textContent = `${decodedBuffer.sampleRate.toLocaleString()} Hz`;
                DOM.metaChannels.textContent = decodedBuffer.numberOfChannels === 1 ? 'Mono (1 Ch)' : 'Stereo (2 Ch)';

                generateWaveformPeaks(decodedBuffer);
                drawWaveform();

                // Compute spectrogram if spectrogram view is active
                if (state.activeView === 'spectrogram' || state.activeView === 'dual') {
                    convertAndRenderSpectrogram();
                }
            }, function (err) {
                console.warn('Decode error, falling back:', err);
                generateSyntheticWaveform();
                drawWaveform();
            });
        };
        reader.readAsArrayBuffer(file);

        DOM.playerSection.style.display = 'flex';
        DOM.resultsSection.style.display = 'none';
        DOM.progressSection.style.display = 'none';
        DOM.playerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function loadPreset(presetKey, scrollToView = true) {
        const preset = SAMPLE_PRESETS[presetKey];
        if (!preset) return;

        initAudioContext();
        stopPlayback();
        stopAllStems();

        state.audio.file = null;
        state.audio.filename = preset.filename;
        state.audio.sizeText = '250 KB';
        state.audio.url = preset.url;
        state.spectrogram.computed = false;

        DOM.currentSessionTitle.textContent = preset.title;
        DOM.currentSessionDate.textContent = 'Dataset Sample';
        DOM.metaFileName.textContent = preset.filename;
        DOM.metaFileSize.textContent = '250 KB';
        DOM.metaDuration.textContent = '00:08';
        DOM.totalTimeDisplay.textContent = '00:08';
        DOM.metaSampleRate.textContent = '16,000 Hz';
        DOM.metaChannels.textContent = 'Mono (1 Ch)';
        DOM.audioStateBadge.textContent = 'DATASET MIX LOADED';

        DOM.masterAudio.src = preset.url;
        DOM.masterAudio.load();

        fetch(preset.url)
            .then(res => res.arrayBuffer())
            .then(arrayBuffer => audioCtx.decodeAudioData(arrayBuffer))
            .then(decodedBuffer => {
                state.audio.audioBuffer = decodedBuffer;
                state.audio.duration = decodedBuffer.duration;
                state.audio.sampleRate = decodedBuffer.sampleRate;
                generateWaveformPeaks(decodedBuffer);
                drawWaveform();

                if (state.activeView === 'spectrogram' || state.activeView === 'dual') {
                    convertAndRenderSpectrogram();
                }
            })
            .catch(() => {
                generateSyntheticWaveform();
                drawWaveform();
            });

        DOM.playerSection.style.display = 'flex';
        DOM.resultsSection.style.display = 'none';
        DOM.progressSection.style.display = 'none';

        if (scrollToView) {
            DOM.playerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // ==========================================
    // VIEW SWITCHER (WAVEFORM VS SPECTROGRAM VS DUAL)
    // ==========================================
    function switchView(viewMode) {
        state.activeView = viewMode;

        // Update tab buttons
        DOM.tabWaveform.classList.toggle('active', viewMode === 'waveform');
        DOM.tabSpectrogram.classList.toggle('active', viewMode === 'spectrogram');
        DOM.tabDual.classList.toggle('active', viewMode === 'dual');

        if (viewMode === 'waveform') {
            DOM.waveformPane.style.display = 'flex';
            DOM.spectrogramPane.style.display = 'none';
            DOM.waveformControls.style.display = 'flex';
            DOM.spectrogramControls.style.display = 'none';
            setupCanvases();
            drawWaveform();
        } else if (viewMode === 'spectrogram') {
            DOM.waveformPane.style.display = 'none';
            DOM.spectrogramPane.style.display = 'flex';
            DOM.waveformControls.style.display = 'none';
            DOM.spectrogramControls.style.display = 'flex';
            setupCanvases();
            if (!state.spectrogram.computed && state.audio.audioBuffer) {
                convertAndRenderSpectrogram();
            } else {
                drawSpectrogram();
            }
        } else if (viewMode === 'dual') {
            DOM.waveformPane.style.display = 'flex';
            DOM.spectrogramPane.style.display = 'flex';
            DOM.waveformControls.style.display = 'flex';
            DOM.spectrogramControls.style.display = 'flex';
            setupCanvases();
            drawWaveform();
            if (!state.spectrogram.computed && state.audio.audioBuffer) {
                convertAndRenderSpectrogram();
            } else {
                drawSpectrogram();
            }
        }
    }

    function setupCanvases() {
        // Waveform Canvas
        const waveCanvas = DOM.waveformCanvas;
        const waveRect = waveCanvas.getBoundingClientRect();
        waveCanvas.width = (waveRect.width * (window.devicePixelRatio || 1)) || 800;
        waveCanvas.height = (waveRect.height * (window.devicePixelRatio || 1)) || 120;

        // Spectrogram Canvas
        const specCanvas = DOM.spectrogramCanvas;
        const specRect = specCanvas.getBoundingClientRect();
        specCanvas.width = (specRect.width * (window.devicePixelRatio || 1)) || 800;
        specCanvas.height = (specRect.height * (window.devicePixelRatio || 1)) || 140;
    }

    window.addEventListener('resize', () => {
        setupCanvases();
        drawWaveform();
        if (state.spectrogram.computed) {
            drawSpectrogram();
        }
    });

    // ==========================================
    // 1. WAVEFORM CONVERTER & VIEWER
    // ==========================================
    function generateWaveformPeaks(buffer) {
        const rawData = buffer.getChannelData(0);
        const bars = 140 * state.zoomFactor;
        const blockSize = Math.floor(rawData.length / bars);
        waveformPeaks = [];

        let maxSample = 0;
        let sumSquares = 0;

        for (let i = 0; i < bars; i++) {
            let blockStart = blockSize * i;
            let sum = 0;
            for (let j = 0; j < blockSize; j++) {
                const sample = rawData[blockStart + j] || 0;
                sum += Math.abs(sample);
                const absSample = Math.abs(sample);
                if (absSample > maxSample) maxSample = absSample;
                sumSquares += sample * sample;
            }
            waveformPeaks.push(sum / blockSize);
        }

        // Calculate Peak dBFS and RMS dB
        const peakDbfs = 20 * Math.log10(Math.max(maxSample, 0.0001));
        const rms = Math.sqrt(sumSquares / rawData.length);
        const rmsDbfs = 20 * Math.log10(Math.max(rms, 0.0001));

        state.audio.peakDbfs = peakDbfs;
        state.audio.rmsDbfs = rmsDbfs;
        DOM.peakReadout.textContent = `Peak: ${peakDbfs.toFixed(1)} dBFS | RMS: ${rmsDbfs.toFixed(1)} dB`;
        DOM.waveformStatus.textContent = `Converted: ${rawData.length.toLocaleString()} Samples`;

        // Normalize peaks for rendering
        const maxVal = Math.max(...waveformPeaks, 0.01);
        waveformPeaks = waveformPeaks.map(v => v / maxVal);
    }

    function generateSyntheticWaveform() {
        waveformPeaks = [];
        for (let i = 0; i < 140; i++) {
            waveformPeaks.push(0.15 + Math.random() * 0.75 * Math.sin(i / 10));
        }
    }

    function drawWaveform() {
        const canvas = DOM.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const barColor = isLight ? '#d4d4d8' : '#3f3f46';
        const progressColor = isLight ? '#09090b' : '#ffffff';
        const centerLineColor = isLight ? '#e4e4e7' : '#27272a';

        // Draw center zero-crossing line
        ctx.strokeStyle = centerLineColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        const duration = DOM.masterAudio.duration || state.audio.duration || 1;
        const progressRatio = DOM.masterAudio.currentTime / duration;

        if (waveformPeaks.length === 0) {
            generateSyntheticWaveform();
        }

        const barCount = waveformPeaks.length;
        const barWidth = width / barCount;
        const gap = 1.2 * (window.devicePixelRatio || 1);

        for (let i = 0; i < barCount; i++) {
            const barHeight = Math.max(waveformPeaks[i] * height * 0.85, 3);
            const x = i * barWidth;
            const y = (height - barHeight) / 2;

            if (i / barCount <= progressRatio) {
                ctx.fillStyle = progressColor;
            } else {
                ctx.fillStyle = barColor;
            }

            ctx.fillRect(x + gap / 2, y, barWidth - gap, barHeight);
        }

        // Playhead position
        const playheadX = progressRatio * DOM.waveformContainer.offsetWidth;
        DOM.waveformPlayhead.style.transform = `translateX(${playheadX}px)`;
    }

    // ==========================================
    // 2. SPECTROGRAM CONVERTER & VIEWER (STFT)
    // ==========================================
    function convertAndRenderSpectrogram() {
        if (!state.audio.audioBuffer) {
            DOM.spectrogramStatus.textContent = 'Upload or load an audio file first';
            return;
        }

        DOM.spectrogramStatus.textContent = 'Computing STFT Spectrogram...';

        // Use setTimeout to yield to UI thread so status updates
        setTimeout(() => {
            const buffer = state.audio.audioBuffer;
            const channelData = buffer.getChannelData(0);
            const sampleRate = buffer.sampleRate;

            const fftSize = parseInt(DOM.specFftSizeSelect.value, 10) || 512;
            const windowType = DOM.specWindowSelect.value || 'hann';
            const hopSize = Math.floor(fftSize / 2);

            state.spectrogram.fftSize = fftSize;
            state.spectrogram.windowFunc = windowType;

            // Compute STFT matrix
            const numFrames = Math.floor((channelData.length - fftSize) / hopSize);
            const numBins = Math.floor(fftSize / 2);
            const matrix = [];

            // Precompute window weights
            const windowWeights = new Float32Array(fftSize);
            for (let i = 0; i < fftSize; i++) {
                if (windowType === 'hann') {
                    windowWeights[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
                } else if (windowType === 'hamming') {
                    windowWeights[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (fftSize - 1));
                } else {
                    windowWeights[i] = 1.0;
                }
            }

            let globalMaxMag = 0.00001;

            // Compute STFT frames
            for (let f = 0; f < numFrames; f++) {
                const offset = f * hopSize;
                const frameMags = new Float32Array(numBins);

                // Real DFT for frequency bins
                for (let k = 0; k < numBins; k++) {
                    let real = 0;
                    let imag = 0;
                    const angleStep = (2 * Math.PI * k) / fftSize;

                    // Compute windowed correlation
                    for (let n = 0; n < fftSize; n += 2) { // 2x stride for high speed
                        const val = channelData[offset + n] * windowWeights[n];
                        const angle = angleStep * n;
                        real += val * Math.cos(angle);
                        imag -= val * Math.sin(angle);
                    }

                    const mag = Math.sqrt(real * real + imag * imag);
                    frameMags[k] = mag;
                    if (mag > globalMaxMag) globalMaxMag = mag;
                }

                matrix.push(frameMags);
            }

            state.spectrogram.matrix = matrix;
            state.spectrogram.timeFrames = numFrames;
            state.spectrogram.freqBins = numBins;
            state.spectrogram.computed = true;

            DOM.spectrogramStatus.textContent = `STFT Computed: ${numFrames} Frames × ${numBins} Bins (${fftSize} FFT)`;
            drawSpectrogram();
        }, 30);
    }

    function drawSpectrogram() {
        if (!state.spectrogram.computed || !state.spectrogram.matrix) return;

        const canvas = DOM.spectrogramCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const matrix = state.spectrogram.matrix;
        const numFrames = matrix.length;
        const numBins = state.spectrogram.freqBins;

        if (numFrames === 0 || numBins === 0) return;

        const imgData = ctx.createImageData(width, height);
        const data = imgData.data;

        // Dynamic range threshold in dB
        const minDb = -75;
        const maxDb = 0;

        for (let x = 0; x < width; x++) {
            const frameIdx = Math.floor((x / width) * numFrames);
            const frame = matrix[frameIdx];

            for (let y = 0; y < height; y++) {
                // Invert Y so 0 Hz is at bottom, high freq is at top
                const binIdx = Math.floor(((height - 1 - y) / height) * numBins);
                const mag = frame ? frame[binIdx] : 0;

                // Logarithmic dB scale
                const db = 20 * Math.log10(Math.max(mag, 0.00001));
                const norm = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));

                // Strictly monochrome mapping:
                // Dark mode: 0 (black) to 255 (white)
                // Light mode: 255 (white) to 0 (black)
                let intensity;
                if (!isLight) {
                    intensity = Math.floor(Math.pow(norm, 1.4) * 255);
                } else {
                    intensity = Math.floor(255 - Math.pow(norm, 1.4) * 255);
                }

                const pixelIndex = (y * width + x) * 4;
                data[pixelIndex] = intensity;     // R
                data[pixelIndex + 1] = intensity; // G
                data[pixelIndex + 2] = intensity; // B
                data[pixelIndex + 3] = 255;       // Alpha
            }
        }

        ctx.putImageData(imgData, 0, 0);

        // Update Spectrogram Playhead
        const duration = DOM.masterAudio.duration || state.audio.duration || 1;
        const progressRatio = DOM.masterAudio.currentTime / duration;
        const playheadX = progressRatio * DOM.spectrogramCanvas.offsetWidth;
        DOM.spectrogramPlayhead.style.transform = `translateX(${playheadX}px)`;
    }

    function exportSpectrogramPNG() {
        if (!state.spectrogram.computed) {
            convertAndRenderSpectrogram();
        }
        const canvas = DOM.spectrogramCanvas;
        const link = document.createElement('a');
        link.download = `${state.audio.filename || 'sanada'}_spectrogram.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    // ==========================================
    // AUDIO ANIMATION & TIMELINE PLAYBACK
    // ==========================================
    function startAudioAnimation() {
        if (!state.audio.isPlaying) return;

        if (analyserNode && !DOM.masterAudio.paused) {
            const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
            analyserNode.getByteFrequencyData(dataArray);

            const duration = DOM.masterAudio.duration || 1;
            const currentIdx = Math.floor((DOM.masterAudio.currentTime / duration) * waveformPeaks.length);
            if (waveformPeaks[currentIdx] !== undefined) {
                const liveAmp = (dataArray[8] || 40) / 255;
                waveformPeaks[currentIdx] = Math.max(waveformPeaks[currentIdx], liveAmp * 0.9);
            }
        }

        drawWaveform();

        // Update Spectrogram Playhead
        if (state.spectrogram.computed) {
            const duration = DOM.masterAudio.duration || state.audio.duration || 1;
            const progressRatio = DOM.masterAudio.currentTime / duration;
            const playheadX = progressRatio * DOM.spectrogramCanvas.offsetWidth;
            DOM.spectrogramPlayhead.style.transform = `translateX(${playheadX}px)`;
        }

        updateTimeDisplays();
        animFrameId = requestAnimationFrame(startAudioAnimation);
    }

    function stopAudioAnimation() {
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
        drawWaveform();
        if (state.spectrogram.computed) {
            const duration = DOM.masterAudio.duration || state.audio.duration || 1;
            const progressRatio = DOM.masterAudio.currentTime / duration;
            const playheadX = progressRatio * DOM.spectrogramCanvas.offsetWidth;
            DOM.spectrogramPlayhead.style.transform = `translateX(${playheadX}px)`;
        }
        updateTimeDisplays();
    }

    function togglePlay() {
        if (!DOM.masterAudio.src) return;

        initAudioContext();

        if (DOM.masterAudio.paused) {
            DOM.masterAudio.play().then(() => {
                state.audio.isPlaying = true;
                DOM.playIcon.style.display = 'none';
                DOM.pauseIcon.style.display = 'block';
                startAudioAnimation();
            }).catch(e => console.error('Play failed:', e));
        } else {
            DOM.masterAudio.pause();
            state.audio.isPlaying = false;
            DOM.playIcon.style.display = 'block';
            DOM.pauseIcon.style.display = 'none';
            stopAudioAnimation();
        }
    }

    function stopPlayback() {
        DOM.masterAudio.pause();
        DOM.masterAudio.currentTime = 0;
        state.audio.isPlaying = false;
        DOM.playIcon.style.display = 'block';
        DOM.pauseIcon.style.display = 'none';
        DOM.audioScrubber.value = 0;
        stopAudioAnimation();
    }

    function updateTimeDisplays() {
        const cur = DOM.masterAudio.currentTime || 0;
        const total = DOM.masterAudio.duration || state.audio.duration || 0;

        DOM.currentTimeDisplay.textContent = formatTime(cur);
        DOM.totalTimeDisplay.textContent = formatTime(total);

        if (total > 0) {
            DOM.audioScrubber.value = (cur / total) * 100;
        }
    }

    function seekTo(percentage) {
        const total = DOM.masterAudio.duration || state.audio.duration || 0;
        if (total > 0) {
            DOM.masterAudio.currentTime = (percentage / 100) * total;
            drawWaveform();
            if (state.spectrogram.computed) {
                const playheadX = (percentage / 100) * DOM.spectrogramCanvas.offsetWidth;
                DOM.spectrogramPlayhead.style.transform = `translateX(${playheadX}px)`;
            }
            updateTimeDisplays();
        }
    }

    // ==========================================
    // NEURAL INFERENCE RUNNER
    // ==========================================
    function runInference() {
        if (state.inference.isRunning) return;

        // Check that at least one task is chosen
        if (!state.tasks.tagging && !state.tasks.counting && !state.tasks.separating) {
            alert('Please select at least one task (Tagging, Counting, or Separating) to run.');
            return;
        }

        state.inference.isRunning = true;
        DOM.runInferenceBtn.disabled = true;
        DOM.progressSection.style.display = 'flex';
        DOM.resultsSection.style.display = 'none';
        stopAllStems();

        DOM.progressSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

        const steps = [
            { el: DOM.step1, title: 'Step 1: STFT & Spectrogram Analysis...', pct: 25, delay: 500 },
            { el: DOM.step2, title: 'Step 2: Audio Tagging & Source Count Estimation...', pct: 55, delay: 650 },
            { el: DOM.step3, title: 'Step 3: Neural Mask Demixing & Isolation...', pct: 85, delay: 750 },
            { el: DOM.step4, title: 'Step 4: Waveform Synthesis & Stems Packaging...', pct: 100, delay: 500 }
        ];

        let currentIdx = 0;

        function advanceStep() {
            if (currentIdx >= steps.length) {
                state.inference.isRunning = false;
                DOM.runInferenceBtn.disabled = false;
                DOM.progressSection.style.display = 'none';

                // Display selected blocks based on the 3 options
                DOM.resultBlockTagging.style.display = state.tasks.tagging ? 'flex' : 'none';
                DOM.resultBlockCounting.style.display = state.tasks.counting ? 'flex' : 'none';
                DOM.resultBlockSeparating.style.display = state.tasks.separating ? 'flex' : 'none';

                if (state.audio.filename.includes('3sources') || state.audio.filename.includes('3_sources') || (state.currentSession?.presetKey === 'mix3')) {
                    showResultsForPreset('mix3');
                } else if (state.audio.filename.includes('2sources') || state.audio.filename.includes('2_sources') || (state.currentSession?.presetKey === 'mix2')) {
                    showResultsForPreset('mix2');
                } else {
                    showResultsForCustomAudio();
                }

                DOM.resultsSection.style.display = 'flex';
                DOM.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                return;
            }

            const step = steps[currentIdx];
            DOM.progressStepTitle.textContent = step.title;
            DOM.progressPercentText.textContent = `${step.pct}%`;
            DOM.progressFill.style.width = `${step.pct}%`;

            steps.forEach((s, idx) => {
                if (idx < currentIdx) {
                    s.el.className = 'pipeline-step done';
                } else if (idx === currentIdx) {
                    s.el.className = 'pipeline-step active';
                } else {
                    s.el.className = 'pipeline-step';
                }
            });

            currentIdx++;
            setTimeout(advanceStep, step.delay);
        }

        advanceStep();
    }

    function showResultsForPreset(presetKey) {
        const preset = SAMPLE_PRESETS[presetKey];
        if (!preset) return;

        // 1. Tagging Output
        if (state.tasks.tagging) {
            renderTaggingResults(preset.tags);
        }

        // 2. Counting Output
        if (state.tasks.counting) {
            DOM.estimatedCountNumber.textContent = preset.estimatedCount;
            DOM.estimatedConfidence.textContent = `Model Confidence: ${preset.confidence}`;
            DOM.resultModelName.textContent = MODELS[state.currentModelId].name;
            DOM.compositionalityRating.textContent = preset.polyphony;
            DOM.snrEstimate.textContent = `Est. Source SNR: ${preset.snr}`;
            renderCountingDistribution(preset.distribution);
        }

        // 3. Separating Output
        if (state.tasks.separating) {
            renderStems(preset.stems);
        }
    }

    function showResultsForCustomAudio() {
        const count = state.currentModelId === 'sanada-countnet' ? 4 : 3;

        // 1. Tagging Output
        if (state.tasks.tagging) {
            const customTags = [
                {
                    label: 'Percussive Transient Attack',
                    confidence: '96.2%',
                    pct: 96,
                    category: 'Transient / Impact',
                    freqRange: '2.5 kHz – 8.0 kHz',
                    timeSpan: '0.0s – ' + formatTime(state.audio.duration || 8)
                },
                {
                    label: 'Harmonic Tonal Component',
                    confidence: '94.8%',
                    pct: 95,
                    category: 'Harmonic Formant',
                    freqRange: '300 Hz – 2.8 kHz',
                    timeSpan: '0.0s – ' + formatTime(state.audio.duration || 8)
                },
                {
                    label: 'Stationary Background Ambient',
                    confidence: '91.5%',
                    pct: 91,
                    category: 'Wideband Texture',
                    freqRange: '60 Hz – 1.2 kHz',
                    timeSpan: '0.0s – ' + formatTime(state.audio.duration || 8)
                }
            ];
            renderTaggingResults(customTags);
        }

        // 2. Counting Output
        if (state.tasks.counting) {
            DOM.estimatedCountNumber.textContent = count;
            DOM.estimatedConfidence.textContent = 'Model Confidence: 96.8%';
            DOM.resultModelName.textContent = MODELS[state.currentModelId].name;
            DOM.compositionalityRating.textContent = 'Multi-Source Polyphonic Mixture';
            DOM.snrEstimate.textContent = 'Est. Source SNR: +15.4 dB';

            const customDist = [
                { count: '1 Source', pct: 2.1 },
                { count: '2 Sources', pct: 6.4 },
                { count: '3 Sources', pct: 88.5, isPredicted: count === 3 },
                { count: '4 Sources', pct: 2.6, isPredicted: count === 4 },
                { count: '5+ Sources', pct: 0.4 }
            ];
            renderCountingDistribution(customDist);
        }

        // 3. Separating Output
        if (state.tasks.separating) {
            const stems = [
                {
                    id: 1,
                    name: 'Source 1 — Primary Transient',
                    char: 'Transient sharp attack • De-mixed frequency band: 2.4 kHz - 8.0 kHz',
                    url: state.audio.url,
                    duration: formatTime(state.audio.duration || 8)
                },
                {
                    id: 2,
                    name: 'Source 2 — Harmonic Sustained Formant',
                    char: 'Continuous tonal component • De-mixed frequency band: 300 Hz - 2.2 kHz',
                    url: state.audio.url,
                    duration: formatTime(state.audio.duration || 8)
                },
                {
                    id: 3,
                    name: 'Source 3 — Ambient Background / Noise Floor',
                    char: 'Diffuse stationary acoustic component • SNR threshold: -12 dB',
                    url: state.audio.url,
                    duration: formatTime(state.audio.duration || 8)
                }
            ];
            renderStems(stems);
        }
    }

    // ==========================================
    // RENDER: OPTION 1 TAGGING RESULTS
    // ==========================================
    function renderTaggingResults(tags) {
        DOM.taggingResultsGrid.innerHTML = '';
        DOM.taggingCountSummary.textContent = `${tags.length} Acoustic Sound Classes Detected`;

        tags.forEach(tag => {
            const card = document.createElement('div');
            card.className = 'tag-item-card';

            card.innerHTML = `
                <div class="tag-item-top">
                    <span class="tag-label">${escapeHtml(tag.label)}</span>
                    <span class="tag-confidence">${escapeHtml(tag.confidence)}</span>
                </div>
                <div class="tag-meter">
                    <div class="tag-fill" style="width: ${tag.pct}%;"></div>
                </div>
                <div class="tag-specs">
                    <span>${escapeHtml(tag.category)}</span>
                    <span>${escapeHtml(tag.freqRange)}</span>
                </div>
            `;
            DOM.taggingResultsGrid.appendChild(card);
        });
    }

    // ==========================================
    // RENDER: OPTION 2 COUNTING DISTRIBUTION
    // ==========================================
    function renderCountingDistribution(distList) {
        DOM.distBarsContainer.innerHTML = '';

        distList.forEach(item => {
            const row = document.createElement('div');
            row.className = 'dist-bar-row';

            row.innerHTML = `
                <span class="dist-source-label">${escapeHtml(item.count)}</span>
                <div class="dist-bar-track">
                    <div class="dist-bar-fill" style="width: ${item.pct}%;"></div>
                </div>
                <span class="dist-pct-label">${item.pct.toFixed(1)}%</span>
            `;
            DOM.distBarsContainer.appendChild(row);
        });
    }

    // ==========================================
    // RENDER: OPTION 3 SEPARATING STEMS PREVIEWS
    // ==========================================
    function renderStems(stems) {
        stopAllStems();
        DOM.stemsGrid.innerHTML = '';
        state.stemPlayers = [];

        stems.forEach(stem => {
            const card = document.createElement('div');
            card.className = 'stem-card';
            card.dataset.stemId = stem.id;

            card.innerHTML = `
                <div class="stem-top-bar">
                    <div class="stem-identity">
                        <span class="stem-badge">STEM 0${stem.id}</span>
                        <div class="stem-title-wrap">
                            <span class="stem-name">${escapeHtml(stem.name)}</span>
                            <span class="stem-char">${escapeHtml(stem.char)}</span>
                        </div>
                    </div>
                    <div class="stem-actions">
                        <button type="button" class="btn-tag-action solo-btn" data-stem-id="${stem.id}" title="Solo this sound">SOLO</button>
                        <button type="button" class="btn-tag-action mute-btn" data-stem-id="${stem.id}" title="Mute this sound">MUTE</button>
                        <a href="${stem.url}" download="${state.audio.filename || 'sanada'}_stem_${stem.id}.wav" class="btn-sm-outline" title="Download separated stem">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7 10 12 15 17 10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                            <span>.WAV</span>
                        </a>
                    </div>
                </div>

                <div class="stem-player-row">
                    <button type="button" class="stem-play-btn" data-stem-id="${stem.id}" title="Play / Pause Stem">
                        <svg class="stem-play-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        <svg class="stem-pause-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="display: none;">
                            <rect x="6" y="4" width="4" height="16"></rect>
                            <rect x="14" y="4" width="4" height="16"></rect>
                        </svg>
                    </button>

                    <div class="stem-waveform-box" data-stem-id="${stem.id}">
                        <canvas class="stem-canvas" id="stemCanvas_${stem.id}"></canvas>
                        <div class="stem-playhead" id="stemPlayhead_${stem.id}"></div>
                    </div>

                    <span class="stem-time" id="stemTime_${stem.id}">00:00</span>
                </div>
            `;

            const stemAudio = new Audio(stem.url);
            stemAudio.preload = 'auto';

            const stemObj = {
                id: stem.id,
                audio: stemAudio,
                isPlaying: false,
                isMuted: false,
                isSoloed: false,
                card: card,
                playBtn: card.querySelector('.stem-play-btn'),
                playIcon: card.querySelector('.stem-play-icon'),
                pauseIcon: card.querySelector('.stem-pause-icon'),
                canvas: card.querySelector('.stem-canvas'),
                playhead: card.querySelector('.stem-playhead'),
                timeLabel: card.querySelector('.stem-time'),
                soloBtn: card.querySelector('.solo-btn'),
                muteBtn: card.querySelector('.mute-btn'),
                peaks: []
            };

            for (let i = 0; i < 60; i++) {
                stemObj.peaks.push(0.15 + Math.random() * 0.7);
            }

            stemAudio.addEventListener('timeupdate', () => {
                const cur = stemAudio.currentTime;
                const dur = stemAudio.duration || 8;
                stemObj.timeLabel.textContent = formatTime(cur);

                const pct = cur / dur;
                const boxWidth = card.querySelector('.stem-waveform-box').offsetWidth;
                stemObj.playhead.style.transform = `translateX(${pct * boxWidth}px)`;
            });

            stemAudio.addEventListener('ended', () => {
                stemObj.isPlaying = false;
                stemObj.playIcon.style.display = 'block';
                stemObj.pauseIcon.style.display = 'none';
                stemObj.playhead.style.transform = 'translateX(0px)';
                stemObj.timeLabel.textContent = '00:00';
            });

            stemObj.playBtn.addEventListener('click', () => {
                toggleStemPlayback(stemObj);
            });

            const waveformBox = card.querySelector('.stem-waveform-box');
            waveformBox.addEventListener('click', (e) => {
                const rect = waveformBox.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, clickX / rect.width));
                const dur = stemAudio.duration || 8;
                stemAudio.currentTime = ratio * dur;
                stemObj.playhead.style.transform = `translateX(${clickX}px)`;
            });

            stemObj.soloBtn.addEventListener('click', () => {
                toggleStemSolo(stemObj);
            });

            stemObj.muteBtn.addEventListener('click', () => {
                toggleStemMute(stemObj);
            });

            state.stemPlayers.push(stemObj);
            DOM.stemsGrid.appendChild(card);

            setTimeout(() => drawStemCanvas(stemObj), 20);
        });
    }

    function drawStemCanvas(stemObj) {
        const canvas = stemObj.canvas;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * (window.devicePixelRatio || 1) || 400;
        canvas.height = rect.height * (window.devicePixelRatio || 1) || 36;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        ctx.fillStyle = isLight ? '#71717a' : '#52525b';

        const barCount = stemObj.peaks.length;
        const barWidth = width / barCount;
        const gap = 1 * (window.devicePixelRatio || 1);

        for (let i = 0; i < barCount; i++) {
            const h = Math.max(stemObj.peaks[i] * height * 0.8, 2);
            const x = i * barWidth;
            const y = (height - h) / 2;
            ctx.fillRect(x + gap / 2, y, barWidth - gap, h);
        }
    }

    function toggleStemPlayback(stemObj) {
        if (stemObj.isPlaying) {
            stemObj.audio.pause();
            stemObj.isPlaying = false;
            stemObj.playIcon.style.display = 'block';
            stemObj.pauseIcon.style.display = 'none';
        } else {
            if (!DOM.masterAudio.paused) {
                togglePlay();
            }

            stemObj.audio.play().then(() => {
                stemObj.isPlaying = true;
                stemObj.playIcon.style.display = 'none';
                stemObj.pauseIcon.style.display = 'block';
            }).catch(e => console.error('Stem play failed:', e));
        }
    }

    function toggleStemSolo(targetStem) {
        const isNowSoloed = !targetStem.isSoloed;
        state.stemPlayers.forEach(stem => {
            stem.isSoloed = false;
            stem.soloBtn.classList.remove('active');
        });

        if (isNowSoloed) {
            targetStem.isSoloed = true;
            targetStem.soloBtn.classList.add('active');
            state.stemPlayers.forEach(stem => {
                if (stem.id !== targetStem.id) {
                    stem.audio.muted = true;
                    stem.muteBtn.classList.add('active');
                } else {
                    stem.audio.muted = false;
                    stem.muteBtn.classList.remove('active');
                }
            });
        } else {
            state.stemPlayers.forEach(stem => {
                stem.audio.muted = false;
                stem.muteBtn.classList.remove('active');
            });
        }
    }

    function toggleStemMute(stemObj) {
        stemObj.isMuted = !stemObj.isMuted;
        stemObj.audio.muted = stemObj.isMuted;
        stemObj.muteBtn.classList.toggle('active', stemObj.isMuted);
    }

    function stopAllStems() {
        state.stemPlayers.forEach(stem => {
            stem.audio.pause();
            stem.audio.currentTime = 0;
            stem.isPlaying = false;
            stem.playIcon.style.display = 'block';
            stem.pauseIcon.style.display = 'none';
            stem.playhead.style.transform = 'translateX(0px)';
            stem.timeLabel.textContent = '00:00';
        });
    }

    function playAllStems() {
        if (!DOM.masterAudio.paused) {
            togglePlay();
        }
        state.stemPlayers.forEach(stem => {
            stem.audio.currentTime = 0;
            stem.audio.play().then(() => {
                stem.isPlaying = true;
                stem.playIcon.style.display = 'none';
                stem.pauseIcon.style.display = 'block';
            }).catch(e => console.error(e));
        });
    }

    // ==========================================
    // SAVE SESSION TO HISTORY
    // ==========================================
    function saveCurrentSession() {
        const id = 'sess-' + Date.now();
        const title = DOM.currentSessionTitle.textContent || 'Audio Analysis Session';
        const estimatedCount = parseInt(DOM.estimatedCountNumber.textContent, 10) || 3;

        const newSession = {
            id: id,
            title: title,
            timestamp: 'Today, ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            dateCategory: 'Today',
            modelId: state.currentModelId,
            filename: state.audio.filename || 'custom_recording.wav',
            fileSize: state.audio.sizeText || '512 KB',
            duration: DOM.metaDuration.textContent || '00:08',
            estimatedCount: estimatedCount,
            confidence: DOM.estimatedConfidence.textContent.replace('Model Confidence: ', ''),
            presetKey: state.currentSession?.presetKey || null
        };

        state.sessions.unshift(newSession);
        state.currentSession = newSession;
        saveSessionsToStorage();
        renderSidebarSessions();

        const originalText = DOM.saveSessionBtn.innerHTML;
        DOM.saveSessionBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span>Saved!</span>
        `;
        setTimeout(() => {
            DOM.saveSessionBtn.innerHTML = originalText;
        }, 1500);
    }

    // ==========================================
    // EVENT LISTENERS & BINDINGS
    // ==========================================
    function bindEvents() {
        // Theme Toggle
        DOM.themeToggleBtn.addEventListener('click', toggleTheme);

        // Sidebar Toggle
        DOM.toggleSidebarBtn.addEventListener('click', () => {
            DOM.sidebar.classList.toggle('collapsed');
        });

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
                e.preventDefault();
                DOM.sidebar.classList.toggle('collapsed');
            } else if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
                e.preventDefault();
                togglePlay();
            }
        });

        // New Session Buttons
        DOM.headerNewChatBtn.addEventListener('click', resetToNewSession);
        DOM.sidebarNewBtn.addEventListener('click', resetToNewSession);

        // Find Chat Search
        DOM.findChatInput.addEventListener('input', (e) => {
            handleFindChat(e.target.value);
        });
        DOM.clearSearchBtn.addEventListener('click', clearSearch);

        // Model Selector Modal
        DOM.modelSelectTrigger.addEventListener('click', openModelModal);
        DOM.closeModelModalBtn.addEventListener('click', closeModelModal);
        DOM.cancelModelSelectBtn.addEventListener('click', closeModelModal);

        DOM.modelOptionsList.addEventListener('click', (e) => {
            const card = e.target.closest('.model-option-card');
            if (!card) return;
            tempSelectedModelId = card.dataset.modelId;
            document.querySelectorAll('.model-option-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
        });

        DOM.confirmModelSelectBtn.addEventListener('click', () => {
            selectModel(tempSelectedModelId);
            closeModelModal();
        });

        DOM.modelModalBackdrop.addEventListener('click', (e) => {
            if (e.target === DOM.modelModalBackdrop) {
                closeModelModal();
            }
        });

        // Drop Zone File Input
        DOM.dropZone.addEventListener('click', () => {
            DOM.audioFileInput.click();
        });

        DOM.audioFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });

        DOM.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.add('drag-active');
        });

        DOM.dropZone.addEventListener('dragleave', () => {
            DOM.dropZone.classList.remove('drag-active');
        });

        DOM.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.dropZone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Preset Quick Buttons
        DOM.loadSample3Btn.addEventListener('click', () => loadPreset('mix3'));
        DOM.loadSample2Btn.addEventListener('click', () => loadPreset('mix2'));

        // Replace / Remove Audio Buttons
        DOM.replaceAudioBtn.addEventListener('click', () => DOM.audioFileInput.click());
        DOM.removeAudioBtn.addEventListener('click', resetToNewSession);

        // View Tabs (Waveform vs Spectrogram vs Dual)
        DOM.tabWaveform.addEventListener('click', () => switchView('waveform'));
        DOM.tabSpectrogram.addEventListener('click', () => switchView('spectrogram'));
        DOM.tabDual.addEventListener('click', () => switchView('dual'));

        // Converter Actions
        DOM.btnConvertWaveform.addEventListener('click', () => {
            if (state.audio.audioBuffer) {
                generateWaveformPeaks(state.audio.audioBuffer);
                drawWaveform();
            }
        });

        // Zoom Buttons
        document.querySelectorAll('.zoom-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.zoomFactor = parseInt(btn.dataset.zoom, 10) || 1;
                if (state.audio.audioBuffer) {
                    generateWaveformPeaks(state.audio.audioBuffer);
                    drawWaveform();
                }
            });
        });

        // Spectrogram Controls
        DOM.btnConvertSpectrogram.addEventListener('click', convertAndRenderSpectrogram);
        DOM.btnExportSpectrogram.addEventListener('click', exportSpectrogramPNG);

        // Spectrogram Hover Crosshair & Tooltip
        DOM.spectrogramContainer.addEventListener('mousemove', (e) => {
            if (!state.spectrogram.computed) return;

            const rect = DOM.spectrogramCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
                DOM.spectrogramCrosshair.style.display = 'none';
                DOM.spectrogramTooltip.style.display = 'none';
                return;
            }

            DOM.spectrogramCrosshair.style.display = 'block';
            DOM.spectrogramTooltip.style.display = 'block';

            DOM.spectrogramCrosshair.querySelector('.crosshair-x').style.left = `${mouseX}px`;
            DOM.spectrogramCrosshair.querySelector('.crosshair-y').style.top = `${mouseY}px`;

            const dur = DOM.masterAudio.duration || state.audio.duration || 8;
            const timeAtCursor = (mouseX / rect.width) * dur;
            const maxFreq = (state.audio.sampleRate || 16000) / 2;
            const freqAtCursor = Math.round((1 - mouseY / rect.height) * maxFreq);

            // Read magnitude from matrix
            const numFrames = state.spectrogram.timeFrames || 1;
            const numBins = state.spectrogram.freqBins || 1;
            const frameIdx = Math.floor((mouseX / rect.width) * numFrames);
            const binIdx = Math.floor((1 - mouseY / rect.height) * numBins);

            let db = -80;
            if (state.spectrogram.matrix && state.spectrogram.matrix[frameIdx]) {
                const mag = state.spectrogram.matrix[frameIdx][binIdx] || 0.00001;
                db = (20 * Math.log10(Math.max(mag, 0.00001))).toFixed(1);
            }

            DOM.spectrogramTooltip.textContent = `${timeAtCursor.toFixed(2)}s • ${freqAtCursor.toLocaleString()} Hz • ${db} dBFS`;
        });

        DOM.spectrogramContainer.addEventListener('mouseleave', () => {
            DOM.spectrogramCrosshair.style.display = 'none';
            DOM.spectrogramTooltip.style.display = 'none';
        });

        // Spectrogram Click to Seek
        DOM.spectrogramContainer.addEventListener('click', (e) => {
            const rect = DOM.spectrogramCanvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const ratio = Math.max(0, Math.min(1, clickX / rect.width));
            seekTo(ratio * 100);
        });

        // 3 Tasks Selection Events: Tagging, Counting, Separating
        DOM.taskCardTagging.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT') {
                toggleTask('tagging');
            }
        });
        DOM.checkTaskTagging.addEventListener('change', () => {
            state.tasks.tagging = DOM.checkTaskTagging.checked;
            updateTaskOptionsUI();
        });

        DOM.taskCardCounting.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT') {
                toggleTask('counting');
            }
        });
        DOM.checkTaskCounting.addEventListener('change', () => {
            state.tasks.counting = DOM.checkTaskCounting.checked;
            updateTaskOptionsUI();
        });

        DOM.taskCardSeparating.addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT') {
                toggleTask('separating');
            }
        });
        DOM.checkTaskSeparating.addEventListener('change', () => {
            state.tasks.separating = DOM.checkTaskSeparating.checked;
            updateTaskOptionsUI();
        });

        DOM.btnSelectAllTasks.addEventListener('click', selectAllTasks);

        // Player Controls
        DOM.playPauseBtn.addEventListener('click', togglePlay);
        DOM.stopBtn.addEventListener('click', stopPlayback);

        DOM.audioScrubber.addEventListener('input', (e) => {
            seekTo(parseFloat(e.target.value));
        });

        // Waveform Container Click & Tooltip
        DOM.waveformContainer.addEventListener('click', (e) => {
            const rect = DOM.waveformContainer.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const ratio = Math.max(0, Math.min(1, clickX / rect.width));
            seekTo(ratio * 100);
        });

        DOM.waveformContainer.addEventListener('mousemove', (e) => {
            const rect = DOM.waveformContainer.getBoundingClientRect();
            const hoverX = e.clientX - rect.left;
            const ratio = Math.max(0, Math.min(1, hoverX / rect.width));
            const total = DOM.masterAudio.duration || state.audio.duration || 0;
            DOM.waveformHoverTime.textContent = formatTime(ratio * total);
            DOM.waveformHoverTime.style.display = 'block';
        });

        DOM.waveformContainer.addEventListener('mouseleave', () => {
            DOM.waveformHoverTime.style.display = 'none';
        });

        // Master Audio Ended
        DOM.masterAudio.addEventListener('ended', () => {
            if (state.audio.isLooping) {
                DOM.masterAudio.currentTime = 0;
                DOM.masterAudio.play();
            } else {
                stopPlayback();
            }
        });

        // Volume & Mute
        DOM.volumeSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            DOM.masterAudio.volume = val;
            state.audio.volume = val;
            DOM.masterAudio.muted = false;
            state.audio.isMuted = false;
        });

        DOM.muteBtn.addEventListener('click', () => {
            state.audio.isMuted = !state.audio.isMuted;
            DOM.masterAudio.muted = state.audio.isMuted;
            DOM.muteBtn.classList.toggle('active', state.audio.isMuted);
        });

        // Speed Selector
        DOM.playbackSpeedSelect.addEventListener('change', (e) => {
            const speed = parseFloat(e.target.value);
            DOM.masterAudio.playbackRate = speed;
            state.audio.playbackRate = speed;
        });

        // Loop Toggle
        DOM.loopBtn.addEventListener('click', () => {
            state.audio.isLooping = !state.audio.isLooping;
            DOM.masterAudio.loop = state.audio.isLooping;
            DOM.loopBtn.classList.toggle('active', state.audio.isLooping);
        });

        // Run Inference
        DOM.runInferenceBtn.addEventListener('click', runInference);

        // Stems Controls
        DOM.playAllStemsBtn.addEventListener('click', playAllStems);
        DOM.stopAllStemsBtn.addEventListener('click', stopAllStems);

        // Save Session
        DOM.saveSessionBtn.addEventListener('click', saveCurrentSession);
    }

    // ==========================================
    // UTILITY FUNCTIONS
    // ==========================================
    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '00:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function formatFileSize(bytes) {
        if (!bytes || isNaN(bytes)) return '0 KB';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Bootstrap
    document.addEventListener('DOMContentLoaded', init);

})();

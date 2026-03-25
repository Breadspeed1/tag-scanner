import { CARD_LAYOUT, MAX_SCENE_DIM, MAX_DETECT_DIM } from './config.js';
import { initReference, detect, getRefInfo } from './detector.js';

// OpenCV.js (Emscripten) throws C++ exceptions as raw WASM heap pointers (numbers).
// Decode them with cv.exceptionFromPtr(); fall back gracefully for anything else.
function cvErrMsg(err) {
    if (typeof err === 'number') {
        try { return cv.exceptionFromPtr(err).msg; } catch (_) { return `OpenCV ptr=${err}`; }
    }
    if (!err) return String(err);
    if (typeof err === 'string') return err;
    if (err instanceof Error) {
        // Errors re-thrown from detect() carry the raw OpenCV pointer as .cause
        const base = err.message;
        if (typeof err.cause === 'number') {
            try { return `${base}: ${cv.exceptionFromPtr(err.cause).msg}`; } catch (_) {}
        }
        return base;
    }
    if (typeof err === 'object') return err.msg ?? err.message ?? JSON.stringify(err);
    return String(err);
}

// Scale all scene-space coordinates in detection results to display canvas space.
function scalePointsToDisplay({ cards, debug }, f) {
    for (const p of debug.scenePoints)    { p.x *= f; p.y *= f; }
    for (const m of debug.goodMatches)    { m.scenePt.x *= f; m.scenePt.y *= f; }
    for (const c of cards) {
        for (const p of c.qrCorners)      { p.x *= f; p.y *= f; }
        for (const p of c.skuCorners)     { p.x *= f; p.y *= f; }
        c.center.x *= f; c.center.y *= f;
    }
}

function waitForOpenCV() {
    return new Promise(resolve => {
        const poll = () => {
            if (typeof cv !== 'undefined' && typeof cv.Mat !== 'undefined') resolve();
            else setTimeout(poll, 50);
        };
        poll();
    });
}

function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load ${file.name}`));
        img.src = url;
    });
}

document.addEventListener('alpine:init', () => {
    Alpine.data('scanner', () => {
        // Non-reactive closure variables (DOM elements, workers, images)
        let refImageEl  = null;
        let sceneImageEl = null;   // original full-resolution scene image
        let displayScale = 1;      // ratio of display canvas size to original image
        let ocrWorker   = null;
        let sceneCanvas  = null;
        let overlayCanvas = null;
        let sceneCtx     = null;
        let overlayCtx   = null;

        return {
            // --- Reactive state ---
            cvReady: false,
            refLoaded: false,
            sceneLoaded: false,
            refPreviewSrc: '',
            detecting: false,
            status: '',
            statusLevel: 'info',
            logsOpen: true,
            logs: [],
            cards: [],   // { qrCorners, skuCorners, center, inlierCount, text, pending }
            debugVisible: false,
            debugStats: { refPoints: 0, scenePoints: 0, goodMatches: 0, clusters: 0 },

            get canDetect() {
                return this.cvReady && this.refLoaded && this.sceneLoaded && !this.detecting;
            },

            // --- Logging ---
            _log(msg, level = 'info') {
                const time = new Date().toLocaleTimeString('en', { hour12: false });
                this.logs.unshift({ msg, level, time });
                if (this.logs.length > 200) this.logs.pop();
                const fn = level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'log';
                console[fn](`[${time}] ${msg}`);
            },

            _setStatus(msg, level = 'info') {
                this.status = msg;
                this.statusLevel = level;
                this._log(msg, level);
            },

            // --- Lifecycle (called automatically by Alpine) ---
            async init() {
                sceneCanvas  = document.getElementById('scene-canvas');
                overlayCanvas = document.getElementById('overlay-canvas');
                sceneCtx     = sceneCanvas.getContext('2d');
                overlayCtx   = overlayCanvas.getContext('2d');

                ocrWorker = new Worker('js/ocr-worker.js');
                ocrWorker.onmessage = (e) => this.handleOCRResult(e);
                ocrWorker.onerror   = (e) => {
                    this._log(`OCR worker crashed: ${e.message}`, 'error');
                    // Unblock any pending cards so detecting doesn't stay true forever
                    this.cards = this.cards.map(c =>
                        c.pending ? { ...c, text: '', pending: false } : c
                    );
                    this.detecting = false;
                    this._setStatus('OCR worker failed — see log', 'error');
                };
                this._log('OCR worker started');

                this._log('Waiting for OpenCV.js WASM…');
                this._setStatus('Loading OpenCV.js…');
                await waitForOpenCV();

                if (typeof cv.AKAZE === 'undefined') {
                    this._log('cv.AKAZE is undefined — this OpenCV build may not include features2d', 'error');
                    this._setStatus('OpenCV loaded but AKAZE unavailable — see log', 'error');
                    return;
                }

                this.cvReady = true;
                this._log('OpenCV ready — AKAZE confirmed');
                this._setStatus('Ready. Load a reference QR and a scene image.');
            },

            // --- File inputs ---
            async onRefChange(e) {
                const file = e.target.files[0];
                if (!file) return;
                try {
                    refImageEl = await loadImageFromFile(file);
                    this.refPreviewSrc = refImageEl.src;
                    this.refLoaded = true;
                    this._log(`Reference loaded: ${file.name} (${refImageEl.naturalWidth}×${refImageEl.naturalHeight}px)`);
                } catch (err) {
                    this._log(`Failed to load reference image: ${err.message}`, 'error');
                }
            },

            async onSceneChange(e) {
                const file = e.target.files[0];
                if (!file) return;
                try {
                    const img = await loadImageFromFile(file);
                    sceneImageEl = img;

                    const origW = img.naturalWidth;
                    const origH = img.naturalHeight;

                    // Display canvas is downscaled so the UI stays fast.
                    // Detection will use the original image separately.
                    const scale = Math.min(1, MAX_SCENE_DIM / Math.max(origW, origH));
                    displayScale = scale;
                    const w = Math.round(origW * scale);
                    const h = Math.round(origH * scale);

                    sceneCanvas.width    = w;
                    sceneCanvas.height   = h;
                    overlayCanvas.width  = w;
                    overlayCanvas.height = h;
                    sceneCtx.drawImage(img, 0, 0, w, h);
                    overlayCtx.clearRect(0, 0, w, h);

                    this.sceneLoaded = true;
                    const scaleNote = scale < 1 ? ` (display downscaled to ${w}×${h})` : '';
                    this._log(`Scene loaded: ${file.name} (${origW}×${origH}px${scaleNote})`);
                } catch (err) {
                    this._log(`Failed to load scene image: ${err.message}`, 'error');
                }
            },

            // --- Detection ---
            runDetection() {
                this.detecting = true;
                this.cards = [];
                this._setStatus('Running detection…');

                // Defer so Alpine re-renders the disabled button before the synchronous work blocks
                setTimeout(() => {
                    try {
                        this._doDetection();
                    } catch (err) {
                        this._log(`Detection threw: ${cvErrMsg(err)}`, 'error');
                        this._setStatus(`Detection error — see log`, 'error');
                        this.detecting = false;
                    }
                }, 0);
            },

            _doDetection() {
                this._log('Initializing reference features…');
                try {
                    initReference(refImageEl);
                } catch (err) {
                    throw new Error(`initReference failed: ${cvErrMsg(err)}`);
                }
                const info = getRefInfo();
                this._log(`Reference: ${info.keypointCount} keypoints (${info.width}×${info.height}px)`);
                if (info.keypointCount === 0) {
                    this._log('Zero keypoints on reference — image may be blank or too uniform', 'warn');
                }

                // Detection runs on a temporary canvas at a capped resolution.
                // We never attempt full resolution — large phone photos will OOM the WASM
                // heap, and a failed allocation corrupts it, making smaller retries also fail.
                const origW = sceneImageEl.naturalWidth;
                const origH = sceneImageEl.naturalHeight;
                const maxDim = Math.max(origW, origH);

                // Candidates: MAX_DETECT_DIM (1920), then MAX_SCENE_DIM (1280) as fallback.
                // Deduped and descending so we try the best quality first.
                const candidateScales = [MAX_DETECT_DIM / maxDim, MAX_SCENE_DIM / maxDim]
                    .map(s => Math.min(1, s))
                    .filter((s, i, a) => a.findIndex(x => Math.abs(x - s) < 0.01) === i)
                    .sort((a, b) => b - a);

                let rawCards = null;
                let debug = null;
                let detScale = 1;

                for (const tryScale of candidateScales) {
                    const detW = Math.round(origW * tryScale);
                    const detH = Math.round(origH * tryScale);
                    this._log(`Trying detection at ${detW}×${detH}px…`);

                    const detCanvas = document.createElement('canvas');
                    detCanvas.width  = detW;
                    detCanvas.height = detH;
                    detCanvas.getContext('2d').drawImage(sceneImageEl, 0, 0, detW, detH);

                    const imageData = detCanvas.getContext('2d').getImageData(0, 0, detW, detH);
                    const sceneMat = cv.matFromImageData(imageData);

                    try {
                        ({ cards: rawCards, debug } = detect(sceneMat));
                        sceneMat.delete();
                        detScale = tryScale;
                        this._log(`Detection succeeded at ${detW}×${detH}px`);
                        break;
                    } catch (err) {
                        sceneMat.delete();
                        const msg = cvErrMsg(err);
                        if (msg.includes('Failed to allocate') || msg.includes('Insufficient memory')) {
                            this._log(`OOM at ${detW}×${detH}px, trying smaller…`, 'warn');
                        } else {
                            throw new Error(`detect() failed: ${msg}`);
                        }
                    }
                }

                if (!rawCards) {
                    throw new Error('Detection failed at all scales due to insufficient memory');
                }

                // Map coordinates from detection space → display canvas space
                const coordScale = displayScale / detScale;
                if (coordScale !== 1) {
                    this._log(`Remapping coords by ${coordScale.toFixed(3)} (det→display)`);
                    scalePointsToDisplay(rawCards, debug, coordScale);
                }

                // Render debug images regardless of card count
                this._renderDebug(debug, refImageEl);

                const clusterCount = debug.labels
                    ? new Set(debug.labels.filter(l => l >= 0)).size : 0;
                this._log(`Scene keypoints: ${debug.scenePoints.length} | Good matches: ${debug.goodMatches.length} | Clusters: ${clusterCount}`);
                this._log(`Detection complete: ${rawCards.length} card(s) found`);

                if (rawCards.length === 0) {
                    this._setStatus('No cards detected. Check debug images for clues.', 'warn');
                    this.detecting = false;
                    return;
                }

                let ocrDispatched = 0;
                const newCards = rawCards.map((card, i) => {
                    const entry = {
                        qrCorners:   card.qrCorners,
                        skuCorners:  card.skuCorners,
                        center:      card.center,
                        inlierCount: card.inlierCount,
                        text:    null,
                        pending: !!card.skuCrop,
                    };

                    if (card.skuCrop) {
                        const cropData = new Uint8Array(card.skuCrop.data);
                        ocrWorker.postMessage({
                            cardId: i,
                            cropData,
                            cropWidth:  CARD_LAYOUT.crop_width,
                            cropHeight: CARD_LAYOUT.crop_height,
                        });
                        this._log(`Card ${i + 1}: OCR dispatched (${card.inlierCount} inliers)`);
                        ocrDispatched++;
                    } else {
                        this._log(`Card ${i + 1}: no SKU crop`, 'warn');
                    }

                    card.homography.delete();
                    if (card.skuCrop) card.skuCrop.delete();
                    return entry;
                });

                this.cards = newCards;
                this._drawOverlays();

                if (ocrDispatched === 0) {
                    this._setStatus(`Found ${rawCards.length} card(s) — no SKU regions extracted`, 'warn');
                    this.detecting = false;
                } else {
                    this._setStatus(`Found ${rawCards.length} card(s) — OCR running (${ocrDispatched} job${ocrDispatched !== 1 ? 's' : ''})…`);
                }
            },

            // --- OCR result handler ---
            handleOCRResult(event) {
                const { cardId, text, error } = event.data;

                if (error) {
                    this._log(`Card ${cardId + 1} OCR error: ${error}`, 'error');
                    this.cards[cardId] = { ...this.cards[cardId], text: '', pending: false };
                } else {
                    this._log(`Card ${cardId + 1} OCR result: "${text}"`);
                    this.cards[cardId] = { ...this.cards[cardId], text, pending: false };
                }

                this._drawOverlays();

                const remaining = this.cards.filter(c => c.pending).length;
                if (remaining === 0) {
                    this._log('All OCR jobs complete');
                    this._setStatus(`Done — ${this.cards.length} card(s) found`, 'success');
                    this.detecting = false;
                } else {
                    this._log(`OCR: ${remaining} job(s) remaining`);
                }
            },

            // --- Debug visualization ---
            _renderDebug(debug, refImg) {
                const CLUSTER_COLORS = [
                    '#ff6b6b','#ffd93d','#6bcb77','#4d96ff',
                    '#f72585','#7209b7','#4cc9f0','#fb8500',
                    '#06d6a0','#ef476f','#118ab2','#ffc8dd',
                ];

                const clusterCount = debug.labels
                    ? new Set(debug.labels.filter(l => l >= 0)).size : 0;

                this.debugStats = {
                    refPoints:   debug.refPoints.length,
                    scenePoints: debug.scenePoints.length,
                    goodMatches: debug.goodMatches.length,
                    clusters:    clusterCount,
                };

                // Draw a scene background (at display scale) onto a debug canvas
                const bg = (canvas) => {
                    canvas.width  = sceneCanvas.width;
                    canvas.height = sceneCanvas.height;
                    canvas.getContext('2d').drawImage(sceneCanvas, 0, 0);
                };

                // Draw filled circles
                const dots = (ctx, pts, color, r = 3) => {
                    ctx.fillStyle = color;
                    for (const p of pts) {
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                        ctx.fill();
                    }
                };

                // Radius scaled so dots are visible regardless of image size
                const r = Math.max(3, Math.round(sceneCanvas.width / 200));

                // 1. Reference keypoints
                const refKpCanvas = document.getElementById('debug-ref-kp');
                if (refImg) {
                    refKpCanvas.width  = refImg.naturalWidth;
                    refKpCanvas.height = refImg.naturalHeight;
                    const ctx = refKpCanvas.getContext('2d');
                    ctx.drawImage(refImg, 0, 0);
                    dots(ctx, debug.refPoints, '#00ff44', Math.max(2, Math.round(refImg.naturalWidth / 150)));
                }

                // 2. All scene keypoints
                const sceneKpCanvas = document.getElementById('debug-scene-kp');
                bg(sceneKpCanvas);
                dots(sceneKpCanvas.getContext('2d'), debug.scenePoints, '#00ff44', r);

                // 3. Good matches — scene side highlighted; show count overlay
                const matchCanvas = document.getElementById('debug-matches');
                bg(matchCanvas);
                {
                    const ctx = matchCanvas.getContext('2d');
                    dots(ctx, debug.goodMatches.map(m => m.scenePt), '#ff9900', r * 2);
                    // Label each match with its index so even a single one is obvious
                    ctx.fillStyle = '#fff';
                    ctx.font = `bold ${r * 4}px monospace`;
                    ctx.textAlign = 'center';
                    debug.goodMatches.forEach((m, i) => {
                        ctx.fillText(i + 1, m.scenePt.x, m.scenePt.y - r * 2 - 2);
                    });
                    if (debug.goodMatches.length === 0) {
                        ctx.fillStyle = 'rgba(255,80,80,0.85)';
                        ctx.fillRect(0, 0, matchCanvas.width, matchCanvas.height);
                        ctx.fillStyle = '#fff';
                        ctx.font = `bold ${Math.round(matchCanvas.width / 20)}px sans-serif`;
                        ctx.textAlign = 'center';
                        ctx.fillText('0 good matches', matchCanvas.width / 2, matchCanvas.height / 2);
                    }
                }

                // 4. DBSCAN clusters (colored per cluster, grey = noise)
                const clusterCanvas = document.getElementById('debug-clusters');
                bg(clusterCanvas);
                {
                    const ctx = clusterCanvas.getContext('2d');
                    if (debug.labels) {
                        for (let i = 0; i < debug.goodMatches.length; i++) {
                            const label = debug.labels[i];
                            ctx.fillStyle = label === -1
                                ? 'rgba(150,150,150,0.8)'
                                : CLUSTER_COLORS[label % CLUSTER_COLORS.length];
                            const p = debug.goodMatches[i].scenePt;
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r * 2, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    }
                }

                this.debugVisible = true;
            },

            // --- Canvas overlay ---
            _drawOverlays() {
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                for (const card of this.cards) {
                    this._drawQuad(card.qrCorners, '#00ff44', 2);
                    this._drawQuad(card.skuCorners, '#4488ff', 2);
                    if (card.text) {
                        overlayCtx.font = 'bold 16px monospace';
                        const m = overlayCtx.measureText(card.text);
                        const tx = card.center.x;
                        const ty = card.center.y - 30;
                        overlayCtx.fillStyle = 'rgba(0,0,0,0.7)';
                        overlayCtx.fillRect(tx - m.width / 2 - 5, ty - 17, m.width + 10, 22);
                        overlayCtx.fillStyle = '#fff';
                        overlayCtx.textAlign = 'center';
                        overlayCtx.fillText(card.text, tx, ty);
                    }
                }
            },

            _drawQuad(corners, color, lw) {
                overlayCtx.strokeStyle = color;
                overlayCtx.lineWidth = lw;
                overlayCtx.beginPath();
                overlayCtx.moveTo(corners[0].x, corners[0].y);
                for (let i = 1; i < corners.length; i++) overlayCtx.lineTo(corners[i].x, corners[i].y);
                overlayCtx.closePath();
                overlayCtx.stroke();
            },

            // --- Export ---
            exportCSV() {
                const rows = [['Card', 'SKU', 'Inliers']];
                this.cards.forEach((c, i) => rows.push([`Card ${i + 1}`, c.text ?? '', c.inlierCount]));
                const csv = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'sku-results.csv';
                a.click();
            },
        };
    });
});

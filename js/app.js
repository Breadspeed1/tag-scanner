import { CARD_LAYOUT, MAX_SCENE_DIM, MAX_DETECT_DIM, DETECTION_PARAMS } from './config.js';
import { initDetector, detect, preprocessBase, preprocessLocal } from './detector.js';

// OpenCV.js (Emscripten) throws C++ exceptions as raw WASM heap pointers (numbers).
function cvErrMsg(err) {
    if (typeof err === 'number') {
        try { return cv.exceptionFromPtr(err).msg; } catch (_) { return `OpenCV ptr=${err}`; }
    }
    if (!err) return String(err);
    if (typeof err === 'string') return err;
    if (err instanceof Error) {
        const base = err.message;
        if (typeof err.cause === 'number') {
            try { return `${base}: ${cv.exceptionFromPtr(err.cause).msg}`; } catch (_) { }
        }
        return base;
    }
    if (typeof err === 'object') return err.msg ?? err.message ?? JSON.stringify(err);
    return String(err);
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
        // Non-reactive closure variables
        let ocrWorker = null;
        let sceneCanvas = null;
        let overlayCanvas = null;
        let sceneCtx = null;
        let overlayCtx = null;
        let previewTimer  = null;
        // Off-DOM canvas at near-original resolution, used only for detection.
        let detectCanvas = null;
        let detectCtx = null;
        let detectToDisplay = 1;  // scale factor: display_coords = detect_coords * detectToDisplay
        let previewCanvas = null;

        return {
            // --- Reactive state ---
            cvReady: false,
            sceneLoaded: false,
            detecting: false,
            status: '',
            statusLevel: 'info',
            logsOpen: true,
            logs: [],
            cards: [],  // { qrCorners, skuCorners, center, decoded, text, pending }
            params: { ...DETECTION_PARAMS },

            get canDetect() {
                return this.cvReady && this.sceneLoaded && !this.detecting;
            },

            // --- Logging ---
            _log(msg, level = 'info') {
                const time = new Date().toLocaleTimeString('en', { hour12: false });
                this.logs.unshift({ msg, level, time });
                if (this.logs.length > 200) this.logs.pop();
                console[level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'log'](`[${time}] ${msg}`);
            },

            _setStatus(msg, level = 'info') {
                this.status = msg;
                this.statusLevel = level;
                this._log(msg, level);
            },

            // --- Lifecycle ---
            async init() {
                sceneCanvas = document.getElementById('scene-canvas');
                overlayCanvas = document.getElementById('overlay-canvas');
                previewCanvas = document.getElementById('preprocessed-canvas');
                sceneCtx = sceneCanvas.getContext('2d');
                overlayCtx = overlayCanvas.getContext('2d');

                ocrWorker = new Worker('js/ocr-worker.js');
                ocrWorker.onmessage = (e) => this.handleOCRResult(e);
                ocrWorker.onerror = (e) => {
                    this._log(`OCR worker crashed: ${e.message}`, 'error');
                    this.cards = this.cards.map(c => c.pending ? { ...c, text: '', pending: false } : c);
                    this.detecting = false;
                    this._setStatus('OCR worker failed — see log', 'error');
                };

                this._log('Waiting for OpenCV.js WASM…');
                this._setStatus('Loading OpenCV.js…');
                await waitForOpenCV();

                if (typeof cv.QRCodeDetector === 'undefined') {
                    this._log('cv.QRCodeDetector is undefined — this build may not include objdetect', 'error');
                    this._setStatus('OpenCV loaded but QRCodeDetector unavailable — see log', 'error');
                    return;
                }

                initDetector();
                this.cvReady = true;
                this._log('OpenCV ready — QRCodeDetector initialized');
                this._setStatus('Ready. Load a scene image to begin.');
            },

            // --- File input ---
            async onSceneChange(e) {
                const file = e.target.files[0];
                if (!file) return;
                try {
                    const img = await loadImageFromFile(file);
                    const origW = img.naturalWidth;
                    const origH = img.naturalHeight;

                    // Display canvas — downscaled for rendering
                    const dispScale = Math.min(1, MAX_SCENE_DIM / Math.max(origW, origH));
                    const w = Math.round(origW * dispScale);
                    const h = Math.round(origH * dispScale);

                    sceneCanvas.width = w;
                    sceneCanvas.height = h;
                    overlayCanvas.width = w;
                    overlayCanvas.height = h;
                    sceneCtx.drawImage(img, 0, 0, w, h);
                    overlayCtx.clearRect(0, 0, w, h);

                    // Detection canvas — near-original resolution for accurate QR detection
                    const detScale = Math.min(1, MAX_DETECT_DIM / Math.max(origW, origH));
                    const dw = Math.round(origW * detScale);
                    const dh = Math.round(origH * detScale);
                    detectCanvas = document.createElement('canvas');
                    detectCanvas.width = dw;
                    detectCanvas.height = dh;
                    detectCtx = detectCanvas.getContext('2d');
                    detectCtx.drawImage(img, 0, 0, dw, dh);
                    detectToDisplay = w / dw;

                    this.cards = [];
                    this.sceneLoaded = true;
                    this._log(`Scene loaded: ${file.name} (${origW}x${origH}px, display: ${w}x${h}, detect: ${dw}x${dh})`);
                    this._drawOverlays();
                    clearTimeout(previewTimer);
                    previewTimer = setTimeout(() => this.updatePreview(), 250);
                } catch (err) {
                    this._log(`Failed to load scene image: ${err.message}`, 'error');
                }
            },

            // --- Param tuning ---
            onParamChange() {
                if (!this.cvReady || !this.sceneLoaded) return;
                this._drawOverlays();                   // instant — just canvas drawing
                clearTimeout(previewTimer);
                previewTimer = setTimeout(() => this.updatePreview(), 250);
            },

            updatePreview() {
                if (!this.cvReady || !this.sceneLoaded) return;
                try {
                    const DW = detectCanvas.width;
                    const DH = detectCanvas.height;

                    // Global preprocessing — identical to what detector.js does,
                    // so tuning sliders reflect exactly what detection sees.
                    const imageData = detectCtx.getImageData(0, 0, DW, DH);
                    const src       = cv.matFromImageData(imageData);
                    const baseGray  = preprocessBase(src, this.params);
                    src.delete();
                    const fullBin = preprocessLocal(baseGray, this.params);
                    baseGray.delete();

                    // Downscale the full-res binary output to display size for rendering.
                    const displayOut = new cv.Mat();
                    cv.resize(fullBin, displayOut, new cv.Size(sceneCanvas.width, sceneCanvas.height), 0, 0, cv.INTER_AREA);
                    fullBin.delete();

                    cv.imshow(previewCanvas, displayOut);
                    displayOut.delete();

                    overlayCanvas.width  = previewCanvas.width;
                    overlayCanvas.height = previewCanvas.height;
                } catch (e) {
                    this._log(`Preview error: ${e.message}`, 'warn');
                }
            },

            // --- Detection ---
            runDetection() {
                this.detecting = true;
                this.cards = [];
                this._setStatus('Running QR detection…');

                setTimeout(() => {
                    try {
                        this._doDetection();
                    } catch (err) {
                        this._log(`Detection error: ${cvErrMsg(err)}`, 'error');
                        this._setStatus('Detection error — see log', 'error');
                        this.detecting = false;
                    }
                }, 0);
            },

            _doDetection() {
                const imageData = detectCtx.getImageData(0, 0, detectCanvas.width, detectCanvas.height);
                const sceneMat = cv.matFromImageData(imageData);

                let result;
                try {
                    result = detect(sceneMat, this.params);
                } catch (err) {
                    sceneMat.delete();
                    throw new Error(`detect() failed: ${cvErrMsg(err)}`);
                }
                sceneMat.delete();

                // Scale corners from detection-canvas coords → display-canvas coords.
                const s = detectToDisplay;
                result.cards.forEach(card => {
                    card.qrCorners = card.qrCorners.map(p => ({ x: p.x * s, y: p.y * s }));
                    card.skuCorners = card.skuCorners.map(p => ({ x: p.x * s, y: p.y * s }));
                    card.center = { x: card.center.x * s, y: card.center.y * s };
                });

                const { cards: rawCards, tileCount } = result;
                this._log(`QR detection complete: ${rawCards.length} code(s) found across ${tileCount} tiles`);

                if (rawCards.length === 0) {
                    this._setStatus('No QR codes detected. Try a smaller tile size or higher contrast.', 'warn');
                    this._drawOverlays();
                    this.detecting = false;
                    return;
                }

                let ocrDispatched = 0;
                const newCards = rawCards.map((card, i) => {
                    const entry = {
                        qrCorners: card.qrCorners,
                        skuCorners: card.skuCorners,
                        center: card.center,
                        decoded: card.decoded,
                        text: null,
                        pending: !!card.skuCrop,
                    };

                    if (card.decoded) {
                        this._log(`Card ${i + 1} QR content: "${card.decoded}"`);
                    }

                    if (card.skuCrop) {
                        const cropData = new Uint8Array(card.skuCrop.data);
                        ocrWorker.postMessage({
                            cardId: i,
                            cropData,
                            cropWidth: CARD_LAYOUT.crop_width,
                            cropHeight: CARD_LAYOUT.crop_height,
                        });
                        ocrDispatched++;
                        card.skuCrop.delete();
                    } else {
                        this._log(`Card ${i + 1}: no SKU crop`, 'warn');
                    }

                    return entry;
                });

                this.cards = newCards;
                this._drawOverlays();

                if (ocrDispatched === 0) {
                    this._setStatus(`Found ${rawCards.length} QR code(s) — no SKU crops`, 'warn');
                    this.detecting = false;
                } else {
                    this._setStatus(`Found ${rawCards.length} QR code(s) — OCR running…`);
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
                    this._setStatus(`Done — ${this.cards.length} card(s) found`, 'success');
                    this.detecting = false;
                }
            },

            // --- Canvas overlay ---
            _drawOverlays() {
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

                // Draw tile grid
                if (detectCanvas) {
                    const { tileSize } = this.params;
                    const overlap = Math.round(tileSize * 0.15);
                    const step = tileSize - overlap;
                    const dw = detectCanvas.width;
                    const dh = detectCanvas.height;
                    const s = detectToDisplay;

                    overlayCtx.strokeStyle = 'rgba(255, 200, 0, 0.4)';
                    overlayCtx.lineWidth = 1;
                    overlayCtx.setLineDash([4, 4]);
                    for (let ty = 0; ty < dh; ty += step) {
                        for (let tx = 0; tx < dw; tx += step) {
                            const tw = Math.min(tileSize, dw - tx);
                            const th = Math.min(tileSize, dh - ty);
                            overlayCtx.strokeRect(tx * s, ty * s, tw * s, th * s);
                        }
                    }
                    overlayCtx.setLineDash([]);
                }

                // Draw detected card overlays
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
                const rows = [['Card', 'SKU', 'QR Content']];
                this.cards.forEach((c, i) => rows.push([`Card ${i + 1}`, c.text ?? '', c.decoded ?? '']));
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

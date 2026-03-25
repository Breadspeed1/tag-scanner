import { CARD_LAYOUT } from './config.js';
import { initReference, detect } from './detector.js';

async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}

// App state
let ocrWorker = null;
let refImageEl = null;
let sceneImageEl = null;
let cvReady = false;
let refLoaded = false;
let sceneLoaded = false;

// Cards detected in the last run: [{qrCorners, skuCorners, center}]
let lastCards = [];
// OCR results keyed by card index: Map<index, {text, pending}>
let ocrResults = new Map();

// Canvas elements
let sceneCanvas, sceneCtx, overlayCanvas, overlayCtx;

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function updateDetectButton() {
    document.getElementById('detect-btn').disabled = !(refLoaded && sceneLoaded && cvReady);
}

function loadImageFromFile(file) {
    return new Promise((resolve) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = url;
    });
}

async function onRefChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    refImageEl = await loadImageFromFile(file);
    const preview = document.getElementById('ref-preview');
    preview.src = refImageEl.src;
    preview.hidden = false;
    refLoaded = true;
    updateDetectButton();
}

async function onSceneChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    sceneImageEl = await loadImageFromFile(file);

    sceneCanvas.width = sceneImageEl.naturalWidth;
    sceneCanvas.height = sceneImageEl.naturalHeight;
    overlayCanvas.width = sceneImageEl.naturalWidth;
    overlayCanvas.height = sceneImageEl.naturalHeight;
    sceneCtx.drawImage(sceneImageEl, 0, 0);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    document.getElementById('canvas-container').hidden = false;
    sceneLoaded = true;
    updateDetectButton();
}

function runDetection() {
    setStatus('Running detection…');
    document.getElementById('detect-btn').disabled = true;

    initReference(refImageEl);

    const imageData = sceneCtx.getImageData(0, 0, sceneCanvas.width, sceneCanvas.height);
    const sceneMat = cv.matFromImageData(imageData);
    const cards = detect(sceneMat);
    sceneMat.delete();

    lastCards = cards.map(c => ({ qrCorners: c.qrCorners, skuCorners: c.skuCorners, center: c.center }));
    ocrResults = new Map();

    for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        if (card.skuCrop) {
            ocrResults.set(i, { text: null, pending: true });
            const cropData = new Uint8Array(card.skuCrop.data);
            ocrWorker.postMessage({
                cardId: i,
                cropData,
                cropWidth: CARD_LAYOUT.crop_width,
                cropHeight: CARD_LAYOUT.crop_height,
            });
        }

        card.homography.delete();
        if (card.skuCrop) card.skuCrop.delete();
    }

    drawOverlays();
    updateTable();

    const count = lastCards.length;
    if (count === 0) {
        setStatus('No cards detected. Try adjusting the reference image or scene.');
        document.getElementById('detect-btn').disabled = false;
    } else {
        setStatus(`Found ${count} card${count !== 1 ? 's' : ''}. Running OCR…`);
    }
}

function drawOverlays() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    for (let i = 0; i < lastCards.length; i++) {
        const card = lastCards[i];
        drawQuad(overlayCtx, card.qrCorners, '#00ff44', 2);
        drawQuad(overlayCtx, card.skuCorners, '#4488ff', 2);

        const result = ocrResults.get(i);
        if (result?.text) {
            overlayCtx.font = 'bold 16px monospace';
            const metrics = overlayCtx.measureText(result.text);
            const tx = card.center.x;
            const ty = card.center.y - 30;
            overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.65)';
            overlayCtx.fillRect(tx - metrics.width / 2 - 5, ty - 17, metrics.width + 10, 22);
            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.textAlign = 'center';
            overlayCtx.fillText(result.text, tx, ty);
        }
    }
}

function drawQuad(ctx, corners, color, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.stroke();
}

function handleOCRResult(event) {
    const { cardId, text } = event.data;
    ocrResults.set(cardId, { text, pending: false });
    drawOverlays();
    updateTable();

    const pending = [...ocrResults.values()].filter(r => r.pending).length;
    if (pending === 0) {
        const count = lastCards.length;
        setStatus(`Done. ${count} card${count !== 1 ? 's' : ''} found.`);
        document.getElementById('detect-btn').disabled = false;
    }
}

function updateTable() {
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';
    for (let i = 0; i < lastCards.length; i++) {
        const result = ocrResults.get(i);
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>Card ${i + 1}</td>
            <td class="sku-cell">${
                result?.text
                    ? `<span class="sku-text">${result.text}</span>`
                    : result?.pending
                        ? '<span class="sku-pending">Reading…</span>'
                        : '<span class="sku-empty">—</span>'
            }</td>
        `;
        tbody.appendChild(tr);
    }
    document.getElementById('results').hidden = lastCards.length === 0;
}

function exportCSV() {
    const rows = [['Card', 'SKU']];
    for (let i = 0; i < lastCards.length; i++) {
        rows.push([`Card ${i + 1}`, ocrResults.get(i)?.text ?? '']);
    }
    const csv = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sku-results.csv';
    a.click();
}

async function init() {
    sceneCanvas = document.getElementById('scene-canvas');
    sceneCtx = sceneCanvas.getContext('2d');
    overlayCanvas = document.getElementById('overlay-canvas');
    overlayCtx = overlayCanvas.getContext('2d');

    ocrWorker = new Worker('js/ocr-worker.js');
    ocrWorker.onmessage = handleOCRResult;

    document.getElementById('ref-input').addEventListener('change', onRefChange);
    document.getElementById('scene-input').addEventListener('change', onSceneChange);
    document.getElementById('detect-btn').addEventListener('click', runDetection);
    document.getElementById('export-btn').addEventListener('click', exportCSV);

    setStatus('Loading OpenCV.js…');
    await waitForOpenCV();
    cvReady = true;
    setStatus('Ready. Load a reference QR and a scene image to begin.');
    updateDetectButton();
}

init();

# QR Fiducial Detection & SKU Reader — Technical Specification

## OpenCV.js + Tesseract.js Web Application

---

## 1. System Overview

A static web application that detects identical QR codes on product cards
(potentially behind plastic bags), computes each card's pose, extracts a
rectified crop of the SKU text region, and reads it via OCR.

Everything runs client-side. No server required. Deployable to GitHub Pages.

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser                                                         │
│                                                                  │
│  getUserMedia → <video> → <canvas> → ImageData                   │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────┐                        │
│  │  Detection Pipeline (OpenCV.js)      │                        │
│  │                                      │                        │
│  │  1. AKAZE detect+compute (ref, once) │                        │
│  │  2. AKAZE detect+compute (scene)     │                        │
│  │  3. BFMatcher knnMatch               │                        │
│  │  4. Ratio test filter                │                        │
│  │  5. Cluster matches (DBSCAN)         │                        │
│  │  6. Per cluster: findHomography      │                        │
│  │  7. Per card: warpPerspective crop   │                        │
│  └──────────┬───────────────────────────┘                        │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Overlay Canvas      │  │  OCR Worker (Tesseract.js)       │  │
│  │  - QR bounding quads │  │  - Receives crop ImageData       │  │
│  │  - SKU region quads  │  │  - Returns SKU text              │  │
│  │  - SKU text labels   │  │  - One-shot per new card         │  │
│  └──────────────────────┘  └──────────────────────────────────┘  │
│                                                                  │
│  Card Tracker                                                    │
│  - Deduplicates across frames by position                        │
│  - Caches OCR results                                            │
│  - Prevents redundant OCR calls                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. File Structure

```
/
├── index.html
├── style.css
├── js/
│   ├── app.js              # entry point, camera setup, frame loop
│   ├── detector.js         # OpenCV.js detection pipeline
│   ├── clustering.js       # DBSCAN (only custom algorithm)
│   ├── tracker.js          # cross-frame card deduplication
│   ├── ocr-worker.js       # Web Worker running Tesseract.js
│   └── config.js           # card layout + detection parameters
└── assets/
    └── ref_qr.png          # reference QR code image
```

---

## 3. Configuration

`config.js` — all tunable parameters in one place.

```javascript
export const CARD_LAYOUT = {
    // SKU bounding box in QR-side-length units.
    // QR code occupies [0,0] → [1,1].
    sku_x: 0.0,
    sku_y: 1.1,
    sku_w: 1.0,
    sku_h: 0.4,

    // Output crop resolution (pixels).
    crop_width: 300,
    crop_height: 100,
};

export const DETECTION = {
    // AKAZE
    akaze_threshold: 0.001,         // detector threshold (lower = more features)

    // Matching
    lowe_ratio: 0.75,

    // DBSCAN
    dbscan_eps: 80,                 // px — tune relative to QR size in scene
    dbscan_min_samples: 8,

    // RANSAC (passed to cv.findHomography)
    ransac_reproj_thresh: 5.0,
    min_inliers: 10,
};

export const TRACKER = {
    merge_radius: 60,               // px — max distance to consider same card
    ema_alpha: 0.2,                 // smoothing factor for center position
};

export const CAMERA = {
    width: 1280,
    height: 720,
    facing_mode: 'environment',
};

// Tesseract character whitelist for SKU text
export const OCR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-';
```

---

## 4. Module Specifications

### 4.1 `app.js` — Entry Point

Responsibilities: load OpenCV.js, set up camera, manage the frame loop,
coordinate between detector, tracker, and OCR worker.

```javascript
// Lifecycle:
//
// 1. Poll for OpenCV.js WASM initialization (loaded via CDN script tag)
// 2. Load reference image onto a <canvas>, convert to cv.Mat
// 3. Call detector.initReference(refMat)
// 4. Open camera via getUserMedia
// 5. Start frame loop via requestAnimationFrame
// 6. Initialize Tesseract Web Worker

let refFeatures = null;   // cached reference AKAZE features
let tracker = null;       // CardTracker instance
let ocrWorker = null;     // Web Worker handle

async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}

async function start() {
    await waitForOpenCV();
    refFeatures = initReference(refImageElement);
    tracker = new CardTracker(TRACKER);
    ocrWorker = new Worker('js/ocr-worker.js');
    ocrWorker.onmessage = handleOCRResult;
    await startCamera();
    requestAnimationFrame(processFrame);
}

function processFrame() {
    // 1. Capture frame from video to canvas
    ctx.drawImage(video, 0, 0);
    const imageData = ctx.getImageData(0, 0, width, height);
    const sceneMat = cv.matFromImageData(imageData);

    // 2. Run detection
    const cards = detect(sceneMat, refFeatures);

    // 3. Update tracker, dispatch OCR for new cards
    for (const card of cards) {
        const { id, isNew } = tracker.update(card);
        if (isNew && card.skuCrop) {
            ocrWorker.postMessage({
                cardId: id,
                cropData: card.skuCrop.data,
                cropWidth: CARD_LAYOUT.crop_width,
                cropHeight: CARD_LAYOUT.crop_height,
            });
        }
    }

    // 4. Draw overlays
    drawOverlays(cards, tracker);

    // 5. Cleanup OpenCV Mats (prevent memory leaks)
    sceneMat.delete();
    for (const card of cards) {
        if (card.skuCrop) card.skuCrop.delete();
    }

    requestAnimationFrame(processFrame);
}

function handleOCRResult(event) {
    const { cardId, text } = event.data;
    tracker.setText(cardId, text);
}
```

**Critical: OpenCV.js memory management.** Every `cv.Mat` must be explicitly
`.delete()`d. OpenCV.js allocates on the WASM heap; the JS garbage collector
does not know about these allocations. Failing to delete Mats will leak
memory and crash the tab within minutes during a camera loop.

---

### 4.2 `detector.js` — Detection Pipeline

Two exported functions: `initReference` (called once) and `detect`
(called per frame).

```javascript
import { DETECTION, CARD_LAYOUT } from './config.js';
import { dbscan } from './clustering.js';

// Persistent OpenCV objects (created once, reused per frame).
let akaze = null;
let matcher = null;

// Reference data (computed once).
let refDescriptors = null;
let refKeypoints = null;
let refWidth = 0;
let refHeight = 0;

/**
 * Initialize AKAZE detector and extract reference features.
 * Called once at startup.
 *
 * @param {HTMLImageElement|HTMLCanvasElement} refImage
 * @returns {void}
 */
export function initReference(refImage) {
    // Create AKAZE detector
    akaze = new cv.AKAZE();
    akaze.setThreshold(DETECTION.akaze_threshold);

    // Create brute-force matcher for binary descriptors
    matcher = new cv.BFMatcher(cv.NORM_HAMMING);

    // Read reference image → grayscale
    const refMat = cv.imread(refImage);
    const refGray = new cv.Mat();
    cv.cvtColor(refMat, refGray, cv.COLOR_RGBA2GRAY);

    refWidth = refGray.cols;
    refHeight = refGray.rows;

    // Extract features
    refKeypoints = new cv.KeyPointVector();
    refDescriptors = new cv.Mat();
    akaze.detectAndCompute(refGray, new cv.Mat(), refKeypoints, refDescriptors);

    // Cleanup
    refMat.delete();
    refGray.delete();
}

/**
 * Run the full detection pipeline on a scene frame.
 *
 * @param {cv.Mat} sceneMat - RGBA scene image (from matFromImageData)
 * @returns {Array<DetectedCard>}
 *
 * DetectedCard shape:
 * {
 *   homography: cv.Mat (3x3 float64),
 *   inlierCount: number,
 *   center: { x: number, y: number },
 *   qrCorners: [{x,y}, {x,y}, {x,y}, {x,y}],
 *   skuCorners: [{x,y}, {x,y}, {x,y}, {x,y}],
 *   skuCrop: cv.Mat (grayscale u8, crop_width × crop_height),
 * }
 */
export function detect(sceneMat) {
    // --- Stage 1: Feature extraction ---
    const sceneGray = new cv.Mat();
    cv.cvtColor(sceneMat, sceneGray, cv.COLOR_RGBA2GRAY);

    const sceneKeypoints = new cv.KeyPointVector();
    const sceneDescriptors = new cv.Mat();
    akaze.detectAndCompute(sceneGray, new cv.Mat(), sceneKeypoints, sceneDescriptors);

    if (sceneDescriptors.rows < DETECTION.min_inliers) {
        sceneGray.delete();
        sceneDescriptors.delete();
        return [];
    }

    // --- Stage 2: Descriptor matching + ratio test ---
    const rawMatches = new cv.DMatchVectorVector();
    matcher.knnMatch(refDescriptors, sceneDescriptors, rawMatches, 2);

    const goodMatches = [];
    for (let i = 0; i < rawMatches.size(); i++) {
        const pair = rawMatches.get(i);
        if (pair.size() < 2) continue;
        const m = pair.get(0);
        const n = pair.get(1);
        if (m.distance < DETECTION.lowe_ratio * n.distance) {
            goodMatches.push({
                refIdx: m.queryIdx,
                sceneIdx: m.trainIdx,
                refPt: keypointToPoint(refKeypoints, m.queryIdx),
                scenePt: keypointToPoint(sceneKeypoints, m.trainIdx),
            });
        }
    }

    rawMatches.delete();
    sceneDescriptors.delete();

    if (goodMatches.length < DETECTION.min_inliers) {
        sceneGray.delete();
        sceneKeypoints.delete();
        return [];
    }

    // --- Stage 3: Spatial clustering ---
    const scenePoints = goodMatches.map(m => m.scenePt);
    const labels = dbscan(scenePoints, DETECTION.dbscan_eps, DETECTION.dbscan_min_samples);

    // --- Stage 4: Per-cluster homography + crop ---
    const clusterIds = [...new Set(labels.filter(l => l >= 0))].sort((a, b) => a - b);
    const cards = [];

    for (const clusterId of clusterIds) {
        const clusterMatches = goodMatches.filter((_, i) => labels[i] === clusterId);
        if (clusterMatches.length < 4) continue;

        // Build point arrays for findHomography
        const srcPts = cv.matFromArray(clusterMatches.length, 1, cv.CV_32FC2,
            clusterMatches.flatMap(m => [m.refPt.x, m.refPt.y]));
        const dstPts = cv.matFromArray(clusterMatches.length, 1, cv.CV_32FC2,
            clusterMatches.flatMap(m => [m.scenePt.x, m.scenePt.y]));

        // RANSAC homography
        const mask = new cv.Mat();
        const H = cv.findHomography(srcPts, dstPts, cv.RANSAC,
            DETECTION.ransac_reproj_thresh, mask);

        srcPts.delete();
        dstPts.delete();

        if (H.empty()) {
            H.delete();
            mask.delete();
            continue;
        }

        // Count inliers
        let inlierCount = 0;
        for (let i = 0; i < mask.rows; i++) {
            if (mask.data[i] !== 0) inlierCount++;
        }
        mask.delete();

        if (inlierCount < DETECTION.min_inliers) {
            H.delete();
            continue;
        }

        // Transform QR corners to scene space
        const qrCornersRef = [
            [0, 0], [refWidth, 0],
            [refWidth, refHeight], [0, refHeight],
        ];
        const qrCorners = transformPoints(qrCornersRef, H);

        // Transform SKU corners to scene space
        const sx = CARD_LAYOUT.sku_x * refWidth;
        const sy = CARD_LAYOUT.sku_y * refHeight;
        const sw = CARD_LAYOUT.sku_w * refWidth;
        const sh = CARD_LAYOUT.sku_h * refHeight;
        const skuCornersRef = [
            [sx, sy], [sx + sw, sy],
            [sx + sw, sy + sh], [sx, sy + sh],
        ];
        const skuCorners = transformPoints(skuCornersRef, H);

        // Center = centroid of QR corners
        const center = {
            x: qrCorners.reduce((s, p) => s + p.x, 0) / 4,
            y: qrCorners.reduce((s, p) => s + p.y, 0) / 4,
        };

        // Extract rectified SKU crop
        const skuCrop = extractCrop(sceneGray, skuCorners);

        cards.push({
            homography: H,  // caller must .delete() after use
            inlierCount,
            center,
            qrCorners,
            skuCorners,
            skuCrop,         // caller must .delete() after use
        });
    }

    sceneGray.delete();
    sceneKeypoints.delete();

    return cards;
}


// --- Internal helpers ---

function keypointToPoint(keypoints, idx) {
    const kp = keypoints.get(idx);
    return { x: kp.pt.x, y: kp.pt.y };
}

/**
 * Transform an array of [x, y] points through a 3x3 homography.
 * Returns array of {x, y} objects.
 */
function transformPoints(pts, H) {
    const src = cv.matFromArray(pts.length, 1, cv.CV_32FC2,
        pts.flatMap(([x, y]) => [x, y]));
    const dst = new cv.Mat();
    cv.perspectiveTransform(src, dst, H);

    const result = [];
    for (let i = 0; i < pts.length; i++) {
        result.push({
            x: dst.floatAt(i, 0),
            y: dst.floatAt(i, 1),
        });
    }
    src.delete();
    dst.delete();
    return result;
}

/**
 * Extract a rectified crop of the SKU region.
 *
 * skuCorners: [{x,y} × 4] — the SKU quad in scene space.
 * Returns: cv.Mat (grayscale, crop_width × crop_height). Caller must delete.
 */
function extractCrop(sceneGray, skuCorners) {
    const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2,
        skuCorners.flatMap(p => [p.x, p.y]));
    const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        CARD_LAYOUT.crop_width, 0,
        CARD_LAYOUT.crop_width, CARD_LAYOUT.crop_height,
        0, CARD_LAYOUT.crop_height,
    ]);

    const M = cv.getPerspectiveTransform(srcQuad, dstQuad);
    const crop = new cv.Mat();
    cv.warpPerspective(sceneGray, crop,
        M, new cv.Size(CARD_LAYOUT.crop_width, CARD_LAYOUT.crop_height));

    // Otsu threshold for clean OCR input
    const binary = new cv.Mat();
    cv.threshold(crop, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    srcQuad.delete();
    dstQuad.delete();
    M.delete();
    crop.delete();

    return binary;  // caller owns this Mat
}
```

---

### 4.3 `clustering.js` — DBSCAN

The only algorithm implemented from scratch. OpenCV.js does not expose
a clustering API.

```javascript
/**
 * DBSCAN clustering on 2D points.
 *
 * @param {Array<{x: number, y: number}>} points
 * @param {number} eps - neighborhood radius in pixels
 * @param {number} minSamples - minimum cluster size
 * @returns {Int32Array} label per point (-1 = noise)
 */
export function dbscan(points, eps, minSamples) {
    const n = points.length;
    const labels = new Int32Array(n).fill(-1);
    const visited = new Uint8Array(n);
    const epsSq = eps * eps;
    let clusterId = 0;

    function regionQuery(idx) {
        const neighbors = [];
        const p = points[idx];
        for (let j = 0; j < n; j++) {
            const dx = p.x - points[j].x;
            const dy = p.y - points[j].y;
            if (dx * dx + dy * dy <= epsSq) {
                neighbors.push(j);
            }
        }
        return neighbors;
    }

    for (let i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = 1;

        const neighbors = regionQuery(i);
        if (neighbors.length < minSamples) continue;

        labels[i] = clusterId;
        const queue = [...neighbors];
        let qi = 0;

        while (qi < queue.length) {
            const q = queue[qi++];
            if (!visited[q]) {
                visited[q] = 1;
                const qNeighbors = regionQuery(q);
                if (qNeighbors.length >= minSamples) {
                    for (const nn of qNeighbors) {
                        if (!queue.includes(nn)) queue.push(nn);
                    }
                }
            }
            if (labels[q] === -1) {
                labels[q] = clusterId;
            }
        }
        clusterId++;
    }

    return labels;
}
```

**Performance note:** `queue.includes(nn)` is O(n) per call, making worst
case O(n³). For hundreds of matches this is fine. If it becomes a problem,
replace with a Set:

```javascript
const inQueue = new Uint8Array(n);
// ...
if (!inQueue[nn]) { queue.push(nn); inQueue[nn] = 1; }
```

---

### 4.4 `tracker.js` — Cross-Frame Card Deduplication

Prevents OCR from running on every frame for every card. Matches detected
cards to previously-seen cards by spatial proximity.

```javascript
import { TRACKER } from './config.js';

export class CardTracker {
    constructor(config = TRACKER) {
        this.mergeRadiusSq = config.merge_radius ** 2;
        this.alpha = config.ema_alpha;
        this.cards = new Map();  // id → TrackedCard
        this.nextId = 0;
    }

    /**
     * Update tracker with a newly detected card.
     * Returns { id, isNew } where isNew means this card hasn't been seen before.
     */
    update(detectedCard) {
        const { center } = detectedCard;

        // Find closest known card
        let bestId = null;
        let bestDistSq = Infinity;

        for (const [id, tracked] of this.cards) {
            const dx = center.x - tracked.center.x;
            const dy = center.y - tracked.center.y;
            const distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestId = id;
            }
        }

        // Match to existing card if close enough
        if (bestId !== null && bestDistSq < this.mergeRadiusSq) {
            const tracked = this.cards.get(bestId);
            // EMA smoothing on center position
            tracked.center.x = tracked.center.x * (1 - this.alpha)
                              + center.x * this.alpha;
            tracked.center.y = tracked.center.y * (1 - this.alpha)
                              + center.y * this.alpha;
            tracked.framesSeen++;
            tracked.lastSeen = performance.now();
            return { id: bestId, isNew: false };
        }

        // Register new card
        const id = this.nextId++;
        this.cards.set(id, {
            center: { ...center },
            skuText: null,
            ocrPending: false,
            framesSeen: 1,
            lastSeen: performance.now(),
        });
        return { id, isNew: true };
    }

    setText(id, text) {
        const card = this.cards.get(id);
        if (card) {
            card.skuText = text;
            card.ocrPending = false;
        }
    }

    getText(id) {
        return this.cards.get(id)?.skuText ?? null;
    }

    setOcrPending(id) {
        const card = this.cards.get(id);
        if (card) card.ocrPending = true;
    }

    isOcrNeeded(id) {
        const card = this.cards.get(id);
        return card && !card.skuText && !card.ocrPending;
    }

    /**
     * Remove cards not seen for a given duration.
     * Call periodically to prevent unbounded growth.
     */
    prune(maxAgeMs = 5000) {
        const now = performance.now();
        for (const [id, card] of this.cards) {
            if (now - card.lastSeen > maxAgeMs) {
                this.cards.delete(id);
            }
        }
    }

    /**
     * Get all tracked cards with their current state.
     * Useful for rendering the results table.
     */
    getAll() {
        return [...this.cards.entries()].map(([id, card]) => ({
            id,
            center: card.center,
            skuText: card.skuText,
            framesSeen: card.framesSeen,
        }));
    }
}
```

---

### 4.5 `ocr-worker.js` — Web Worker for Tesseract.js

Runs OCR off the main thread so it doesn't block rendering.

```javascript
importScripts('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');

let worker = null;

async function initWorker() {
    worker = await Tesseract.createWorker('eng');
    await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        tessedit_pageseg_mode: '7',  // single line
    });
}

const ready = initWorker();

self.onmessage = async (event) => {
    await ready;

    const { cardId, cropData, cropWidth, cropHeight } = event.data;

    // cropData is a Uint8Array of grayscale pixels (from the thresholded Mat).
    // Tesseract expects an image — convert to ImageData or a data URL.
    // Since we have raw grayscale, expand to RGBA for Tesseract:
    const rgba = new Uint8ClampedArray(cropWidth * cropHeight * 4);
    for (let i = 0; i < cropData.length; i++) {
        const v = cropData[i];
        rgba[i * 4]     = v;
        rgba[i * 4 + 1] = v;
        rgba[i * 4 + 2] = v;
        rgba[i * 4 + 3] = 255;
    }

    // Create ImageData and convert to a blob URL for Tesseract
    const canvas = new OffscreenCanvas(cropWidth, cropHeight);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(rgba, cropWidth, cropHeight), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });

    const { data: { text } } = await worker.recognize(blob);

    self.postMessage({
        cardId,
        text: text.trim(),
    });
};
```

---

### 4.6 Overlay Rendering

In `app.js` or a separate `renderer.js`. Draws on a `<canvas>` positioned
over the video feed.

```javascript
function drawOverlays(cards, tracker) {
    overlay.clearRect(0, 0, width, height);

    for (const card of cards) {
        // QR bounding quad (green)
        drawQuad(overlay, card.qrCorners, '#00ff00', 2);

        // SKU region quad (blue)
        drawQuad(overlay, card.skuCorners, '#4488ff', 2);

        // SKU text label (if OCR has completed)
        const { id } = tracker.update(card);  // re-lookup
        const text = tracker.getText(id);
        if (text) {
            const { x, y } = card.center;
            overlay.fillStyle = '#ffffff';
            overlay.font = '16px monospace';
            overlay.fillText(text, x - 30, y - 30);
        }
    }
}

function drawQuad(ctx, corners, color, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) {
        ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();
    ctx.stroke();
}
```

---

## 5. OpenCV.js Memory Management Rules

This is the most likely source of bugs. OpenCV.js uses Emscripten's
WASM heap. JS garbage collection does NOT apply to cv.Mat objects.

**Rule 1:** Every `new cv.Mat()`, `cv.matFromArray()`, `cv.matFromImageData()`,
or OpenCV function that returns a Mat must have a corresponding `.delete()`.

**Rule 2:** OpenCV functions that take an output Mat parameter (like
`cv.cvtColor(src, dst, ...)`) allocate internally if dst is empty.
You still own dst and must delete it.

**Rule 3:** `cv.KeyPointVector` and `cv.DMatchVectorVector` also need
`.delete()`.

**Rule 4:** In the frame loop, delete ALL Mats created during that frame
before the next `requestAnimationFrame`. A 720p RGBA Mat is ~3.7MB.
At 30fps, leaking one per frame exhausts the WASM heap in seconds.

**Practical pattern:**

```javascript
function processFrame() {
    const mats = [];  // track all Mats created this frame

    const scene = cv.matFromImageData(imageData);
    mats.push(scene);

    const gray = new cv.Mat();
    mats.push(gray);
    cv.cvtColor(scene, gray, cv.COLOR_RGBA2GRAY);

    // ... pipeline ...

    // Cleanup everything
    for (const m of mats) m.delete();

    requestAnimationFrame(processFrame);
}
```

Or use a helper:

```javascript
class MatPool {
    constructor() { this.mats = []; }
    track(mat) { this.mats.push(mat); return mat; }
    flush() { for (const m of this.mats) m.delete(); this.mats = []; }
}
```

---

## 6. OpenCV.js Loading

The `@techstark/opencv-js` package on jsDelivr is the recommended CDN
source. It's actively maintained (OpenCV 4.13.0 as of January 2026),
built with `SINGLE_FILE=1` so the WASM binary is embedded in the JS —
no separate `.wasm` file to serve. The tradeoff is ~9MB file size.

**Do NOT use `docs.opencv.org/4.x/opencv.js`** — it returns 403 Forbidden.

**CDN script tag:**
```html
<script async
    src="https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js">
</script>
```

**Wait for ready (polling):**

OpenCV.js loading involves both script download and WASM compilation.
The simplest reliable approach is polling for `cv.Mat` to exist. This
avoids race conditions between `onRuntimeInitialized` callbacks and
ES module execution order.

```javascript
async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}
```

50ms polling is imperceptible to the user. The loop exits as soon as
the WASM runtime finishes compiling, typically 1-3 seconds on modern
hardware.

**Verifying the build includes AKAZE:**

After loading, check `typeof cv.AKAZE` in the browser console. If it's
`"undefined"`, the build doesn't include the `features2d` module and
you'll need to fall back to `cv.ORB` (always included) or use a custom
build. The `@techstark` builds include `features2d`.

**Custom build (optional, for smaller bundle):**

```bash
python platforms/js/build_js.py build_js \
    --build_wasm \
    --cmake_option="-DBUILD_LIST=features2d,calib3d,imgproc"
```

Strips video I/O, objdetect, photo, etc. and cuts the bundle to ~2-3MB.
Self-host the resulting `opencv.js` and reference it directly.

---

## 7. HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QR SKU Reader</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="camera-container">
        <video id="video" autoplay playsinline></video>
        <canvas id="overlay"></canvas>
    </div>

    <div id="controls">
        <label>
            Reference QR:
            <input type="file" id="ref-input" accept="image/*">
        </label>
        <button id="start-btn" disabled>Start Detection</button>
    </div>

    <div id="results">
        <h2>Detected Cards</h2>
        <table id="results-table">
            <thead>
                <tr>
                    <th>Card #</th>
                    <th>SKU</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
        <button id="export-btn">Export CSV</button>
    </div>

    <!-- OpenCV.js (CDN, WASM embedded via SINGLE_FILE) -->
    <script async
        src="https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js">
    </script>

    <!-- App (ES modules) -->
    <script type="module" src="js/app.js"></script>
</body>
</html>
```

**CSS note:** `#camera-container` uses `position: relative`, video and
overlay canvas are both `position: absolute` with identical dimensions.
The overlay canvas sits on top of the video with `pointer-events: none`.

---

## 8. Performance Budget

| Stage               | Expected Time (720p) | Notes                                     |
| ------------------- | -------------------- | ----------------------------------------- |
| AKAZE extraction    | 50–100ms             | Biggest cost. Reduce input res if needed. |
| knnMatch            | 10–20ms              | Hamming, hardware-accelerated in WASM.    |
| Ratio test + DBSCAN | <5ms                 | Trivial at match counts <1000.            |
| findHomography ×N   | 2–5ms per card       | RANSAC is fast on small point sets.       |
| warpPerspective ×N  | <1ms per card        | Small output crop.                        |
| **Total**           | **70–140ms**         | **~7–14 fps**                             |

OCR (Tesseract.js) takes 100–500ms per crop but runs asynchronously in
a Worker and only fires once per new card. It does not affect frame rate.

If the frame budget is too tight, process every Nth frame and interpolate
bounding box positions between detections using the tracker's EMA.

---

## 9. Deployment

Static files only. Any static hosting works.

```bash
# GitHub Pages (from your existing fc.aidenvoth.com setup)
git add .
git commit -m "add SKU reader"
git push

# Or any static server for local dev
npx serve .
# or
python3 -m http.server 8080
```

No build step required unless you opt for a JS bundler. The app is
vanilla JS with ES modules. OpenCV.js is loaded as a script tag.

---

## 10. Test Plan

| Test                                             | Method                                       |
| ------------------------------------------------ | -------------------------------------------- |
| DBSCAN produces correct clusters                 | Unit test with known point layouts           |
| DBSCAN handles noise correctly                   | Points farther than eps → label -1           |
| Tracker deduplicates same card across frames     | Feed same center 10× → one id                |
| Tracker registers new card at different position | Feed two centers → two ids                   |
| Tracker prunes stale cards                       | Feed one center, wait, prune → empty         |
| Detection finds QR in clean image                | Load test image, verify card count           |
| Detection handles no-QR scene                    | Load image without QR → empty results        |
| Homography is reasonable                         | Check reprojected corners form a convex quad |
| Crop extraction produces readable output         | Save crop, manually verify                   |
| OCR reads known SKU correctly                    | Feed crop of known SKU → match text          |
| Memory doesn't leak during frame loop            | Monitor WASM heap over 60s                   |

**Synthetic integration test:** Programmatically create a scene image by
applying known affine transforms to the reference QR image (paste 3 copies
at known positions/rotations onto a white background). Run detection.
Verify correct card count and that recovered QR corners match the known
transforms within 5px tolerance.
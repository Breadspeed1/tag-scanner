# QR Fiducial Detection & SKU Reader — Technical Specification

## OpenCV.js + Tesseract.js Web Application

---

## 1. System Overview

A static web application that detects identical QR codes on product cards,
computes each card's pose from the QR corner points, extracts a rectified
crop of the SKU text region, and reads it via OCR.

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
│  │  1. QRCodeDetector.detectMulti       │                        │
│  │     → 4 corners per QR code          │                        │
│  │  2. getPerspectiveTransform          │                        │
│  │     → homography per card            │                        │
│  │  3. warpPerspective                  │                        │
│  │     → rectified SKU crop             │                        │
│  └──────────┬───────────────────────────┘                        │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Overlay Canvas      │  │  OCR Worker (Tesseract.js)       │  │
│  │  - QR bounding quads │  │  - Receives crop ImageData       │  │
│  │  - SKU region quads  │  │  - Returns SKU text              │  │
│  │  - SKU text labels   │  │  - Runs per detected card        │  │
│  └──────────────────────┘  └──────────────────────────────────┘  │
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
│   ├── ocr-worker.js       # Web Worker running Tesseract.js
│   └── config.js           # card layout + detection parameters
└── assets/
    └── ref_qr.png          # reference QR code image (for layout calibration only)
```

No feature matching, clustering, or tracking modules. The QR detector
gives us corners directly.

---

## 3. Configuration

`config.js` — all tunable parameters in one place.

```javascript
export const CARD_LAYOUT = {
    // SKU bounding box relative to the QR code's detected corners.
    // The QR code occupies [0,0] → [1,1] where the unit is the QR
    // side length. Corners from detectMulti are in order:
    //   [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left
    //
    // Measure from a physical card: how far is the SKU text from the
    // QR code, expressed as multiples of the QR side length?
    sku_x: 0.0,        // left edge relative to QR left
    sku_y: 1.1,        // top edge relative to QR top (1.1 = just below)
    sku_w: 1.0,        // width in QR-side-lengths
    sku_h: 0.4,        // height in QR-side-lengths

    // Output crop resolution in pixels.
    crop_width: 300,
    crop_height: 100,
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
coordinate between detector and OCR worker.

```javascript
import { initDetector, detect } from './detector.js';
import { CARD_LAYOUT, CAMERA } from './config.js';

let ocrWorker = null;
let detecting = false;

async function waitForOpenCV() {
    while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
    }
}

async function start() {
    await waitForOpenCV();
    initDetector();
    ocrWorker = new Worker('js/ocr-worker.js');
    ocrWorker.onmessage = handleOCRResult;
    await startCamera();
    requestAnimationFrame(processFrame);
}

function processFrame() {
    if (detecting) {
        requestAnimationFrame(processFrame);
        return;
    }
    detecting = true;

    // 1. Capture frame
    ctx.drawImage(video, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const sceneMat = cv.matFromImageData(imageData);

    // 2. Detect QR codes + extract crops
    const cards = detect(sceneMat);

    // 3. Draw overlays
    overlay.clearRect(0, 0, canvas.width, canvas.height);
    for (const card of cards) {
        drawQuad(overlay, card.qrCorners, '#00ff00', 2);
        drawQuad(overlay, card.skuCorners, '#4488ff', 2);
    }

    // 4. Send crops to OCR worker
    for (let i = 0; i < cards.length; i++) {
        if (cards[i].skuCrop) {
            ocrWorker.postMessage({
                cardIndex: i,
                cropData: new Uint8Array(cards[i].skuCrop.data),
                cropWidth: CARD_LAYOUT.crop_width,
                cropHeight: CARD_LAYOUT.crop_height,
            });
        }
    }

    // 5. Cleanup OpenCV Mats
    sceneMat.delete();
    for (const card of cards) {
        if (card.skuCrop) card.skuCrop.delete();
    }

    detecting = false;
    requestAnimationFrame(processFrame);
}

function handleOCRResult(event) {
    const { cardIndex, text } = event.data;
    updateResultsTable(cardIndex, text);
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

start();
```

**Critical: OpenCV.js memory management.** Every `cv.Mat` must be explicitly
`.delete()`d. OpenCV.js allocates on the WASM heap; the JS garbage collector
does not know about these allocations. Failing to delete Mats will leak
memory and crash the tab within minutes during a camera loop.

---

### 4.2 `detector.js` — Detection Pipeline

The entire detection module. No feature extraction, no matching, no
clustering. `detectMulti` does the hard work.

```javascript
import { CARD_LAYOUT } from './config.js';

let qrDetector = null;

/**
 * Initialize the QR code detector. Called once at startup.
 */
export function initDetector() {
    qrDetector = new cv.QRCodeDetector();
}

/**
 * Detect all QR codes in a scene and extract SKU crops.
 *
 * @param {cv.Mat} sceneMat - RGBA scene image
 * @returns {Array<DetectedCard>}
 *
 * DetectedCard shape:
 * {
 *   qrCorners: [{x,y}, {x,y}, {x,y}, {x,y}],
 *   skuCorners: [{x,y}, {x,y}, {x,y}, {x,y}],
 *   skuCrop: cv.Mat | null,  // grayscale, crop_width × crop_height
 *   decoded: string,          // decoded QR content (bonus — free data)
 * }
 */
export function detect(sceneMat) {
    const sceneGray = new cv.Mat();
    cv.cvtColor(sceneMat, sceneGray, cv.COLOR_RGBA2GRAY);

    // detectMulti returns decoded strings + corner points for each QR
    const points = new cv.Mat();
    const decoded = new cv.StringVector();

    let found = false;
    try {
        found = qrDetector.detectAndDecodeMulti(sceneGray, decoded, points);
    } catch (e) {
        // detectAndDecodeMulti can throw on some builds.
        // Fall back to detectMulti (corners only, no decode).
        try {
            found = qrDetector.detectMulti(sceneGray, points);
        } catch (e2) {
            sceneGray.delete();
            points.delete();
            decoded.delete();
            return [];
        }
    }

    if (!found || points.empty()) {
        sceneGray.delete();
        points.delete();
        decoded.delete();
        return [];
    }

    // points.data32F contains all floats: N codes × 4 corners × 2 coords
    const floats = points.data32F;
    const numQR = floats.length / 8;  // 4 corners × 2 floats each
    const cards = [];

    for (let q = 0; q < numQR; q++) {
        const baseIdx = q * 8;

        const qrCorners = [
            { x: floats[baseIdx + 0], y: floats[baseIdx + 1] },  // top-left
            { x: floats[baseIdx + 2], y: floats[baseIdx + 3] },  // top-right
            { x: floats[baseIdx + 4], y: floats[baseIdx + 5] },  // bottom-right
            { x: floats[baseIdx + 6], y: floats[baseIdx + 7] },  // bottom-left
        ];

        // Compute QR side length (average of top and left edges)
        const topLen = Math.hypot(
            qrCorners[1].x - qrCorners[0].x,
            qrCorners[1].y - qrCorners[0].y,
        );
        const leftLen = Math.hypot(
            qrCorners[3].x - qrCorners[0].x,
            qrCorners[3].y - qrCorners[0].y,
        );
        const qrSize = (topLen + leftLen) / 2;

        // Compute local coordinate axes from the detected corners.
        // axisX: unit vector along the top edge (left → right)
        // axisY: unit vector along the left edge (top → bottom)
        const axisX = {
            x: (qrCorners[1].x - qrCorners[0].x) / topLen,
            y: (qrCorners[1].y - qrCorners[0].y) / topLen,
        };
        const axisY = {
            x: (qrCorners[3].x - qrCorners[0].x) / leftLen,
            y: (qrCorners[3].y - qrCorners[0].y) / leftLen,
        };

        // SKU corners in scene space
        const origin = qrCorners[0];
        const skuCorners = computeSkuCorners(origin, axisX, axisY, qrSize);

        // Extract rectified SKU crop
        const skuCrop = extractCrop(sceneGray, skuCorners);

        cards.push({
            qrCorners,
            skuCorners,
            skuCrop,
            decoded: decoded.size() > q ? decoded.get(q) : '',
        });
    }

    sceneGray.delete();
    points.delete();
    decoded.delete();

    return cards;
}


// --- Internal helpers ---

/**
 * Compute the 4 SKU region corners in scene-image space.
 *
 * Uses the QR code's detected corners to establish a local coordinate
 * frame, then places the SKU box according to CARD_LAYOUT offsets.
 */
function computeSkuCorners(origin, axisX, axisY, qrSize) {
    const { sku_x, sku_y, sku_w, sku_h } = CARD_LAYOUT;

    function toScene(lx, ly) {
        return {
            x: origin.x + (lx * qrSize) * axisX.x + (ly * qrSize) * axisY.x,
            y: origin.y + (lx * qrSize) * axisX.y + (ly * qrSize) * axisY.y,
        };
    }

    return [
        toScene(sku_x,         sku_y),
        toScene(sku_x + sku_w, sku_y),
        toScene(sku_x + sku_w, sku_y + sku_h),
        toScene(sku_x,         sku_y + sku_h),
    ];
}

/**
 * Extract a rectified crop of the SKU region via perspective warp.
 *
 * @param {cv.Mat} sceneGray - grayscale scene image
 * @param {Array<{x,y}>} skuCorners - 4 corners in scene space
 * @returns {cv.Mat} - grayscale thresholded crop. Caller must .delete().
 */
function extractCrop(sceneGray, skuCorners) {
    const { crop_width, crop_height } = CARD_LAYOUT;

    const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2,
        skuCorners.flatMap(p => [p.x, p.y]));
    const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        crop_width, 0,
        crop_width, crop_height,
        0, crop_height,
    ]);

    const M = cv.getPerspectiveTransform(srcQuad, dstQuad);
    const crop = new cv.Mat();
    cv.warpPerspective(
        sceneGray, crop, M,
        new cv.Size(crop_width, crop_height),
    );

    // Otsu threshold for clean OCR input
    const binary = new cv.Mat();
    cv.threshold(crop, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    srcQuad.delete();
    dstQuad.delete();
    M.delete();
    crop.delete();

    return binary; // caller must .delete()
}
```

**That's the entire detection pipeline.** ~150 lines including comments,
replacing what was ~300+ lines with AKAZE + matching + DBSCAN + RANSAC.

---

### 4.3 `ocr-worker.js` — Web Worker for Tesseract.js

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

    const { cardIndex, cropData, cropWidth, cropHeight } = event.data;

    // cropData is a Uint8Array of grayscale pixels (thresholded).
    // Expand to RGBA for Tesseract.
    const rgba = new Uint8ClampedArray(cropWidth * cropHeight * 4);
    for (let i = 0; i < cropData.length; i++) {
        const v = cropData[i];
        rgba[i * 4]     = v;
        rgba[i * 4 + 1] = v;
        rgba[i * 4 + 2] = v;
        rgba[i * 4 + 3] = 255;
    }

    const canvas = new OffscreenCanvas(cropWidth, cropHeight);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(rgba, cropWidth, cropHeight), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });

    const { data: { text } } = await worker.recognize(blob);

    self.postMessage({
        cardIndex,
        text: text.trim(),
    });
};
```

---

## 5. OpenCV.js Memory Management

Every `cv.Mat` must be explicitly `.delete()`d. OpenCV.js allocates on
the WASM heap; the JS garbage collector does not track these allocations.

**In the frame loop, delete ALL Mats created during that frame.**
A 720p RGBA Mat is ~3.7MB. Leaking one per frame exhausts the WASM
heap in seconds.

Use a pool helper to make this less error-prone:

```javascript
class MatPool {
    constructor() { this.mats = []; }
    track(mat) { this.mats.push(mat); return mat; }
    flush() {
        for (const m of this.mats) {
            try { m.delete(); } catch {}
        }
        this.mats = [];
    }
}

// Usage in frame loop:
const pool = new MatPool();
const scene = pool.track(cv.matFromImageData(imageData));
const gray = pool.track(new cv.Mat());
cv.cvtColor(scene, gray, cv.COLOR_RGBA2GRAY);
// ... use gray ...
pool.flush(); // cleans up everything
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

**Verifying the build includes QRCodeDetector:**

After loading, check in the console:
```javascript
console.log(typeof cv.QRCodeDetector);  // should be "function"
```

`QRCodeDetector` is in the `objdetect` module, which is included in
the default `@techstark` builds.

**Custom build (optional, for smaller bundle):**

```bash
python platforms/js/build_js.py build_js \
    --build_wasm \
    --cmake_option="-DBUILD_LIST=objdetect,calib3d,imgproc"
```

Strips everything except what you need. Self-host the output.

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
        <button id="capture-btn">Capture & Read</button>
    </div>

    <div id="results">
        <h2>Detected Cards</h2>
        <table id="results-table">
            <thead>
                <tr>
                    <th>Card #</th>
                    <th>SKU</th>
                    <th>QR Content</th>
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

## 8. Preprocessing for Difficult Conditions

If `detectMulti` struggles with plastic bags or poor lighting, apply
preprocessing to the grayscale image before detection:

```javascript
function preprocess(gray) {
    // 1. Gaussian blur to smooth plastic bag texture
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(3, 3), 0);

    // 2. CLAHE for local contrast enhancement
    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
    const enhanced = new cv.Mat();
    clahe.apply(blurred, enhanced);

    blurred.delete();
    return enhanced; // caller must .delete()
}
```

Try detection on the raw grayscale first. Only add preprocessing if
needed — it costs ~5-10ms per frame.

---

## 9. Performance Budget

| Stage                   | Expected Time (720p) | Notes                  |
| ----------------------- | -------------------- | ---------------------- |
| detectAndDecodeMulti    | 20–50ms              | Scales with QR count   |
| getPerspectiveTransform | <1ms per card        | 4-point exact solution |
| warpPerspective         | <1ms per card        | Small output crop      |
| threshold (Otsu)        | <1ms per card        | On the small crop      |
| **Total**               | **25–60ms**          | **~16–40 fps**         |

This is dramatically faster than AKAZE (~100ms+ just for extraction).

OCR (Tesseract.js) takes 100–500ms per crop but runs asynchronously in
a Worker. It does not affect frame rate.

---

## 10. Deployment

Static files only. Any static hosting works.

```bash
# Local dev with Bun
bun --serve .

# GitHub Pages
git add .
git commit -m "add SKU reader"
git push
```

No build step required. The app is vanilla JS with ES modules.
OpenCV.js and Tesseract.js are loaded from CDN.

For intellisense during development:
```bash
bun init
bun add -d @techstark/opencv-js
```

Add a `jsconfig.json`:
```json
{
    "compilerOptions": {
        "checkJs": true,
        "target": "es2022",
        "module": "es2022"
    },
    "include": ["js/**/*.js"]
}
```

Add `/// <reference types="@techstark/opencv-js" />` at the top of
files that use OpenCV for autocomplete.

---

## 11. Test Plan

| Test                                             | Method                                   |
| ------------------------------------------------ | ---------------------------------------- |
| QR detection finds codes in clean image          | Load test image, verify card count       |
| QR detection handles no-QR scene                 | Load plain image → empty results         |
| SKU corners are correctly positioned             | Draw skuCorners overlay, verify visually |
| Crop extraction produces readable output         | Save crop, verify text is axis-aligned   |
| OCR reads known SKU correctly                    | Feed crop of known SKU → match text      |
| Memory doesn't leak during frame loop            | Monitor WASM heap over 60s               |
| Preprocessing improves detection through plastic | Compare detect count with/without        |
| Export CSV produces valid output                 | Click export, open file, check format    |

**Synthetic test:** Print a QR code several times on a sheet of paper with
SKU labels in known positions. Photograph it. Run detection. Verify correct
count and that OCR reads the expected SKUs.
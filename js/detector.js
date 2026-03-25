import { CARD_LAYOUT } from './config.js';

let qrDetector = null;

export function initDetector() {
    if (qrDetector) qrDetector.delete();
    qrDetector = new cv.QRCodeDetector();
}

/**
 * Detect all QR codes in a scene.
 *
 * Preprocessing (adaptive threshold) runs on the FULL image in one pass so
 * every pixel has consistent neighborhood context — no tile-boundary artifacts.
 * The tiling is only for the QR detector, which struggles with many small codes
 * in a large image but handles small regions well.
 *
 * @param {cv.Mat} sceneMat  - RGBA scene image (detection-canvas resolution)
 * @param {object} params    - { tileSize, upscale, contrast, brightness, sharpen, blockSize, cVal }
 * @returns {{ cards: DetectedCard[], tileCount: number }}
 */
export function detect(sceneMat, params = {}) {
    const { tileSize = 1000, upscale = 1 } = params;

    // Global preprocessing on full image — adaptive threshold sees full context.
    const baseGray = preprocessBase(sceneMat, params);
    const fullBin  = preprocessLocal(baseGray, params);

    const W = fullBin.cols;
    const H = fullBin.rows;

    const overlap = Math.round(tileSize * 0.5);
    const step    = tileSize - overlap;
    const rawHits = [];
    let tileCount = 0;

    for (let ty = 0; ty < H; ty += step) {
        for (let tx = 0; tx < W; tx += step) {
            const tw = Math.min(tileSize, W - tx);
            const th = Math.min(tileSize, H - ty);
            tileCount++;

            // ROI from the already-preprocessed binary image — no clone needed
            // since there is no further neighborhood operation on these tiles.
            const roi = fullBin.roi(new cv.Rect(tx, ty, tw, th));

            let detectMat = roi;
            if (upscale > 1) {
                detectMat = new cv.Mat();
                cv.resize(roi, detectMat, new cv.Size(
                    Math.round(tw * upscale), Math.round(th * upscale)
                ), 0, 0, cv.INTER_LINEAR);
            }

            const hits = _detectInMat(detectMat);
            if (upscale > 1) detectMat.delete();
            roi.delete();

            const inv = 1 / upscale;
            hits.forEach(({ qrCorners, decoded }) => {
                rawHits.push({
                    qrCorners: qrCorners.map(p => ({ x: p.x * inv + tx, y: p.y * inv + ty })),
                    decoded,
                });
            });
        }
    }

    fullBin.delete();

    const unique = _deduplicate(rawHits, tileSize * 0.1);

    // Use baseGray (pre-binary) for SKU crop extraction — better for Otsu OCR.
    const cards = unique.map(({ qrCorners, decoded }) => {
        const topLen  = Math.hypot(qrCorners[1].x - qrCorners[0].x, qrCorners[1].y - qrCorners[0].y);
        const leftLen = Math.hypot(qrCorners[3].x - qrCorners[0].x, qrCorners[3].y - qrCorners[0].y);
        const qrSize  = (topLen + leftLen) / 2;

        const axisX = {
            x: (qrCorners[1].x - qrCorners[0].x) / topLen,
            y: (qrCorners[1].y - qrCorners[0].y) / topLen,
        };
        const axisY = {
            x: (qrCorners[3].x - qrCorners[0].x) / leftLen,
            y: (qrCorners[3].y - qrCorners[0].y) / leftLen,
        };

        const skuCorners = computeSkuCorners(qrCorners[0], axisX, axisY, qrSize);
        const center = {
            x: (qrCorners[0].x + qrCorners[1].x + qrCorners[2].x + qrCorners[3].x) / 4,
            y: (qrCorners[0].y + qrCorners[1].y + qrCorners[2].y + qrCorners[3].y) / 4,
        };

        return {
            qrCorners,
            skuCorners,
            center,
            skuCrop: extractCrop(baseGray, skuCorners),
            decoded,
        };
    });

    baseGray.delete();
    return { cards, tileCount };
}

/**
 * Global preprocessing: RGBA → grayscale → contrast/brightness → sharpen.
 * Safe to run on the full image; no large-neighborhood operations.
 * Exported for use in the preview renderer.
 */
export function preprocessBase(sceneMat, params = {}) {
    const { contrast = 1, brightness = 0, sharpen = 0 } = params;

    const gray = new cv.Mat();
    cv.cvtColor(sceneMat, gray, cv.COLOR_RGBA2GRAY);

    let working = new cv.Mat();
    cv.convertScaleAbs(gray, working, contrast, brightness);
    gray.delete();

    if (sharpen > 0) {
        const k = cv.matFromArray(3, 3, cv.CV_32F, [
            0, -sharpen, 0,
            -sharpen, 1 + 4 * sharpen, -sharpen,
            0, -sharpen, 0,
        ]);
        const sharpened = new cv.Mat();
        cv.filter2D(working, sharpened, -1, k);
        k.delete();
        working.delete();
        working = sharpened;
    }

    return working;
}

/**
 * Local preprocessing: adaptive threshold + morphological close.
 * Should be called on the full image (or a standalone clone) for consistent results.
 * Exported for use in the preview renderer.
 */
export function preprocessLocal(grayMat, params = {}) {
    const { blockSize = 51, cVal = 10 } = params;

    const binary = new cv.Mat();
    cv.adaptiveThreshold(
        grayMat, binary, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        blockSize,
        cVal,
    );

    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const closed = new cv.Mat();
    cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel);
    kernel.delete();
    binary.delete();

    return closed;
}


// --- Internal helpers ---

function _detectInMat(mat) {
    const points  = new cv.Mat();
    const decoded = new cv.StringVector();
    let found = false;

    try {
        found = qrDetector.detectAndDecodeMulti(mat, decoded, points);
    } catch (_) {
        try {
            found = qrDetector.detectMulti(mat, points);
        } catch (e2) {
            points.delete(); decoded.delete();
            throw Object.assign(new Error('QR detectMulti failed'), { cause: e2 });
        }
    }

    if (!found || points.empty()) {
        points.delete(); decoded.delete();
        return [];
    }

    const floats = points.data32F;
    const numQR  = floats.length / 8;
    const results = [];

    for (let q = 0; q < numQR; q++) {
        const b = q * 8;
        results.push({
            qrCorners: [
                { x: floats[b],     y: floats[b + 1] },
                { x: floats[b + 2], y: floats[b + 3] },
                { x: floats[b + 4], y: floats[b + 5] },
                { x: floats[b + 6], y: floats[b + 7] },
            ],
            decoded: decoded.size() > q ? decoded.get(q) : '',
        });
    }

    points.delete();
    decoded.delete();
    return results;
}

function _deduplicate(detections, threshold) {
    const kept = [];
    for (const det of detections) {
        const cx = (det.qrCorners[0].x + det.qrCorners[1].x + det.qrCorners[2].x + det.qrCorners[3].x) / 4;
        const cy = (det.qrCorners[0].y + det.qrCorners[1].y + det.qrCorners[2].y + det.qrCorners[3].y) / 4;
        const dup = kept.some(k => {
            const kx = (k.qrCorners[0].x + k.qrCorners[1].x + k.qrCorners[2].x + k.qrCorners[3].x) / 4;
            const ky = (k.qrCorners[0].y + k.qrCorners[1].y + k.qrCorners[2].y + k.qrCorners[3].y) / 4;
            return Math.hypot(cx - kx, cy - ky) < threshold;
        });
        if (!dup) kept.push(det);
    }
    return kept;
}

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

function extractCrop(sceneGray, skuCorners) {
    const { crop_width, crop_height } = CARD_LAYOUT;

    const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2, skuCorners.flatMap(p => [p.x, p.y]));
    const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0, crop_width, 0, crop_width, crop_height, 0, crop_height,
    ]);

    const M    = cv.getPerspectiveTransform(srcQuad, dstQuad);
    const crop = new cv.Mat();
    cv.warpPerspective(sceneGray, crop, M, new cv.Size(crop_width, crop_height));

    const binary = new cv.Mat();
    cv.threshold(crop, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    srcQuad.delete(); dstQuad.delete(); M.delete(); crop.delete();
    return binary;
}

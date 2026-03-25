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
 * Called once at startup (or whenever the reference image changes).
 *
 * @param {HTMLImageElement|HTMLCanvasElement} refImage
 */
export function initReference(refImage) {
    // Clean up previous state
    if (akaze) { akaze.delete(); akaze = null; }
    if (matcher) { matcher.delete(); matcher = null; }
    if (refDescriptors) { refDescriptors.delete(); refDescriptors = null; }
    if (refKeypoints) { refKeypoints.delete(); refKeypoints = null; }

    akaze = new cv.AKAZE();
    akaze.setThreshold(DETECTION.akaze_threshold);

    matcher = new cv.BFMatcher(cv.NORM_HAMMING);

    const refMat = cv.imread(refImage);
    const refGray = new cv.Mat();
    cv.cvtColor(refMat, refGray, cv.COLOR_RGBA2GRAY);

    refWidth = refGray.cols;
    refHeight = refGray.rows;

    refKeypoints = new cv.KeyPointVector();
    refDescriptors = new cv.Mat();
    const emptyMask = new cv.Mat();
    akaze.detectAndCompute(refGray, emptyMask, refKeypoints, refDescriptors);

    refMat.delete();
    refGray.delete();
    emptyMask.delete();
}

/**
 * Run the full detection pipeline on a scene frame.
 *
 * @param {cv.Mat} sceneMat - RGBA scene image (from matFromImageData or cv.imread)
 * @returns {Array<DetectedCard>}
 *
 * DetectedCard shape:
 * {
 *   homography: cv.Mat (3x3),      — caller must .delete()
 *   inlierCount: number,
 *   center: { x, y },
 *   qrCorners: [{x,y} × 4],
 *   skuCorners: [{x,y} × 4],
 *   skuCrop: cv.Mat (grayscale),   — caller must .delete()
 * }
 */
export function detect(sceneMat) {
    if (!akaze || !refDescriptors) return [];

    // --- Stage 1: Feature extraction ---
    const sceneGray = new cv.Mat();
    cv.cvtColor(sceneMat, sceneGray, cv.COLOR_RGBA2GRAY);

    const sceneKeypoints = new cv.KeyPointVector();
    const sceneDescriptors = new cv.Mat();
    const emptyMask = new cv.Mat();
    akaze.detectAndCompute(sceneGray, emptyMask, sceneKeypoints, sceneDescriptors);
    emptyMask.delete();

    if (sceneDescriptors.rows < DETECTION.min_inliers) {
        sceneGray.delete();
        sceneKeypoints.delete();
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

        const srcPts = cv.matFromArray(clusterMatches.length, 1, cv.CV_32FC2,
            clusterMatches.flatMap(m => [m.refPt.x, m.refPt.y]));
        const dstPts = cv.matFromArray(clusterMatches.length, 1, cv.CV_32FC2,
            clusterMatches.flatMap(m => [m.scenePt.x, m.scenePt.y]));

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

        const center = {
            x: qrCorners.reduce((s, p) => s + p.x, 0) / 4,
            y: qrCorners.reduce((s, p) => s + p.y, 0) / 4,
        };

        const skuCrop = extractCrop(sceneGray, skuCorners);

        cards.push({
            homography: H,
            inlierCount,
            center,
            qrCorners,
            skuCorners,
            skuCrop,
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
 * Returns a grayscale cv.Mat (crop_width × crop_height). Caller must delete.
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

    const binary = new cv.Mat();
    cv.threshold(crop, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    srcQuad.delete();
    dstQuad.delete();
    M.delete();
    crop.delete();

    return binary;
}

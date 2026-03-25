import { DETECTION, CARD_LAYOUT } from './config.js';
import { dbscan } from './clustering.js';

// Persistent OpenCV objects (created once, reused per run).
let akaze = null;
let matcher = null;

// Reference data (computed once).
let refDescriptors = null;
let refKeypoints = null;
let refWidth = 0;
let refHeight = 0;

export function initReference(refImage) {
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
    const mask = new cv.Mat();
    akaze.detectAndCompute(refGray, mask, refKeypoints, refDescriptors);

    refMat.delete();
    refGray.delete();
    mask.delete();
}

export function getRefInfo() {
    return {
        keypointCount: refKeypoints ? refKeypoints.size() : 0,
        width: refWidth,
        height: refHeight,
    };
}

/** Extract all keypoint positions from a KeyPointVector into a plain JS array. */
function extractKpPositions(kv) {
    const pts = [];
    for (let i = 0; i < kv.size(); i++) {
        const kp = kv.get(i);
        pts.push({ x: kp.pt.x, y: kp.pt.y });
    }
    return pts;
}

/**
 * Run the full detection pipeline.
 *
 * Returns { cards, debug } where debug always contains intermediate data
 * regardless of whether cards were found, to enable visualization.
 *
 * debug shape:
 * {
 *   refPoints:    [{x,y}]          — all reference keypoints
 *   scenePoints:  [{x,y}]          — all scene keypoints (after detectAndCompute)
 *   goodMatches:  [{refPt,scenePt}] — after Lowe ratio test
 *   labels:       Int32Array|null   — DBSCAN cluster label per goodMatch (-1 = noise)
 * }
 */
export function detect(sceneMat) {
    const debug = {
        refPoints:   extractKpPositions(refKeypoints ?? new cv.KeyPointVector()),
        scenePoints: [],
        goodMatches: [],
        labels:      null,
    };

    if (!akaze || !refDescriptors) return { cards: [], debug };

    // --- Stage 1: Feature extraction ---
    const sceneGray = new cv.Mat();
    try {
        cv.cvtColor(sceneMat, sceneGray, cv.COLOR_RGBA2GRAY);
    } catch (e) {
        sceneGray.delete();
        throw Object.assign(new Error('stage1: cvtColor'), { cause: e });
    }

    const sceneKeypoints = new cv.KeyPointVector();
    const sceneDescriptors = new cv.Mat();
    const emptyMask = new cv.Mat();
    try {
        akaze.detectAndCompute(sceneGray, emptyMask, sceneKeypoints, sceneDescriptors);
    } catch (e) {
        sceneGray.delete(); sceneKeypoints.delete(); sceneDescriptors.delete(); emptyMask.delete();
        throw Object.assign(new Error('stage1: detectAndCompute (scene)'), { cause: e });
    }
    emptyMask.delete();

    // Capture scene keypoints for debug before any early return
    debug.scenePoints = extractKpPositions(sceneKeypoints);

    if (sceneDescriptors.rows < DETECTION.min_inliers) {
        sceneGray.delete(); sceneKeypoints.delete(); sceneDescriptors.delete();
        return { cards: [], debug };
    }

    // --- Stage 2: Descriptor matching + ratio test ---
    const rawMatches = new cv.DMatchVectorVector();
    try {
        matcher.knnMatch(refDescriptors, sceneDescriptors, rawMatches, 2);
    } catch (e) {
        sceneGray.delete(); sceneKeypoints.delete(); sceneDescriptors.delete(); rawMatches.delete();
        throw Object.assign(new Error('stage2: knnMatch'), { cause: e });
    }

    const goodMatches = [];
    for (let i = 0; i < rawMatches.size(); i++) {
        const pair = rawMatches.get(i);
        if (pair.size() < 2) continue;
        const m = pair.get(0);
        const n = pair.get(1);
        if (m.distance < DETECTION.lowe_ratio * n.distance) {
            goodMatches.push({
                refPt:   keypointToPoint(refKeypoints, m.queryIdx),
                scenePt: keypointToPoint(sceneKeypoints, m.trainIdx),
            });
        }
    }

    rawMatches.delete();
    sceneDescriptors.delete();

    // Capture good matches before early return
    debug.goodMatches = goodMatches;

    if (goodMatches.length < DETECTION.min_inliers) {
        sceneGray.delete(); sceneKeypoints.delete();
        return { cards: [], debug };
    }

    // --- Stage 3: Spatial clustering ---
    const scenePts = goodMatches.map(m => m.scenePt);
    const labels = dbscan(scenePts, DETECTION.dbscan_eps, DETECTION.dbscan_min_samples);
    debug.labels = labels;

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
        const H = cv.findHomography(srcPts, dstPts, cv.RANSAC, DETECTION.ransac_reproj_thresh, mask);
        srcPts.delete();
        dstPts.delete();

        if (H.empty()) { H.delete(); mask.delete(); continue; }

        let inlierCount = 0;
        for (let i = 0; i < mask.rows; i++) {
            if (mask.data[i] !== 0) inlierCount++;
        }
        mask.delete();

        if (inlierCount < DETECTION.min_inliers) { H.delete(); continue; }

        const qrCornersRef = [[0,0],[refWidth,0],[refWidth,refHeight],[0,refHeight]];
        const qrCorners = transformPoints(qrCornersRef, H);

        const sx = CARD_LAYOUT.sku_x * refWidth;
        const sy = CARD_LAYOUT.sku_y * refHeight;
        const sw = CARD_LAYOUT.sku_w * refWidth;
        const sh = CARD_LAYOUT.sku_h * refHeight;
        const skuCorners = transformPoints([[sx,sy],[sx+sw,sy],[sx+sw,sy+sh],[sx,sy+sh]], H);

        const center = {
            x: qrCorners.reduce((s, p) => s + p.x, 0) / 4,
            y: qrCorners.reduce((s, p) => s + p.y, 0) / 4,
        };

        cards.push({
            homography: H,
            inlierCount,
            center,
            qrCorners,
            skuCorners,
            skuCrop: extractCrop(sceneGray, skuCorners),
        });
    }

    sceneGray.delete();
    sceneKeypoints.delete();

    return { cards, debug };
}


// --- Internal helpers ---

function keypointToPoint(keypoints, idx) {
    const kp = keypoints.get(idx);
    return { x: kp.pt.x, y: kp.pt.y };
}

function transformPoints(pts, H) {
    const src = cv.matFromArray(pts.length, 1, cv.CV_32FC2, pts.flatMap(([x, y]) => [x, y]));
    const dst = new cv.Mat();
    cv.perspectiveTransform(src, dst, H);
    const result = [];
    for (let i = 0; i < pts.length; i++) {
        result.push({ x: dst.floatAt(i, 0), y: dst.floatAt(i, 1) });
    }
    src.delete();
    dst.delete();
    return result;
}

function extractCrop(sceneGray, skuCorners) {
    const srcQuad = cv.matFromArray(4, 1, cv.CV_32FC2, skuCorners.flatMap(p => [p.x, p.y]));
    const dstQuad = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0, CARD_LAYOUT.crop_width, 0,
        CARD_LAYOUT.crop_width, CARD_LAYOUT.crop_height, 0, CARD_LAYOUT.crop_height,
    ]);

    const M = cv.getPerspectiveTransform(srcQuad, dstQuad);
    const crop = new cv.Mat();
    cv.warpPerspective(sceneGray, crop, M, new cv.Size(CARD_LAYOUT.crop_width, CARD_LAYOUT.crop_height));

    const binary = new cv.Mat();
    cv.threshold(crop, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    srcQuad.delete(); dstQuad.delete(); M.delete(); crop.delete();
    return binary;
}

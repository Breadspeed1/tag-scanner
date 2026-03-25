import { DETECTION, CARD_LAYOUT } from './config.js'
import { dbscan } from './clustering.js'

let akaze = null;
let matcher = null;

let refDescriptors = null;
let refKeypoints = null;
let refWidth = 0;
let refHeight = 0;

export function initReference(refImage) {
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

    akaze.detectAndCompute(refGray, new cv.Mat(), refKeypoints, refDescriptors);

    refMat.delete();
    refGray.delete();
}
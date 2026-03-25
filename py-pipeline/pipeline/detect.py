import cv2
import numpy as np


def _pick_largest_quad(points):
    best_corners = None
    best_area = 0
    for quad in points:
        quad = quad.reshape(4, 2)
        area = cv2.contourArea(quad.astype(np.float32))
        if area > best_area:
            best_area = area
            best_corners = quad
    return best_corners


def _preprocess_crop(crop, method):
    if method == 'none':
        return crop
    elif method == 'sharpen':
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(crop, -1, k)
    elif method == 'unsharp':
        blur = cv2.GaussianBlur(crop, (0, 0), 3)
        return cv2.addWeighted(crop, 1.5, blur, -0.5, 0)
    elif method == 'clahe':
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cl.apply(gray)
    return crop


def _subpix_refine(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    refined = cv2.cornerSubPix(gray, corners.copy(), (5, 5), (-1, -1), criteria)
    return refined


def detect_qr_in_aoi(img, box, pad=120, min_crop_size=400,
                     detector='aruco+standard', preprocessing='none',
                     subpix=False, min_module_size=4.0, max_colors_mismatch=0.2):
    x, y, w, h = box
    img_h, img_w = img.shape[:2]

    rx = max(0, x - pad)
    ry = max(0, y - pad)
    rw = min(w + 2 * pad, img_w - rx)
    rh = min(h + 2 * pad, img_h - ry)

    crop = img[ry:ry + rh, rx:rx + rw]

    scale = 1.0
    min_dim = min(rw, rh)
    if min_dim < min_crop_size:
        scale = min_crop_size / min_dim
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    crop_pp = _preprocess_crop(crop, preprocessing)

    # Build ArUco detector with custom params
    aruco_det = cv2.QRCodeDetectorAruco()
    params = aruco_det.getDetectorParameters()
    params.minModuleSizeInPyramid = min_module_size
    params.maxColorsMismatch = max_colors_mismatch
    aruco_det.setDetectorParameters(params)

    standard_det = cv2.QRCodeDetector()

    best = None

    if detector in ('aruco', 'aruco+standard'):
        ret_a, pts_a = aruco_det.detectMulti(crop_pp)
        if ret_a and pts_a is not None and len(pts_a) > 0:
            best = _pick_largest_quad(pts_a)

    if detector == 'standard' or (detector == 'aruco+standard' and best is not None):
        ret_s, pts_s = standard_det.detectMulti(crop_pp)
        if ret_s and pts_s is not None and len(pts_s) > 0:
            best = _pick_largest_quad(pts_s)

    if detector == 'standard' and best is None:
        ret_s, pts_s = standard_det.detectMulti(crop_pp)
        if ret_s and pts_s is not None and len(pts_s) > 0:
            best = _pick_largest_quad(pts_s)

    if best is None:
        return None

    corners = best.astype(np.float32)
    corners /= scale
    corners[:, 0] += rx
    corners[:, 1] += ry

    if subpix:
        corners = _subpix_refine(img, corners)

    return corners

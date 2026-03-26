import cv2
import numpy as np


MIN_SQUARENESS = 0.5

# Margin (in pixels, in crop space) around each quad edge to search for
# strong edges during edge-line refinement.
EDGE_SEARCH_MARGIN = 15


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


def _quad_squareness(corners):
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return min(sides) / max(sides) if max(sides) > 0 else 0


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


def _intersect_lines(line_a, line_b):
    """Intersect two lines defined as (point, direction) tuples."""
    p1, d1 = line_a
    p2, d2 = line_b
    # Solve p1 + t*d1 = p2 + s*d2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return p1 + t * d1


def _refine_corners_by_edges(gray, corners, expand=0.3, search_margin=EDGE_SEARCH_MARGIN):
    """
    Refine quad corners by adaptive-thresholding an expanded ROI to get a
    clean binary of the white QR background, finding edge pixels on the
    outer boundary, assigning them to the four quad sides, fitting a line
    per side, and intersecting adjacent lines for refined corners.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    center = corners.mean(axis=0)
    expanded = center + (corners - center) * (1 + expand)
    h, w = gray.shape[:2]

    x_min = max(0, int(expanded[:, 0].min()))
    y_min = max(0, int(expanded[:, 1].min()))
    x_max = min(w, int(expanded[:, 0].max()))
    y_max = min(h, int(expanded[:, 1].max()))

    if x_max - x_min < 10 or y_max - y_min < 10:
        return corners

    roi = gray[y_min:y_max, x_min:x_max]

    # Adaptive threshold — white QR background becomes 255
    binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
    # Find the largest contour — outer boundary of the white QR region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return corners

    orig_area = cv2.contourArea(corners.astype(np.float32))
    best_cnt = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area and area > orig_area * 0.3:
            best_area = area
            best_cnt = cnt
    if best_cnt is None:
        return corners

    # Contour points in crop space
    edge_pts = best_cnt.reshape(-1, 2).astype(np.float32)
    edge_pts[:, 0] += x_min
    edge_pts[:, 1] += y_min

    avg_side = np.mean([np.linalg.norm(corners[(k + 1) % 4] - corners[k]) for k in range(4)])
    refined_lines = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1:
            refined_lines.append(None)
            continue

        edge_dir = edge_vec / edge_len
        normal = np.array([-edge_dir[1], edge_dir[0]])
        to_center = center - (p1 + p2) / 2
        if np.dot(normal, to_center) > 0:
            normal = -normal

        # Select edge pixels near this quad side:
        # - within search_margin perpendicular distance
        # - between 10%..90% along the edge (avoid corners)
        # - on the outward side (positive normal distance)
        dp = edge_pts - p1
        along = dp @ edge_dir
        perp = dp @ normal

        mask = ((along > edge_len * 0.1) & (along < edge_len * 0.9) &
                (perp > -2) & (perp < search_margin))
        candidates = edge_pts[mask]

        if len(candidates) < 5:
            refined_lines.append(None)
            continue

        # Bin along the edge direction, take median perpendicular distance
        # per bin to resist glare/noise outliers that push the contour out
        along_vals = (candidates - p1) @ edge_dir
        perp_vals = (candidates - p1) @ normal
        n_bins = max(3, int(edge_len / 8))
        bins = np.linspace(edge_len * 0.1, edge_len * 0.9, n_bins + 1)
        outer_pts = []
        for b in range(n_bins):
            in_bin = (along_vals >= bins[b]) & (along_vals < bins[b + 1])
            if not np.any(in_bin):
                continue
            bin_perps = perp_vals[in_bin]
            bin_alongs = along_vals[in_bin]
            med_perp = np.median(bin_perps)
            med_along = np.median(bin_alongs)
            outer_pts.append(p1 + med_along * edge_dir + med_perp * normal)

        if len(outer_pts) < 3:
            refined_lines.append(None)
            continue

        pts_arr = np.array(outer_pts, dtype=np.float32).reshape(-1, 1, 2)
        line = cv2.fitLine(pts_arr, cv2.DIST_HUBER, 0, 0.01, 0.01)
        refined_lines.append((np.array([line[2][0], line[3][0]]),
                              np.array([line[0][0], line[1][0]])))

    # Intersect adjacent lines for corners
    refined_corners = corners.copy()
    for i in range(4):
        line_a = refined_lines[i]
        line_b = refined_lines[(i + 1) % 4]
        if line_a is None or line_b is None:
            continue
        pt = _intersect_lines(line_a, line_b)
        if pt is not None and np.linalg.norm(pt - corners[(i + 1) % 4]) < avg_side * 0.4:
            refined_corners[(i + 1) % 4] = pt

    return refined_corners


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

    aruco_det = cv2.QRCodeDetectorAruco()
    params = aruco_det.getDetectorParameters()
    params.minModuleSizeInPyramid = min_module_size
    params.maxColorsMismatch = max_colors_mismatch
    aruco_det.setDetectorParameters(params)

    standard_det = cv2.QRCodeDetector()

    aruco_quad = None
    standard_quad = None

    if detector in ('aruco', 'aruco+standard'):
        ret_a, pts_a = aruco_det.detectMulti(crop_pp)
        if ret_a and pts_a is not None and len(pts_a) > 0:
            aruco_quad = _pick_largest_quad(pts_a)

    if detector in ('standard', 'aruco+standard'):
        ret_s, pts_s = standard_det.detectMulti(crop_pp)
        if ret_s and pts_s is not None and len(pts_s) > 0:
            standard_quad = _pick_largest_quad(pts_s)

    # Pick the best quad: prefer whichever has higher squareness
    best = None
    if detector == 'aruco':
        best = aruco_quad
    elif detector == 'standard':
        best = standard_quad
    elif detector == 'aruco+standard':
        if aruco_quad is not None and standard_quad is not None:
            sq_a = _quad_squareness(aruco_quad)
            sq_s = _quad_squareness(standard_quad)
            best = standard_quad if sq_s >= sq_a else aruco_quad
        else:
            best = aruco_quad if aruco_quad is not None else standard_quad

    if best is None:
        return None

    if _quad_squareness(best) < MIN_SQUARENESS:
        return None

    # Refine corners by snapping each edge to the strongest gradient
    best = _refine_corners_by_edges(crop, best.astype(np.float32))

    corners = best.astype(np.float32)
    corners /= scale
    corners[:, 0] += rx
    corners[:, 1] += ry

    if subpix:
        from pipeline.detect import _subpix_refine
        corners = _subpix_refine(img, corners)

    return corners

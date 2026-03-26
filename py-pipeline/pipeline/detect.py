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
    # Morphological closing to bridge gaps from glare
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
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


def _run_aruco_detector(crop_pp, min_module_size, max_colors_mismatch):
    """Run ArUco QR detector. Returns quad or None."""
    det = cv2.QRCodeDetectorAruco()
    params = det.getDetectorParameters()
    params.minModuleSizeInPyramid = min_module_size
    params.maxColorsMismatch = max_colors_mismatch
    det.setDetectorParameters(params)
    ret, pts = det.detectMulti(crop_pp)
    if ret and pts is not None and len(pts) > 0:
        return _pick_largest_quad(pts)
    return None


def _run_standard_detector(crop_pp):
    """Run standard QR detector. Returns quad or None."""
    det = cv2.QRCodeDetector()
    ret, pts = det.detectMulti(crop_pp)
    if ret and pts is not None and len(pts) > 0:
        return _pick_largest_quad(pts)
    return None


def _run_wechat_detector(crop):
    """Run WeChat DNN-based QR detector. Returns quad or None."""
    try:
        wechat = cv2.wechat_qrcode.WeChatQRCode()
        results, points = wechat.detectAndDecode(crop)
        if points is not None and len(points) > 0:
            return _pick_largest_quad(points)
    except Exception:
        pass
    return None


def _pick_best_candidate(candidates):
    """From a list of (name, quad) pairs, pick the one with highest squareness."""
    best = None
    best_sq = -1
    best_name = None
    for name, quad in candidates:
        if quad is None:
            continue
        sq = _quad_squareness(quad)
        if sq > best_sq:
            best_sq = sq
            best = quad
            best_name = name
    return best, best_name, best_sq


def _detect_finder_patterns(gray_crop):
    """
    Detect QR finder patterns by looking for nested contours with the
    characteristic 1:1:3:1:1 ratio. Returns a 4-corner quad or None.
    """
    if len(gray_crop.shape) == 3:
        gray_crop = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None

    hierarchy = hierarchy[0]  # shape: (N, 4) — [next, prev, child, parent]

    # Find contours with at least 2 levels of nesting (parent > child > grandchild)
    finder_centers = []
    finder_sizes = []

    for i in range(len(contours)):
        # Check this contour has a child
        child = hierarchy[i][2]
        if child == -1:
            continue
        # Check the child has a grandchild
        grandchild = hierarchy[child][2]
        if grandchild == -1:
            continue

        # Check area ratios roughly match finder pattern (outer:middle:inner ~ 49:25:9)
        area_outer = cv2.contourArea(contours[i])
        area_middle = cv2.contourArea(contours[child])
        area_inner = cv2.contourArea(contours[grandchild])

        if area_outer < 50 or area_middle < 10 or area_inner < 2:
            continue

        ratio_mid = area_middle / area_outer
        ratio_inner = area_inner / area_outer

        # Expected: middle/outer ~ 0.51, inner/outer ~ 0.18
        # Allow wide tolerance
        if not (0.2 < ratio_mid < 0.8 and 0.05 < ratio_inner < 0.45):
            continue

        # Also check the outer contour is roughly square
        rect = cv2.minAreaRect(contours[i])
        w, h = rect[1]
        if w < 1 or h < 1:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < 0.5:
            continue

        M = cv2.moments(contours[i])
        if M['m00'] < 1:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        finder_centers.append(np.array([cx, cy]))
        finder_sizes.append(np.sqrt(area_outer))

    if len(finder_centers) < 3:
        return None

    # Deduplicate: cluster centers within half the median size
    centers = np.array(finder_centers)
    sizes = np.array(finder_sizes)
    median_size = np.median(sizes)

    unique = []
    unique_sizes = []
    used = set()
    for i in range(len(centers)):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(centers)):
            if j in used:
                continue
            if np.linalg.norm(centers[i] - centers[j]) < median_size * 0.5:
                cluster.append(j)
                used.add(j)
        used.add(i)
        # Take the one with the largest area
        best_idx = max(cluster, key=lambda k: sizes[k])
        unique.append(centers[best_idx])
        unique_sizes.append(sizes[best_idx])

    if len(unique) < 3:
        return None

    # Pick the 3 finder patterns closest in size to each other
    unique = np.array(unique)
    unique_sizes = np.array(unique_sizes)

    if len(unique) > 3:
        # Pick 3 with most consistent sizes
        best_triple = None
        best_size_var = float('inf')
        from itertools import combinations
        for combo in combinations(range(len(unique)), 3):
            s = unique_sizes[list(combo)]
            var = s.std() / s.mean() if s.mean() > 0 else float('inf')
            if var < best_size_var:
                best_size_var = var
                best_triple = combo
        if best_triple is None:
            return None
        unique = unique[list(best_triple)]
        unique_sizes = unique_sizes[list(best_triple)]

    # Identify which is the top-left (the one at the right angle)
    # The top-left finder pattern is the one where the other two form a ~90 degree angle
    p0, p1, p2 = unique[0], unique[1], unique[2]
    combos = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
    best_corner = 0
    best_angle_err = float('inf')
    for corner, a, b in combos:
        v1 = unique[a] - unique[corner]
        v2 = unique[b] - unique[corner]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle_err = abs(cos_angle)  # cos(90°) = 0
        if angle_err < best_angle_err:
            best_angle_err = angle_err
            best_corner = corner

    # Reorder: top-left, then the two others
    tl_idx = best_corner
    other = [i for i in range(3) if i != tl_idx]

    tl = unique[tl_idx]
    pa = unique[other[0]]
    pb = unique[other[1]]

    # Determine which is top-right vs bottom-left using cross product
    va = pa - tl
    vb = pb - tl
    cross = va[0] * vb[1] - va[1] * vb[0]
    if cross > 0:
        tr, bl = pa, pb
    else:
        tr, bl = pb, pa

    # Infer bottom-right: BR = TR + BL - TL
    br = tr + bl - tl

    # Expand from finder centers to QR boundary
    # Each finder pattern center is ~3.5 modules from the QR edge
    # The finder pattern outer size is ~7 modules
    # So offset from center to edge is ~3.5 modules = half the finder width
    avg_finder_size = unique_sizes.mean()
    # finder outer contour is ~7 modules, so half-width = size/2
    # QR boundary is ~0.5 modules outside the finder edge
    half_finder = avg_finder_size / 2 * 1.1  # slight extra

    # Expand each corner outward from center
    center = (tl + tr + bl + br) / 4
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    for i in range(4):
        direction = quad[i] - center
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        quad[i] = quad[i] + direction * half_finder * 0.3

    return quad


def _dewarp_redetect(crop, corners, detector, min_module_size, max_colors_mismatch):
    """Dewarp a detected QR using initial corners, re-detect on frontal view, map back."""
    size = 300
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    dewarped = cv2.warpPerspective(crop, M, (size, size))

    # Re-detect on the dewarped (frontal) image — try all detectors
    candidates = [
        ('aruco', _run_aruco_detector(dewarped, min_module_size, max_colors_mismatch)),
        ('standard', _run_standard_detector(dewarped)),
        ('wechat', _run_wechat_detector(dewarped)),
    ]
    new_quad, _, _ = _pick_best_candidate(candidates)
    if new_quad is None:
        return corners

    # Map the refined corners back to crop space using true matrix inverse
    # (not getPerspectiveTransform which depends on corner ordering)
    _, M_inv = cv2.invert(M)
    new_corners = cv2.perspectiveTransform(
        new_quad.reshape(1, -1, 2).astype(np.float32), M_inv
    )
    result = new_corners.reshape(4, 2)

    # Sanity check: refined corners shouldn't be too far from originals
    max_drift = np.mean([np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(4)]) * 0.5
    if np.max(np.linalg.norm(result - corners, axis=1)) > max_drift:
        return corners

    return result


def detect_qr_in_aoi(img, box, pad=120, min_crop_size=400,
                     detector='aruco+standard', preprocessing='none',
                     subpix=False, min_module_size=4.0, max_colors_mismatch=0.2,
                     diagnostic=False):
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

    diag = {
        'crop': crop.copy(),
        'scale': scale,
        'roi': (rx, ry, rw, rh),
        'attempts': [],
        'winner': None,
        'squareness_filter': None,
        'dewarp': None,
        'corners_before_refine': None,
        'corners_after_refine': None,
    } if diagnostic else None

    # Run ALL detectors and collect candidates — pick best by squareness
    candidates = []

    # --- Original preprocessing ---
    crop_pp = _preprocess_crop(crop, preprocessing)
    pp_input = crop_pp.copy() if len(crop_pp.shape) == 3 else cv2.cvtColor(crop_pp, cv2.COLOR_GRAY2BGR)

    aruco_q = _run_aruco_detector(crop_pp, min_module_size, max_colors_mismatch)
    candidates.append(('aruco', aruco_q))
    if diagnostic:
        diag['attempts'].append({'name': 'aruco', 'input': pp_input.copy(),
                                 'quad': aruco_q.copy() if aruco_q is not None else None})

    standard_q = _run_standard_detector(crop_pp)
    candidates.append(('standard', standard_q))
    if diagnostic:
        diag['attempts'].append({'name': 'standard', 'input': pp_input.copy(),
                                 'quad': standard_q.copy() if standard_q is not None else None})

    wechat_q = _run_wechat_detector(crop)
    candidates.append(('wechat', wechat_q))
    if diagnostic:
        diag['attempts'].append({'name': 'wechat', 'input': crop.copy(),
                                 'quad': wechat_q.copy() if wechat_q is not None else None})

    # --- CLAHE preprocessing ---
    if preprocessing != 'clahe':
        crop_clahe = _preprocess_crop(crop, 'clahe')
        clahe_bgr = crop_clahe if len(crop_clahe.shape) == 3 else cv2.cvtColor(crop_clahe, cv2.COLOR_GRAY2BGR)

        aruco_clahe_q = _run_aruco_detector(crop_clahe, min_module_size, max_colors_mismatch)
        candidates.append(('aruco+clahe', aruco_clahe_q))
        if diagnostic:
            diag['attempts'].append({'name': 'aruco (clahe)', 'input': clahe_bgr.copy(),
                                     'quad': aruco_clahe_q.copy() if aruco_clahe_q is not None else None})

        standard_clahe_q = _run_standard_detector(crop_clahe)
        candidates.append(('standard+clahe', standard_clahe_q))
        if diagnostic:
            diag['attempts'].append({'name': 'standard (clahe)', 'input': clahe_bgr.copy(),
                                     'quad': standard_clahe_q.copy() if standard_clahe_q is not None else None})

        wechat_clahe_q = _run_wechat_detector(clahe_bgr)
        candidates.append(('wechat+clahe', wechat_clahe_q))
        if diagnostic:
            diag['attempts'].append({'name': 'wechat (clahe)', 'input': clahe_bgr.copy(),
                                     'quad': wechat_clahe_q.copy() if wechat_clahe_q is not None else None})

    # --- Finder pattern fallback ---
    finder_q = _detect_finder_patterns(crop)
    candidates.append(('finder', finder_q))
    if diagnostic:
        diag['attempts'].append({'name': 'finder patterns', 'input': crop.copy(),
                                 'quad': finder_q.copy() if finder_q is not None else None})

    # Pick best candidate by squareness
    best, winner_name, best_sq = _pick_best_candidate(candidates)

    if diagnostic and best is not None:
        diag['winner'] = winner_name

    if best is None:
        if diagnostic:
            return None, diag
        return None

    if best_sq < MIN_SQUARENESS:
        if diagnostic:
            diag['squareness_filter'] = best_sq
            return None, diag
        return None

    # Iterative dewarp-redetect for heavy perspective
    if best_sq < 0.85:
        dewarped_best = _dewarp_redetect(crop, best.astype(np.float32), detector,
                                         min_module_size, max_colors_mismatch)
        if diagnostic:
            diag['dewarp'] = {
                'before': best.copy(),
                'after': dewarped_best.copy(),
                'squareness': best_sq,
            }
        best = dewarped_best

    if diagnostic:
        diag['corners_before_refine'] = best.copy()

    best = _refine_corners_by_edges(crop, best.astype(np.float32))

    if diagnostic:
        diag['corners_after_refine'] = best.copy()

    corners = best.astype(np.float32)
    corners /= scale
    corners[:, 0] += rx
    corners[:, 1] += ry

    if subpix:
        from pipeline.detect import _subpix_refine
        corners = _subpix_refine(img, corners)

    if diagnostic:
        return corners, diag
    return corners

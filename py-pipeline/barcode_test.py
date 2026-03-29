import sys
import cv2
import numpy as np
import zxingcpp


ZXING_FORMATS = zxingcpp.BarcodeFormat.EAN13 | zxingcpp.BarcodeFormat.UPCA


# --- Energy map & candidate finding ---

def compute_barcode_energy(gray, kernel_sizes=(21, 31, 41)):
    """Barcode detector using structure tensor coherence.
    Barcodes have parallel edges (high coherence), QR codes and text don't.
    Multi-scale: computes at several kernel sizes and takes the max,
    so both small and large barcodes produce strong energy."""
    # CLAHE first to boost contrast through plastic bags
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cl.apply(gray)

    sobel_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)

    combined = np.zeros(gray.shape, dtype=np.float32)
    for kernel_size in kernel_sizes:
        # Structure tensor components (blurred outer products of gradient)
        Jxx = cv2.blur(sobel_x * sobel_x, (kernel_size, kernel_size))
        Jyy = cv2.blur(sobel_y * sobel_y, (kernel_size, kernel_size))
        Jxy = cv2.blur(sobel_x * sobel_y, (kernel_size, kernel_size))

        # Eigenvalues of the 2x2 structure tensor
        trace = Jxx + Jyy
        diff = Jxx - Jyy
        disc = np.sqrt(diff * diff + 4 * Jxy * Jxy)
        lambda1 = (trace + disc) / 2
        lambda2 = (trace - disc) / 2

        # Coherence: (λ1 - λ2)² / (λ1 + λ2)²  → 1 for parallel edges, 0 for isotropic
        denom = (lambda1 + lambda2) ** 2
        coherence = np.where(denom > 1e-6, (lambda1 - lambda2) ** 2 / denom, 0)

        # Weight by edge strength, then square to push barcodes up and crush noise
        cv2.normalize(trace, trace, 0, 1, cv2.NORM_MINMAX)
        energy = (coherence * trace) ** 2
        np.maximum(combined, energy, out=combined)

    cv2.normalize(combined, combined, 0, 255, cv2.NORM_MINMAX)
    return combined.astype(np.uint8)


def find_candidates(energy, min_area=300, max_area=80000):
    """Threshold energy map and find candidate barcode regions.
    Uses mean + std of nonzero pixels as a dynamic threshold that adapts
    to each image's energy distribution."""
    nonzero = energy[energy > 0]
    if len(nonzero) > 100:
        # Dynamic threshold adapts to image energy distribution,
        # capped so it never exceeds the fixed baseline for easy images
        dynamic = nonzero.mean() + nonzero.std()
        fixed = energy.max() * 0.08
        thresh_val = min(dynamic, fixed)
    else:
        thresh_val = energy.max() * 0.08
    _, binary = cv2.threshold(energy, thresh_val, 255, cv2.THRESH_BINARY)

    # Close to merge barcode bar fragments into one blob
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (64, 64))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    # Open to kill small noise blobs (text, card edges)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boundingRect(cnt)
        candidates.append({
            'bbox': box,
            'min_area_rect': rect,
            'area': area,
            'contour': cnt,
        })

    return candidates, binary


# --- Decoding ---

def decode_zxing(image):
    """Run zxing on an image, return detections."""
    results = zxingcpp.read_barcodes(image, formats=ZXING_FORMATS)
    detections = []
    for r in results:
        pos = r.position
        pts = np.array([
            [pos.top_left.x, pos.top_left.y],
            [pos.top_right.x, pos.top_right.y],
            [pos.bottom_right.x, pos.bottom_right.y],
            [pos.bottom_left.x, pos.bottom_left.y],
        ], dtype=np.float64)
        cx, cy = pts.mean(axis=0)
        detections.append({
            'value': r.text,
            'type': str(r.format),
            'polygon': pts,
            'center': (cx, cy),
        })
    return detections


def estimate_bar_angle(gray_crop):
    """Estimate the angle needed to rotate barcode bars to vertical.
    The structure tensor dominant eigenvector is the gradient direction
    (perpendicular to bars). Bar direction = gradient_angle + 90°.
    Returns the rotation angle to apply to make bars vertical."""
    sx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = np.mean(sx * sx)
    Jyy = np.mean(sy * sy)
    Jxy = np.mean(sx * sy)
    # Dominant gradient direction (perpendicular to bars)
    grad_angle = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    # Bar direction = grad_angle + 90°
    # To make bars vertical (90°), rotate by: 90 - bar_direction = -grad_angle
    return -grad_angle


def _try_decode_crop(gray, candidate, pad, min_crop_size, offset=(0, 0)):
    """Try to decode a barcode from a padded crop around a candidate.
    Returns (detections_list, success_bool)."""
    h, w = gray.shape
    bx, by, bw, bh = candidate['bbox']
    dx, dy = offset

    # Pad the crop (with optional offset for jittering)
    x0 = max(0, bx - pad + dx)
    y0 = max(0, by - pad + dy)
    x1 = min(w, bx + bw + pad + dx)
    y1 = min(h, by + bh + pad + dy)
    crop = gray[y0:y1, x0:x1]

    # Upscale small crops
    crop_h, crop_w = crop.shape
    scale = 1.0
    if max(crop_w, crop_h) < min_crop_size:
        scale = min_crop_size / max(crop_w, crop_h)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Estimate rotation to make bars vertical — two independent estimates
    deskew_angle = estimate_bar_angle(crop)

    # minAreaRect angle as a second estimate (based on blob shape, not gradients)
    rect = candidate['min_area_rect']
    rect_angle = rect[2]  # OpenCV minAreaRect angle in [-90, 0)
    rw, rh = rect[1]
    # Make the long axis vertical: if wider than tall, rotate 90°
    if rw > rh:
        rect_angle += 90

    # Build angle search: deskew estimate with fine adjustments, then coarse sweep
    angles_to_try = [deskew_angle]
    for delta in [-15, -10, -5, 5, 10, 15]:
        angles_to_try.append(deskew_angle + delta)
    # Wider deltas to handle when structure tensor is confused by non-barcode edges
    for delta in [-60, -45, -30, 30, 45, 60]:
        angles_to_try.append(deskew_angle + delta)
    # minAreaRect-based estimate and its adjustments
    angles_to_try.append(rect_angle)
    for delta in [-15, -10, -5, 5, 10, 15]:
        angles_to_try.append(rect_angle + delta)
    # Also try no rotation and 90° offsets
    angles_to_try.extend([0, deskew_angle + 90, deskew_angle - 90])

    # CLAHE preprocessing
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop_clahe = cl.apply(crop)

    all_dets = []
    for rot_angle in angles_to_try:
        for img_variant in [crop, crop_clahe]:
            if abs(rot_angle) > 0.5:
                center = (img_variant.shape[1] // 2, img_variant.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
                rotated = cv2.warpAffine(img_variant, M, (img_variant.shape[1], img_variant.shape[0]),
                                         borderValue=255)
            else:
                rotated = img_variant

            dets = decode_zxing(rotated)
            for det in dets:
                pts = det['polygon']
                if abs(rot_angle) > 0.5:
                    M_inv = cv2.getRotationMatrix2D(center, -rot_angle, 1.0)
                    ones = np.ones((len(pts), 1))
                    pts_h = np.hstack([pts, ones])
                    pts = (M_inv @ pts_h.T).T
                pts = pts / scale + [x0, y0]
                det['polygon'] = pts.astype(np.float64)
                cx, cy = pts.mean(axis=0)
                det['center'] = (cx, cy)
                all_dets.append(det)

            if all_dets:
                return all_dets
        if all_dets:
            return all_dets

    return all_dets


def decode_candidate(gray, candidate, pad=150, min_crop_size=300):
    """Crop a candidate region, deskew using structure tensor, try decoding.
    If the primary crop fails, jitters the crop position to handle
    cases where small shifts in centering affect decode success."""
    # Primary attempt
    dets = _try_decode_crop(gray, candidate, pad, min_crop_size)
    if dets:
        return dets

    # Jitter: try shifted crop positions
    jitter = 20
    for dx, dy in [(-jitter, 0), (jitter, 0), (0, -jitter), (0, jitter),
                   (-jitter, -jitter), (jitter, -jitter), (-jitter, jitter), (jitter, jitter)]:
        dets = _try_decode_crop(gray, candidate, pad, min_crop_size, offset=(dx, dy))
        if dets:
            return dets

    return []


# --- Deduplication ---

def normalize_value(value):
    """EAN-13 with leading 0 is the same as UPC-A."""
    v = value.strip()
    if v.startswith('0') and len(v) == 13:
        v = v[1:]
    return v


def deduplicate_by_blob(candidates, all_detections_per_candidate):
    """One detection per candidate blob. Pick most common value if multiple."""
    results = []
    for i, dets in enumerate(all_detections_per_candidate):
        if not dets:
            continue
        # Vote on value
        values = [normalize_value(d['value']) for d in dets]
        from collections import Counter
        most_common = Counter(values).most_common(1)[0][0]
        # Pick the detection with that value (first one)
        best = next(d for d in dets if normalize_value(d['value']) == most_common)
        best['norm_value'] = most_common
        best['candidate_idx'] = i
        results.append(best)
    return results


# --- Fallback tiled scan ---

def tiled_fallback(gray, existing_centers, grid=(6, 6), overlap=0.3, radius=150):
    """Scan all tiles with rotation, dedup against existing detections."""
    
    return []

    h, w = gray.shape
    rows, cols = grid
    tile_h, tile_w = h // rows, w // cols
    pad_y, pad_x = int(tile_h * overlap), int(tile_w * overlap)

    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    new_dets = []
    all_centers = list(existing_centers)

    angles = [0, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25]

    for r in range(rows):
        for c in range(cols):
            y0 = max(0, r * tile_h - pad_y)
            y1 = min(h, (r + 1) * tile_h + pad_y)
            x0 = max(0, c * tile_w - pad_x)
            x1 = min(w, (c + 1) * tile_w + pad_x)

            tile = gray[y0:y1, x0:x1]
            found_in_tile = False

            for img in [tile, cl.apply(tile)]:
                if found_in_tile:
                    break
                for angle in angles:
                    if abs(angle) > 0.5:
                        center = (img.shape[1] // 2, img.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
                    else:
                        rotated = img

                    dets = decode_zxing(rotated)
                    for det in dets:
                        pts = det['polygon']
                        if abs(angle) > 0.5:
                            ones = np.ones((len(pts), 1))
                            pts_h = np.hstack([pts, ones])
                            pts = (M_inv @ pts_h.T).T
                        pts = pts + [x0, y0]
                        cx, cy = pts.mean(axis=0)

                        # Check if this is a new detection
                        is_dup = any(abs(cx - ex) < radius and abs(cy - ey) < radius
                                     for ex, ey in all_centers)
                        if not is_dup:
                            det['polygon'] = pts.astype(np.float64)
                            det['center'] = (cx, cy)
                            det['norm_value'] = normalize_value(det['value'])
                            new_dets.append(det)
                            all_centers.append((cx, cy))
                            found_in_tile = True

    # Dedup the fallback results
    unique = []
    all_centers = list(existing_centers)
    for det in new_dets:
        cx, cy = det['center']
        is_dup = any(abs(cx - ex) < radius and abs(cy - ey) < radius
                     for ex, ey in all_centers)
        if not is_dup:
            unique.append(det)
            all_centers.append((cx, cy))
    return unique


def spatial_dedup(detections, radius=200):
    """Final pass: merge detections that are spatially close."""
    unique = []
    for det in detections:
        cx, cy = det['center']
        is_dup = any(abs(cx - u['center'][0]) < radius and abs(cy - u['center'][1]) < radius
                     for u in unique)
        if not is_dup:
            unique.append(det)
    return unique


# --- Visualization ---

def draw_results(img, detections, candidates=None):
    vis = img.copy()
    # Draw candidate blobs in gray
    if candidates:
        for cand in candidates:
            bx, by, bw, bh = cand['bbox']
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255),
    ]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        cx, cy = int(det['center'][0]), int(det['center'][1])
        cv2.circle(vis, (cx, cy), 30, color, 4)
        cv2.putText(vis, f"#{i}", (cx + 35, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        pts = det['polygon'].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, color, 3)
    return vis


# --- Main ---

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'img/barcodes-test.jpg'
    img = cv2.imread(path)
    if img is None:
        print(f'Could not read image: {path}')
        return

    print(f'Image: {img.shape[1]}x{img.shape[0]}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Find candidate barcode regions
    energy = compute_barcode_energy(gray)
    candidates, mask = find_candidates(energy)
    print(f'Found {len(candidates)} candidate regions')

    # Save energy map and mask for debugging
    cv2.imwrite('debug/debug_barcode_energy.jpg', energy)
    cv2.imwrite('debug/debug_barcode_mask.jpg', mask)

    # Step 2: Decode each candidate
    all_dets_per_candidate = []
    for i, cand in enumerate(candidates):
        dets = decode_candidate(gray, cand)
        all_dets_per_candidate.append(dets)

    # Step 3: Deduplicate by blob, then spatial dedup for overlapping blobs
    results = deduplicate_by_blob(candidates, all_dets_per_candidate)
    results = spatial_dedup(results)
    print(f'Decoded {len(results)} barcodes from candidates')

    # Step 4: Fallback tiled scan for uncovered areas
    existing_centers = [(d['center'][0], d['center'][1]) for d in results]
    fallback = tiled_fallback(gray, existing_centers)
    if fallback:
        print(f'Fallback found {len(fallback)} additional barcodes')
        results.extend(fallback)

    print(f'\n=== Detected barcodes: {len(results)} ===')
    for i, det in enumerate(results):
        cx, cy = det['center']
        print(f'  #{i} [{det["type"]}] {det["norm_value"]}  @ ({cx:.0f}, {cy:.0f})')

    vis = draw_results(img, results, candidates)
    cv2.imwrite('debug/debug_barcodes.jpg', vis)
    print('Saved debug/debug_barcodes.jpg')


if __name__ == '__main__':
    main()

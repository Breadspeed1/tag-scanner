from .energy import compute_energy_map, energy_to_mask, find_qr_blobs
from .detect import detect_qr_in_aoi
from .homography import compute_sku_corners, extract_crop
from .grid import infer_grid, find_missing_positions
from .viz import draw_results


DEFAULT_CONFIG = {
    'energy_kernel': 31,
    'morph_kernel': 15,
    'percentile': 95,
    'min_area': 1000,
    'pad': 120,
    'min_crop_size': 400,
    'detector': 'aruco+standard',
    'preprocessing': 'none',
    'subpix': False,
    'min_module_size': 4.0,
    'max_colors_mismatch': 0.2,
    'sku_x': -1.5,
    'sku_y': 1.3,
    'sku_w': 2.5,
    'sku_h': 0.4,
    'sku_crop_w': 300,
    'sku_crop_h': 100,
}


def detect_qr_codes(img, config=None, debug=False):
    import cv2
    import numpy as np
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    energy = compute_energy_map(gray, kernel_size=cfg['energy_kernel'])
    mask = energy_to_mask(energy, morph_kernel_size=cfg['morph_kernel'], percentile=cfg['percentile'])
    boxes = find_qr_blobs(mask, min_area=cfg['min_area'])

    results = []
    for box in boxes:
        corners = detect_qr_in_aoi(
            img, box,
            pad=cfg['pad'],
            min_crop_size=cfg['min_crop_size'],
            detector=cfg['detector'],
            preprocessing=cfg['preprocessing'],
            subpix=cfg['subpix'],
            min_module_size=cfg['min_module_size'],
            max_colors_mismatch=cfg['max_colors_mismatch'],
        )
        if corners is None:
            continue

        sku_corners = compute_sku_corners(
            corners, cfg['sku_x'], cfg['sku_y'], cfg['sku_w'], cfg['sku_h']
        )
        sku_crop = extract_crop(img, sku_corners, cfg['sku_crop_w'], cfg['sku_crop_h'])

        results.append({
            'qr_corners': corners,
            'sku_corners': sku_corners,
            'sku_crop': sku_crop,
            'bbox': box,
        })

    # Grid inference pass: predict missing positions and retry
    if len(results) >= 4:
        det_centers = [r['qr_corners'].mean(axis=0) for r in results]
        grid_positions, H, assigned = infer_grid(det_centers)
        missing = find_missing_positions(grid_positions, assigned, det_centers)

        if missing:
            # Estimate typical QR size from detections
            qr_sizes = []
            for r in results:
                c = r['qr_corners']
                sides = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
                qr_sizes.append(np.mean(sides))
            avg_qr_size = np.median(qr_sizes)
            box_size = int(avg_qr_size * 1.5)

            img_h, img_w = img.shape[:2]
            for pos in missing:
                px, py = int(pos[0]), int(pos[1])
                # Skip if predicted position is outside the image
                if px < 0 or py < 0 or px >= img_w or py >= img_h:
                    continue
                # Create a synthetic box centered on predicted position
                bx = max(0, px - box_size // 2)
                by = max(0, py - box_size // 2)
                bw = min(box_size, img_w - bx)
                bh = min(box_size, img_h - by)
                if bw < 20 or bh < 20:
                    continue
                synth_box = (bx, by, bw, bh)
                corners = detect_qr_in_aoi(
                    img, synth_box,
                    pad=cfg['pad'],
                    min_crop_size=cfg['min_crop_size'],
                    detector=cfg['detector'],
                    preprocessing=cfg['preprocessing'],
                    subpix=cfg['subpix'],
                    min_module_size=max(2.0, cfg['min_module_size'] - 1),
                    max_colors_mismatch=min(0.4, cfg['max_colors_mismatch'] + 0.1),
                )
                if corners is None:
                    continue

                # Check it's not a duplicate
                new_center = corners.mean(axis=0)
                duplicate = False
                for r in results:
                    if np.linalg.norm(r['qr_corners'].mean(axis=0) - new_center) < 50:
                        duplicate = True
                        break
                if duplicate:
                    continue

                sku_corners = compute_sku_corners(
                    corners, cfg['sku_x'], cfg['sku_y'], cfg['sku_w'], cfg['sku_h']
                )
                sku_crop = extract_crop(img, sku_corners, cfg['sku_crop_w'], cfg['sku_crop_h'])

                results.append({
                    'qr_corners': corners,
                    'sku_corners': sku_corners,
                    'sku_crop': sku_crop,
                    'bbox': synth_box,
                })

    if debug:
        return results, energy, mask
    return results

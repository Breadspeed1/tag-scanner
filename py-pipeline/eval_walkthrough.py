"""
Comprehensive pipeline walkthrough — emits annotated images for every stage.

Output structure (in debug/walkthrough/<image_name>/):
  01_energy.jpg          — energy map
  02_mask.jpg            — binary mask after thresholding + morphology
  03_blobs.jpg           — all blobs drawn on image
  04_detections.jpg      — initial detections (before grid inference)
  05_grid.jpg            — grid inference: detected (green), predicted missing (red circles)
  06_final.jpg           — final result vs GT

  aoi_NNN/
    crop.jpg             — the AOI crop fed to detectors
    att_1_opencv.jpg     — attempt 1 result (green quad if found, red X if not)
    att_2_wechat.jpg     — attempt 2 ...
    att_3_clahe.jpg      — attempt 3 ...
    att_4_finder.jpg     — attempt 4 ...
    refine.jpg           — corners before (yellow) and after (green) refinement
    result.jpg           — final detection mapped back to full image context
"""

import json
import os
import sys
import cv2
import numpy as np

from pipeline import DEFAULT_CONFIG
from pipeline.energy import compute_energy_map, energy_to_mask, find_qr_blobs
from pipeline.detect import detect_qr_in_aoi, _quad_squareness
from pipeline.homography import compute_sku_corners, extract_crop
from pipeline.grid import infer_grid, find_missing_positions


IMG_DIR = 'img'
JSON_PREFIX = 'qrs_dataset 2026-03-25 23-20-04_'
MATCH_DIST = 100

IMAGES = [
    'large_clear.jpg',
    'large_semiclear.jpg',
    'large_semiclear2.jpg',
    'large_semiclear3.jpg',
    'large_unclear.jpg',
]


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)
    quads = []
    for obj in data['annotation']['objects']:
        pts = np.array(obj['points']['exterior'], dtype=np.float32)
        if pts.shape == (4, 2):
            quads.append(pts)
    return quads


def quad_center(quad):
    return quad.mean(axis=0)


def match_detections(gt_quads, det_corners_list, max_dist=MATCH_DIST):
    gt_centers = [quad_center(q) for q in gt_quads]
    det_centers = [quad_center(c) for c in det_corners_list]

    matched_gt = set()
    matched_det = set()
    pairs = []
    for i, gc in enumerate(gt_centers):
        for j, dc in enumerate(det_centers):
            dist = np.linalg.norm(gc - dc)
            if dist < max_dist:
                pairs.append((dist, i, j))
    pairs.sort()
    for dist, i, j in pairs:
        if i in matched_gt or j in matched_det:
            continue
        matched_gt.add(i)
        matched_det.add(j)

    missed = [i for i in range(len(gt_quads)) if i not in matched_gt]
    fps = [j for j in range(len(det_corners_list)) if j not in matched_det]
    return list(matched_gt), list(matched_det), missed, fps


def draw_quad(img, quad, color, thickness=2):
    pts = quad.astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


def draw_quad_on_crop(crop, quad, color, thickness=2, label=None):
    vis = crop.copy() if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if quad is not None:
        pts = quad.astype(np.int32)
        cv2.polylines(vis, [pts], True, color, thickness)
        for i, pt in enumerate(pts):
            cv2.circle(vis, tuple(pt), 4, color, -1)
            cv2.putText(vis, str(i), tuple(pt + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        h, w = vis.shape[:2]
        cv2.line(vis, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)
        cv2.line(vis, (w - 10, 10), (10, h - 10), (0, 0, 255), 3)
    if label:
        cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def process_image(img_name, out_dir, cfg):
    img_path = os.path.join(IMG_DIR, img_name)
    json_path = os.path.join(IMG_DIR, f'{JSON_PREFIX}{img_name}.json')

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        print(f'  SKIPPED (missing file)')
        return None

    img = cv2.imread(img_path)
    gt_quads = load_gt(json_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    os.makedirs(out_dir, exist_ok=True)

    # --- Stage 1: Energy map ---
    energy = compute_energy_map(gray, kernel_size=cfg['energy_kernel'])
    energy_color = cv2.applyColorMap(energy, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, '01_energy.jpg'), energy_color)

    # --- Stage 2: Binary mask ---
    mask = energy_to_mask(energy, morph_kernel_size=cfg['morph_kernel'], percentile=cfg['percentile'])
    cv2.imwrite(os.path.join(out_dir, '02_mask.jpg'), mask)

    # --- Stage 3: Blobs ---
    boxes = find_qr_blobs(mask, min_area=cfg['min_area'])
    vis_blobs = img.copy()
    for i, (bx, by, bw, bh) in enumerate(boxes):
        cv2.rectangle(vis_blobs, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
        cv2.putText(vis_blobs, str(i), (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    # Also draw GT for reference
    for q in gt_quads:
        draw_quad(vis_blobs, q, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, '03_blobs.jpg'), vis_blobs)

    # --- Stage 4: Per-AOI detection with diagnostics ---
    initial_results = []  # (corners, diag, box_idx)
    failed_aois = []      # (diag, box_idx)

    for i, box in enumerate(boxes):
        result = detect_qr_in_aoi(
            img, box,
            pad=cfg['pad'],
            min_crop_size=cfg['min_crop_size'],
            detector=cfg['detector'],
            preprocessing=cfg['preprocessing'],
            subpix=cfg['subpix'],
            min_module_size=cfg['min_module_size'],
            max_colors_mismatch=cfg['max_colors_mismatch'],
            diagnostic=True,
        )

        corners, diag = result

        aoi_dir = os.path.join(out_dir, f'aoi_{i:03d}')
        os.makedirs(aoi_dir, exist_ok=True)

        # Save crop
        cv2.imwrite(os.path.join(aoi_dir, 'crop.jpg'), diag['crop'])

        # Save each attempt
        for j, att in enumerate(diag['attempts']):
            label = att['name']
            found = att['quad'] is not None
            color = (0, 255, 0) if found else (0, 0, 255)
            status = 'FOUND' if found else 'MISS'
            vis_att = draw_quad_on_crop(att['input'], att['quad'], color,
                                        label=f"{label}: {status}")
            cv2.imwrite(os.path.join(aoi_dir, f'att_{j+1}_{att["name"].replace(" ", "_").replace("(", "").replace(")", "")}.jpg'), vis_att)

        # Save refinement before/after
        if diag['corners_before_refine'] is not None and diag['corners_after_refine'] is not None:
            vis_refine = diag['crop'].copy()
            if len(vis_refine.shape) == 2:
                vis_refine = cv2.cvtColor(vis_refine, cv2.COLOR_GRAY2BGR)
            # Before in yellow
            draw_quad(vis_refine, diag['corners_before_refine'], (0, 255, 255), 2)
            # After in green
            draw_quad(vis_refine, diag['corners_after_refine'], (0, 255, 0), 2)
            cv2.putText(vis_refine, 'yellow=before  green=after', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(aoi_dir, 'refine.jpg'), vis_refine)

        # Save dewarp info
        if diag['dewarp'] is not None:
            dw = diag['dewarp']
            vis_dw = diag['crop'].copy()
            if len(vis_dw.shape) == 2:
                vis_dw = cv2.cvtColor(vis_dw, cv2.COLOR_GRAY2BGR)
            draw_quad(vis_dw, dw['before'], (0, 255, 255), 2)
            draw_quad(vis_dw, dw['after'], (0, 255, 0), 2)
            cv2.putText(vis_dw, f'dewarp sq={dw["squareness"]:.2f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(aoi_dir, 'dewarp.jpg'), vis_dw)

        # Save squareness filter info
        if diag['squareness_filter'] is not None:
            vis_sq = diag['crop'].copy()
            if len(vis_sq.shape) == 2:
                vis_sq = cv2.cvtColor(vis_sq, cv2.COLOR_GRAY2BGR)
            # Draw the quad that was rejected
            for att in diag['attempts']:
                if att['quad'] is not None:
                    draw_quad(vis_sq, att['quad'], (0, 0, 255), 2)
                    break
            cv2.putText(vis_sq, f'REJECTED sq={diag["squareness_filter"]:.3f} < {0.5}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(aoi_dir, 'rejected_squareness.jpg'), vis_sq)

        # Save outcome summary
        if corners is not None:
            # Draw result in context: crop with final quad
            vis_result = diag['crop'].copy()
            if len(vis_result.shape) == 2:
                vis_result = cv2.cvtColor(vis_result, cv2.COLOR_GRAY2BGR)
            # Map final corners back to crop space
            rx_d, ry_d = diag['roi'][0], diag['roi'][1]
            crop_corners = corners.copy()
            crop_corners[:, 0] = (crop_corners[:, 0] - rx_d) * diag['scale']
            crop_corners[:, 1] = (crop_corners[:, 1] - ry_d) * diag['scale']
            draw_quad(vis_result, crop_corners, (0, 255, 0), 2)
            cv2.putText(vis_result, f'DETECTED via {diag["winner"]}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(aoi_dir, 'result.jpg'), vis_result)

            initial_results.append((corners, diag, i))
        else:
            failed_aois.append((diag, i))

    # --- Stage 4b: Initial detections overview ---
    vis_det = img.copy()
    for corners, diag, idx in initial_results:
        draw_quad(vis_det, corners, (0, 255, 0), 2)
        c = quad_center(corners).astype(int)
        cv2.putText(vis_det, f'{idx}:{diag["winner"][:3]}', tuple(c - [0, 10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for diag, idx in failed_aois:
        rx, ry, rw, rh = diag['roi']
        cv2.rectangle(vis_det, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
        cv2.putText(vis_det, f'{idx}:fail', (rx, ry - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # Draw GT
    for q in gt_quads:
        draw_quad(vis_det, q, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, '04_detections.jpg'), vis_det)

    # --- Stage 5: Grid inference ---
    all_corners = [c for c, _, _ in initial_results]
    grid_detected_from = ['initial'] * len(all_corners)

    if len(all_corners) >= 4:
        det_centers = [quad_center(c) for c in all_corners]
        grid_positions, H, assigned = infer_grid(det_centers)
        missing = find_missing_positions(grid_positions, assigned, det_centers)

        vis_grid = img.copy()
        # Draw all detections
        for c in all_corners:
            draw_quad(vis_grid, c, (0, 255, 0), 2)
        # Draw grid cells
        if grid_positions:
            for (r, col_idx), pos in grid_positions.items():
                px, py = int(pos[0]), int(pos[1])
                if (r, col_idx) in assigned:
                    cv2.circle(vis_grid, (px, py), 8, (0, 255, 0), 2)
                else:
                    cv2.circle(vis_grid, (px, py), 12, (0, 0, 255), 3)
                    cv2.putText(vis_grid, f'{r},{col_idx}', (px + 15, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # Draw GT
        for q in gt_quads:
            draw_quad(vis_grid, q, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, '05_grid.jpg'), vis_grid)

        # Try to detect at missing positions
        if missing:
            qr_sizes = []
            for c in all_corners:
                sides = [np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]
                qr_sizes.append(np.mean(sides))
            avg_qr_size = np.median(qr_sizes)
            box_size = int(avg_qr_size * 1.5)

            img_h, img_w = img.shape[:2]
            grid_dir = os.path.join(out_dir, 'grid_attempts')
            os.makedirs(grid_dir, exist_ok=True)

            for mi, pos in enumerate(missing):
                px, py = int(pos[0]), int(pos[1])
                if px < 0 or py < 0 or px >= img_w or py >= img_h:
                    continue
                bx = max(0, px - box_size // 2)
                by = max(0, py - box_size // 2)
                bw = min(box_size, img_w - bx)
                bh = min(box_size, img_h - by)
                if bw < 20 or bh < 20:
                    continue
                synth_box = (bx, by, bw, bh)

                result = detect_qr_in_aoi(
                    img, synth_box,
                    pad=cfg['pad'],
                    min_crop_size=cfg['min_crop_size'],
                    detector=cfg['detector'],
                    preprocessing=cfg['preprocessing'],
                    subpix=cfg['subpix'],
                    min_module_size=max(2.0, cfg['min_module_size'] - 1),
                    max_colors_mismatch=min(0.4, cfg['max_colors_mismatch'] + 0.1),
                    diagnostic=True,
                )
                corners, diag = result

                # Save grid attempt crop + result
                vis_ga = diag['crop'].copy()
                if len(vis_ga.shape) == 2:
                    vis_ga = cv2.cvtColor(vis_ga, cv2.COLOR_GRAY2BGR)
                if corners is not None:
                    crop_corners = corners.copy()
                    rx_d, ry_d = diag['roi'][0], diag['roi'][1]
                    crop_corners[:, 0] = (crop_corners[:, 0] - rx_d) * diag['scale']
                    crop_corners[:, 1] = (crop_corners[:, 1] - ry_d) * diag['scale']
                    draw_quad(vis_ga, crop_corners, (0, 255, 0), 2)
                    cv2.putText(vis_ga, f'grid_{mi}: FOUND via {diag["winner"]}', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_ga, f'grid_{mi}: MISS', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(grid_dir, f'grid_{mi:03d}.jpg'), vis_ga)

                if corners is not None:
                    new_center = quad_center(corners)
                    duplicate = False
                    for c in all_corners:
                        if np.linalg.norm(quad_center(c) - new_center) < 50:
                            duplicate = True
                            break
                    if not duplicate:
                        all_corners.append(corners)
                        grid_detected_from.append('grid')
    else:
        vis_grid = img.copy()
        for c in all_corners:
            draw_quad(vis_grid, c, (0, 255, 0), 2)
        cv2.putText(vis_grid, 'Grid inference skipped (<4 detections)', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, '05_grid.jpg'), vis_grid)

    # --- Stage 6: Final result vs GT ---
    matched_gt, matched_det, missed_gt, fp_det = match_detections(gt_quads, all_corners)

    vis_final = img.copy()
    # Matched GT in thin yellow
    for i in matched_gt:
        draw_quad(vis_final, gt_quads[i], (0, 255, 255), 1)
    # Missed GT in red with X marker
    for i in missed_gt:
        draw_quad(vis_final, gt_quads[i], (0, 0, 255), 3)
        c = quad_center(gt_quads[i]).astype(int)
        cv2.drawMarker(vis_final, tuple(c), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    # Matched detections in green, labeled with source
    for j in matched_det:
        color = (0, 255, 0) if grid_detected_from[j] == 'initial' else (255, 255, 0)
        draw_quad(vis_final, all_corners[j], color, 2)
        c = quad_center(all_corners[j]).astype(int)
        label = 'G' if grid_detected_from[j] == 'grid' else ''
        if label:
            cv2.putText(vis_final, label, tuple(c + [10, -10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # FP in magenta
    for j in fp_det:
        draw_quad(vis_final, all_corners[j], (255, 0, 255), 3)
        c = quad_center(all_corners[j]).astype(int)
        cv2.drawMarker(vis_final, tuple(c), (255, 0, 255), cv2.MARKER_CROSS, 40, 3)

    n_gt = len(gt_quads)
    n_match = len(matched_gt)
    n_miss = len(missed_gt)
    n_fp = len(fp_det)
    recall = n_match / n_gt if n_gt > 0 else 0

    cv2.putText(vis_final,
                f'GT={n_gt} Match={n_match} Miss={n_miss} FP={n_fp} Recall={recall:.0%}',
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    cv2.putText(vis_final,
                f'GT={n_gt} Match={n_match} Miss={n_miss} FP={n_fp} Recall={recall:.0%}',
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    cv2.imwrite(os.path.join(out_dir, '06_final.jpg'), vis_final)

    return {
        'gt': n_gt, 'det': len(all_corners), 'match': n_match,
        'miss': n_miss, 'fp': n_fp, 'recall': recall,
    }


def main():
    base_dir = 'debug/walkthrough'
    os.makedirs(base_dir, exist_ok=True)

    cfg = {**DEFAULT_CONFIG}

    # Optional: only run specific image
    only = None
    if len(sys.argv) > 1:
        only = sys.argv[1]

    total = {'gt': 0, 'det': 0, 'match': 0, 'miss': 0, 'fp': 0}

    print(f"{'Image':<25} {'GT':>4} {'Det':>4} {'Match':>5} {'Miss':>5} {'FP':>4} {'Recall':>7}")
    print('-' * 60)

    for img_name in IMAGES:
        if only and only not in img_name:
            continue
        base = os.path.splitext(img_name)[0]
        out_dir = os.path.join(base_dir, base)
        print(f'{img_name:<25}', end=' ', flush=True)
        stats = process_image(img_name, out_dir, cfg)
        if stats is None:
            print('SKIPPED')
            continue
        for k in total:
            total[k] += stats[k]
        print(f'{stats["gt"]:>4} {stats["det"]:>4} {stats["match"]:>5} {stats["miss"]:>5} {stats["fp"]:>4} {stats["recall"]:>6.1%}')

    print('-' * 60)
    total_recall = total['match'] / total['gt'] if total['gt'] > 0 else 0
    total_prec = total['match'] / total['det'] if total['det'] > 0 else 0
    print(f'{"TOTAL":<25} {total["gt"]:>4} {total["det"]:>4} {total["match"]:>5} {total["miss"]:>5} {total["fp"]:>4} {total_recall:>6.1%}')
    print(f'Precision: {total_prec:.1%}')
    print(f'\nOutput: {base_dir}/')


if __name__ == '__main__':
    main()

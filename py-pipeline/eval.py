import json
import os
import sys
import cv2
import numpy as np
from pipeline import detect_qr_codes, draw_results


IMG_DIR = 'img'
JSON_PREFIX = 'qrs_dataset 2026-03-25 23-20-04_'
MATCH_DIST = 100  # max center distance to count as a match

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


def match_detections(gt_quads, det_results, max_dist=MATCH_DIST):
    gt_centers = [quad_center(q) for q in gt_quads]
    det_centers = [quad_center(r['qr_corners']) for r in det_results]

    matched_gt = set()
    matched_det = set()

    # Greedy matching by closest pairs
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
    fps = [j for j in range(len(det_results)) if j not in matched_det]

    return list(matched_gt), list(matched_det), missed, fps


def draw_eval(img, gt_quads, det_results, matched_gt, matched_det, missed_gt, fp_det):
    vis = img.copy()

    # Matched GT in green
    for i in matched_gt:
        pts = gt_quads[i].astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    # Missed GT in red
    for i in missed_gt:
        pts = gt_quads[i].astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
        c = quad_center(gt_quads[i]).astype(int)
        cv2.drawMarker(vis, tuple(c), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)

    # Matched detections in cyan
    for j in matched_det:
        pts = det_results[j]['qr_corners'].astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 0), 2)

    # FP detections in magenta
    for j in fp_det:
        pts = det_results[j]['qr_corners'].astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 0, 255), 3)
        c = quad_center(det_results[j]['qr_corners']).astype(int)
        cv2.drawMarker(vis, tuple(c), (255, 0, 255), cv2.MARKER_CROSS, 40, 3)

    return vis


def main():
    os.makedirs('debug', exist_ok=True)

    config = None
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        config = json.loads(sys.argv[2])

    total_gt = 0
    total_det = 0
    total_matched = 0
    total_missed = 0
    total_fp = 0

    print(f"{'Image':<25} {'GT':>4} {'Det':>4} {'Match':>5} {'Miss':>5} {'FP':>4} {'Recall':>7}")
    print('-' * 60)

    for img_name in IMAGES:
        img_path = os.path.join(IMG_DIR, img_name)
        json_path = os.path.join(IMG_DIR, f'{JSON_PREFIX}{img_name}.json')

        if not os.path.exists(img_path) or not os.path.exists(json_path):
            print(f'{img_name:<25} SKIPPED (missing file)')
            continue

        img = cv2.imread(img_path)
        gt_quads = load_gt(json_path)
        results = detect_qr_codes(img, config=config)

        matched_gt, matched_det, missed, fps = match_detections(gt_quads, results)

        n_gt = len(gt_quads)
        n_det = len(results)
        n_matched = len(matched_gt)
        n_missed = len(missed)
        n_fp = len(fps)
        recall = n_matched / n_gt if n_gt > 0 else 0

        total_gt += n_gt
        total_det += n_det
        total_matched += n_matched
        total_missed += n_missed
        total_fp += n_fp

        print(f'{img_name:<25} {n_gt:>4} {n_det:>4} {n_matched:>5} {n_missed:>5} {n_fp:>4} {recall:>6.1%}')

        vis = draw_eval(img, gt_quads, results, matched_gt, matched_det, missed, fps)
        base = os.path.splitext(img_name)[0]
        cv2.imwrite(f'debug/eval_{base}.jpg', vis)

    print('-' * 60)
    total_recall = total_matched / total_gt if total_gt > 0 else 0
    total_prec = total_matched / total_det if total_det > 0 else 0
    print(f'{"TOTAL":<25} {total_gt:>4} {total_det:>4} {total_matched:>5} {total_missed:>5} {total_fp:>4} {total_recall:>6.1%}')
    print(f'Precision: {total_prec:.1%}')


if __name__ == '__main__':
    main()

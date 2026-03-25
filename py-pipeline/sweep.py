import csv
import hashlib
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from pipeline import DEFAULT_CONFIG, detect_qr_codes
from pipeline.viz import draw_results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def quad_squareness(corners):
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return min(sides) / max(sides) if max(sides) > 0 else 0


def compute_metrics(results):
    n = len(results)
    if n == 0:
        return {'n_detections': 0, 'mean_squareness': 0.0}
    sq = [quad_squareness(r['qr_corners']) for r in results]
    return {
        'n_detections': n,
        'mean_squareness': float(np.mean(sq)),
    }


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_config(img_path, config, save_dir=None):
    img = cv2.imread(img_path)
    if img is None:
        return None

    results = detect_qr_codes(img, config=config)
    metrics = compute_metrics(results)

    if save_dir is not None:
        vis = draw_results(img, results)
        os.makedirs(save_dir, exist_ok=True)
        cfg_hash = config_hash(config)
        img_stem = Path(img_path).stem
        out_path = os.path.join(save_dir, f'{img_stem}_{cfg_hash}.jpg')
        cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        metrics['detection_image'] = out_path

    return metrics


def config_hash(config):
    s = str(sorted(config.items()))
    return hashlib.md5(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _worker(args):
    img_path, config, save_dir, config_id = args
    try:
        metrics = evaluate_config(img_path, config, save_dir)
        if metrics is None:
            return None
        return {
            'config_id': config_id,
            'image': Path(img_path).name,
            **config,
            **metrics,
        }
    except Exception as e:
        return {
            'config_id': config_id,
            'image': Path(img_path).name,
            'error': str(e),
            **config,
            'n_detections': -1,
            'mean_squareness': 0.0,
        }


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

ENERGY_GRID = {
    'energy_kernel': [31, 41, 51, 61, 81, 101],
}

MASK_GRID = {
    'percentile': [93, 95, 97, 98, 99],
    'morph_kernel': [7, 11, 15, 21],
}

BLOB_GRID = {
    'min_area': [1000, 2000, 3000, 5000],
}

AOI_GRID = {
    'pad': [60, 80, 100, 120, 160],
    'min_crop_size': [200, 300, 400, 600],
}

DETECT_GRID = {
    'detector': ['aruco', 'standard', 'aruco+standard'],
    'preprocessing': ['none', 'sharpen', 'unsharp', 'clahe'],
    'subpix': [True, False],
    'min_module_size': [2.0, 3.0, 4.0, 6.0],
    'max_colors_mismatch': [0.1, 0.2, 0.3],
}


def expand_grid(grid):
    keys = list(grid.keys())
    vals = list(grid.values())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def build_configs_per_stage(base):
    configs = []

    # Phase A: sweep each stage independently
    for grid, label in [
        (ENERGY_GRID, 'energy'),
        (MASK_GRID, 'mask'),
        (BLOB_GRID, 'blob'),
        (AOI_GRID, 'aoi'),
        (DETECT_GRID, 'detect'),
    ]:
        for overrides in expand_grid(grid):
            cfg = {**base, **overrides}
            configs.append((cfg, f'A_{label}'))

    return configs


def build_full_cartesian(base):
    # Split into two groups to keep total manageable:
    # Group 1: energy/mask/blob params (cartesian within group)
    preprocess_grid = {**ENERGY_GRID, **MASK_GRID, **BLOB_GRID}
    # Group 2: AOI/detection params (cartesian within group)
    detect_grid = {**AOI_GRID, **DETECT_GRID}

    configs = []
    for overrides in expand_grid(preprocess_grid):
        cfg = {**base, **overrides}
        configs.append((cfg, 'B_preprocess'))

    for overrides in expand_grid(detect_grid):
        cfg = {**base, **overrides}
        configs.append((cfg, 'B_detect'))

    return configs


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

def run_sweep(image_paths, output_dir='sweep_results', max_workers=None, full=True):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    base = dict(DEFAULT_CONFIG)

    if full:
        config_list = build_full_cartesian(base)
    else:
        config_list = build_configs_per_stage(base)

    # Build work items: (image, config) pairs
    work = []
    for idx, (cfg, phase) in enumerate(config_list):
        cfg_id = f'{phase}_{idx:06d}'
        for img_path in image_paths:
            work.append((img_path, cfg, img_dir, cfg_id))

    total = len(work)
    n_configs = len(config_list)
    n_images = len(image_paths)
    print(f'Sweep: {n_configs} configs x {n_images} images = {total} evaluations')
    print(f'Output: {output_dir}/')

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 32)
    print(f'Workers: {max_workers}')

    csv_path = os.path.join(output_dir, 'results.csv')
    fieldnames = None
    results_written = 0
    start_time = time.time()

    with open(csv_path, 'w', newline='') as f:
        writer = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, w): i for i, w in enumerate(work)}

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                if writer is None:
                    fieldnames = list(result.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()

                writer.writerow(result)
                results_written += 1

                # Progress
                elapsed = time.time() - start_time
                rate = results_written / elapsed if elapsed > 0 else 0
                eta = (total - results_written) / rate if rate > 0 else 0
                pct = results_written / total * 100
                print(
                    f'\r  [{results_written}/{total}] {pct:.1f}%  '
                    f'{rate:.1f} eval/s  ETA {eta:.0f}s  ',
                    end='', flush=True,
                )

    elapsed = time.time() - start_time
    print(f'\nDone: {results_written} results in {elapsed:.1f}s')
    print(f'CSV: {csv_path}')
    print(f'Images: {img_dir}/')

    return csv_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline parameter sweep')
    parser.add_argument('--images', nargs='*', default=None,
                        help='Image paths (default: all in img/)')
    parser.add_argument('--output', default='sweep_results',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                        help='Max parallel workers')
    parser.add_argument('--per-stage', action='store_true',
                        help='Only sweep per-stage (not full cartesian)')
    args = parser.parse_args()

    if args.images:
        paths = args.images
    else:
        paths = sorted(glob('img/*.jpg') + glob('img/*.png'))

    if not paths:
        print('No images found')
        sys.exit(1)

    print(f'Images: {paths}')
    run_sweep(paths, output_dir=args.output, max_workers=args.workers, full=not args.per_stage)

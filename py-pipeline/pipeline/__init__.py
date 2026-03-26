from .energy import compute_energy_map, energy_to_mask, find_qr_blobs
from .detect import detect_qr_in_aoi
from .homography import compute_sku_corners, extract_crop
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

    if debug:
        return results, energy, mask
    return results

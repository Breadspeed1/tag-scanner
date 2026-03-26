import cv2
import numpy as np


def compute_sku_corners(qr_corners, sku_x, sku_y, sku_w, sku_h):
    # QR code occupies unit square [0,0] -> [1,1]
    qr_unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    M = cv2.getPerspectiveTransform(qr_unit, qr_corners.astype(np.float32))

    sku_unit = np.float32([
        [sku_x, sku_y],
        [sku_x + sku_w, sku_y],
        [sku_x + sku_w, sku_y + sku_h],
        [sku_x, sku_y + sku_h],
    ])

    sku_img = cv2.perspectiveTransform(sku_unit.reshape(1, -1, 2), M)
    return sku_img.reshape(4, 2)


def extract_crop(img, sku_corners, crop_w=300, crop_h=100, pad_frac=0.1):
    # Expand corners outward by pad_frac to be less sensitive to corner errors
    if pad_frac > 0:
        center = sku_corners.mean(axis=0)
        sku_corners = center + (sku_corners - center) * (1 + pad_frac)

    dst = np.float32([
        [0, 0],
        [crop_w, 0],
        [crop_w, crop_h],
        [0, crop_h],
    ])

    M = cv2.getPerspectiveTransform(sku_corners.astype(np.float32), dst)
    crop = cv2.warpPerspective(img, M, (crop_w, crop_h))

    return crop

export const DETECTION = {
    akaze_threshold: 0.001,
    lowe_ratio: 0.75,

    dbscan_eps: 80,
    dbscan_min_samples: 8,

    ransac_reproj_thresh: 5.0,
    min_inliers: 10,
}

export const CARD_LAYOUT = {
    // SKU bounding box in QR-side-length units.
    // QR code occupies [0,0] → [1,1].
    sku_x: 0.0,
    sku_y: 1.1,
    sku_w: 1.0,
    sku_h: 0.4,

    // Output crop resolution (pixels).
    crop_width: 300,
    crop_height: 100,
};

export const OCR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
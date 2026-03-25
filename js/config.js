export const CARD_LAYOUT = {
    // SKU bounding box relative to the QR code's detected corners.
    // QR occupies [0,0]→[1,1] where the unit is the QR side length.
    // Corners from detectMulti: [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left
    //
    // Measure from a physical card: how far is the SKU text from the QR,
    // expressed as multiples of the QR side length?
    sku_x: -1.5,    // left edge relative to QR left
    sku_y: 1.4,    // top edge below QR (1.1 = just below the code)
    sku_w: 2.5,    // width in QR-side-lengths
    sku_h: 0.3,    // height in QR-side-lengths

    // Output crop resolution in pixels.
    crop_width: 300,
    crop_height: 100,
};

// Display canvas is downscaled to this (pixels on longest side).
export const MAX_SCENE_DIM = 4096;

// Detection runs on a separate canvas at up to this resolution.
// 4096 is effectively full-res for a 3072×4080 phone photo.
export const MAX_DETECT_DIM = 4096;

// Default preprocessing parameters for the tuning panel.
export const DETECTION_PARAMS = {
    tileSize: 850,   // px per tile side; smaller = more tiles, fewer QRs per tile
    upscale: 1.0,   // upscale each tile before detection (1 = off)
    contrast: 1.0,   // convertScaleAbs alpha (0.5–3)
    brightness: -35,    // additive brightness offset (-128..128)
    sharpen: 0.1,   // unsharp-mask strength (0 = off)
    blockSize: 23,   // adaptiveThreshold neighborhood (odd) — scales with image resolution
    cVal: 9,     // adaptiveThreshold C constant (subtract from local mean)
};

export const OCR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-';

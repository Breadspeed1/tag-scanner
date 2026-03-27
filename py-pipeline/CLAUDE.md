# QR Code Detection Pipeline

## Project Overview

Computer vision pipeline that scans photos of grids of packaged products. Each product has a card inside a plastic bag with a QR code (used only as a localization anchor — contains no useful data) and a SKU/barcode region. The pipeline detects QR codes, extracts precise corner quads, then uses perspective homography to crop the nearby SKU text region for OCR.

## Architecture

```
pipeline/
  energy.py      — Blob detection: bidirectional Sobel energy map → threshold → morphology → contours
  detect.py      — QR detection: runs all detectors on each AOI crop, picks best by squareness
  homography.py  — Perspective math: QR corners → unit square homography → SKU region extraction
  grid.py        — Grid inference: clusters detections into rows/cols, predicts missing positions
  viz.py         — Debug visualization
  __init__.py    — Orchestration: energy → blobs → detect per blob → grid inference → results
```

## Key Design Decisions

- **Energy map uses geometric mean of Sobel_x² and Sobel_y²** — finds 2D texture (QR codes) while suppressing 1D texture (barcodes) and uniform glare
- **All detectors run on every AOI** (aruco, standard, wechat, finder patterns × original + CLAHE preprocessing), best squareness wins — finder patterns win ~60-75% of the time because geometric centers are very precise
- **Corner refinement**: adaptive threshold → outer contour → assign edge pixels to quad sides → median-per-bin (resists glare outliers) → fitLine per side → intersect adjacent lines
- **Refinement snaps to the white→gray border** (outer edge of white QR background), NOT the QR module boundary
- **Grid inference** uses nearest-neighbor distances for spacing, k-means on pairwise angles for grid directions, homography from grid indices to image positions

## Ground Truth

- Annotations in `img/*.json` (Supervisely format): `data['annotation']['objects'][i]['points']['exterior']`
- JSON filename pattern: `qrs_dataset 2026-03-25 23-20-04_<image>.json`
- **GT outlines the QR module area**, not the outer white border — so detected corners will be systematically slightly outside GT quads
- **GT corners are approximate** (hand-annotated with subpixel inference) — don't use GT squareness as a quality filter

## Running

```bash
uv run python main.py [image_path]         # Interactive viewer
uv run python eval.py                       # Quick recall/precision table
uv run python eval_walkthrough.py [filter]  # Full per-stage annotated images in debug/walkthrough/
```

## Config

All tunable parameters are in `DEFAULT_CONFIG` in `pipeline/__init__.py`. Key ones:
- `energy_kernel` (31), `morph_kernel` (15), `percentile` (95) — blob detection sensitivity
- `pad` (120), `min_crop_size` (400) — AOI crop sizing
- `min_module_size` (4.0), `max_colors_mismatch` (0.2) — ArUco detector params
- `sku_x/y/w/h` — SKU region position relative to QR code in QR-width units

## Known Issues

- Corner refinement can be pulled outward by glare (contour follows glare-brightened region)
- `large_unclear` back rows (~30 codes) are too small/distant for any detector — grid inference helps but can't fully recover
- Finder pattern detector produces false positives on non-QR nested contours occasionally

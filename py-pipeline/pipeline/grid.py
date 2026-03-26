import cv2
import numpy as np
from itertools import combinations


def _nearest_neighbor_dists(centers):
    """Compute nearest-neighbor distance for each center."""
    nn_dists = []
    for i in range(len(centers)):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = float('inf')
        nn_dists.append(dists.min())
    return np.array(nn_dists)


def _cluster_1d(values, expected_spacing):
    """Cluster 1D values into groups using expected grid spacing."""
    if len(values) == 0:
        return []
    values = np.sort(values)

    clusters = [[values[0]]]
    for i in range(1, len(values)):
        if values[i] - values[i - 1] > expected_spacing * 0.5:
            clusters.append([])
        clusters[-1].append(values[i])

    return [np.mean(c) for c in clusters]


def infer_grid(centers, min_points=4):
    """
    Given detected QR centers, infer a regular grid and return predicted
    positions for ALL grid cells (including missing ones).

    Returns:
        grid_positions: dict mapping (row, col) to predicted (x, y) position
        H: homography from grid-index space to image space
        assigned: dict mapping (row, col) to index in input centers
    """
    if len(centers) < min_points:
        return None, None, None

    centers = np.array(centers, dtype=np.float32)

    # Use nearest-neighbor distance median as the grid spacing estimate
    nn_dists = _nearest_neighbor_dists(centers)
    base_dist = np.median(nn_dists)
    if base_dist < 10:
        return None, None, None

    # Find vectors that are approximately 1x grid spacing
    vectors = []
    for i, j in combinations(range(len(centers)), 2):
        v = centers[j] - centers[i]
        dist = np.linalg.norm(v)
        if base_dist * 0.5 < dist < base_dist * 1.5:
            vectors.append(v)

    if len(vectors) < 2:
        return None, None, None

    vectors = np.array(vectors)

    # Cluster directions into ~horizontal and ~vertical
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % np.pi
    angle_data = angles.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

    try:
        _, labels, _ = cv2.kmeans(angle_data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return None, None, None

    labels = labels.flatten()
    dir1_vecs = vectors[labels == 0]
    dir2_vecs = vectors[labels == 1]

    if len(dir1_vecs) == 0 or len(dir2_vecs) == 0:
        return None, None, None

    def avg_direction(vecs):
        ref = vecs[0]
        aligned = [(-v if np.dot(v, ref) < 0 else v) for v in vecs]
        return np.mean(aligned, axis=0)

    d1 = avg_direction(dir1_vecs)
    d2 = avg_direction(dir2_vecs)

    # Ensure d1 is more horizontal
    if abs(np.arctan2(d1[1], d1[0])) > abs(np.arctan2(d2[1], d2[0])):
        d1, d2 = d2, d1

    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    spacing1 = np.linalg.norm(d1)
    spacing2 = np.linalg.norm(d2)

    proj1 = centers @ d1_norm
    proj2 = centers @ d2_norm

    cols = _cluster_1d(proj1, spacing1)
    rows = _cluster_1d(proj2, spacing2)

    if len(cols) < 2 or len(rows) < 2:
        return None, None, None

    # Assign each center to nearest (row, col)
    assigned = {}
    for idx, c in enumerate(centers):
        p1 = np.dot(c, d1_norm)
        p2 = np.dot(c, d2_norm)
        col_idx = int(np.argmin([abs(p1 - cc) for cc in cols]))
        row_idx = int(np.argmin([abs(p2 - rc) for rc in rows]))
        key = (row_idx, col_idx)
        if key not in assigned:
            assigned[key] = idx

    # Build correspondence: grid indices -> image positions
    grid_pts = []
    img_pts = []
    for (r, c), idx in assigned.items():
        grid_pts.append([c, r])
        img_pts.append(centers[idx])

    grid_pts = np.array(grid_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    if len(grid_pts) < 4:
        return None, None, None

    H, _ = cv2.findHomography(grid_pts, img_pts, cv2.RANSAC, base_dist * 0.3)
    if H is None:
        return None, None, None

    # Predict positions for ALL grid cells
    grid_positions = {}
    for r in range(len(rows)):
        for c in range(len(cols)):
            pt = np.array([[[c, r]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pt, H)
            grid_positions[(r, c)] = img_pt[0, 0]

    return grid_positions, H, assigned


def find_missing_positions(grid_positions, assigned, detected_centers, match_dist=80):
    """Find grid positions where no detection exists."""
    if grid_positions is None:
        return []

    detected = np.array(detected_centers, dtype=np.float32) if detected_centers else np.empty((0, 2))
    missing = []

    for (r, c), pos in grid_positions.items():
        if (r, c) in assigned:
            continue
        if len(detected) > 0:
            dists = np.linalg.norm(detected - pos, axis=1)
            if dists.min() < match_dist:
                continue
        missing.append(pos)

    return missing

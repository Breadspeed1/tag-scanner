import cv2


def draw_results(img, results):
    vis = img.copy()

    for i, det in enumerate(results):
        pts = det['qr_corners'].astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)

        sku_pts = det['sku_corners'].astype(int)
        cv2.polylines(vis, [sku_pts], True, (255, 100, 0), 3)

        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vis, f'#{i}', (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis

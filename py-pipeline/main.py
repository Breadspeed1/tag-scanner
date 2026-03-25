import sys
import cv2
from pipeline import detect_qr_codes, draw_results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'img/large_clear.jpg'
    img = cv2.imread(path)
    if img is None:
        print(f'Could not read image: {path}')
        return

    print(f'Image: {img.shape[1]}x{img.shape[0]}')

    results, energy, mask = detect_qr_codes(img, debug=True)
    print(f'Detected {len(results)} QR code(s)')

    cv2.namedWindow('Energy', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Energy', 960, 540)
    cv2.imshow('Energy', energy)

    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 960, 540)
    cv2.imshow('Mask', mask)

    vis = draw_results(img, results)
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detections', 960, 540)
    cv2.imshow('Detections', vis)

    for i, det in enumerate(results):
        cv2.namedWindow(f'SKU #{i}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'SKU #{i}', det['sku_crop'])

    print('Press any key to close...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('debug/debug_energy.jpg', energy)
    cv2.imwrite('debug/debug_mask.jpg', mask)
    cv2.imwrite('debug/debug_detections.jpg', vis)
    for i, det in enumerate(results):
        cv2.imwrite(f'debug/debug_sku_{i}.png', det['sku_crop'])
    print('Debug images saved.')


if __name__ == '__main__':
    main()

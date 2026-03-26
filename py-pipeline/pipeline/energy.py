import cv2
import numpy as np


def compute_energy_map(gray, kernel_size=61):
    # Bidirectional gradient: high only where there's texture in BOTH x and y
    # This finds QR codes (2D checkerboard) and suppresses barcodes (1D) and glare
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Blur each gradient independently first to get local directional energy
    gx = cv2.blur(sobel_x * sobel_x, ksize=(kernel_size, kernel_size))
    gy = cv2.blur(sobel_y * sobel_y, ksize=(kernel_size, kernel_size))

    # Geometric mean: high only when BOTH directions have energy
    energy = np.sqrt(gx * gy)

    cv2.normalize(energy, energy, 0, 255, cv2.NORM_MINMAX)
    return energy.astype(np.uint8)


def energy_to_mask(energy, morph_kernel_size=15, percentile=98):
    thresh_val = np.percentile(energy, percentile)
    _, binary = cv2.threshold(energy, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def find_qr_blobs(binary, min_area=3000):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        boxes.append(cv2.boundingRect(cnt))

    return boxes

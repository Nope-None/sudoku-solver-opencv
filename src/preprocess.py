"""
src/preprocess.py
Image preprocessing: grayscale → blur → threshold → contour → warp → cells.
"""

import cv2
import numpy as np


def preprocess(image):
    """Convert to grayscale, blur, and apply adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh


def find_board_contour(thresh):
    """
    Find the largest quadrilateral contour — the Sudoku grid.
    Returns 4 corner points or None.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def order_points(pts):
    """Order 4 corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_board(image, corners, size=450):
    """Apply perspective transform to get a top-down view of the board."""
    src = order_points(corners)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped, M, src


def extract_cells(warped_gray):
    """Slice the warped board into 81 individual cell images."""
    cell_size = warped_gray.shape[0] // 9
    cells = []
    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            cells.append(warped_gray[y1:y2, x1:x2])
    return cells

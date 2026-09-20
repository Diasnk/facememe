import cv2
import numpy as np

# MediaPipe face mesh inner-lip contour indices (itsgiving).
INNER_LIPS = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def tongue_score(frame, face, hands, jaw_ready):
    """Fraction of the mouth opening that reads pink (tongue vs teeth/throat)."""
    if not jaw_ready:
        return 0.0
    if any(_dist(h.palm, face.mouth) < 0.7 * face.width for h in hands):
        return 0.0

    poly = face.pts[INNER_LIPS].astype(np.int32)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0

    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = int(x1), int(y1)
    roi = frame[y0 : y1 + 1, x0 : x1 + 1]
    if roi.size == 0:
        return 0.0

    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    k = max(3, int(0.15 * (y1 - y0)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    n = int(np.count_nonzero(mask))
    if n < 40:
        return 0.0

    hue, sat, val = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    pink = ((hue < 12) | (hue > 160)) & (sat > 70) & (val > 110)
    return float(np.count_nonzero(pink & (mask > 0)) / n)

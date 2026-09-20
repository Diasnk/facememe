import numpy as np

# MediaPipe hand topology (21 landmarks).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

RAISED_Y_RATIO = 0.4


def _dist(a, b):
    return float(np.linalg.norm(a - b))


class Hand:
    """Pixel-space hand geometry from MediaPipe hand landmarks."""

    def __init__(self, pts, palm, thumb, index, middle, open, raised, horizontal, vertical):
        self.pts = pts
        self.palm = palm
        self.thumb = thumb
        self.index = index
        self.middle = middle
        self.open = open
        self.raised = raised
        self.horizontal = horizontal
        self.vertical = vertical

    @classmethod
    def from_landmarks(cls, lms, width, height):
        pts = np.array([[lm.x * width, lm.y * height] for lm in lms], dtype=np.float32)
        palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)
        thumb, index, middle = pts[4], pts[8], pts[12]
        d = pts[9] - pts[0]
        horizontal = abs(d[0]) > 1.5 * abs(d[1])
        vertical = abs(d[1]) > 1.5 * abs(d[0])
        extended = [
            _dist(pts[0], pts[tip]) > 1.2 * _dist(pts[0], pts[tip - 2])
            for tip in (8, 12, 16, 20)
        ]
        open_hand = sum(extended) >= 3
        raised = bool(palm[1] < RAISED_Y_RATIO * height)
        return cls(
            pts=pts,
            palm=palm,
            thumb=thumb,
            index=index,
            middle=middle,
            open=open_hand,
            raised=raised,
            horizontal=horizontal,
            vertical=vertical,
        )

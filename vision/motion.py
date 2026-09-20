import numpy as np


def _dist(a, b):
    return float(np.linalg.norm(a - b))


class Motion:
    """Smoothed hand speed across frames, in face-widths per frame."""

    def __init__(self):
        self.prev = []
        self.energy = 0.0
        self.fw = 200.0

    def update(self, hands, face):
        if face is not None:
            self.fw = max(face.width, 1.0)

        cur = [h.palm for h in hands]
        speed = 0.0
        if cur and self.prev:
            moved = [min(_dist(c, p) for p in self.prev) for c in cur]
            moved = [m for m in moved if m < self.fw]
            if moved:
                speed = max(moved) / self.fw

        self.energy = 0.8 * self.energy + 0.2 * speed
        self.prev = cur
        return self.energy

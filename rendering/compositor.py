import numpy as np

from rendering.assets import overlay

FACE_SCALE = 2.0
SMOOTH = 0.7


class OverlayTracker:
    """EMA-smoothed face center/height + draw reaction sprites."""

    def __init__(self):
        self.center = None
        self.height = None

    def reset(self):
        self.center = None
        self.height = None

    def draw(self, frame, face, shown, assets, shown_since_ms):
        if face is None or not shown or shown not in assets:
            if shown is None:
                self.reset()
            return frame

        target_center = np.array(face.center, dtype=np.float32)
        target_h = float(face.height) * FACE_SCALE
        if self.center is None:
            self.center = target_center
            self.height = target_h
        else:
            self.center = SMOOTH * self.center + (1.0 - SMOOTH) * target_center
            self.height = SMOOTH * self.height + (1.0 - SMOOTH) * target_h

        fh, fw = frame.shape[:2]
        asset = assets[shown]
        height = int(min(self.height, fh * 0.98, (fw * 0.98) / asset.aspect)) // 8 * 8
        height = max(height, 8)
        idx = asset.frame_at(int(shown_since_ms))
        sprite = asset.scaled(idx, height)
        sh, sw = sprite.shape[:2]
        x = int(self.center[0] - sw / 2)
        y = int(self.center[1] - sh / 2 - 0.05 * sh)
        return overlay(frame, sprite, x, y)

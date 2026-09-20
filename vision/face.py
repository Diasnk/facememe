import numpy as np


class Face:
    """Normalized MediaPipe face landmarks converted to pixel-space geometry."""

    def __init__(
        self,
        pts,
        bbox,
        center,
        width,
        height,
        nose,
        chin,
        mouth,
        eye_y,
        top,
        blendshapes,
        turn_signed,
    ):
        self.pts = pts
        self.bbox = bbox
        self.center = center
        self.width = width
        self.height = height
        self.nose = nose
        self.chin = chin
        self.mouth = mouth
        self.eye_y = eye_y
        self.top = top
        self.blendshapes = blendshapes
        self.turn_signed = turn_signed

    @classmethod
    def from_landmarks(cls, lms, blendshapes, width, height):
        pts = np.array([[lm.x * width, lm.y * height] for lm in lms], dtype=np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        bbox = (int(x0), int(y0), int(x1), int(y1))
        face_w = float(x1 - x0)
        face_h = float(y1 - y0)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        nose = pts[1]
        chin = pts[152]
        mouth = (pts[13] + pts[14]) / 2.0
        eye_y = float((pts[33][1] + pts[263][1]) / 2.0)
        top = pts[10]
        cheek_left, cheek_right = pts[234], pts[454]
        turn_signed = float(
            (nose[0] - cheek_left[0]) / max(cheek_right[0] - cheek_left[0], 1e-3) - 0.5
        )
        bs = {c.category_name: c.score for c in (blendshapes or [])}
        return cls(
            pts=pts,
            bbox=bbox,
            center=center,
            width=face_w,
            height=face_h,
            nose=nose,
            chin=chin,
            mouth=mouth,
            eye_y=eye_y,
            top=top,
            blendshapes=bs,
            turn_signed=turn_signed,
        )

    def blendshape(self, name, default=0.0):
        return self.blendshapes.get(name, default)

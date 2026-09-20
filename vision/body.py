import numpy as np

# Pose landmark indices: left/right shoulder, elbow, wrist.
# Connections use those indices into the full pose landmark list.
UPPER_BODY_CONNECTIONS = (
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
)

VISIBILITY_THRESHOLD = 0.5


class Body:
    """Upper-body pose: shoulders, elbows, wrists + elbows_up flag."""

    def __init__(self, pts, shoulders, elbows, wrists, seen, elbows_up):
        self.pts = pts
        self.shoulders = shoulders
        self.elbows = elbows
        self.wrists = wrists
        self.seen = seen
        self.elbows_up = elbows_up

    @classmethod
    def from_landmarks(cls, lms, width, height):
        pts = np.array([[lm.x * width, lm.y * height] for lm in lms], dtype=np.float32)
        shoulders = pts[[11, 12]]
        elbows = pts[[13, 14]]
        wrists = pts[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        seen = min(vis) > VISIBILITY_THRESHOLD
        shoulder_y = float(shoulders[:, 1].mean())
        elbows_up = seen and bool((elbows[:, 1] < shoulder_y).all())
        return cls(
            pts=pts,
            shoulders=shoulders,
            elbows=elbows,
            wrists=wrists,
            seen=seen,
            elbows_up=elbows_up,
        )

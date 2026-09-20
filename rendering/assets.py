import os

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")


class Asset:
    """One reaction: BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / float(h)
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            width = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(
                self.frames[idx], (width, height), interpolation=cv2.INTER_AREA
            )
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    if not os.path.isdir(ASSETS_DIR):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg")
    for fn in sorted(os.listdir(ASSETS_DIR)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(ASSETS_DIR, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)

    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence

        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(frame.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(frame.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]

    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)

    print(
        f"  {pose:16s} {os.path.basename(path)} "
        f"({len(frames)} frame{'s' if len(frames) != 1 else ''})"
    )
    return Asset(frames, durations)


def load_all_assets(reaction_names):
    print("Loading reaction assets...")
    return {name: load_asset(name) for name in reaction_names}


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y)."""
    fh, fw = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, fw), min(y + sh, fh)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * s[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)
    return frame

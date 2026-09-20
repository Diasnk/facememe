import json
import os
import time

CALIB_SECONDS = 7.0
CALIB_WARMUP = 1.5
CALIB_MIN_SAMPLES = 30
SIGMA_FLOOR = 0.015
SIGMA_CEIL = 0.080
CALIB_VERSION = 1

GENERIC_SIGMA = 0.035
GENERIC_MEAN = {
    "jawOpen": 0.08,
    "eyeSquintLeft": 0.10,
    "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10,
    "eyeBlinkRight": 0.10,
    "noseSneerLeft": 0.03,
    "noseSneerRight": 0.03,
    "browDownLeft": 0.06,
    "browDownRight": 0.06,
    "mouthFrownLeft": 0.05,
    "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05,
    "mouthUpperUpRight": 0.05,
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(_PROJECT_ROOT, "data", "calibration.json")


class Baseline:
    """Your resting face: a mean and a wobble for every channel."""

    def __init__(self, mean=None, sigma=None, samples=0, made=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.generic = not self.mean

    def z(self, name, value):
        """How far above your neutral this channel is, in standard deviations."""
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        return (value - m) / self.sigma.get(name, SIGMA_CEIL)

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    def save(self, path=None):
        path = path or CALIB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": CALIB_VERSION,
                    "made": self.made,
                    "samples": self.samples,
                    "mean": self.mean,
                    "sigma": self.sigma,
                },
                fh,
                indent=1,
                sort_keys=True,
            )

    @staticmethod
    def load(path=None):
        path = path or CALIB_PATH
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Baseline()
        if data.get("version") != CALIB_VERSION or not data.get("mean"):
            return Baseline()
        return Baseline(
            data["mean"],
            data.get("sigma", {}),
            data.get("samples", 0),
            data.get("made"),
        )


class Collector:
    """Running mean and standard deviation per channel over the calibration window."""

    def __init__(self):
        self.n = 0
        self.s = {}
        self.ss = {}

    def add(self, face):
        self.n += 1
        for name, v in list(face.blendshapes.items()) + [("turn_signed", face.turn_signed)]:
            self.s[name] = self.s.get(name, 0.0) + v
            self.ss[name] = self.ss.get(name, 0.0) + v * v

    def finish(self):
        mean, sigma = {}, {}
        for name, total in self.s.items():
            m = total / self.n
            var = max(self.ss[name] / self.n - m * m, 0.0)
            mean[name] = round(m, 5)
            sigma[name] = round(min(max(var ** 0.5, SIGMA_FLOOR), SIGMA_CEIL), 5)
        sigma["turn_signed"] = min(max(sigma.get("turn_signed", 0.02), 0.01), 0.10)
        return Baseline(mean, sigma, self.n, time.strftime("%Y-%m-%d %H:%M"))


def calibration_warnings(base):
    """Catch common calibration mistakes: mid-expression or fidgeting."""
    out = []
    if base.mean.get("jawOpen", 0) > 0.30:
        out.append("your mouth looks like it was open — don't talk during calibration")
    if max(base.mean.get("noseSneerLeft", 0), base.mean.get("noseSneerRight", 0)) > 0.15:
        out.append("your nose was scrunched — hold a bored face, not a reaction")
    if max(base.mean.get("browInnerUp", 0), base.mean.get("browOuterUpLeft", 0)) > 0.35:
        out.append("your eyebrows were up — relax them")
    pinned = sum(1 for v in base.sigma.values() if v >= SIGMA_CEIL)
    if pinned > 12:
        out.append("you moved a lot — sit still and try again for a tighter baseline")
    return out

import time

from reactions.classifier import REACTIONS

HOLD_FRAMES = 10
ARM = {
    "spin": 15,
    "suspicious": 8,
    "talking_to_wall": 6,
    "dance": 6,
    "crashing_out": 4,
    "open_mouth": 4,
    "tongue_out": 5,
    "disgusted": 5,
    "heart": 3,
    "time_out": 3,
    "cover_nose": 3,
    "nose_closed": 3,
    "flirty": 3,
    "hand_up": 3,
}


class ReactionState:
    """Arm counters + hold frames so reaction labels do not flicker every frame."""

    def __init__(self):
        self.arm = {name: 0 for name in REACTIONS}
        self.shown = None
        self.hold = 0
        self.shown_since = time.monotonic()

    def update(self, raw_label):
        fired = None
        for name in REACTIONS:
            if raw_label == name:
                self.arm[name] += 1
            else:
                self.arm[name] = 0
            if raw_label == name and self.arm[name] >= ARM.get(name, 3):
                fired = name

        if fired:
            if fired != self.shown:
                self.shown = fired
                self.hold = HOLD_FRAMES
                self.shown_since = time.monotonic()
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.shown = None

        return self.shown

    def elapsed_ms(self):
        if self.shown is None:
            return 0
        return int((time.monotonic() - self.shown_since) * 1000)

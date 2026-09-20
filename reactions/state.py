import time

from reactions.classifier import REACTIONS

# Wall-clock arm/hold so latency does not stretch when FPS drops.
# Values ≈ former frame counts at ~30 FPS, with a slightly tighter hold.
HOLD_MS = 250
ARM_MS = {
    "spin": 400,
    "suspicious": 250,
    "talking_to_wall": 200,
    "dance": 200,
    "crashing_out": 130,
    "open_mouth": 130,
    "tongue_out": 160,
    "disgusted": 160,
    "heart": 100,
    "time_out": 100,
    "cover_nose": 100,
    "nose_closed": 100,
    "flirty": 100,
    "hand_up": 100,
}


class ReactionState:
    """Time-based arm + hold so reaction labels do not flicker."""

    def __init__(self):
        self.arm_since = {name: None for name in REACTIONS}
        self.shown = None
        self.hold_until = 0.0
        self.shown_since = time.monotonic()

    def update(self, raw_label):
        now = time.monotonic()
        fired = None
        for name in REACTIONS:
            if raw_label == name:
                if self.arm_since[name] is None:
                    self.arm_since[name] = now
                needed = ARM_MS.get(name, 100) / 1000.0
                if now - self.arm_since[name] >= needed:
                    fired = name
            else:
                self.arm_since[name] = None

        if fired:
            if fired != self.shown:
                self.shown = fired
                self.shown_since = now
            self.hold_until = now + HOLD_MS / 1000.0
        elif now >= self.hold_until:
            self.shown = None

        return self.shown

    def elapsed_ms(self):
        if self.shown is None:
            return 0
        return int((time.monotonic() - self.shown_since) * 1000)

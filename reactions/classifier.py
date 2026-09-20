import numpy as np

REACTIONS = (
    "time_out",
    "heart",
    "cover_nose",
    "crashing_out",
    "dance",
    "nose_closed",
    "flirty",
    "hand_up",
    "tongue_out",
    "open_mouth",
    "disgusted",
    "talking_to_wall",
    "suspicious",
    "spin",
)

Z = dict(
    jaw_open=6.0,
    scream_jaw=3.5,
    tongue_jaw=3.5,
    sneer=4.5,
    disgust=14.0,
    squint=4.0,
)

FLOOR = dict(
    jaw_open=0.30,
    scream_jaw=0.18,
    tongue_jaw=0.18,
    sneer=0.06,
    squint=0.18,
)

T = dict(
    tongue=0.5,
    head_turn=0.15,
    gesture=0.035,
)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def over(key, m, zkey, rawkey):
    """Sigma above neutral AND a raw floor, so tiny sigma can't hair-trigger."""
    return m[zkey] >= Z[key] and m[rawkey] >= FLOOR[key]


def decide(face, hands, body, gesture, m, tongue=0.0):
    """Return (reaction_label or None, debug dict). Priority: first match wins."""
    debug = {"hands": len(hands), "gesture": gesture, "tongue": tongue}

    if face is None:
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), debug

    if not m:
        return None, debug

    fw = face.width
    near = lambda a, b, k: _dist(a, b) < k * fw
    elbows_up = bool(body and body.elbows_up)
    debug.update(m)
    debug["elbows_up"] = elbows_up
    screaming = over("scream_jaw", m, "z_jaw", "jaw")

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):
            if (
                top.horizontal
                and under.vertical
                and top.palm[1] < under.palm[1]
                and near(under.middle, top.palm, 0.6)
            ):
                return "time_out", debug
        if (
            near(a.index, b.index, 0.3)
            and near(a.thumb, b.thumb, 0.3)
            and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1])
        ):
            return "heart", debug
        if near(a.palm, face.mouth, 0.6) and near(b.palm, face.mouth, 0.6):
            return "cover_nose", debug

        def on_head(h):
            return (
                h.palm[1] < face.eye_y
                and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                and h.palm[1] > face.top[1] - 0.8 * face.height
            )

        if on_head(a) and on_head(b) and screaming:
            return "crashing_out", debug

    def near_head(h):
        return abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.height

    if elbows_up and hands and all(near_head(h) for h in hands):
        return ("crashing_out" if screaming else "dance"), debug

    if elbows_up:
        return "dance", debug

    for h in hands:
        if (
            near(h.thumb, face.nose, 0.35)
            and near(h.index, face.nose, 0.35)
            and near(h.thumb, h.index, 0.3)
        ):
            return "nose_closed", debug
        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", debug
        if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.8 * fw:
            return "hand_up", debug

    if tongue > T["tongue"]:
        return "tongue_out", debug
    if over("jaw_open", m, "z_jaw", "jaw"):
        return "open_mouth", debug
    if over("sneer", m, "z_sneer", "sneer") or m["z_disgust"] >= Z["disgust"]:
        return "disgusted", debug
    if hands and gesture > T["gesture"]:
        return "talking_to_wall", debug
    if m["turn"] > T["head_turn"] and over("squint", m, "z_squint", "squint"):
        return "suspicious", debug

    return None, debug

Z_CAP = 8.0


def measure(face, baseline):
    """Every expression channel, raw and in sigma above your own neutral."""

    def zpair(name):
        return (
            baseline.z(name + "Left", face.blendshape(name + "Left"))
            + baseline.z(name + "Right", face.blendshape(name + "Right"))
        ) / 2

    def pair(name):
        return (face.blendshape(name + "Left") + face.blendshape(name + "Right")) / 2

    m = {
        "jaw": face.blendshape("jawOpen"),
        "z_jaw": baseline.z("jawOpen", face.blendshape("jawOpen")),
        "sneer": pair("noseSneer"),
        "z_sneer": zpair("noseSneer"),
        "z_brow": zpair("browDown"),
        "z_frown": zpair("mouthFrown"),
        "z_lip": zpair("mouthUpperUp"),
        "squint": max(pair("eyeSquint"), pair("eyeBlink")),
        "z_squint": max(zpair("eyeSquint"), zpair("eyeBlink")),
        "turn": abs(face.turn_signed - baseline.neutral_turn),
    }

    def cap(v):
        return min(v, Z_CAP)

    m["z_disgust"] = (
        2 * cap(m["z_sneer"]) + cap(m["z_brow"]) + cap(m["z_frown"]) + cap(m["z_lip"])
    )
    return m

import time

import cv2

from calibration.baseline import (
    CALIB_MIN_SAMPLES,
    CALIB_PATH,
    CALIB_SECONDS,
    CALIB_WARMUP,
    Baseline,
    Collector,
    calibration_warnings,
)
from camera.capture import open_camera, read_frame, release_camera
from reactions.classifier import REACTIONS, decide, over
from reactions.state import ReactionState
from rendering.assets import load_all_assets
from rendering.compositor import OverlayTracker
from vision.body import UPPER_BODY_CONNECTIONS, Body
from vision.detectors import (
    create_face_landmarker,
    create_hand_landmarker,
    create_pose_landmarker,
    detect_face,
    detect_hands,
    detect_pose,
    frame_to_mp_image,
)
from vision.face import Face
from vision.features import measure
from vision.hands import HAND_CONNECTIONS, Hand
from vision.models import ensure_models
from vision.motion import Motion
from vision.tongue import tongue_score


class TimestampClock:
    """Strictly increasing millisecond timestamps for MediaPipe VIDEO mode."""

    def __init__(self):
        self._t0 = time.monotonic()
        self._last = -1

    def next(self):
        self._last = max(int((time.monotonic() - self._t0) * 1000), self._last + 1)
        return self._last


def draw_face(frame, face):
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for point in (face.nose, face.mouth, face.chin):
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)


def draw_hands(frame, hands):
    color = (255, 255, 0)  # cyan in BGR
    for hand in hands:
        for a, b in HAND_CONNECTIONS:
            pa = (int(hand.pts[a][0]), int(hand.pts[a][1]))
            pb = (int(hand.pts[b][0]), int(hand.pts[b][1]))
            cv2.line(frame, pa, pb, color, 2)
        for tip in (hand.thumb, hand.index, hand.middle, hand.palm):
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, color, -1)

        labels = []
        if hand.open:
            labels.append("OPEN")
        if hand.raised:
            labels.append("RAISED")
        if labels:
            text = " ".join(labels)
            origin = (int(hand.palm[0]) + 8, int(hand.palm[1]) - 8)
            cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_body(frame, body):
    if not body.seen:
        return
    color = (255, 0, 255)  # magenta in BGR
    for a, b in UPPER_BODY_CONNECTIONS:
        pa = (int(body.pts[a][0]), int(body.pts[a][1]))
        pb = (int(body.pts[b][0]), int(body.pts[b][1]))
        cv2.line(frame, pa, pb, color, 2)
    for point in list(body.shoulders) + list(body.elbows) + list(body.wrists):
        cv2.circle(frame, (int(point[0]), int(point[1])), 4, color, -1)
    if body.elbows_up:
        mid = body.shoulders.mean(axis=0)
        origin = (int(mid[0]) - 40, int(mid[1]) - 16)
        cv2.putText(frame, "ELBOWS UP", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_calibration(frame, elapsed, samples, face):
    h, w = frame.shape[:2]
    left = max(0.0, CALIB_SECONDS - elapsed)
    cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "CALIBRATING - hold a neutral face",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    status = f"{left:0.1f}s  {samples} frames"
    if face is None:
        status += "  NO FACE"
    cv2.putText(
        frame,
        status,
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255) if face is not None else (0, 140, 255),
        2,
    )
    done = int(w * min(elapsed / CALIB_SECONDS, 1.0))
    cv2.rectangle(frame, (0, 84), (done, 96), (0, 220, 0), -1)
    if face is not None:
        draw_face(frame, face)


def run_calibration(cap, face_landmarker, clock):
    print(
        f"\nCalibrating for {CALIB_SECONDS:.0f}s. Sit normally, look at the camera, "
        "hold a bored/neutral face.\nBlinking is fine. Don't talk, smile, or raise eyebrows."
    )
    print("Press q to cancel.")

    collector = Collector()
    start = time.monotonic()
    seen_face = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed > CALIB_SECONDS:
            break

        frame = read_frame(cap, mirror=True)
        if frame is None:
            break

        h, w = frame.shape[:2]
        mp_image = frame_to_mp_image(frame)
        result = detect_face(face_landmarker, mp_image, clock.next())
        face = None
        if result.face_landmarks:
            blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
            face = Face.from_landmarks(result.face_landmarks[0], blendshapes, w, h)
            seen_face += 1
            if elapsed > CALIB_WARMUP and face.blendshapes:
                collector.add(face)

        draw_calibration(frame, elapsed, collector.n, face)
        cv2.imshow("FaceCam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration cancelled.")
            return None

    if collector.n < CALIB_MIN_SAMPLES:
        reason = " — your face was never detected" if not seen_face else ""
        print(
            f"Calibration failed: only {collector.n} usable frames{reason}.\n"
            "Light your face from the front, sit head-and-shoulders in frame, try again (press c)."
        )
        return None

    baseline = collector.finish()
    print(f"Calibrated on {baseline.samples} frames. Neutral face:")
    for name in ("jawOpen", "noseSneerLeft", "browDownLeft", "mouthFrownLeft", "eyeSquintLeft"):
        if name in baseline.mean:
            print(f"  {name:16s} {baseline.mean[name]:.3f} ± {baseline.sigma[name]:.3f}")
    for warning in calibration_warnings(baseline):
        print(f"  ! {warning}")

    baseline.save(CALIB_PATH)
    print(f"Saved {CALIB_PATH}")
    return baseline


def draw_feature_hud(frame, feats, energy, shown, raw):
    lines = [
        f"showing: {shown or '-'}   raw: {raw or '-'}",
        f"motion: {energy:.2f}",
        f"z_jaw: {feats['z_jaw']:.2f}",
        f"z_sneer: {feats['z_sneer']:.2f}",
        f"turn: {feats['turn']:.2f}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26


def main():
    print("FACECAM starting...")
    print("Keys: q quit | c recalibrate")

    model_paths = ensure_models()
    face_landmarker = create_face_landmarker(model_paths["face_landmarker.task"])
    hand_landmarker = create_hand_landmarker(model_paths["hand_landmarker.task"])
    pose_landmarker = create_pose_landmarker(model_paths["pose_landmarker_lite.task"])
    clock = TimestampClock()
    motion = Motion()
    reaction_state = ReactionState()
    assets = load_all_assets(REACTIONS)
    overlay_tracker = OverlayTracker()

    baseline = Baseline.load(CALIB_PATH)
    if baseline.generic:
        print("No calibration found — starting calibration first.")

    cap = None
    try:
        cap = open_camera(index=0, width=1280, height=720)

        if baseline.generic:
            calibrated = run_calibration(cap, face_landmarker, clock)
            if calibrated is None:
                print("Using generic baseline until you press c.")
            else:
                baseline = calibrated

        while True:
            frame = read_frame(cap, mirror=True)
            if frame is None:
                print("Failed to read frame from camera.")
                break

            h, w = frame.shape[:2]
            ts = clock.next()
            mp_image = frame_to_mp_image(frame)

            face = None
            face_result = detect_face(face_landmarker, mp_image, ts)
            if face_result.face_landmarks:
                blendshapes = (
                    face_result.face_blendshapes[0] if face_result.face_blendshapes else None
                )
                face = Face.from_landmarks(face_result.face_landmarks[0], blendshapes, w, h)
                draw_face(frame, face)

            hand_result = detect_hands(hand_landmarker, mp_image, ts)
            hands = [
                Hand.from_landmarks(lms, w, h)
                for lms in hand_result.hand_landmarks
            ]
            if hands:
                draw_hands(frame, hands)

            body = None
            pose_result = detect_pose(pose_landmarker, mp_image, ts)
            if pose_result.pose_landmarks:
                body = Body.from_landmarks(pose_result.pose_landmarks[0], w, h)
                draw_body(frame, body)

            energy = motion.update(hands, face)
            feats = measure(face, baseline) if face is not None else {}
            tongue = 0.0
            if face is not None and feats:
                jaw_ready = over("tongue_jaw", feats, "z_jaw", "jaw")
                tongue = tongue_score(frame, face, hands, jaw_ready)
            raw, _dbg = decide(face, hands, body, energy, feats, tongue=tongue)
            shown = reaction_state.update(raw)

            overlay_tracker.draw(
                frame,
                face,
                shown,
                assets,
                reaction_state.elapsed_ms(),
            )

            if feats:
                draw_feature_hud(frame, feats, energy, shown, raw)
            else:
                cv2.putText(
                    frame,
                    f"showing: {shown or '-'}   motion: {energy:.2f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("FaceCam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrated = run_calibration(cap, face_landmarker, clock)
                if calibrated is not None:
                    baseline = calibrated
                    reaction_state = ReactionState()
                    overlay_tracker.reset()
    finally:
        face_landmarker.close()
        hand_landmarker.close()
        pose_landmarker.close()
        release_camera(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

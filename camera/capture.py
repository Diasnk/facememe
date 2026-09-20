import cv2


def open_camera(index=0, width=1280, height=720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def read_frame(cap, mirror=True):
    ok, frame = cap.read()
    if not ok:
        return None
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def release_camera(cap):
    if cap is not None:
        cap.release()

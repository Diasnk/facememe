import pyvirtualcam


def open_virtual_camera(width, height, fps=30):
    """Open a pyvirtualcam device in BGR. Returns camera or None on failure."""
    try:
        cam = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        print(f"Virtual camera: '{cam.device}' <- pick this camera in Zoom / Meet / Discord")
        return cam
    except Exception as exc:
        print(f"Virtual camera unavailable ({exc}).")
        print("Install OBS Studio for the Windows virtual-camera backend, or run with --no-vcam.")
        return None


def send_frame(cam, bgr_frame):
    if cam is None:
        return
    cam.send(bgr_frame)
    cam.sleep_until_next_frame()


def close_virtual_camera(cam):
    if cam is not None:
        cam.close()

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


def frame_to_mp_image(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def create_face_landmarker(model_path):
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker(model_path):
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_pose_landmarker(model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_face(landmarker, mp_image, timestamp_ms):
    """Run Face Landmarker on a MediaPipe Image. Returns the raw result."""
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def detect_hands(landmarker, mp_image, timestamp_ms):
    """Run Hand Landmarker on a MediaPipe Image. Returns the raw result."""
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def detect_pose(landmarker, mp_image, timestamp_ms):
    """Run Pose Landmarker on a MediaPipe Image. Returns the raw result."""
    return landmarker.detect_for_video(mp_image, timestamp_ms)

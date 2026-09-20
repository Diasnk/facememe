import os
import urllib.request

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def ensure_models():
    """Download MediaPipe .task files into models/ if missing. Return name -> path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths

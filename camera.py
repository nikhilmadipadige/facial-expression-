"""
Camera and Face Detection Module

Handles webcam capture and face detection using MediaPipe.
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============== CONFIGURATION ==============
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


# ============== GLOBAL VARIABLES ==============
_camera = None
_face_landmarker = None
_mp_module = None


# ============== MODEL DOWNLOAD ==============
def get_model_path():
    """Download MediaPipe model if not exists and return path."""
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / MODEL_FILENAME

    if not model_path.exists():
        print("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print(f"Model downloaded to: {model_path}")

    return str(model_path)


# ============== FACE DETECTOR INITIALIZATION ==============
def init_face_detector():
    """Initialize the MediaPipe face landmarker."""
    global _face_landmarker, _mp_module

    if _face_landmarker is not None:
        return _face_landmarker

    model_path = get_model_path()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    _mp_module = mp

    print("Face detector initialized successfully")
    return _face_landmarker


# ============== CAMERA FUNCTIONS ==============
def get_camera():
    """Get or create camera instance."""
    global _camera

    if _camera is not None and _camera.isOpened():
        return _camera

    # Try DirectShow on Windows for better compatibility
    _camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not _camera.isOpened():
        _camera = cv2.VideoCapture(0)

    if not _camera.isOpened():
        print("Error: Could not open camera")
        return None

    # Set camera properties
    _camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    _camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    print(f"Camera opened: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    return _camera


def release_camera():
    """Release camera resources."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
        print("Camera released")


def read_frame():
    """Read a frame from camera."""
    camera = get_camera()
    if camera is None:
        return None

    ret, frame = camera.read()
    if not ret:
        return None

    # Flip horizontally for mirror effect
    return cv2.flip(frame, 1)


# ============== FACE DETECTION ==============
def detect_face(frame):
    """
    Detect face in frame and return cropped face image with bounding box.

    Returns:
        tuple: (face_image, bbox, landmarks) or (None, None, None) if no face
    """
    global _face_landmarker, _mp_module

    if _face_landmarker is None:
        init_face_detector()

    if frame is None or frame.size == 0:
        return None, None, None

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = _mp_module.Image(
        image_format=_mp_module.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect faces
    results = _face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None, None, None

    # Get first face landmarks
    face_landmarks = results.face_landmarks[0]
    h, w = frame.shape[:2]

    # Calculate bounding box from landmarks
    x_coords = [lm.x * w for lm in face_landmarks]
    y_coords = [lm.y * h for lm in face_landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add padding (20%)
    padding = 0.2
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * padding
    pad_y = box_h * padding

    x = max(0, int(x_min - pad_x))
    y = max(0, int(y_min - pad_y))
    bw = min(w - x, int(box_w + 2 * pad_x))
    bh = min(h - y, int(box_h + 2 * pad_y))

    bbox = (x, y, bw, bh)

    # Crop face
    face_image = frame[y:y+bh, x:x+bw].copy()

    # Extract landmark coordinates
    landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]

    return face_image, bbox, landmarks


def draw_face_box(frame, bbox, label=None, confidence=None):
    """Draw bounding box and label on frame."""
    if bbox is None:
        return frame

    x, y, w, h = bbox
    color = (0, 255, 0)  # Green

    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Draw label if provided
    if label:
        text = f"{label}"
        if confidence is not None:
            text += f": {confidence:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Background rectangle
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)

        # Text
        cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)

    return frame


def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on frame."""
    if landmarks is None:
        return frame

    # Key points to draw
    key_indices = [33, 133, 362, 263, 1, 61, 291, 199]

    for idx in key_indices:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    return frame

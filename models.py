"""
Expression Recognition Models Module

Provides FER and DeepFace wrappers for facial expression recognition.
"""

import time
import numpy as np


# ============== EXPRESSION LABELS ==============
EXPRESSION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Label mapping for different libraries
LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}


# ============== GLOBAL MODEL INSTANCES ==============
_fer_detector = None
_deepface_module = None


# ============== FER MODEL ==============
def load_fer_model():
    """Load FER model."""
    global _fer_detector

    if _fer_detector is not None:
        return True

    try:
        # Try different import methods for different FER versions
        try:
            from fer import FER
        except ImportError:
            try:
                from fer.fer import FER
            except ImportError:
                from fer.classes import FER

        print("Loading FER model...")
        _fer_detector = FER(mtcnn=False)
        print("FER model loaded successfully")
        return True

    except ImportError as e:
        print(f"FER library not installed: {e}")
        print("Install with: pip install fer")
        return False
    except Exception as e:
        print(f"Failed to load FER model: {e}")
        return False


def predict_with_fer(face_image):
    """
    Predict expression using FER model.

    Args:
        face_image: BGR face image (cropped)

    Returns:
        dict: {expression, confidence, all_scores, time_ms} or None
    """
    global _fer_detector

    if _fer_detector is None:
        if not load_fer_model():
            return None

    if face_image is None or face_image.size == 0:
        return None

    try:
        start_time = time.time()

        # Detect emotions
        results = _fer_detector.detect_emotions(face_image)

        processing_time = (time.time() - start_time) * 1000

        if not results:
            return None

        # Get emotions from first face
        emotions = results[0].get("emotions", {})
        if not emotions:
            return None

        # Map to standard labels
        all_scores = {}
        for label, score in emotions.items():
            std_label = LABEL_MAP.get(label, label.capitalize())
            all_scores[std_label] = float(score)

        # Find dominant expression
        dominant = max(all_scores, key=all_scores.get)
        confidence = all_scores[dominant]

        return {
            "expression": dominant,
            "confidence": confidence,
            "all_scores": all_scores,
            "time_ms": processing_time,
            "model": "FER"
        }

    except Exception as e:
        print(f"FER prediction error: {e}")
        return None


# ============== DEEPFACE MODEL ==============
def load_deepface_model():
    """Load DeepFace model."""
    global _deepface_module

    if _deepface_module is not None:
        return True

    try:
        from deepface import DeepFace

        print("Loading DeepFace model...")
        _deepface_module = DeepFace

        # Warm up with dummy image
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            _deepface_module.analyze(
                dummy,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )
        except:
            pass

        print("DeepFace model loaded successfully")
        return True

    except ImportError as e:
        print(f"DeepFace library not installed: {e}")
        print("Install with: pip install deepface tf-keras")
        return False
    except Exception as e:
        print(f"Failed to load DeepFace model: {e}")
        return False


def predict_with_deepface(face_image):
    """
    Predict expression using DeepFace model.

    Args:
        face_image: BGR face image (cropped)

    Returns:
        dict: {expression, confidence, all_scores, time_ms} or None
    """
    global _deepface_module

    if _deepface_module is None:
        if not load_deepface_model():
            return None

    if face_image is None or face_image.size == 0:
        return None

    try:
        start_time = time.time()

        # Analyze face
        results = _deepface_module.analyze(
            face_image,
            actions=["emotion"],
            detector_backend="skip",  # We already detected the face
            enforce_detection=False,
            silent=True
        )

        processing_time = (time.time() - start_time) * 1000

        # Handle list or dict result
        if isinstance(results, list):
            if not results:
                return None
            result = results[0]
        else:
            result = results

        emotions = result.get("emotion", {})
        if not emotions:
            return None

        # Map to standard labels (DeepFace returns percentages)
        all_scores = {}
        for label, score in emotions.items():
            std_label = LABEL_MAP.get(label.lower(), label.capitalize())
            all_scores[std_label] = float(score) / 100.0  # Convert to 0-1 range

        # Find dominant expression
        dominant = result.get("dominant_emotion", "").capitalize()
        dominant = LABEL_MAP.get(dominant.lower(), dominant)

        if dominant not in all_scores:
            dominant = max(all_scores, key=all_scores.get)

        confidence = all_scores.get(dominant, 0.0)

        return {
            "expression": dominant,
            "confidence": confidence,
            "all_scores": all_scores,
            "time_ms": processing_time,
            "model": "DeepFace"
        }

    except Exception as e:
        print(f"DeepFace prediction error: {e}")
        return None


# ============== UNIFIED PREDICTION FUNCTION ==============
def predict_expression(face_image, model_name="FER"):
    """
    Predict expression using the specified model.

    Args:
        face_image: BGR face image (cropped)
        model_name: "FER" or "DeepFace"

    Returns:
        dict: {expression, confidence, all_scores, time_ms, model} or None
    """
    if model_name == "FER":
        return predict_with_fer(face_image)
    elif model_name == "DeepFace":
        return predict_with_deepface(face_image)
    else:
        print(f"Unknown model: {model_name}")
        return None


# ============== MODEL LIST ==============
AVAILABLE_MODELS = ["FER", "DeepFace"]


def get_available_models():
    """Get list of available model names."""
    return AVAILABLE_MODELS

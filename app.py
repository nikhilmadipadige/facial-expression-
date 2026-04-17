"""
Facial Expression Detection - Streamlit Application

A simple real-time facial expression detection app using FER and DeepFace.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import time

# Import our modules
from camera import (
    get_camera, release_camera, read_frame,
    detect_face, draw_face_box, draw_landmarks,
    init_face_detector
)
from models import (
    predict_expression, get_available_models,
    load_fer_model, load_deepface_model
)


# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Facial Expression Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============== EXPRESSION COLORS ==============
EXPRESSION_COLORS = {
    "Angry": "#FF4B4B",
    "Disgust": "#8B4513",
    "Fear": "#9932CC",
    "Happy": "#FFD700",
    "Sad": "#4169E1",
    "Surprise": "#FF6347",
    "Neutral": "#808080"
}


# ============== SIDEBAR ==============
def render_sidebar():
    """Render sidebar controls and return settings."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")

        # Model Selection
        st.subheader("🤖 Expression Model")
        selected_model = st.selectbox(
            "Select Algorithm",
            options=get_available_models(),
            help="Choose the facial expression recognition algorithm"
        )

        # Model descriptions
        if selected_model == "FER":
            st.info("**FER Library**\n- Pre-trained CNN model\n- Fast inference")
        else:
            st.info("**DeepFace**\n- Multiple backends\n- Robust detection")

        st.markdown("---")

        # Camera Controls
        st.subheader("📷 Camera")
        camera_enabled = st.toggle("Enable Camera", value=True)

        st.markdown("---")

        # Display Options
        st.subheader("🎨 Display")
        show_landmarks = st.checkbox("Show Landmarks", value=True)
        show_all_scores = st.checkbox("Show All Scores", value=True)

        st.markdown("---")

        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        **Expressions Detected:**
        - Angry, Disgust, Fear
        - Happy, Sad, Surprise
        - Neutral

        Uses MediaPipe for face detection.
        """)

    return {
        "model": selected_model,
        "camera_enabled": camera_enabled,
        "show_landmarks": show_landmarks,
        "show_all_scores": show_all_scores
    }


# ============== MAIN VIEW ==============
def render_header():
    """Render the main header."""
    st.title("😊 Facial Expression Detection")
    st.markdown("Real-time facial expression recognition using machine learning")


def render_expression_result(result, show_all_scores=True):
    """Render expression recognition result."""
    if result is None:
        st.warning("No expression detected")
        return

    expression = result["expression"]
    confidence = result["confidence"]
    color = EXPRESSION_COLORS.get(expression, "#808080")

    # Main expression display
    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border-left: 4px solid {color};
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        ">
            <h2 style="margin: 0; color: {color};">{expression}</h2>
            <p style="margin: 5px 0 0 0; font-size: 1.2em;">
                Confidence: <strong>{confidence:.1%}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # All scores
    if show_all_scores and "all_scores" in result:
        st.markdown("**All Expressions:**")
        sorted_scores = sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for expr, score in sorted_scores:
            st.progress(score, text=f"{expr}: {score:.1%}")

    # Performance
    st.markdown("---")
    st.caption(f"Processing time: {result.get('time_ms', 0):.0f}ms | Model: {result.get('model', 'N/A')}")


# ============== MAIN LOOP ==============
def run_camera_loop(settings):
    """Run the camera capture and analysis loop."""

    # Initialize face detector
    with st.spinner("Initializing face detector..."):
        init_face_detector()

    # Pre-load selected model
    with st.spinner(f"Loading {settings['model']} model..."):
        if settings["model"] == "FER":
            load_fer_model()
        else:
            load_deepface_model()

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Camera Feed")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("📊 Analysis Results")
        results_placeholder = st.empty()

    # Stop button
    stop_button = st.button("⏹️ Stop Camera", key="stop_camera")

    # FPS tracking
    fps_times = []

    # Main loop
    while settings["camera_enabled"] and not stop_button:
        # Track FPS
        loop_start = time.time()

        # Read frame
        frame = read_frame()
        if frame is None:
            st.error("Failed to read from camera")
            break

        # Detect face
        face_image, bbox, landmarks = detect_face(frame)

        # Predict expression
        result = None
        if face_image is not None:
            result = predict_expression(face_image, settings["model"])

        # Draw annotations
        if bbox is not None:
            label = result["expression"] if result else None
            conf = result["confidence"] if result else None
            frame = draw_face_box(frame, bbox, label, conf)

            if settings["show_landmarks"] and landmarks:
                frame = draw_landmarks(frame, landmarks)

        # Calculate FPS
        fps_times.append(time.time() - loop_start)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0

        # Draw FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update display
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        with results_placeholder.container():
            if result:
                render_expression_result(result, settings["show_all_scores"])
            elif bbox is not None:
                st.info("Analyzing expression...")
            else:
                st.warning("No face detected")
                st.markdown(
                    """
                    <div style="
                        background-color: #f0f0f0;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                    ">
                        <p style="color: #888;">
                            Position your face in front of the camera
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Small delay
        time.sleep(0.01)

        # Check stop button
        if stop_button:
            break

    # Cleanup
    release_camera()


def render_camera_off():
    """Render placeholder when camera is off."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Camera Feed")
        st.info("Camera is disabled. Enable it from the sidebar.")

        # Placeholder image
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (50, 50, 50)
        cv2.putText(
            placeholder, "Camera Off",
            (220, 250), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (150, 150, 150), 2
        )
        st.image(placeholder, channels="RGB", use_container_width=True)

    with col2:
        st.subheader("📊 Analysis Results")
        st.warning("Enable camera to start detection")


# ============== MAIN ==============
def main():
    """Main application entry point."""

    # Render header
    render_header()

    # Render sidebar and get settings
    settings = render_sidebar()

    # Run camera or show placeholder
    if settings["camera_enabled"]:
        run_camera_loop(settings)
    else:
        render_camera_off()
        release_camera()


if __name__ == "__main__":
    main()

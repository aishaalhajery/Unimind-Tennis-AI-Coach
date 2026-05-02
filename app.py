"""
Tennis Coach Live Demo — Streamlit App

Upload a tennis image to get action classification and posture feedback.
"""

import os
import numpy as np
import joblib
import streamlit as st
from PIL import Image

from pose_utils import (
    extract_landmarks,
    compute_joint_angles,
    generate_feedback,
    draw_pose_on_image,
)

MODEL_PATH = "random_forest_tennis_model.pkl"
SCALER_PATH = "scaler.pkl"


@st.cache_resource
def load_model():
    """Load the trained model and scaler if available."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


def main():
    st.set_page_config(page_title="Tennis Coach AI", layout="wide")

    # --- Sidebar ---
    with st.sidebar:
        st.title("Tennis Coach AI")
        st.markdown(
            "Upload a photo of a tennis player to get:\n"
            "- **Action classification** (forehand, backhand, serve, ready position)\n"
            "- **Posture correction feedback**"
        )
        st.divider()

        model, scaler = load_model()
        if model is not None:
            st.success("Model loaded")
        else:
            st.warning(
                "Model files not found. Place `random_forest_tennis_model.pkl` "
                "and `scaler.pkl` in the `live_demo/` directory. "
                "Pose detection and feedback still work without them."
            )

        st.divider()
        st.caption("UniMind Project — AI Tennis Coach")

    # --- Main area ---
    st.header("Upload a Tennis Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)

    # --- Pose detection ---
    with st.spinner("Detecting pose..."):
        features, pose_landmarks = extract_landmarks(image_rgb)

    if features is None:
        col1, _ = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
        st.error(
            "No pose detected. Make sure the image clearly shows a person's full body."
        )
        return

    annotated_image = draw_pose_on_image(image_rgb, pose_landmarks)

    # --- Display images side by side ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original Image", use_container_width=True)
    with col2:
        st.image(annotated_image, caption="Detected Pose", use_container_width=True)

    st.divider()

    # --- Classification ---
    predicted_action = None
    if model is not None and scaler is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        predicted_action = model.predict(features_scaled)[0]
        st.metric(label="Predicted Action", value=predicted_action.replace("_", " ").title())
    else:
        st.info("Classification skipped — model files not loaded.")

    st.divider()

    # --- Joint angles ---
    angles = compute_joint_angles(pose_landmarks)

    st.subheader("Joint Angles")
    angle_cols = st.columns(len(angles))
    for i, (joint, angle) in enumerate(angles.items()):
        with angle_cols[i]:
            label = joint.replace("_", " ").title()
            st.metric(label=label, value=f"{angle}°")

    st.divider()

    # --- Feedback ---
    st.subheader("Posture Feedback")
    action_for_feedback = predicted_action if predicted_action else "ready_position"

    if predicted_action is None:
        st.caption(
            "No model loaded — showing feedback against 'ready position' as default."
        )

    feedback_items = generate_feedback(action_for_feedback, angles)
    for item in feedback_items:
        if "Great form" in item:
            st.success(item)
        else:
            st.warning(item)


if __name__ == "__main__":
    main()

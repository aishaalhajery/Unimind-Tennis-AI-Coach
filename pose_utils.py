"""
Shared utilities for the Tennis Coach Live Demo.

Provides MediaPipe pose extraction, joint angle calculation,
posture feedback generation, and pose visualization.
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ACTIONS = ["backhand", "forehand", "ready_position", "serve"]

# MediaPipe landmark indices used for angle calculations
LANDMARK = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Ideal angle ranges per action (degrees).
# Format: (min_angle, max_angle)
IDEAL_ANGLE_RANGES = {
    "serve": {
        "right_elbow": (140, 180),
        "left_elbow": (90, 170),
        "right_knee": (140, 175),
        "left_knee": (140, 175),
        "right_shoulder": (140, 180),
        "left_shoulder": (60, 140),
    },
    "forehand": {
        "right_elbow": (90, 155),
        "left_elbow": (100, 170),
        "right_knee": (130, 175),
        "left_knee": (130, 175),
        "right_shoulder": (60, 130),
        "left_shoulder": (40, 100),
    },
    "backhand": {
        "right_elbow": (100, 165),
        "left_elbow": (90, 155),
        "right_knee": (130, 175),
        "left_knee": (130, 175),
        "right_shoulder": (50, 130),
        "left_shoulder": (60, 140),
    },
    "ready_position": {
        "right_elbow": (100, 155),
        "left_elbow": (100, 155),
        "right_knee": (120, 155),
        "left_knee": (120, 155),
        "right_shoulder": (30, 75),
        "left_shoulder": (30, 75),
    },
}

FEEDBACK_TEMPLATES = {
    "right_elbow": {
        "too_low": "Extend your right arm more — your elbow is too bent.",
        "too_high": "Bend your right elbow a bit more — your arm is too straight.",
    },
    "left_elbow": {
        "too_low": "Extend your left arm more — your elbow is too bent.",
        "too_high": "Bend your left elbow a bit more — your arm is too straight.",
    },
    "right_knee": {
        "too_low": "Your right knee is bent too much — straighten up slightly.",
        "too_high": "Bend your right knee more — your stance is too upright.",
    },
    "left_knee": {
        "too_low": "Your left knee is bent too much — straighten up slightly.",
        "too_high": "Bend your left knee more — your stance is too upright.",
    },
    "right_shoulder": {
        "too_low": "Raise your right arm higher.",
        "too_high": "Lower your right arm a bit.",
    },
    "left_shoulder": {
        "too_low": "Raise your left arm higher.",
        "too_high": "Lower your left arm a bit.",
    },
}


def extract_landmarks(image_rgb):
    """Run MediaPipe Pose on an RGB image and return the 99-feature vector.

    Returns:
        features: np.ndarray of shape (99,) with x,y,z for 33 landmarks,
                  or None if no pose was detected.
        pose_landmarks: the raw MediaPipe pose_landmarks object (for drawing),
                        or None.
    """
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features), results.pose_landmarks


def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at point b given three points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _get_point(landmarks, idx):
    """Get (x, y) for a landmark by index."""
    lm = landmarks.landmark[idx]
    return [lm.x, lm.y]


def compute_joint_angles(pose_landmarks):
    """Compute key joint angles from MediaPipe pose landmarks.

    Returns a dict mapping joint name to angle in degrees.
    """
    angles = {}

    angle_definitions = {
        "right_elbow": (
            LANDMARK["right_shoulder"],
            LANDMARK["right_elbow"],
            LANDMARK["right_wrist"],
        ),
        "left_elbow": (
            LANDMARK["left_shoulder"],
            LANDMARK["left_elbow"],
            LANDMARK["left_wrist"],
        ),
        "right_knee": (
            LANDMARK["right_hip"],
            LANDMARK["right_knee"],
            LANDMARK["right_ankle"],
        ),
        "left_knee": (
            LANDMARK["left_hip"],
            LANDMARK["left_knee"],
            LANDMARK["left_ankle"],
        ),
        "right_shoulder": (
            LANDMARK["right_hip"],
            LANDMARK["right_shoulder"],
            LANDMARK["right_elbow"],
        ),
        "left_shoulder": (
            LANDMARK["left_hip"],
            LANDMARK["left_shoulder"],
            LANDMARK["left_elbow"],
        ),
    }

    for name, (idx_a, idx_b, idx_c) in angle_definitions.items():
        a = _get_point(pose_landmarks, idx_a)
        b = _get_point(pose_landmarks, idx_b)
        c = _get_point(pose_landmarks, idx_c)
        angles[name] = round(calculate_angle(a, b, c), 1)

    return angles


def generate_feedback(action, angles):
    """Compare computed angles against ideal ranges for the predicted action.

    Returns a list of feedback strings. An empty list means good form.
    """
    if action not in IDEAL_ANGLE_RANGES:
        return ["Unknown action — cannot provide feedback."]

    ideal = IDEAL_ANGLE_RANGES[action]
    feedback = []

    for joint, (lo, hi) in ideal.items():
        if joint not in angles:
            continue
        value = angles[joint]
        templates = FEEDBACK_TEMPLATES.get(joint, {})
        if value < lo:
            msg = templates.get("too_low", f"Adjust your {joint.replace('_', ' ')}.")
            feedback.append(f"{msg} (current: {value}°, ideal: {lo}°–{hi}°)")
        elif value > hi:
            msg = templates.get("too_high", f"Adjust your {joint.replace('_', ' ')}.")
            feedback.append(f"{msg} (current: {value}°, ideal: {lo}°–{hi}°)")

    if not feedback:
        feedback.append("Great form! Your posture looks good for this action.")

    return feedback


def draw_pose_on_image(image_rgb, pose_landmarks):
    """Draw the MediaPipe pose skeleton on a copy of the image.

    Returns the annotated image (RGB).
    """
    annotated = image_rgb.copy()
    mp_drawing.draw_landmarks(
        annotated,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated

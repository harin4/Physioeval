"""
pose_detector.py — Enhanced MediaPipe pose detection with skeleton overlay,
angle annotation, and per-frame timestamp tracking.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, Dict
import tempfile
import os

from app.core.logger import logger
from app.core.config import settings
from app.models.schemas import PoseData, PoseKeypoint


class PoseDetector:
    """
    MediaPipe-based pose detection service.
    Extracts body keypoints, draws annotated skeleton overlays,
    and tracks per-frame timestamps for signal analysis.
    """

    LANDMARK_NAMES = {
        0: "nose",
        11: "left_shoulder",  12: "right_shoulder",
        13: "left_elbow",     14: "right_elbow",
        15: "left_wrist",     16: "right_wrist",
        23: "left_hip",       24: "right_hip",
        25: "left_knee",      26: "right_knee",
        27: "left_ankle",     28: "right_ankle",
    }

    # Colours (BGR)
    SKELETON_COLOR = (0, 255, 128)    # neon green
    ANGLE_COLOR    = (255, 220, 0)    # gold
    WARN_COLOR     = (0, 60, 255)     # red
    TEXT_BG_COLOR  = (20, 20, 20)

    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_style   = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("PoseDetector initialised (enhanced v2)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_image(self, image_path: str) -> Optional[PoseData]:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        return self._process_frame(frame, frame_index=0, timestamp_ms=0.0)

    def detect_from_frame(self, frame: np.ndarray, frame_index: int = 0) -> Optional[PoseData]:
        return self._process_frame(frame, frame_index, timestamp_ms=float(frame_index) * 33.3)

    def detect_from_video(
        self,
        video_path: str,
        sample_rate: int = 3,
        annotated_output_path: Optional[str] = None,
        blur_faces: bool = False,
    ) -> List[PoseData]:
        """
        Detect poses from video.
        If annotated_output_path is provided, writes an annotated video there.
        sample_rate: analyse every Nth frame (annotation draws every frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps      = cap.get(cv2.CAP_PROP_FPS) or settings.ANNOTATED_VIDEO_FPS
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if annotated_output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_output_path, fourcc, fps, (width, height))

        results: List[PoseData] = []
        frame_count = 0

        # rolling angle for annotation overlay
        current_angle: Optional[float] = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if frame_count % sample_rate == 0:
                pose_data = self._process_frame(frame, frame_index=frame_count,
                                                timestamp_ms=timestamp_ms)
                if pose_data:
                    results.append(pose_data)
                    ang = extract_shoulder_angle(pose_data)
                    if ang is not None:
                        current_angle = ang

            # Always annotate for smooth video output
            if writer:
                annotated = self._draw_overlay(frame, current_angle)
                if blur_faces:
                    annotated = self._blur_frame_faces(annotated)
                writer.write(annotated)

            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        logger.info(f"Video: {frame_count} frames, {len(results)} analysed, "
                    f"annotated={'yes' if annotated_output_path else 'no'}")
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp_ms: float
    ) -> Optional[PoseData]:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return PoseData(keypoints={}, pose_detected=False,
                            frame_index=frame_index, timestamp_ms=timestamp_ms)

        keypoints = {}
        for idx, name in self.LANDMARK_NAMES.items():
            lm = result.pose_landmarks.landmark[idx]
            keypoints[name] = PoseKeypoint(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
            )

        return PoseData(keypoints=keypoints, pose_detected=True,
                        frame_index=frame_index, timestamp_ms=timestamp_ms)

    def _draw_overlay(self, frame: np.ndarray, angle: Optional[float]) -> np.ndarray:
        """Draw skeleton lines, joint dots, and angle badge onto frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            # Draw MediaPipe skeleton
            self.mp_drawing.draw_landmarks(
                out,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=self.SKELETON_COLOR, thickness=2, circle_radius=4
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(200, 200, 200), thickness=2
                ),
            )

        # Angle badge (top-left)
        if angle is not None:
            label = f"Shoulder Angle: {angle:.1f} deg"
            color = self.SKELETON_COLOR if angle >= settings.ARM_RAISE_CORRECT_ANGLE else (
                self.ANGLE_COLOR if angle >= settings.ARM_RAISE_PARTIAL_ANGLE else self.WARN_COLOR
            )
            _draw_text_badge(out, label, (16, 36), color)

        # Header bar
        _draw_text_badge(out, "PhysioEval Enhanced v2", (16, h - 20),
                         (180, 180, 180), font_scale=0.45)
        return out

    def _blur_frame_faces(self, frame: np.ndarray) -> np.ndarray:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (99, 99), 30)
        return frame

    def blur_image_faces(self, input_path: str, output_path: str) -> None:
        frame = cv2.imread(input_path)
        if frame is not None:
            blurred = self._blur_frame_faces(frame)
            cv2.imwrite(output_path, blurred)

    def close(self):
        self.pose.close()


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def calculate_angle(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> float:
    """Angle at B formed by A-B-C, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return round(float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))), 2)


def extract_shoulder_angle(pose_data: PoseData) -> Optional[float]:
    """Shoulder raise angle: elbow–shoulder–hip (best-visibility side)."""
    kp = pose_data.keypoints
    left_ok  = all(k in kp for k in ["left_shoulder",  "left_elbow",  "left_hip"])
    right_ok = all(k in kp for k in ["right_shoulder", "right_elbow", "right_hip"])
    if not left_ok and not right_ok:
        return None

    def vis(side):
        return sum(kp[f"{side}_{j}"].visibility
                   for j in ["shoulder", "elbow", "hip"])

    if left_ok and right_ok:
        side = "left" if vis("left") >= vis("right") else "right"
    else:
        side = "left" if left_ok else "right"

    shoulder = (kp[f"{side}_shoulder"].x, kp[f"{side}_shoulder"].y)
    elbow    = (kp[f"{side}_elbow"].x,    kp[f"{side}_elbow"].y)
    hip      = (kp[f"{side}_hip"].x,      kp[f"{side}_hip"].y)
    return calculate_angle(elbow, shoulder, hip)


def extract_hip_x(pose_data: PoseData) -> Optional[float]:
    """Returns normalised mean hip x-coordinate (for compensation detection)."""
    kp = pose_data.keypoints
    xs = [kp[k].x for k in ["left_hip", "right_hip"] if k in kp]
    return float(np.mean(xs)) if xs else None


# ---------------------------------------------------------------------------
# Drawing utility
# ---------------------------------------------------------------------------

def _draw_text_badge(
    img: np.ndarray, text: str, origin: Tuple[int, int],
    color=(255, 255, 255), font_scale: float = 0.55, thickness: int = 1
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    pad = 4
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

"""
scoring_engine.py — Enhanced rule-based + signal-aware scoring engine.

Integrates:
  - Classic angle-based scoring (arm raise)
  - SignalAnalyzer results (tremor, fatigue, compensation)
  - SageMaker disorder probability (when enabled)
  - Clinical recommendation generation
"""
from __future__ import annotations

import statistics
from typing import List, Optional, Tuple

from app.models.schemas import (
    EvaluationStatus, JointAngles, ExerciseType,
    SignalAnalysis, TremorLevel, FatigueLevel,
)
from app.services.pose_detector import PoseData, extract_shoulder_angle, extract_hip_x
from app.services.signal_analyzer import SignalAnalyzer, call_sagemaker_classifier
from app.core.config import settings
from app.core.logger import logger


class ScoringEngine:
    """
    Comprehensive scoring engine combining classical rules with clinical signals.
    """

    def __init__(self):
        self.signal_analyzer = SignalAnalyzer()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        exercise_type: ExerciseType,
        pose_frames: List[PoseData],
        fps: float = 15.0,
    ) -> Tuple[float, EvaluationStatus, str, Optional[JointAngles],
               Optional[SignalAnalysis], List[str]]:
        """
        Returns:
            (score, status, feedback, joint_angles, signal_analysis, recommendations)
        """
        if exercise_type == ExerciseType.ARM_RAISE:
            return self._evaluate_arm_raise(pose_frames, fps)
        raise ValueError(f"Unsupported exercise type: {exercise_type}")

    def evaluate_arm_raise(
        self,
        pose_frames: List[PoseData],
        fps: float = 15.0,
    ) -> Tuple[float, EvaluationStatus, str, Optional[JointAngles],
               Optional[SignalAnalysis], List[str]]:
        """Public wrapper for arm raise evaluation."""
        return self._evaluate_arm_raise(pose_frames, fps)

    # ------------------------------------------------------------------
    # Arm raise evaluation
    # ------------------------------------------------------------------

    def _evaluate_arm_raise(
        self, pose_frames: List[PoseData], fps: float
    ) -> Tuple[float, EvaluationStatus, str, Optional[JointAngles],
               Optional[SignalAnalysis], List[str]]:

        detected = [f for f in pose_frames if f.pose_detected]
        if not detected:
            return (
                0.0, EvaluationStatus.NO_POSE_DETECTED,
                "No human body detected. Ensure you are fully visible and well-lit.",
                None, None, [],
            )

        # --- Angle extraction ---
        angles = [extract_shoulder_angle(f) for f in detected]
        angles = [a for a in angles if a is not None]

        if not angles:
            return (
                0.0, EvaluationStatus.NO_POSE_DETECTED,
                "Could not compute shoulder angle. Ensure shoulders, elbows, and hips are visible.",
                None, None, [],
            )

        best_angle = max(angles)
        avg_angle  = statistics.mean(angles)
        logger.info(f"Arm raise — best: {best_angle:.1f}°, avg: {avg_angle:.1f}°")

        # --- Base score ---
        correct_thresh = settings.ARM_RAISE_CORRECT_ANGLE  # 150°
        partial_thresh = settings.ARM_RAISE_PARTIAL_ANGLE  # 90°

        if best_angle >= correct_thresh:
            base_score = self._map_score(best_angle, correct_thresh, 180.0, 80.0, 100.0)
            status     = EvaluationStatus.CORRECT
            feedback   = (
                f"✅ Excellent! Peak shoulder angle: {best_angle:.1f}° "
                f"(target ≥ {correct_thresh}°). Great range of motion!"
            )
        elif best_angle >= partial_thresh:
            base_score = self._map_score(best_angle, partial_thresh, correct_thresh, 40.0, 79.0)
            status     = EvaluationStatus.PARTIAL
            feedback   = (
                f"⚠️ Partial arm raise. Peak: {best_angle:.1f}° "
                f"(target ≥ {correct_thresh}°). "
                f"Raise arm {correct_thresh - best_angle:.0f}° higher."
            )
        else:
            base_score = self._map_score(best_angle, 0.0, partial_thresh, 0.0, 39.0)
            status     = EvaluationStatus.INCORRECT
            feedback   = (
                f"❌ Insufficient arm raise. Peak: {best_angle:.1f}° "
                f"(target ≥ {correct_thresh}°). "
                f"Consult your therapist if pain limits movement."
            )

        joint_angles = JointAngles(shoulder_angle=round(best_angle, 2))

        # --- Signal analysis ---
        hip_x_timeline = [extract_hip_x(f) for f in detected]
        hip_x_clean    = [x for x in hip_x_timeline if x is not None]

        signal = self.signal_analyzer.analyse(
            angle_timeline=angles,
            hip_x_timeline=hip_x_clean if hip_x_clean else None,
            fps=fps,
        )

        # Upgrade disorder probability via SageMaker if enabled
        if settings.SAGEMAKER_ENABLED:
            sm_prob, sm_label = call_sagemaker_classifier(signal.tremor_score, best_angle)
            if sm_prob > signal.disorder_probability:
                signal.disorder_probability = sm_prob
                signal.disorder_label = sm_label

        # --- Penalise score for clinical findings ---
        final_score = base_score
        if signal.tremor_level == TremorLevel.MODERATE:
            final_score *= 0.92
        elif signal.tremor_level == TremorLevel.SEVERE:
            final_score *= 0.80
        if signal.fatigue_level == FatigueLevel.MODERATE:
            final_score *= 0.94
        elif signal.fatigue_level == FatigueLevel.SEVERE:
            final_score *= 0.85
        if signal.compensation_detected:
            final_score *= 0.90

        final_score = round(max(0.0, min(100.0, final_score)), 1)

        # --- Recommendations ---
        recs = self._generate_recommendations(signal, status)

        return (final_score, status, feedback, joint_angles, signal, recs)

    # ------------------------------------------------------------------
    # Recommendation generator
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self, signal: SignalAnalysis, status: EvaluationStatus
    ) -> List[str]:
        recs: List[str] = []

        if signal.tremor_level in (TremorLevel.MODERATE, TremorLevel.SEVERE):
            recs.append(
                "🔴 Moderate-to-severe shakiness detected. "
                "Consider neurological assessment — tremor may indicate Parkinson's, "
                "essential tremor, or medication side effects."
            )
        elif signal.tremor_level == TremorLevel.MILD:
            recs.append(
                "🟡 Mild shakiness observed. Ensure adequate hydration and rest. "
                "Monitor over multiple sessions."
            )

        if signal.fatigue_level in (FatigueLevel.MODERATE, FatigueLevel.SEVERE):
            recs.append(
                "🔴 Significant muscle fatigue detected. "
                "Reduce repetitions per session and increase rest intervals. "
                "Inform your physiotherapist."
            )
        elif signal.fatigue_level == FatigueLevel.MILD:
            recs.append("🟡 Mild fatigue detected. Ensure adequate warm-up and recovery time.")

        if signal.compensation_detected:
            recs.append(
                "⚠️ Compensation pattern detected. "
                "Focus on isolating shoulder muscles. "
                "Your therapist may prescribe targeted rotator cuff strengthening."
            )

        if signal.rom_restriction_angle is not None:
            recs.append(
                f"🔴 Range of Motion restriction at ~{signal.rom_restriction_angle:.0f}°. "
                "This may indicate Adhesive Capsulitis (Frozen Shoulder). "
                "Consult your physiotherapist for manual therapy or corticosteroid options."
            )

        if signal.disorder_probability and signal.disorder_probability > 0.4:
            recs.append(
                f"⚕️ Clinical indicators suggest possible: {signal.disorder_label}. "
                "Please consult a medical professional for formal diagnosis."
            )

        if not recs:
            if status == EvaluationStatus.CORRECT:
                recs.append("✅ Movement quality is excellent. Continue current exercise plan.")
            else:
                recs.append("💪 Work on increasing range of motion with guided physiotherapy.")

        return recs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _map_score(
        value: float, in_min: float, in_max: float, out_min: float, out_max: float
    ) -> float:
        if in_max == in_min:
            return out_min
        ratio = (value - in_min) / (in_max - in_min)
        ratio = max(0.0, min(1.0, ratio))
        return out_min + ratio * (out_max - out_min)

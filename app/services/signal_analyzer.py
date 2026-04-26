"""
signal_analyzer.py — Clinical biomechanical signal analysis.

Detects:
  - Tremor / shakiness via jerk (3rd derivative of position) RMS
  - Muscle fatigue via velocity decay across repetitions
  - Compensation (body lean) via hip drift
  - ROM restriction (angle plateau before target)
  - Smoothness via spectral arc length (SAL) metric

All thresholds are configurable via settings.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from app.core.config import settings
from app.core.logger import logger
from app.models.schemas import SignalAnalysis, TremorLevel, FatigueLevel


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class SignalAnalyzer:
    """
    Analyses a time-series of joint angles and hip positions to produce
    a SignalAnalysis object containing clinical indicators.
    """

    def analyse(
        self,
        angle_timeline: List[float],
        hip_x_timeline: Optional[List[float]] = None,
        fps: float = 15.0,
    ) -> SignalAnalysis:
        """
        Parameters
        ----------
        angle_timeline   : shoulder angles (degrees) across sampled frames
        hip_x_timeline   : normalised hip x positions (same length as angles)
        fps              : effective sampling rate of the angle data
        """
        if len(angle_timeline) < 5:
            logger.warning("Not enough frames for signal analysis (need ≥ 5)")
            return SignalAnalysis(
                tremor_score=0.0,
                fatigue_score=0.0,
                angle_timeline=angle_timeline or [],
                smoothness_curve=[],
            )

        angles = np.array(angle_timeline, dtype=float)

        # 1. Tremor
        tremor_score, tremor_freq = self._detect_tremor(angles, fps)
        tremor_level = self._classify_tremor(tremor_score)

        # 2. Fatigue
        fatigue_score, velocity_decay = self._detect_fatigue(angles, fps)
        fatigue_level = self._classify_fatigue(fatigue_score)

        # 3. Compensation / body lean
        comp_detected, comp_details = self._detect_compensation(
            angles, hip_x_timeline
        )

        # 4. ROM restriction
        rom_restriction = self._detect_rom_restriction(angles)

        # 5. Smoothness curve (jerk profile, normalised)
        smoothness = self._compute_smoothness_curve(angles)

        # 6. Simple disorder probability heuristic
        # (replaced by SageMaker in cloud; this is the fallback)
        disorder_prob, disorder_label = self._disorder_heuristic(
            tremor_score, fatigue_score, rom_restriction
        )

        return SignalAnalysis(
            tremor_score=round(float(tremor_score), 2),
            tremor_level=tremor_level,
            tremor_frequency_hz=round(float(tremor_freq), 2) if tremor_freq else None,
            fatigue_score=round(float(fatigue_score), 2),
            fatigue_level=fatigue_level,
            velocity_decay_percent=round(float(velocity_decay), 2) if velocity_decay is not None else None,
            compensation_detected=comp_detected,
            compensation_details=comp_details,
            rom_restriction_angle=round(float(rom_restriction), 2) if rom_restriction is not None else None,
            smoothness_curve=[round(v, 3) for v in smoothness],
            angle_timeline=[round(a, 2) for a in angle_timeline],
            disorder_probability=round(float(disorder_prob), 3),
            disorder_label=disorder_label,
        )

    # ------------------------------------------------------------------
    # Tremor detection
    # ------------------------------------------------------------------

    def _detect_tremor(self, angles: np.ndarray, fps: float) -> Tuple[float, Optional[float]]:
        """
        Compute jerk (3rd derivative) RMS as tremor indicator.
        Also estimate dominant tremor frequency via FFT.
        Tremor score 0–100.
        """
        if len(angles) < 6:
            return 0.0, None

        # Low-pass filter to separate tremor from intentional movement
        nyq = fps / 2.0
        if nyq > 1.0:
            try:
                b, a = scipy_signal.butter(2, min(4.0 / nyq, 0.99), btype="low")
                smooth = scipy_signal.filtfilt(b, a, angles)
            except Exception:
                smooth = angles
        else:
            smooth = angles

        residual = angles - smooth   # high-freq noise = tremor signal

        # Jerk of residual
        dt   = 1.0 / fps
        vel  = np.gradient(residual, dt)
        acc  = np.gradient(vel, dt)
        jerk = np.gradient(acc, dt)
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))

        # Normalise to 0–100 score
        MAX_JERK = 5000.0  # empirically chosen upper bound (deg/s³)
        tremor_score = min(100.0, (jerk_rms / MAX_JERK) * 100.0)

        # Dominant tremor frequency via FFT
        tremor_freq = None
        if len(residual) >= 8:
            try:
                freqs = np.fft.rfftfreq(len(residual), d=dt)
                fft   = np.abs(np.fft.rfft(residual))
                # Only look at 2–12 Hz (pathological tremor range)
                mask  = (freqs >= 2.0) & (freqs <= 12.0)
                if mask.any():
                    peak_idx = np.argmax(fft[mask])
                    tremor_freq = float(freqs[mask][peak_idx])
            except Exception:
                pass

        return tremor_score, tremor_freq

    def _classify_tremor(self, score: float) -> TremorLevel:
        if score >= settings.TREMOR_SEVERE_THRESHOLD:
            return TremorLevel.SEVERE
        elif score >= settings.TREMOR_MODERATE_THRESHOLD:
            return TremorLevel.MODERATE
        elif score >= settings.TREMOR_MILD_THRESHOLD:
            return TremorLevel.MILD
        return TremorLevel.NONE

    # ------------------------------------------------------------------
    # Fatigue detection
    # ------------------------------------------------------------------

    def _detect_fatigue(self, angles: np.ndarray, fps: float) -> Tuple[float, Optional[float]]:
        """
        Detect fatigue as velocity decay across the session.
        Compare mean angular velocity in first quarter vs last quarter.
        Fatigue score 0–100.
        """
        if len(angles) < 8:
            return 0.0, None

        dt  = 1.0 / fps
        vel = np.abs(np.gradient(angles, dt))   # absolute angular velocity

        q = max(1, len(vel) // 4)
        early_vel = float(np.mean(vel[:q]))
        late_vel  = float(np.mean(vel[-q:]))

        if early_vel < 1e-3:
            return 0.0, 0.0

        decay_pct  = ((early_vel - late_vel) / early_vel) * 100.0
        decay_pct  = max(0.0, decay_pct)  # only count actual decay

        MAX_DECAY  = 80.0
        fatigue_score = min(100.0, (decay_pct / MAX_DECAY) * 100.0)

        return fatigue_score, decay_pct

    def _classify_fatigue(self, score: float) -> FatigueLevel:
        if score >= settings.FATIGUE_SEVERE_DECAY * (100.0 / 80.0):
            return FatigueLevel.SEVERE
        elif score >= settings.FATIGUE_MODERATE_DECAY * (100.0 / 80.0):
            return FatigueLevel.MODERATE
        elif score >= settings.FATIGUE_MILD_DECAY * (100.0 / 80.0):
            return FatigueLevel.MILD
        return FatigueLevel.NONE

    # ------------------------------------------------------------------
    # Compensation detection
    # ------------------------------------------------------------------

    def _detect_compensation(
        self,
        angles: np.ndarray,
        hip_x_timeline: Optional[List[float]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect body lean compensation: if hip drifts > threshold while
        arm raise is near peak angle, patient is using torso.
        """
        if not hip_x_timeline or len(hip_x_timeline) < 4:
            return False, None

        hip_x   = np.array(hip_x_timeline[:len(angles)])
        if len(hip_x) < 4:
            return False, None

        baseline = float(np.median(hip_x[:max(1, len(hip_x) // 4)]))
        drift    = np.abs(hip_x - baseline)
        max_drift = float(np.max(drift))

        if max_drift > settings.COMPENSATION_LEAN_THRESHOLD:
            direction = "left" if hip_x[np.argmax(drift)] < baseline else "right"
            detail = (
                f"⚠️ Compensation detected: torso lean to the {direction} "
                f"({max_drift:.3f} normalised units). "
                f"You may be using your back to raise your arm instead of your shoulder muscles."
            )
            return True, detail

        return False, None

    # ------------------------------------------------------------------
    # ROM restriction
    # ------------------------------------------------------------------

    def _detect_rom_restriction(self, angles: np.ndarray) -> Optional[float]:
        """
        Detect if movement plateaus well below target (possible frozen shoulder / capsulitis).
        Looks for a sustained plateau below the correct threshold.
        Returns the plateau angle, or None.
        """
        target = settings.ARM_RAISE_CORRECT_ANGLE
        peak   = float(np.max(angles))

        if peak >= target:
            return None   # reached target, no restriction

        # Check plateau: last 20% of frames near peak
        tail = angles[int(len(angles) * 0.7):]
        if len(tail) < 2:
            return None

        tail_std = float(np.std(tail))
        tail_mean = float(np.mean(tail))

        # If movement is "stuck" (std < 5°) well below target
        if tail_std < 5.0 and tail_mean < (target - 20.0):
            return round(tail_mean, 1)

        return None

    # ------------------------------------------------------------------
    # Smoothness (Spectral Arc Length proxy)
    # ------------------------------------------------------------------

    def _compute_smoothness_curve(self, angles: np.ndarray) -> List[float]:
        """
        Returns a normalised jerk-squared curve for the Gradio line plot.
        Positive = smooth region, Negative = jerky region.
        Normalised to [-1, 1].
        """
        if len(angles) < 4:
            return [0.0] * len(angles)

        jerk = np.gradient(np.gradient(np.gradient(angles)))
        j_abs = np.abs(jerk)
        max_j = float(np.max(j_abs)) if np.max(j_abs) > 0 else 1.0
        smoothness = 1.0 - (j_abs / max_j)   # 1=smooth, 0=jerky
        return smoothness.tolist()

    # ------------------------------------------------------------------
    # Simple disorder heuristic (SageMaker replacement fallback)
    # ------------------------------------------------------------------

    def _disorder_heuristic(
        self,
        tremor_score: float,
        fatigue_score: float,
        rom_restriction: Optional[float],
    ) -> Tuple[float, str]:
        """
        Rule-based disorder probability when SageMaker is unavailable.
        Returns (probability 0–1, label string).
        """
        score = 0.0
        labels = []

        if tremor_score >= 45:
            score += 0.35
            labels.append("Neurological Tremor")
        if fatigue_score >= 50:
            score += 0.25
            labels.append("Muscle Weakness / Fatigue")
        if rom_restriction is not None:
            score += 0.30
            labels.append("ROM Restriction / Frozen Shoulder (Adhesive Capsulitis)")

        score = min(1.0, score)
        label = " + ".join(labels) if labels else "No significant disorder indicated"

        return score, label


# ---------------------------------------------------------------------------
# SageMaker wrapper (calls real endpoint when enabled)
# ---------------------------------------------------------------------------

def call_sagemaker_classifier(
    tremor_score: float, shoulder_angle: float
) -> Tuple[float, str]:
    """
    Calls the SageMaker Random Forest endpoint.
    Falls back silently if endpoint unavailable.
    Returns (probability, label).
    """
    try:
        import boto3, json
        from app.core.config import settings

        if not settings.SAGEMAKER_ENABLED or not settings.SAGEMAKER_ENDPOINT:
            return 0.0, "SageMaker not configured"

        client  = boto3.client("sagemaker-runtime", region_name=settings.AWS_REGION)
        payload = json.dumps({"tremor_score": tremor_score, "angle": shoulder_angle})
        resp    = client.invoke_endpoint(
            EndpointName=settings.SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(resp["Body"].read())
        return float(result.get("probability", 0.0)), result.get("label", "Unknown")

    except Exception as e:
        logger.warning(f"SageMaker call failed (using heuristic): {e}")
        return 0.0, "SageMaker unavailable"

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ExerciseType(str, Enum):
    ARM_RAISE = "arm_raise"


class EvaluationStatus(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NO_POSE_DETECTED = "no_pose_detected"


class TremorLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class FatigueLevel(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class JointAngles(BaseModel):
    shoulder_angle: Optional[float] = None
    elbow_angle: Optional[float] = None
    hip_angle: Optional[float] = None
    additional: Optional[Dict[str, float]] = {}


class PoseKeypoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float


class PoseData(BaseModel):
    keypoints: Dict[str, PoseKeypoint]
    pose_detected: bool
    frame_index: Optional[int] = None
    timestamp_ms: Optional[float] = None


class SignalAnalysis(BaseModel):
    """Clinical-grade signal analysis results."""
    tremor_score: float = Field(ge=0.0, le=100.0, description="0=none, 100=severe tremor")
    tremor_level: TremorLevel = TremorLevel.NONE
    tremor_frequency_hz: Optional[float] = None
    fatigue_score: float = Field(ge=0.0, le=100.0, description="0=none, 100=severe fatigue")
    fatigue_level: FatigueLevel = FatigueLevel.NONE
    velocity_decay_percent: Optional[float] = None
    compensation_detected: bool = False
    compensation_details: Optional[str] = None
    rom_restriction_angle: Optional[float] = None
    smoothness_curve: Optional[List[float]] = None
    angle_timeline: Optional[List[float]] = None
    disorder_probability: Optional[float] = None
    disorder_label: Optional[str] = None


class EvaluationResponse(BaseModel):
    success: bool
    evaluation_id: str
    score: float
    status: EvaluationStatus
    feedback: str
    joint_angles: Optional[JointAngles] = None
    signal_analysis: Optional[SignalAnalysis] = None
    frames_analyzed: int
    timestamp: str
    face_blurred: bool = False
    message: str = "Evaluation complete"
    recommendations: List[str] = []
    annotated_video_url: Optional[str] = None


class HistoryRecord(BaseModel):
    evaluation_id: str
    exercise_type: str
    score: float
    status: str
    timestamp: str
    feedback: str
    tremor_level: Optional[str] = None
    fatigue_level: Optional[str] = None


class HistoryResponse(BaseModel):
    success: bool
    records: List[HistoryRecord]
    count: int

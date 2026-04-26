import os
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    EvaluationResponse, HistoryResponse, HistoryRecord,
    ExerciseType, EvaluationStatus,
)
from app.services.pose_detector import PoseDetector, extract_shoulder_angle
from app.services.scoring_engine import ScoringEngine
from app.services.storage_service import StorageService
from app.services.results_service import ResultsService
from app.services.ml_pipeline import MLPipeline
from app.monitoring.metrics import record_evaluation
from app.utils.validators import validate_upload_file, check_file_size
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()

pose_detector   = PoseDetector()
scoring_engine  = ScoringEngine()
storage_service = StorageService()
results_service = ResultsService()
ml_pipeline     = MLPipeline()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_exercise(
    file: UploadFile = File(..., description="Video (mp4/avi/mov) or image (jpg/png)"),
    exercise_type: ExerciseType = Form(default=ExerciseType.ARM_RAISE),
    session_id: Optional[str] = Form(default=None),
    blur_faces: bool = Form(default=True, description="Apply face blurring for HIPAA compliance"),
):
    """Upload a video or image and evaluate the exercise."""
    evaluation_id = str(uuid.uuid4())
    tmp_path = None

    try:
        media_type = validate_upload_file(file)
        logger.info(f"[{evaluation_id}] Received {media_type}: {file.filename}")

        suffix = os.path.splitext(file.filename or "upload")[-1] or (
            ".mp4" if media_type == "video" else ".jpg"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        check_file_size(tmp_path)
        media_url = storage_service.upload_file(tmp_path, file.filename or f"upload{suffix}")

        face_blurred = False
        if blur_faces and media_type == "image":
            blurred_path = tmp_path.replace(suffix, f"_blurred{suffix}")
            pose_detector.blur_image_faces(tmp_path, blurred_path)
            face_blurred = True

        fps = 10.0
        if media_type == "video":
            pose_frames = pose_detector.detect_from_video(tmp_path, sample_rate=3)
        else:
            single = pose_detector.detect_from_image(tmp_path)
            pose_frames = [single] if single else []

        frames_analyzed = len(pose_frames)
        score, status, feedback, joint_angles, signal, recs = scoring_engine.evaluate(
            exercise_type, pose_frames, fps=fps
        )

        # ── ML Pipeline: extract features & predict ──────────────────────────
        angle_timeline = signal.angle_timeline if signal and signal.angle_timeline else []
        hip_x_timeline = None
        if signal and signal.smoothness_curve:
            hip_x_timeline = signal.smoothness_curve  # reuse available timeline

        ml_result = ml_pipeline.predict(
            angle_timeline=angle_timeline,
            hip_x_timeline=hip_x_timeline,
        )
        ml_confidence   = ml_result.get("confidence")
        ml_is_anomaly   = ml_result.get("is_anomaly", False)
        ml_label        = ml_result.get("label", status.value)
        ml_available    = ml_result.get("ml_available", False)

        # If ML model disagrees strongly and is high-confidence, append note
        if ml_available and ml_label != status.value and (ml_confidence or 0) > 0.85:
            feedback += (
                f" [ML model ({ml_result.get('source','rf')}) "
                f"suggests '{ml_label}' with {ml_confidence:.0%} confidence]"
            )
        if ml_is_anomaly:
            recs.insert(0, "⚠️ Unusual movement pattern detected — please repeat the exercise clearly in frame.")

        # Trigger auto-retrain check (non-blocking — only logs)
        try:
            ml_pipeline.maybe_retrain(threshold=200)
        except Exception:
            pass

        # ── Prometheus metrics ───────────────────────────────────────────────
        record_evaluation(
            score=score,
            status=status.value,
            frames=frames_analyzed,
            ml_confidence=ml_confidence,
            is_anomaly=ml_is_anomaly,
        )

        timestamp = datetime.utcnow().isoformat()
        record = {
            "evaluation_id": evaluation_id,
            "exercise_type": exercise_type.value,
            "score": str(score),
            "status": status.value,
            "feedback": feedback,
            "joint_angles": joint_angles.model_dump() if joint_angles else {},
            "signal_analysis": signal.model_dump() if signal else {},
            "frames_analyzed": frames_analyzed,
            "media_url": media_url,
            "session_id": session_id or "",
            "timestamp": timestamp,
            "face_blurred": face_blurred,
            "ml_label": ml_label,
            "ml_confidence": ml_confidence,
            "ml_is_anomaly": ml_is_anomaly,
            "ml_available": ml_available,
        }
        results_service.save_result(record)

        return EvaluationResponse(
            success=True,
            evaluation_id=evaluation_id,
            score=score,
            status=status,
            feedback=feedback,
            joint_angles=joint_angles,
            signal_analysis=signal,
            frames_analyzed=frames_analyzed,
            timestamp=timestamp,
            face_blurred=face_blurred,
            recommendations=recs,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{evaluation_id}] Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/result/{evaluation_id}")
async def get_result(evaluation_id: str):
    result = results_service.get_result(evaluation_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found")
    return {"success": True, "result": result}


@router.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = Query(default=20, ge=1, le=100)):
    records_raw = results_service.get_all_results(limit=limit)
    records = [
        HistoryRecord(
            evaluation_id=r.get("evaluation_id", ""),
            exercise_type=r.get("exercise_type", ""),
            score=float(r.get("score", 0)),
            status=r.get("status", ""),
            timestamp=r.get("timestamp", ""),
            feedback=r.get("feedback", ""),
            tremor_level=r.get("signal_analysis", {}).get("tremor_level"),
            fatigue_level=r.get("signal_analysis", {}).get("fatigue_level"),
        )
        for r in records_raw
    ]
    return HistoryResponse(success=True, records=records, count=len(records))


@router.get("/exercises")
async def list_exercises():
    return {
        "exercises": [
            {
                "id": "arm_raise",
                "name": "Arm Raise",
                "description": "Raise arm to shoulder height or above",
                "correct_angle": f">= {settings.ARM_RAISE_CORRECT_ANGLE}°",
                "partial_angle": f"{settings.ARM_RAISE_PARTIAL_ANGLE}° – {settings.ARM_RAISE_CORRECT_ANGLE}°",
            }
        ],
        "signal_analysis_features": [
            "tremor_detection", "fatigue_analysis",
            "compensation_detection", "rom_restriction", "disorder_prediction",
        ],
        "privacy_features": ["face_blurring"],
    }


@router.get("/ml/status")
async def ml_status():
    from app.services.ml_pipeline import CLASSIFIER_PATH, SKLEARN_AVAILABLE, MLFLOW_AVAILABLE
    return {
        "ml_available": bool(SKLEARN_AVAILABLE and CLASSIFIER_PATH.exists()),
        "model_path": str(CLASSIFIER_PATH),
        "sklearn_available": bool(SKLEARN_AVAILABLE),
        "mlflow_available": bool(MLFLOW_AVAILABLE),
    }

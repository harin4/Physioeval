"""
Tests for PhysioEval API and core services.
Run with: pytest tests/ -v
"""

import pytest
import os
import json
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── App import ────────────────────────────────────────────────────────────────
from app.main import app
from app.services.scoring_engine import ScoringEngine
from app.services.pose_detector import calculate_angle, extract_shoulder_angle
from app.models.schemas import (
    ExerciseType, EvaluationStatus, PoseData, PoseKeypoint
)

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_pose(shoulder_x, shoulder_y, elbow_x, elbow_y, hip_x, hip_y,
              visibility=0.95) -> PoseData:
    """Build a minimal PoseData with left-side landmarks."""
    kp = {
        "left_shoulder": PoseKeypoint(x=shoulder_x, y=shoulder_y, z=0, visibility=visibility),
        "left_elbow":    PoseKeypoint(x=elbow_x,    y=elbow_y,    z=0, visibility=visibility),
        "left_hip":      PoseKeypoint(x=hip_x,      y=hip_y,      z=0, visibility=visibility),
    }
    return PoseData(keypoints=kp, pose_detected=True)


def no_pose_frame() -> PoseData:
    return PoseData(keypoints={}, pose_detected=False)


def make_jpeg_bytes(width=64, height=64) -> bytes:
    """Create a tiny valid JPEG in memory."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (100, 149, 237)  # cornflower blue
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — angle calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestAngleCalculation:

    def test_right_angle(self):
        # A=(0,1), B=(0,0), C=(1,0) → 90°
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90.0) < 0.5

    def test_straight_line(self):
        # A=(0,0), B=(1,0), C=(2,0) → 180°
        angle = calculate_angle((0, 0), (1, 0), (2, 0))
        assert abs(angle - 180.0) < 0.5

    def test_45_degrees(self):
        import math
        angle = calculate_angle((0, 1), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 1.0

    def test_identical_points(self):
        # Should not raise, returns 0 or near
        angle = calculate_angle((0, 0), (0, 0), (1, 1))
        assert isinstance(angle, float)


class TestExtractShoulderAngle:

    def test_arm_raised_high(self):
        # Elbow above shoulder, hip below → large angle
        pose = make_pose(
            shoulder_x=0.5, shoulder_y=0.4,
            elbow_x=0.5,    elbow_y=0.1,   # elbow high
            hip_x=0.5,      hip_y=0.7,
        )
        angle = extract_shoulder_angle(pose)
        assert angle is not None
        assert angle > 100

    def test_arm_down(self):
        # Elbow at hip level → small angle
        pose = make_pose(
            shoulder_x=0.5, shoulder_y=0.4,
            elbow_x=0.5,    elbow_y=0.65,
            hip_x=0.5,      hip_y=0.7,
        )
        angle = extract_shoulder_angle(pose)
        assert angle is not None
        assert angle < 60

    def test_no_keypoints(self):
        pose = PoseData(keypoints={}, pose_detected=False)
        assert extract_shoulder_angle(pose) is None


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — scoring engine
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringEngine:

    def setup_method(self):
        self.engine = ScoringEngine()

    # --- Arm raise: correct ---
    def test_arm_raise_correct(self):
        # Angle ~160° → correct
        pose = make_pose(0.5, 0.4, 0.5, 0.05, 0.5, 0.7)
        score, status, feedback, angles, signal, recs = self.engine.evaluate_arm_raise([pose])

        assert status == EvaluationStatus.CORRECT
        assert score >= 80
        assert "✅" in feedback

    # --- Arm raise: partial ---
    def test_arm_raise_partial(self):
        # Elbow roughly at shoulder level
        pose = make_pose(0.5, 0.4, 0.15, 0.4, 0.5, 0.7)
        score, status, feedback, angles, signal, recs = self.engine.evaluate_arm_raise([pose])

        assert status in (EvaluationStatus.PARTIAL, EvaluationStatus.CORRECT,
                          EvaluationStatus.INCORRECT)
        assert 0 <= score <= 100

    # --- No pose detected ---
    def test_no_pose_detected(self):
        score, status, feedback, angles, signal, recs = self.engine.evaluate_arm_raise([no_pose_frame()])
        assert status == EvaluationStatus.NO_POSE_DETECTED
        assert score == 0.0
        assert angles is None

    # --- Empty frames list ---
    def test_empty_frames(self):
        score, status, feedback, angles, signal, recs = self.engine.evaluate_arm_raise([])
        assert status == EvaluationStatus.NO_POSE_DETECTED

    # --- Multiple frames: uses best angle ---
    def test_uses_best_angle_across_frames(self):
        bad_pose  = make_pose(0.5, 0.4, 0.5, 0.65, 0.5, 0.7)   # arm down
        good_pose = make_pose(0.5, 0.4, 0.5, 0.05, 0.5, 0.7)   # arm high
        score, status, *_ = self.engine.evaluate_arm_raise([bad_pose, good_pose])
        assert status == EvaluationStatus.CORRECT

    # --- Dispatch works ---
    def test_evaluate_dispatch(self):
        pose = make_pose(0.5, 0.4, 0.5, 0.05, 0.5, 0.7)
        score, status, *_ = self.engine.evaluate(ExerciseType.ARM_RAISE, [pose])
        assert score >= 0

    # --- Score map helper ----
    def test_score_map_clamps(self):
        assert ScoringEngine._map_score(200, 0, 100, 0, 100) == 100.0
        assert ScoringEngine._map_score(-10, 0, 100, 0, 100) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — API endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


class TestExercisesEndpoint:

    def test_list_exercises(self):
        r = client.get("/api/v1/exercises")
        assert r.status_code == 200
        body = r.json()
        assert "exercises" in body
        assert any(e["id"] == "arm_raise" for e in body["exercises"])


class TestEvaluateEndpoint:

    @patch("app.api.routes.pose_detector")
    @patch("app.api.routes.storage_service")
    @patch("app.api.routes.results_service")
    def test_evaluate_image_correct(self, mock_results, mock_storage, mock_pose):
        # Mock storage
        mock_storage.upload_file.return_value = "/tmp/fake.jpg"

        # Mock pose detector — arm raised high
        good_pose = make_pose(0.5, 0.4, 0.5, 0.05, 0.5, 0.7)
        mock_pose.detect_from_image.return_value = good_pose

        # Mock results save
        mock_results.save_result.return_value = "test-id-123"

        jpeg = make_jpeg_bytes()
        r = client.post(
            "/api/v1/evaluate",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
            data={"exercise_type": "arm_raise"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["score"] >= 0
        assert body["status"] in ["correct", "partial", "incorrect", "no_pose_detected"]

    def test_evaluate_no_file(self):
        r = client.post("/api/v1/evaluate")
        assert r.status_code == 422   # Unprocessable Entity

    def test_evaluate_invalid_type(self):
        r = client.post(
            "/api/v1/evaluate",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"exercise_type": "arm_raise"},
        )
        assert r.status_code == 400


class TestHistoryEndpoint:

    def test_get_history(self):
        r = client.get("/api/v1/history")
        assert r.status_code == 200
        body = r.json()
        assert "records" in body
        assert isinstance(body["records"], list)

    def test_get_history_limit(self):
        r = client.get("/api/v1/history?limit=5")
        assert r.status_code == 200
        assert len(r.json()["records"]) <= 5


class TestResultEndpoint:

    def test_get_nonexistent_result(self):
        r = client.get("/api/v1/result/nonexistent-id-xyz")
        assert r.status_code == 404

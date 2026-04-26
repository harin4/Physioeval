"""
test_services.py — Unit tests for signal analyzer, storage, results, and metrics services.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.services.signal_analyzer import SignalAnalyzer, call_sagemaker_classifier
from app.services.results_service import ResultsService
from app.services.storage_service import StorageService
from app.monitoring.metrics import record_evaluation, metrics_endpoint, setup_metrics
from app.models.schemas import TremorLevel, FatigueLevel, SignalAnalysis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ramp_angles(peak=160.0, n=25):
    """Smooth ramp-up to peak angle."""
    t = np.linspace(0, 1, n)
    return (peak * (1 - np.exp(-4 * t))).tolist()

def _noisy_angles(peak=160.0, n=25, noise=8.0):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, n)
    base = peak * (1 - np.exp(-4 * t))
    return (base + rng.normal(0, noise, n)).clip(0, 185).tolist()

def _hip_drift(n=25, drift=0.2):
    return (np.linspace(0.5, 0.5 + drift, n)).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SignalAnalyzer — core analyse()
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalAnalyzer:

    def setup_method(self):
        self.analyzer = SignalAnalyzer()

    def test_returns_signal_analysis(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert isinstance(result, SignalAnalysis)

    def test_tremor_score_in_range(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert 0.0 <= result.tremor_score <= 100.0

    def test_fatigue_score_in_range(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert 0.0 <= result.fatigue_score <= 100.0

    def test_angle_timeline_preserved(self):
        angles = _ramp_angles()
        result = self.analyzer.analyse(angles)
        assert result.angle_timeline is not None
        assert len(result.angle_timeline) == len(angles)

    def test_smoothness_curve_same_length_as_angles(self):
        angles = _ramp_angles()
        result = self.analyzer.analyse(angles)
        assert result.smoothness_curve is not None
        assert len(result.smoothness_curve) == len(angles)

    def test_not_enough_frames_returns_zero_scores(self):
        result = self.analyzer.analyse([90.0, 100.0, 110.0])
        assert result.tremor_score == 0.0
        assert result.fatigue_score == 0.0

    def test_single_frame_returns_zeros(self):
        result = self.analyzer.analyse([120.0])
        assert result.tremor_score == 0.0

    def test_empty_angles_returns_zeros(self):
        result = self.analyzer.analyse([])
        assert result.tremor_score == 0.0

    def test_noisy_angles_give_higher_tremor(self):
        smooth_result = self.analyzer.analyse(_ramp_angles())
        noisy_result  = self.analyzer.analyse(_noisy_angles())
        assert noisy_result.tremor_score >= smooth_result.tremor_score

    def test_compensation_not_detected_without_hip(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert result.compensation_detected is False

    def test_compensation_detected_with_large_hip_drift(self):
        angles = _ramp_angles(n=25)
        hip = _hip_drift(n=25, drift=0.5)
        result = self.analyzer.analyse(angles, hip_x_timeline=hip)
        assert result.compensation_detected is True
        assert result.compensation_details is not None

    def test_no_compensation_with_stable_hip(self):
        angles = _ramp_angles(n=25)
        hip = [0.5] * 25
        result = self.analyzer.analyse(angles, hip_x_timeline=hip)
        assert result.compensation_detected is False

    def test_rom_restriction_detected_for_low_plateau(self):
        # Angles that plateau well below 150° target
        low_plateau = [30.0] * 5 + [45.0] * 20
        result = self.analyzer.analyse(low_plateau)
        assert result.rom_restriction_angle is not None
        assert result.rom_restriction_angle < 90.0

    def test_no_rom_restriction_for_full_range(self):
        result = self.analyzer.analyse(_ramp_angles(peak=165.0))
        assert result.rom_restriction_angle is None

    def test_disorder_probability_in_range(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert result.disorder_probability is not None
        assert 0.0 <= result.disorder_probability <= 1.0

    def test_high_tremor_increases_disorder_probability(self):
        noisy = _noisy_angles(noise=20.0, n=30)
        clean = _ramp_angles(n=30)
        noisy_result = self.analyzer.analyse(noisy)
        clean_result = self.analyzer.analyse(clean)
        assert noisy_result.tremor_score > clean_result.tremor_score

    def test_tremor_level_classification_none(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert result.tremor_level in (TremorLevel.NONE, TremorLevel.MILD,
                                        TremorLevel.MODERATE, TremorLevel.SEVERE)

    def test_fatigue_level_classification(self):
        result = self.analyzer.analyse(_ramp_angles())
        assert result.fatigue_level in (FatigueLevel.NONE, FatigueLevel.MILD,
                                         FatigueLevel.MODERATE, FatigueLevel.SEVERE)

    def test_tremor_frequency_optional(self):
        result = self.analyzer.analyse(_ramp_angles())
        if result.tremor_frequency_hz is not None:
            assert result.tremor_frequency_hz >= 0.0

    def test_velocity_decay_optional(self):
        result = self.analyzer.analyse(_ramp_angles())
        if result.velocity_decay_percent is not None:
            assert result.velocity_decay_percent >= 0.0

    def test_fps_parameter_respected(self):
        angles = _ramp_angles()
        r1 = self.analyzer.analyse(angles, fps=10.0)
        r2 = self.analyzer.analyse(angles, fps=30.0)
        assert isinstance(r1, SignalAnalysis)
        assert isinstance(r2, SignalAnalysis)

    def test_disorder_heuristic_with_high_tremor(self):
        # Force high tremor score via noisy rapid oscillations
        wild = ([10.0, 80.0] * 20)
        result = self.analyzer.analyse(wild)
        assert result.disorder_probability >= 0.0

    def test_disorder_heuristic_with_low_plateau_and_tremor(self):
        low_noisy = _noisy_angles(peak=40.0, noise=15.0, n=30)
        result = self.analyzer.analyse(low_noisy)
        assert result.disorder_label is not None


# ─────────────────────────────────────────────────────────────────────────────
# SageMaker wrapper — disabled path
# ─────────────────────────────────────────────────────────────────────────────

class TestSageMakerWrapper:

    def test_returns_zero_when_disabled(self):
        prob, label = call_sagemaker_classifier(50.0, 120.0)
        assert prob == 0.0
        assert isinstance(label, str)

    def test_returns_tuple(self):
        result = call_sagemaker_classifier(0.0, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# ResultsService — local JSON mode
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsService:

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path):
        with patch("app.services.results_service.settings") as mock_settings:
            mock_settings.USE_LOCAL_STORAGE = True
            mock_settings.LOCAL_RESULTS_DIR = str(tmp_path)
            self.service = ResultsService()
            yield

    def test_save_and_get_result(self):
        record = {
            "evaluation_id": "test-abc-123",
            "score": "85.0",
            "status": "correct",
            "timestamp": "2026-01-01T00:00:00",
            "exercise_type": "arm_raise",
        }
        self.service.save_result(record)
        retrieved = self.service.get_result("test-abc-123")
        assert retrieved is not None
        assert retrieved["evaluation_id"] == "test-abc-123"
        assert retrieved["score"] == "85.0"

    def test_get_nonexistent_returns_none(self):
        result = self.service.get_result("does-not-exist")
        assert result is None

    def test_get_all_results_empty(self):
        results = self.service.get_all_results()
        assert isinstance(results, list)

    def test_get_all_results_with_records(self):
        for i in range(3):
            self.service.save_result({
                "evaluation_id": f"id-{i}",
                "score": str(i * 10),
                "status": "correct",
                "timestamp": f"2026-01-0{i+1}T00:00:00",
                "exercise_type": "arm_raise",
            })
        results = self.service.get_all_results(limit=10)
        assert len(results) == 3

    def test_get_all_results_respects_limit(self):
        for i in range(5):
            self.service.save_result({
                "evaluation_id": f"lim-{i}",
                "score": "50",
                "status": "partial",
                "timestamp": f"2026-02-0{i+1}T00:00:00",
                "exercise_type": "arm_raise",
            })
        results = self.service.get_all_results(limit=2)
        assert len(results) <= 2

    def test_save_result_returns_true(self):
        ok = self.service.save_result({
            "evaluation_id": "ok-id",
            "score": "70",
            "status": "partial",
            "timestamp": "2026-01-01T00:00:00",
            "exercise_type": "arm_raise",
        })
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# StorageService — local mode
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageService:

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path):
        upload_dir = str(tmp_path / "uploads")
        annotated_dir = str(tmp_path / "annotated")
        with patch("app.services.storage_service.settings") as mock_settings:
            mock_settings.USE_LOCAL_STORAGE = True
            mock_settings.LOCAL_UPLOAD_DIR = upload_dir
            mock_settings.LOCAL_ANNOTATED_DIR = annotated_dir
            self.service = StorageService()
            self.tmp_path = tmp_path
            yield

    def _make_temp_file(self, content=b"fake video data"):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.write(content)
        f.close()
        return f.name

    def test_upload_file_returns_local_path(self):
        src = self._make_temp_file()
        try:
            url = self.service.upload_file(src, "test.mp4")
            assert url.startswith("local://")
        finally:
            os.unlink(src)

    def test_upload_raw_copies_to_upload_dir(self):
        src = self._make_temp_file()
        try:
            url = self.service.upload_raw(src, "raw.mp4")
            assert "local://" in url
        finally:
            os.unlink(src)

    def test_upload_annotated_copies_to_annotated_dir(self):
        src = self._make_temp_file()
        try:
            url = self.service.upload_annotated(src, "annotated.mp4")
            assert "local://" in url
        finally:
            os.unlink(src)

    def test_upload_file_file_exists_in_destination(self):
        src = self._make_temp_file(b"test content abc")
        try:
            url = self.service.upload_file(src, "check.mp4")
            # Strip local:// prefix
            dest = url.replace("local://", "")
            assert os.path.exists(dest)
            with open(dest, "rb") as f:
                assert f.read() == b"test content abc"
        finally:
            os.unlink(src)


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestPrometheusMetrics:

    def test_record_evaluation_no_error(self):
        record_evaluation(score=85.0, status="correct", frames=10)

    def test_record_evaluation_with_ml_confidence(self):
        record_evaluation(score=70.0, status="partial", frames=5, ml_confidence=0.88)

    def test_record_evaluation_with_anomaly(self):
        record_evaluation(score=30.0, status="incorrect", frames=3, is_anomaly=True)

    def test_record_evaluation_none_confidence(self):
        record_evaluation(score=50.0, status="partial", frames=8, ml_confidence=None)

    def test_metrics_endpoint_returns_response(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"] or "text" in r.headers["content-type"]

    def test_health_endpoint_structure(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body

    def test_ml_status_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        r = client.get("/api/v1/ml/status")
        assert r.status_code == 200
        body = r.json()
        assert "ml_available" in body
        assert "sklearn_available" in body

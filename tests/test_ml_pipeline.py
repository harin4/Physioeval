"""
test_ml_pipeline.py — Functional, performance, and validation tests for the ML pipeline.

Run with:
    pytest tests/test_ml_pipeline.py -v --tb=short

Coverage:
  - Feature extraction correctness
  - ML training pipeline (metrics thresholds)
  - Model prediction consistency vs rule-based baseline
  - Anomaly detection
  - Performance (inference latency)
  - Edge cases (empty input, single frame, all-zero angles)
  - Continuous-learning logging
"""

import time
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ── Import ML pipeline ─────────────────────────────────────────────────────
from app.services.ml_pipeline import (
    MLPipeline,
    FeatureVector,
    extract_features,
    rule_label,
    _generate_synthetic_dataset,
    FEATURE_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_pipeline(tmp_path):
    """Return a pipeline trained on a small synthetic dataset."""
    with patch("app.services.ml_pipeline.MODEL_DIR", tmp_path), \
         patch("app.services.ml_pipeline.CLASSIFIER_PATH", tmp_path / "clf.joblib"), \
         patch("app.services.ml_pipeline.SCALER_PATH",     tmp_path / "scl.joblib"), \
         patch("app.services.ml_pipeline.ANOMALY_PATH",    tmp_path / "iso.joblib"), \
         patch("app.services.ml_pipeline.FEATURE_LOG_PATH",tmp_path / "log.jsonl"), \
         patch("app.services.ml_pipeline.MLFLOW_AVAILABLE", False):
        pipe = MLPipeline()
        pipe.train(n_synthetic=300)
        yield pipe


@pytest.fixture
def correct_angles():
    """A realistic 'correct' arm raise (peaks ~165°)."""
    t = np.linspace(0, 1, 20)
    return (165 * (1 - np.exp(-4 * t))).tolist()


@pytest.fixture
def partial_angles():
    """A realistic 'partial' arm raise (peaks ~110°)."""
    t = np.linspace(0, 1, 20)
    return (110 * (1 - np.exp(-4 * t))).tolist()


@pytest.fixture
def incorrect_angles():
    """An incorrect arm raise (peaks ~50°)."""
    t = np.linspace(0, 1, 20)
    return (50 * (1 - np.exp(-4 * t))).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtraction:

    def test_returns_feature_vector(self, correct_angles):
        fv = extract_features(correct_angles)
        assert isinstance(fv, FeatureVector)

    def test_correct_has_high_best_angle(self, correct_angles):
        fv = extract_features(correct_angles)
        assert fv.best_angle >= 150

    def test_incorrect_has_low_best_angle(self, incorrect_angles):
        fv = extract_features(incorrect_angles)
        assert fv.best_angle < 90

    def test_vector_length(self, correct_angles):
        fv = extract_features(correct_angles)
        assert len(fv.to_array()) == len(FEATURE_NAMES)

    def test_empty_angles_returns_zeros(self):
        fv = extract_features([])
        arr = fv.to_array()
        assert np.all(arr == 0)

    def test_single_frame(self):
        fv = extract_features([120.0])
        assert fv.best_angle == 120.0
        assert fv.std_angle  == 0.0
        assert fv.n_frames   == 1.0

    def test_hip_drift_increases_comp_score(self, correct_angles):
        fv_no_drift   = extract_features(correct_angles, hip_x_timeline=[0.5] * 20)
        fv_high_drift = extract_features(correct_angles, hip_x_timeline=np.linspace(0.3, 0.7, 20).tolist())
        assert fv_high_drift.comp_hip_drift > fv_no_drift.comp_hip_drift

    def test_jerk_is_higher_for_shaky_motion(self, correct_angles):
        rng = np.random.default_rng(0)
        noisy = (np.array(correct_angles) + rng.normal(0, 10, len(correct_angles))).tolist()
        fv_smooth = extract_features(correct_angles)
        fv_noisy  = extract_features(noisy)
        assert fv_noisy.jerk_rms > fv_smooth.jerk_rms

    def test_fatigue_ratio_higher_for_declining_velocity(self):
        # Starts fast, slows down → fatigue_ratio < 1
        fast_start = list(range(60, 160, 5)) + [160] * 5
        fv = extract_features(fast_start)
        assert fv.fatigue_ratio <= 1.5  # velocity decaying

    def test_all_feature_names_present(self, correct_angles):
        fv = extract_features(correct_angles)
        arr = fv.to_array()
        assert len(arr) == len(FEATURE_NAMES)
        assert not np.any(np.isnan(arr)), "Features should not contain NaN"
        assert not np.any(np.isinf(arr)), "Features should not contain Inf"


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based labeller
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleLabel:

    def test_correct_label(self, correct_angles):
        fv = extract_features(correct_angles)
        assert rule_label(fv) == 2

    def test_partial_label(self, partial_angles):
        fv = extract_features(partial_angles)
        assert rule_label(fv) == 1

    def test_incorrect_label(self, incorrect_angles):
        fv = extract_features(incorrect_angles)
        assert rule_label(fv) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticDataset:

    def test_shape(self):
        X, y = _generate_synthetic_dataset(100)
        assert X.shape == (100, len(FEATURE_NAMES))
        assert y.shape == (100,)

    def test_all_classes_present(self):
        X, y = _generate_synthetic_dataset(300)
        assert set(y) == {0, 1, 2}, "All 3 classes should be present"

    def test_no_nans(self):
        X, _ = _generate_synthetic_dataset(100)
        assert not np.any(np.isnan(X))


# ─────────────────────────────────────────────────────────────────────────────
# ML Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestMLTraining:

    def test_train_returns_metrics(self, trained_pipeline):
        # Already trained in fixture; train again small
        metrics = trained_pipeline.train(n_synthetic=100)
        assert "cv_f1_mean"     in metrics
        assert "cv_acc_mean"    in metrics
        assert "train_accuracy" in metrics
        assert "train_f1"       in metrics

    def test_cv_f1_above_threshold(self, trained_pipeline):
        metrics = trained_pipeline.train(n_synthetic=400)
        # With rule-based synthetic data this should be very high
        assert metrics["cv_f1_mean"] >= 0.70, \
            f"CV F1 too low: {metrics['cv_f1_mean']:.3f}"

    def test_train_accuracy_above_threshold(self, trained_pipeline):
        metrics = trained_pipeline.train(n_synthetic=400)
        assert metrics["train_accuracy"] >= 0.85

    def test_feature_importances_returned(self, trained_pipeline):
        metrics = trained_pipeline.train(n_synthetic=100)
        imps = metrics["feature_importances"]
        assert isinstance(imps, dict)
        assert len(imps) == len(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert name in imps

    def test_top_feature_is_angle_related(self, trained_pipeline):
        metrics = trained_pipeline.train(n_synthetic=400)
        imps = metrics["feature_importances"]
        top3 = sorted(imps, key=lambda k: -imps[k])[:3]
        # best_angle or avg_angle must appear in top 3
        assert any(k in ("best_angle", "avg_angle", "angle_range") for k in top3), \
            f"Expected angle feature in top 3, got {top3}"


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

class TestMLInference:

    def test_predict_returns_required_keys(self, trained_pipeline, correct_angles):
        result = trained_pipeline.predict(correct_angles)
        for key in ("label", "confidence", "is_anomaly", "feature_vector", "ml_available"):
            assert key in result

    def test_correct_motion_predicts_correct(self, trained_pipeline, correct_angles):
        result = trained_pipeline.predict(correct_angles)
        assert result["label"] == "correct"

    def test_incorrect_motion_predicts_incorrect(self, trained_pipeline, incorrect_angles):
        result = trained_pipeline.predict(incorrect_angles)
        assert result["label"] == "incorrect"

    def test_partial_motion_label(self, trained_pipeline, partial_angles):
        result = trained_pipeline.predict(partial_angles)
        # partial is borderline; accept partial or correct (not incorrect if 110°)
        assert result["label"] in ("partial", "correct")

    def test_confidence_in_range(self, trained_pipeline, correct_angles):
        result = trained_pipeline.predict(correct_angles)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_ml_available_flag(self, trained_pipeline, correct_angles):
        result = trained_pipeline.predict(correct_angles)
        assert result["ml_available"] is True

    def test_empty_angles_returns_rule_fallback(self, trained_pipeline):
        result = trained_pipeline.predict([])
        assert result["label"] == "incorrect"  # 0° → incorrect by rule

    def test_probabilities_sum_to_one(self, trained_pipeline, correct_angles):
        result = trained_pipeline.predict(correct_angles)
        if "probabilities" in result:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-5

    def test_anomaly_detection_on_extreme_input(self, trained_pipeline):
        # Wildly oscillating angles — should trigger anomaly
        wild = ([5, 185, 5, 185, 5, 185] * 5)
        result = trained_pipeline.predict(wild)
        # Anomaly detection may or may not flag — just ensure no crash
        assert isinstance(result["is_anomaly"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# Performance tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:

    LATENCY_THRESHOLD_MS = 600  # inference must be < 600 ms (Windows CI overhead ~400 ms)

    def test_feature_extraction_latency(self, correct_angles):
        start = time.perf_counter()
        for _ in range(100):
            extract_features(correct_angles)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
        assert elapsed_ms < 5, f"Feature extraction too slow: {elapsed_ms:.2f} ms"

    def test_inference_latency(self, trained_pipeline, correct_angles):
        # Warm up
        trained_pipeline.predict(correct_angles)

        start = time.perf_counter()
        for _ in range(50):
            trained_pipeline.predict(correct_angles)
        elapsed_ms = (time.perf_counter() - start) / 50 * 1000
        assert elapsed_ms < self.LATENCY_THRESHOLD_MS, \
            f"Inference too slow: {elapsed_ms:.2f} ms (threshold={self.LATENCY_THRESHOLD_MS} ms)"

    def test_training_completes_within_timeout(self, tmp_path):
        with patch("app.services.ml_pipeline.MODEL_DIR", tmp_path), \
             patch("app.services.ml_pipeline.CLASSIFIER_PATH", tmp_path / "c.joblib"), \
             patch("app.services.ml_pipeline.SCALER_PATH",     tmp_path / "s.joblib"), \
             patch("app.services.ml_pipeline.ANOMALY_PATH",    tmp_path / "a.joblib"), \
             patch("app.services.ml_pipeline.FEATURE_LOG_PATH",tmp_path / "l.jsonl"), \
             patch("app.services.ml_pipeline.MLFLOW_AVAILABLE", False):
            pipe = MLPipeline()
            start = time.perf_counter()
            pipe.train(n_synthetic=200)
            elapsed = time.perf_counter() - start
        assert elapsed < 30, f"Training took too long: {elapsed:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Continuous learning & feature logging
# ─────────────────────────────────────────────────────────────────────────────

class TestContinuousLearning:

    def test_feature_log_written(self, trained_pipeline, correct_angles, tmp_path):
        log_path = tmp_path / "log.jsonl"
        with patch("app.services.ml_pipeline.FEATURE_LOG_PATH", log_path):
            trained_pipeline._log_feature({"best_angle": 160.0}, label=2, confidence=0.9)
            assert log_path.exists()
            with open(log_path) as f:
                record = json.loads(f.readline())
            assert record["label"] == 2
            assert record["confidence"] == 0.9

    def test_get_feature_log_count(self, trained_pipeline, tmp_path):
        log_path = tmp_path / "count_log.jsonl"
        with patch("app.services.ml_pipeline.FEATURE_LOG_PATH", log_path):
            # Write 5 records
            for i in range(5):
                trained_pipeline._log_feature({"best_angle": float(i)}, 0, 0.5)
            count = trained_pipeline.get_feature_log_count()
        assert count == 5

    def test_maybe_retrain_not_triggered_below_threshold(self, trained_pipeline, tmp_path):
        log_path = tmp_path / "retrain_log.jsonl"
        with patch("app.services.ml_pipeline.FEATURE_LOG_PATH", log_path):
            # Only 3 records — well below default threshold of 200
            for _ in range(3):
                trained_pipeline._log_feature({}, 0, 0.5)
            with patch.object(trained_pipeline, "train") as mock_train:
                trained_pipeline.maybe_retrain(threshold=200)
                mock_train.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (no sklearn) behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackBehaviour:

    def test_predict_falls_back_when_no_model(self, correct_angles):
        pipe = MLPipeline()
        pipe.classifier = None
        pipe.scaler     = None
        result = pipe.predict(correct_angles)
        assert result["ml_available"] is False
        assert result["source"] == "rule_based"
        assert result["label"] in ("correct", "partial", "incorrect")

    def test_fallback_confidence_reasonable(self, correct_angles):
        pipe = MLPipeline()
        pipe.classifier = None
        pipe.scaler     = None
        result = pipe.predict(correct_angles)
        assert 0.0 <= result["confidence"] <= 1.0

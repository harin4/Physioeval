"""
ml_pipeline.py — MLflow-tracked ML workflow for PhysioEval.

Responsibilities:
  1. Feature extraction from pose frame sequences
  2. Rule-augmented ML classifier training (scikit-learn IsolationForest + RandomForest)
  3. MLflow experiment tracking (params, metrics, artifacts, model registry)
  4. Inference with confidence scores
  5. Automated retraining trigger when enough labelled data accumulates

Run standalone to train:
    python -m app.services.ml_pipeline

Or import and call:
    from app.services.ml_pipeline import MLPipeline
    pipeline = MLPipeline()
    result = pipeline.predict(features)
"""
from __future__ import annotations

import json
import os
import time
import uuid
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── MLflow (optional — graceful fallback if not installed) ──────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── scikit-learn ────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report,
        confusion_matrix,
    )
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.core.config import settings
from app.core.logger import logger

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True)
CLASSIFIER_PATH   = MODEL_DIR / "rf_classifier.joblib"
SCALER_PATH       = MODEL_DIR / "scaler.joblib"
ANOMALY_PATH      = MODEL_DIR / "isolation_forest.joblib"
FEATURE_LOG_PATH  = MODEL_DIR / "feature_log.jsonl"
MLFLOW_EXPERIMENT = "physio-eval-arm-raise"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# ── Feature vector definition (14-dim) ──────────────────────────────────────
FEATURE_NAMES = [
    "best_angle",          # peak shoulder angle (°)
    "avg_angle",           # mean shoulder angle
    "std_angle",           # std dev (variability / tremor proxy)
    "angle_range",         # max − min  (ROM span)
    "velocity_mean",       # mean frame-to-frame velocity (°/frame)
    "velocity_std",        # velocity jitter
    "jerk_rms",            # root-mean-square jerk (tremor)
    "fatigue_ratio",       # velocity in last third / first third
    "smoothness",          # 1 / (1 + jerk_rms)  rescaled
    "comp_hip_drift",      # hip x std (compensation proxy)
    "rising_time_frames",  # frames to reach 80 % of peak
    "plateau_angle",       # angle at which motion stalls (ROM restriction)
    "n_frames",            # number of valid frames
    "visibility_mean",     # avg landmark visibility
]


@dataclass
class FeatureVector:
    best_angle: float
    avg_angle: float
    std_angle: float
    angle_range: float
    velocity_mean: float
    velocity_std: float
    jerk_rms: float
    fatigue_ratio: float
    smoothness: float
    comp_hip_drift: float
    rising_time_frames: float
    plateau_angle: float
    n_frames: float
    visibility_mean: float

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)


# ── Feature extractor ────────────────────────────────────────────────────────

def extract_features(
    angle_timeline: List[float],
    hip_x_timeline: Optional[List[float]] = None,
    visibility_timeline: Optional[List[float]] = None,
) -> FeatureVector:
    """Convert raw timeseries into a fixed-length feature vector."""
    arr = np.array(angle_timeline, dtype=float)
    n   = len(arr)

    if n == 0:
        return FeatureVector(*([0.0] * len(FEATURE_NAMES)))

    best_angle   = float(arr.max())
    avg_angle    = float(arr.mean())
    std_angle    = float(arr.std()) if n > 1 else 0.0
    angle_range  = float(arr.max() - arr.min())

    # velocity & jerk
    vel  = np.abs(np.diff(arr)) if n > 1 else np.array([0.0])
    vel_mean = float(vel.mean())
    vel_std  = float(vel.std()) if len(vel) > 1 else 0.0
    jerk = np.abs(np.diff(vel)) if len(vel) > 1 else np.array([0.0])
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))

    # fatigue: velocity ratio (last third vs first third)
    third = max(1, n // 3)
    first_vel = float(np.abs(np.diff(arr[:third])).mean()) if third > 1 else 1e-9
    last_vel  = float(np.abs(np.diff(arr[-third:])).mean()) if third > 1 else first_vel
    fatigue_ratio = last_vel / (first_vel + 1e-9)

    # smoothness proxy
    smoothness = float(1.0 / (1.0 + jerk_rms))

    # compensation — hip drift
    if hip_x_timeline and len(hip_x_timeline) > 1:
        comp_hip_drift = float(np.std(hip_x_timeline))
    else:
        comp_hip_drift = 0.0

    # rising time: frames to hit 80 % of peak
    target = 0.8 * best_angle
    rising_time = float(n)
    for i, a in enumerate(arr):
        if a >= target:
            rising_time = float(i + 1)
            break

    # plateau angle: first angle that doesn't improve by > 1° over remaining
    plateau_angle = best_angle
    for i in range(n - 5):
        if arr[i:].max() - arr[i] < 1.0:
            plateau_angle = float(arr[i])
            break

    visibility_mean = float(np.mean(visibility_timeline)) if visibility_timeline else 0.9

    return FeatureVector(
        best_angle=best_angle,
        avg_angle=avg_angle,
        std_angle=std_angle,
        angle_range=angle_range,
        velocity_mean=vel_mean,
        velocity_std=vel_std,
        jerk_rms=jerk_rms,
        fatigue_ratio=fatigue_ratio,
        smoothness=smoothness,
        comp_hip_drift=comp_hip_drift,
        rising_time_frames=rising_time,
        plateau_angle=plateau_angle,
        n_frames=float(n),
        visibility_mean=visibility_mean,
    )


# ── Rule-based labeller (used to generate synthetic training labels) ─────────

def rule_label(fv: FeatureVector) -> int:
    """
    Returns 0=incorrect, 1=partial, 2=correct
    Mirrors the scoring engine logic so the ML model learns the same decision
    boundary but generalises to noisy, real-world input.
    """
    if fv.best_angle >= settings.ARM_RAISE_CORRECT_ANGLE:
        return 2
    if fv.best_angle >= settings.ARM_RAISE_PARTIAL_ANGLE:
        return 1
    return 0


# ── Synthetic dataset generator ───────────────────────────────────────────────

def _generate_synthetic_dataset(n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a realistic synthetic training set by sampling plausible
    biomechanical timeseries and extracting features from them.
    """
    rng = np.random.default_rng(42)
    X_list, y_list = [], []

    for _ in range(n_samples):
        # Random peak angle 20° – 185°
        peak = rng.uniform(20, 185)
        n_fr = rng.integers(5, 40)
        # Smooth ramp-up with gaussian noise
        t     = np.linspace(0, 1, n_fr)
        ramp  = peak * (1 - np.exp(-3 * t))
        noise = rng.normal(0, rng.uniform(0.5, 8.0), n_fr)
        angles = np.clip(ramp + noise, 0, 185).tolist()

        hip_drift = rng.uniform(0, 0.15, n_fr).tolist()
        vis       = rng.uniform(0.7, 1.0, n_fr).tolist()

        fv = extract_features(angles, hip_drift, vis)
        X_list.append(fv.to_array())
        y_list.append(rule_label(fv))

    return np.array(X_list), np.array(y_list)


# ── Main MLPipeline class ────────────────────────────────────────────────────

class MLPipeline:
    """
    Wraps training, evaluation, MLflow logging, and inference.
    Falls back to rule-based scoring when models are unavailable.
    """

    def __init__(self):
        self.classifier: Optional[Any] = None
        self.scaler: Optional[Any]     = None
        self.anomaly_detector: Optional[Any] = None
        self._load_models()

    # ── Model persistence ─────────────────────────────────────────────────────

    def _load_models(self):
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not installed — ML pipeline disabled.")
            return
        if CLASSIFIER_PATH.exists() and SCALER_PATH.exists():
            try:
                self.classifier = joblib.load(CLASSIFIER_PATH)
                self.scaler     = joblib.load(SCALER_PATH)
                if ANOMALY_PATH.exists():
                    self.anomaly_detector = joblib.load(ANOMALY_PATH)
                logger.info("ML models loaded from disk.")
            except Exception as e:
                logger.warning(f"Could not load ML models: {e}")

    def _save_models(self):
        joblib.dump(self.classifier,      CLASSIFIER_PATH)
        joblib.dump(self.scaler,          SCALER_PATH)
        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, ANOMALY_PATH)
        logger.info(f"Models saved to {MODEL_DIR}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_synthetic: int = 600,
    ) -> Dict[str, Any]:
        """
        Train classifier + anomaly detector.
        If X/y not supplied, generates a synthetic dataset.
        Logs everything to MLflow.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        if X is None or y is None:
            logger.info(f"Generating synthetic training set ({n_synthetic} samples)…")
            X, y = _generate_synthetic_dataset(n_synthetic)

        # ── Setup MLflow ─────────────────────────────────────────────────────
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            run_ctx = mlflow.start_run(run_name=f"train_{int(time.time())}")
        else:
            run_ctx = _NullContext()

        metrics: Dict[str, Any] = {}

        with run_ctx:
            # ── Params ───────────────────────────────────────────────────────
            params = {
                "n_estimators":   200,
                "max_depth":      8,
                "min_samples_leaf": 3,
                "n_features":     len(FEATURE_NAMES),
                "n_samples_train": len(X),
                "contamination":  0.05,
            }
            if MLFLOW_AVAILABLE:
                mlflow.log_params(params)

            # ── Preprocessing ─────────────────────────────────────────────────
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ── Cross-validation ──────────────────────────────────────────────
            clf_cv = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf_cv, X_scaled, y, cv=cv, scoring="f1_weighted")
            cv_acc    = cross_val_score(clf_cv, X_scaled, y, cv=cv, scoring="accuracy")

            metrics["cv_f1_mean"]     = float(cv_scores.mean())
            metrics["cv_f1_std"]      = float(cv_scores.std())
            metrics["cv_acc_mean"]    = float(cv_acc.mean())
            logger.info(
                f"CV F1={metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f} "
                f"Acc={metrics['cv_acc_mean']:.3f}"
            )

            # ── Final fit ─────────────────────────────────────────────────────
            clf_final = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            clf_final.fit(X_scaled, y)
            y_pred = clf_final.predict(X_scaled)

            metrics["train_accuracy"]  = float(accuracy_score(y, y_pred))
            metrics["train_f1"]        = float(f1_score(y, y_pred, average="weighted"))
            metrics["train_precision"] = float(precision_score(y, y_pred, average="weighted"))
            metrics["train_recall"]    = float(recall_score(y, y_pred, average="weighted"))

            # feature importance
            importances = dict(zip(FEATURE_NAMES, clf_final.feature_importances_.tolist()))
            metrics["feature_importances"] = importances
            logger.info(f"Top features: {sorted(importances.items(), key=lambda x:-x[1])[:5]}")

            # ── Anomaly detector ──────────────────────────────────────────────
            iso_forest = IsolationForest(
                contamination=params["contamination"],
                random_state=42,
                n_jobs=-1,
            )
            iso_forest.fit(X_scaled)
            anomaly_preds = iso_forest.predict(X_scaled)
            metrics["anomaly_fraction"] = float((anomaly_preds == -1).mean())

            # ── MLflow log ────────────────────────────────────────────────────
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                mlflow.log_dict(importances, "feature_importances.json")
                mlflow.sklearn.log_model(clf_final, "random_forest_classifier")
                mlflow.sklearn.log_model(scaler,    "feature_scaler")
                mlflow.sklearn.log_model(iso_forest,"isolation_forest")
                report = classification_report(y, y_pred, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")
                cm = confusion_matrix(y, y_pred).tolist()
                mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")

            # ── Store on self & disk ──────────────────────────────────────────
            self.classifier      = clf_final
            self.scaler          = scaler
            self.anomaly_detector = iso_forest
            self._save_models()

        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        angle_timeline: List[float],
        hip_x_timeline: Optional[List[float]] = None,
        visibility_timeline: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a prediction dict:
          label        : 'correct' | 'partial' | 'incorrect'
          confidence   : 0.0 – 1.0
          is_anomaly   : bool (unusual movement pattern)
          feature_vector : dict of extracted features
          ml_available : bool
        """
        fv = extract_features(angle_timeline, hip_x_timeline, visibility_timeline)
        fv_dict = asdict(fv)

        # Rule-based fallback
        rule = rule_label(fv)
        label_map = {0: "incorrect", 1: "partial", 2: "correct"}
        fallback = {
            "label":          label_map[rule],
            "confidence":     0.7,
            "is_anomaly":     False,
            "feature_vector": fv_dict,
            "ml_available":   False,
            "source":         "rule_based",
        }

        if not SKLEARN_AVAILABLE or self.classifier is None or self.scaler is None:
            return fallback

        try:
            X = fv.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            pred_class = int(self.classifier.predict(X_scaled)[0])
            proba      = self.classifier.predict_proba(X_scaled)[0]
            confidence = float(proba[pred_class])

            is_anomaly = False
            if self.anomaly_detector is not None:
                is_anomaly = bool(self.anomaly_detector.predict(X_scaled)[0] == -1)

            # Log feature for continuous learning
            self._log_feature(fv_dict, pred_class, confidence)

            return {
                "label":          label_map.get(pred_class, "incorrect"),
                "confidence":     confidence,
                "probabilities":  {label_map[i]: float(p) for i, p in enumerate(proba)},
                "is_anomaly":     is_anomaly,
                "feature_vector": fv_dict,
                "ml_available":   True,
                "source":         "random_forest",
            }
        except Exception as e:
            logger.warning(f"ML inference failed, using rule fallback: {e}")
            return fallback

    # ── Continuous learning: log features ─────────────────────────────────────

    def _log_feature(self, fv_dict: dict, label: int, confidence: float):
        """Append feature + predicted label to JSONL for future retraining."""
        record = {
            "id":         str(uuid.uuid4()),
            "timestamp":  time.time(),
            "features":   fv_dict,
            "label":      label,
            "confidence": confidence,
        }
        try:
            with open(FEATURE_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Feature log write failed: {e}")

    def get_feature_log_count(self) -> int:
        if not FEATURE_LOG_PATH.exists():
            return 0
        with open(FEATURE_LOG_PATH) as f:
            return sum(1 for _ in f)

    def maybe_retrain(self, threshold: int = 200):
        """Trigger retraining if we've accumulated enough real feature logs."""
        count = self.get_feature_log_count()
        if count >= threshold:
            logger.info(f"Auto-retraining triggered ({count} logged samples)…")
            self.train(n_synthetic=max(400, count * 2))


# ── Null context for when MLflow is unavailable ───────────────────────────────

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure stdout handles Unicode (MLflow 3.x logs emoji in run URLs)
    import sys, io
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("PhysioEval - ML Pipeline Training")
    print("=" * 60)

    pipeline = MLPipeline()
    metrics  = pipeline.train(n_synthetic=800)

    print("\n📊 Training Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")

    print("\n🔍 Test Inference:")
    sample_angles = [10, 30, 60, 90, 120, 145, 155, 158, 157, 159]
    result = pipeline.predict(sample_angles)
    print(f"  Label:      {result['label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Anomaly:    {result['is_anomaly']}")
    print(f"  Source:     {result['source']}")
    print("\n✅ Done.")

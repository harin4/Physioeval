"""
metrics.py — Prometheus-compatible metrics for PhysioEval.

Exposes:
  GET /metrics  → Prometheus text format (scraped by Prometheus / Grafana)

Tracked metrics:
  physio_requests_total          — counter by endpoint, method, status
  physio_request_duration_seconds — histogram of latency
  physio_score_histogram         — histogram of evaluation scores
  physio_frames_analyzed_total   — counter of frames processed
  physio_evaluation_status_total — counter by status (correct/partial/incorrect)
  physio_ml_confidence_histogram — histogram of ML model confidence
  physio_anomaly_detections_total— counter of anomaly flags
  physio_active_requests         — gauge of in-flight requests

Usage:
  Add to FastAPI in main.py:
    from app.monitoring.metrics import setup_metrics
    setup_metrics(app)
"""
from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute

from app.core.logger import logger

# ── Try to import prometheus_client; fall back to stub ───────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — /metrics will return stub data")


# ── Metric definitions ────────────────────────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "physio_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    REQUEST_LATENCY = Histogram(
        "physio_request_duration_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    SCORE_HISTOGRAM = Histogram(
        "physio_score_histogram",
        "Distribution of evaluation scores (0–100)",
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    FRAMES_COUNTER = Counter(
        "physio_frames_analyzed_total",
        "Total video frames analyzed",
    )
    EVAL_STATUS_COUNTER = Counter(
        "physio_evaluation_status_total",
        "Evaluations by outcome",
        ["status"],
    )
    ML_CONFIDENCE_HISTOGRAM = Histogram(
        "physio_ml_confidence_histogram",
        "Distribution of ML model prediction confidence",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    )
    ANOMALY_COUNTER = Counter(
        "physio_anomaly_detections_total",
        "Total anomalous movement patterns detected",
    )
    ACTIVE_REQUESTS = Gauge(
        "physio_active_requests",
        "Number of requests currently being processed",
    )
else:
    # Stub objects — silently ignore observations
    class _Stub:
        def labels(self, **kw): return self
        def inc(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def time(self): return _NullCtx()

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    REQUEST_COUNT = _Stub()
    REQUEST_LATENCY = _Stub()
    SCORE_HISTOGRAM = _Stub()
    FRAMES_COUNTER  = _Stub()
    EVAL_STATUS_COUNTER = _Stub()
    ML_CONFIDENCE_HISTOGRAM = _Stub()
    ANOMALY_COUNTER = _Stub()
    ACTIVE_REQUESTS = _Stub()


# ── Public helper functions (called from routes.py) ───────────────────────────

def record_evaluation(
    score: float,
    status: str,
    frames: int,
    ml_confidence: float | None = None,
    is_anomaly: bool = False,
):
    """Call this after every successful evaluation."""
    SCORE_HISTOGRAM.observe(score)
    FRAMES_COUNTER.inc(frames)
    EVAL_STATUS_COUNTER.labels(status=status).inc()
    if ml_confidence is not None:
        ML_CONFIDENCE_HISTOGRAM.observe(ml_confidence)
    if is_anomaly:
        ANOMALY_COUNTER.inc()


# ── Middleware ────────────────────────────────────────────────────────────────

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """ASGI middleware that times every request and records Prometheus metrics."""
    endpoint = request.url.path
    method   = request.method

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as exc:
        status_code = "500"
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        raise
    finally:
        ACTIVE_REQUESTS.dec()

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    return response


# ── /metrics endpoint ────────────────────────────────────────────────────────

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape endpoint."""
    if PROMETHEUS_AVAILABLE:
        data    = generate_latest()
        ctype   = CONTENT_TYPE_LATEST
    else:
        # Return a minimal stub so Prometheus doesn't error
        data  = b"# PhysioEval metrics stub (prometheus_client not installed)\n"
        ctype = "text/plain; version=0.0.4; charset=utf-8"
    return Response(content=data, media_type=ctype)


# ── Setup helper ─────────────────────────────────────────────────────────────

def setup_metrics(app: FastAPI):
    """Register middleware and /metrics route on the FastAPI app."""
    app.middleware("http")(metrics_middleware)
    app.add_route("/metrics", metrics_endpoint, include_in_schema=False)
    logger.info("Prometheus metrics enabled at /metrics")

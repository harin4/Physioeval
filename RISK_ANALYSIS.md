# PhysioEval — Risk Analysis & Mitigation

## 1. Risk Register

| # | Risk | Category | Likelihood | Impact | Score | Mitigation |
|---|------|----------|-----------|--------|-------|-----------|
| R1 | Pose detection fails on low-quality video | Technical | High | High | 9 | Frame-quality pre-filter; multi-sample-rate fallback; user guidance UI |
| R2 | ML model drifts as real patient data diverges from synthetic training | ML/AI | Medium | High | 6 | MLflow model versioning; auto-retrain trigger at 200 logged samples; CV F1 threshold alert |
| R3 | S3 or DynamoDB outage causes evaluation failure | Cloud/Infra | Low | High | 4 | `USE_LOCAL_STORAGE` fallback flag; retry logic in `StorageService`; AWS Multi-AZ deployment |
| R4 | Patient PHI exposed via uploaded video | Security/Privacy | Medium | Critical | 10 | Face blurring (HIPAA flag); presigned S3 URLs (15 min TTL); no PII in DynamoDB; TLS everywhere |
| R5 | False "correct" score for dangerous posture | Clinical | Medium | Critical | 10 | Clinical disclaimer in UI; disorder_probability flag; therapist review workflow |
| R6 | High-latency inference degrades UX (> 5 s) | Performance | Medium | Medium | 4 | Prometheus p95 alert; EC2 autoscaling; async video processing |
| R7 | Docker image vulnerabilities | Security | High | Medium | 6 | Trivy scan in CI; pin base image digest; Dependabot on requirements.txt |
| R8 | Adversarial input (oversized file, malformed video) | Security | Medium | Low | 2 | File-type/size validation in `validators.py`; exception handling in routes |
| R9 | Single-exercise scope limits clinical value | Business | High | Low | 3 | Backlog: multi-exercise support; clearly communicated MVP scope |
| R10 | AWS cost overrun | Financial | Low | Medium | 2 | CloudWatch billing alarms; S3 lifecycle policy; DynamoDB on-demand billing |

*Score = Likelihood × Impact (1=Low, 3=Med, 5=High)*

---

## 2. Top Risk Deep-Dives

### R4 — PHI Exposure (Score 10)

**Threat model:**
- Attacker intercepts uploaded video containing patient face
- Internal employee downloads results with identifiable metadata

**Controls:**
```
Upload → Face blur (OpenCV) → S3 (AES-256 SSE-C) → presigned URL (TTL 15 min)
                                                   → DynamoDB (no face data stored)
API    → HTTPS only → API key header (X-API-Key)
Logging→ CloudWatch (no file content, only evaluation_id)
```

**Residual risk:** Low — patient identity not reconstructible from stored data.

---

### R5 — Clinical False Positive (Score 10)

**Threat model:**
- Patient performs dangerous compensatory movement scored as "correct"
- System used as sole clinical decision-maker

**Controls:**
- Compensation detection penalty (−10 % score)
- `disorder_probability` flag triggers mandatory therapist review text
- UI disclaimer: *"This tool is an aid, not a medical diagnosis"*
- Recommendations always include "Consult your physiotherapist"

**Residual risk:** Medium — inherent to any automated clinical system. Mitigated by workflow design (therapist remains responsible).

---

### R2 — ML Model Drift (Score 6)

**Threat model:**
- Synthetic training data doesn't reflect real patient diversity
- Model accuracy degrades silently over time

**Controls:**
```python
# Auto-retrain when real data accumulates
pipeline.maybe_retrain(threshold=200)

# MLflow tracks every run; compare new cv_f1 to baseline
# Prometheus alert: physio_ml_confidence_histogram p10 < 0.6
```

**Residual risk:** Low-Medium — MLflow versioning allows instant rollback to last good model.

---

## 3. Scalability Strategy

### Current (MVP)
- Single EC2 t3.medium or Render free tier
- Synchronous request processing
- Local file storage fallback

### Scale-Out Path

```
Users (100+/day)
      │
      ▼
CloudFront CDN  ──→  Load Balancer (ALB)
                             │
                    ┌────────┴────────┐
                 EC2 Auto-Scaling Group (2–10 instances)
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                   SQS Queue (async video jobs)
                    └────────┬────────┘
                             │
                    Lambda Workers (inference)
                             │
                    ┌────────┴────────┐
                   S3 (videos)   DynamoDB (results)
```

### Cost Optimisation
| Resource | MVP Cost | Scale-Out |
|----------|----------|-----------|
| EC2 t3.medium | ~$30/mo | Auto-scale, Spot instances (−70 %) |
| S3 | < $1/mo | Lifecycle: Glacier after 90 days |
| DynamoDB | < $1/mo | On-demand billing |
| Total MVP | ~$35/mo | ~$150/mo at 10k evaluations/day |

---

## 4. Security Checklist

- [x] Input validation: file type + size enforced in `validators.py`
- [x] Face blurring: OpenCV DNN blur for HIPAA compliance
- [x] HTTPS: enforced at ALB / Render layer
- [x] No secrets in code: all credentials via environment variables / `.env`
- [x] Docker: non-root user in container
- [x] CI: Trivy vulnerability scan (add to `.github/workflows/ci.yml`)
- [ ] TODO: API key authentication (currently open)
- [ ] TODO: Rate limiting (FastAPI `slowapi`)
- [ ] TODO: WAF rules on CloudFront

---

## 5. Reliability Design

```
SLA Target: 99.5 % uptime (≈ 3.6 hr/month downtime budget)

Health checks:  GET /health  (30 s interval, 3 retries)
Auto-restart:   Docker restart: unless-stopped
Circuit breaker: AWS S3/DynamoDB calls wrapped in try/except with local fallback
Logging:        CloudWatch + structured JSON logs (app/core/logger.py)
Alerting:       Prometheus → Grafana alerts → email/PagerDuty
```

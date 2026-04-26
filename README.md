<<<<<<< HEAD
# Physioeval
=======
# 🦾 PhysioEval Enhanced — AI Biomechanical Framework

> **"An AI-driven Biomechanical Framework that utilizes MediaPipe for pose estimation and AWS Lambda for serverless processing. Unlike basic trackers, this system implements Signal Frequency Analysis to detect tremors and muscle fatigue, providing clinical-grade feedback through a Gradio-based digital-twin dashboard, while ensuring data privacy via face-blurring."**

---

## 🆕 What's New in v2.0

| Feature | v1 (Original) | v2 (Enhanced) |
|---|---|---|
| Pose Detection | ✅ MediaPipe | ✅ MediaPipe |
| Angle Scoring | ✅ Rule-based | ✅ Rule-based |
| **Tremor Detection** | ❌ | ✅ Jerk + PSD analysis |
| **Fatigue Analysis** | ❌ | ✅ Velocity-drop per rep |
| **Compensation Detection** | ❌ | ✅ Trunk lean angle |
| **ROM Restriction** | ❌ | ✅ Stall detection |
| **Disorder Prediction** | ❌ | ✅ Rule-based RF classifier |
| **Face Blurring** | ❌ | ✅ Haar cascade + Gaussian blur |
| **Frontend** | HTML page | ✅ Gradio dashboard |
| **Overlay Video** | ❌ | ✅ Skeleton + angle rendered |
| **S3 Event → Lambda** | ❌ | ✅ Event-driven pipeline |
| **Signal Smoothness Plot** | ❌ | ✅ Frame-by-frame chart |

---

## 🏗️ Architecture

```
User (Gradio UI / REST API)
        ↓
FastAPI on EC2 / Render
        ↓
┌─────────────────────────────────────────┐
│         Processing Pipeline             │
│  1. Face Blur (HIPAA Privacy)           │
│  2. Pose Detection (MediaPipe)          │
│  3. Angle Extraction                    │
│  4. Signal Analysis (SciPy)            │
│     ├── Tremor (Jerk + PSD)            │
│     ├── Fatigue (Rep velocity drop)    │
│     ├── Compensation (Trunk lean)      │
│     ├── ROM Restriction (Stall detect) │
│     └── Disorder Prediction (Rules)    │
│  5. Composite Scoring                  │
└─────────────────────────────────────────┘
        ↓
AWS Storage Layer
  ├── S3 (videos) → Lambda trigger
  └── DynamoDB (results + signal data)
        ↓
Gradio Dashboard
  ├── Skeleton Overlay Video
  ├── Score Gauge (0–100)
  ├── Tremor Indicator (Red/Yellow/Green)
  ├── Fatigue Monitor
  ├── Compensation Alert
  ├── ROM Report
  ├── Disorder Flag (clinical note)
  └── Smoothness Curve (frame analytics)
```

---

## 🧠 Signal Analysis Details

### Tremor Detection
- Computes **velocity → acceleration → jerk** from angle time-series
- Performs **Welch Power Spectral Density** analysis
- Flags tremor-band power (3–12 Hz) as neurological vs mechanical
- Levels: `none` / `mild` / `moderate` / `severe`

### Fatigue Analysis
- Detects repetitions using **peak-finding** (SciPy)
- Measures **peak speed** at each rep using velocity signal
- Calculates % velocity drop from Rep 1 → Last Rep
- Levels: `none` (< 20% drop) → `severe` (> 60% drop)

### Compensation Detection
- Computes **mid-shoulder to mid-hip vector**
- Measures deviation from vertical (trunk lean angle)
- > 15° → flags lateral trunk compensation

### ROM Restriction
- Sliding-window stall detection (< 3° motion over 5 frames)
- Classifies by stall angle: < 90° = Frozen Shoulder, 90–120° = Rotator Cuff, etc.

### Disorder Prediction
- Rule-based ensemble mimicking a Random Forest
- Inputs: tremor score, fatigue score, compensation flag, ROM restriction
- Outputs: disorder name, probability (0–1), confidence, clinical note
- **Conditions detected**: Neurological Tremor, Adhesive Capsulitis, Rotator Cuff Weakness, Shoulder Impingement, Deconditioning

---
venv310\Scripts\activate

## 🚀 Quick Start

### Local Development

```bash
# Clone and enter
cd physio-enhanced

# Install dependencies
pip install -r requirements.txt

# Run (FastAPI + Gradio mounted)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open dashboard
open http://localhost:8000/gradio

# API docs
open http://localhost:8000/docs
```

### Docker

```bash
docker build -t physio-eval-v2 .
docker run -p 8000:8000 \
  -e USE_LOCAL_STORAGE=true \
  physio-eval-v2
```

### With AWS

```bash
# Set credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
export S3_BUCKET_NAME=physio-eval-uploads
export DYNAMODB_TABLE_NAME=physio-eval-results
export USE_LOCAL_STORAGE=false

# Provision AWS resources
python scripts/setup_aws.py

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Reference

### `POST /api/v1/evaluate`

Upload a video or image and get full biomechanical evaluation.

**Form fields:**
| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | MP4/MOV/AVI/JPG/PNG |
| `exercise_type` | str | `arm_raise` | Exercise to evaluate |
| `session_id` | str | optional | Patient/session ID |
| `blur_faces` | bool | `true` | HIPAA face blurring |

**Response includes:**
- `score` — 0–100 composite score
- `status` — correct / partial / incorrect / no_pose_detected
- `feedback` — full clinical text
- `signal_analysis` — tremor, fatigue, compensation, rom, disorder_prediction
- `face_blurred` — whether blurring was applied

### `GET /api/v1/history`
Returns past evaluations including tremor and fatigue levels.

### `GET /api/v1/exercises`
Lists supported exercises and signal analysis capabilities.

---

## ☁️ AWS Services

| Service | Purpose |
|---|---|
| **EC2 / Render** | Hosts FastAPI + Gradio |
| **S3** | Video/image storage, HIPAA-compliant |
| **DynamoDB** | Results + full signal analysis JSON |
| **Lambda** | Event-driven trigger on S3 upload |
| **CloudWatch** | API logs, error tracking |

---

## 🔒 Privacy & Compliance

- **Face Blurring** — Haar cascade detects faces; strong Gaussian blur applied before storage and display
- **S3 Public Access Blocked** — All buckets have public access fully disabled
- **DynamoDB PITR** — Point-in-time recovery enabled for audit compliance
- **No PII in logs** — Evaluation IDs are UUIDs; no patient names stored

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Test coverage includes:
- Angle calculation geometry
- Scoring logic (correct / partial / incorrect / no pose)
- Signal analyzer (tremor, fatigue, ROM, smoothness)
- Trunk lean extraction
- API endpoint integration (mocked AWS)

---

## ⚠️ Disclaimer

This system is a **research and educational tool**. All clinical flags and disorder predictions are **AI-generated estimates** based on biomechanical rules and are **NOT medical diagnoses**. Always consult a qualified healthcare professional for clinical decisions.
>>>>>>> 2c813c8 (feat: add full MLflow/CI-CD/testing/monitoring stack)

"""
gradio_app.py — PhysioEval Enhanced
Gradio-based digital-twin dashboard replacing the old frontend/index.html.
Launch standalone: python -m app.gradio_app
Or accessed at /gradio when mounted in FastAPI via main.py.
"""

import os, json, tempfile, uuid
from datetime import datetime
from typing import List
import numpy as np

import gradio as gr

from app.services.pose_detector import PoseDetector
from app.services.scoring_engine import ScoringEngine
from app.services.storage_service import StorageService
from app.services.results_service import ResultsService
from app.models.schemas import ExerciseType
from app.core.logger import logger

pose_detector   = PoseDetector()
scoring_engine  = ScoringEngine()
storage_service = StorageService()
results_service = ResultsService()


# ── Processing ────────────────────────────────────────────────────────────────

def process_video(video_path, session_id, blur_faces):
    if video_path is None:
        return _empty_outputs()

    evaluation_id = str(uuid.uuid4())
    try:
        overlay_path = os.path.join(tempfile.gettempdir(), f"overlay_{evaluation_id}.mp4")
        pose_frames = pose_detector.detect_from_video(
            video_path, sample_rate=3,
            annotated_output_path=overlay_path,
            blur_faces=blur_faces,
        )
        fps = 10.0
        score, status, feedback, joint_angles, signal, _ = scoring_engine.evaluate(
            ExerciseType.ARM_RAISE,
            pose_frames
        )

        if not os.path.exists(overlay_path):
            overlay_path = video_path

        record = {
            "evaluation_id": evaluation_id, "exercise_type": "arm_raise",
            "score": str(score), "status": status.value, "feedback": feedback,
            "joint_angles": joint_angles.model_dump() if joint_angles else {},
            "signal_analysis": signal.model_dump() if signal else {},
            "frames_analyzed": len(pose_frames),
            "media_url": storage_service.upload_file(video_path, f"{evaluation_id}.mp4"),
            "session_id": session_id or "",
            "timestamp": datetime.utcnow().isoformat(),
            "face_blurred": blur_faces,
        }
        results_service.save_result(record)

        return (
            overlay_path, _score_html(score, status.value),
            _tremor_html(signal), _fatigue_html(signal),
            _comp_html(signal), _rom_html(signal),
            _disorder_html(signal), _smoothness_data(signal),
            json.dumps(record, indent=2, default=str),
        )
    except Exception as e:
        logger.error(f"Gradio evaluation error: {e}", exc_info=True)
        return (None, _score_html(0), f"❌ Error: {e}", "", "", "", "", [], "{}")


def process_image(image_path, session_id, blur_faces):
    if image_path is None:
        return _empty_outputs()
    evaluation_id = str(uuid.uuid4())
    try:
        single = pose_detector.detect_from_image(image_path)
        pose_frames = [single] if single else []
        score, status, feedback, joint_angles, signal, _ = scoring_engine.evaluate(
            ExerciseType.ARM_RAISE,
            pose_frames
        )

        out_img = image_path
        if blur_faces:
            out_img = os.path.join(tempfile.gettempdir(), f"blurred_{evaluation_id}.jpg")
            pose_detector.blur_image_faces(image_path, out_img)

        raw = json.dumps({
            "evaluation_id": evaluation_id, "score": score,
            "status": status.value, "feedback": feedback,
            "signal_analysis": signal.model_dump() if signal else {},
        }, indent=2, default=str)

        return (
            out_img, _score_html(score, status.value),
            _tremor_html(signal), _fatigue_html(signal),
            _comp_html(signal), _rom_html(signal),
            _disorder_html(signal), _smoothness_data(signal), raw,
        )
    except Exception as e:
        return (None, _score_html(0), f"❌ {e}", "", "", "", "", [], "{}")


# ── HTML Builders ─────────────────────────────────────────────────────────────

_STATUS_COLORS = {
    "correct": "#22c55e", "partial": "#f59e0b",
    "incorrect": "#ef4444", "no_pose_detected": "#6b7280",
}

def _score_html(score, status="no_pose_detected"):
    color = _STATUS_COLORS.get(status, "#6b7280")
    pct   = int(score)
    label = status.replace("_", " ").title()
    return f"""<div style="text-align:center;padding:20px;background:#1e1e2e;border-radius:12px;">
  <svg viewBox="0 0 36 36" width="150" height="150">
    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
      fill="none" stroke="#2d2d3f" stroke-width="3"/>
    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
      fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{pct}, 100"/>
    <text x="18" y="20.5" text-anchor="middle" fill="white" font-size="8" font-weight="bold">{pct}</text>
  </svg>
  <div style="color:{color};font-size:1.1rem;font-weight:600;">{label}</div>
</div>"""

def _badge(level):
    c = {"none":"#22c55e","mild":"#f59e0b","moderate":"#f97316","severe":"#ef4444"}.get(level,"#6b7280")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:12px;font-size:0.82rem;font-weight:700;">{level.upper()}</span>'

def _card(title, content, bg="#2d2d3f"):
    return f'<div style="background:{bg};border-radius:10px;padding:14px;margin:6px 0;color:#e2e8f0;"><b style="font-size:0.97rem;">{title}</b><div style="margin-top:8px;font-size:0.9rem;line-height:1.6;">{content}</div></div>'

def _tremor_html(s):
    if not s:
        return _card("🫨 Tremor", "Not available")
    level = s.tremor_level.value
    bg = {"none":"#1a2e1a","mild":"#2e2a12","moderate":"#2e1e0a","severe":"#2e0a0a"}.get(level, "#2d2d3f")
    freq_str = f"{s.tremor_frequency_hz:.1f} Hz" if s.tremor_frequency_hz is not None else "N/A"
    return _card(
        "🫨 Tremor / Shakiness",
        f"{_badge(level)}<br><br><b>Score:</b> {s.tremor_score:.1f}/100 | <b>Dominant Freq:</b> {freq_str}",
        bg,
    )

def _fatigue_html(s):
    if not s:
        return _card("💪 Fatigue", "Not available")
    level = s.fatigue_level.value
    bg = {"none":"#1a2e1a","mild":"#2e2a12","moderate":"#2e1e0a","severe":"#2e0a0a"}.get(level, "#2d2d3f")
    decay_str = f"{s.velocity_decay_percent:.1f}%" if s.velocity_decay_percent is not None else "N/A"
    return _card(
        "💪 Muscle Fatigue",
        f"{_badge(level)}<br><br><b>Score:</b> {s.fatigue_score:.1f}/100 | <b>Velocity Decay:</b> {decay_str}",
        bg,
    )

def _comp_html(s):
    if not s:
        return _card("🔄 Compensation", "Not available")
    detected = s.compensation_detected
    bg = "#2e1e0a" if detected else "#1a2e1a"
    detail = s.compensation_details or ("No compensation pattern detected." if not detected else "Compensation pattern detected.")
    icon = "⚠️" if detected else "✅"
    return _card(f"{icon} Compensation / Trunk Alignment", detail, bg)

def _rom_html(s):
    if not s:
        return _card("📐 Range of Motion", "Not available")
    restricted = s.rom_restriction_angle is not None
    bg = "#2e1e0a" if restricted else "#1a2e1a"
    if restricted:
        content = f"ROM Restriction detected at ~{s.rom_restriction_angle:.0f}°. May indicate Frozen Shoulder / Adhesive Capsulitis."
        icon = "⚠️"
    else:
        content = "Full range of motion achieved."
        icon = "✅"
    return _card(f"{icon} Range of Motion", content, bg)

def _disorder_html(s):
    if not s or not s.disorder_probability or s.disorder_probability < 0.1:
        return _card("🏥 Clinical Flag", "✅ No disorder pattern detected.", "#1a2e1a")
    label = s.disorder_label or "Unspecified"
    prob  = s.disorder_probability
    color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.3 else "#22c55e"
    return _card(
        "🏥 Biomechanical Disorder Prediction",
        f"<b style='color:#f87171;'>{label}</b><br>"
        f"<b>Probability:</b> <span style='color:{color};font-weight:700;'>{prob*100:.0f}%</span><br><br>"
        f"<small style='color:#9ca3af;'>⚠️ AI flag only — not a medical diagnosis. Consult a clinician.</small>",
        "#2a1a2a",
    )

def _smoothness_data(s):
    if not s or not s.smoothness_curve:
        return []
    curve   = s.smoothness_curve
    angles  = s.angle_timeline or ([0.0] * len(curve))
    # Compute velocity from angle timeline differences
    ang_arr = np.array(angles[:len(curve)])
    vel_arr = np.gradient(ang_arr) if len(ang_arr) > 1 else np.zeros(len(ang_arr))
    return [
        {"Frame": i, "Angle (°)": round(ang_arr[i] if i < len(ang_arr) else 0.0, 2),
         "Velocity": round(float(vel_arr[i]) if i < len(vel_arr) else 0.0, 3),
         "Smoothness": round(float(v), 3)}
        for i, v in enumerate(curve)
    ]

def _empty_outputs():
    return (None, _score_html(0), "⬆️ Upload media to begin.", "", "", "", "", [], "{}")


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="violet", secondary_hue="blue", neutral_hue="slate",
).set(
    body_background_fill="#0f0f1a",
    block_background_fill="#1e1e2e",
    block_label_text_color="#a78bfa",
    input_background_fill="#2d2d3f",
    button_primary_background_fill="#7c3aed",
)

with gr.Blocks(title="PhysioEval — AI Biomechanical Framework") as demo:
    gr.Markdown("""
# 🦾 PhysioEval — AI Biomechanical Framework
**Signal Analysis · Tremor Detection · Fatigue Monitoring · Clinical Disorder Flags**
> *MediaPipe · AWS · Gradio · SciPy Signal Processing*
""")

    with gr.Tabs():
        with gr.TabItem("📹 Evaluate Exercise"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    with gr.Tab("Video"):
                        video_in  = gr.Video(label="Upload Video (MP4 / MOV / AVI)", height=260)
                        btn_video = gr.Button("🚀 Analyse Video", variant="primary")
                    with gr.Tab("Image"):
                        image_in  = gr.Image(label="Upload Image", type="filepath", height=260)
                        btn_image = gr.Button("🚀 Analyse Image", variant="primary")
                    session_id = gr.Textbox(label="Patient / Session ID (optional)", placeholder="e.g. PT-001")
                    blur_faces = gr.Checkbox(label="🔒 Face Blurring (HIPAA Privacy)", value=True)

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    with gr.Row():
                        overlay_out = gr.Video(label="Skeleton Overlay", height=260)
                        score_out   = gr.HTML(label="Score Gauge")
                    with gr.Row():
                        tremor_out  = gr.HTML()
                        fatigue_out = gr.HTML()
                    with gr.Row():
                        comp_out = gr.HTML()
                        rom_out  = gr.HTML()
                    disorder_out = gr.HTML()

        with gr.TabItem("📊 Smoothness Analytics"):
            gr.Markdown("### Movement Smoothness Curve — Angle · Velocity · Smoothness per Frame")
            smoothness_plot = gr.DataFrame(
                label="Smoothness Data",
                headers=["Frame", "Angle (°)", "Velocity", "Smoothness"],
                datatype=["number", "number", "number", "number"],
                column_count=(4, "fixed"),
            )

        with gr.TabItem("🔬 Raw Report JSON"):
            raw_json_out = gr.Code(language="json", label="Full Evaluation JSON", lines=40)

        with gr.TabItem("📋 Session History"):
            gr.Markdown("### Past Evaluations")
            refresh_btn = gr.Button("🔄 Refresh")
            history_out = gr.DataFrame(
                headers=["ID", "Exercise", "Score", "Status", "Tremor", "Fatigue", "Timestamp"],
                datatype=["str","str","number","str","str","str","str"],
            )
            def load_history():
                rows = []
                for r in results_service.get_all_results(limit=50):
                    sig = r.get("signal_analysis", {})
                    t = sig.get("tremor_level", "—") if sig else "—"
                    f = sig.get("fatigue_level", "—") if sig else "—"
                    rows.append([
                        r.get("evaluation_id","")[:8]+"…",
                        r.get("exercise_type",""),
                        float(r.get("score",0)),
                        r.get("status",""),
                        t, f,
                        r.get("timestamp","")[:19],
                    ])
                return rows
            refresh_btn.click(fn=load_history, outputs=history_out)

    _outs = [overlay_out, score_out, tremor_out, fatigue_out, comp_out, rom_out, disorder_out, smoothness_plot, raw_json_out]
    btn_video.click(fn=process_video, inputs=[video_in, session_id, blur_faces], outputs=_outs)
    btn_image.click(fn=process_image, inputs=[image_in, session_id, blur_faces], outputs=_outs)

    gr.Markdown("""
---
<center><small>⚕️ <b>Disclaimer:</b> AI-generated clinical flags are NOT medical diagnoses. Consult a qualified clinician. 🔒 Face blurring for HIPAA compliance.</small></center>
""")


def create_gradio_app():
    return demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

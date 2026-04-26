from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.api.routes import router
from app.gradio_app import create_gradio_app
from app.monitoring.metrics import setup_metrics

app = FastAPI(title="PhysioEval API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "PhysioEval API running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "physio-eval", "version": "2.0.0"}


gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

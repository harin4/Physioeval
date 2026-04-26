from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PhysioEval Enhanced"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    VERSION: str = "2.0.0"

    # AWS Core
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # S3
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "physio-eval-uploads")
    S3_ANNOTATED_PREFIX: str = "annotated/"
    S3_RAW_PREFIX: str = "raw/"

    # DynamoDB
    DYNAMODB_TABLE_NAME: str = os.getenv("DYNAMODB_TABLE_NAME", "physio-eval-results")

    # AWS Rekognition — face blurring
    REKOGNITION_ENABLED: bool = os.getenv("REKOGNITION_ENABLED", "false").lower() == "true"
    FACE_BLUR_RADIUS: int = 31  # Gaussian blur kernel size (must be odd)

    # AWS SageMaker — disorder probability
    SAGEMAKER_ENDPOINT: Optional[str] = os.getenv("SAGEMAKER_ENDPOINT")
    SAGEMAKER_ENABLED: bool = os.getenv("SAGEMAKER_ENABLED", "false").lower() == "true"

    # AWS Lambda — event-driven trigger (set via S3 notification, documented only)
    LAMBDA_FUNCTION_NAME: Optional[str] = os.getenv("LAMBDA_FUNCTION_NAME")

    # Upload limits
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/mov", "video/quicktime"]
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/jpg"]

    # Scoring thresholds
    ARM_RAISE_CORRECT_ANGLE: float = 150.0
    ARM_RAISE_PARTIAL_ANGLE: float = 90.0

    # Signal analysis thresholds
    TREMOR_MILD_THRESHOLD: float = 20.0       # jerk RMS threshold → mild
    TREMOR_MODERATE_THRESHOLD: float = 45.0   # jerk RMS threshold → moderate
    TREMOR_SEVERE_THRESHOLD: float = 70.0     # jerk RMS threshold → severe
    FATIGUE_MILD_DECAY: float = 20.0          # velocity decay % → mild
    FATIGUE_MODERATE_DECAY: float = 40.0      # velocity decay % → moderate
    FATIGUE_SEVERE_DECAY: float = 60.0        # velocity decay % → severe
    COMPENSATION_LEAN_THRESHOLD: float = 0.08 # normalised hip drift to flag lean

    # Video annotation
    DRAW_SKELETON: bool = True
    DRAW_ANGLES: bool = True
    ANNOTATED_VIDEO_FPS: int = 15

    # Local fallback (when AWS not configured)
    USE_LOCAL_STORAGE: bool = os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true"
    LOCAL_UPLOAD_DIR: str = os.getenv("LOCAL_UPLOAD_DIR", "./uploads")
    LOCAL_RESULTS_DIR: str = os.getenv("LOCAL_RESULTS_DIR", "./results")
    LOCAL_ANNOTATED_DIR: str = os.getenv("LOCAL_ANNOTATED_DIR", "./annotated")

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()

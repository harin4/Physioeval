import os
import mimetypes
from fastapi import UploadFile, HTTPException

from app.core.config import settings
from app.core.logger import logger


ALLOWED_TYPES = (
    settings.ALLOWED_VIDEO_TYPES + settings.ALLOWED_IMAGE_TYPES
)


def validate_upload_file(file: UploadFile) -> str:
    """
    Validate uploaded file type and size.
    Returns 'video' or 'image'.
    Raises HTTPException on invalid input.
    """
    content_type = file.content_type or ""

    # Check by declared content type
    is_video = content_type in settings.ALLOWED_VIDEO_TYPES
    is_image = content_type in settings.ALLOWED_IMAGE_TYPES

    if not is_video and not is_image:
        # Try by extension as fallback
        name = file.filename or ""
        ext = os.path.splitext(name)[-1].lower()
        video_exts = [".mp4", ".avi", ".mov"]
        image_exts = [".jpg", ".jpeg", ".png"]
        if ext in video_exts:
            is_video = True
        elif ext in image_exts:
            is_image = True
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}. Allowed: mp4, avi, mov, jpg, png",
            )

    return "video" if is_video else "image"


def check_file_size(file_path: str) -> None:
    """Raise HTTPException if file exceeds max size."""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {settings.MAX_FILE_SIZE_MB} MB",
        )

"""
storage_service.py — S3 upload with event notification hooks.

Adds:
  - Separate prefixes for raw vs annotated media
  - S3 event notification tagging (triggers Lambda)
  - Presigned URL generation for response
  - Local fallback unchanged
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logger import logger


class StorageService:
    """Handles file storage in S3 or local filesystem."""

    def __init__(self):
        self._s3 = None
        if not settings.USE_LOCAL_STORAGE:
            try:
                import boto3
                self._s3 = boto3.client(
                    "s3",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
                logger.info(f"StorageService: S3 bucket '{settings.S3_BUCKET_NAME}'")
            except Exception as e:
                logger.warning(f"S3 init failed, falling back to local: {e}")
                self._s3 = None

        if self._s3 is None:
            os.makedirs(settings.LOCAL_UPLOAD_DIR,    exist_ok=True)
            os.makedirs(settings.LOCAL_ANNOTATED_DIR, exist_ok=True)
            logger.info("StorageService: local filesystem mode")

    # ------------------------------------------------------------------

    def upload_raw(self, local_path: str, filename: str) -> str:
        """Upload raw (unprocessed) media. Returns URL / local path."""
        key = f"{settings.S3_RAW_PREFIX}{uuid.uuid4()}_{filename}"
        return self._upload(local_path, key, tag="raw")

    def upload_annotated(self, local_path: str, filename: str) -> str:
        """
        Upload annotated/face-blurred video.
        S3 Event Notification on this prefix triggers Lambda for async analysis.
        Returns URL / local path.
        """
        key = f"{settings.S3_ANNOTATED_PREFIX}{uuid.uuid4()}_{filename}"
        return self._upload(local_path, key, tag="annotated")

    # Legacy alias
    def upload_file(self, local_path: str, filename: str) -> str:
        return self.upload_raw(local_path, filename)

    # ------------------------------------------------------------------

    def _upload(self, local_path: str, key: str, tag: str = "") -> str:
        if self._s3:
            return self._upload_s3(local_path, key, tag)
        return self._copy_local(local_path, tag)

    def _upload_s3(self, local_path: str, key: str, tag: str) -> str:
        try:
            extra = {}
            if tag:
                extra["Tagging"] = f"type={tag}"   # triggers S3 event filter if configured

            self._s3.upload_file(
                local_path,
                settings.S3_BUCKET_NAME,
                key,
                ExtraArgs=extra,
            )
            # Return presigned URL (24-hour access)
            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=86400,
            )
            logger.info(f"Uploaded to S3: {key}")
            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return self._copy_local(local_path, tag)

    def _copy_local(self, local_path: str, tag: str) -> str:
        dest_dir = (
            settings.LOCAL_ANNOTATED_DIR if tag == "annotated"
            else settings.LOCAL_UPLOAD_DIR
        )
        fname = os.path.basename(local_path)
        dest  = os.path.join(dest_dir, fname)
        shutil.copy2(local_path, dest)
        return f"local://{dest}"

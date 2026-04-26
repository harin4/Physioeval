"""results_service.py — DynamoDB / local JSON persistence (unchanged structure, extended schema)."""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.core.config import settings
from app.core.logger import logger


class ResultsService:

    def __init__(self):
        self._table = None
        if not settings.USE_LOCAL_STORAGE:
            try:
                import boto3
                ddb = boto3.resource(
                    "dynamodb",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
                self._table = ddb.Table(settings.DYNAMODB_TABLE_NAME)
                logger.info(f"DynamoDB table: {settings.DYNAMODB_TABLE_NAME}")
            except Exception as e:
                logger.warning(f"DynamoDB init failed, using local: {e}")

        if self._table is None:
            os.makedirs(settings.LOCAL_RESULTS_DIR, exist_ok=True)
            logger.info("ResultsService: local JSON mode")

    def save_result(self, record: Dict[str, Any]) -> bool:
        try:
            if self._table:
                self._table.put_item(Item=record)
            else:
                path = os.path.join(
                    settings.LOCAL_RESULTS_DIR,
                    f"{record['evaluation_id']}.json"
                )
                with open(path, "w") as f:
                    json.dump(record, f, indent=2, default=str)
            logger.info(f"Result saved: {record.get('evaluation_id')}")
            return True
        except Exception as e:
            logger.error(f"Save result failed: {e}")
            return False

    def get_result(self, evaluation_id: str) -> Optional[Dict]:
        try:
            if self._table:
                resp = self._table.get_item(Key={"evaluation_id": evaluation_id})
                return resp.get("Item")
            path = os.path.join(settings.LOCAL_RESULTS_DIR, f"{evaluation_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Get result failed: {e}")
        return None

    def get_all_results(self, limit: int = 20) -> List[Dict]:
        try:
            if self._table:
                resp = self._table.scan(Limit=limit)
                items = resp.get("Items", [])
                items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return items[:limit]
            results = []
            for fname in os.listdir(settings.LOCAL_RESULTS_DIR):
                if fname.endswith(".json"):
                    with open(os.path.join(settings.LOCAL_RESULTS_DIR, fname)) as f:
                        results.append(json.load(f))
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"Get all results failed: {e}")
            return []

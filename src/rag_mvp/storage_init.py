"""Create the configured S3-compatible bucket for a fresh Compose deployment."""

from __future__ import annotations

import os
import time

from boto3 import Session
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger


def main() -> None:
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000").strip()
    access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    bucket = os.environ.get("MINIO_BUCKET", "edu-materials").strip()
    region = os.environ.get("MINIO_REGION", "us-east-1").strip()
    if not endpoint or not access_key or not secret_key or not bucket:
        raise RuntimeError("MINIO_ENDPOINT, credentials, and MINIO_BUCKET are required")

    client = Session().client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    deadline = time.monotonic() + 120
    while True:
        try:
            client.list_buckets()
            break
        except (BotoCoreError, ClientError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("Object storage did not become ready within 120 seconds") from exc
            time.sleep(2)

    try:
        client.head_bucket(Bucket=bucket)
        logger.info("Object storage bucket already exists: {}", bucket)
        return
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if status not in {400, 404}:
            raise

    client.create_bucket(Bucket=bucket)
    logger.info("Object storage bucket created: {}", bucket)

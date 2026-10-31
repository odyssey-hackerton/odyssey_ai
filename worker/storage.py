import asyncio

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from worker.config import Settings

RESULTS_PREFIX = "results/"


def result_key(job_id: str, ext: str) -> str:
    return f"{RESULTS_PREFIX}{job_id}.{ext}"


class Storage:
    """AWS S3. 원본은 originals/ 에서 읽고, 결과는 results/ 에 쓴다.

    boto3는 동기 라이브러리라 네트워크 호출은 스레드에서 실행한다.
    """

    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.configured = bool(settings.aws_access_key_id and settings.aws_secret_access_key)
        self._client: BaseClient = boto3.client(
            "s3",
            region_name=settings.s3_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(
                signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}
            ),
        )

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

        return await asyncio.to_thread(_get)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    async def delete(self, key: str) -> None:
        """Delete the exact temporary key, including retained S3 versions."""
        if not key.startswith("originals/"):
            raise ValueError("only temporary originals may be deleted")

        def _delete() -> None:
            pages = self._client.get_paginator("list_object_versions")
            for page in pages.paginate(Bucket=self.bucket, Prefix=key):
                objects = [
                    {"Key": key, "VersionId": item["VersionId"]}
                    for item in page.get("Versions", []) + page.get("DeleteMarkers", [])
                    if item["Key"] == key
                ]
                if objects:
                    result = self._client.delete_objects(
                        Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True}
                    )
                    if result.get("Errors"):
                        raise RuntimeError("temporary image deletion failed")

        await asyncio.to_thread(_delete)

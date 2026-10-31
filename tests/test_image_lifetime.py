import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image, PngImagePlugin
from redis.exceptions import ConnectionError as RedisConnectionError

from stages.types import Assessment
from tests.test_worker import FakeRedis, FakeStorage, job
from worker.config import Settings
from worker.contract import Finding, ResultMessage
from worker.main import OriginalCleanupError, edited_key, execution_key, handle, process


def image_job():
    image = Image.new("RGB", (20, 20), "white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private", "synthetic-original-metadata")
    out = BytesIO()
    image.save(out, format="PNG", pnginfo=metadata)
    msg = job("image", "originals/test.png")
    storage = FakeStorage()
    storage.objects[msg.object_key] = out.getvalue()
    return msg, storage, FakeRedis()


async def test_detection_deletes_before_analysis_and_recovers_without_original(monkeypatch):
    msg, storage, redis = image_job()
    detector = AsyncMock(return_value=[])
    monkeypatch.setattr("worker.pipeline.detect", detector)

    async def analyze(media, findings):
        assert media.image is None
        assert msg.object_key not in storage.objects
        assert await redis.get(f"{execution_key(msg)}:checkpoint")
        return Assessment(risky=False, risk=0, findings=findings)

    monkeypatch.setattr("worker.pipeline.analyze", analyze)
    first = await process(redis, storage, Settings(), msg)
    replay = await process(redis, storage, Settings(), msg)
    assert replay == first and first.status == "AWAITING_USER"
    detector.assert_awaited_once()


async def test_empty_findings_still_strip_metadata_and_delete_before_verification(monkeypatch):
    msg, storage, redis = image_job()
    original = storage.objects[msg.object_key]
    msg = msg.model_copy(update={"user_choices": [], "initial_findings": []})

    async def verify(media, watchlist=None):
        assert msg.object_key not in storage.objects
        assert edited_key(msg) in storage.objects
        with Image.open(BytesIO(media.image)) as image:
            assert "private" not in image.info
        return []

    monkeypatch.setattr("worker.pipeline.detect", verify)
    await handle(redis, storage, Settings(), "1-0", {"payload": msg.model_dump_json()})
    result = ResultMessage.model_validate_json(redis.results[0]["payload"])
    assert result.status == "DONE" and result.iterations == 1
    assert storage.objects[result.result_object_key] != original
    assert not any(key.startswith("originals/") for key in storage.objects)


async def test_verification_restart_uses_edited_checkpoint(monkeypatch):
    msg, storage, redis = image_job()
    msg = msg.model_copy(update={"user_choices": [], "initial_findings": []})
    monkeypatch.setattr("worker.pipeline.detect", AsyncMock(side_effect=RuntimeError("restart")))
    with pytest.raises(RuntimeError):
        await process(redis, storage, Settings(), msg)
    assert msg.object_key not in storage.objects
    assert json.loads(await redis.get(f"{execution_key(msg)}:checkpoint"))["stage"] == "edited"
    monkeypatch.setattr(
        "worker.pipeline.transform", AsyncMock(side_effect=AssertionError("re-edit"))
    )
    monkeypatch.setattr("worker.pipeline.detect", AsyncMock(return_value=[]))
    result = await process(redis, storage, Settings(), msg)
    assert result.status == "DONE"


async def test_completed_execution_replays_without_model_or_original(monkeypatch):
    msg, storage, redis = image_job()
    detector = AsyncMock(return_value=[])
    monkeypatch.setattr("worker.pipeline.detect", detector)
    fields = {"payload": msg.model_dump_json()}
    await handle(redis, storage, Settings(), "1-0", fields)
    await handle(redis, storage, Settings(), "1-0", fields)
    assert redis.results[0] == redis.results[1]
    assert len(redis.acks) == 2
    detector.assert_awaited_once()


async def test_failed_model_still_deletes_original(monkeypatch):
    msg, storage, redis = image_job()
    monkeypatch.setattr("worker.pipeline.detect", AsyncMock(side_effect=TimeoutError()))
    await handle(redis, storage, Settings(), "1-0", {"payload": msg.model_dump_json()})
    assert msg.object_key not in storage.objects
    assert json.loads(redis.results[0]["payload"])["status"] == "FAILED"


async def test_failed_deletion_blocks_publication_and_retries_from_checkpoint(monkeypatch):
    msg, storage, redis = image_job()
    detector = AsyncMock(return_value=[])
    monkeypatch.setattr("worker.pipeline.detect", detector)
    delete = storage.delete
    storage.delete = AsyncMock(side_effect=RuntimeError("S3 unavailable"))
    fields = {"payload": msg.model_dump_json()}
    with pytest.raises(OriginalCleanupError):
        await handle(redis, storage, Settings(), "1-0", fields)
    assert not redis.acks and not redis.results
    storage.delete = delete
    await handle(redis, storage, Settings(), "1-0", fields)
    detector.assert_awaited_once()
    assert msg.object_key not in storage.objects


async def test_held_execution_lock_does_not_run_or_ack(monkeypatch):
    msg, storage, redis = image_job()
    redis.lock(f"{execution_key(msg)}:lock").acquire.return_value = False
    detector = AsyncMock()
    monkeypatch.setattr("worker.pipeline.detect", detector)
    await handle(redis, storage, Settings(), "1-0", {"payload": msg.model_dump_json()})
    detector.assert_not_called()
    assert not redis.acks


async def test_selected_snapshot_still_applies_after_reupload(monkeypatch):
    msg, storage, redis = image_job()
    msg = msg.model_copy(
        update={
            "user_choices": [],
            "initial_findings": [
                Finding(type="face", source="vlm", bbox=[0, 0, 10, 10], action="mask")
            ],
        }
    )
    detector = AsyncMock(return_value=[])
    monkeypatch.setattr("worker.pipeline.detect", detector)
    result = await process(redis, storage, Settings(), msg)
    with Image.open(BytesIO(storage.objects[result.result_object_key])) as image:
        assert image.getpixel((5, 5)) == (20, 33, 56)
    detector.assert_awaited_once()  # 재검증만 호출한다.


async def test_checkpoint_failure_still_deletes_original(monkeypatch):
    msg, storage, redis = image_job()
    monkeypatch.setattr("worker.pipeline.detect", AsyncMock(return_value=[]))
    redis.setex = AsyncMock(side_effect=RedisConnectionError())
    with pytest.raises(RedisConnectionError):
        await handle(redis, storage, Settings(), "1-0", {"payload": msg.model_dump_json()})
    assert msg.object_key not in storage.objects
    assert not redis.acks and not redis.results

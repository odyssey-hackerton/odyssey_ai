import uuid
from datetime import UTC, datetime

import pytest

from stages.types import Assessment, Media
from worker.config import Settings
from worker.contract import JobMessage, JobStatus, ResultMessage
from worker.main import process
from worker.pipeline import run_pipeline


class FakeStorage:
    configured = True

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    async def delete(self, key):
        self.objects.pop(key, None)


class FakePipe:
    def __init__(self, store: dict) -> None:
        self.store = store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def hset(self, key, mapping):
        self.store.setdefault(key, []).append(dict(mapping))

    def expire(self, key, ttl):
        pass

    async def execute(self):
        pass


class FakeRedis:
    def __init__(self) -> None:
        self.progress: dict[str, list[dict]] = {}
        self.values = {}
        self.results = []
        self.acks = []
        self.locks = {}

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)

    async def xadd(self, stream, fields, **kwargs):
        self.results.append(fields)

    async def xack(self, *args):
        self.acks.append(args)

    def lock(self, key, **kwargs):
        from unittest.mock import AsyncMock

        if key not in self.locks:
            self.locks[key] = AsyncMock(acquire=AsyncMock(return_value=True))
        return self.locks[key]

    def pipeline(self, transaction=True):
        return FakePipe(self.progress)


def job(input_type, object_key=None, text=None) -> JobMessage:
    return JobMessage(
        job_id=uuid.uuid4(),
        execution_id=uuid.uuid4(),
        input_type=input_type,
        object_key=object_key,
        text=text,
        created_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("input_type", "has_image", "text"),
    [("image", True, None), ("text", False, "010-1234-5678"), ("image_text", True, "본문")],
)
async def test_detection_waits_for_user(input_type, has_image, text, monkeypatch):
    """각 입력은 탐지 결과를 먼저 반환하고 사용자 선택을 기다린다."""
    from unittest.mock import AsyncMock

    monkeypatch.setattr("worker.pipeline.detect", AsyncMock(return_value=[]))
    storage, redis = FakeStorage(), FakeRedis()
    key = None
    if has_image:
        key = "originals/x.png"
        storage.objects[key] = b"png-bytes"
    msg = job(input_type, key, text)

    result = await process(redis, storage, Settings(), msg)

    assert isinstance(result, ResultMessage) and result.status == "AWAITING_USER"
    assert result.result_text is None and result.result_object_key is None
    statuses = [p["status"] for p in redis.progress[f"job:{msg.job_id}"]]
    assert statuses == [JobStatus.DETECTING, JobStatus.ANALYZING]


async def test_pipeline_loops_until_safe(monkeypatch):
    """위험이 남아 있으면 변환 → 재검증을 반복하고, 안전해지면 멈춘다."""
    import worker.pipeline as p

    calls = {"analyze": 0}

    async def fake_detect(media, watchlist=None):
        return [{"type": "phone_number"}]

    async def fake_analyze(media, findings):
        calls["analyze"] += 1
        risky = calls["analyze"] < 3  # 처음 분석 + 1회차 재검증은 위험, 2회차에 안전
        return Assessment(risky=risky, risk=0.8 if risky else 0.1, findings=findings)

    async def fake_transform(media, assessment):
        return Media(text="masked")

    monkeypatch.setattr(p, "detect", fake_detect)
    monkeypatch.setattr(p, "analyze", fake_analyze)
    monkeypatch.setattr(p, "transform", fake_transform)

    seen = []

    async def report(status, iteration=0, message=None):
        seen.append((status, iteration))

    r = await run_pipeline(Media(text="010-1234-5678"), max_iterations=3, report=report)
    assert r.iterations == 2
    assert (r.before.risk, r.after.risk) == (0.8, 0.1)
    assert r.output.text == "masked"
    assert seen[-1] == (JobStatus.VERIFYING, 2)


async def test_pipeline_stops_at_max_iterations(monkeypatch):
    import worker.pipeline as p

    async def always_risky(media, findings):
        return Assessment(risky=True, risk=0.9)

    from unittest.mock import AsyncMock

    monkeypatch.setattr(p, "detect", AsyncMock(return_value=[]))
    monkeypatch.setattr(p, "transform", AsyncMock(return_value=Media(text="x")))
    monkeypatch.setattr(p, "analyze", always_risky)

    async def report(*a, **k):
        pass

    r = await run_pipeline(Media(text="x"), max_iterations=3, report=report)
    assert r.iterations == 3

import json
from io import BytesIO
from unittest.mock import AsyncMock
from urllib.error import HTTPError

import pytest
from PIL import Image

from stages import gemini
from stages.detect import Detection, detect
from stages.images import open_image, png_bytes
from stages.transform import transform
from stages.types import Assessment, Media
from worker.contract import UserChoice
from worker.pipeline import run_pipeline


def candidate(**changes):
    return (
        dict(
            type="phone_number",
            source="text",
            title="연락처",
            description="연락처 노출",
            risk_level="high",
            score=0.9,
            box_2d=None,
            text="010-0000-0000",
        )
        | changes
    )


@pytest.fixture(autouse=True)
def no_provider_calls(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Tests must not call an external AI service")

    monkeypatch.setattr(gemini, "urlopen", fail)


async def test_real_edit_flow_uses_snapshot_and_verifies_once(monkeypatch):
    """사진+본문 → 탐지 → 선택 → 실제 픽셀/본문 수정 → 잔여 위험 재검증."""
    image = png_bytes(Image.new("RGB", (100, 80), "white"))
    media = Media(image=image, image_ext="png", text="010-0000-0000 그대로 010-0000-0000")
    provider = AsyncMock(
        side_effect=[
            {
                "findings": [
                    candidate(),
                    candidate(source="image", text=None, box_2d=[100, 100, 500, 500]),
                ]
            },
            {"findings": [candidate(source="image", text=None, box_2d=[100, 100, 500, 500])]},
        ]
    )
    monkeypatch.setattr("stages.detect.generate_json", provider)
    crop = AsyncMock(return_value={"complete": True, "exposes": []})
    monkeypatch.setattr("stages.risk.generate_json", crop)
    findings = await detect(media)
    assert findings[1]["bbox"] == [10, 8, 50, 40]
    assert "text" not in findings[0] and len(findings[0]["spans"]) == 2
    result = await run_pipeline(
        media,
        max_iterations=3,
        report=AsyncMock(),
        initial_findings=findings,
        user_choices=[UserChoice(idx=0, action="mask"), UserChoice(idx=1, action="skip")],
    )
    assert result.output.text == "[가림] 그대로 [가림]"
    # 남겨 둔 전화번호는 재검증에서 다시 잡힌다. 단독 연락처는 ENISA 기준 낮음(1×0.75+0.5).
    assert result.iterations == 1 and not result.after.risky
    assert [f["risk_level"] for f in result.after.findings] == ["low"]
    crop.assert_awaited_once()  # 재검증 때 남은 조각 하나만 VLM 에 넣는다
    assert provider.await_count == 2  # 최초 분석 + 재검증, 선택 전 재탐지 없음
    assert open_image(result.output.image).getpixel((20, 20)) == (255, 255, 255)
    masked = await transform(media, Assessment(risky=True, risk=0.9, findings=[findings[1]]))
    assert open_image(masked.image).getpixel((20, 20)) == (20, 33, 56)
    assert masked.text == media.text and media.image == image


@pytest.mark.parametrize(
    "finding",
    [
        candidate(text="not in input"),
        candidate(source="image", text=None, box_2d=[0, 20, 100, 10]),
        candidate(score=float("nan")),
        candidate(risk_level="safe"),
    ],
)
async def test_invalid_findings_fail_closed(monkeypatch, finding):
    monkeypatch.setattr(
        "stages.detect.generate_json", AsyncMock(return_value={"findings": [finding]})
    )
    with pytest.raises(gemini.AIServiceError):
        await detect(Media(text="010-0000-0000", image=png_bytes(Image.new("RGB", (10, 10)))))


async def test_interactions_request_and_output(monkeypatch):
    def request(payload):
        assert payload["store"] is False
        assert payload["response_format"]["mime_type"] == "application/json"
        return {
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": json.dumps({"findings": []})}],
                }
            ],
        }

    monkeypatch.setattr(gemini, "_request", request)
    assert await gemini.generate_json([], {}, "instruction") == {"findings": []}


@pytest.mark.parametrize(
    "response",
    [
        {"status": "incomplete"},
        {"status": "completed", "steps": []},
        {"status": "completed", "steps": None},
        {
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "not JSON"}]}],
        },
    ],
)
async def test_provider_failure_never_becomes_safe(monkeypatch, response):
    monkeypatch.setattr(gemini, "_request", lambda payload: response)
    with pytest.raises(gemini.AIServiceError):
        await gemini.generate_json([], {}, "instruction")


def test_rate_limit_has_no_retry_or_secret_in_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    gemini.get_settings.cache_clear()
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, 429, "private provider body", {}, None)

    monkeypatch.setattr(gemini, "urlopen", fail)
    try:
        with pytest.raises(gemini.AIServiceError, match="HTTP 429") as error:
            gemini._request({})
        assert "private" not in str(error.value) and len(calls) == 1
    finally:
        gemini.get_settings.cache_clear()


async def test_orientation_metadata_and_overlapping_text():
    image = Image.new("RGB", (30, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    raw = BytesIO()
    image.save(raw, format="JPEG", exif=exif)
    assert open_image(raw.getvalue()).size == (20, 30)
    media = Media(image=raw.getvalue(), text="abcdefghi")
    result = await transform(
        media,
        Assessment(
            risky=True,
            risk=0.9,
            findings=[
                {"source": "text", "spans": [[1, 4]], "action": "remove"},
                {"source": "text", "spans": [[3, 6]], "action": "mask"},
                {"source": "exif", "action": "remove"},
            ],
        ),
    )
    assert result.text == "a[가림]ghi"
    with Image.open(BytesIO(result.image)) as output:
        assert not output.getexif() and output.size == (20, 30)


async def test_choices_require_original_findings():
    with pytest.raises(ValueError, match="initial findings"):
        await run_pipeline(Media(text="x"), max_iterations=3, report=AsyncMock(), user_choices=[])


async def test_removing_entire_caption_finishes_without_provider():
    result = await run_pipeline(
        Media(text="secret"),
        max_iterations=3,
        report=AsyncMock(),
        initial_findings=[
            {
                "type": "name",
                "source": "text",
                "spans": [[0, 6]],
                "risk_level": "high",
                "action": "mask",
            }
        ],
        user_choices=[UserChoice(idx=0, action="remove")],
    )
    assert result.output.text == "" and not result.after.risky


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
async def test_transparent_rgb_never_becomes_visible(format):
    image = Image.new("RGBA", (20, 20), (255, 0, 0, 0))
    image.putpixel((0, 0), (0, 0, 0, 255))
    raw = BytesIO()
    image.save(raw, format=format, lossless=True)
    decoded = open_image(raw.getvalue())
    assert decoded.getpixel((5, 5)) == (255, 255, 255)
    assert decoded.getpixel((0, 0)) == (0, 0, 0)
    result = await transform(
        Media(image=raw.getvalue(), text="secret"),
        Assessment(
            risky=True,
            risk=0.9,
            findings=[{"source": "text", "spans": [[0, 6]], "action": "remove"}],
        ),
    )
    assert open_image(result.image).getpixel((5, 5)) == (255, 255, 255)


def test_detection_schema_sent_without_limits_gemini_rejects():
    sent = json.dumps(gemini.serving_schema(Detection.model_json_schema()))
    for key in ("maxLength", "minLength", "maxItems", "minimum", "maximum"):
        assert key not in sent, key
    assert '"findings"' in sent and '"box_2d"' in sent


def test_serving_schema_keeps_property_names():
    schema = {"properties": {"maximum": {"type": "number", "maximum": 3}}, "maxItems": 2}
    assert gemini.serving_schema(schema) == {"properties": {"maximum": {"type": "number"}}}


async def test_long_findings_are_cut_and_boxes_clamped(monkeypatch):
    long = candidate(
        source="image",
        text=None,
        title="가" * 200,
        description="나" * 900,
        box_2d=[-5, 0, 1004, 500],
        score=1.2,
    )
    monkeypatch.setattr(
        "stages.detect.generate_json", AsyncMock(return_value={"findings": [long] * 40})
    )
    findings = await detect(Media(image=png_bytes(Image.new("RGB", (100, 100)))))
    assert len(findings) == 30
    assert len(findings[0]["title"]) == 80 and len(findings[0]["description"]) == 300
    assert findings[0]["bbox"] == [0, 0, 50, 100] and findings[0]["score"] == 1.0


def test_overloaded_model_is_retried_but_not_forever(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    gemini.get_settings.cache_clear()
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    calls = []

    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b'{"status": "completed"}'

    def flaky(request, **kwargs):
        calls.append(request)
        if len(calls) < 3:
            raise HTTPError(request.full_url, 503, "busy", {}, None)
        return Ok()

    def busy(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, 503, "busy", {}, None)

    try:
        monkeypatch.setattr(gemini, "urlopen", flaky)
        assert gemini._request({}) == {"status": "completed"} and len(calls) == 3
        calls.clear()
        monkeypatch.setattr(gemini, "urlopen", busy)
        with pytest.raises(gemini.AIServiceError, match="HTTP 503"):
            gemini._request({})
        assert len(calls) == 1 + len(gemini.RETRY_DELAYS)
    finally:
        gemini.get_settings.cache_clear()

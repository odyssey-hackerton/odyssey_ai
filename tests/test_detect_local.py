"""에이전트1 로컬 탐지(OCR·PII·EXIF)와 병합 규칙. 외부 모델·API 는 호출하지 않는다."""

from io import BytesIO
from unittest.mock import AsyncMock

from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from stages import detect as d
from stages import ocr
from stages.exif import exif_findings
from stages.images import png_bytes
from stages.pii import detect_pii
from stages.types import Media
from worker.config import get_settings


def test_regex_pii_and_dedupe():
    spans = detect_pii(
        "김나리 010-1234-5821 / 서울 성북구 보문로 2 한누리빌 302동 1203호 / 880101-1234567"
    )
    types = {s.type for s in spans}
    assert {"phone_number", "apt_dong_ho", "rrn"} <= types
    # 주민번호 안의 숫자열이 송장번호로 중복 잡히지 않는다
    assert sum(1 for s in spans if s.type == "tracking_number") == 0


def test_text_findings_have_spans_not_text():
    f = d._text_findings("연락처 010-0000-0000 입니다")
    assert f[0]["source"] == "text" and f[0]["spans"] == [[4, 17]]
    assert "text" not in f[0] and f[0]["risk_level"] == "high" and f[0]["action"] == "mask"


def test_ocr_findings_bbox_inside_image(monkeypatch):
    img = Image.new("RGB", (200, 100), "white")
    monkeypatch.setattr(
        d, "read_lines", lambda image: [ocr.OcrLine("전화 010-1234-5678", 0.9, [10, 20, 190, 40])]
    )
    f = d._ocr_findings(img)
    assert len(f) == 1 and f[0]["source"] == "ocr" and f[0]["type"] == "phone_number"
    x1, y1, x2, y2 = f[0]["bbox"]
    assert 10 <= x1 < x2 <= 190 and (y1, y2) == (20, 40)


def test_exif_gps_rounded():
    exif = Image.Exif()
    gps = exif.get_ifd(34853)
    gps[1], gps[2] = "N", (IFDRational(37), IFDRational(35), IFDRational(27))
    gps[3], gps[4] = "E", (IFDRational(127), IFDRational(1), IFDRational(7))
    exif[271], exif[36867] = "Apple", "2026:10:31 18:42:10"
    raw = BytesIO()
    Image.new("RGB", (10, 10)).save(raw, format="JPEG", exif=exif)
    f = {x["type"]: x for x in exif_findings(raw.getvalue())}
    assert f["gps"]["details"] == {"lat": 37.59, "lon": 127.02} and f["gps"]["action"] == "remove"
    assert f["capture_time"]["details"]["date"] == "2026:10:31"


def test_merge_prefers_vlm_and_dedupes_spans():
    vlm = [
        {
            "type": "phone_number",
            "source": "text",
            "spans": [[0, 13]],
            "score": 0.9,
            "action": "mask",
        }
    ]
    local = [
        {
            "type": "phone_number",
            "source": "text",
            "detector": "regex",
            "spans": [[0, 13]],
            "score": 0.85,
        },
        {"type": "face", "source": "ocr", "bbox": [0, 0, 10, 10], "score": 0.5},
    ]
    merged = d._merge(vlm, local)
    assert len(merged) == 2 and merged[0]["detectors"] == ["regex", "text"]


def test_merge_iou_same_box():
    a = [{"type": "face", "source": "vlm", "bbox": [0, 0, 100, 100], "score": 0.7}]
    b = [{"type": "face", "source": "ocr", "bbox": [5, 5, 100, 100], "score": 0.9}]
    merged = d._merge(a, b)
    assert len(merged) == 1 and merged[0]["score"] == 0.9


async def test_detect_without_vlm(monkeypatch):
    monkeypatch.setenv("DETECT_USE_VLM", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(d, "generate_json", AsyncMock(side_effect=AssertionError("no VLM call")))
    monkeypatch.setattr(d, "read_lines", lambda image: [])
    try:
        media = Media(
            image=png_bytes(Image.new("RGB", (50, 50))), image_ext="png", text="010-1111-2222"
        )
        findings = await d.detect(media)
    finally:
        get_settings.cache_clear()
    assert [f["type"] for f in findings] == ["phone_number"]


async def test_vlm_provider_dispatch(monkeypatch):
    from stages import gemini, qwen, vlm

    monkeypatch.setattr(gemini, "generate_json", AsyncMock(return_value={"findings": [], "p": "g"}))
    monkeypatch.setattr(qwen, "generate_json", AsyncMock(return_value={"findings": [], "p": "q"}))
    for provider, expected in (("gemini", "g"), ("qwen", "q")):
        monkeypatch.setenv("VLM_PROVIDER", provider)
        get_settings.cache_clear()
        try:
            assert (await vlm.generate_json([], {}, "i"))["p"] == expected
        finally:
            get_settings.cache_clear()


def test_qwen_request_shape():
    from stages import qwen

    parts = qwen._to_openai_content(
        [
            {"type": "image", "data": "QUJD", "mime_type": "image/png"},
            {"type": "text", "text": "hi"},
        ]
    )
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,QUJD")
    assert parts[1] == {"type": "text", "text": "hi"}

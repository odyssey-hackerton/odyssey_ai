"""② 위험 분석: VLM 추출·조합과 ENISA 심각도 점수."""

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from stages import gemini
from stages.images import png_bytes
from stages.risk import analyze, enrich
from stages.types import Media


@pytest.fixture(autouse=True)
def no_provider_calls(monkeypatch):
    monkeypatch.setattr(gemini, "urlopen", lambda *a, **k: pytest.fail("no external AI call"))


def f(type_, **extra):
    return {"type": type_, "source": "vlm", **extra}


async def test_no_findings_is_safe():
    a = await analyze(Media(text="x"), [])
    assert (a.risky, a.risk, a.findings) == (False, 0.0, [])


async def test_location_clue_alone_is_low():
    # 상호 간판: DPC 2 × EI 0.25 + 0.5 = 1.0 → 낮음
    a = await analyze(Media(), [f("store_name", entity_id="f0")])
    assert not a.risky
    assert (a.findings[0]["severity"], a.findings[0]["risk_level"]) == (1.0, "low")


async def test_face_with_location_clue_is_medium():
    # 얼굴(EI 0.75)이 같이 있으면 간판 SE = 2 × 0.75 + 0.5 = 2.0 → 중간, risky
    a = await analyze(Media(), [f("face", entity_id="f0"), f("store_name", entity_id="f1")])
    assert a.risky
    face, store = a.findings
    assert (store["severity"], store["risk_level"], store["priority"]) == (2.0, "medium", 1)
    assert (face["severity"], face["priority"]) == (1.25, 2)


async def test_rrn_is_very_high_and_order_is_kept():
    a = await analyze(Media(), [f("date", entity_id="f0"), f("rrn", entity_id="f1")])
    assert a.risk == 1.0 and a.risky
    assert [x["type"] for x in a.findings] == ["date", "rrn"]  # 인덱스 유지
    assert a.findings[1]["risk_level"] == "high"


async def test_partial_lowers_identifiability():
    full = await analyze(Media(), [f("face", complete=True)])
    part = await analyze(Media(), [f("face", complete=False)])
    assert (full.findings[0]["ei"], part.findings[0]["ei"]) == (0.75, 0.5)


async def test_inferred_home_uses_source_identifiability():
    findings = [
        f("face", entity_id="f0"),
        f("bus_stop", entity_id="f1"),
        {"type": "inferred_home", "kind": "home", "source": "agent2", "sources": ["f0", "f1"]},
    ]
    a = await analyze(Media(), findings)
    # 사는 곳: DPC 3 × EI 0.75(얼굴) + 0.5 = 2.75
    assert a.findings[2]["severity"] == 2.75 and a.findings[2]["ei"] == 0.75


async def test_enrich_reads_each_crop_then_combines(monkeypatch):
    image = png_bytes(Image.new("RGB", (100, 100), "white"))
    calls = []

    async def vlm(content, schema, instruction, provider=None):
        calls.append(schema["title"])
        if schema["title"] == "CropInfo":
            return {"complete": False, "exposes": [{"kind": "location", "description": "동네"}]}
        return {
            "inferences": [
                {"kind": "home", "title": "사는 동네", "description": "d", "sources": [0, 1]},
                {"kind": "other", "title": "단독", "description": "d", "sources": [0]},
            ]
        }

    monkeypatch.setattr("stages.risk.generate_json", vlm)
    findings = [
        f("store_name", bbox=[0, 0, 50, 50], entity_id="f0"),
        f("bus_stop", bbox=[50, 50, 100, 100], entity_id="f1"),
        {"type": "phone_number", "source": "text", "spans": [[0, 3]], "entity_id": "f2"},
    ]
    out = await enrich(Media(image=image, text="abc"), findings)
    assert calls == ["CropInfo", "CropInfo", "Combination"]
    assert out[0]["complete"] is False and out[0]["exposes"][0]["kind"] == "location"
    assert "exposes" not in out[2]  # 글 항목은 조각이 없다
    assert len(out) == 4  # 조합이 아닌(출처 1개) 추론은 버린다
    assert out[3] | {"sources": ["f0", "f1"], "source": "agent2", "action": "skip"} == out[3]
    assert "complete" not in findings[0]  # 입력은 바꾸지 않는다


async def test_enrich_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        "stages.risk.get_settings",
        lambda: type("S", (), {"risk_use_vlm": False, "risk_vlm_provider": None}),
    )
    monkeypatch.setattr("stages.risk.generate_json", AsyncMock(side_effect=AssertionError))
    findings = [f("face", bbox=[0, 0, 5, 5]), f("name")]
    assert await enrich(Media(image=png_bytes(Image.new("RGB", (9, 9)))), findings) == findings


async def test_invalid_vlm_output_fails_closed(monkeypatch):
    monkeypatch.setattr("stages.risk.generate_json", AsyncMock(return_value={"bad": 1}))
    with pytest.raises(gemini.AIServiceError):
        await enrich(
            Media(image=png_bytes(Image.new("RGB", (20, 20)))), [f("face", bbox=[0, 0, 10, 10])]
        )


async def test_risk_provider_can_differ_from_detection(monkeypatch):
    from stages import qwen, vlm

    monkeypatch.setattr(gemini, "generate_json", AsyncMock(side_effect=AssertionError))
    monkeypatch.setattr(qwen, "generate_json", AsyncMock(return_value={"p": "q"}))
    assert (await vlm.generate_json([], {}, "i", provider="qwen"))["p"] == "q"


def test_small_crop_is_upscaled():
    from stages.risk import MIN_CROP_SIDE, _crop

    crop = _crop(Image.new("RGB", (200, 200)), [10, 10, 60, 46])
    assert min(crop.size) == MIN_CROP_SIDE and crop.width > crop.height

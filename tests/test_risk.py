"""② 위험 분석: VLM 은 스텁으로 바꾸고, 판정 검사와 점수 계산을 확인한다."""

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from stages import risk
from stages.context import RULES, Context
from stages.gemini import AIServiceError
from stages.images import png_bytes
from stages.types import Media
from worker.config import get_settings


@pytest.fixture(autouse=True)
def vlm_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "risk_use_vlm", True)


def finding(i, type_="store_name", category="location", **extra):
    return {
        "entity_id": f"f{i}",
        "type": type_,
        "category": category,
        "title": "간판",
        "source": "vlm",
        "bbox": [10, 10, 50, 40],
        "action": "mask",
        **extra,
    }


def judged(i, s, lk, issue=""):
    return {"entity_id": f"f{i}", "reason": "근거", "severity": s, "likelihood": lk, "issue": issue}


MEDIA = Media(image=png_bytes(Image.new("RGB", (100, 80), "white")), image_ext="png")


async def test_judge_covers_every_object_and_keeps_valid_combos(monkeypatch):
    provider = AsyncMock(
        side_effect=[
            {"objects": [judged(0, 2, 3)]},  # f1 누락 → 다시 묻는다
            {"objects": [judged(1, 1, 4), judged(9, 4, 4)]},  # 없는 id 는 무시
            {
                "combos": [
                    {
                        "title": "위치 추정: 거리(1km)",
                        "reason": "간판 두 개로 거리가 좁혀짐",
                        "sources": ["f0", "f1", "f7"],
                        "severity": 3,
                        "likelihood": 3,
                    },
                    {
                        "title": "x",
                        "reason": "y",
                        "sources": ["f0"],
                        "severity": 4,
                        "likelihood": 4,
                    },
                ]
            },
        ]
    )
    monkeypatch.setattr(risk.vlm, "generate_json", provider)
    findings, combos = await risk.judge(MEDIA, [finding(0), finding(1)])

    assert [(f["severity"], f["likelihood"]) for f in findings] == [(2, 3), (1, 4)]
    assert all(f["reason"] for f in findings)
    assert combos == [
        {
            "title": "위치 추정: 거리(1km)",
            "reason": "간판 두 개로 거리가 좁혀짐",
            "sources": ["f0", "f1"],
            "severity": 3,
            "likelihood": 3,
        }
    ]
    # 재질문에는 빠진 객체만, 좌표는 0~1000 box_2d 로 보낸다
    retry_text = provider.call_args_list[1].args[0][-1]["text"]
    assert "id=f1" in retry_text and "id=f0" not in retry_text
    assert "box_2d=[125, 100, 500, 500]" in retry_text


async def test_judge_fails_when_object_still_missing(monkeypatch):
    provider = AsyncMock(return_value={"objects": [judged(0, 2, 2)]})
    monkeypatch.setattr(risk.vlm, "generate_json", provider)
    with pytest.raises(AIServiceError):
        await risk.judge(MEDIA, [finding(0), finding(1)])


async def test_judge_rejects_out_of_scale(monkeypatch):
    provider = AsyncMock(return_value={"objects": [judged(0, 5, 2)]})
    monkeypatch.setattr(risk.vlm, "generate_json", provider)
    with pytest.raises(AIServiceError):
        await risk.judge(MEDIA, [finding(0)])


async def test_judge_off_returns_input(monkeypatch):
    monkeypatch.setattr(get_settings(), "risk_use_vlm", False)
    items = [finding(0)]
    assert await risk.judge(MEDIA, items) == (items, [])


@pytest.mark.parametrize(
    ("s", "lk", "expected"),
    [(1, 3, (3, "low")), (2, 2, (4, "medium")), (4, 2, (8, "medium")), (3, 3, (9, "high"))],
)
def test_score_bands(s, lk, expected):
    assert risk.score({"severity": s, "likelihood": lk}) == expected


def test_score_hot_issue_and_rule_floor():
    assert risk.score({"severity": 2, "likelihood": 3, "issue": "선거"}, {"선거"}) == (9, "high")
    assert risk.score({"severity": 4, "likelihood": 3, "issue": "선거"}, {"선거"}) == (12, "high")
    assert risk.score({"severity": 1, "likelihood": 1, "type": "rrn"}) == (12, "high")
    ballot = next(r["item"] for r in RULES if r["category"] == "sensitive")
    assert risk.score({"severity": 1, "likelihood": 2, "watch_item": ballot}) == (12, "high")


async def test_analyze_objects_and_combos_separately():
    findings = [finding(0, severity=1, likelihood=2), finding(1, severity=1, likelihood=3)]
    combos = [
        {"title": "위치", "reason": "r", "sources": ["f0", "f1"], "severity": 3, "likelihood": 4}
    ]
    context = Context(country="KR", day="2026-11-01")
    result = await risk.analyze(Media(), findings, combos, context)

    assert [f["risk_level"] for f in result.findings] == ["low", "low"]  # 조합이 객체를 안 바꿈
    assert result.combos[0]["risk_level"] == "high"
    assert result.risky and result.risk == 0.75


async def test_analyze_stale_issue_does_not_raise_severity():
    context = Context(
        country="KR",
        day="2026-11-01",
        issues=[{"title": "선거", "kind": "election", "category": "sensitive", "weight": 0.3}],
    )
    result = await risk.analyze(
        Media(), [finding(0, severity=2, likelihood=3, issue="선거")], [], context
    )
    assert result.findings[0]["risk_score"] == 6


async def test_analyze_without_judgement_uses_detect_level():
    result = await risk.analyze(Media(), [finding(0, risk_level="high")])
    assert result.risky and result.risk == 0.9 and result.combos == []
    assert (await risk.analyze(Media(), [])).risky is False

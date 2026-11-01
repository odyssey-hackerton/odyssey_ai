"""④ 재검증: ② 코드 재사용, 처리 이력 전달, 처리 전후 객체 상태."""

from unittest.mock import AsyncMock

from stages import risk, verify
from stages.types import Assessment, Media


def obj(i, type_="store_name", bbox=(10, 10, 50, 50), action="mask", **extra):
    return {"entity_id": f"f{i}", "type": type_, "bbox": list(bbox), "action": action, **extra}


def test_compare_statuses():
    before = [
        obj(0, "face", action="blur", title="얼굴"),  # 다시 나오고 medium → remaining
        obj(1, bbox=(100, 100, 150, 150)),  # 사라짐 → removed
        obj(2, "face", (200, 200, 260, 260), action="skip"),  # 사용자가 둠 → kept
        obj(3, "bus_stop", (300, 10, 340, 50)),  # 다시 나왔지만 low → weakened
        {"entity_id": "f4", "type": "metadata", "source": "exif", "action": "remove"},
    ]
    after = [
        obj(9, "face", (12, 12, 52, 52), risk_level="medium", reason="윤곽이 보임"),
        obj(8, "face", (201, 201, 261, 261), risk_level="high"),
        obj(7, "bus_stop", (301, 11, 341, 51), risk_level="low"),
        obj(6, "other", (400, 400, 440, 440), risk_level="medium", reason="야간 외식"),
        obj(5, "other", (500, 500, 540, 540), risk_level="low"),  # 새로 찾았지만 low → 빼기
    ]
    result = verify.compare(before, after)
    status = {i["entity_id"]: i["status"] for i in result["items"]}
    assert status == {
        "f0": "remaining",
        "f1": "removed",
        "f2": "kept",
        "f3": "weakened",
        "f4": "removed",
    }
    f0 = result["items"][0]
    assert f0["recommend"] == "mask" and f0["after"]["reason"] == "윤곽이 보임"
    assert f0["message"] == (
        "얼굴: 블러 처리했지만 아직 알아볼 수 있어요. "
        "윤곽이 보임. 덮기로 다시 처리하는 걸 추천해요."
    )
    assert "더 이상 보이지 않아요" in result["items"][1]["message"]
    assert result["items"][2]["message"].startswith("face: 그대로 두기로 한 정보예요.")
    assert result["new_items"][0]["message"] == "처음엔 못 찾은 정보예요. other: 야간 외식."
    assert [n["entity_id"] for n in result["new_items"]] == ["f6"]
    assert (result["remaining"], result["new"], result["clean"]) == (1, 1, False)


def test_compare_needs_same_type_and_overlap():
    before = [obj(0)]
    assert verify.compare(before, [obj(1, "bus_stop")])["items"][0]["status"] == "removed"
    assert (
        verify.compare(before, [obj(1, bbox=(40, 40, 90, 90))])["items"][0]["status"] == "removed"
    )


async def test_verify_reuses_risk_assess_with_history(monkeypatch):
    calls = []

    async def fake_assess(media, context, processed=None):
        calls.append(processed)
        return Assessment(risky=False, risk=0.1, findings=[])

    monkeypatch.setattr(risk, "assess", fake_assess)
    before = Assessment(True, 0.9, [obj(0)])
    after, result = await verify.verify(Media(text="x"), None, before)
    assert calls == [before.findings] and after.risk == 0.1
    assert result["clean"] and result["removed"] == 1


async def test_edited_area_is_told_to_judge(monkeypatch):
    monkeypatch.setattr(risk, "detect", AsyncMock(return_value=[obj(0, "face", (12, 12, 52, 52))]))
    provider = AsyncMock(
        return_value={
            "objects": [
                {"entity_id": "f0", "reason": "윤곽", "severity": 2, "likelihood": 2, "issue": ""}
            ]
        }
    )
    monkeypatch.setattr(risk.vlm, "generate_json", provider)
    monkeypatch.setattr(risk.get_settings(), "risk_use_vlm", True)
    findings, _ = await risk.find_and_judge(
        Media(text="x"), None, processed=[obj(0, "face", action="blur")]
    )
    assert findings[0]["edited_as"] == "blur"
    assert "이미 블러 처리됐다" in provider.call_args_list[0].args[0][-1]["text"]


def test_risk_total_counts_new_items_as_already_there():
    # 실제 서버 실행 예: 처음 4개 (4, 8, 6, 4) 모두 사라지고, 새로 2개 (4, 8)
    before = [
        obj(0, "bus_stop", (0, 0, 10, 10), risk_score=4),
        obj(1, "other", (20, 0, 30, 10), risk_score=8),
        obj(2, "other", (40, 0, 50, 10), risk_score=6),
        obj(3, "store_name", (60, 0, 70, 10), risk_score=4),
    ]
    after = [
        obj(9, "bus_stop", (0, 100, 50, 150), risk_level="medium", risk_score=4),
        obj(8, "store_name", (200, 100, 250, 150), risk_level="medium", risk_score=8),
    ]
    assert verify.compare(before, after)["risk_total"] == {
        "before": 34,
        "after": 12,
        "reduction": 0.647,
    }


def test_risk_total_uses_after_score_for_matched_and_handles_empty():
    before = [obj(0, "face", risk_score=12), obj(1, bbox=(100, 100, 150, 150), risk_score=4)]
    after = [obj(9, "face", (11, 11, 51, 51), risk_level="low", risk_score=2)]
    assert verify.compare(before, after)["risk_total"] == {
        "before": 16,
        "after": 2,
        "reduction": 0.875,
    }
    assert verify.compare([], [])["risk_total"] == {"before": 0, "after": 0, "reduction": None}

"""④ 재검증: ② 코드 재사용과 처리 전후 객체 비교."""

from stages import risk, verify
from stages.types import Assessment, Media


def obj(i, type_="store_name", bbox=(10, 10, 50, 50), action="mask"):
    return {"entity_id": f"f{i}", "type": type_, "bbox": list(bbox), "action": action}


def test_compare_counts_remaining_removed_new_and_skipped():
    before = [
        obj(0),  # 처리했는데 같은 자리에 다시 탐지 → remaining
        obj(1, bbox=(100, 100, 150, 150)),  # 사라짐 → removed
        obj(2, "face", (200, 200, 260, 260), action="skip"),  # 사용자가 남겨둠
        {"entity_id": "f3", "type": "metadata", "source": "exif", "action": "remove"},
    ]
    after = [
        obj(9, bbox=(12, 12, 52, 52)),  # f0 과 겹침 (IoU ≥ 0.5)
        obj(8, "face", (201, 201, 261, 261)),  # skip 한 얼굴
        obj(7, bbox=(300, 300, 340, 340)),  # 새로 생김 (재생성된 가짜 간판 등)
    ]
    result = verify.compare(before, after)
    assert result["remaining"] == ["f0"]
    assert result["removed"] == 2  # f1, f3(EXIF)
    assert result["kept_by_user"] == ["f2"]
    assert result["new"] == 1
    assert (result["before"], result["after"], result["reduced"]) == (4, 3, True)


def test_compare_needs_same_type_and_overlap():
    before = [obj(0)]
    assert verify.compare(before, [obj(1, "bus_stop")])["remaining"] == []
    assert verify.compare(before, [obj(1, bbox=(40, 40, 90, 90))])["remaining"] == []


async def test_verify_reuses_risk_assess(monkeypatch):
    calls = []

    async def fake_assess(media, context):
        calls.append(media)
        return Assessment(risky=False, risk=0.1, findings=[])

    monkeypatch.setattr(risk, "assess", fake_assess)
    media = Media(text="x")
    after, result = await verify.verify(media, None, Assessment(True, 0.9, [obj(0)]))
    assert calls == [media] and after.risk == 0.1
    assert result["removed"] == 1 and result["remaining"] == []

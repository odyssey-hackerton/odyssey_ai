"""④ 재검증: ② 코드를 그대로 다시 돌리고, 처리 전 객체와 짝지어 상태를 붙인다.

- 점수: `risk.assess` 를 그대로 부른다 (같은 모델·프롬프트·S×L).
  ② 를 고치면 ④ 도 같이 바뀐다. 이 파일에는 ② 판정·점수 로직을 두지 않는다.
  이미 편집된 자리는 ② 판정에 "처리 이력"으로 알려준다 (risk.find_and_judge 의 processed).
- 짝짓기: 처리 전 객체와 재검증 객체가 같은 type 이고 박스가 risk.IOU_MATCH 이상 겹치면 같은 객체.
  박스 없는 항목(캡션·EXIF)은 type 만 본다. 모델 호출 없음.
- 원본 사진은 이미 지워졌으므로 사진끼리가 아니라 처리 전 판정 ↔ 재검증 판정을 비교한다.

상태 (items[].status)
- removed   : 처리했고 재검증에서 같은 자리에 안 나옴
- weakened  : 같은 자리에서 다시 나왔지만 ② 등급이 low
- remaining : 같은 자리에서 다시 나왔고 ② 등급이 medium 이상 → 경고
- kept      : 사용자가 skip (그대로 둠)
new_items: 처리 전 객체와 짝이 없고 medium 이상인 재검증 객체 (처음엔 못 찾은 단서)
"""

from typing import Any

from stages import risk
from stages.context import Context
from stages.types import Assessment, Media

Findings = list[dict[str, Any]]

# 아직 남았을 때 추천하는 더 센 처리
STRONGER = {"blur": "mask", "mask": "remove"}


def _same(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("type") != after.get("type"):
        return False
    if before.get("bbox") and after.get("bbox"):
        return risk.iou(before["bbox"], after["bbox"]) >= risk.IOU_MATCH
    return not before.get("bbox") and not after.get("bbox")


def _base(f: dict[str, Any]) -> dict[str, Any]:
    keys = ("entity_id", "type", "category", "title", "source", "bbox", "spans")
    return {k: f[k] for k in keys if f.get(k) is not None}


def _after_info(a: dict[str, Any]) -> dict[str, Any]:
    keys = ("entity_id", "reason", "risk_level", "risk_score")
    return {k: a[k] for k in keys if a.get(k) is not None}


def compare(before: Findings, after: Findings) -> dict[str, Any]:
    """처리 전후 객체 비교. 백엔드에는 details.verification 으로 간다."""
    matched: set[int] = set()
    items = []
    for f in before:
        action = f.get("action", "mask")
        hit = next((i for i, a in enumerate(after) if i not in matched and _same(f, a)), None)
        item = {**_base(f), "action": action}
        if hit is not None:
            matched.add(hit)
            item["after"] = _after_info(after[hit])
        if action == "skip":
            item["status"] = "kept"
        elif hit is None:
            item["status"] = "removed"
        elif after[hit].get("risk_level", "medium") == "low":
            item["status"] = "weakened"
        else:
            item["status"] = "remaining"
            if action in STRONGER:
                item["recommend"] = STRONGER[action]
        items.append(item)

    new_items = [
        {**_base(a), **_after_info(a)}
        for i, a in enumerate(after)
        if i not in matched and a.get("risk_level", "medium") != "low"
    ]
    count = {
        s: sum(i["status"] == s for i in items)
        for s in ("removed", "weakened", "remaining", "kept")
    }
    return {
        "before": len(before),
        "after": len(after),
        **count,
        "new": len(new_items),
        "reduced": len(after) < len(before),
        "clean": count["remaining"] == 0 and not new_items,  # 경고할 것이 없음
        "items": items,
        "new_items": new_items,
    }


async def verify(
    media: Media, context: Context | None, before: Assessment
) -> tuple[Assessment, dict[str, Any]]:
    """④ = ② 그대로 (risk.assess, 처리 이력 전달) + 처리 전후 비교."""
    after = await risk.assess(media, context, processed=before.findings)
    return after, compare(before.findings, after.findings)

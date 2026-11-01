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

risk_total: 위험 총량 (객체별 risk_score 합) 처리 전/후 비교
- before: 처리 전 객체 점수 합 + 재검증에서 새로 찾은 객체 점수 합
  (새로 찾은 객체도 원본 사진에 원래 있던 것이라 처리 전 위험에 넣는다. 점수는 재검증 판정 값)
- after : 재검증에서 찾은 객체 전부의 점수 합
- reduction: (before - after) / before. 짝지은 객체 점수가 처리 후 오르지 않는 한 0 이상

message: 화면에 그대로 띄울 한 문장. 모델 호출 없이 상태·처리·② reason 으로 만든다.
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


ACTION_LABELS = {**risk.EDIT_LABELS, "skip": "그대로 두기"}


def _name(f: dict[str, Any]) -> str:
    return f.get("title") or f.get("type") or "항목"


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text.endswith((".", "!", "?")) else text + "."


def _message(item: dict[str, Any]) -> str:
    name, done = _name(item), ACTION_LABELS.get(item["action"], item["action"])
    reason = _sentence(item.get("after", {}).get("reason", ""))
    if item["status"] == "removed":
        return f"{name}: {done} 처리했고, 다시 확인했을 때 더 이상 보이지 않아요."
    if item["status"] == "weakened":
        return f"{name}: {done} 처리했어요. 흔적은 남았지만 알아낼 수 있는 정보는 거의 없어요."
    if item["status"] == "remaining":
        more = ACTION_LABELS.get(item.get("recommend", ""), "")
        tail = f" {more}로 다시 처리하는 걸 추천해요." if more else ""
        return f"{name}: {done} 처리했지만 아직 알아볼 수 있어요. {reason}{tail}".strip()
    return f"{name}: 그대로 두기로 한 정보예요. {reason}".strip()


def _new_message(a: dict[str, Any]) -> str:
    return f"처음엔 못 찾은 정보예요. {_name(a)}: {_sentence(a.get('reason', ''))}".strip()


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
        item["message"] = _message(item)
        items.append(item)

    new_items = [
        {**_base(a), **_after_info(a), "message": _new_message(a)}
        for i, a in enumerate(after)
        if i not in matched and a.get("risk_level", "medium") != "low"
    ]
    count = {
        s: sum(i["status"] == s for i in items)
        for s in ("removed", "weakened", "remaining", "kept")
    }
    return {
        "risk_total": _risk_total(before, after, matched),
        "before": len(before),
        "after": len(after),
        **count,
        "new": len(new_items),
        "reduced": len(after) < len(before),
        "clean": count["remaining"] == 0 and not new_items,  # 경고할 것이 없음
        "items": items,
        "new_items": new_items,
    }


def _score(f: dict[str, Any]) -> int:
    return int(f.get("risk_score") or 0)


def _risk_total(before: Findings, after: Findings, matched: set[int]) -> dict[str, Any]:
    unmatched = sum(_score(a) for i, a in enumerate(after) if i not in matched)
    total_before = sum(_score(f) for f in before) + unmatched
    total_after = sum(_score(a) for a in after)
    reduction = round((total_before - total_after) / total_before, 3) if total_before else None
    return {"before": total_before, "after": total_after, "reduction": reduction}


async def verify(
    media: Media, context: Context | None, before: Assessment
) -> tuple[Assessment, dict[str, Any]]:
    """④ = ② 그대로 (risk.assess, 처리 이력 전달) + 처리 전후 비교."""
    after = await risk.assess(media, context, processed=before.findings)
    return after, compare(before.findings, after.findings)

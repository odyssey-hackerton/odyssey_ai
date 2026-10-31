"""④ 재검증: ② 코드를 그대로 다시 돌리고, 처리 전후 객체를 비교한다.

- 점수: `risk.assess` 를 그대로 부른다 (같은 모델·프롬프트·S×L).
  ② 를 고치면 ④ 도 같이 바뀐다. 이 파일에는 ② 판정·점수 로직을 두지 않는다.
- 객체 감소: 처리 전 객체가 처리 후에도 같은 자리에 다시 탐지되는지 본다.
  같은 type 이고 박스가 IOU_MATCH 이상 겹치면 "그대로 남음".
  박스 없는 항목(캡션·EXIF)은 type 만 본다.
  사용자가 skip 한 객체는 남아 있어도 정상이라 따로 센다.
"""

from typing import Any

from stages import risk
from stages.context import Context
from stages.types import Assessment, Media

# 같은 객체로 볼 박스 겹침. PASCAL VOC·COCO 의 기본 매칭 기준 0.5.
IOU_MATCH = 0.5

Findings = list[dict[str, Any]]


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("type") != after.get("type"):
        return False
    if before.get("bbox") and after.get("bbox"):
        return _iou(before["bbox"], after["bbox"]) >= IOU_MATCH
    return not before.get("bbox") and not after.get("bbox")


def compare(before: Findings, after: Findings) -> dict[str, Any]:
    """처리 전후 객체 비교. 백엔드에는 details.verification 으로 간다."""
    targeted = [f for f in before if f.get("action") != "skip"]
    kept = [f for f in before if f.get("action") == "skip"]
    matched: set[int] = set()
    remaining = []
    for f in targeted:
        hit = next((i for i, a in enumerate(after) if i not in matched and _same(f, a)), None)
        if hit is not None:
            matched.add(hit)
            remaining.append(f.get("entity_id"))
    for f in kept:  # skip 한 객체와 짝지어진 것은 "새로 생김"에서 뺀다
        hit = next((i for i, a in enumerate(after) if i not in matched and _same(f, a)), None)
        if hit is not None:
            matched.add(hit)
    return {
        "before": len(before),
        "after": len(after),
        "targeted": len(targeted),
        "removed": len(targeted) - len(remaining),
        "remaining": remaining,  # 처리했는데도 같은 자리에 다시 탐지된 원래 객체
        "kept_by_user": [f.get("entity_id") for f in kept],
        "new": len(after) - len(matched),  # 처리 후 새로 탐지된 객체 (재생성된 가짜 단서 등)
        "reduced": len(after) < len(before),
    }


async def verify(
    media: Media, context: Context | None, before: Assessment
) -> tuple[Assessment, dict[str, Any]]:
    """④ = ② 그대로 (risk.assess) + 처리 전후 비교."""
    after = await risk.assess(media, context)
    return after, compare(before.findings, after.findings)

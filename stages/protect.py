"""③ 에이전트3 정책 (Adaptive Protection Policy).

①·② 가 만든 findings 에 보호 정책을 얹는다. 새로 탐지하거나 점수를 다시 계산하지 않는다.

findings 항목에 추가하는 필드 (프론트 flow.ts 가 읽는 이름 그대로)
- category: 편집 도구 분류 "text" | "object" | "location". ② 의 8분류는 risk_category 로 옮긴다.
- action:   추천 보호 방법. 프론트 ALLOWED_ACTIONS 와 같은 집합에서 고른다.
- policy:   서버가 사용자 편집을 검사할 때 쓰는 규칙. 백엔드(app/services/policy.py)도 같은 규칙으로
            /transform 요청을 422 로 거절하고, 워커는 Phase 2 에서 한 번 더 검사한다.
    level:           "locked" | "high" | "medium" | "low"
    allowed_actions: 이 영역에 실제로 적용할 수 있는 action
    anchor_bbox:     최초 탐지 좌표. locked/high 는 이 안쪽으로 줄일 수 없다
    allow_shrink / allow_delete / allow_partial: 축소·× 삭제(skip)·부분마스킹 허용 여부
    regenerate_candidate: 재생성 후보(G_i) 여부. 후보만 allowed_actions 에 regenerate 가 있다

재생성 후보 G_i = 1[ t_i ∈ T_gen ∧ L_i ≥ medium ∧ L_i ≠ locked ∧ Q_i ]
- T_gen: 간판·상호(전체 허용)와 학교·아파트·정류장·위치 단서(조건부). 얼굴·번호판·PII 는 제외
- Q_i:   영역 품질. bbox 가 사진 안에 있고, 짧은 변 ≥ REGEN_MIN_SIDE_PX,
         면적 ≤ 사진의 REGEN_MAX_AREA_RATIO (조건부 type 은 REGEN_MAX_AREA_RATIO_CONDITIONAL)
후보여도 추천(action)은 그대로 가림이다. 기준값은 초기 설계값이라 평가하면서 조정한다.

프론트는 도구 목록·크기 조절 규칙을 코드에 고정하고 있으므로(flow.ts ALLOWED_ACTIONS, 12px 축소,
× 삭제) 정책 위반은 UI 에서 막히지 않는다. 그래서 서버가 거절(422 + 메시지)한다.
"""

from typing import Any

from stages.catalog import TYPE_CATEGORY
from stages.risk import RULE_FLOOR_TYPES, RULE_FLOOR_WATCH

POLICY_VERSION = "agent3.policy.v1"

# 프론트 flow.ts ALLOWED_ACTIONS + 이미지재생성(사진 아래 패널, 모든 유형에서 사용 가능)
# regenerate 는 재생성 후보(G_i)에만 policy_for 가 붙인다.
UI_ACTIONS: dict[str, list[str]] = {
    "text": ["mask", "blur", "partialMask", "remove"],
    "object": ["mask", "blur", "mosaic"],
    "location": ["mask"],
}
IMAGE_ACTIONS = {"mask", "blur", "mosaic", "partialMask", "remove", "regenerate", "skip"}
TEXT_ACTIONS = {"mask", "remove", "skip"}

# 글자로 읽히는 개인정보. OCR 이 아니어도(VLM) 텍스트 도구(블러·부분마스킹·텍스트제거)가 맞다.
TEXT_TYPES = {
    "name",
    "phone_number",
    "email",
    "address",
    "apt_dong_ho",
    "zipcode",
    "rrn",
    "account_number",
    "card_number",
    "business_id",
    "tracking_number",
    "user_id",
    "url",
    "date",
    "absence_period",
}
# 간판·표지는 글자가 있어도 "간접 위치 단서"로 둔다 (가림 + 이미지재생성 패널).
# 텍스트 도구를 주고 싶으면 이 집합을 TEXT_TYPES 로 옮긴다.
SIGN_TYPES = {
    "store_name",
    "building_sign",
    "pharmacy_name",
    "school_sign",
    "apt_sign",
    "bus_stop",
}
OBJECT_TYPES = {"face", "license_plate", "delivery_label", "id_card"}
# 부분 노출만으로도 식별되는 것: 전체 불투명 가림만 허용
LOCKED_TYPES = RULE_FLOOR_TYPES | {"account_number", "card_number"}
# 자동 추천에서 블러·모자이크를 쓰지 않는 고위험 유형 (사용자가 고르는 것은 막지 않는다)
ALWAYS_MASK_TYPES = LOCKED_TYPES | {
    "face",
    "license_plate",
    "phone_number",
    "address",
    "tracking_number",
}


# ---------------------------------------------------------------- 재생성 후보 (G_i)
# 초기 설계값. 평가(재생성 품질·④ 재검증)를 보면서 조정한다.
REGEN_MIN_LEVEL = "medium"  # L_i ≥ medium (locked 는 제외)
REGEN_MIN_SIDE_PX = 16  # bbox 짧은 변 최소 길이(px)
REGEN_MAX_AREA_RATIO = 0.40  # bbox 면적 / 사진 면적 상한
REGEN_MAX_AREA_RATIO_CONDITIONAL = 0.25  # 조건부 type 의 상한

# 간판(vlm, 간판 전체 box)·상호(ocr, 글자 box). 같은 type 이 source 로 구분된다.
REGEN_TYPES = {"store_name", "building_sign", "pharmacy_name"}
REGEN_SOURCES = {"vlm", "ocr"}
# 조건부: 학교·아파트 표지, 정류장, 위치 단서(other + risk_category=location)
REGEN_CONDITIONAL_TYPES = {"school_sign", "apt_sign", "bus_stop"}
REGEN_CONDITIONAL_LOCATION_TYPES = {"other"}
_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


def regen_class(finding: dict[str, Any]) -> str | None:
    """t_i ∈ T_gen 이면 "full" 또는 "conditional", 아니면 None."""
    type_ = finding.get("type", "")
    if type_ in REGEN_TYPES and finding.get("source") in REGEN_SOURCES:
        return "full"
    if type_ in REGEN_CONDITIONAL_TYPES:
        return "conditional"
    risk_category = finding.get("risk_category") or finding.get("category")
    if type_ in REGEN_CONDITIONAL_LOCATION_TYPES and risk_category == "location":
        return "conditional"
    return None


def region_ok(bbox: Any, image_size: tuple[int, int] | None, max_area_ratio: float) -> bool:
    """Q_i: 사진 안의 bbox, 짧은 변 ≥ REGEN_MIN_SIDE_PX, 면적 ≤ max_area_ratio."""
    box = _box(bbox)
    if box is None or image_size is None:
        return False
    w, h = image_size
    if w <= 0 or h <= 0:
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return False
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < REGEN_MIN_SIDE_PX:
        return False
    return bw * bh <= max_area_ratio * w * h


def is_regen_candidate(
    finding: dict[str, Any], level: str, image_size: tuple[int, int] | None
) -> bool:
    """G_i = 1[ t_i ∈ T_gen ∧ L_i ≥ medium ∧ L_i ≠ locked ∧ Q_i ]. 사진 크기 모르면 0."""
    kind = regen_class(finding)
    if kind is None or level == "locked":
        return False
    if _LEVEL_RANK.get(level, -1) < _LEVEL_RANK[REGEN_MIN_LEVEL]:
        return False
    ratio = REGEN_MAX_AREA_RATIO if kind == "full" else REGEN_MAX_AREA_RATIO_CONDITIONAL
    return region_ok(finding.get("bbox"), image_size, ratio)


def ui_category(finding: dict[str, Any]) -> str:
    source = finding.get("source")
    type_ = finding.get("type", "")
    if source in {"text", "ocr"}:
        return "text"
    if type_ in TEXT_TYPES:
        return "text"
    if type_ in OBJECT_TYPES:
        return "object"
    if type_ in SIGN_TYPES:
        return "location"
    risk_category = finding.get("risk_category") or finding.get("category")
    if type_ in {"other", "context_item"} and risk_category == "location":
        return "location"
    return "object"


def protection_level(finding: dict[str, Any]) -> str:
    if finding.get("type") in LOCKED_TYPES or finding.get("watch_item") in RULE_FLOOR_WATCH:
        return "locked"
    level = finding.get("risk_level", "medium")
    return level if level in {"high", "low"} else "medium"


def recommend(finding: dict[str, Any], category: str, level: str) -> str:
    """보안 최소 조건을 먼저 만족하고, 그 안에서 덜 거슬리는 방법을 고른다."""
    if category == "location" or level in {"locked", "high"}:
        return "mask"
    if finding.get("type") in SIGN_TYPES:  # 간판·표지는 OCR(글자 box)로 잡혀도 가림
        return "mask"
    if finding.get("type") in ALWAYS_MASK_TYPES:
        return "mask"
    if category == "object":
        return "mosaic" if level == "low" else "blur"
    return "blur"  # text, medium/low


def policy_for(
    finding: dict[str, Any],
    category: str,
    level: str,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    strict = level in {"locked", "high"}
    allowed = ["mask"] if level == "locked" else list(UI_ACTIONS[category])
    if strict and "partialMask" in allowed:
        allowed.remove("partialMask")
    candidate = is_regen_candidate(finding, level, image_size)
    if candidate:
        allowed.append("regenerate")
    return {
        "version": POLICY_VERSION,
        "level": level,
        "allowed_actions": allowed,
        "anchor_bbox": list(finding["bbox"]) if finding.get("bbox") else None,
        "allow_shrink": not strict,
        "allow_delete": not strict,
        "allow_partial": not strict,
        "regenerate_candidate": candidate,
    }


def apply_policy(
    findings: list[dict[str, Any]], image_size: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    """리포트로 나가는 findings 에 category(UI)·action·policy 를 붙인다. 입력은 바꾸지 않는다.

    image_size(가로, 세로 px)는 재생성 후보의 영역 품질(Q_i) 검사에 쓴다. 없으면 재생성 후보가 없다.
    """
    out = []
    for f in findings:
        f = dict(f)
        if f.get("source") == "exif":
            f.setdefault("policy", {"version": POLICY_VERSION, "level": "high"})
            f["action"] = "remove"
            out.append(f)
            continue
        if "risk_category" not in f:
            f["risk_category"] = f.get("category") or TYPE_CATEGORY.get(f.get("type", ""))
        category = ui_category(f)
        level = protection_level(f)
        f["category"] = category
        f["action"] = recommend(f, category, level) if f.get("source") != "text" else "mask"
        f["policy"] = policy_for(f, category, level, image_size)
        out.append(f)
    return out


# ---------------------------------------------------------------- 사용자 편집 검사
class PolicyViolation(ValueError):
    """사용자가 보낸 편집이 보호 최소 정책보다 약하다. 메시지는 그대로 사용자에게 보여도 된다."""


def _box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in box):
        return None
    return box


def check_choice(
    finding: dict[str, Any],
    action: str,
    bbox: Any = None,
    mask_box: Any = None,
    generation_id: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """한 항목의 사용자 선택이 정책 안에 있는지 검사한다. 위반이면 PolicyViolation."""
    title = finding.get("title") or finding.get("type", "")
    policy = finding.get("policy") or {}
    is_text = finding.get("source") == "text"
    is_file = finding.get("source") == "exif"

    if is_text or is_file:
        if action not in TEXT_ACTIONS:
            raise PolicyViolation(f"'{title}'에는 {action} 처리를 쓸 수 없어요.")
        return
    if action not in IMAGE_ACTIONS:
        raise PolicyViolation(f"'{title}'에는 {action} 처리를 쓸 수 없어요.")

    level = policy.get("level", "medium")
    allowed = set(policy.get("allowed_actions") or UI_ACTIONS.get(finding.get("category", ""), []))
    if action == "skip":
        if not policy.get("allow_delete", True):
            raise PolicyViolation(
                f"'{title}'은(는) 위험도가 높아 그대로 둘 수 없어요. 가림을 유지해 주세요."
            )
    elif allowed and action not in allowed:
        if level == "locked":
            raise PolicyViolation(f"'{title}'은(는) 전체 가림만 할 수 있어요.")
        if action == "partialMask" and not policy.get("allow_partial", True):
            raise PolicyViolation(f"'{title}'은(는) 위험도가 높아 부분마스킹을 쓸 수 없어요.")
        raise PolicyViolation(f"'{title}'에는 {action} 처리를 쓸 수 없어요.")

    if action == "partialMask":
        part = _box(mask_box)
        if part is None or not (0 <= part[0] < part[2] <= 1 and 0 <= part[1] < part[3] <= 1):
            raise PolicyViolation(f"'{title}'의 부분마스킹 범위를 사진에서 지정해 주세요.")
    if action == "regenerate" and not generation_id:
        raise PolicyViolation(
            f"'{title}'의 이미지를 먼저 생성하거나 다른 편집 방식을 선택해 주세요."
        )

    if bbox is None or action == "skip":
        return
    box = _box(bbox)
    if box is None or not (box[0] < box[2] and box[1] < box[3]):
        raise PolicyViolation(f"'{title}'의 영역 좌표가 올바르지 않아요.")
    if image_size is not None:
        w, h = image_size
        if not (0 <= box[0] and 0 <= box[1] and box[2] <= w and box[3] <= h):
            raise PolicyViolation(f"'{title}'의 영역이 사진 범위를 벗어났어요.")
    anchor = _box(policy.get("anchor_bbox")) or _box(finding.get("bbox"))
    if anchor and not policy.get("allow_shrink", True):
        tolerance = 0.5  # 화면 좌표 반올림
        if (
            box[0] > anchor[0] + tolerance
            or box[1] > anchor[1] + tolerance
            or box[2] < anchor[2] - tolerance
            or box[3] < anchor[3] - tolerance
        ):
            raise PolicyViolation(
                f"'{title}'은(는) 위험도가 높아 처음 찾은 영역보다 작게 줄일 수 없어요."
            )

"""위험도 8개 카테고리. ⓪ 맥락, ① 탐지, ② 위험 분석이 같은 값을 쓴다.

위험도 = 모르는 사람이 사진(+글)에서 알아낼 수 있는 모든 것.
"""

from typing import Literal

Category = Literal[
    "location",  # 위치 (정밀도 포함)
    "time_movement",  # 시간·동선
    "identity",  # 신원
    "contact_finance",  # 연락·금융
    "attributes",  # 개인 속성 (나이, 성별, 직업 등)
    "sensitive",  # 민감 정보 (건강, 정치, 종교 등)
    "relationships",  # 관계 (가족, 아이, 동료)
    "lifestyle",  # 생활·소유물
]

CATEGORIES: tuple[str, ...] = (
    "location",
    "time_movement",
    "identity",
    "contact_finance",
    "attributes",
    "sensitive",
    "relationships",
    "lifestyle",
)

LABELS: dict[str, str] = {
    "location": "위치",
    "time_movement": "시간·동선",
    "identity": "신원",
    "contact_finance": "연락·금융",
    "attributes": "개인 속성",
    "sensitive": "민감 정보",
    "relationships": "관계",
    "lifestyle": "생활·소유물",
}

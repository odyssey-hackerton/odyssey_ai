from dataclasses import dataclass, field
from typing import Any


@dataclass
class Media:
    """처리할 입력. 입력 종류에 따라 image, text 중 하나 또는 둘 다 있다."""

    image: bytes | None = None
    image_ext: str | None = None  # jpg, png, webp
    text: str | None = None


@dataclass
class Assessment:
    """② 위험 분석 결과."""

    risky: bool  # 보호 처리가 필요한지
    risk: float  # 위험도 0~1
    # 탐지 항목. 키 예시: type, source, bbox, score, action (contract.Finding 참고)
    findings: list[dict[str, Any]] = field(default_factory=list)

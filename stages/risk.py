"""② 위험 분석·재검증: 탐지 항목을 보고 위험도를 매긴다. 변환 후 재검증에도 같은 함수를 쓴다."""

from typing import Any

from stages.types import Assessment, Media


async def analyze(media: Media, findings: list[dict[str, Any]]) -> Assessment:
    # 위험 이유와 등급은 탐지 API에서 함께 받는다. 점수 집계에 별도 유료 호출은 없다.
    weights = {"low": 0.25, "medium": 0.6, "high": 0.9}
    risk = max((weights.get(f.get("risk_level"), 0.6) for f in findings), default=0.0)
    return Assessment(risky=bool(findings), risk=risk, findings=findings)

"""② 위험 분석·재검증: 탐지 항목을 보고 위험도를 매긴다. 변환 후 재검증에도 같은 함수를 쓴다."""

from typing import Any

from stages.types import Assessment, Media


async def analyze(media: Media, findings: list[dict[str, Any]]) -> Assessment:
    # TODO(AI): 항목 조합으로 위험도 계산 (예: 주소 + 얼굴이면 높음)
    return Assessment(risky=bool(findings), risk=0.0, findings=findings)

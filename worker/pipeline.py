"""① 탐지 → ② 위험 분석 → ③ 변환 → ② 재검증 (위험이 남아 있으면 최대 max_iterations 회 반복).

단계가 바뀔 때마다 report 로 진행 상태를 알리면 프론트 진행 표시에 그대로 나온다.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from stages.detect import detect
from stages.risk import analyze
from stages.transform import transform
from stages.types import Assessment, Media
from worker.contract import JobStatus, UserChoice

Report = Callable[..., Awaitable[None]]  # report(status, iteration=0, message=None)


@dataclass
class DetectionResult:
    before: Assessment  # 탐지·분석 결과 (변환 전)


@dataclass
class PipelineResult:
    output: Media
    before: Assessment  # 처리 전 분석
    after: Assessment  # 마지막 재검증 결과 (처리하지 않았으면 before 와 같다)
    iterations: int


def _apply_user_choices(assessment: Assessment, choices: list[UserChoice]) -> Assessment:
    """사용자 선택으로 findings의 action 필드를 덮어쓴다."""
    findings: list[dict[str, Any]] = list(assessment.findings)
    seen = set()
    for choice in choices:
        if (
            choice.idx in seen
            or not 0 <= choice.idx < len(findings)
            or choice.action not in {"blur", "mask", "remove", "skip"}
        ):
            raise ValueError("invalid user choice")
        seen.add(choice.idx)
        findings[choice.idx] = {**findings[choice.idx], "action": choice.action}
    return Assessment(risky=assessment.risky, risk=assessment.risk, findings=findings)


async def run_detection_phase(media: Media, *, report: Report) -> DetectionResult:
    """Phase 1: 탐지·분석만 실행하고 결과를 반환한다. 변환은 하지 않는다."""
    await report(JobStatus.DETECTING, message="개인정보 탐지 중")
    findings = await detect(media)

    await report(JobStatus.ANALYZING, message="위험도 분석 중")
    before = await analyze(media, findings)

    return DetectionResult(before=before)


async def run_pipeline(
    media: Media,
    *,
    max_iterations: int,
    report: Report,
    user_choices: list[UserChoice] | None = None,
    initial_findings: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    """Phase 2: 최초 탐지 결과에 사용자 선택을 반영해 변환하고 한 번 재검증한다."""
    if user_choices is not None:
        if initial_findings is None:
            raise ValueError("initial findings are required for user-selected edits")
        findings = initial_findings
    else:
        await report(JobStatus.DETECTING, message="개인정보 탐지 중")
        findings = await detect(media)

    await report(JobStatus.ANALYZING, message="위험도 분석 중")
    before = await analyze(media, findings)

    # 사용자가 선택한 action으로 findings 덮어쓰기
    if user_choices:
        before = _apply_user_choices(before, user_choices)

    current, after, iterations = media, before, 0
    while after.risky and iterations < max_iterations:
        iterations += 1
        await report(JobStatus.TRANSFORMING, iteration=iterations, message="보호 처리 중")
        current = await transform(current, after)

        await report(JobStatus.VERIFYING, iteration=iterations, message="재검증 중")
        after = await analyze(current, await detect(current))

        # 사용자가 남겨둔 단서를 자동 반복으로 다시 덮어쓰지 않는다.
        if user_choices is not None:
            break

    return PipelineResult(output=current, before=before, after=after, iterations=iterations)

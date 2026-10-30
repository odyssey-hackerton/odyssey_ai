"""① 탐지 → ② 위험 분석 → ③ 변환 → ② 재검증 (위험이 남아 있으면 최대 max_iterations 회 반복).

단계가 바뀔 때마다 report 로 진행 상태를 알리면 프론트 진행 표시에 그대로 나온다.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from stages.detect import detect
from stages.risk import analyze
from stages.transform import transform
from stages.types import Assessment, Media
from worker.contract import JobStatus

Report = Callable[..., Awaitable[None]]  # report(status, iteration=0, message=None)


@dataclass
class PipelineResult:
    output: Media
    before: Assessment  # 처리 전 분석
    after: Assessment  # 마지막 재검증 결과 (처리하지 않았으면 before 와 같다)
    iterations: int


async def run_pipeline(media: Media, *, max_iterations: int, report: Report) -> PipelineResult:
    await report(JobStatus.DETECTING, message="개인정보 탐지 중")
    findings = await detect(media)

    await report(JobStatus.ANALYZING, message="위험도 분석 중")
    before = await analyze(media, findings)

    current, after, iterations = media, before, 0
    while after.risky and iterations < max_iterations:
        iterations += 1
        await report(JobStatus.TRANSFORMING, iteration=iterations, message="보호 처리 중")
        current = await transform(current, after)

        await report(JobStatus.VERIFYING, iteration=iterations, message="재검증 중")
        after = await analyze(current, await detect(current))

    return PipelineResult(output=current, before=before, after=after, iterations=iterations)

"""⓪ 맥락 → ① 탐지 → ② 위험 분석 → ③ 정책(에이전트3) → 사용자 선택 → ③ 변환 → ④ 재검증.

(사용자 선택이 없으면 위험이 남는 동안 최대 max_iterations 회 반복한다.)
② 와 ④ 는 같은 코드(stages.risk.find_and_judge / analyze / assess)를 쓴다.
③ 에이전트3은 두 곳에서 일한다. Phase 1 끝과 ④ 결과에서 findings 에 편집 분류·추천·정책을 붙여
프론트에 보여 주고(stages/protect.py), Phase 2 에서 사용자가 고른 편집을 정책으로 검사한 뒤 최종
픽셀을 그린다(stages/transform.py). 이미지재생성 패치는 API 가 만들어 Redis 에 둔 것을 합성한다.

단계가 바뀔 때마다 report 로 진행 상태를 알리면 프론트 진행 표시에 그대로 나온다.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from stages import risk
from stages.context import build_context
from stages.images import open_image
from stages.protect import apply_policy, check_choice
from stages.transform import transform
from stages.types import Assessment, Media
from stages.verify import verify
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
    verification: dict[str, Any] | None = None  # ④ 처리 전후 객체 비교 (재검증했을 때만)


Findings = list[dict[str, Any]]


def _apply_user_choices(
    assessment: Assessment, choices: list[UserChoice], image_size: tuple[int, int] | None = None
) -> Assessment:
    """사용자 선택(처리 방법·조절한 영역·부분마스킹·생성 패치)을 findings 에 반영한다.

    에이전트3 정책(policy)보다 약한 편집은 PolicyViolation 으로 거절한다. 백엔드가 같은 규칙으로
    422 를 돌려주지만, 큐에 직접 넣는 클라이언트도 있으므로 워커가 다시 검사한다.
    """
    findings: list[dict[str, Any]] = list(assessment.findings)
    seen = set()
    for choice in choices:
        if choice.idx in seen or not 0 <= choice.idx < len(findings):
            raise ValueError("invalid user choice")
        seen.add(choice.idx)
        finding = findings[choice.idx]
        check_choice(
            finding,
            choice.action,
            bbox=choice.bbox,
            mask_box=choice.mask_box,
            generation_id=choice.generation_id,
            image_size=image_size,
        )
        updated = {**finding, "action": choice.action}
        if choice.bbox is not None and finding.get("bbox") is not None:
            updated["bbox"] = [float(v) for v in choice.bbox]
        if choice.mask_box is not None:
            updated["mask_box"] = [float(v) for v in choice.mask_box]
        if choice.generation_id:
            updated["generation_id"] = choice.generation_id
        findings[choice.idx] = updated
    return Assessment(
        risky=assessment.risky, risk=assessment.risk, findings=findings, combos=assessment.combos
    )


def _with_policy(assessment: Assessment, image_size: tuple[int, int] | None = None) -> Assessment:
    """③ 에이전트3 정책을 붙인 결과. 리포트(프론트)로 나가는 findings 는 항상 이 형태다."""
    return Assessment(
        risky=assessment.risky,
        risk=assessment.risk,
        findings=apply_policy(assessment.findings, image_size),
        combos=assessment.combos,
    )


def _image_size(media: Media) -> tuple[int, int] | None:
    if media.image is None:
        return None
    try:
        return open_image(media.image).size
    except Exception:  # 읽을 수 없으면 크기 없이 진행 (재생성 후보 없음)
        return None


async def run_detection_phase(
    media: Media,
    *,
    report: Report,
    initial_findings: list[dict[str, Any]] | None = None,
    initial_combos: list[dict[str, Any]] | None = None,
    on_detected: Callable[[Findings, Findings], Awaitable[None]] | None = None,
) -> DetectionResult:
    """Phase 1: 탐지·분석만 실행하고 결과를 반환한다. 변환은 하지 않는다."""
    await report(JobStatus.DETECTING, message="개인정보 탐지 중")
    context = await build_context()  # ⓪ 하루 캐시. 실패하지 않는다 (최악이면 고정 규칙만)
    # 재생성 후보 영역 검사(Q_i)에 쓴다. 원본을 지우기 전에 크기를 잡아 둔다.
    size = await asyncio.to_thread(_image_size, media)
    if initial_findings is not None:
        findings, combos = initial_findings, initial_combos or []
    else:
        # ② 판정은 원본 사진이 필요하므로 원본을 지우기 전에 끝낸다.
        findings, combos = await risk.find_and_judge(media, context)
    if on_detected is not None:
        await on_detected(findings, combos)
        media.image = None

    await report(JobStatus.ANALYZING, message="위험도 분석 중")
    before = _with_policy(await risk.analyze(media, findings, combos, context), size)

    return DetectionResult(before=before)


async def run_pipeline(
    media: Media,
    *,
    max_iterations: int,
    report: Report,
    user_choices: list[UserChoice] | None = None,
    initial_findings: list[dict[str, Any]] | None = None,
    resume_output: Media | None = None,
    on_transformed: Callable[[Media], Awaitable[None]] | None = None,
    patches: dict[str, bytes] | None = None,
) -> PipelineResult:
    """Phase 2: 최초 탐지 결과에 사용자 선택을 반영해 변환하고 한 번 재검증한다."""
    context = await build_context()
    combos: Findings = []  # 1단계 조합 경고는 백엔드가 details 로 이미 받았다
    if user_choices is not None:
        if initial_findings is None:
            raise ValueError("initial findings are required for user-selected edits")
        findings = initial_findings
    else:
        await report(JobStatus.DETECTING, message="개인정보 탐지 중")
        findings, combos = await risk.find_and_judge(media, context)

    await report(JobStatus.ANALYZING, message="위험도 분석 중")
    before = await risk.analyze(media, findings, combos, context)
    if user_choices is None:
        before = _with_policy(before, await asyncio.to_thread(_image_size, media))

    # 사용자가 선택한 처리·영역으로 findings 덮어쓰기 (정책 위반은 여기서 거절)
    if user_choices:
        size = await asyncio.to_thread(_image_size, media) if resume_output is None else None
        before = _apply_user_choices(before, user_choices, size)

    current, after, iterations = media, before, 0
    verification = None
    while (
        after.risky or (user_choices is not None and iterations == 0)
    ) and iterations < max_iterations:
        iterations += 1
        await report(JobStatus.TRANSFORMING, iteration=iterations, message="보호 처리 중")
        current = (
            resume_output if resume_output is not None else await transform(current, after, patches)
        )
        resume_output = None
        if on_transformed is not None:
            await on_transformed(current)
            media.image = None

        await report(JobStatus.VERIFYING, iteration=iterations, message="재검증 중")
        after, verification = await verify(current, context, before)
        after = _with_policy(after)

        # 사용자가 남겨둔 단서를 자동 반복으로 다시 덮어쓰지 않는다.
        if user_choices is not None:
            break

    return PipelineResult(
        output=current,
        before=before,
        after=after,
        iterations=iterations,
        verification=verification,
    )

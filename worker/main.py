"""AI 워커. 일반 서버의 Redis `jobs` 에서 작업을 가져와 처리하고 `results` 로 결과를 보낸다.

    python -m worker.main

워커는 Redis(일반 서버 사설 IP), S3와 Gemini HTTPS API에 접속한다. Postgres에는 접속하지 않는다.
원본 글·사진, presigned URL은 로그에 남기지 않는다.
"""

import asyncio
import contextlib
import json
import logging
import signal
from pathlib import PurePosixPath

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from stages.gemini import AIServiceError
from stages.protect import PolicyViolation
from stages.types import Media
from worker.config import Settings, get_settings
from worker.contract import JobMessage, JobStatus, ResultMessage
from worker.pipeline import run_detection_phase, run_pipeline
from worker.queue import create_redis, ensure_group, load_patches, send_result, set_progress
from worker.storage import Storage, result_key

logger = logging.getLogger("worker")

CLAIM_CHECK_SECONDS = 60
RETRY_SECONDS = 5


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)


class OriginalCleanupError(RuntimeError):
    """Keep the job pending until S3 confirms that the original is gone."""


def execution_key(msg: JobMessage) -> str:
    return f"execution:{msg.execution_id}"


def edited_key(msg: JobMessage) -> str:
    return f"originals/{msg.job_id}/{msg.execution_id}-edited.png"


async def discard_original(storage: Storage, key: str | None) -> None:
    if key:
        try:
            await storage.delete(key)
        except Exception:
            raise OriginalCleanupError("temporary image cleanup must be retried") from None


async def process(
    redis: Redis, storage: Storage, settings: Settings, msg: JobMessage
) -> ResultMessage:
    job_id = str(msg.job_id)
    checkpoint_key = f"{execution_key(msg)}:checkpoint"
    saved = await redis.get(checkpoint_key)
    checkpoint = json.loads(saved) if saved else None

    async def report(status: JobStatus, iteration: int = 0, message: str | None = None) -> None:
        await set_progress(
            redis,
            job_id,
            status,
            iteration=iteration,
            message=message,
            ttl_seconds=settings.progress_ttl_seconds,
        )

    media = Media(text=msg.text)
    try:
        if msg.object_key and checkpoint is None:
            media.image = await storage.get(msg.object_key)
            media.image_ext = PurePosixPath(msg.object_key).suffix.lstrip(".").lower()

        if msg.user_choices is None:

            async def detected(findings, combos):
                # 원본 삭제 전에 복구에 필요한 탐지·판정 결과만 저장한다.
                await redis.setex(
                    checkpoint_key,
                    settings.progress_ttl_seconds,
                    json.dumps({"stage": "detected", "findings": findings, "combos": combos}),
                )
                await discard_original(storage, msg.object_key)

            detection = await run_detection_phase(
                media,
                report=report,
                on_detected=detected,
                initial_findings=checkpoint["findings"] if checkpoint else None,
                initial_combos=checkpoint.get("combos") if checkpoint else None,
            )
            return ResultMessage(
                job_id=msg.job_id,
                execution_id=msg.execution_id,
                status="AWAITING_USER",
                risk_detected=detection.before.risky,
                risk_before=detection.before.risk,
                findings=detection.before.findings,
                # 조합 경고 (② 2단계). 객체 findings 와 별개로 보여준다.
                details={"worker": settings.worker_name, "combos": detection.before.combos},
            )

        async def transformed(output: Media):
            nonlocal checkpoint
            if checkpoint is None:
                if output.image is not None:
                    await storage.put(edited_key(msg), output.image, "image/png")
                checkpoint = {
                    "stage": "edited",
                    "text": output.text,
                    "has_image": output.image is not None,
                }
                await redis.setex(
                    checkpoint_key, settings.progress_ttl_seconds, json.dumps(checkpoint)
                )
            await discard_original(storage, msg.object_key)

        # 편집 화면에서 API 가 만들어 Redis 에 둔 이미지재생성 패치
        patches = await load_patches(
            redis, [c.generation_id or "" for c in msg.user_choices if c.action == "regenerate"]
        )
        resume_output = None
        if checkpoint:
            resume_output = Media(
                image=await storage.get(edited_key(msg)) if checkpoint["has_image"] else None,
                image_ext="png" if checkpoint["has_image"] else None,
                text=checkpoint["text"],
            )
        result = await run_pipeline(
            media,
            max_iterations=msg.options.max_iterations,
            report=report,
            user_choices=msg.user_choices,
            initial_findings=(
                [f.model_dump(exclude_none=True) for f in msg.initial_findings]
                if msg.initial_findings is not None
                else None
            ),
            resume_output=resume_output,
            on_transformed=transformed,
            patches=patches,
        )
        out = result.output
        out_key = None
        if out.image is not None:
            out_key = result_key(f"{job_id}/{msg.execution_id}", "png")
            await storage.put(out_key, out.image, "image/png")
        return ResultMessage(
            job_id=msg.job_id,
            execution_id=msg.execution_id,
            status="DONE",
            risk_detected=result.before.risky,
            risk_before=result.before.risk,
            risk_after=result.after.risk,
            iterations=result.iterations,
            result_object_key=out_key,
            result_text=out.text,
            findings=result.after.findings,
            details={
                "worker": settings.worker_name,
                "model": settings.gemini_model,
                "processed_findings": result.before.findings,
                "remaining_risk": result.after.risky,
                "combos": result.after.combos,
                # ④ 처리 전후 객체 비교 (재검증했을 때만)
                **({"verification": result.verification} if result.verification else {}),
            },
        )
    finally:
        media.image = None
        await discard_original(storage, msg.object_key)


async def handle(
    redis: Redis, storage: Storage, settings: Settings, msg_id: str, fields: dict[str, str]
) -> None:
    """Persist the outcome before publishing; retries never repeat a completed edit."""
    try:
        msg = JobMessage.model_validate_json(fields.get("payload", ""))
    except ValidationError:
        logger.error("invalid job message %s dropped", msg_id)
        await redis.xack(settings.jobs_stream, settings.worker_group, msg_id)
        return

    key = execution_key(msg)
    lock = redis.lock(f"{key}:lock", timeout=settings.claim_idle_seconds, blocking=False)
    if not await lock.acquire():
        return  # 다른 워커가 처리 중이다. ack하지 않고 결과를 기다린다.

    owner = asyncio.current_task()

    async def renew_lock():
        try:
            while True:
                await asyncio.sleep(max(1, settings.claim_idle_seconds / 3))
                await lock.extend(settings.claim_idle_seconds, replace_ttl=True)
        except Exception:
            owner.cancel()  # 잠금을 잃은 워커는 결과를 쓰거나 ack하지 않는다.
            raise

    renewal = asyncio.create_task(renew_lock())
    try:
        cached = await redis.get(f"{key}:result")
        if cached:
            result = ResultMessage.model_validate_json(cached)
        else:
            try:
                result = await process(redis, storage, settings, msg)
            except (OriginalCleanupError, RedisConnectionError, RedisTimeoutError):
                raise
            except Exception as error:
                # AIServiceError·PolicyViolation 메시지에는 키·원문이 없다. 다른 예외는 이름만.
                safe = isinstance(error, (AIServiceError, PolicyViolation))
                reason = f": {error}" if safe else ""
                logger.warning("job %s failed (%s%s)", msg.job_id, type(error).__name__, reason)
                result = ResultMessage(
                    job_id=msg.job_id,
                    execution_id=msg.execution_id,
                    status="FAILED",
                    error=(
                        str(error)
                        if isinstance(error, PolicyViolation)
                        else "처리하지 못했습니다. 원본 파일을 다시 업로드해 주세요"
                    ),
                )
            await redis.setex(
                f"{key}:result", settings.progress_ttl_seconds, result.model_dump_json()
            )

        # 실패·타임아웃·결과 전송 재시도에서도 원본과 임시 편집본을 정리한다.
        await discard_original(storage, msg.object_key)
        if msg.object_key and msg.user_choices is not None:
            await discard_original(storage, edited_key(msg))
        await send_result(redis, settings.results_stream, result, settings.stream_maxlen)
        await redis.xack(settings.jobs_stream, settings.worker_group, msg_id)
        await redis.delete(f"{key}:checkpoint")
        logger.info("job %s -> %s", msg.job_id, result.status)
    finally:
        renewal.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await renewal
        with contextlib.suppress(Exception):
            await lock.release()


async def claim_stale(redis: Redis, storage: Storage, settings: Settings) -> None:
    """처리 도중 워커가 죽어 ack 되지 않은 작업을 다시 가져와 처리한다."""
    _, entries, _ = await redis.xautoclaim(
        settings.jobs_stream,
        settings.worker_group,
        settings.worker_name,
        min_idle_time=settings.claim_idle_seconds * 1000,
        start_id="0-0",
        count=10,
    )
    for msg_id, fields in entries:
        if fields:  # 이미 잘린(삭제된) 메시지는 fields 가 비어 있다
            logger.info("reclaimed stale job message %s", msg_id)
            await handle(redis, storage, settings, msg_id, fields)


async def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    redis = create_redis(settings.redis_url)
    storage = Storage(settings)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    # 일반 서버가 재시작 중이어도 죽지 않고 다시 붙을 때까지 기다린다.
    while not stop.is_set():
        try:
            await redis.ping()
            await ensure_group(redis, settings.jobs_stream, settings.worker_group)
            break
        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning("redis not reachable (%s), retry in %ss", e, RETRY_SECONDS)
            await _sleep_or_stop(stop, RETRY_SECONDS)
    logger.info(
        "worker %s started: redis ok, stream=%s group=%s, s3 keys %s",
        settings.worker_name,
        settings.jobs_stream,
        settings.worker_group,
        "set" if storage.configured else "NOT set (image jobs will fail)",
    )

    next_claim = 0.0
    while not stop.is_set():
        try:
            if loop.time() >= next_claim:
                await claim_stale(redis, storage, settings)
                next_claim = loop.time() + CLAIM_CHECK_SECONDS

            resp = await redis.xreadgroup(
                settings.worker_group,
                settings.worker_name,
                {settings.jobs_stream: ">"},
                count=1,
                block=5000,
            )
            for _, entries in resp or []:
                for msg_id, fields in entries:
                    await handle(redis, storage, settings, msg_id, fields)
        except (RedisConnectionError, RedisTimeoutError) as e:
            # 처리 중이던 작업은 ack 되지 않았으므로 다시 붙은 뒤 claim_stale 이 가져간다.
            logger.warning("redis connection lost (%s), retry in %ss", e, RETRY_SECONDS)
            await _sleep_or_stop(stop, RETRY_SECONDS)
        except ResponseError as e:
            if "NOGROUP" not in str(e):
                raise
            # Redis 데이터가 초기화되어 그룹이 사라진 경우
            await ensure_group(redis, settings.jobs_stream, settings.worker_group)
        except Exception as error:
            logger.warning(
                "worker operation failed (%s), pending job will retry", type(error).__name__
            )
            await _sleep_or_stop(stop, RETRY_SECONDS)

    logger.info("worker %s stopped", settings.worker_name)
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())

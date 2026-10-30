"""Redis 연결, 진행 상태, 결과 전송. 메시지 형식은 contract.py, 키 이름은 백엔드와 같아야 한다."""

from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from worker.contract import ResultMessage

# XREADGROUP BLOCK(5초)보다 길어야 대기 중에 타임아웃이 나지 않는다.
SOCKET_TIMEOUT_SECONDS = 30


def create_redis(url: str) -> Redis:
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=5,
        health_check_interval=30,
    )


async def ensure_group(redis: Redis, stream: str, group: str) -> None:
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def set_progress(
    redis: Redis,
    job_id: str,
    status: str,
    *,
    iteration: int = 0,
    message: str | None = None,
    ttl_seconds: int = 24 * 3600,
) -> None:
    """진행 상태 Redis Hash `job:{job_id}`. 프론트 진행 표시에 그대로 쓰인다."""
    key = f"job:{job_id}"
    mapping = {
        "status": status,
        "iteration": str(iteration),
        "message": message or "",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl_seconds)
        await pipe.execute()


async def send_result(redis: Redis, stream: str, result: ResultMessage, maxlen: int) -> None:
    await redis.xadd(stream, {"payload": result.model_dump_json()}, maxlen=maxlen, approximate=True)

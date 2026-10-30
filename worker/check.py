"""GPU 서버 → 일반 서버 통신 확인. 작업을 처리하지 않고 연결만 확인한다.

    python -m worker.check

Redis 접속·인증, 작업 큐, S3 버킷 접근을 차례로 확인하고 하나라도 실패하면 종료 코드 1.
"""

import asyncio
import sys

from redis.exceptions import ResponseError

from worker.config import get_settings
from worker.queue import create_redis
from worker.storage import Storage


async def main() -> int:
    s = get_settings()
    ok = True

    redis = create_redis(s.redis_url)
    try:
        await redis.ping()
        print("✔ Redis 접속·인증")
        length = await redis.xlen(s.jobs_stream)
        try:
            groups = [g["name"] for g in await redis.xinfo_groups(s.jobs_stream)]
        except ResponseError:  # 스트림이 아직 없음
            groups = []
        group_text = ", ".join(groups) or "없음 (워커가 처음 뜰 때 생성)"
        print(f"✔ 작업 큐 '{s.jobs_stream}': 메시지 {length}개, 그룹 {group_text}")
    except Exception as e:
        ok = False
        print(f"✖ Redis: {type(e).__name__}: {e}")
        print("  REDIS_URL 의 IP·비밀번호, 일반 서버 REDIS_BIND, ACG(6379) 규칙을 확인하세요.")
    finally:
        await redis.aclose()

    storage = Storage(s)
    if not storage.configured:
        print("- S3: AWS 키가 없어 건너뜀 (글 작업만 처리 가능)")
    else:
        # 워커 권한(results/ 쓰기)으로 할 수 있는 확인: 작은 파일을 하나 써 본다.
        key = f"results/_check-{s.worker_name}.txt"
        try:
            await storage.put(key, b"ok", "text/plain")
            print(f"✔ S3 버킷 '{s.s3_bucket}' 쓰기 ({key})")
        except Exception as e:
            ok = False
            print(f"✖ S3 버킷 '{s.s3_bucket}': {type(e).__name__}: {e}")
            print("  AWS 키, 버킷 이름, 리전, IAM 권한(results/* PutObject)을 확인하세요.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

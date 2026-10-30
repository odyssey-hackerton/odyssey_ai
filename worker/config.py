import socket
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env)로 주입되는 설정. 이름은 대소문자 구분 없이 매칭된다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # 일반 서버의 Redis (사설 IP). 예: redis://:비밀번호@10.1.1.27:6379/0
    redis_url: str = "redis://localhost:6379/0"

    # AWS S3. 사진 작업에만 필요하다 (글만 있는 작업은 S3를 쓰지 않는다).
    s3_bucket: str = "odyssey-images"
    s3_region: str = "ap-northeast-2"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # 큐 (API-워커 계약 v1). 백엔드 설정과 같아야 한다.
    jobs_stream: str = "jobs"
    results_stream: str = "results"
    worker_group: str = "workers"
    stream_maxlen: int = 10000
    progress_ttl_seconds: int = 24 * 3600

    # 같은 그룹 안에서 이 워커를 구분하는 이름. 워커를 여러 대 띄우면 서로 달라야 한다.
    worker_name: str = f"worker-{socket.gethostname()}"
    # 처리 도중 워커가 죽어 남은 작업을 이 시간(초)이 지나면 다시 가져온다.
    # 작업 하나가 이보다 오래 걸리면 늘린다.
    claim_idle_seconds: int = 600


@lru_cache
def get_settings() -> Settings:
    return Settings()

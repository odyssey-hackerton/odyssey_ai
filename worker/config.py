import socket
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env)로 주입되는 설정. 이름은 대소문자 구분 없이 매칭된다."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # 테스트용 무료 Gemini 프로젝트 키. 유료 모델/폴백은 자동 호출하지 않는다.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: int = Field(default=90, ge=1, le=300)

    # ① 탐지(에이전트1) VLM 제공자: gemini (기본) | qwen (vLLM OpenAI 호환 서버, stages/qwen.py)
    vlm_provider: Literal["gemini", "qwen"] = "gemini"
    qwen_base_url: str = "http://localhost:8001/v1"
    qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_api_key: SecretStr | None = None

    # ① 탐지(에이전트1) 로컬 모델. 모델이 없으면 해당 모듈만 건너뛴다.
    detect_use_vlm: bool = True  # False 면 Gemini 를 호출하지 않고 OCR·PII·EXIF 만 쓴다
    detect_use_ocr: bool = (
        False  # True 면 PaddleOCR 로컬 OCR 도 돌린다 (기본은 Gemini 가 글자까지 읽음)
    )
    veil_dir: str = "models/veil"  # hf download 1T/veil-pii-ko-lite --local-dir models/veil
    paddle_device: str = "cpu"  # "cpu" | "gpu:0"
    paddle_mkldnn: bool = False  # CPU oneDNN 가속. std::exception 이 나면 False 유지

    # ⓪ 맥락: 국가+날짜의 사회 이슈 (stages/context.py). 키가 없으면 고정 규칙만 쓴다.
    tavily_api_key: SecretStr | None = None
    context_use_search: bool = True
    context_country: str = "KR"
    context_timezone: str = "Asia/Seoul"
    context_timeout_seconds: int = Field(default=15, ge=1, le=120)

    # 일반 서버의 Redis (사설 IP). 예: redis://:비밀번호@10.1.1.27:6379/0
    redis_url: str = "redis://localhost:6379/0"

    # AWS S3. 사진 작업에만 필요하다 (글만 있는 작업은 S3를 쓰지 않는다).
    s3_bucket: str = "odyssey-hackarton-images"
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
    claim_idle_seconds: int = Field(default=600, ge=3)


@lru_cache
def get_settings() -> Settings:
    return Settings()

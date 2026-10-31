"""위치 공격 설정. worker/config.py 를 건드리지 않도록 따로 읽는다 (.env 공유)."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kakao_rest_api_key: SecretStr | None = None  # developers.kakao.com REST API 키
    tavily_api_key: SecretStr | None = None
    http_timeout_seconds: int = Field(default=10, ge=1, le=60)

    attack_image_max_px: int = 1024  # 공격자에게 보내는 사본 크기
    search_budget: int = Field(default=4, ge=0, le=8)  # 공격자 A 검색 횟수 상한
    alt_threshold: float = 0.50  # 확신도가 이보다 낮으면 대안 후보를 받는다

    geoclip_top_k: int = Field(default=5, ge=1, le=50)
    geoclip_device: str = "auto"  # auto | cpu | cuda


@lru_cache
def get_location_settings() -> LocationSettings:
    return LocationSettings()

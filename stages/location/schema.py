"""공격 결과 형식. 에이전트2 I/O 명세의 attacks[] 한 칸과 같다."""

import math
from enum import StrEnum

from pydantic import BaseModel, Field


class Level(StrEnum):
    """정밀도 단계. 뒤로 갈수록 넓다."""

    STREET = "street"
    DISTRICT = "district"
    CITY = "city"
    PROVINCE = "province"
    COUNTRY = "country"
    UNKNOWN = "unknown"


# 단계별 대표 반경(km). 근거 확인 허용 거리와 radius_km 에 쓴다.
RADIUS_KM = {
    Level.STREET: 1.0,
    Level.DISTRICT: 5.0,
    Level.CITY: 25.0,
    Level.PROVINCE: 200.0,
    Level.COUNTRY: 1000.0,
    Level.UNKNOWN: 20000.0,
}
ORDER = list(RADIUS_KM)  # 정밀한 순서

# 논문 precision_level → 우리 단계
PAPER_LEVEL = {
    "building/poi": Level.STREET,
    "building": Level.STREET,
    "poi": Level.STREET,
    "street": Level.STREET,
    "neighborhood": Level.DISTRICT,
    "city": Level.CITY,
    "state/district": Level.PROVINCE,
    "state": Level.PROVINCE,
    "district": Level.PROVINCE,
    "country": Level.COUNTRY,
}


def map_paper_level(value: str | None) -> Level:
    if not value:
        return Level.UNKNOWN
    return PAPER_LEVEL.get(value.strip().lower(), Level.UNKNOWN)


def coarser(level: Level) -> Level:
    """한 단계 넓힌다 (근거가 어긋날 때)."""
    i = ORDER.index(level)
    return ORDER[min(i + 1, len(ORDER) - 1)]


def level_from_radius(radius_km: float) -> Level:
    for level in ORDER:
        if radius_km <= RADIUS_KM[level]:
            return level
    return Level.UNKNOWN


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


class Place(BaseModel):
    name: str
    address: str | None = None
    lat: float
    lon: float


class PoiResult(BaseModel):
    """상호·주소 검색 한 건. hits 가 1이면 결정적 단서, 수백이면 약한 단서."""

    entity_id: str | None
    query: str
    kind: str  # keyword | address
    hits: int  # 전국 검색 건수
    places: list[Place] = Field(default_factory=list)  # 상위 몇 곳만


class Guess(BaseModel):
    attacker: str  # llm_geotrace | llm_direct | geoclip | llm_adaptive
    model: str
    level: Level = Level.UNKNOWN
    region: str | None = None  # 예: "서울 성북구 보문동"
    # 좌표는 계산에만 쓰고 DB에는 H3 칸으로만 저장한다.
    lat: float | None = None
    lon: float | None = None
    radius_km: float = RADIUS_KM[Level.UNKNOWN]
    confidence: float = 0.0
    clues_used: list[str] = Field(default_factory=list)  # entity_id 또는 단서 값
    reasoning: list[str] = Field(default_factory=list)
    alternatives: list[dict] = Field(default_factory=list)
    # 근거 확인 결과. None = 아직 안 함
    grounded: bool | None = None
    ground_note: str | None = None
    new_clues: list[dict] = Field(default_factory=list)  # ①이 못 찾은 단서 (entity_id a1, a2…)

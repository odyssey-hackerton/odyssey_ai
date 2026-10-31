"""카카오 로컬 API: 상호 키워드 검색, 주소 검색, 좌표 → 행정구역."""

import logging

from stages.location.http import SearchError, get_json
from stages.location.schema import Place, PoiResult
from stages.location.settings import get_location_settings

log = logging.getLogger(__name__)
BASE = "https://dapi.kakao.com/v2/local"
TOP_PLACES = 5


def _headers() -> dict[str, str]:
    key = get_location_settings().kakao_rest_api_key
    if key is None:
        raise SearchError("KAKAO_REST_API_KEY is not configured")
    return {"Authorization": f"KakaoAK {key.get_secret_value()}"}


def enabled() -> bool:
    return get_location_settings().kakao_rest_api_key is not None


async def keyword(query: str, entity_id: str | None = None) -> PoiResult:
    """상호·랜드마크 이름. hits = 전국 매장 수 (특이도)."""
    data = await get_json(f"{BASE}/search/keyword.json", {"query": query, "size": 15}, _headers())
    places = [
        Place(
            name=d.get("place_name", ""),
            address=d.get("road_address_name") or d.get("address_name"),
            lat=float(d["y"]),
            lon=float(d["x"]),
        )
        for d in data.get("documents", [])[:TOP_PLACES]
    ]
    hits = int(data.get("meta", {}).get("total_count", len(places)))
    return PoiResult(entity_id=entity_id, query=query, kind="keyword", hits=hits, places=places)


async def address(query: str, entity_id: str | None = None) -> PoiResult:
    """도로명·지번 주소."""
    data = await get_json(f"{BASE}/search/address.json", {"query": query, "size": 10}, _headers())
    places = [
        Place(
            name=d.get("address_name", ""),
            address=d.get("address_name"),
            lat=float(d["y"]),
            lon=float(d["x"]),
        )
        for d in data.get("documents", [])[:TOP_PLACES]
    ]
    hits = int(data.get("meta", {}).get("total_count", len(places)))
    return PoiResult(entity_id=entity_id, query=query, kind="address", hits=hits, places=places)


async def region_name(lat: float, lon: float) -> str | None:
    """좌표 → "서울특별시 성북구 보문동". 키가 없거나 해외면 None."""
    if not enabled():
        return None
    try:
        data = await get_json(f"{BASE}/geo/coord2regioncode.json", {"x": lon, "y": lat}, _headers())
    except SearchError:
        return None
    docs = data.get("documents", [])
    doc = next((d for d in docs if d.get("region_type") == "H"), docs[0] if docs else None)
    return doc.get("address_name") if doc else None


async def lookup(query: str, kind: str, entity_id: str | None = None) -> PoiResult | None:
    """검색 실패는 None. 실패해도 공격은 계속한다."""
    try:
        return await (address(query, entity_id) if kind == "address" else keyword(query, entity_id))
    except SearchError as error:
        log.warning("kakao %s search failed: %s", kind, error)
        return None

"""Tavily 웹 검색. 공격자 A의 검색 도구."""

import logging

from stages.location.http import SearchError, post_json
from stages.location.settings import get_location_settings

log = logging.getLogger(__name__)


def enabled() -> bool:
    return get_location_settings().tavily_api_key is not None


async def search(query: str, max_results: int = 3) -> list[str]:
    """결과 요약 문장 목록. 실패하면 빈 목록."""
    key = get_location_settings().tavily_api_key
    if key is None:
        return []
    try:
        data = await post_json(
            "https://api.tavily.com/search",
            {"query": query, "max_results": max_results, "search_depth": "basic"},
            {"Authorization": f"Bearer {key.get_secret_value()}"},
        )
    except SearchError as error:
        log.warning("web search failed: %s", error)
        return []
    return [
        f"{r.get('title', '')}: {(r.get('content') or '')[:300]}" for r in data.get("results", [])
    ]

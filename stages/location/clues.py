"""① 탐지 결과(findings)에서 위치 단서를 고르고, 검색어와 공격자 프롬프트용 요약을 만든다."""

import asyncio
from typing import Any

from stages.location import kakao
from stages.location.schema import PoiResult

# 지도 검색에 넣을 탐지 종류. 주소는 주소 검색, 나머지는 키워드 검색
ADDRESS_TYPES = {"address", "tracking_number"}
KEYWORD_TYPES = {
    "landmark",
    "store",
    "store_name",
    "school_sign",
    "school",
    "sign",
    "poi",
    "building_name",
    "apartment",
    "bus_stop",
    "subway",
}


def entity_id(finding: dict[str, Any], index: int) -> str:
    """①이 entity_id 를 주면 그대로, 아니면 순번으로 붙인다."""
    return str(finding.get("entity_id") or f"f{index}")


def with_ids(findings: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(entity_id(f, i), f) for i, f in enumerate(findings)]


def _query(finding: dict[str, Any]) -> str | None:
    """검색어. ①이 normalized/text 를 주면 쓰고, 없으면 제목."""
    for key in ("normalized", "text", "value"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:100]
    return None


def search_targets(findings: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(entity_id, 검색어, kind) 목록."""
    targets = []
    for eid, f in with_ids(findings):
        ftype = str(f.get("type", "")).lower()
        query = _query(f)
        if not query:
            continue
        if ftype in ADDRESS_TYPES:
            targets.append((eid, query, "address"))
        elif ftype in KEYWORD_TYPES:
            targets.append((eid, query, "keyword"))
    return targets


async def pre_search(findings: list[dict[str, Any]]) -> list[PoiResult]:
    """1단계 병렬 검색: 공격자보다 먼저 돌려 결과를 프롬프트에 넣는다."""
    if not kakao.enabled():
        return []
    targets = search_targets(findings)
    results = await asyncio.gather(*(kakao.lookup(q, k, eid) for eid, q, k in targets))
    return [r for r in results if r is not None]


def summarize_entities(findings: list[dict[str, Any]]) -> str:
    """공격자에게 보여줄 ① 결과 요약. GPS·EXIF 값은 넣지 않는다."""
    lines = []
    for eid, f in with_ids(findings):
        if f.get("source") == "exif" or f.get("type") in {"gps", "metadata"}:
            continue
        desc = f.get("title") or f.get("type")
        text = _query(f)
        lines.append(
            f"- {eid}: type={f.get('type')} ({desc})" + (f' text="{text}"' if text else "")
        )
    return "\n".join(lines) or "(none)"


def summarize_poi(results: list[PoiResult]) -> str:
    lines = []
    for r in results:
        if r.hits == 0:
            lines.append(f'- "{r.query}" ({r.entity_id}) → 0 hits')
        elif r.hits == 1 and r.places:
            p = r.places[0]
            lines.append(f'- "{r.query}" ({r.entity_id}) → exactly 1 hit: {p.name}, {p.address}')
        else:
            where = "; ".join(f"{p.address}" for p in r.places[:3])
            weak = " (weak clue: chain or common name)" if r.hits > 20 else ""
            lines.append(f'- "{r.query}" ({r.entity_id}) → {r.hits} hits{weak}, e.g. {where}')
    return "\n".join(lines) or "(no map results)"

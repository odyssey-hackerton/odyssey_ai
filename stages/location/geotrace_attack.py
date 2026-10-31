"""공격자 A: GeoTrace 2단계 프롬프트 수정판.

1) 단서 추출 → 2) 검색 계획(최대 search_budget 회) → 검색 실행 → 3) 국가→거리 순 최종 추론.
Gemini 호출 3번. GPS 는 주지 않고 EXIF 를 지운 1024px 사본만 보낸다.
"""

import asyncio
import base64
import json
import logging
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ValidationError

from stages.gemini import AIServiceError, generate_json
from stages.images import open_image, png_bytes
from stages.location import clues as clue_utils
from stages.location import kakao, prompts, web
from stages.location.schema import RADIUS_KM, Guess, PoiResult, map_paper_level
from stages.location.settings import get_location_settings
from worker.config import get_settings

log = logging.getLogger(__name__)
NAME = "llm_geotrace"


# ---- 응답 형식 ----
# Gemini 는 길이·범위 제약이 많은 스키마를 "too many states" 로 거부한다.
# 그래서 스키마에는 제약을 넣지 않고, 받은 뒤에 잘라내거나 범위 안으로 맞춘다.
Conf = Annotated[float, AfterValidator(lambda v: min(1.0, max(0.0, v)))]
Lat = Annotated[float, AfterValidator(lambda v: min(90.0, max(-90.0, v)))]
Lon = Annotated[float, AfterValidator(lambda v: min(180.0, max(-180.0, v)))]


def _cut(n: int):
    return AfterValidator(lambda v: v[:n])


class Clue(BaseModel):
    type: str
    value: Annotated[str, _cut(300)]
    confidence: Conf
    entity_id: str | None = None
    bbox: list[float] | None = None


class Stage1(BaseModel):
    clues: Annotated[list[Clue], _cut(40)]
    likely_continents: list[str] = []
    likely_hemisphere: str = "unknown"
    urban_or_rural: str = "unknown"
    driving_side: str = "unknown"
    overall_region_guess: str = ""
    reasoning: str = ""


class Search(BaseModel):
    tool: Literal["kakao_local_search", "web_search"]
    query: Annotated[str, _cut(100)]


class Plan(BaseModel):
    searches: Annotated[list[Search], _cut(8)]


class Step(BaseModel):
    hypothesis: str = ""
    evidence: str = ""
    confidence: Conf = 0.0


class Chain(BaseModel):
    country: Step
    state_district: Step
    city_town: Step
    neighborhood_ward: Step
    street_poi: Step


class Alternative(BaseModel):
    description: str
    latitude: float
    longitude: float
    confidence: Conf


class Final(BaseModel):
    latitude: Lat
    longitude: Lon
    description: str
    precision_level: str
    confidence: Conf
    rationale: str = ""
    clues_used: list[str] = []
    alternatives: Annotated[list[Alternative], _cut(3)] = []


class Stage2(BaseModel):
    chain_of_thought: Chain
    final_answer: Final


# ---- 단계 ----
def attack_image(image_bytes: bytes) -> dict[str, str]:
    """EXIF 를 지운 작은 사본. 원본 크기·GPS 는 공격자에게 가지 않는다."""
    image = open_image(image_bytes)
    side = get_location_settings().attack_image_max_px
    image.thumbnail((side, side))
    return {
        "type": "image",
        "data": base64.b64encode(png_bytes(image)).decode(),
        "mime_type": "image/png",
    }


async def _ask(model: type[BaseModel], content: list[dict[str, Any]], instruction: str):
    raw = await generate_json(content, model.model_json_schema(), instruction)
    try:
        return model.model_validate(raw)
    except ValidationError:
        raise AIServiceError(f"attacker {model.__name__} failed schema validation") from None


def summarize_clues(stage1: Stage1) -> str:
    return "\n" + "\n".join(
        f"- [{c.type}] {c.value} (conf {c.confidence:.2f}"
        + (f", {c.entity_id}" if c.entity_id else "")
        + ")"
        for c in stage1.clues
    )


async def run_searches(plan: Plan, budget: int) -> tuple[list[PoiResult], list[str]]:
    """계획한 검색 실행. 상한을 넘는 요청은 버린다."""
    searches = plan.searches[:budget]
    kakao_jobs = [
        kakao.lookup(s.query, "keyword")
        for s in searches
        if s.tool == "kakao_local_search" and kakao.enabled()
    ]
    web_queries = [s.query for s in searches if s.tool == "web_search" and web.enabled()]
    poi, pages = await asyncio.gather(
        asyncio.gather(*kakao_jobs), asyncio.gather(*(web.search(q) for q in web_queries))
    )
    texts = [
        f'web "{q}":\n  ' + "\n  ".join(r) for q, r in zip(web_queries, pages, strict=True) if r
    ]
    return [p for p in poi if p is not None], texts


def new_clues(stage1: Stage1, image_id: str = "img") -> list[dict[str, Any]]:
    """①이 못 찾은 단서에 a1, a2… 번호를 붙인다 (I/O 명세 5-1 source=agent2)."""
    out = []
    for c in stage1.clues:
        if c.entity_id is None and c.confidence >= 0.5:
            out.append(
                {
                    "entity_id": f"{image_id}:a{len(out) + 1}",
                    "source": "agent2",
                    "type": c.type,
                    "value": c.value,
                    "confidence": c.confidence,
                    "box_2d": c.bbox,
                }
            )
    return out


def to_guess(stage2: Stage2, stage1: Stage1, image_id: str) -> Guess:
    f = stage2.final_answer
    level = map_paper_level(f.precision_level)
    chain = stage2.chain_of_thought
    steps = {
        "country": chain.country,
        "province": chain.state_district,
        "city": chain.city_town,
        "district": chain.neighborhood_ward,
        "street": chain.street_poi,
    }
    reasoning = [
        f"{name}: {step.hypothesis} — {step.evidence}"[:300]
        for name, step in steps.items()
        if step.hypothesis
    ]
    return Guess(
        attacker=NAME,
        model=get_settings().gemini_model,
        level=level,
        region=f.description[:200],
        lat=f.latitude,
        lon=f.longitude,
        radius_km=RADIUS_KM[level],
        confidence=f.confidence,
        clues_used=f.clues_used[:20],
        reasoning=reasoning,
        alternatives=[a.model_dump() for a in f.alternatives],
        new_clues=new_clues(stage1, image_id),
    )


async def attack(
    image_bytes: bytes,
    findings: list[dict[str, Any]],
    *,
    poi_results: list[PoiResult] | None = None,
    geoclip_regions: str = "(none)",
    image_id: str = "img",
) -> tuple[Guess, list[PoiResult]]:
    """공격 1회. 반환: (추정, 이번에 쓴 지도 검색 결과 전체 — 근거 확인에 쓴다)."""
    s = get_location_settings()
    image = await asyncio.to_thread(attack_image, image_bytes)
    poi = list(poi_results or [])

    stage1 = await _ask(
        Stage1,
        [image],
        prompts.STAGE1.format(entities_summary=clue_utils.summarize_entities(findings)),
    )
    clue_summary = summarize_clues(stage1)

    searched: list[str] = []
    if s.search_budget and (kakao.enabled() or web.enabled()):
        plan = await _ask(
            Plan,
            [{"type": "text", "text": "Plan the searches."}],
            prompts.PLAN.format(
                clue_summary=clue_summary,
                poi_results=clue_utils.summarize_poi(poi),
                geoclip_regions=geoclip_regions,
                budget=s.search_budget,
            ),
        )
        extra, searched = await run_searches(plan, s.search_budget)
        poi += extra

    stage2 = await _ask(
        Stage2,
        [image],
        prompts.STAGE2.format(
            clue_summary=clue_summary,
            clue_raw_json=json.dumps(stage1.model_dump(), ensure_ascii=False),
            poi_results=clue_utils.summarize_poi(poi),
            search_results="\n".join(searched) or "(none)",
            geoclip_regions=geoclip_regions,
            alt_threshold=s.alt_threshold,
        ),
    )
    return to_guess(stage2, stage1, image_id), poi

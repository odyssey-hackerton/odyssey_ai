"""공격자 A: GeoTrace 2단계 프롬프트 수정판.

1) 단서 추출 → 2) 검색 계획(최대 search_budget 회) → 검색 실행 → 3) 국가→거리 순 최종 추론.
Gemini 호출 3번. GPS 는 주지 않고 EXIF 를 지운 1024px 사본만 보낸다.
"""

import asyncio
import base64
import json
import logging
import re
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


async def _ask(
    model: type[BaseModel],
    content: list[dict[str, Any]],
    instruction: str,
    thinking_level: str | None = None,
):
    raw = await generate_json(
        content,
        model.model_json_schema(),
        instruction,
        max_output_tokens=get_location_settings().attack_max_output_tokens,
        thinking_level=thinking_level,
    )
    try:
        return model.model_validate(raw)
    except ValidationError:
        raise AIServiceError(f"attacker {model.__name__} failed schema validation") from None


def summarize_clues(stage1: Stage1) -> str:
    """단서에 c1, c2… 번호를 붙여 보여준다. Stage 2 는 이 번호로 clues_used 를 보고한다."""
    return "\n" + "\n".join(
        f"- c{i} [{c.type}] {c.value} (conf {c.confidence:.2f}"
        + (f", {c.entity_id}" if c.entity_id else "")
        + ")"
        for i, c in enumerate(stage1.clues, 1)
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


def image_size(image_bytes: bytes) -> tuple[int, int]:
    """공격자에게 보낸 사진과 같은 방향(EXIF 회전 적용)의 원본 크기."""
    return open_image(image_bytes).size


def to_pixels(box: list[float] | None, size: tuple[int, int]) -> list[int] | None:
    """Gemini 박스 [ymin, xmin, ymax, xmax] (0~1000) → 원본 픽셀 [x1, y1, x2, y2]."""
    if not box or len(box) != 4:
        return None
    w, h = size
    y1, x1, y2, x2 = (min(1000.0, max(0.0, v)) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)]


def _used_positions(used: list[str], clues: list[Clue]) -> list[int]:
    """clues_used 를 단서 위치로 바꾼다. c3 같은 번호, entity_id, 단서 문장 순으로 맞춘다."""
    out: list[int] = []
    for u in used:
        u = u.strip()
        m = re.fullmatch(r"[cC](\d+)", u)
        if m and 1 <= int(m.group(1)) <= len(clues):
            i = int(m.group(1)) - 1
        else:
            i = next(
                (
                    k
                    for k, c in enumerate(clues)
                    if u == c.entity_id or u == c.value or (len(u) > 3 and u in c.value)
                ),
                None,
            )
        if i is not None and i not in out:
            out.append(i)
    return out


def label_clues(
    clues: list[Clue], used: list[str], image_id: str, size: tuple[int, int]
) -> tuple[list[dict[str, Any]], list[str]]:
    """①이 못 찾은 단서에 a1, a2… 번호를 붙이고 (I/O 명세 5-1 source=agent2),
    clues_used 를 entity_id 목록으로 바꾼다. 공격자가 쓴 단서는 신뢰도가 낮아도 번호를 받는다."""
    positions = _used_positions(used, clues)
    ids: list[str | None] = []
    new: list[dict[str, Any]] = []
    for i, c in enumerate(clues):
        if c.entity_id:
            ids.append(c.entity_id)
        elif c.confidence >= 0.5 or i in positions:
            eid = f"{image_id}:a{len(new) + 1}"
            ids.append(eid)
            new.append(
                {
                    "entity_id": eid,
                    "source": "agent2",
                    "type": c.type,
                    "value": c.value,
                    "confidence": c.confidence,
                    "bbox": to_pixels(c.bbox, size),
                }
            )
        else:
            ids.append(None)
    return new, [ids[i] for i in positions if ids[i]]


def to_guess(stage2: Stage2, stage1: Stage1, image_id: str, size: tuple[int, int]) -> Guess:
    f = stage2.final_answer
    new, used = label_clues(stage1.clues, f.clues_used, image_id, size)
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
        clues_used=used[:20],
        reasoning=reasoning,
        alternatives=[a.model_dump() for a in f.alternatives],
        new_clues=new,
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
    size = await asyncio.to_thread(image_size, image_bytes)
    return to_guess(stage2, stage1, image_id, size), poi

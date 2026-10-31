"""공격자 B: 단계·검색 없이 "여기 어디야?" 한 번 묻는다 (Gemini 호출 1번).

A 와 같은 Guess 를 돌려줘서 근거 확인·집계를 그대로 쓴다.
"""

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel

from stages.location import clues as clue_utils
from stages.location import prompts
from stages.location.geotrace_attack import (
    Clue,
    Conf,
    Lat,
    Lon,
    _ask,
    _cut,
    attack_image,
    image_size,
    label_clues,
)
from stages.location.schema import RADIUS_KM, Guess, map_paper_level
from worker.config import get_settings

NAME = "llm_direct"


class Direct(BaseModel):
    clues: Annotated[list[Clue], _cut(20)] = []
    description: str
    precision_level: str
    confidence: Conf
    latitude: Lat
    longitude: Lon
    rationale: str = ""
    clues_used: list[str] = []


def to_guess(answer: Direct, image_id: str, size: tuple[int, int]) -> Guess:
    new, used = label_clues(answer.clues, answer.clues_used, image_id, size)
    level = map_paper_level(answer.precision_level)
    return Guess(
        attacker=NAME,
        model=get_settings().gemini_model,
        level=level,
        region=answer.description[:200],
        lat=answer.latitude,
        lon=answer.longitude,
        radius_km=RADIUS_KM[level],
        confidence=answer.confidence,
        clues_used=used[:20],
        reasoning=[answer.rationale[:300]] if answer.rationale else [],
        new_clues=new,
    )


async def attack(
    image_bytes: bytes, findings: list[dict[str, Any]], *, image_id: str = "img"
) -> Guess:
    image = await asyncio.to_thread(attack_image, image_bytes)
    answer = await _ask(
        Direct,
        [image],
        prompts.DIRECT.format(entities_summary=clue_utils.summarize_entities(findings)),
    )
    size = await asyncio.to_thread(image_size, image_bytes)
    return to_guess(answer, image_id, size)

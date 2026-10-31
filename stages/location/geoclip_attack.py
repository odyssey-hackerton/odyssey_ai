"""공격자 C: GeoCLIP. 글자 없이 풍경 분위기만으로 상위 후보 좌표를 낸다.

설치: uv sync --group geo  (torch, geoclip. 첫 실행 때 CLIP ViT-L/14 가중치 약 1.7GB 다운로드)
VRAM: 위치 후보 10만 곳 특징값을 한 번만 계산해 캐시하므로 호출마다 늘지 않는다. 약 2~4GB.
"""

import asyncio
import logging
from functools import lru_cache

from stages.images import open_image
from stages.location import kakao
from stages.location.schema import Guess, Level, haversine_km, level_from_radius
from stages.location.settings import get_location_settings

log = logging.getLogger(__name__)
NAME = "geoclip"
CONCENTRATION_KM = 25.0  # 1위 후보 반경 안에 모인 확률을 확신도로 본다


class GeoClipUnavailable(RuntimeError):
    """torch/geoclip 이 설치되지 않았다."""


@lru_cache(maxsize=1)
def _load():
    try:
        import torch
        import torch.nn.functional as F
        from geoclip import GeoCLIP
    except ImportError:
        raise GeoClipUnavailable("install with: uv sync --group geo") from None
    setting = get_location_settings().geoclip_device
    device = ("cuda" if torch.cuda.is_available() else "cpu") if setting == "auto" else setting
    model = GeoCLIP().to(device).eval()
    with torch.no_grad():
        gallery = model.gps_gallery.to(device)
        # 위치 특징은 사진과 무관하므로 한 번만 계산한다.
        loc = F.normalize(model.location_encoder(gallery), dim=1)
    log.info("GeoCLIP loaded on %s (%d gallery points)", device, gallery.shape[0])
    return torch, F, model, gallery.cpu(), loc, device


def _predict(image_bytes: bytes, top_k: int) -> list[tuple[float, float, float]]:
    torch, F, model, gallery, loc, device = _load()
    image = open_image(image_bytes)  # EXIF 회전만 반영. GPS 는 읽지 않는다
    pixels = model.image_encoder.preprocess_image(image).to(device)
    with torch.no_grad():
        feat = F.normalize(model.image_encoder(pixels), dim=1)
        probs = (model.logit_scale.exp() * feat @ loc.t()).softmax(dim=-1)[0]
        top = torch.topk(probs, top_k)
    coords = gallery.index_select(0, top.indices.cpu())
    probs_top = top.values.cpu()
    return [(float(c[0]), float(c[1]), float(p)) for c, p in zip(coords, probs_top, strict=True)]


def to_guess(candidates: list[tuple[float, float, float]]) -> Guess:
    """상위 후보 → Guess. 반경은 1위 주변으로 확률이 얼마나 모였는지로 정한다."""
    if not candidates:
        return Guess(attacker=NAME, model=NAME)
    lat, lon, _ = candidates[0]
    total = sum(p for *_, p in candidates) or 1.0
    dists = [haversine_km(lat, lon, la, lo) for la, lo, _ in candidates]
    # 확률 가중 평균 거리 = 후보들이 퍼진 정도. 최소 1km
    spread = max(1.0, sum(d * p for d, (*_, p) in zip(dists, candidates, strict=True)) / total)
    pairs = list(zip(dists, candidates, strict=True))
    near = sum(p for d, (*_, p) in pairs if d <= CONCENTRATION_KM) / total
    level = level_from_radius(spread)
    # GeoCLIP 은 분위기 모델이라 거리 수준은 주장하지 않는다.
    if level == Level.STREET:
        level, spread = Level.DISTRICT, max(spread, 5.0)
    return Guess(
        attacker=NAME,
        model=NAME,
        level=level,
        lat=lat,
        lon=lon,
        radius_km=round(spread, 1),
        confidence=round(near, 3),
        grounded=False,  # 근거(글자) 없이 분위기로만 추정
        ground_note="scene_only",
        alternatives=[
            {"lat": la, "lon": lo, "p": round(p / total, 3)} for la, lo, p in candidates[1:]
        ],
    )


async def attack(image_bytes: bytes) -> Guess:
    top_k = get_location_settings().geoclip_top_k
    candidates = await asyncio.to_thread(_predict, image_bytes, top_k)
    guess = to_guess(candidates)
    if guess.lat is not None:
        guess.region = await kakao.region_name(guess.lat, guess.lon)
    return guess


async def candidate_regions(guess: Guess) -> str:
    """공격자 A 프롬프트에 넣을 후보 지역 이름 (좌표 대신 이름만)."""
    points = [(guess.lat, guess.lon)] + [(a["lat"], a["lon"]) for a in guess.alternatives]
    names = []
    for lat, lon in points:
        if lat is None:
            continue
        name = await kakao.region_name(lat, lon)
        names.append(name or f"({lat:.1f}, {lon:.1f})")
    return ", ".join(dict.fromkeys(names)) or "(none)"


def preload() -> None:
    """워커 시작 때 미리 올려 두면 첫 작업이 느리지 않다."""
    _load()

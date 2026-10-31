"""근거 확인: 공격자가 말한 위치가 지도 검색 좌표 근처인지 본다 (간단 버전).

- 공격자 추정 반경(단계 대표값) 안에 검색 좌표가 하나라도 있으면 grounded=True, 확신도 유지
- 검색 결과는 있는데 모두 멀면 grounded=False, 확신도 ×0.5, 단계 한 칸 넓힘
- 비교할 검색 결과가 없으면 grounded=False(확인 불가), 확신도 ×0.8
검색 결과 개수가 1곳인 단서 근처면 "결정적 단서"로 표시한다.
"""

from stages.location.schema import RADIUS_KM, Guess, PoiResult, coarser, haversine_km

MISMATCH_PENALTY = 0.5
UNVERIFIED_PENALTY = 0.8
MIN_TOLERANCE_KM = 2.0  # 송장 주소와 실제 사진 위치가 조금 떨어져도 맞는 것으로 본다


def nearest(guess: Guess, poi: list[PoiResult]) -> tuple[float, PoiResult] | None:
    best = None
    for result in poi:
        for place in result.places:
            d = haversine_km(guess.lat, guess.lon, place.lat, place.lon)
            if best is None or d < best[0]:
                best = (d, result)
    return best


def ground(guess: Guess, poi: list[PoiResult]) -> Guess:
    """이미 판정된(grounded 가 정해진) 결과는 건드리지 않는다."""
    if guess.grounded is not None:
        return guess
    out = guess.model_copy(deep=True)
    if out.lat is None or out.lon is None:
        out.grounded, out.ground_note = False, "no_coordinates"
        out.confidence = round(out.confidence * UNVERIFIED_PENALTY, 3)
        return out
    hit = nearest(out, poi)
    if hit is None:
        out.grounded, out.ground_note = False, "unverified"
        out.confidence = round(out.confidence * UNVERIFIED_PENALTY, 3)
        return out
    distance, result = hit
    tolerance = max(MIN_TOLERANCE_KM, RADIUS_KM[out.level])
    if distance <= tolerance:
        out.grounded = True
        out.ground_note = f'{distance:.1f}km from "{result.query}" ({result.hits} hits)'
        if result.hits == 1:
            out.ground_note += ", decisive"
        if result.entity_id and result.entity_id not in out.clues_used:
            out.clues_used.append(result.entity_id)
    else:
        out.grounded = False
        out.ground_note = f"nearest map result {distance:.0f}km away"
        out.confidence = round(out.confidence * MISMATCH_PENALTY, 3)
        out.level = coarser(out.level)
        out.radius_km = RADIUS_KM[out.level]
    return out

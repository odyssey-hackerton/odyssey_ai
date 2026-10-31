"""② 위치 추론 공격: 공격자가 사진만 보고 촬영 위치를 얼마나 좁힐 수 있는지 잰다.

- 공격자 A (`geotrace_attack`): GeoTrace 논문 2단계 프롬프트 수정판.
  단서 추출 → 검색 → 국가→거리 순 추론
- 공격자 C (`geoclip_attack`): GeoCLIP. 글자 없이 풍경 분위기로 상위 후보 좌표
- 근거 확인 (`grounding`): 공격자가 말한 위치가 카카오 검색 좌표 근처인지 비교

GPS(EXIF)는 공격자에게 절대 주지 않는다. 사진은 EXIF를 지운 1024px 사본만 보낸다.
"""

from stages.location.schema import Guess, Level, PoiResult

__all__ = ["Guess", "Level", "PoiResult"]

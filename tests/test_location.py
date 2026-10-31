"""② 위치 공격: 외부 API 없이 도는 단위 테스트 (Gemini·카카오·Tavily 는 가짜로 바꾼다)."""

from io import BytesIO

import pytest
from PIL import Image

from stages.location import (
    clues,
    direct_attack,
    geoclip_attack,
    geotrace_attack,
    grounding,
    kakao,
    prompts,
    web,
)
from stages.location.schema import Guess, Level, Place, PoiResult

FINDINGS = [
    {"type": "store", "source": "vlm", "title": "가게 간판", "text": "보문베이커리"},
    {"type": "address", "source": "vlm", "title": "송장 주소", "text": "서울 성북구 보문로 1"},
    {"type": "face", "source": "vlm", "title": "얼굴"},
    {"type": "metadata", "source": "exif", "title": "사진 파일 정보"},
]
POI = [
    PoiResult(
        entity_id="f0",
        query="보문베이커리",
        kind="keyword",
        hits=1,
        places=[Place(name="보문베이커리", address="서울 성북구", lat=37.585, lon=127.018)],
    )
]


def jpeg_with_gps() -> bytes:
    image = Image.new("RGB", (2000, 1500), "white")
    exif = Image.Exif()
    exif[0x8825] = {2: (37.0, 35.0, 0.0), 4: (127.0, 1.0, 0.0)}  # GPSInfo
    out = BytesIO()
    image.save(out, format="JPEG", exif=exif)
    return out.getvalue()


def test_prompts_format_without_missing_keys():
    prompts.STAGE1.format(entities_summary="-")
    prompts.PLAN.format(clue_summary="-", poi_results="-", geoclip_regions="-", budget=4)
    text = prompts.STAGE2.format(
        clue_summary="-",
        clue_raw_json="{}",
        poi_results="-",
        search_results="-",
        geoclip_regions="-",
        alt_threshold=0.5,
    )
    assert "0.50" in text


def test_search_targets_and_summary_skip_exif():
    assert clues.search_targets(FINDINGS) == [
        ("f0", "보문베이커리", "keyword"),
        ("f1", "서울 성북구 보문로 1", "address"),
    ]
    summary = clues.summarize_entities(FINDINGS)
    assert "f3" not in summary and "보문베이커리" in summary


def test_attack_image_drops_exif_and_shrinks():
    part = geotrace_attack.attack_image(jpeg_with_gps())
    import base64

    with Image.open(BytesIO(base64.b64decode(part["data"]))) as image:
        assert max(image.size) <= 1024
        assert not image.getexif()


def guess(level=Level.STREET, lat=37.586, lon=127.019, conf=0.9) -> Guess:
    return Guess(attacker="llm_geotrace", model="m", level=level, lat=lat, lon=lon, confidence=conf)


def test_grounding_near_map_hit():
    out = grounding.ground(guess(), POI)
    assert out.grounded and out.confidence == 0.9 and "decisive" in out.ground_note
    assert "f0" in out.clues_used


def test_grounding_far_from_map_hit_downgrades():
    out = grounding.ground(guess(lat=35.1, lon=129.0), POI)  # 부산
    assert out.grounded is False and out.confidence == 0.45 and out.level == Level.DISTRICT


def test_grounding_without_map_results():
    out = grounding.ground(guess(), [])
    assert out.grounded is False and out.ground_note == "unverified" and out.confidence == 0.72


def test_geoclip_concentrated_vs_spread():
    seoul = [(37.58, 127.01, 0.4), (37.59, 127.02, 0.3), (37.57, 127.0, 0.2)]
    g = geoclip_attack.to_guess(seoul)
    assert g.level == Level.DISTRICT and g.confidence > 0.9 and g.grounded is False
    world = [(37.5, 127.0, 0.3), (35.6, 139.7, 0.3), (48.8, 2.3, 0.3)]
    assert geoclip_attack.to_guess(world).level in {Level.COUNTRY, Level.UNKNOWN}


STAGE1 = {
    "clues": [
        {"type": "signage", "value": "보문베이커리", "confidence": 0.95, "entity_id": "f0"},
        {
            "type": "kr_bus_route",
            "value": "green bus 1111",
            "confidence": 0.7,
            "entity_id": None,
            "bbox": [100, 100, 200, 300],
        },
    ],
    "overall_region_guess": "East Asia",
}
PLAN = {
    "searches": [
        {"tool": "web_search", "query": "1111 bus route"},
        {"tool": "kakao_local_search", "query": "보문역"},
    ]
    * 3
}
STEP = {"hypothesis": "h", "evidence": "e", "confidence": 0.9}
STAGE2 = {
    "chain_of_thought": {
        k: STEP
        for k in ["country", "state_district", "city_town", "neighborhood_ward", "street_poi"]
    },
    "final_answer": {
        "latitude": 37.586,
        "longitude": 127.019,
        "description": "보문동, 서울, 한국",
        "precision_level": "building/POI",
        "confidence": 0.93,
        "clues_used": ["c1", "c2"],
    },
}


@pytest.fixture
def fake_apis(monkeypatch):
    calls = {"prompts": [], "kakao": [], "web": []}

    async def fake_generate(content, schema, instruction, **kw):
        assert kw["max_output_tokens"] >= 8192  # 생각 토큰 때문에 4096 이면 잘린다
        calls["prompts"].append(instruction)
        return {"Stage1": STAGE1, "Plan": PLAN, "Stage2": STAGE2}[schema["title"]]

    async def fake_lookup(query, kind, entity_id=None):
        calls["kakao"].append(query)
        return PoiResult(entity_id=entity_id, query=query, kind=kind, hits=3, places=[])

    async def fake_web(query, max_results=3):
        calls["web"].append(query)
        return ["route info"]

    monkeypatch.setattr(geotrace_attack, "generate_json", fake_generate)
    monkeypatch.setattr(kakao, "enabled", lambda: True)
    monkeypatch.setattr(kakao, "lookup", fake_lookup)
    monkeypatch.setattr(web, "enabled", lambda: True)
    monkeypatch.setattr(web, "search", fake_web)
    return calls


async def test_geotrace_attack_end_to_end(fake_apis):
    g, poi = await geotrace_attack.attack(
        jpeg_with_gps(), FINDINGS, poi_results=POI, image_id="img_001"
    )
    assert g.attacker == "llm_geotrace" and g.level == Level.STREET and g.confidence == 0.93
    assert len(fake_apis["prompts"]) == 3
    assert len(fake_apis["kakao"]) + len(fake_apis["web"]) == 4  # 검색 상한
    assert g.new_clues[0]["entity_id"] == "img_001:a1"
    assert g.clues_used == ["f0", "img_001:a1"]  # c 번호 → entity_id
    # [ymin,xmin,ymax,xmax] 0~1000 → 2000x1500 원본 픽셀 [x1,y1,x2,y2]
    assert g.new_clues[0]["bbox"] == [200, 150, 600, 300]
    assert "box_2d" not in g.new_clues[0]
    assert "37.5" not in fake_apis["prompts"][0]  # GPS 값이 프롬프트에 없다
    assert grounding.ground(g, poi).grounded is True


def test_attacker_schemas_have_no_constraints_gemini_rejects():
    import json

    models = (geotrace_attack.Stage1, geotrace_attack.Plan, geotrace_attack.Stage2)
    for model in (*models, direct_attack.Direct):
        text = json.dumps(model.model_json_schema())
        for key in ("maxLength", "minLength", "maxItems", "minimum", "maximum"):
            assert key not in text, (model.__name__, key)


def test_attacker_output_is_clamped_and_cut():
    clue = geotrace_attack.Clue(type="signage", value="x" * 500, confidence=1.7)
    assert len(clue.value) == 300 and clue.confidence == 1.0
    stage1 = geotrace_attack.Stage1(clues=[clue.model_dump()] * 50)
    assert len(stage1.clues) == 40


DIRECT = {
    "clues": [
        {"type": "signage", "value": "보문베이커리", "confidence": 0.9, "entity_id": "f0"},
        {"type": "kr_bus_route", "value": "1111 버스", "confidence": 0.7, "bbox": [1, 2, 3, 4]},
    ],
    "description": "서울 성북구 보문동 보문로",
    "precision_level": "street",
    "confidence": 0.8,
    "latitude": 37.586,
    "longitude": 127.019,
    "rationale": "간판과 버스 번호",
    "clues_used": ["f0", "1111 버스"],
}


async def test_direct_attack_returns_same_guess_shape(monkeypatch):
    calls = []

    async def fake_generate(content, schema, instruction, **kw):
        assert kw["thinking_level"] == "low"
        calls.append(instruction)
        assert schema["title"] == "Direct"
        return DIRECT

    monkeypatch.setattr(geotrace_attack, "generate_json", fake_generate)
    g = await direct_attack.attack(jpeg_with_gps(), FINDINGS, image_id="img_001")
    assert len(calls) == 1 and "37.5" not in calls[0]
    assert g.attacker == "llm_direct" and g.level == Level.STREET and g.confidence == 0.8
    assert g.clues_used == ["f0", "img_001:a1"]  # 문장으로 와도 id 로 바꾼다
    assert [c["entity_id"] for c in g.new_clues] == ["img_001:a1"]
    assert set(g.model_dump()) == set(guess().model_dump())  # A 와 같은 형식
    assert grounding.ground(g, POI).grounded is True


def test_direct_prompt_formats():
    prompts.DIRECT.format(entities_summary="-")


def test_label_clues_maps_used_and_keeps_low_confidence_used_clue():
    clues_ = [
        geotrace_attack.Clue(type="signage", value="SPAO 강남", confidence=0.9),
        geotrace_attack.Clue(type="flora", value="은행나무", confidence=0.3),
        geotrace_attack.Clue(type="sky_weather", value="맑음", confidence=0.2),
        geotrace_attack.Clue(type="signage", value="x", confidence=0.9, bbox=[5, 5, 5, 5]),
    ]
    new, used = geotrace_attack.label_clues(
        clues_, ["c2", "c9", "SPAO 강남", "c2"], "img", (100, 100)
    )
    assert used == ["img:a2", "img:a1"]  # 없는 번호는 버리고, 중복은 한 번만
    assert [c["value"] for c in new] == ["SPAO 강남", "은행나무", "x"]  # 맑음은 안 씀 + 신뢰도 낮음
    assert new[2]["bbox"] is None  # 넓이 0 박스


def test_to_pixels():
    assert geotrace_attack.to_pixels([0, 0, 1000, 1000], (640, 480)) == [0, 0, 640, 480]
    assert geotrace_attack.to_pixels([500, 1200, 0, 0], (100, 100)) == [0, 0, 100, 50]
    assert geotrace_attack.to_pixels(None, (1, 1)) is None


def test_prompts_ask_korean_region_and_clue_numbers():
    assert "Korean" in prompts.DIRECT and "c1" in prompts.DIRECT
    assert "Korean" in prompts.STAGE2 and "c1" in prompts.STAGE2
    assert "- c1 [signage]" in geotrace_attack.summarize_clues(geotrace_attack.Stage1(**STAGE1))

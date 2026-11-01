"""③ 에이전트3: 정책(분류·추천·거절), 최종 렌더(프론트 미리보기와 같은 규칙), 이미지재생성 도구."""

import base64
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from stages import protect
from stages import transform as tf
from stages.images import open_image, png_bytes
from stages.protect import PolicyViolation, apply_policy, check_choice
from stages.types import Assessment, Media
from tests.test_worker import FakeRedis, FakeStorage
from worker.config import Settings
from worker.contract import UserChoice, generation_key
from worker.main import process
from worker.pipeline import _apply_user_choices, run_pipeline

GREEN, RED = (0, 255, 0), (255, 0, 0)
SIZE = (100, 80)  # photo() 크기


def photo() -> bytes:
    image = Image.new("RGB", (100, 80), GREEN)
    image.paste(RED, (20, 20, 60, 40))
    return png_bytes(image)


def finding(**changes):
    return {
        "entity_id": "f0",
        "type": "face",
        "source": "vlm",
        "title": "얼굴",
        "category": "identity",
        "risk_level": "high",
        "bbox": [20, 20, 60, 40],
        "action": "mask",
    } | changes


# ---------------------------------------------------------------- 정책
@pytest.mark.parametrize(
    ("changes", "category", "action"),
    [
        ({}, "object", "mask"),  # 얼굴: 객체, 고위험 → 가림
        ({"type": "address", "source": "ocr", "category": "location"}, "text", "mask"),
        ({"type": "tracking_number", "category": "location"}, "text", "mask"),
        ({"type": "license_plate"}, "object", "mask"),
        (
            {"type": "store_name", "category": "location", "risk_level": "medium"},
            "location",
            "mask",
        ),
        ({"type": "other", "category": "location", "risk_level": "medium"}, "location", "mask"),
        ({"type": "other", "category": "lifestyle", "risk_level": "low"}, "object", "mosaic"),
        ({"type": "other", "category": "attributes", "risk_level": "medium"}, "object", "blur"),
        ({"type": "name", "risk_level": "medium"}, "text", "blur"),
        (
            {"type": "phone_number", "source": "text", "spans": [[0, 5]], "bbox": None},
            "text",
            "mask",
        ),
    ],
)
def test_policy_maps_agent2_findings_to_editor_categories(changes, category, action):
    """② 의 8분류 category 는 risk_category 로 보존하고, 프론트의 category 는 도구 분류가 된다."""
    [out] = apply_policy([finding(**changes)])
    assert out["category"] == category and out["action"] == action
    assert out["risk_category"] == finding(**changes)["category"]
    assert out["policy"]["version"] == protect.POLICY_VERSION
    assert out["policy"]["anchor_bbox"] == (out["bbox"] if out.get("bbox") else None)


def test_locked_types_only_allow_full_mask_and_high_risk_keeps_anchor():
    [rrn] = apply_policy([finding(type="rrn", source="ocr")])
    assert rrn["policy"]["level"] == "locked"
    assert rrn["policy"]["allowed_actions"] == ["mask"]
    [face] = apply_policy([finding()], SIZE)
    assert face["policy"]["level"] == "high"
    assert face["policy"]["allowed_actions"] == ["mask", "blur", "mosaic"]  # 얼굴은 재생성 제외
    assert not face["policy"]["allow_shrink"] and not face["policy"]["allow_delete"]
    [low] = apply_policy([finding(type="other", category="lifestyle", risk_level="low")])
    assert low["policy"]["allow_shrink"] and low["policy"]["allow_delete"]
    assert "exif" in apply_policy([{"type": "gps", "source": "exif"}])[0]["source"]


@pytest.mark.parametrize(
    ("changes", "choice", "message"),
    [
        ({"type": "rrn", "source": "ocr"}, {"action": "blur"}, "전체 가림만"),
        ({}, {"action": "skip"}, "그대로 둘 수 없어요"),
        ({}, {"action": "partialMask", "mask_box": [0, 0, 0.5, 1]}, "부분마스킹"),
        ({}, {"action": "mask", "bbox": [25, 22, 60, 40]}, "작게 줄일 수 없어요"),
        ({}, {"action": "regenerate", "bbox": [20, 20, 60, 40]}, "쓸 수 없어요"),
        (
            {"type": "store_name", "category": "location", "risk_level": "medium"},
            {"action": "regenerate", "bbox": [20, 20, 60, 40]},
            "먼저 생성",
        ),
        ({}, {"action": "mask", "bbox": [0, 0, 200, 40]}, "사진 범위"),
        ({"type": "name", "risk_level": "low"}, {"action": "partialMask"}, "범위를 사진에서"),
        (
            {"type": "other", "category": "location", "risk_level": "medium"},
            {"action": "regenerate"},
            "먼저 생성",
        ),
        ({"source": "text", "spans": [[0, 3]], "bbox": None}, {"action": "blur"}, "쓸 수 없어요"),
    ],
)
def test_weaker_than_policy_edits_are_rejected_with_user_message(changes, choice, message):
    [f] = apply_policy([finding(**changes)], SIZE)
    with pytest.raises(PolicyViolation, match=message):
        check_choice(f, image_size=(100, 80), **choice)


def test_allowed_edits_pass_including_expansion_and_low_risk_shrink():
    [face] = apply_policy([finding()])
    check_choice(face, "blur", bbox=[10, 10, 70, 50], image_size=(100, 80))
    check_choice(face, "mask", bbox=[20.4, 20.2, 59.8, 39.9], image_size=(100, 80))  # 반올림 허용
    [low] = apply_policy([finding(type="other", category="lifestyle", risk_level="low")])
    check_choice(low, "skip")
    check_choice(low, "mosaic", bbox=[30, 25, 50, 35], image_size=(100, 80))
    [sign] = apply_policy(
        [finding(type="store_name", category="location", risk_level="medium")], SIZE
    )
    check_choice(
        sign, "regenerate", bbox=[20, 20, 60, 40], generation_id="abc", image_size=(100, 80)
    )
    [caption] = apply_policy([finding(source="text", spans=[[0, 3]], bbox=None)])
    check_choice(caption, "skip")
    check_choice(caption, "remove")


# ---------------------------------------------------------------- 재생성 후보 (G_i)
def regen(image_size=SIZE, **changes):
    [out] = apply_policy([finding(**changes)], image_size)
    return out


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "store_name", "source": "vlm"},  # SIGNBOARD: 간판 전체 box
        {"type": "building_sign", "source": "vlm"},
        {"type": "pharmacy_name", "source": "vlm"},
        {"type": "store_name", "source": "ocr"},  # SHOP_NAME: 글자 box
        {"type": "pharmacy_name", "source": "ocr", "risk_level": "high"},
        {"type": "school_sign"},  # 조건부
        {"type": "apt_sign"},
        {"type": "bus_stop"},
        {"type": "other", "category": "location"},
    ],
)
def test_regen_candidates_get_regenerate_but_still_default_to_mask(changes):
    changes = {"risk_level": "medium"} | changes
    out = regen(**changes)
    assert out["policy"]["regenerate_candidate"] is True
    assert "regenerate" in out["policy"]["allowed_actions"]
    # 후보여도 추천 보호는 바뀌지 않는다 (간판은 가림)
    assert out["action"] == regen(None, **changes)["action"]
    if changes.get("source", "vlm") == "vlm":
        assert out["action"] == "mask"


@pytest.mark.parametrize(
    "changes",
    [
        {"type": "face", "risk_level": "medium"},  # FACE 제외
        {"type": "license_plate"},
        {"type": "rrn", "source": "ocr"},
        {"type": "account_number", "source": "ocr"},
        {"type": "id_card"},
        {"type": "delivery_label"},
        {"type": "tracking_number", "source": "ocr"},
        {"type": "other", "category": "lifestyle", "risk_level": "medium"},  # 위치 단서 아님
        {"type": "store_name", "source": "text", "spans": [[0, 3]], "bbox": None},  # 캡션
        {"type": "store_name", "risk_level": "low"},  # L_i < medium
        {"type": "school_sign", "watch_item": "military"},  # locked
    ],
)
def test_non_candidates_never_get_regenerate(changes, monkeypatch):
    monkeypatch.setattr(protect, "RULE_FLOOR_WATCH", {"military"})
    out = regen(**changes)
    assert not out["policy"].get("regenerate_candidate")
    assert "regenerate" not in out["policy"].get("allowed_actions", [])


@pytest.mark.parametrize(
    ("bbox", "image_size", "ok"),
    [
        ([10, 10, 26, 26], (100, 100), True),  # 짧은 변 16px 경계
        ([10, 10, 25, 40], (100, 100), False),  # 짧은 변 15px
        ([0, 0, 40, 100], (100, 100), True),  # 면적 40% 경계
        ([0, 0, 41, 100], (100, 100), False),  # 41%
        ([-1, 10, 30, 40], (100, 100), False),  # 사진 밖
        ([80, 10, 101, 40], (100, 100), False),
        ([30, 30, 20, 40], (100, 100), False),  # 뒤집힌 좌표
        ([10, 10, 40, 40], None, False),  # 사진 크기 모름
        (None, (100, 100), False),
    ],
)
def test_region_quality_q_i(bbox, image_size, ok):
    out = regen(image_size, type="store_name", risk_level="medium", bbox=bbox)
    assert out["policy"]["regenerate_candidate"] is ok


def test_conditional_types_use_smaller_area_limit():
    box_30pct = [0, 0, 30, 100]
    assert regen((100, 100), type="store_name", bbox=box_30pct)["policy"]["regenerate_candidate"]
    for changes in ({"type": "school_sign"}, {"type": "other", "category": "location"}):
        out = regen((100, 100), bbox=box_30pct, risk_level="medium", **changes)
        assert out["policy"]["regenerate_candidate"] is False
    out = regen((100, 100), type="bus_stop", bbox=[0, 0, 25, 100], risk_level="medium")
    assert out["policy"]["regenerate_candidate"] is True  # 25% 경계


def test_regen_thresholds_are_tunable(monkeypatch):
    monkeypatch.setattr(protect, "REGEN_MIN_SIDE_PX", 32)
    assert not regen(type="store_name", bbox=[20, 20, 60, 40])["policy"]["regenerate_candidate"]
    monkeypatch.setattr(protect, "REGEN_MIN_SIDE_PX", 16)
    monkeypatch.setattr(protect, "REGEN_MIN_LEVEL", "high")
    assert not regen(type="store_name", risk_level="medium")["policy"]["regenerate_candidate"]
    assert regen(type="store_name", risk_level="high")["policy"]["regenerate_candidate"]


# ---------------------------------------------------------------- 렌더
def pixels(findings, patches=None):
    media = tf._transform(
        Media(image=photo(), image_ext="png"), Assessment(True, 0.9, findings), patches
    )
    image = open_image(media.image)
    return {
        "outside": image.getpixel((5, 5)),
        "left": image.getpixel((21, 25)),
        "center": image.getpixel((40, 25)),
    }


def test_render_matches_browser_preview_rules():
    """프론트 media.ts renderPhoto 와 같은 결과: 바깥 보존, 가림 단색, 부분마스킹은 지정 범위만."""
    for action in ["mask", "blur", "mosaic", "partialMask", "remove", "skip"]:
        extra = {"mask_box": [0.25, 0, 0.75, 1]} if action == "partialMask" else {}
        result = pixels([finding(action=action, **extra)])
        assert result["outside"] == GREEN, action
        if action == "mask":
            assert result["center"] == tf.MASK_COLOR
        if action == "partialMask":
            assert result["left"] == RED and result["center"] == tf.MASK_COLOR
        if action == "skip":
            assert result["center"] == RED
        if action == "remove":
            assert result["center"] == GREEN  # 주변 배경으로 채움
    left = pixels([finding(action="partialMask", mask_box=[0, 0, 0.25, 1])])
    assert left["left"] == tf.MASK_COLOR and left["center"] == RED


def test_blur_and_mosaic_change_the_region_without_touching_outside():
    image = Image.new("RGB", (100, 80), GREEN)
    for x in range(20, 60):
        for y in range(20, 40):
            image.putpixel((x, y), RED if (x + y) % 2 else (0, 0, 255))
    media = Media(image=png_bytes(image), image_ext="png")
    for action in ["blur", "mosaic"]:
        out = open_image(
            tf._transform(media, Assessment(True, 0.9, [finding(action=action)])).image
        )
        assert out.getpixel((5, 5)) == GREEN and out.getpixel((61, 25)) == GREEN
        assert out.getpixel((40, 30)) not in {RED, (0, 0, 255)}


def test_generated_patch_is_pasted_and_missing_patch_falls_back_to_mask():
    patch = png_bytes(Image.new("RGB", (8, 4), (1, 2, 3)))
    patched = pixels([finding(action="regenerate", generation_id="g1")], {"g1": patch})
    assert patched["center"] == (1, 2, 3) and patched["outside"] == GREEN
    fallback = pixels([finding(action="regenerate", generation_id="expired")], {})
    assert fallback["center"] == tf.MASK_COLOR


def test_restorative_edits_cannot_overwrite_opaque_protection():
    patch = png_bytes(Image.new("RGB", (8, 4), (1, 2, 3)))
    findings = [
        finding(entity_id="a", action="mask", bbox=[20, 20, 60, 40]),
        finding(entity_id="b", action="regenerate", generation_id="g", bbox=[10, 10, 70, 50]),
    ]
    result = pixels(findings, {"g": patch})
    assert result["center"] == tf.MASK_COLOR and result["outside"] == GREEN


def test_partial_mask_needs_a_selection_and_inpainting_needs_background():
    with pytest.raises(ValueError, match="partial mask"):
        pixels([finding(action="partialMask")])
    with pytest.raises(ValueError, match="background"):
        pixels([finding(action="remove", bbox=[0, 0, 100, 80])])
    with pytest.raises(ValueError, match="unsupported"):
        pixels([finding(action="sticker")])


# ---------------------------------------------------------------- Phase 2 연동
async def test_user_edits_reach_the_renderer_and_policy_is_enforced(monkeypatch):
    initial = apply_policy(
        [
            finding(),
            finding(
                entity_id="f1",
                type="other",
                category="lifestyle",
                risk_level="low",
                bbox=[70, 50, 90, 70],
            ),
        ]
    )
    assessment = Assessment(True, 0.9, initial)
    edited = _apply_user_choices(
        assessment,
        [
            UserChoice(idx=0, action="blur", bbox=[10, 10, 70, 50]),
            UserChoice(idx=1, action="skip"),
        ],
        (100, 80),
    )
    assert edited.findings[0]["bbox"] == [10, 10, 70, 50] and edited.findings[0]["action"] == "blur"
    assert edited.findings[1]["action"] == "skip"
    with pytest.raises(PolicyViolation):
        _apply_user_choices(assessment, [UserChoice(idx=0, action="skip")], (100, 80))
    with pytest.raises(ValueError):
        _apply_user_choices(assessment, [UserChoice(idx=5, action="mask")], (100, 80))

    monkeypatch.setattr("stages.risk.detect", AsyncMock(return_value=[]))
    result = await run_pipeline(
        Media(image=photo(), image_ext="png"),
        max_iterations=3,
        report=AsyncMock(),
        user_choices=[UserChoice(idx=0, action="partialMask", mask_box=[0.5, 0, 1, 1])],
        initial_findings=apply_policy([finding(type="name", risk_level="low")]),
    )
    out = open_image(result.output.image)
    assert out.getpixel((25, 25)) == RED and out.getpixel((50, 25)) == tf.MASK_COLOR


async def test_policy_violation_fails_the_job_with_its_message(monkeypatch):
    storage, redis = FakeStorage(), FakeRedis()
    storage.objects["originals/x.png"] = photo()
    from worker.contract import Finding, JobMessage

    msg = JobMessage(
        job_id=uuid.uuid4(),
        execution_id=uuid.uuid4(),
        input_type="image",
        object_key="originals/x.png",
        created_at=datetime.now(UTC),
        user_choices=[UserChoice(idx=0, action="skip")],
        initial_findings=[Finding.model_validate(f) for f in apply_policy([finding()])],
    )
    from worker.main import handle

    await handle(redis, storage, Settings(), "1-0", {"payload": msg.model_dump_json()})
    [sent] = redis.results
    payload = json.loads(sent["payload"])
    assert payload["status"] == "FAILED" and "그대로 둘 수 없어요" in payload["error"]


async def test_phase1_report_carries_policy_fields(monkeypatch):
    monkeypatch.setattr("stages.risk.detect", AsyncMock(return_value=[finding()]))
    storage, redis = FakeStorage(), FakeRedis()
    storage.objects["originals/x.png"] = photo()
    from worker.contract import JobMessage

    msg = JobMessage(
        job_id=uuid.uuid4(),
        execution_id=uuid.uuid4(),
        input_type="image",
        object_key="originals/x.png",
        created_at=datetime.now(UTC),
    )
    result = await process(redis, storage, Settings(), msg)
    [f] = result.findings
    dumped = f.model_dump()
    assert dumped["category"] == "object" and dumped["policy"]["level"] == "high"
    assert dumped["risk_category"] == "identity"


async def test_phase2_loads_patches_stored_by_the_api():
    from worker.queue import load_patches

    redis = FakeRedis()
    patch = png_bytes(Image.new("RGB", (8, 4), (1, 2, 3)))
    redis.values[generation_key("ab12")] = base64.b64encode(patch).decode()
    redis.values[generation_key("bad")] = "not base64!"
    assert await load_patches(redis, ["ab12", "bad", "missing", "", "ab12"]) == {"ab12": patch}


def test_verify_knows_new_edit_actions():
    """④ 재검증이 모자이크·부분 덮기·배경 재생성도 처리 이력으로 읽고, 남으면 가림을 추천한다."""
    from stages.verify import compare

    before = [finding(action="mosaic"), finding(entity_id="f1", type="name", action="regenerate")]
    after = [finding(entity_id="g0", risk_level="medium", reason="윤곽이 보여요")]
    result = compare(before, after)
    [mosaic, regenerated] = result["items"]
    assert mosaic["status"] == "remaining" and mosaic["recommend"] == "mask"
    assert "모자이크 처리했지만" in mosaic["message"] and "덮기로 다시" in mosaic["message"]
    assert regenerated["status"] == "removed" and "배경 재생성 처리했고" in regenerated["message"]

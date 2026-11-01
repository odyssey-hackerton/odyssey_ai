"""③ 변환 (에이전트3 렌더러): 사용자가 고른 보호 방법을 실제 픽셀·본문에 적용한다.

프론트 편집 화면(frontend/src/media.ts renderPhoto)의 미리보기와 같은 규칙으로 그린다.
- mask:        불투명 단색 (20, 33, 56)
- blur:        영역을 max(w,h)/768 비율로 줄인 뒤 반경 16/scale 박스 블러, 다시 키움
- mosaic:      16px 격자로 줄였다가 최근접 확대
- partialMask: bbox 안에서 사용자가 지정한 상대 사각형(mask_box, 0~1)만 가림
- remove:      텍스트제거. OpenCV Telea 인페인팅 (주변 24px 포함, 최대 512px 로 축소 처리)
- regenerate:  워커가 Gemini 로 만들어 둔 패치(generation_id)를 영역에 합성. 없으면 가림
- skip:        그대로
복원형(remove, regenerate)을 먼저 그려서 불투명 보호를 덮어쓰지 못하게 한다.
EXIF 는 PNG 재인코딩으로 제거되고, 글은 remove=삭제, 그 외 "[가림]" 치환(겹침 병합)이다.
"""

import asyncio
import math
from io import BytesIO
from typing import Any

from PIL import Image, ImageFilter

from stages.images import open_image, png_bytes
from stages.types import Assessment, Media

MASK_COLOR = (20, 33, 56)
STRENGTH = 16  # 블러 반경·모자이크 격자 (프론트와 같은 고정값)
INPAINT_PAD = 24
INPAINT_MAX = 512
RESTORATIVE = {"remove", "regenerate"}
IMAGE_ACTIONS = {"mask", "blur", "mosaic", "partialMask", "remove", "regenerate", "skip"}
TEXT_ACTIONS = {"mask", "remove", "skip"}


def _pixel_box(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("image finding has no valid rectangle")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
        raise ValueError("invalid rectangle coordinates")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("rectangle outside image")
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def _partial_box(bbox: list[float], mask_box: Any) -> list[float]:
    if (
        not isinstance(mask_box, list)
        or len(mask_box) != 4
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in mask_box)
    ):
        raise ValueError("partial mask has no selection")
    px1, py1, px2, py2 = (min(1.0, max(0.0, float(v))) for v in mask_box)
    if not (px1 < px2 and py1 < py2):
        raise ValueError("partial mask has no selection")
    x1, y1, x2, y2 = bbox
    return [x1 + (x2 - x1) * px1, y1 + (y2 - y1) * py1, x1 + (x2 - x1) * px2, y1 + (y2 - y1) * py2]


def blur_region(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    region = image.crop(box)
    w, h = region.size
    scale = max(1.0, max(w, h) / 768)
    small = region.resize((max(1, math.ceil(w / scale)), max(1, math.ceil(h / scale))))
    small = small.filter(ImageFilter.BoxBlur(max(1, STRENGTH / scale)))
    image.paste(small.resize((w, h), Image.Resampling.BILINEAR), box)


def mosaic_region(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    region = image.crop(box)
    w, h = region.size
    small = region.resize(
        (max(1, math.ceil(w / STRENGTH)), max(1, math.ceil(h / STRENGTH))), Image.Resampling.BOX
    )
    image.paste(small.resize((w, h), Image.Resampling.NEAREST), box)


def inpaint_region(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    """선택 영역을 주변 배경으로 채운다 (프론트 inpaint.ts 와 같은 Telea, 24px 패딩, 512px 상한)."""
    import cv2
    import numpy as np

    width, height = image.size
    x, y, x2, y2 = box
    if x == 0 and y == 0 and x2 == width and y2 == height:
        raise ValueError("inpainting needs surrounding background")
    left, top = max(0, x - INPAINT_PAD), max(0, y - INPAINT_PAD)
    right, bottom = min(width, x2 + INPAINT_PAD), min(height, y2 + INPAINT_PAD)
    crop = image.crop((left, top, right, bottom))
    cw, ch = crop.size
    scale = min(1.0, INPAINT_MAX / max(cw, ch))
    sw, sh = max(1, round(cw * scale)), max(1, round(ch * scale))
    small = np.asarray(crop.resize((sw, sh), Image.Resampling.BILINEAR).convert("RGB"))
    mask = np.zeros((sh, sw), dtype=np.uint8)
    sx, sy = sw / cw, sh / ch
    mask[
        math.floor((y - top) * sy) : math.ceil((y2 - top) * sy),
        math.floor((x - left) * sx) : math.ceil((x2 - left) * sx),
    ] = 255
    filled = cv2.inpaint(np.ascontiguousarray(small), mask, 3, cv2.INPAINT_TELEA)
    restored = Image.fromarray(filled).resize((cw, ch), Image.Resampling.BILINEAR)
    # 바깥 픽셀은 보존하고 선택 영역만 합성한다.
    image.paste(restored.crop((x - left, y - top, x2 - left, y2 - top)), (x, y))


def paste_patch(image: Image.Image, box: tuple[int, int, int, int], patch: bytes) -> None:
    with Image.open(BytesIO(patch)) as source:
        source.load()
        generated = source.convert("RGB")
    x, y, x2, y2 = box
    image.paste(generated.resize((x2 - x, y2 - y), Image.Resampling.LANCZOS), (x, y))


def render_image(
    image: Image.Image, findings: list[dict[str, Any]], patches: dict[str, bytes]
) -> None:
    ordered = sorted(findings, key=lambda f: f.get("action") not in RESTORATIVE)
    for finding in ordered:
        action = finding.get("action", "mask")
        if action not in IMAGE_ACTIONS:
            raise ValueError("unsupported edit action")
        if action == "skip":
            continue
        bbox = finding.get("bbox")
        if not isinstance(bbox, list):
            raise ValueError("image finding has no valid rectangle")
        if action == "partialMask":
            bbox = _partial_box(bbox, finding.get("mask_box"))
        box = _pixel_box(bbox, image.width, image.height)
        if action == "regenerate":
            patch = patches.get(finding.get("generation_id") or "")
            if patch is None:
                action = "mask"  # 생성 결과가 없으면 안전한 쪽(가림)으로
            else:
                paste_patch(image, box, patch)
                continue
        if action == "blur":
            blur_region(image, box)
        elif action == "mosaic":
            mosaic_region(image, box)
        elif action == "remove":
            inpaint_region(image, box)
        else:
            image.paste(MASK_COLOR, box)


def render_text(text: str, findings: list[dict[str, Any]]) -> str:
    replacements = []
    for finding in findings:
        action = finding.get("action", "mask")
        if action not in TEXT_ACTIONS:
            raise ValueError("unsupported edit action")
        if action == "skip":
            continue
        spans = finding.get("spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("text finding has no valid spans")
        for span in spans:
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= len(text)
            ):
                raise ValueError("text finding has invalid spans")
            replacements.append((*span, "" if action == "remove" else "[가림]"))
    # 겹친 범위를 합친 뒤 뒤에서부터 편집하여 원문 좌표를 유지한다.
    merged = []
    for start, end, replacement in sorted(replacements):
        if merged and start < merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (previous[0], max(previous[1], end), previous[2] or replacement)
        else:
            merged.append((start, end, replacement))
    for start, end, replacement in reversed(merged):
        text = text[:start] + replacement + text[end:]
    return text


def _transform(
    media: Media, assessment: Assessment, patches: dict[str, bytes] | None = None
) -> Media:
    image = open_image(media.image) if media.image is not None else None
    text = media.text
    image_findings, text_findings = [], []
    for finding in assessment.findings:
        if finding.get("action", "mask") == "skip":
            continue
        if finding.get("source") == "exif":
            continue  # PNG 재인코딩 시 메타데이터가 제거된다.
        if finding.get("source") == "text":
            if text is None:
                raise ValueError("text finding has no valid spans")
            text_findings.append(finding)
        else:
            if image is None:
                raise ValueError("image finding has no valid rectangle")
            image_findings.append(finding)
    if image is not None:
        render_image(image, image_findings, patches or {})
    if text is not None:
        text = render_text(text, text_findings)
    return Media(
        image=png_bytes(image) if image else None, image_ext="png" if image else None, text=text
    )


async def transform(
    media: Media, assessment: Assessment, patches: dict[str, bytes] | None = None
) -> Media:
    return await asyncio.to_thread(_transform, media, assessment, patches)

"""③ 변환: 위험한 부분을 블러·마스킹·제거한다."""

import asyncio
import math

from PIL import ImageFilter

from stages.images import open_image, png_bytes
from stages.types import Assessment, Media


def _transform(media: Media, assessment: Assessment) -> Media:
    image = open_image(media.image) if media.image is not None else None
    text = media.text
    replacements = []
    for finding in assessment.findings:
        action = finding.get("action", "mask")
        if action not in {"blur", "mask", "remove", "skip"}:
            raise ValueError("unsupported edit action")
        if action == "skip":
            continue
        if finding.get("source") == "exif":
            continue  # PNG 재인코딩 시 메타데이터가 제거된다.
        if finding.get("source") == "agent2":
            continue  # ② 조합 정보는 자기 영역이 없다. 가릴 곳은 sources 의 ① 항목이다.
        if finding.get("source") == "text":
            spans = finding.get("spans")
            if not isinstance(spans, list) or not spans or text is None:
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
            continue
        bbox = finding.get("bbox")
        if image is None or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("image finding has no valid rectangle")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
            raise ValueError("invalid rectangle coordinates")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("rectangle outside image")
        box = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        if action == "blur":
            region = image.crop(box)
            image.paste(region.filter(ImageFilter.GaussianBlur(max(12, min(region.size) / 4))), box)
        else:
            # shortcut: remove도 불투명 가림이다. 자연스러운 제거는 인페인팅 API 연결 시 교체한다.
            image.paste((20, 33, 56), box)
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
    return Media(
        image=png_bytes(image) if image else None, image_ext="png" if image else None, text=text
    )


async def transform(media: Media, assessment: Assessment) -> Media:
    return await asyncio.to_thread(_transform, media, assessment)

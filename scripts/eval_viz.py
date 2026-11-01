"""평가·데모 스크립트 공용: 처리 단계별 사진 저장 (①② 탐지 박스 → ③ 처리 결과 → ④ 재검증 상태)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from stages.images import open_image

STATUS_COLOR = {"removed": "lime", "weakened": "green", "remaining": "red", "kept": "gray"}
LEVEL_COLOR = {"high": "red", "medium": "orange", "low": "yellow"}
_FONTS = [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _font(size: int):
    for p in _FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size), True
            except OSError:
                continue
    return ImageFont.load_default(), False


def _draw(img: Image.Image, boxes) -> Image.Image:
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    size = max(14, img.width // 60)
    font, korean = _font(size)
    for bbox, color, label, ascii_label in boxes:
        d.rectangle(bbox, outline=color, width=max(3, img.width // 300))
        text = label if korean else ascii_label
        x, y = bbox[0] + 2, max(0, bbox[1] - size - 4)
        d.rectangle(d.textbbox((x, y), text, font=font), fill="black")
        d.text((x, y), text, fill=color, font=font)
    return img


def save_stages(
    out_dir: Path, stem: str, original: bytes, findings, output: bytes | None, verification=None
):
    """{stem}_1_detect.jpg / _2_output.jpg / _3_verify.jpg 를 out_dir 에 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    det = [
        (
            f["bbox"],
            LEVEL_COLOR.get(f.get("risk_level"), "yellow"),
            f"{f['entity_id']} {f.get('title') or f.get('type')} ({f.get('risk_score', '-')})",
            f"{f['entity_id']} {f.get('type')} ({f.get('risk_score', '-')})",
        )
        for f in findings
        if f.get("bbox")
    ]
    _draw(open_image(original), det).save(out_dir / f"{stem}_1_detect.jpg", quality=92)
    if not output:
        return
    out_img = open_image(output)
    out_img.convert("RGB").save(out_dir / f"{stem}_2_output.jpg", quality=95)
    v = verification or {}
    boxes = [
        (
            it["bbox"],
            STATUS_COLOR.get(it["status"], "white"),
            f"{it['entity_id']} {it['status']}",
            f"{it['entity_id']} {it['status']}",
        )
        for it in v.get("items", [])
        if it.get("bbox")
    ]
    boxes += [
        (n["bbox"], "magenta", f"NEW {n.get('title') or n.get('type')}", f"NEW {n.get('type')}")
        for n in v.get("new_items", [])
        if n.get("bbox")
    ]
    _draw(out_img, boxes).save(out_dir / f"{stem}_3_verify.jpg", quality=92)

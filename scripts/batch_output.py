# ruff: noqa: E501, ASYNC240  (단독 실행 도우미 스크립트)
"""사진 → ①② → ③ 변환 결과만 저장 (④ 재검증 없음, 빠름). 결과: out_eval/outputs/<이름>_output.jpg

uv run python scripts/batch_output.py                          # eval_data/images/KakaoTalk_*.jpg
uv run python scripts/batch_output.py a.jpg b.jpg 사진폴더/      # 파일·폴더 여러 개
uv run python scripts/batch_output.py 사진폴더/ -o 결과폴더
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 레포 루트

from stages.images import open_image  # noqa: E402
from stages.types import Media  # noqa: E402
from worker.pipeline import (
    run_detection_phase,  # noqa: E402
    transform,  # noqa: E402
)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


async def quiet(*_a, **_k):
    return None


def inputs(args: list[str]):
    if not args:
        return sorted(Path("eval_data/images").glob("KakaoTalk_*"))
    out = []
    for a in args:
        p = Path(a)
        out += sorted(x for x in p.glob("*") if x.suffix.lower() in EXTS) if p.is_dir() else [p]
    return out


async def main():
    args = sys.argv[1:]
    dst = Path("out_eval/outputs")
    if "-o" in args:
        i = args.index("-o")
        dst = Path(args[i + 1])
        args = args[:i] + args[i + 2 :]
    dst.mkdir(parents=True, exist_ok=True)
    items = inputs(args)
    print(f"사진 {len(items)}장 → {dst}")
    for n, p in enumerate(items, 1):
        try:
            raw = p.read_bytes()
            media = Media(image=raw, image_ext=p.suffix.lstrip(".").lower().replace("jpeg", "jpg"))
            det = await run_detection_phase(media, report=quiet)
            before = det.before
            for f in before.findings:  # ② 추천 처리 그대로 (EXIF 는 remove)
                if f.get("source") == "exif" and f.get("action") == "blur":
                    f["action"] = "remove"
            out = await transform(Media(image=raw, image_ext=media.image_ext), before)
            if out.image:
                open_image(out.image).convert("RGB").save(dst / f"{p.stem}_output.jpg", quality=95)
            print(
                f"[{n}/{len(items)}] {p.name}: 항목 {len(before.findings)}개 → {p.stem}_output.jpg"
            )
        except Exception as e:  # 한 장 실패해도 계속
            print(f"[{n}/{len(items)}] {p.name}: 실패 {e!r}"[:300])


asyncio.run(main())

# ruff: noqa: E501, ASYNC240  (단독 실행 도우미 스크립트)
"""사진 한 장 → ①② 분석 → ③ 처리(② 추천 그대로) → ④ 재검증 → output.jpg

uv run python scripts/make_output.py test.jpg              # → output.jpg, output_result.json
uv run python scripts/make_output.py test.jpg out.jpg "캡션 문구"
ROUNDS=3 uv run python scripts/make_output.py test.jpg     # 자동 반복: 재검증에서 새로 찾은 것도 다시 처리 (최대 3회)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 레포 루트

from stages.images import open_image
from stages.types import Media
from worker.contract import UserChoice
from worker.pipeline import run_detection_phase, run_pipeline


async def report(status, iteration=0, message=None):
    print(f"  [{status}] {message or ''}")


async def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output.jpg")
    caption = sys.argv[3] if len(sys.argv) > 3 else None
    raw = src.read_bytes()
    ext = src.suffix.lstrip(".").lower().replace("jpeg", "jpg")

    det = await run_detection_phase(Media(image=raw, image_ext=ext, text=caption), report=report)
    findings = det.before.findings
    print(f"처리 전 risk={det.before.risk:.3f}, 찾은 항목 {len(findings)}개")
    choices = []
    for i, f in enumerate(findings):
        act = f.get("action", "mask")
        if f.get("source") == "exif" and act == "blur":
            act = "remove"
        choices.append(UserChoice(idx=i, action=act))
        print(f"  {f['entity_id']} {f.get('type')} {f.get('risk_level')} → {act}")

    rounds = int(os.environ.get("ROUNDS", "1"))
    if rounds > 1:  # 사용자 선택 없이 파이프라인 자동 반복 (위험이 남으면 ③→④ 다시)
        res = await run_pipeline(
            Media(image=raw, image_ext=ext, text=caption), max_iterations=rounds, report=report
        )
    else:
        res = await run_pipeline(
            Media(image=raw, image_ext=ext, text=caption),
            max_iterations=1,
            report=report,
            user_choices=choices,
            initial_findings=findings,
        )
    print(f"반복 {res.iterations}회")
    if res.output.image:
        open_image(res.output.image).convert("RGB").save(dst, quality=95)
    from eval_viz import save_stages

    save_stages(
        dst.parent / f"{dst.stem}_steps",
        src.stem,
        raw,
        findings,
        res.output.image,
        res.verification,
    )
    v = res.verification or {}
    print(f"처리 후 risk={res.after.risk:.3f} clean={v.get('clean')} → {dst}")
    t = v.get("risk_total") or {}
    if t:
        pct = f"{t['reduction'] * 100:.0f}% 감소" if t.get("reduction") is not None else "-"
        print(
            f"위험 총량 {t['before']} → {t['after']} ({pct})  "
            f"처음 찾은 {v.get('before')}개 중 {v.get('removed')}개 제거, 새로 찾은 {v.get('new')}개"
        )
    dst.with_name(dst.stem + "_result.json").write_text(
        json.dumps(
            {"before": findings, "verification": v, "caption_after": res.output.text},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "utf-8",
    )


asyncio.run(main())

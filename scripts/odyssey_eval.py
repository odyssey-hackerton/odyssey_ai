# ruff: noqa: E501, ASYNC230  (단독 실행 도우미 스크립트)
"""⓪①② 성능 평가 올인원 (DIPA2 샘플 20장). odyssey_ai 레포 루트에서 실행.

  uv run python scripts/odyssey_eval.py                 # 이미지 받기 → 박스 확인 그림 → Gemini 3회 → Qwen 1회 → 비교표
  uv run python scripts/odyssey_eval.py --only gemini   # 하나만
  uv run python scripts/odyssey_eval.py --runs 3 --qwen-runs 3 --skip-context

결과: eval_data/images/_check/ (정답 박스 그림), out_eval/<gemini|qwen>/summary.md, out_eval/compare.md
Qwen 은 .env 를 안 바꿔도 된다 (이 스크립트가 VLM_PROVIDER=qwen 으로 따로 실행). QWEN_BASE_URL 은 .env 값 사용.
채점 기준
- ① 물체: 정답(라벨러 4명 중 2명 이상이 위협으로 고른 물체)과 IoU≥0.5 매칭 (PASCAL VOC·COCO). recall / precision.
  precision 은 하한값: DIPA 는 위협이라 고른 물체만 라벨이 있어서, 맞게 찾았는데 라벨이 없는 물체도 오답으로 센다.
- ① 카테고리: DIPA 정보 유형 → 우리 카테고리 허용 집합에 들면 정답 (README 표).
- ② 위험도: ② risk_score(0~16) 와 DIPA 심각도 평균(7점 Likert)의 Spearman 순위 상관. 구간 임계값 없음.
- ② 안정성: 같은 물체의 3회 risk_level 이 모두 같은 비율, risk_score 표준편차.
- 위험 낮은 사진(최대 심각도 ≤2.5): ② risky=False 비율.
- ⓪: 3회 watchlist 항목의 Jaccard (실행 간 겹침). 관련성은 csv 에 사람이 표시.
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 레포 루트 (stages, worker)
DATA = Path("eval_data")
IMAGES = DATA / "images"
# 정답: DIPA2 (Xu et al., IMWUT 2024) 에서 뽑은 20장, 물체 단위로 묶음. CC BY 4.0
GT = json.loads(Path(__file__).with_name("dipa2_sample20.json").read_text("utf-8"))


# ---------------------------------------------------------------- 1. 이미지 받기 + 박스 확인
def image_urls(name, source):
    stem = name.rsplit(".", 1)[0]
    if source == "LVIS":
        return [
            f"http://images.cocodataset.org/{s}/{int(stem):012d}.jpg"
            for s in ("train2017", "val2017")
        ]
    return [
        f"https://s3.amazonaws.com/open-images-dataset/{s}/{stem}.jpg"
        for s in ("train", "validation", "test")
    ]


def download():
    IMAGES.mkdir(parents=True, exist_ok=True)
    fail = []
    for name, g in GT.items():
        dst = IMAGES / name
        if dst.exists():
            continue
        for u in image_urls(name, g["source"]):
            try:
                with urllib.request.urlopen(u, timeout=30) as r:
                    dst.write_bytes(r.read())
                break
            except Exception:
                continue
        else:
            fail.append(name)
    print(
        f"이미지 {len(GT) - len(fail)}/{len(GT)}장 준비", "실패: " + ", ".join(fail) if fail else ""
    )


def draw_check():
    from PIL import Image, ImageDraw

    out = IMAGES / "_check"
    out.mkdir(exist_ok=True)
    for name, g in GT.items():
        p = IMAGES / name
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        for o in g["objects"]:
            x, y, w, h = o["bbox_xywh"]
            c = "red" if o["votes"] >= 2 else "gray"
            d.rectangle([x, y, x + w, y + h], outline=c, width=4)
            d.text((x + 4, y + 4), f"{o['label']} v{o['votes']} s{o['severity_mean']}", fill=c)
        im.save(out / name)
    print("정답 박스 그림:", out, "(빨간 박스가 물체에 안 맞으면 알려줘)")


# ---------------------------------------------------------------- 2. 평가 (provider 하나, 별도 프로세스)
ALLOWED = {
    "personal_info": {"identity", "attributes", "contact_finance"},
    "location": {"location"},
    "preferences": {"lifestyle", "sensitive"},
    "social_circle": {"relationships"},
    "others_private": {"identity", "contact_finance", "sensitive"},
}
PRIMARY = {  # 정답 물체를 ② 에 바로 넣을 때 쓰는 카테고리 (정보 유형 최다표)
    "personal_info": "identity",
    "location": "location",
    "preferences": "lifestyle",
    "social_circle": "relationships",
    "others_private": "identity",
    "other": "lifestyle",
}
IOU_MATCH = 0.5
MIN_VOTES = 2


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def gt_objects(g):
    out = []
    for i, o in enumerate(g["objects"]):
        if o["votes"] < MIN_VOTES:
            continue
        types = {k: v for k, v in o["info_types"].items() if k != "other"}
        out.append(
            {
                **o,
                "gid": f"g{i}",
                "box": xyxy(o["bbox_xywh"]),
                "top_type": max(types, key=types.get) if types else "other",
                "allowed": set().union(*(ALLOWED[t] for t in types)) if types else set(),
            }
        )
    return out


def match(gts, finds):
    """IoU 큰 쌍부터 1:1 매칭."""
    pairs = sorted(
        (
            (iou(g["box"], f["bbox"]), gi, fi)
            for gi, g in enumerate(gts)
            for fi, f in enumerate(finds)
            if f.get("bbox")
        ),
        reverse=True,
    )
    used_g, used_f, out = set(), set(), {}
    for v, gi, fi in pairs:
        if v < IOU_MATCH:
            break
        if gi in used_g or fi in used_f:
            continue
        used_g.add(gi)
        used_f.add(fi)
        out[gi] = fi
    return out


def spearman(x, y):
    if len(x) < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def media_of(path: Path):
    ext = path.suffix.lstrip(".").lower().replace("jpeg", "jpg")
    return Media(image=path.read_bytes(), image_ext=ext)


async def run_context(country, day, runs):
    out = []
    for _ in range(runs):
        ctx_mod._cache.clear()
        c = await build_context(country, day)
        out.append(c)
    return out


async def run_image(name, g, media, context, runs):
    gts = gt_objects(g)
    rec = {"image": name, "max_severity": g["max_severity"], "e2e": [], "oracle": []}
    for _ in range(runs):
        t = time.monotonic()
        try:
            findings = await detect(media, watchlist=context.watchlist if context else None)
            judged, combos = await judge(media, findings, context)
            a = await analyze(media, judged, combos, context)
            rec["e2e"].append(
                {
                    "risky": a.risky,
                    "risk": a.risk,
                    "sec": round(time.monotonic() - t, 1),
                    "findings": [
                        {
                            k: f.get(k)
                            for k in (
                                "entity_id",
                                "type",
                                "category",
                                "bbox",
                                "title",
                                "severity",
                                "likelihood",
                                "risk_score",
                                "risk_level",
                                "reason",
                            )
                        }
                        for f in a.findings
                    ],
                }
            )
        except Exception as e:  # 한 장 실패해도 나머지는 계속
            rec["e2e"].append({"error": repr(e)})
        if not gts:
            continue
        oracle = [
            {
                "entity_id": o["gid"],
                "type": "other",
                "source": "image",
                "bbox": o["box"],
                "category": PRIMARY[o["top_type"]],
                "title": o["label"],
                "description": o["label"],
                "risk_level": "medium",
                "score": 0.5,
            }
            for o in gts
        ]
        try:
            judged, combos = await judge(media, oracle, context)
            a = await analyze(media, judged, combos, context)
            rec["oracle"].append(
                {
                    f["entity_id"]: {
                        "risk_score": f.get("risk_score"),
                        "risk_level": f.get("risk_level"),
                        "reason": f.get("reason"),
                    }
                    for f in a.findings
                }
            )
        except Exception as e:
            rec["oracle"].append({"error": repr(e)})
    return rec, gts


def pct(a, b):
    return round(a / b, 3) if b else None


def score_all(records, gtmap):
    tp = fn = fp = cat_ok = cat_n = 0
    e2e_x, e2e_y, or_x, or_y = [], [], [], []
    stable, sd = [], []
    low_ok = low_n = errors = 0
    for rec in records:
        gts = gtmap[rec["image"]]
        for run in rec["e2e"]:
            if "error" in run:
                errors += 1
                continue
            finds = [f for f in run["findings"] if f.get("bbox")]
            m = match(gts, finds)
            tp += len(m)
            fn += len(gts) - len(m)
            fp += len(finds) - len(m)
            for gi, fi in m.items():
                f, g = finds[fi], gts[gi]
                if g["allowed"]:
                    cat_n += 1
                    cat_ok += f.get("category") in g["allowed"]
                if f.get("risk_score") is not None:
                    e2e_x.append(f["risk_score"])
                    e2e_y.append(g["severity_mean"])
            if rec["max_severity"] <= 2.5:
                low_n += 1
                low_ok += not run["risky"]
        runs = [r for r in rec["oracle"] if "error" not in r]
        errors += len(rec["oracle"]) - len(runs)
        for g in gts:
            vals = [
                r[g["gid"]] for r in runs if g["gid"] in r and r[g["gid"]]["risk_score"] is not None
            ]
            if not vals:
                continue
            or_x.append(statistics.mean(v["risk_score"] for v in vals))
            or_y.append(g["severity_mean"])
            if len(vals) > 1:
                stable.append(len({v["risk_level"] for v in vals}) == 1)
                sd.append(statistics.pstdev(v["risk_score"] for v in vals))
    return {
        "①_recall@IoU0.5": pct(tp, tp + fn),
        "①_precision@IoU0.5(하한)": pct(tp, tp + fp),
        "①_category_acc": pct(cat_ok, cat_n),
        "①_matched": tp,
        "①_gt_total": tp + fn,
        "②_spearman_e2e": spearman(e2e_x, e2e_y),
        "②_n_e2e": len(e2e_x),
        "②_spearman_oracle": spearman(or_x, or_y),
        "②_n_oracle": len(or_x),
        "②_same_level_all_runs": pct(sum(stable), len(stable)),
        "②_score_sd_mean": round(statistics.mean(sd), 2) if sd else None,
        "low_risk_images_not_risky": pct(low_ok, low_n),
        "errors": errors,
    }


def jaccard_runs(contexts):
    sets = [{w.get("item", "").strip() for w in c.watchlist} for c in contexts]
    vals = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :] if a | b]
    return round(statistics.mean(vals), 3) if vals else None


async def evaluate(label, runs, skip_context, country, day):
    global build_context, detect, analyze, iou, judge, ctx_mod, Media
    from stages import context as ctx_mod
    from stages.context import build_context
    from stages.detect import detect
    from stages.risk import analyze, iou, judge
    from stages.types import Media

    out = Path("out_eval") / label
    out.mkdir(parents=True, exist_ok=True)
    summary = {"label": label, "provider": os.environ.get("VLM_PROVIDER"), "runs": runs, "day": day}
    context = None
    t0 = time.monotonic()
    if not skip_context:
        print(f"[{label}] ⓪ 맥락 {runs}회")
        contexts = await run_context(country, date.fromisoformat(day), runs)
        context = contexts[0]
        summary["⓪_searched"] = [c.searched for c in contexts]
        summary["⓪_watchlist_size"] = [len(c.watchlist) for c in contexts]
        summary["⓪_jaccard_between_runs"] = jaccard_runs(contexts)
        with open(out / "context_watchlist.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["run", "item", "category", "why", "issue", "관련있음(1/0) ← 사람이 채움"])
            for i, c in enumerate(contexts):
                for x in c.watchlist:
                    w.writerow(
                        [i, x.get("item"), x.get("category"), x.get("why"), x.get("issue", ""), ""]
                    )
    records, gtmap = [], {}
    with open(out / "raw.jsonl", "w", encoding="utf-8") as raw:
        for n, (name, g) in enumerate(GT.items(), 1):
            path = IMAGES / name
            if not path.exists():
                continue
            print(
                f"[{label}] {n}/{len(GT)} {name} ({g['dominant_type']}, 최대 심각도 {g['max_severity']})"
            )
            rec, gts = await run_image(name, g, media_of(path), context, runs)
            gtmap[name] = gts
            records.append(rec)
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw.flush()
    summary |= score_all(records, gtmap)
    summary["images"] = len(records)
    summary["minutes"] = round((time.monotonic() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    (out / "summary.md").write_text(
        "\n".join(["| 지표 | 값 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]),
        "utf-8",
    )
    print(f"[{label}] 완료 → {out}/summary.md")


def compare():
    rows = {}
    for p in sorted(Path("out_eval").glob("*/summary.json")):
        rows[p.parent.name] = json.loads(p.read_text("utf-8"))
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    lines = ["| 지표 | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    lines += [
        f"| {k} | " + " | ".join(str(r.get(k, "")) for r in rows.values()) + " |" for k in keys
    ]
    Path("out_eval/compare.md").write_text("\n".join(lines), "utf-8")
    print("\n" + "\n".join(lines) + "\n\n저장: out_eval/compare.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["gemini", "qwen"])
    p.add_argument("--runs", type=int, default=3, help="Gemini 반복 횟수")
    p.add_argument("--qwen-runs", type=int, default=1, help="Qwen 반복 횟수 (느려서 기본 1)")
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--country", default="KR")
    p.add_argument("--day", default=date.today().isoformat())
    p.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a._worker:
        asyncio.run(evaluate(a._worker[0], int(a._worker[1]), a.skip_context, a.country, a.day))
        return
    download()
    draw_check()
    plan = [("gemini", a.runs), ("qwen", a.qwen_runs)]
    for provider, runs in plan:
        if a.only and provider != a.only:
            continue
        cmd = [
            sys.executable,
            __file__,
            "--_worker",
            provider,
            str(runs),
            "--country",
            a.country,
            "--day",
            a.day,
        ]
        if a.skip_context:
            cmd.append("--skip-context")
        r = subprocess.run(
            cmd, env={**os.environ, "VLM_PROVIDER": provider, "PYTHONPATH": os.getcwd()}
        )
        if r.returncode:
            print(f"[{provider}] 실패 (코드 {r.returncode}), 다음으로 넘어감")
    compare()


if __name__ == "__main__":
    main()

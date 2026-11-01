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


async def run_verify(media, findings, stem=None):
    """④: ② 추천 처리 그대로 ③ → ④ 1회 (scripts/make_output.py 와 같은 흐름)."""
    from worker.contract import UserChoice
    from worker.pipeline import run_pipeline

    async def quiet(*_a, **_k):
        return None

    choices = []
    for i, f in enumerate(findings):
        act = f.get("action", "mask")
        if f.get("source") == "exif" and act == "blur":
            act = "remove"
        choices.append(UserChoice(idx=i, action=act))
    res = await run_pipeline(
        Media(image=media.image, image_ext=media.image_ext),
        max_iterations=1,
        report=quiet,
        user_choices=choices,
        initial_findings=findings,
    )
    v = res.verification or {}
    if stem:
        from eval_viz import save_stages

        save_stages(
            Path("out_eval") / "images",
            f"{os.environ.get('VLM_PROVIDER')}_{stem}",
            media.image,
            findings,
            res.output.image,
            v,
        )
    keys = ("before", "removed", "weakened", "remaining", "kept", "new", "clean", "risk_total")
    return {k: v.get(k) for k in keys} | {
        "risk_before": res.before.risk,
        "risk_after": res.after.risk,
    }


async def run_image(name, g, media, context, runs, with_verify=True):
    gts = gt_objects(g)
    rec = {"image": name, "max_severity": g["max_severity"], "e2e": [], "oracle": []}
    first = None
    for _ in range(runs):
        t = time.monotonic()
        try:
            findings = await detect(media, watchlist=context.watchlist if context else None)
            judged, combos = await judge(media, findings, context)
            a = await analyze(media, judged, combos, context)
            first = first or a
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
    if with_verify and first is not None and first.findings:
        try:
            rec["verify"] = await run_verify(media, first.findings, Path(name).stem)
        except Exception as e:
            rec["verify"] = {"error": repr(e)}
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
            rec, gts = await run_image(
                name, g, media_of(path), context, runs, os.environ.get("EVAL_VERIFY", "1") == "1"
            )
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


# ---------------------------------------------------------------- 3. 발표용 표 (out_eval/tables.md)
TYPE_KO = {
    "personal_info": "개인정보",
    "location": "촬영 위치",
    "preferences": "취향·생활",
    "social_circle": "인간관계",
    "others_private": "타인 정보",
}


def _load_runs():
    runs = {}
    for d in sorted(Path("out_eval").glob("*/")):
        sj, rj = d / "summary.json", d / "raw.jsonl"
        if sj.exists() and rj.exists():
            raws = [json.loads(x) for x in rj.read_text("utf-8").splitlines() if x.strip()]
            runs[d.name] = (json.loads(sj.read_text("utf-8")), raws)
    return runs


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("-" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _p(v):
    return None if v is None else f"{v * 100:.0f}%"


def _by_type(raws):
    hit, tot = {t: 0 for t in TYPE_KO}, {t: 0 for t in TYPE_KO}
    for rec in raws:
        gts = gt_objects(GT[rec["image"]])
        for run in rec["e2e"]:
            if "error" in run:
                continue
            m = match(gts, [f for f in run["findings"] if f.get("bbox")])
            for gi, g in enumerate(gts):
                if g["top_type"] in tot:
                    tot[g["top_type"]] += 1
                    hit[g["top_type"]] += gi in m
    return {t: (pct(hit[t], tot[t]), tot[t]) for t in TYPE_KO}


def _verify(raws):
    vs = [r["verify"] for r in raws if r.get("verify") and "error" not in r["verify"]]
    if not vs:
        return None
    b = sum(v["before"] or 0 for v in vs)
    tb = sum((v.get("risk_total") or {}).get("before", 0) for v in vs)
    ta = sum((v.get("risk_total") or {}).get("after", 0) for v in vs)
    return {
        "n": len(vs),
        "before": b,
        "removed": sum(v["removed"] or 0 for v in vs),
        "weakened": sum(v["weakened"] or 0 for v in vs),
        "remaining": sum(v["remaining"] or 0 for v in vs),
        "new": sum(v["new"] or 0 for v in vs),
        "clean": pct(sum(bool(v["clean"]) for v in vs), len(vs)),
        "total": (tb, ta, pct(tb - ta, tb)),
        "risk": (
            round(statistics.mean(v["risk_before"] for v in vs), 3),
            round(statistics.mean(v["risk_after"] for v in vs), 3),
        ),
    }


def write_tables():
    runs = _load_runs()
    if not runs:
        return
    names = list(runs)
    S = {n: runs[n][0] for n in names}
    V = {n: _verify(runs[n][1]) for n in names}
    md = ["# ⓪①②④ 성능 평가 (DIPA2 20장)", ""]
    md.append("## 표 1. 한눈에 보기")
    rows = [
        ["⓪ 맥락 실행 간 일치 (Jaccard)"] + [S[n].get("⓪_jaccard_between_runs") for n in names],
        ["① 탐지 recall (IoU≥0.5)"] + [_p(S[n].get("①_recall@IoU0.5")) for n in names],
        ["① 탐지 precision (하한)"] + [_p(S[n].get("①_precision@IoU0.5(하한)")) for n in names],
        ["① 카테고리 정확도"] + [_p(S[n].get("①_category_acc")) for n in names],
        ["② 사람 심각도와 순위 상관 ρ (정답 물체)"]
        + [S[n].get("②_spearman_oracle") for n in names],
        ["② 같은 판정 반복 일치율"] + [_p(S[n].get("②_same_level_all_runs")) for n in names],
        ["④ 처음 찾은 항목 제거율"]
        + [_p(pct(V[n]["removed"], V[n]["before"])) if V[n] else None for n in names],
        ["④ 위험 총량 감소율"] + [_p(V[n]["total"][2]) if V[n] else None for n in names],
        ["소요 시간 (분)"] + [S[n].get("minutes") for n in names],
    ]
    md += [_table(["지표"] + names, rows), ""]
    md.append("## 표 2. ① 정보 유형별 탐지 recall")
    bt = {n: _by_type(runs[n][1]) for n in names}
    rows = [[TYPE_KO[t]] + [f"{_p(bt[n][t][0])} (n={bt[n][t][1]})" for n in names] for t in TYPE_KO]
    md += [_table(["DIPA 정보 유형"] + names, rows), ""]
    md.append("## 표 3. ② 위험도 판정")
    rows = [
        ["순위 상관 ρ, 정답 물체 투입"] + [S[n].get("②_spearman_oracle") for n in names],
        ["순위 상관 ρ, ①→② 전체 흐름"] + [S[n].get("②_spearman_e2e") for n in names],
        ["비교한 물체 수 (정답 / 전체 흐름)"]
        + [f"{S[n].get('②_n_oracle')} / {S[n].get('②_n_e2e')}" for n in names],
        ["반복 실행 판정 일치율"] + [_p(S[n].get("②_same_level_all_runs")) for n in names],
        ["반복 실행 점수 표준편차 (0~16)"] + [S[n].get("②_score_sd_mean") for n in names],
        ["위험 낮은 사진을 위험 없음으로"]
        + [_p(S[n].get("low_risk_images_not_risky")) for n in names],
    ]
    md += [_table(["지표"] + names, rows), ""]
    md.append("## 표 4. ④ 재검증 (② 추천 처리 1회)")
    rows = []
    for label, f in [
        ("사진 수", lambda v: v["n"]),
        ("처음 찾은 항목", lambda v: v["before"]),
        ("제거 / 약해짐 / 남음", lambda v: f"{v['removed']} / {v['weakened']} / {v['remaining']}"),
        ("재검증에서 새로 찾음", lambda v: v["new"]),
        (
            "위험 총량 처리 전 → 후",
            lambda v: f"{v['total'][0]} → {v['total'][1]} ({_p(v['total'][2])} 감소)",
        ),
        ("최고 위험도 평균 처리 전 → 후", lambda v: f"{v['risk'][0]} → {v['risk'][1]}"),
        ("경고 없이 끝난 사진 (clean)", lambda v: _p(v["clean"])),
    ]:
        rows.append([label] + [f(V[n]) if V[n] else None for n in names])
    md += [_table(["지표"] + names, rows), ""]
    md.append(
        "기준: ① IoU≥0.5 (PASCAL VOC·COCO), ② Spearman ρ vs DIPA2 7점 심각도 (Xu et al., IMWUT 2024), "
        "precision 은 DIPA 가 위협 물체만 라벨해서 하한값."
    )
    Path("out_eval/tables.md").write_text("\n".join(md), "utf-8")
    print("\n" + "\n".join(md) + "\n\n발표용 표: out_eval/tables.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["gemini", "qwen"])
    p.add_argument("--runs", type=int, default=3, help="Gemini 반복 횟수")
    p.add_argument("--qwen-runs", type=int, default=1, help="Qwen 반복 횟수 (느려서 기본 1)")
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="④ 재검증 생략 (빠르게)")
    p.add_argument("--country", default="KR")
    p.add_argument("--day", default=date.today().isoformat())
    p.add_argument("--tables", action="store_true", help="이미 돌린 결과로 표만 다시 만들기")
    p.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.tables:
        compare()
        write_tables()
        return
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
            cmd,
            env={
                **os.environ,
                "VLM_PROVIDER": provider,
                "EVAL_VERIFY": "0" if a.no_verify else "1",
                "PYTHONPATH": os.getcwd(),
            },
        )
        if r.returncode:
            print(f"[{provider}] 실패 (코드 {r.returncode}), 다음으로 넘어감")
    compare()
    write_tables()


if __name__ == "__main__":
    main()

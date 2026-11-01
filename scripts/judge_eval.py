# ruff: noqa: E501, ASYNC230, ASYNC240  (단독 실행 도우미 스크립트)
"""채점관 모델 평가 (LLM-as-a-judge, Zheng et al. NeurIPS 2023). 정답 데이터셋 없이 ①②④ 를 잰다.

  uv run python scripts/judge_eval.py                 # Gemini → Qwen 순서로, 사진 10장
  uv run python scripts/judge_eval.py --only gemini
  uv run python scripts/judge_eval.py --tables        # 이미 돌린 결과로 표만

사진: eval_data/judge/ 의 jpg/png 전부 + DIPA 샘플 5장 (eval_data/images, odyssey_eval.py 가 받아 둔 것).
채점관: JUDGE_MODEL (없으면 키로 쓸 수 있는 최신 gemini-*-pro 자동 선택). 테스트 대상과 다른, 더 큰 모델이어야 한다. Gemini API 키 사용.

채점
- ① 사진 + ① 결과(박스) 를 채점관에 보여주고: 항목마다 맞는 단서인지(valid), 심각도 1~4, 놓친 단서 목록.
  precision = valid / 찾은 수, recall = valid / (valid + 놓친 것), 심각도 3 이상 recall 따로.
- ② ② 의 S(심각도 1~4) vs 채점관 심각도: 가중 κ (quadratic, Cohen 1968), 정확 일치율, risk_score Spearman.
- ④ ② 추천대로 ③ 처리한 사진을 채점관에 보여주고: 처음 항목이 아직 알아볼 수 있는지 + 새 단서.
  제거율 = 1 - 아직 보임 / 처음 항목 수.
심각도 기준은 ② 와 같은 CNIL PIA 4단계 (1 무시 가능, 2 제한적, 3 상당함, 4 최대).
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 레포 루트 (stages, worker)

JUDGE_DIR = Path("eval_data/judge")
DIPA_DIR = Path("eval_data/images")
# DIPA 샘플 중 글자·위치·화면 단서가 있는 5장 (번호판, 고층건물, 사무실 건물, 모니터, 간판)
DIPA_PICK = [
    "049cbdf58c353959.jpg",
    "9ba50cb2198a23d8.jpg",
    "709340946095cf0d.jpg",
    "105804.jpg",
    "29715.jpg",
]
OUT = Path("out_eval")


_JUDGE = None


def _pick_pro_model():
    """API 키로 쓸 수 있는 모델 목록에서 가장 최신 gemini-*-pro 를 고른다."""
    import re
    import urllib.request

    from worker.config import get_settings

    key = get_settings().gemini_api_key
    if not key:
        return None
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key.get_secret_value()},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            models = json.loads(r.read()).get("models", [])
    except Exception as e:
        print("모델 목록 조회 실패:", repr(e)[:120])
        return None
    names = [m["name"].removeprefix("models/") for m in models]
    pros = [
        n
        for n in names
        if re.match(r"gemini-[\d.]+-pro", n) and not re.search(r"tts|image|audio|vision", n)
    ]

    def rank(n):
        ver = tuple(int(x) for x in re.findall(r"\d+", n.split("-")[1]))
        return (ver, "preview" not in n and "exp" not in n, -len(n))

    print("쓸 수 있는 Pro 모델:", ", ".join(sorted(pros, key=rank, reverse=True)[:5]) or "없음")
    return max(pros, key=rank) if pros else None


def judge_model():
    """JUDGE_MODEL: 환경변수 → .env → 키로 쓸 수 있는 최신 Pro 자동 선택."""
    global _JUDGE
    if _JUDGE:
        return _JUDGE
    _JUDGE = os.environ.get("JUDGE_MODEL")
    env = Path(".env")
    if not _JUDGE and env.exists():
        for line in env.read_text("utf-8").splitlines():
            if line.strip().startswith("JUDGE_MODEL="):
                _JUDGE = line.split("=", 1)[1].strip()
    if not _JUDGE:
        _JUDGE = _pick_pro_model()
        if not _JUDGE:
            raise SystemExit("채점관 모델을 못 찾음: .env 에 JUDGE_MODEL=<Pro 모델명> 을 넣어줘")
    print("채점관 모델:", _JUDGE)
    return _JUDGE


SEVERITY = "1 무시 가능, 2 제한적, 3 상당함, 4 최대 (CNIL PIA 심각도 4단계)"
CATS = "location, time_movement, identity, contact_finance, attributes, sensitive, relationships, lifestyle"

J1 = f"""너는 사진 프라이버시 평가 채점관이다. 사진을 처음 보는 낯선 사람이 사진에서 알아낼 수 있는 개인 정보 단서를 기준으로 채점한다.
입력: 사진과 탐지 시스템이 찾은 항목 목록 (id, 제목, 박스 box_2d=[ymin,xmin,ymax,xmax] 0~1000).
1) 항목마다 valid: 박스 자리에 실제로 그 단서가 있고 낯선 사람에게 의미 있는 정보면 true. 아니면 false.
   severity: 이 단서가 노출됐을 때 심각도 {SEVERITY}.
2) missed: 목록에 없지만 사진에 있는 단서. 사소한 배경 물체는 넣지 말고 실제로 정보가 되는 것만.
   category 는 {CATS} 중 하나.
판단 근거는 사진에 보이는 것만 쓴다. 추측으로 단서를 만들지 않는다."""

J2 = f"""너는 사진 프라이버시 평가 채점관이다. 아래 사진은 보호 처리(가리기·블러 등)가 끝난 사진이다.
입력: 처리 전에 찾았던 항목 목록 (id, 제목, 박스).
1) 항목마다 still_visible: 처리 후 사진에서 그 단서를 아직 알아볼 수 있으면 true.
2) new_clues: 처리 후 사진에 남아 있는, 목록에 없는 의미 있는 단서. severity {SEVERITY}."""

S1 = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "valid": {"type": "boolean"},
                    "severity": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["id", "valid", "severity"],
            },
        },
        "missed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {"type": "integer"},
                },
                "required": ["title", "category", "severity"],
            },
        },
    },
    "required": ["items", "missed"],
}
S2 = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "still_visible": {"type": "boolean"}},
                "required": ["id", "still_visible"],
            },
        },
        "new_clues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "severity": {"type": "integer"}},
                "required": ["title", "severity"],
            },
        },
    },
    "required": ["items", "new_clues"],
}


def photos():
    out = sorted(
        p for p in JUDGE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    out += [DIPA_DIR / n for n in DIPA_PICK if (DIPA_DIR / n).exists()]
    return out


def lines(findings, size):
    w, h = size
    rows = []
    for f in findings:
        box = ""
        if f.get("bbox"):
            x1, y1, x2, y2 = f["bbox"]
            box = f" box_2d={[round(y1 * 1000 / h), round(x1 * 1000 / w), round(y2 * 1000 / h), round(x2 * 1000 / w)]}"
        rows.append(
            f"- id={f['entity_id']} [{f.get('category')}] {f.get('title', f.get('type'))}{box}"
        )
    return "\n".join(rows) or "(없음)"


async def ask_judge(image_bytes, text, instruction, schema):
    from stages import gemini
    from stages.risk import _image_block
    from worker.config import get_settings

    block, _ = await asyncio.to_thread(_image_block, image_bytes)
    s = get_settings()
    old = s.gemini_model
    s.gemini_model = judge_model()
    try:
        return await gemini.generate_json(
            [block, {"type": "text", "text": text}], schema, instruction, max_output_tokens=16384
        )
    finally:
        s.gemini_model = old


async def run_one(path):
    from eval_viz import save_stages

    from stages.context import build_context
    from stages.images import open_image
    from stages.types import Media
    from worker.contract import UserChoice
    from worker.pipeline import run_detection_phase, run_pipeline

    async def quiet(*_a, **_k):
        return None

    raw = path.read_bytes()
    ext = path.suffix.lstrip(".").lower().replace("jpeg", "jpg")
    size = open_image(raw).size
    rec = {"image": path.name}
    t = time.monotonic()
    await build_context()
    det = await run_detection_phase(Media(image=raw, image_ext=ext), report=quiet)
    findings = det.before.findings
    rec["findings"] = [
        {
            k: f.get(k)
            for k in (
                "entity_id",
                "type",
                "category",
                "title",
                "bbox",
                "severity",
                "likelihood",
                "risk_score",
                "risk_level",
                "action",
                "source",
            )
        }
        for f in findings
    ]
    rec["risk_before"] = det.before.risk
    j1 = await ask_judge(raw, "탐지 항목:\n" + lines(findings, size), J1, S1)
    rec["judge1"] = j1
    if findings:
        choices = []
        for i, f in enumerate(findings):
            act = f.get("action", "mask")
            if f.get("source") == "exif" and act == "blur":
                act = "remove"
            choices.append(UserChoice(idx=i, action=act))
        res = await run_pipeline(
            Media(image=raw, image_ext=ext),
            max_iterations=1,
            report=quiet,
            user_choices=choices,
            initial_findings=findings,
        )
        v = res.verification or {}
        rec["verify"] = {
            k: v.get(k)
            for k in ("before", "removed", "weakened", "remaining", "new", "clean", "risk_total")
        }
        rec["risk_after"] = res.after.risk
        save_stages(
            OUT / "judge_images",
            f"{os.environ.get('VLM_PROVIDER')}_{path.stem}",
            raw,
            det.before.findings,
            res.output.image,
            res.verification,
        )
        if res.output.image:
            rec["judge2"] = await ask_judge(
                res.output.image, "처리 전 항목:\n" + lines(findings, size), J2, S2
            )
    rec["sec"] = round(time.monotonic() - t, 1)
    return rec


async def evaluate(label):
    out = OUT / f"judge_{label}"
    out.mkdir(parents=True, exist_ok=True)
    items = photos()
    print(f"[{label}] 사진 {len(items)}장")
    with open(out / "raw.jsonl", "w", encoding="utf-8") as raw:
        for n, p in enumerate(items, 1):
            print(f"[{label}] {n}/{len(items)} {p.name}")
            try:
                rec = await run_one(p)
            except Exception as e:  # 한 장 실패해도 계속
                rec = {"image": p.name, "error": repr(e)}
                print("   실패:", repr(e)[:200])
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw.flush()


# ---------------------------------------------------------------- 지표
def pct(a, b):
    return round(a / b, 3) if b else None


def qwk(a, b, k=4):
    """quadratic weighted kappa (Cohen 1968), 등급 1..k."""
    n = len(a)
    if n < 2:
        return None
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[x - 1][y - 1] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * ra[i] * cb[j] / n
    return round(1 - num / den, 3) if den else None


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2 + 1
            i = j + 1
        return r

    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def clamp(s):
    return min(4, max(1, int(s or 1)))


def score(recs):
    found = valid = missed = hi_valid = hi_missed = 0
    s_ours, s_judge, risk, errors = [], [], [], 0
    before_n = still = new_after = 0
    secs = []
    for r in recs:
        if "error" in r or "judge1" not in r:
            errors += 1
            continue
        secs.append(r["sec"])
        byid = {f["entity_id"]: f for f in r["findings"]}
        for it in r["judge1"].get("items", []):
            f = byid.get(it.get("id"))
            if f is None:
                continue
            found += 1
            sev = clamp(it.get("severity"))
            if it.get("valid"):
                valid += 1
                hi_valid += sev >= 3
                if f.get("severity"):
                    s_ours.append(clamp(f["severity"]))
                    s_judge.append(sev)
                if f.get("risk_score") is not None:
                    risk.append((f["risk_score"], sev))
        found += len(set(byid) - {it.get("id") for it in r["judge1"].get("items", [])})
        for m in r["judge1"].get("missed", []):
            missed += 1
            hi_missed += clamp(m.get("severity")) >= 3
        if "judge2" in r:
            items = r["judge2"].get("items", [])
            before_n += len(byid)
            still += sum(bool(i.get("still_visible")) for i in items if i.get("id") in byid)
            new_after += len(r["judge2"].get("new_clues", []))
    exact = pct(sum(a == b for a, b in zip(s_ours, s_judge, strict=True)), len(s_ours))
    return {
        "사진 수": len(recs) - errors,
        "① 찾은 항목 / 맞는 단서 / 놓친 단서": f"{found} / {valid} / {missed}",
        "① precision (채점관 기준)": pct(valid, found),
        "① recall (채점관 기준)": pct(valid, valid + missed),
        "① 심각도 3 이상 recall": pct(hi_valid, hi_valid + hi_missed),
        "② 심각도 가중 κ (quadratic)": qwk(s_ours, s_judge),
        "② 심각도 정확 일치율": exact,
        "② risk_score ↔ 채점관 심각도 Spearman": spearman(
            [a for a, _ in risk], [b for _, b in risk]
        ),
        "④ 처리 후 제거율 (채점관 기준)": pct(before_n - still, before_n),
        "④ 처리 후 새로 보이는 단서": new_after,
        "사진당 평균 시간 (초)": round(statistics.mean(secs), 1) if secs else None,
        "실패": errors,
    }


def tables():
    cols = {}
    for d in sorted(OUT.glob("judge_*/raw.jsonl")):
        recs = [json.loads(x) for x in d.read_text("utf-8").splitlines() if x.strip()]
        cols[d.parent.name.removeprefix("judge_")] = score(recs)
    if not cols:
        print("결과 없음")
        return
    names = list(cols)
    keys = list(next(iter(cols.values())))

    def fmt(k, v):
        if v is None:
            return "-"
        if isinstance(v, float) and ("precision" in k or "recall" in k or "율" in k):
            return f"{v * 100:.0f}%"
        return str(v)

    md = [
        "# 채점관 모델 평가 (LLM-as-a-judge)",
        "",
        f"채점관: {judge_model()}. 기준: Zheng et al. NeurIPS 2023, 가중 κ Cohen 1968, 심각도 CNIL PIA 4단계.",
        "채점관도 모델이라 정답이 아니라 '강한 모델 기준'이다.",
        "",
        "| 지표 | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    md += ["| " + k + " | " + " | ".join(fmt(k, cols[n][k]) for n in names) + " |" for k in keys]
    OUT.joinpath("tables_judge.md").write_text("\n".join(md), "utf-8")
    print(
        "\n".join(md) + "\n\n저장: out_eval/tables_judge.md (처리된 사진: out_eval/judge_images/)"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["gemini", "qwen"])
    p.add_argument("--tables", action="store_true")
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a._worker:
        asyncio.run(evaluate(a._worker))
        return
    if not a.tables:
        if not photos():
            print(
                "사진이 없음: eval_data/judge/ 에 넣거나 odyssey_eval.py 를 먼저 한 번 돌려 DIPA 사진을 받아 둬"
            )
            return
        os.environ["JUDGE_MODEL"] = judge_model()
        for provider in ("gemini", "qwen"):
            if a.only and provider != a.only:
                continue
            r = subprocess.run(
                [sys.executable, __file__, "--_worker", provider],
                env={**os.environ, "VLM_PROVIDER": provider, "PYTHONPATH": os.getcwd()},
            )
            if r.returncode:
                print(f"[{provider}] 실패 (코드 {r.returncode})")
    tables()


if __name__ == "__main__":
    main()

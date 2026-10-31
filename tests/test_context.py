"""⓪ 맥락: 외부 검색·LLM 은 부르지 않고 스텁으로 확인한다."""

from datetime import date

import pytest
from pydantic import SecretStr

from stages import context
from stages.http import SearchError
from worker.config import get_settings

DAY = date(2026, 10, 31)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    context._cache.clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tavily_api_key", SecretStr("test-key"))
    monkeypatch.setattr(settings, "context_use_search", True)
    yield
    context._cache.clear()


def _news(calls):
    async def post_json(url, payload, headers, *, http_timeout=15):
        calls.append(payload["query"])
        if "선거" in payload["query"]:
            return {
                "results": [
                    {
                        "title": "지방선거 D-30",
                        "content": "후보 등록",
                        "published_date": "2026-10-20",
                    }
                ]
            }
        return {"results": []}

    return post_json


def _summary(calls):
    async def generate_json(content, schema, instruction):
        calls.append(content[0]["text"])
        return {
            "watchlist": [
                {
                    "item": "선거 홍보물",
                    "category": "sensitive",
                    "why": "지지 성향",
                    "issue": "지방선거",
                }
            ],
            "issues": [
                {
                    "title": "지방선거",
                    "kind": "election",
                    "category": "sensitive",
                    "started": "2026-10-20",
                }
            ],
            "note": "선거철이라 정치 성향 노출 위험이 크다.",
        }

    return generate_json


async def test_rules_only_without_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "tavily_api_key", SecretStr(""))

    async def fail(*args, **kwargs):
        raise AssertionError("must not search")

    monkeypatch.setattr(context, "post_json", fail)
    result = await context.build_context("KR", DAY)
    assert not result.searched
    assert [w["item"] for w in result.watchlist] == [r["item"] for r in context.RULES]
    assert result.issues == [] and result.note == ""


async def test_search_and_summary(monkeypatch):
    queries, prompts = [], []
    monkeypatch.setattr(context, "post_json", _news(queries))
    monkeypatch.setattr(context.vlm, "generate_json", _summary(prompts))
    result = await context.build_context("KR", DAY)

    assert result.searched
    assert len(queries) == len(context.SEARCH_TERMS)
    # 검색어에는 국가·날짜·일반 단어만
    assert all(q.startswith("한국 ") and q.endswith("2026년 10월") for q in queries)
    # 결과가 없는 검색 종류는 요약 입력에 넣지 않는다
    assert "kind=election" in prompts[0] and "kind=protest" not in prompts[0]
    assert result.watchlist[: len(context.RULES)] == [
        {**r, "source": "rule"} for r in context.RULES
    ]
    assert result.watchlist[-1]["source"] == "context"
    assert result.issues == [
        {
            "title": "지방선거",
            "kind": "election",
            "category": "sensitive",
            "started": "2026-10-20",
            "weight": 1.0,  # 반감기 미측정 → 감쇠 없음
        }
    ]


async def test_cached_per_day(monkeypatch):
    queries, prompts = [], []
    monkeypatch.setattr(context, "post_json", _news(queries))
    monkeypatch.setattr(context.vlm, "generate_json", _summary(prompts))
    first = await context.build_context("KR", DAY)
    second = await context.build_context("KR", DAY)
    assert first is second and len(prompts) == 1
    await context.build_context("KR", date(2026, 11, 1))
    assert len(prompts) == 2 and ("KR", DAY.isoformat()) not in context._cache


async def test_search_failure_falls_back(monkeypatch):
    async def down(*args, **kwargs):
        raise SearchError("connection failed or timed out")

    monkeypatch.setattr(context, "post_json", down)
    result = await context.build_context("KR", DAY)
    assert not result.searched and len(result.watchlist) == len(context.RULES)
    # 실패는 짧게만 캐시한다
    assert context._cache[("KR", DAY.isoformat())][1] is not None


async def test_bad_summary_falls_back(monkeypatch):
    monkeypatch.setattr(context, "post_json", _news([]))

    async def bad(content, schema, instruction):
        return {"watchlist": [{"item": "x", "category": "nope", "why": "y"}], "issues": []}

    monkeypatch.setattr(context.vlm, "generate_json", bad)
    result = await context.build_context("KR", DAY)
    assert not result.searched and result.issues == []


def test_issue_weight_decay(monkeypatch):
    monkeypatch.setitem(context.HALF_LIFE_DAYS, "election", 10.0)
    assert context.issue_weight("election", "2026-10-21", DAY) == 0.5
    assert context.issue_weight("election", "", DAY) == 1.0
    assert context.issue_weight("protest", "2026-10-01", DAY) == 1.0

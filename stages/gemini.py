"""Gemini REST 호출. 키·원문·모델 응답은 로그나 오류에 포함하지 않는다."""

import asyncio
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worker.config import get_settings


class AIServiceError(RuntimeError):
    """호출 실패는 안전한 빈 결과로 취급하지 않는다."""


# Gemini 는 길이·범위 제약이 많은 스키마를 "too many states"(HTTP 400)로 거부한다.
# 보낼 때는 제약을 빼고, 받은 응답은 호출한 쪽의 pydantic 모델로 다시 검사한다.
_LIMIT_KEYS = frozenset(
    {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "pattern",
    }
)

# 503(모델 과부하)은 잠깐 뒤 다시 보낸다. 429(한도 초과)는 다시 보내지 않는다.
RETRY_DELAYS = (2, 5)


def serving_schema(schema: Any) -> Any:
    """Gemini 에 보낼 응답 스키마. 길이·범위 제약만 뺀다 (속성 이름은 건드리지 않는다)."""
    if isinstance(schema, list):
        return [serving_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key in ("properties", "$defs"):
            out[key] = {name: serving_schema(sub) for name, sub in value.items()}
        elif key not in _LIMIT_KEYS:
            out[key] = serving_schema(value)
    return out


def _send(request: Request, timeout: int) -> bytes:
    for delay in (*RETRY_DELAYS, None):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read(1024 * 1024 + 1)
        except HTTPError as error:
            if error.code != 503 or delay is None:
                raise
        time.sleep(delay)
    raise AssertionError("unreachable")


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise AIServiceError("GEMINI_API_KEY is not configured")
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key.get_secret_value(),
        },
        method="POST",
    )
    try:
        body = _send(request, settings.gemini_timeout_seconds)
        if len(body) > 1024 * 1024:
            raise AIServiceError("Gemini response exceeded the size limit")
        result = json.loads(body)
    except HTTPError as error:
        raise AIServiceError(f"Gemini HTTP {error.code}; no paid fallback was called") from None
    except (URLError, TimeoutError, OSError):
        raise AIServiceError("Gemini connection failed or timed out") from None
    except (ValueError, UnicodeError):
        raise AIServiceError("Gemini returned invalid JSON") from None
    if not isinstance(result, dict):
        raise AIServiceError("Gemini returned an invalid response")
    return result


async def generate_json(
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    instruction: str,
    *,
    max_output_tokens: int = 4096,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    """max_output_tokens 는 생각(thinking) 토큰까지 포함한다. 넘치면 답이 잘린다."""
    generation_config: dict[str, Any] = {"temperature": 0.1, "max_output_tokens": max_output_tokens}
    if thinking_level:
        generation_config["thinking_level"] = thinking_level  # low | medium | high
    result = await asyncio.to_thread(
        _request,
        {
            "model": get_settings().gemini_model,
            "input": content,
            "system_instruction": instruction,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": serving_schema(schema),
            },
            "generation_config": generation_config,
            "store": False,
        },
    )
    if result.get("status") != "completed":
        # 상태값만 남긴다 (incomplete = 토큰 한도, 그 외 = 차단 등). 원문은 넣지 않는다.
        status = str(result.get("status"))[:20]
        raise AIServiceError(f"Gemini analysis was blocked or did not complete (status={status})")
    # Interactions API의 최종 텍스트만 사용한다. 생각/도구 출력은 결과가 아니다.
    try:
        text = "".join(
            part["text"]
            for step in result["steps"]
            if step["type"] == "model_output"
            for part in step["content"]
            if part["type"] == "text"
        )
    except (KeyError, TypeError):
        raise AIServiceError("Gemini returned invalid output steps") from None
    if not text.strip():
        raise AIServiceError("Gemini returned no analysis text")
    try:
        decoded = json.loads(text)
    except ValueError:
        raise AIServiceError("Gemini analysis was not valid JSON") from None
    if not isinstance(decoded, dict):
        raise AIServiceError("Gemini analysis must be an object")
    return decoded

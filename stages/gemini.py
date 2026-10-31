"""Gemini REST 호출. 키·원문·모델 응답은 로그나 오류에 포함하지 않는다."""

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worker.config import get_settings


class AIServiceError(RuntimeError):
    """호출 실패는 안전한 빈 결과로 취급하지 않는다."""


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
        with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
            body = response.read(1024 * 1024 + 1)
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
    """max_output_tokens 는 생각(thinking) 토큰까지 포함한다. 생각이 길면 답이 잘려 incomplete 가 된다."""
    generation_config: dict[str, Any] = {"temperature": 0.1, "max_output_tokens": max_output_tokens}
    if thinking_level:
        generation_config["thinking_level"] = thinking_level  # low | medium | high
    result = await asyncio.to_thread(
        _request,
        {
            "model": get_settings().gemini_model,
            "input": content,
            "system_instruction": instruction,
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
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

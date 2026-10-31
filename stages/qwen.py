"""로컬 VLM (Qwen3-VL 등) — vLLM 의 OpenAI 호환 API 로 호출한다.
키·원문·응답은 로그에 남기지 않는다.

GPU 서버에서 예:
  vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 16384 \
    --limit-mm-per-prompt image=1
.env:
  VLM_PROVIDER=qwen
  QWEN_BASE_URL=http://localhost:8001/v1
  QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
gemini.py 와 같은 content 블록(image: data+mime_type / text)을 받아 OpenAI chat 형식으로 바꾸고,
response_format=json_schema 로 JSON 만 받는다.
"""

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from stages.gemini import AIServiceError
from worker.config import get_settings


def _to_openai_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block['mime_type']};base64,{block['data']}"},
                }
            )
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
    return parts


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.qwen_api_key:
        headers["Authorization"] = f"Bearer {settings.qwen_api_key.get_secret_value()}"
    request = Request(
        f"{settings.qwen_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise AIServiceError("local VLM response exceeded the size limit")
        result = json.loads(body)
    except HTTPError as error:
        raise AIServiceError(f"local VLM HTTP {error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise AIServiceError("local VLM connection failed or timed out") from None
    except (ValueError, UnicodeError):
        raise AIServiceError("local VLM returned invalid JSON") from None
    if not isinstance(result, dict):
        raise AIServiceError("local VLM returned an invalid response")
    return result


async def generate_json(
    content: list[dict[str, Any]], schema: dict[str, Any], instruction: str
) -> dict[str, Any]:
    settings = get_settings()
    result = await asyncio.to_thread(
        _request,
        {
            "model": settings.qwen_model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": _to_openai_content(content)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "detection", "schema": schema},
            },
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    )
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AIServiceError("local VLM returned no choices") from None
    if not isinstance(text, str) or not text.strip():
        raise AIServiceError("local VLM returned no analysis text")
    try:
        decoded = json.loads(text)
    except ValueError:
        raise AIServiceError("local VLM analysis was not valid JSON") from None
    if not isinstance(decoded, dict):
        raise AIServiceError("local VLM analysis must be an object")
    return decoded

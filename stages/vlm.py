"""VLM 제공자 디스패처. detect.py 는 이 모듈의 generate_json 만 부른다.

VLM_PROVIDER=gemini (기본)  → stages.gemini  (Interactions API, 무료 키)
VLM_PROVIDER=qwen           → stages.qwen    (vLLM OpenAI 호환 서버의 Qwen3-VL 등, GPU 서버 로컬)
입력·출력 형식은 두 제공자가 동일하다: content 블록 리스트 + JSON 스키마 → dict.
"""

from typing import Any

from stages import gemini, qwen
from stages.gemini import AIServiceError
from worker.config import get_settings

__all__ = ["AIServiceError", "generate_json"]


async def generate_json(
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    instruction: str,
    *,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """max_output_tokens 는 Gemini 에만 적용한다 (생각 토큰 포함 한도). None 이면 기본값."""
    provider = get_settings().vlm_provider
    if provider == "gemini":
        if max_output_tokens is None:
            return await gemini.generate_json(content, schema, instruction)
        return await gemini.generate_json(
            content, schema, instruction, max_output_tokens=max_output_tokens
        )
    if provider == "qwen":
        return await qwen.generate_json(content, schema, instruction)
    raise AIServiceError(f"unknown VLM_PROVIDER: {provider}")

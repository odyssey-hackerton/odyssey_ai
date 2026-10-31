"""외부 API POST (웹 검색 등). 키·검색어·응답은 로그나 오류 메시지에 남기지 않는다."""

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SearchError(RuntimeError):
    """검색 실패. 호출한 쪽은 검색 없이 계속 진행한다."""


def _call(url: str, headers: dict[str, str], body: bytes, timeout: int) -> Any:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(2 * 1024 * 1024))
    except HTTPError as error:
        raise SearchError(f"HTTP {error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise SearchError("connection failed or timed out") from None
    except ValueError:
        raise SearchError("invalid JSON") from None


async def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], *, http_timeout: int = 15
) -> Any:
    headers = {"Content-Type": "application/json", **headers}
    return await asyncio.to_thread(_call, url, headers, json.dumps(payload).encode(), http_timeout)

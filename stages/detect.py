"""① 탐지: 사진(OCR, 얼굴, EXIF 등)과 글에서 개인정보 후보를 찾는다."""

from typing import Any

from stages.types import Media


async def detect(media: Media) -> list[dict[str, Any]]:
    # TODO(AI): PaddleOCR, 얼굴 탐지, EXIF 확인, 글 분석
    # 반환 예: [{"type": "address", "source": "ocr", "bbox": [40, 60, 320, 110], "score": 0.91}]
    return []

"""PaddleOCR PP-OCRv5 korean. 입력은 open_image()로 EXIF 회전을 적용한 RGB 이미지라서
반환 좌표가 transform.py 의 편집 좌표와 같은 기준이다."""

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from worker.config import get_settings

log = logging.getLogger(__name__)
MODEL_NAME = "korean_PP-OCRv5_mobile_rec"


@dataclass
class OcrLine:
    text: str
    score: float
    bbox: list[float]  # [x1, y1, x2, y2]


@lru_cache
def _engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        log.warning("paddleocr 미설치. OCR 건너뜀 — README '탐지 모델 설치' 참고")
        return None
    settings = get_settings()
    return PaddleOCR(
        text_recognition_model_name=MODEL_NAME,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device=settings.paddle_device,
        # CPU 추론에서 oneDNN 이 `RuntimeError: std::exception` 을 내는 서버가 있다 → 끌 수 있게
        enable_mkldnn=settings.paddle_mkldnn,
    )


def read_lines(image: Image.Image) -> list[OcrLine]:
    engine = _engine()
    if engine is None:
        return []
    array = np.asarray(image)[:, :, ::-1]  # PaddleOCR 는 BGR
    lines: list[OcrLine] = []
    for res in engine.predict(array):
        data = res.json["res"] if hasattr(res, "json") else res
        polys = data.get("rec_polys") or data.get("dt_polys") or []
        for text, score, poly in zip(data["rec_texts"], data["rec_scores"], polys, strict=False):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            lines.append(OcrLine(text, float(score), [min(xs), min(ys), max(xs), max(ys)]))
    return lines

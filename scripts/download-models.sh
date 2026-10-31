#!/usr/bin/env bash
# ① 탐지 로컬 모델 준비: Veil-PII-Ko-Lite 다운로드 + PP-OCRv5 korean 모델 캐시.
# 로컬:  uv run bash scripts/download-models.sh   (uv run 이 venv 를 PATH 에 올린다)
# Docker: 빌드 중 자동 실행 (Dockerfile 참고)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models
hf download 1T/veil-pii-ko-lite --local-dir models/veil
#   → models/veil/model.int8.onnx, veil.py, 토크나이저 (Apache-2.0)

python - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
          use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True)
print("PP-OCRv5 korean 모델 캐시 완료")
PY

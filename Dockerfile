# odyssey-ai: GPU 서버에서 도는 AI 워커.
# Gemini API와 Pillow 편집은 GPU 없이 python:3.12-slim에서 동작한다.
# PaddleOCR 등 GPU 모델을 넣을 때 CUDA 베이스 이미지로 바꾸고 `docker run --gpus all` 로 실행한다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

# PaddleOCR(opencv) 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /srv

# 의존성 레이어를 먼저 만들어 코드만 바뀌면 빠르게 다시 빌드되게 한다.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY worker/ worker/
COPY stages/ stages/
COPY scripts/download-models.sh scripts/
# 모델은 이미지에 굽는다 (Veil ~150MB, PP-OCRv5 korean ~20MB). 빌드 시 네트워크 필요.
ENV PADDLE_PDX_CACHE_HOME=/srv/.paddlex HF_HOME=/srv/.hf
RUN bash scripts/download-models.sh
RUN useradd --create-home --uid 1000 app && chown -R app /srv
USER app

# 통신 확인만 할 때: docker run --rm ... odyssey-ai python -m worker.check
CMD ["python", "-m", "worker.main"]

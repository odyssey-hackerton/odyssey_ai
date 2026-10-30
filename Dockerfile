# odyssey-ai: GPU 서버에서 도는 AI 워커.
# 지금은 모델이 없어 python:3.12-slim 으로 충분하다.
# PaddleOCR 등 GPU 모델을 넣을 때 CUDA 베이스 이미지로 바꾸고 `docker run --gpus all` 로 실행한다.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /srv

# 의존성 레이어를 먼저 만들어 코드만 바뀌면 빠르게 다시 빌드되게 한다.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY worker/ worker/
COPY stages/ stages/
RUN useradd --create-home --uid 1000 app && chown -R app /srv
USER app

# 통신 확인만 할 때: docker run --rm ... odyssey-ai python -m worker.check
CMD ["python", "-m", "worker.main"]

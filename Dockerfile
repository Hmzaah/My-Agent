# CPU image for the API. Talks to an OpenAI-compatible server (e.g. Ollama in docker-compose)
# and runs embeddings + reranking locally with sentence-transformers.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[embeddings]"

COPY data/sample ./data/sample
COPY eval ./eval

RUN useradd --create-home --uid 1000 jolt && chown -R jolt /app
USER jolt

ENV JOLT_LLM_BACKEND=openai \
    JOLT_OPENAI_BASE_URL=http://ollama:11434/v1 \
    JOLT_INDEX_DIR=/app/data/index

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["jolt", "serve", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn rag.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
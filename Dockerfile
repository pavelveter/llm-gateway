# ── build stage (uv) ──────────────────────────────────────
FROM ghcr.io/astral-sh/uv:latest AS uv-bin

FROM python:3.11-slim AS builder
COPY --from=uv-bin /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .
RUN uv sync --frozen --no-dev

# ── runtime stage (slim) ───────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY *.py .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "4000"]

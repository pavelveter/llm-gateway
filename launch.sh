#!/usr/bin/env bash

set -e

echo "[boot] starting llm-gateway..."

if [ ! -f ".env" ]; then
  echo "[error] .env not found — copy .env.example to .env and configure backends"
  exit 1
fi

uv venv .venv >/dev/null 2>&1 || true
source .venv/bin/activate

uv sync --frozen --no-dev 2>/dev/null || uv pip install -q fastapi httpx uvicorn pydantic aiolimiter

echo "[ok] dependencies ready"

uv run uvicorn gateway:app \
  --host 0.0.0.0 \
  --port "${PORT:-4000}" \
  --reload

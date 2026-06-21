#!/usr/bin/env bash

set -e

echo "[boot] starting LLM environment..."

# ===== CONFIG =====
OLLAMA_HOST="http://localhost:11434"
OLLAMA_PID=""
INTERCEPTOR_PID=""

# ===== CLEANUP =====
cleanup() {
  echo ""
  echo "[shutdown] stopping services..."

  if [[ -n "$INTERCEPTOR_PID" ]]; then
    kill "$INTERCEPTOR_PID" 2>/dev/null || true
    echo "[shutdown] interceptor stopped"
  fi

  if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    echo "[shutdown] ollama stopped"
  fi

  exit 0
}

trap cleanup SIGINT SIGTERM

# ===== 1. START OLLAMA =====
echo "[1/2] checking ollama..."

if ! command -v ollama &> /dev/null; then
  echo "[error] ollama not installed"
  exit 1
fi

if ! lsof -i :11434 > /dev/null 2>&1; then
  echo "[start] launching ollama server..."
  ollama serve > ollama.log 2>&1 &
  OLLAMA_PID=$!
  sleep 2
  echo "[ok] ollama started (pid=$OLLAMA_PID)"
else
  echo "[ok] ollama already running"
fi

# ===== 2. OPTIONAL INTERCEPTOR =====
echo "[2/2] starting interceptor..."

if [ -f "./main.py" ]; then
  uv venv .venv >/dev/null 2>&1 || true
  source .venv/bin/activate

  uv pip install -q fastapi uvicorn httpx slowapi

  uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-4000}" \
    --reload \
    > interceptor.log 2>&1 &

  INTERCEPTOR_PID=$!

  echo "[ok] interceptor running (pid=$INTERCEPTOR_PID)"
else
  echo "[skip] no interceptor (main.py not found)"
fi

echo ""
echo "[ready] system is up"
echo ""
echo "  Model:    ${MODEL:-minimax-m2.7:cloud}"
echo "  Ollama:   http://localhost:11434"
echo "  Gateway:  http://localhost:${PORT:-4000}"
echo "  Health:   http://localhost:${PORT:-4000}/health"
echo ""
echo "logs:"
echo "  tail -f ollama.log"
echo "  tail -f interceptor.log"
echo ""

wait

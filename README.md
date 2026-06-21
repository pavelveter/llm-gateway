# llm-gateway

OpenAI-compatible API gateway for local Ollama models.

Proxies `/v1/chat/completions` to a local Ollama instance with context trimming, response caching, and optional rate limiting.

## Quick start

```bash
./launch.sh
```

This starts Ollama (if not running) and the FastAPI gateway on port 4000.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL` | `minimax-m2.7:cloud` | Model to use |
| `PORT` | `4000` | Gateway port |
| `MAX_MESSAGES` | `6` | Max messages to keep in context |
| `CACHE_MAX_SIZE` | `256` | Max cached responses |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL in seconds |
| `RATE_LIMIT` | _(disabled)_ | Rate limit per client, e.g. `30/minute`. Requires `pip install slowapi` |
| `SYSTEM_PROMPT` | `You are a helpful coding assistant.` | System prompt injected into every request |

## Example

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

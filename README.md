# llm-gateway

An OpenAI-compatible LLM proxy gateway with automatic backend failover, rate-limit awareness, and streaming support.

## Features

- **Multi-backend failover** — configure multiple upstream LLM APIs; the gateway routes to the healthiest one
- **Smart error propagation** — returns proper HTTP status codes (429, 504, 502, 503) and `Retry-After` headers to clients instead of a generic 503
- **Streaming** — real SSE pass-through from upstream backends with failover
- **Rate limiting** — per-backend RPM limits via token bucket (aiolimiter)
- **Cooldown scoring** — backends that fail get temporary cooldowns (10s for timeouts, 15s for other errors); the gateway routes around them
- **Latency-weighted selection** — among healthy backends, the fastest one is picked
- **Chat logging** — requests and responses logged to rotating files
- **Connection pooling** — shared httpx.AsyncClient per backend (with optional per-backend proxy support)
- **Circuit breaker** — exponential cooldown on repeated backend failures (15s → 30s → 60s → ...)
- **Request body validation** — rejects payloads exceeding `LLM_MAX_REQUEST_BYTES` with 413
- **Request tracing** — `X-Request-ID` header in all chat responses for troubleshooting
- **Model aliasing** — set `BACKEND_N_MODEL` to route specific model names to specific backends
- **Per-backend proxy** — set `BACKEND_N_PROXY` to route a backend through HTTP(S) or SOCKS5 proxy
- **Base model fallback** — set `BASE_MODEL` as a default model; backends without `BACKEND_N_MODEL` use it instead
- **Metrics** — `/metrics` endpoint with per-backend latency stats (p50, p99), failure counts
- **Health ping** — `/health/backends` endpoint with live HEAD requests to verify backend reachability
- **Docker** — multi-stage build with uv, slim runtime image, healthcheck

## Quick start

### 1. Configure backends

Copy `.env.example` to `.env` and fill in your API URLs and keys:

```env
BACKEND_1_URL=https://api.openai.com/v1/chat/completions
BACKEND_1_KEY=sk-your-key-here
# BACKEND_1_MODEL=gpt-4o
# BACKEND_1_PROXY=socks5://user:pass@127.0.0.1:1080

BACKEND_2_URL=https://api.anthropic.com/v1/messages
BACKEND_2_KEY=sk-ant-another-key-here
# BACKEND_2_PROXY=http://proxy.example.com:8080

# Default model sent to backends that don't specify BACKEND_N_MODEL
# BASE_MODEL=gpt-4o-mini
```

Add as many backends as you need: `BACKEND_3_URL`/`BACKEND_3_KEY`, etc.

### 2. Run with Docker Compose

```bash
docker compose up --build
```

The gateway listens on `http://localhost:4000`.

### 3. Send a request

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or use `payload.json`:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Backend health status |
| `/health/backends` | GET | Live ping of each backend |
| `/metrics` | GET | Latency stats, failure counts, queue info |
| `/v1/models` | GET | Available models (proxy) |
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |

### Streaming

Set `"stream": true` in the request body. The gateway passes SSE chunks from the upstream backend directly to the client.

## Architecture

```
Client → FastAPI → [stream?] → stream_backend() → BackendManager.call_stream() → Upstream
                  ↘ [non-stream] → Queue → Worker → BackendManager.call() → Upstream
```

### Module layout

| Module | Responsibility |
|---|---|
| `gateway.py` | FastAPI app, endpoints, streaming helper |
| `backend.py` | Backend state, scoring, HTTP calls, domain exceptions |
| `worker.py` | Queue consumer, failover loop for non-streaming requests |
| `config.py` | ENV-based configuration |
| `models.py` | Pydantic request models |
| `logger.py` | Rotating file loggers (system + chat) |

### Request flow (non-streaming)

1. Request arrives at `/v1/chat/completions`
2. Placed in `asyncio.Queue` (max `LLM_QUEUE_MAX` entries)
3. One of `LLM_WORKERS` worker coroutines picks it up
4. Worker sorts backends by score (latency + failures)
5. Tries backends in order, skipping those on cooldown
6. First successful response is returned to the client
7. If all fail, returns the appropriate error status code

### Request flow (streaming)

Streaming bypasses the worker queue entirely:

1. `stream_backend()` iterates backends by score
2. Opens an SSE stream to the first available backend
3. Yields chunks directly to the client
4. On HTTP-level failure (429, 5xx, connection refused) before any data is flushed, the next backend is tried transparently. If the stream fails mid-flight, remaining backends are attempted but the client may see a gap or truncated response. When all backends are exhausted, an error SSE event with `[DONE]` is sent.

## Error handling

The gateway maps upstream failures to meaningful HTTP status codes:

| Upstream errors | Client gets |
|---|---|
| All backends return 429 | `429` + `Retry-After` header |
| All backends timeout | `504 Gateway Timeout` |
| All backends return 5xx | `502 Bad Gateway` |
| Mixed error types | `503 Service Unavailable` |
| Queue full | `503 Queue full` |

For streaming clients, errors are delivered as SSE `data:` events with `[DONE]` markers.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_N_URL` | *required* | OpenAI-compatible chat completions endpoint |
| `BACKEND_N_KEY` | *required* | API key (sent as `Bearer <key>`) |
| `BACKEND_N_MODEL` | *(empty)* | Route this model name exclusively to this backend |
| `BACKEND_N_PROXY` | *(empty)* | Route this backend through a proxy (`http://`, `https://`, or `socks5://`) |
| `BASE_MODEL` | *(empty)* | Default model for backends without `BACKEND_N_MODEL` (fallback chain: `BACKEND_N_MODEL` > `BASE_MODEL` > request model) |
| `LLM_RPM_LIMIT` | `38` | Max requests per minute per backend |
| `LLM_QUEUE_MAX` | `100` | Max pending non-streaming requests |
| `LLM_WORKERS` | `2` | Number of worker coroutines |
| `LLM_STREAM_CONCURRENCY` | `20` | Max simultaneously-active stream drivers. Workers *detach* stream jobs into background tasks, so dispatcher capacity (`LLM_WORKERS`) is decoupled from upstream-bound concurrency (`LLM_STREAM_CONCURRENCY`). |
| `LLM_MAX_REQUEST_BYTES` | `1048576` | Max request body size (1 MB). Requests exceeding this get 413. |
| `LLM_REQUEST_TIMEOUT` | `120` | Gateway-level request timeout (seconds) |
| `PYTHONUNBUFFERED` | `1` | Set in docker-compose for real-time logs |

## Logs

Two rotating log files in `./logs/`:

- **`system.log`** — request tracing, backend selection, errors
- **`chat.log`** — user prompts and assistant responses (JSON lines)

## Development

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run uvicorn gateway:app --host 0.0.0.0 --port 4000 --reload
```

Set environment variables from `.env` before running (or use `dotenv`).

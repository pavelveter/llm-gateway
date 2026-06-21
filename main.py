import hashlib
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-gateway")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "minimax-m2.7:cloud")
PORT = int(os.getenv("PORT", "4000"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "6"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful coding assistant.")


# -------- cache with TTL + max size --------

class TTLCache:
    def __init__(self, max_size: int = 256, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = (value, time.time())

    def __len__(self) -> int:
        return len(self._store)


cache = TTLCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)


# -------- rate limiting (optional) --------

limiter = None
if RATE_LIMIT:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    logger.info("Rate limiting enabled: %s", RATE_LIMIT)


# -------- app --------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LLM Gateway starting — model=%s ollama=%s", MODEL, OLLAMA_URL)
    yield
    logger.info("LLM Gateway shutting down")


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
if limiter:
    app.state.limiter = limiter


# -------- utils --------

def hash_prompt(messages: list[dict], model: str) -> str:
    raw = f"{model}:{messages}".encode()
    return hashlib.sha256(raw).hexdigest()


def trim(messages: list[dict]) -> list[dict]:
    return messages[-MAX_MESSAGES:]


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


def make_response(content: str, model: str, cached: bool = False) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "cached": cached,
    }


# -------- endpoints --------

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "cache_size": len(cache)}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "ollama",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    messages = data.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages is required"})

    start = time.time()

    messages = trim(messages)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    key = hash_prompt(messages, MODEL)

    cached = cache.get(key)
    if cached is not None:
        logger.info("cache_hit tokens_in=%d", estimate_tokens(messages))
        return make_response(cached, MODEL, cached=True)

    tokens = estimate_tokens(messages)
    logger.info("tokens_in=%d", tokens)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                },
            )
            r.raise_for_status()
            body = r.json()
    except httpx.ConnectError:
        logger.error("Ollama unreachable at %s", OLLAMA_URL)
        return JSONResponse(
            status_code=502,
            content={"error": "Ollama is not available"},
        )
    except httpx.HTTPStatusError as e:
        logger.error("Ollama returned %d: %s", e.response.status_code, e.response.text[:200])
        return JSONResponse(
            status_code=502,
            content={"error": f"Ollama error: {e.response.status_code}"},
        )
    except Exception as e:
        logger.exception("Unexpected error calling Ollama")
        return JSONResponse(status_code=500, content={"error": str(e)})

    response = body.get("message", {}).get("content", "")
    cache.set(key, response)

    latency = time.time() - start
    logger.info("latency=%.2fs tokens_in=%d", latency, tokens)

    return make_response(response, MODEL, cached=False)

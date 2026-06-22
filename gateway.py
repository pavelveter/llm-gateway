from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend import BackendManager, Backend
from config import CFG
from logger import sys_log, chat_log
from models import ChatCompletionRequest
from worker import WorkerManager, QueueFull, classify_errors, get_retry_after


manager: BackendManager | None = None
workers: WorkerManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, workers
    backends = [Backend(name=n, url=u, key=k) for n, u, k in CFG.backends]
    if not backends:
        sys_log.error("no backends configured — set BACKEND_N_URL and BACKEND_N_KEY")
        raise SystemExit(1)
    manager = BackendManager(backends)
    workers = WorkerManager(manager)
    workers.start()
    sys_log.info("gateway started with %d backend(s)", len(backends))
    yield
    workers.stop()
    sys_log.info("gateway stopped")


app = FastAPI(title="llm-gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": [
            {"name": b.name, "healthy": b.healthy, "score": round(b.score, 4)}
            for b in manager.backends
        ],
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": b.name, "object": "model", "created": 0, "owned_by": "proxy"}
            for b in manager.backends
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    req = ChatCompletionRequest(**body)
    payload = req.model_dump(exclude_none=True)

    chat_log.log_request(model=req.model or "unknown", messages=[m.model_dump() for m in req.messages])

    if stream:
        return StreamingResponse(_stream_backend(payload, req.model or "unknown"), media_type="text/event-stream")

    try:
        result = await workers.submit(payload)
    except QueueFull:
        return JSONResponse(status_code=503, content={"error": "queue full"})
    except Exception as exc:
        errors = [exc]
        code = classify_errors(errors)
        resp = {"error": f"all backends failed: {code}"}
        if code == 429:
            return JSONResponse(status_code=429, content=resp, headers={"Retry-After": str(get_retry_after(errors))})
        return JSONResponse(status_code=code, content=resp)

    chat_log.log_response(model=req.model or "unknown", content=result.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return result


async def _stream_backend(payload: dict, model: str):
    start = time.time()
    try:
        async for chunk in manager.call_stream(payload):
            yield chunk
    except Exception as exc:
        sys_log.error("stream error: %s", exc)
        yield f"data: {__import__('json').dumps({'error': str(exc)})}\n\n"
    yield b"data: [DONE]\n\n"
    latency = time.time() - start
    chat_log.log_response(model=model, content="[stream]", latency=latency)

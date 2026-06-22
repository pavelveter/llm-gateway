from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend import BackendManager
from config import (
    CORS_ORIGINS,
    MAX_REQUEST_BYTES,
    QUEUE_MAX,
    REQUEST_TIMEOUT,
    STREAM_CONCURRENCY,
    WORKERS,
    load_backends,
)
from logger import setup_logging
from models import ChatRequest
from worker import FutureWrapper, StreamJob, _relay_stream, drive_stream, worker

__all__ = ["app"]

log, chat_log = setup_logging()

backend_mgr = BackendManager()
request_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)


@asynccontextmanager
async def lifespan(_: FastAPI):
    backends = load_backends()
    backend_mgr.load(backends)

    stop_event = asyncio.Event()
    workers = [
        asyncio.create_task(
            worker(
                request_queue,
                backend_mgr,
                stop_event,
                stream_concurrency=STREAM_CONCURRENCY,
            )
        )
        for _ in range(WORKERS)
    ]
    log.info(
        "Workers started: %d (stream_concurrency=%d)",
        WORKERS, STREAM_CONCURRENCY,
    )
    try:
        yield
    finally:
        log.info("Shutting down: signalling workers to stop")
        stop_event.set()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await backend_mgr.close()
        log.info("Workers stopped")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"},
        )
    response = await call_next(request)
    return response


# ── helpers / metadata endpoints ─────────────────────────────


@app.get("/health")
async def health() -> dict:
    if not backend_mgr.loaded:
        return {"status": "degraded", "backends": [], "error": "no backends loaded"}
    return {"status": "ok", "backends": backend_mgr.health()}


@app.get("/health/backends")
async def health_backends() -> dict:
    if not backend_mgr.loaded:
        return {"backends": []}
    results = await asyncio.gather(
        *(backend_mgr.ping(b) for b in backend_mgr.backends),
        return_exceptions=True,
    )
    backends = []
    for r in results:
        if isinstance(r, Exception):
            backends.append({"status": "error", "error": type(r).__name__})
        else:
            backends.append(r)
    return {"backends": backends}


@app.get("/metrics")
async def metrics() -> dict:
    return backend_mgr.metrics()


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": b["name"], "object": "model", "created": 0, "owned_by": "proxy"}
            for b in backend_mgr.backends
        ],
    }


# ── chat completions ─────────────────────────────────────────


class _NoopRequest:
    """Sentinel used when no FastAPI Request is available."""

    async def is_disconnected(self) -> bool:
        return False


async def stream_backend(
    payload: dict, trace_id: str, request: Request | None = None
) -> AsyncIterator[str]:
    """Legacy entry point kept for tests and ad-hoc callers."""
    job = StreamJob(payload=payload, trace_id=trace_id)
    task = asyncio.create_task(
        drive_stream(job, backend_mgr), name=f"stream-drive-{trace_id}"
    )
    task.add_done_callback(_log_background_task_failure)

    req = request if request is not None else _NoopRequest()
    try:
        async for chunk in _relay_stream(job, req):
            yield chunk
    finally:
        job.disconnected.set()


def _log_background_task_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception("background stream task failed: %s", repr(exc))


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    trace_id = str(uuid.uuid4())[:8]
    log.info("[%s] INCOMING stream=%s model=%s", trace_id, req.stream, req.model)

    payload: dict = {
        "model": req.model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
    }
    for key in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "stop"):
        val = getattr(req, key, None)
        if val is not None:
            payload[key] = val

    if req.stream:
        job = StreamJob(payload=payload, trace_id=trace_id)
        try:
            request_queue.put_nowait(job)
        except asyncio.QueueFull:
            log.error("[%s] QUEUE FULL (stream)", trace_id)
            raise HTTPException(status_code=503, detail="Queue full")
        return StreamingResponse(
            _relay_stream(job, request),
            media_type="text/event-stream",
            headers={"X-Request-ID": trace_id},
        )

    wrapper = FutureWrapper()
    wrapper.trace_id = trace_id
    try:
        request_queue.put_nowait((payload, wrapper))
    except asyncio.QueueFull:
        log.error("[%s] QUEUE FULL", trace_id)
        raise HTTPException(status_code=503, detail="Queue full")

    try:
        result = await asyncio.wait_for(wrapper.future, timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        log.error("[%s] REQUEST TIMEOUT after %.0fs", trace_id, REQUEST_TIMEOUT)
        raise HTTPException(status_code=504, detail="Gateway request timeout")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("[%s] REQUEST FAILED: %s", trace_id, type(e).__name__)
        raise HTTPException(status_code=503, detail="LLM upstream error")

    try:
        user_messages = [m.content for m in req.messages if m.role == "user"]
        assistant_response = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        chat_log.info(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "model": req.model,
                    "user": user_messages,
                    "assistant": assistant_response[:500],
                },
                ensure_ascii=False,
            )
        )
    except Exception as e:
        log.exception("[%s] CHAT LOGGING FAILED: %s", trace_id, type(e).__name__)

    return JSONResponse(content=result, headers={"X-Request-ID": trace_id})

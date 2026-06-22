import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend import BackendManager
from config import QUEUE_MAX, STREAM_CONCURRENCY, WORKERS, load_backends
from logger import setup_logging
from models import ChatRequest
from worker import FutureWrapper, StreamJob, _relay_stream, drive_stream, worker

log, chat_log = setup_logging()

backend_mgr = BackendManager()
request_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Modern FastAPI startup/shutdown: load backends, spawn workers,
    then on shutdown signal them to stop and drain.

    Workers dispatch but do NOT block on streams: each ``StreamJob`` is
    detached into a ``_supervised_drive_stream`` task bounded by
    ``STREAM_CONCURRENCY``.  This way ``WORKERS`` only governs
    dispatcher throughput, not upstream-bound concurrency.
    """
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
        f"Workers started: {WORKERS} (stream_concurrency={STREAM_CONCURRENCY})"
    )
    try:
        yield
    finally:
        log.info("Shutting down: signalling workers to stop")
        stop_event.set()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        log.info("Workers stopped")


app = FastAPI(lifespan=lifespan)


# ── helpers / metadata endpoints ─────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Liveness + per-backend health snapshot."""
    return {"status": "ok", "backends": backend_mgr.health()}


@app.get("/v1/models")
async def models() -> dict:
    """OpenAI-compatible model list."""
    return {
        "object": "list",
        "data": [{"id": "llm-gateway", "object": "model"}],
    }


# ── chat completions ─────────────────────────────────────────


class _NoopRequest:
    """Sentinel used when no FastAPI Request is available (legacy tests,
    ``stream_backend`` direct callers).  Always reports connected."""

    async def is_disconnected(self) -> bool:
        return False


async def stream_backend(
    payload: dict, trace_id: str, request: Request | None = None
) -> AsyncIterator[str]:
    """Legacy entry point kept for tests and ad-hoc callers.

    Production code goes through :func:`chat` which uses
    :func:`_relay_stream` directly against a worker-drained
    :class:`StreamJob`.  This helper spawns the worker-drain in the
    background and relays, sharing the same code path.
    """
    job = StreamJob(payload=payload, trace_id=trace_id)
    task = asyncio.create_task(
        drive_stream(job, backend_mgr), name=f"stream-drive-{trace_id}"
    )
    # Surface any unhandled exception from the background drain so it
    # doesn't get swallowed by asyncio's "task exception was never
    # retrieved" warning.
    task.add_done_callback(_log_background_task_failure)

    req = request if request is not None else _NoopRequest()
    try:
        async for chunk in _relay_stream(job, req):
            yield chunk
    finally:
        # The relay's own `finally:` already sets `job.disconnected`,
        # which makes the worker exit; this is a defence-in-depth
        # signal in case the relay bailed before that path.
        job.disconnected.set()


def _log_background_task_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception(f"background stream task failed: {repr(exc)}")


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    trace_id = str(uuid.uuid4())[:8]
    log.info(f"[{trace_id}] INCOMING stream={req.stream}")

    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
    }

    if req.stream:
        job = StreamJob(payload=payload, trace_id=trace_id)
        try:
            request_queue.put_nowait(job)
        except asyncio.QueueFull:
            log.error(f"[{trace_id}] QUEUE FULL (stream)")
            raise HTTPException(status_code=503, detail="Queue full")
        return StreamingResponse(
            _relay_stream(job, request),
            media_type="text/event-stream",
        )

    # Non-stream: enqueue a future-bearing job and wait for the worker.
    wrapper = FutureWrapper()
    wrapper.trace_id = trace_id
    try:
        request_queue.put_nowait((payload, wrapper))
    except asyncio.QueueFull:
        log.error(f"[{trace_id}] QUEUE FULL")
        raise HTTPException(status_code=503, detail="Queue full")

    try:
        result = await wrapper.future
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[{trace_id}] REQUEST FAILED: {repr(e)}")
        raise HTTPException(status_code=503, detail="LLM upstream timeout")

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
                    "assistant": assistant_response,
                },
                ensure_ascii=False,
            )
        )
    except Exception as e:
        log.exception(f"[{trace_id}] CHAT LOGGING FAILED: {repr(e)}")

    return result

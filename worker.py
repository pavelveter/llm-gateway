from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException
from starlette.requests import Request

from backend import (
    BackendError,
    BackendManager,
    BackendRateLimit,
    BackendTimeout,
    aggregate_errors,
)

log = logging.getLogger("system")

__all__ = [
    "FutureWrapper",
    "StreamJob",
    "worker",
    "drive_stream",
    "_relay_stream",
]


# ── queue job wrappers ──────────────────────────────────────


class FutureWrapper:
    """Thin wrapper around an asyncio Future with a trace_id."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.future: asyncio.Future = self.loop.create_future()
        self.trace_id: str = str(uuid.uuid4())[:8]

    def set_result(self, value: object) -> None:
        if not self.future.done():
            self.future.set_result(value)

    def set_exception(self, exc: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(exc)


class StreamJob:
    """Job wrapper for streaming requests."""

    _QUEUE_MAX = 256
    _PUSH_TIMEOUT = 2.0

    def __init__(self, payload: dict, trace_id: str | None = None) -> None:
        self.payload = payload
        self.trace_id: str = trace_id or str(uuid.uuid4())[:8]
        self.chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAX)
        self.disconnected: asyncio.Event = asyncio.Event()
        self.error: Exception | None = None


# ── worker ───────────────────────────────────────────────────


async def worker(
    request_queue: asyncio.Queue,
    backend_mgr: BackendManager,
    stop_event: asyncio.Event,
    *,
    stream_concurrency: int = 20,
) -> None:
    stream_concurrency = max(stream_concurrency, 1)
    stream_sem = asyncio.Semaphore(stream_concurrency)
    active_drivers: set[asyncio.Task] = set()

    try:
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            try:
                if isinstance(item, StreamJob):
                    log.info(
                        "[%s] DISPATCH STREAM (active=%d/%d)",
                        item.trace_id, len(active_drivers), stream_concurrency,
                    )
                    task = asyncio.create_task(
                        _supervised_drive_stream(item, backend_mgr, stream_sem),
                        name=f"stream-driver-{item.trace_id}",
                    )
                    active_drivers.add(task)
                    task.add_done_callback(
                        functools.partial(
                            _on_driver_done,
                            drivers=active_drivers,
                            trace_id=item.trace_id,
                        )
                    )
                else:
                    payload, fut = item  # type: ignore[misc]
                    await _drive_request(payload, fut, backend_mgr)  # type: ignore[arg-type]
            except Exception as e:
                log.exception("[worker] FATAL: %s", repr(e))
                if isinstance(item, StreamJob):
                    item.error = e
                    try:
                        item.chunk_queue.put_nowait(None)
                    except asyncio.QueueFull:  # pragma: no cover
                        item.disconnected.set()
                elif isinstance(item, tuple) and len(item) == 2:
                    fut = item[1]
                    if hasattr(fut, "set_exception") and not fut.future.done():
                        fut.set_exception(e)
            finally:
                request_queue.task_done()
    finally:
        snapshot = list(active_drivers)
        if snapshot:
            log.info(
                "Shutting down: cancelling %d active stream drivers",
                len(snapshot),
            )
            for t in snapshot:
                t.cancel()
            await asyncio.gather(*snapshot, return_exceptions=True)


def _on_driver_done(
    task: asyncio.Task, *, drivers: set, trace_id: str
) -> None:
    drivers.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception("[%s] stream driver crashed: %s", trace_id, repr(exc))


# ── non-stream path ──────────────────────────────────────────


async def _drive_request(
    payload: dict, fut: FutureWrapper, backend_mgr: BackendManager
) -> None:
    ordered = backend_mgr.ordered_backends(jitter=True)
    errors: list[Exception] = []

    for backend in ordered:
        name: str = backend["name"]

        if backend_mgr.is_on_cooldown(name):
            log.warning("[%s] SKIP %s — cooldown active", fut.trace_id, name)
            continue

        try:
            log.info("[%s] → %s", fut.trace_id, name)
            result = await backend_mgr.call(backend, payload, fut.trace_id)
            backend_mgr.mark_success(name)
            fut.set_result(result)
            log.info("[%s] SUCCESS %s", fut.trace_id, name)
            return

        except BackendTimeout as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=10.0)
            log.warning(
                "[%s] TIMEOUT %s (failures=%d)",
                fut.trace_id, name, backend_mgr.failures[name],
            )

        except BackendRateLimit as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] RATE-LIMITED %s (failures=%d)",
                fut.trace_id, name, backend_mgr.failures[name],
            )

        except BackendError as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] ERROR %s: %s (failures=%d)",
                fut.trace_id, name, e, backend_mgr.failures[name],
            )

        except Exception as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] UNEXPECTED %s: %s (failures=%d)",
                fut.trace_id, name, type(e).__name__, backend_mgr.failures[name],
            )

    status_code, detail, retry_after = aggregate_errors(errors)
    log.error(
        "[%s] ALL BACKENDS FAILED status=%d errors=%s",
        fut.trace_id, status_code, [type(e).__name__ for e in errors],
    )
    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    fut.set_exception(
        HTTPException(
            status_code=status_code,
            detail=detail,
            headers=headers,
        )
    )


# ── stream path ──────────────────────────────────────────────


async def drive_stream(
    job: StreamJob, backend_mgr: BackendManager
) -> None:
    await _drive_stream(job, backend_mgr)


async def _supervised_drive_stream(
    job: StreamJob,
    backend_mgr: BackendManager,
    sem: asyncio.Semaphore,
) -> None:
    try:
        async with sem:
            log.info("[%s] STREAM DRIVER ACQUIRED SLOT", job.trace_id)
            await _drive_stream(job, backend_mgr)
    except asyncio.CancelledError:
        log.warning("[%s] STREAM DRIVER CANCELLED (shutdown)", job.trace_id)
        job.disconnected.set()
        raise


async def _drive_stream(job: StreamJob, backend_mgr: BackendManager) -> None:
    ordered = backend_mgr.ordered_backends(jitter=True)
    errors: list[Exception] = []

    for backend in ordered:
        name: str = backend["name"]

        if backend_mgr.is_on_cooldown(name):
            log.warning("[%s] SKIP %s — cooldown active", job.trace_id, name)
            continue

        if job.disconnected.is_set():
            log.warning(
                "[%s] CLIENT DISCONNECTED before %s — skipping", job.trace_id, name,
            )
            _send_sentinel(job)
            return

        try:
            log.info("[%s] STREAM → %s", job.trace_id, name)
            async for line in backend_mgr.call_stream(
                backend, job.payload, job.trace_id
            ):
                if job.disconnected.is_set():
                    log.warning(
                        "[%s] CLIENT DISCONNECTED, aborting upstream %s",
                        job.trace_id, name,
                    )
                    _send_sentinel(job)
                    return

                chunk = f"{line}\n" if line else "\n"
                try:
                    await asyncio.wait_for(
                        job.chunk_queue.put(chunk),
                        timeout=StreamJob._PUSH_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "[%s] PUSH TIMEOUT, aborting upstream %s",
                        job.trace_id, name,
                    )
                    _send_sentinel(job)
                    return

            backend_mgr.mark_success(name)
            log.info("[%s] STREAM SUCCESS %s", job.trace_id, name)
            _send_sentinel(job)
            return

        except BackendTimeout as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=10.0)
            log.warning(
                "[%s] STREAM TIMEOUT %s (failures=%d)",
                job.trace_id, name, backend_mgr.failures[name],
            )
        except BackendRateLimit as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] STREAM RATE-LIMITED %s (failures=%d)",
                job.trace_id, name, backend_mgr.failures[name],
            )
        except BackendError as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] STREAM ERROR %s: %s (failures=%d)",
                job.trace_id, name, e, backend_mgr.failures[name],
            )
        except Exception as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                "[%s] STREAM UNEXPECTED %s: %s (failures=%d)",
                job.trace_id, name, type(e).__name__, backend_mgr.failures[name],
            )

    status_code, detail, retry_after = aggregate_errors(errors)
    log.error(
        "[%s] STREAM ALL BACKENDS FAILED status=%d errors=%s",
        job.trace_id, status_code, [type(e).__name__ for e in errors],
    )
    error_obj: dict = {
        "error": {
            "message": detail,
            "type": "upstream_error",
            "code": status_code,
        }
    }
    if retry_after:
        error_obj["error"]["retry_after"] = retry_after
    try:
        job.chunk_queue.put_nowait(f"data: {json.dumps(error_obj)}\n\n")
        job.chunk_queue.put_nowait("data: [DONE]\n\n")
    except asyncio.QueueFull:  # pragma: no cover
        pass
    job.error = BackendError(detail)
    _send_sentinel(job)


def _send_sentinel(job: StreamJob) -> None:
    try:
        job.chunk_queue.put_nowait(None)
    except asyncio.QueueFull:
        job.disconnected.set()


# ── relay (consumed by the HTTP layer) ───────────────────────


async def _relay_stream(
    job: StreamJob, request: Request
) -> AsyncIterator[str]:
    try:
        while True:
            if await request.is_disconnected():
                job.disconnected.set()
                log.warning("[%s] CLIENT DISCONNECTED (relay-side)", job.trace_id)
                return

            try:
                chunk = await asyncio.wait_for(
                    job.chunk_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                return

            yield chunk
    finally:
        job.disconnected.set()

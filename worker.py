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


# ── queue job wrappers ──────────────────────────────────────


class FutureWrapper:
    """Thin wrapper around an asyncio Future with a trace_id.

    Used for non-streaming requests.  The HTTP handler awaits
    ``future``; the worker resolves it with ``set_result`` /
    ``set_exception``.
    """

    def __init__(self) -> None:
        # `get_running_loop` is the Python ≥3.10 way; works under
        # FastAPI/uvicorn and pytest-asyncio where a loop is alive.
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
    """Job wrapper for streaming requests.

    The worker drives ``call_stream`` and pushes each line into
    ``chunk_queue``.  The HTTP relay reads from ``chunk_queue`` and
    forwards chunks to the client.  When the relay exits (client
    disconnect, normal end, error) it sets ``disconnected`` so the
    worker can abort the open upstream stream without marking the
    backend as failed.
    """

    # Bounded so a stale (dead-relay) job can't grow forever; chosen
    # large enough that real SSE traffic flows under backpressure but
    # small enough that a stuck client doesn't stockpile megabytes.
    _QUEUE_MAX = 256

    # If the worker can't push a chunk within this many seconds, the
    # client relay is assumed dead and the upstream is aborted.
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
    """Pull requests from the queue and try backends sequentially.

    Handles two job shapes:

    * ``(payload, FutureWrapper)`` — non-streaming; the worker
      resolves the future inline (typically fast).
    * ``StreamJob`` — streaming.  Stream generation can run for tens
      of seconds, so the worker does NOT block on it; it spawns a
      *driver* coroutine under a semaphore and moves on.  This
      decouples dispatcher capacity (workers) from upstream-bound
      concurrency (drivers), so a burst of slow streams can't pin
      every worker and starve incoming non-stream requests of
      dispatcher capacity.

    ``stream_concurrency`` caps the number of driver coroutines that
    may be reading from an upstream concurrently; further stream
    requests sit in the queue and wait their turn at the semaphore.

    On timeout the failed backend gets a shorter cooldown (10 s);
    other failures get the standard 15 s.  If *all* backends fail the
    request is answered with an appropriate HTTP status.
    """
    # Clamp so callers that pass 0 (e.g. from a misconfigured env
    # var) don't get an `asyncio.Semaphore(0)` -- which would block
    # forever waiting for a release that never comes.
    stream_concurrency = max(stream_concurrency, 1)
    stream_sem = asyncio.Semaphore(stream_concurrency)
    # Track detached drivers so we can cancel them on shutdown rather
    # than leaking tasks that hold open upstream sockets.
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
                        f"[{item.trace_id}] DISPATCH STREAM (active={len(active_drivers)}/"
                        f"{stream_concurrency})"
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
                    # Worker is immediately free for the next queue item.
                else:
                    payload, fut = item  # type: ignore[misc]
                    await _drive_request(payload, fut, backend_mgr)  # type: ignore[arg-type]
            except Exception as e:
                log.exception(f"[worker] FATAL: {repr(e)}")
                if isinstance(item, StreamJob):
                    # best effort — wake the relay with a final error marker
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
        # Whether we exited via stop_event (graceful) or via
        # task.cancel() (lifespan teardown), don't strand driver
        # tasks holding upstream sockets open.  Cancel them; the
        # supervised wrapper translates that into a relay-side
        # disconnect so the HTTP handler exits gracefully.
        #
        # Snapshot the set first: ``_on_driver_done`` mutates
        # ``active_drivers`` synchronously when a driver completes
        # during cancellation, and CPython raises ``RuntimeError:
        # Set changed size during iteration`` if we iterate the live
        # set while callbacks are discarding from underneath us.
        snapshot = list(active_drivers)
        if snapshot:
            log.info(
                f"Shutting down: cancelling {len(snapshot)} active stream drivers"
            )
            for t in snapshot:
                t.cancel()
            await asyncio.gather(*snapshot, return_exceptions=True)


def _on_driver_done(
    task: asyncio.Task, *, drivers: set, trace_id: str
) -> None:
    """Per-driver done callback: drop from active set, surface crashes.

    Always called by the event loop once the task transitions to a
    final state.  We use ``functools.partial`` at registration time so
    the per-task ``trace_id`` and ``drivers`` set are bound at call
    time, not at callback time (avoiding the classic late-binding pitfall).
    """
    drivers.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.exception(f"[{trace_id}] stream driver crashed: {repr(exc)}")


# ── non-stream path ──────────────────────────────────────────


async def _drive_request(
    payload: dict, fut: FutureWrapper, backend_mgr: BackendManager
) -> None:
    """Try backends in order for a non-streaming request.

    Uses the jittered backend ordering so a burst of simultaneous
    worker dispatches doesn't all hit the top-ranked backend and
    trigger a 429 storm upstream.
    """
    ordered = backend_mgr.ordered_backends(jitter=True)
    errors: list[Exception] = []

    for backend in ordered:
        name: str = backend["name"]

        if backend_mgr.is_on_cooldown(name):
            log.warning(f"[{fut.trace_id}] SKIP {name} — cooldown active")
            continue

        try:
            log.info(f"[{fut.trace_id}] → {name}")
            result = await backend_mgr.call(backend, payload, fut.trace_id)
            backend_mgr.mark_success(name)
            fut.set_result(result)
            log.info(f"[{fut.trace_id}] SUCCESS {name}")
            return

        except BackendTimeout as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=10.0)
            log.warning(
                f"[{fut.trace_id}] TIMEOUT {name} "
                f"(failures={backend_mgr.failures[name]})"
            )

        except BackendRateLimit as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{fut.trace_id}] RATE-LIMITED {name} "
                f"(failures={backend_mgr.failures[name]})"
            )

        except BackendError as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{fut.trace_id}] ERROR {name}: {e} "
                f"(failures={backend_mgr.failures[name]})"
            )

        except Exception as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{fut.trace_id}] UNEXPECTED {name}: {repr(e)} "
                f"(failures={backend_mgr.failures[name]})"
            )

    # for-loop completed without an early return → all backends failed.
    status_code, detail, retry_after = aggregate_errors(errors)
    log.error(
        f"[{fut.trace_id}] ALL BACKENDS FAILED "
        f"status={status_code} errors={[repr(e) for e in errors]}"
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
    """Public alias of :func:`_drive_stream` for callers outside this module.

    Runs the stream drive *inline* — useful for tests, the legacy
    ``stream_backend`` helper, and ad-hoc scripts.  Production callers
    go through :func:`worker`, which detaches the drive into a
    background task so dispatcher capacity isn't tied to stream
    duration.
    """
    await _drive_stream(job, backend_mgr)


async def _supervised_drive_stream(
    job: StreamJob,
    backend_mgr: BackendManager,
    sem: asyncio.Semaphore,
) -> None:
    """Wrap :func:`_drive_stream` in a semaphore slot.

    * Acquires a slot before talking to the upstream so concurrent
      stream count is bounded.
    * Translates the worker's ``cancel()`` (shutdown path) into a
      relay-side disconnect before propagating: without this the
      upstream HTTP client keeps reading and ``aclose`` only lands
      when the generator is GC'd.
    """
    try:
        async with sem:
            log.info(f"[{job.trace_id}] STREAM DRIVER ACQUIRED SLOT")
            await _drive_stream(job, backend_mgr)
    except asyncio.CancelledError:
        # Server-level shutdown.  Signal relay so SSE response closes
        # *now*, then let the task end.
        log.warning(f"[{job.trace_id}] STREAM DRIVER CANCELLED (shutdown)")
        job.disconnected.set()
        raise


async def _drive_stream(job: StreamJob, backend_mgr: BackendManager) -> None:
    """Try backends in order for a streaming job.

    Pushes raw SSE lines (with trailing ``\\n``) into ``job.chunk_queue``
    and a ``None`` sentinel at the end.  If every backend fails, queues
    a final SSE error event followed by ``[DONE]`` so the client sees a
    well-formed stream termination.

    On a client disconnect (``job.disconnected`` set) the worker aborts
    the upstream stream and returns WITHOUT calling ``mark_failure``:
    the upstream itself is healthy, the user just left — penalising the
    backend for that would degrade the pool unfairly.
    """
    ordered = backend_mgr.ordered_backends(jitter=True)
    errors: list[Exception] = []

    for backend in ordered:
        name: str = backend["name"]

        if backend_mgr.is_on_cooldown(name):
            log.warning(f"[{job.trace_id}] SKIP {name} — cooldown active")
            continue

        # If the relay already told us the client is gone, skip this
        # backend entirely rather than opening a doomed upstream socket.
        if job.disconnected.is_set():
            log.warning(
                f"[{job.trace_id}] CLIENT DISCONNECTED before {name} — skipping"
            )
            _send_sentinel(job)
            return

        try:
            log.info(f"[{job.trace_id}] STREAM → {name}")
            async for line in backend_mgr.call_stream(
                backend, job.payload, job.trace_id
            ):
                if job.disconnected.is_set():
                    # Disconnect path: upstream looks healthy, do NOT
                    # penalise the backend — just stop the read loop.
                    log.warning(
                        f"[{job.trace_id}] CLIENT DISCONNECTED, aborting upstream {name}"
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
                    # Queue stalled → relay isn't draining anymore.
                    log.warning(
                        f"[{job.trace_id}] PUSH TIMEOUT, aborting upstream {name}"
                    )
                    _send_sentinel(job)
                    return

            # Stream completed cleanly; record latency and end the job.
            backend_mgr.mark_success(name)
            log.info(f"[{job.trace_id}] STREAM SUCCESS {name}")
            _send_sentinel(job)
            return

        except BackendTimeout as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=10.0)
            log.warning(
                f"[{job.trace_id}] STREAM TIMEOUT {name} "
                f"(failures={backend_mgr.failures[name]})"
            )
        except BackendRateLimit as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{job.trace_id}] STREAM RATE-LIMITED {name} "
                f"(failures={backend_mgr.failures[name]})"
            )
        except BackendError as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{job.trace_id}] STREAM ERROR {name}: {e} "
                f"(failures={backend_mgr.failures[name]})"
            )
        except Exception as e:
            errors.append(e)
            backend_mgr.mark_failure(name, cooldown=15.0)
            log.warning(
                f"[{job.trace_id}] STREAM UNEXPECTED {name}: {repr(e)} "
                f"(failures={backend_mgr.failures[name]})"
            )

    # All backends failed: emit a final SSE error event so the client
    # sees proper status + a clean DONE marker.
    status_code, detail, retry_after = aggregate_errors(errors)
    log.error(
        f"[{job.trace_id}] STREAM ALL BACKENDS FAILED "
        f"status={status_code} errors={[repr(e) for e in errors]}"
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
    """Best-effort push of the end-of-stream marker.

    Falls back to setting ``job.disconnected`` if the chunk queue is
    saturated so the relay bails out even if the sentinel never lands.
    """
    try:
        job.chunk_queue.put_nowait(None)
    except asyncio.QueueFull:
        # Sentinel lost in noise — but relay's `finally` already
        # sets this event anyway, so the call is best-effort cleanup.
        job.disconnected.set()


# ── relay (consumed by the HTTP layer) ───────────────────────


async def _relay_stream(
    job: StreamJob, request: Request
) -> AsyncIterator[str]:
    """Forward chunks from ``job.chunk_queue`` to the SSE response.

    Polls ``request.is_disconnected()`` between chunks with a 1-second
    timeout so disconnects surface quickly even if no chunks are
    flowing.  Always sets ``job.disconnected`` on exit so workers
    abort their open upstream streams.
    """
    try:
        while True:
            if await request.is_disconnected():
                job.disconnected.set()
                log.warning(
                    f"[{job.trace_id}] CLIENT DISCONNECTED (relay-side)"
                )
                return

            try:
                chunk = await asyncio.wait_for(
                    job.chunk_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                # Worker is done with this job.
                return

            yield chunk
    finally:
        # Guarantee the worker sees the disconnect on every exit path
        # (client gone, normal end, worker signaled error, exception).
        job.disconnected.set()

import asyncio
import time as time_module
from collections import deque
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend import (
    BackendError,
    BackendManager,
    BackendNetworkError,
    BackendRateLimit,
    BackendServerError,
    BackendTimeout,
)
from worker import FutureWrapper, StreamJob, drive_stream, worker


# ── helpers ──────────────────────────────────────────────────


def _make_mgr() -> BackendManager:
    m = BackendManager()
    m.load(
        [
            ("backend-1", "https://api1.example.com/v1", "key-aaaa"),
            ("backend-2", "https://api2.example.com/v1", "key-bbbb"),
        ]
    )
    return m


def _drain_queue(queue: asyncio.Queue) -> list:
    out: list = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── FutureWrapper ────────────────────────────────────────────


class TestFutureWrapper:
    @pytest.mark.asyncio
    async def test_set_result(self) -> None:
        fw = FutureWrapper()
        fw.set_result({"ok": True})
        assert fw.future.done()
        assert fw.future.result() == {"ok": True}

    @pytest.mark.asyncio
    async def test_set_exception(self) -> None:
        fw = FutureWrapper()
        exc = HTTPException(status_code=500, detail="boom")
        fw.set_exception(exc)
        assert fw.future.done()
        with pytest.raises(HTTPException) as ctx:
            fw.future.result()
        assert ctx.value.status_code == 500

    @pytest.mark.asyncio
    async def test_set_result_twice_is_idempotent(self) -> None:
        fw = FutureWrapper()
        fw.set_result({"first": True})
        fw.set_result({"second": True})
        assert fw.future.result() == {"first": True}

    @pytest.mark.asyncio
    async def test_set_exception_twice_is_idempotent(self) -> None:
        fw = FutureWrapper()
        fw.set_exception(HTTPException(status_code=400, detail="a"))
        fw.set_exception(HTTPException(status_code=500, detail="b"))
        with pytest.raises(HTTPException) as ctx:
            fw.future.result()
        assert ctx.value.status_code == 400

    @pytest.mark.asyncio
    async def test_set_result_blocks_set_exception(self) -> None:
        fw = FutureWrapper()
        fw.set_result({"ok": True})
        fw.set_exception(HTTPException(status_code=500, detail="boom"))
        assert fw.future.result() == {"ok": True}

    @pytest.mark.asyncio
    async def test_set_exception_blocks_set_result(self) -> None:
        fw = FutureWrapper()
        fw.set_exception(HTTPException(status_code=400))
        fw.set_result({"ok": True})
        with pytest.raises(HTTPException):
            fw.future.result()

    @pytest.mark.asyncio
    async def test_trace_id_is_non_empty_string(self) -> None:
        fw1 = FutureWrapper()
        fw2 = FutureWrapper()
        assert isinstance(fw1.trace_id, str)
        assert len(fw1.trace_id) > 0
        assert fw1.trace_id != fw2.trace_id


# ── StreamJob ────────────────────────────────────────────────


class TestStreamJob:
    def test_default_trace_id_is_random(self) -> None:
        s1 = StreamJob({"foo": 1})
        s2 = StreamJob({"foo": 2})
        assert s1.trace_id != s2.trace_id
        assert len(s1.trace_id) > 0

    def test_provided_trace_id_used(self) -> None:
        s = StreamJob({"foo": 1}, trace_id="custom-id")
        assert s.trace_id == "custom-id"

    def test_initial_state(self) -> None:
        s = StreamJob({"foo": 1})
        assert s.payload == {"foo": 1}
        assert s.chunk_queue.empty()
        assert not s.disconnected.is_set()
        assert s.error is None

    def test_chunk_queue_is_bounded(self) -> None:
        s = StreamJob({})
        assert s.chunk_queue.maxsize == StreamJob._QUEUE_MAX


# ── worker ───────────────────────────────────────────────────


class TestWorker:
    @pytest.mark.asyncio
    async def test_success_first_backend(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        response = {"choices": [{"message": {"content": "hello"}}]}
        mgr.call = AsyncMock(return_value=response)  # type: ignore[method-assign]

        fut = FutureWrapper()
        fut.trace_id = "test-001"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        result = await asyncio.wait_for(fut.future, timeout=2.0)
        assert result == response
        assert mgr.failures["backend-1"] == 0

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_fail_first_succeed_second(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        response = {"choices": [{"message": {"content": "fallback"}}]}
        call_mock = AsyncMock()
        call_mock.side_effect = [BackendTimeout("timeout"), response]
        mgr.call = call_mock  # type: ignore[method-assign]

        fut = FutureWrapper()
        fut.trace_id = "test-002"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        result = await asyncio.wait_for(fut.future, timeout=2.0)
        assert result == response
        # One backend failed, one succeeded (order depends on jitter)
        total_failures = sum(mgr.failures.values())
        assert total_failures == 1

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_all_timeout_returns_504(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendTimeout("timeout")
        )

        fut = FutureWrapper()
        fut.trace_id = "test-003"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 504
        assert "timed out" in ctx.value.detail.lower()
        assert mgr.failures["backend-1"] == 1
        assert mgr.failures["backend-2"] == 1

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_all_rate_limited_returns_429_with_retry_after(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                BackendRateLimit("rl1", retry_after=15),
                BackendRateLimit("rl2", retry_after=30),
            ]
        )

        fut = FutureWrapper()
        fut.trace_id = "test-004"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 429
        assert ctx.value.headers.get("Retry-After") == "30"

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_all_server_error_returns_502(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                BackendServerError("e1", status_code=500),
                BackendServerError("e2", status_code=503),
            ]
        )

        fut = FutureWrapper()
        fut.trace_id = "test-005"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 502
        assert "Retry-After" not in ctx.value.headers

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_all_network_error_returns_502(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendNetworkError("dns fail")
        )

        fut = FutureWrapper()
        fut.trace_id = "test-006"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 502
        assert "unreachable" in ctx.value.detail.lower()

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_mixed_errors_returns_503(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                BackendTimeout("timeout"),
                BackendRateLimit("rate-limited"),
            ]
        )

        fut = FutureWrapper()
        fut.trace_id = "test-007"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 503

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_skip_backend_on_cooldown(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.cooldowns["backend-1"] = time_module.time() + 999
        response = {"choices": [{"message": {"content": "only-backend-2"}}]}
        mgr.call = AsyncMock(return_value=response)  # type: ignore[method-assign]

        fut = FutureWrapper()
        fut.trace_id = "test-008"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        result = await asyncio.wait_for(fut.future, timeout=2.0)
        assert result == response

        assert mgr.call.call_count == 1
        call_args = mgr.call.call_args[0]
        assert call_args[0]["name"] == "backend-2"

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_unexpected_exception_marked_as_failure(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError("something unexpected")
        )

        fut = FutureWrapper()
        fut.trace_id = "test-009"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 503
        assert mgr.failures["backend-1"] == 1

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_zero_retry_after_not_included_in_headers(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendRateLimit("rl", retry_after=0)
        )

        fut = FutureWrapper()
        fut.trace_id = "test-010"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        with pytest.raises(HTTPException) as ctx:
            await asyncio.wait_for(fut.future, timeout=2.0)

        assert ctx.value.status_code == 429
        assert "Retry-After" not in ctx.value.headers

        await _cancel(task)

    @pytest.mark.asyncio
    async def test_only_usable_backend_after_3_failures(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        mgr.failures["backend-1"] = 3
        response = {"choices": [{"message": {"content": "only-2"}}]}
        mgr.call = AsyncMock(return_value=response)  # type: ignore[method-assign]

        fut = FutureWrapper()
        fut.trace_id = "test-011"
        queue.put_nowait(({"model": "gpt-4o", "messages": []}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))

        result = await asyncio.wait_for(fut.future, timeout=2.0)
        assert result == response
        assert mgr.call.call_count == 1
        assert mgr.call.call_args[0][0]["name"] == "backend-2"

        await _cancel(task)


# ── worker / jitter integration ──────────────────────────────


class TestWorkerUsesJitteredOrdering:
    @pytest.mark.asyncio
    async def test_worker_passes_jitter_true_to_ordered_backends(self) -> None:
        mgr = _make_mgr()
        mgr.call = AsyncMock(return_value={"ok": True})  # type: ignore[method-assign]

        captured_kwargs: list[dict] = []
        original_ordered = mgr.ordered_backends

        def spy(*args, **kwargs):
            captured_kwargs.append(kwargs.get("jitter", False))
            return original_ordered(*args, **kwargs)

        mgr.ordered_backends = spy  # type: ignore[method-assign]

        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        fut = FutureWrapper()
        fut.trace_id = "jitter-001"
        queue.put_nowait(({"x": 1}, fut))

        stop = asyncio.Event()
        task = asyncio.create_task(worker(queue, mgr, stop))
        await asyncio.wait_for(fut.future, timeout=2.0)
        await _cancel(task)

        assert captured_kwargs, "ordered_backends was not called"
        assert all(kwargs is True for kwargs in captured_kwargs), (
            f"Worker should always request jittered ordering; saw {captured_kwargs}"
        )


# ── drive_stream ─────────────────────────────────────────────


class TestDriveStream:
    @pytest.mark.asyncio
    async def test_success_pushes_lines_and_sentinel(self) -> None:
        mgr = _make_mgr()

        async def mock_stream(backend, payload, trace_id):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield ""
            yield "data: [DONE]"

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        job = StreamJob({"x": 1}, trace_id="ds-001")
        await drive_stream(job, mgr)

        chunks = _drain_queue(job.chunk_queue)
        assert chunks[-1] is None
        body = "".join(chunks[:-1])
        assert '"content":"hi"' in body
        assert "[DONE]" in body
        assert mgr.failures["backend-1"] == 0

    @pytest.mark.asyncio
    async def test_skips_cooldown_backend(self) -> None:
        mgr = _make_mgr()
        mgr.cooldowns["backend-1"] = time_module.time() + 999

        called: list[str] = []

        async def mock_stream(backend, payload, trace_id):
            called.append(backend["name"])
            yield 'data: line'
            yield "data: [DONE]"

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        job = StreamJob({"x": 1}, trace_id="ds-cd")
        await drive_stream(job, mgr)

        assert called == ["backend-2"]

    @pytest.mark.asyncio
    async def test_disconnect_set_before_dispatch_skips_backends(self) -> None:
        mgr = _make_mgr()

        async def should_not_run(backend, payload, trace_id):
            raise AssertionError("call_stream must not run if client is gone")

        mgr.call_stream = should_not_run  # type: ignore[method-assign]

        job = StreamJob({"x": 1}, trace_id="ds-pre")
        job.disconnected.set()
        await drive_stream(job, mgr)

        assert mgr.failures["backend-1"] == 0
        assert mgr.failures["backend-2"] == 0

    @pytest.mark.asyncio
    async def test_disconnect_during_iteration_aborts_no_failure(self) -> None:
        mgr = _make_mgr()
        job = StreamJob({"x": 1}, trace_id="ds-disc")

        async def mock_stream(backend, payload, trace_id):
            yield 'data: line1'
            job.disconnected.set()
            yield 'data: line2'
            yield 'data: line3'

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        await drive_stream(job, mgr)

        chunks = _drain_queue(job.chunk_queue)
        body = "".join(c for c in chunks if c is not None)
        assert "line1" in body
        assert "line2" not in body
        assert "line3" not in body
        assert chunks[-1] is None

        assert mgr.failures["backend-1"] == 0
        assert mgr.failures["backend-2"] == 0

    @pytest.mark.asyncio
    async def test_push_timeout_aborts_no_failure(self) -> None:
        mgr = _make_mgr()
        job = StreamJob({"x": 1}, trace_id="ds-push")

        for _ in range(StreamJob._QUEUE_MAX):
            job.chunk_queue.put_nowait("stuck")

        async def mock_stream(backend, payload, trace_id):
            yield 'data: line'
            yield 'data: line2'

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        await drive_stream(job, mgr)

        assert mgr.failures["backend-1"] == 0

    @pytest.mark.asyncio
    async def test_all_backends_fail_emits_error_sse_event(self) -> None:
        mgr = _make_mgr()

        async def mock_stream(backend, payload, trace_id):
            yield
            raise BackendTimeout(f"{backend['name']}: timeout")

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        job = StreamJob({"x": 1}, trace_id="ds-fail")
        await drive_stream(job, mgr)

        chunks = _drain_queue(job.chunk_queue)
        body = "".join(c for c in chunks if c is not None)

        assert "[DONE]" in body
        assert ('"code": 504' in body) or ('"code":504' in body)
        assert "timed out" in body.lower()

        assert mgr.failures["backend-1"] == 1
        assert mgr.failures["backend-2"] == 1


# ── worker / detach architecture ─────────────────────────────


class TestWorkerDetachesStreams:
    @pytest.mark.asyncio
    async def test_worker_does_not_block_on_stream(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        stream_acquired = asyncio.Event()
        stream_can_finish = asyncio.Event()

        async def mock_stream(backend, payload, trace_id):
            stream_acquired.set()
            await stream_can_finish.wait()
            yield 'data: line'
            yield "data: [DONE]"

        mgr.call_stream = mock_stream  # type: ignore[method-assign]
        mgr.call = AsyncMock(return_value={"ok": True})  # type: ignore[method-assign]

        queue.put_nowait(StreamJob({"x": 1}, trace_id="detach-slow"))

        stop = asyncio.Event()
        w_task = asyncio.create_task(
            worker(queue, mgr, stop, stream_concurrency=2)
        )

        await stream_acquired.wait()

        fut = FutureWrapper()
        fut.trace_id = "ns-fast"
        queue.put_nowait(({"y": 2}, fut))

        result = await asyncio.wait_for(fut.future, timeout=1.0)
        assert result == {"ok": True}

        stream_can_finish.set()
        stop.set()
        await w_task

    @pytest.mark.asyncio
    async def test_stream_concurrency_cap_serializes_drivers(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        s1_entered = asyncio.Event()
        s1_can_finish = asyncio.Event()
        s2_entered = asyncio.Event()

        async def mock_stream(backend, payload, trace_id):
            if trace_id == "s1":
                s1_entered.set()
                await s1_can_finish.wait()
            else:
                s2_entered.set()
            yield 'data: line'
            yield "data: [DONE]"

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        queue.put_nowait(StreamJob({"x": "1"}, trace_id="s1"))
        queue.put_nowait(StreamJob({"x": "2"}, trace_id="s2"))

        stop = asyncio.Event()
        w_task = asyncio.create_task(
            worker(queue, mgr, stop, stream_concurrency=1)
        )

        await s1_entered.wait()

        await asyncio.sleep(0)
        assert not s2_entered.is_set(), (
            "s2 entered despite stream_concurrency=1"
        )

        s1_can_finish.set()
        await s2_entered.wait()

        stop.set()
        await w_task

    @pytest.mark.asyncio
    async def test_shutdown_cancels_active_drivers(self) -> None:
        mgr = _make_mgr()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        stream_started = asyncio.Event()

        async def mock_stream(backend, payload, trace_id):
            stream_started.set()
            yield 'data: line'
            await asyncio.sleep(5.0)
            yield 'data: line2'

        mgr.call_stream = mock_stream  # type: ignore[method-assign]

        queue.put_nowait(StreamJob({"x": 1}, trace_id="shutdown"))

        stop = asyncio.Event()
        w_task = asyncio.create_task(
            worker(queue, mgr, stop, stream_concurrency=2)
        )

        await stream_started.wait()

        stop.set()
        await asyncio.wait_for(w_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_driver_callback_cleans_up_active_set(self) -> None:
        import functools
        from worker import _on_driver_done

        drivers: set = set()

        async def boom():
            raise RuntimeError("kaboom")

        task = asyncio.create_task(boom())
        drivers.add(task)
        task.add_done_callback(
            functools.partial(
                _on_driver_done, drivers=drivers, trace_id="boom-1"
            )
        )

        with pytest.raises(RuntimeError, match="kaboom"):
            await task

        assert task not in drivers
        assert isinstance(task.exception(), RuntimeError)

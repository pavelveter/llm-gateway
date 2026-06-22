import asyncio
import json
import time as time_module
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from backend import BackendManager
from worker import worker as worker_fn

import gateway as gw


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset global state before each test so tests don't leak."""
    from config import QUEUE_MAX

    gw.backend_mgr = BackendManager()
    gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient — does NOT trigger startup events."""
    return TestClient(gw.app)


@pytest.fixture
def configured_client(client: TestClient) -> TestClient:
    """Load test backends into the manager and return the client."""
    gw.backend_mgr.load(
        [
            ("backend-1", "https://api1.example.com/v1/chat/completions", "key-aaaa"),
            ("backend-2", "https://api2.example.com/v1/chat/completions", "key-bbbb"),
        ]
    )
    return client


def _spawn_worker() -> tuple[asyncio.Task, asyncio.Event]:
    """Spawn a worker draining the gateway queue; return (task, stop_event)."""
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        worker_fn(gw.request_queue, gw.backend_mgr, stop_event)
    )
    return task, stop_event


async def _drain_worker(task: asyncio.Task, stop_event: asyncio.Event) -> None:
    stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── health ───────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_shows_no_backends_when_none_loaded(
        self, client: TestClient
    ) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["backends"] == []

    def test_health_shows_loaded_backends(
        self, configured_client: TestClient
    ) -> None:
        resp = configured_client.get("/health")
        assert resp.status_code == 200
        backends = resp.json()["backends"]
        assert len(backends) == 2
        assert {b["name"] for b in backends} == {"backend-1", "backend-2"}

    def test_health_reflects_failures(
        self, configured_client: TestClient
    ) -> None:
        gw.backend_mgr.failures["backend-1"] = 3
        resp = configured_client.get("/health")
        entry = next(
            b for b in resp.json()["backends"] if b["name"] == "backend-1"
        )
        assert entry["failures"] == 3


# ── models ───────────────────────────────────────────────────


class TestModels:
    def test_models_returns_list(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "llm-gateway"


# ── chat completions — non-streaming ─────────────────────────


class TestChatCompletionsNonStreaming:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        from config import QUEUE_MAX

        gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)
        gw.backend_mgr = BackendManager()
        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com/v1/chat/completions", "k1"),
                ("b2", "https://api2.example.com/v1/chat/completions", "k2"),
            ]
        )

        response = {"choices": [{"message": {"content": "Hello!"}}]}
        gw.backend_mgr.call = AsyncMock(return_value=response)  # type: ignore[method-assign]

        w_task, stop_event = _spawn_worker()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert gw.backend_mgr.failures["b1"] == 0

        await _drain_worker(w_task, stop_event)

    @pytest.mark.asyncio
    async def test_all_timeout_returns_504(self) -> None:
        from config import QUEUE_MAX
        from backend import BackendTimeout

        gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)
        gw.backend_mgr = BackendManager()
        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com/v1/chat/completions", "k1"),
                ("b2", "https://api2.example.com/v1/chat/completions", "k2"),
            ]
        )
        gw.backend_mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendTimeout("timeout")
        )

        w_task, stop_event = _spawn_worker()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()

        await _drain_worker(w_task, stop_event)

    @pytest.mark.asyncio
    async def test_all_rate_limited_returns_429_with_retry_after(self) -> None:
        from config import QUEUE_MAX
        from backend import BackendRateLimit

        gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)
        gw.backend_mgr = BackendManager()
        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com/v1/chat/completions", "k1"),
                ("b2", "https://api2.example.com/v1/chat/completions", "k2"),
            ]
        )
        gw.backend_mgr.call = AsyncMock(  # type: ignore[method-assign]
            side_effect=BackendRateLimit("rl", retry_after=30)
        )

        w_task, stop_event = _spawn_worker()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "30"

        await _drain_worker(w_task, stop_event)

    def test_queue_full_returns_503(self, client: TestClient) -> None:
        from config import QUEUE_MAX

        for _ in range(QUEUE_MAX):
            gw.request_queue.put_nowait(({}, object()))

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 503
        assert "Queue full" in resp.json()["detail"]


# ── chat completions — streaming (direct stream_backend) ─────


class TestChatCompletionsStreaming:
    @pytest.mark.asyncio
    async def test_stream_backend_success(self) -> None:
        from gateway import stream_backend

        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com", "k1"),
                ("b2", "https://api2.example.com", "k2"),
            ]
        )

        async def mock_stream(backend, payload, trace_id):
            yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
            yield ""
            yield "data: [DONE]"

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        chunks: list[str] = []
        async for chunk in stream_backend(
            {"model": "gpt-4o", "messages": []}, "t1"
        ):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "hello" in output
        assert "[DONE]" in output

    @pytest.mark.asyncio
    async def test_stream_backend_first_fails_second_succeeds(self) -> None:
        from gateway import stream_backend
        from backend import BackendTimeout

        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com", "k1"),
                ("b2", "https://api2.example.com", "k2"),
            ]
        )

        call_count = 0

        async def mock_stream(backend, payload, trace_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendTimeout("b1: timeout")
            yield 'data: {"choices":[{"delta":{"content":"fallback"}}]}'
            yield ""
            yield "data: [DONE]"

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        chunks: list[str] = []
        async for chunk in stream_backend(
            {"model": "gpt-4o", "messages": []}, "t2"
        ):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "fallback" in output
        assert gw.backend_mgr.failures["b1"] == 1
        assert gw.backend_mgr.failures["b2"] == 0

    @pytest.mark.asyncio
    async def test_stream_backend_all_fail_returns_error_sse(self) -> None:
        from gateway import stream_backend
        from backend import BackendTimeout

        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com", "k1"),
                ("b2", "https://api2.example.com", "k2"),
            ]
        )

        async def mock_stream(backend, payload, trace_id):
            yield
            raise BackendTimeout(f"{backend['name']}: timeout")

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        chunks: list[str] = []
        async for chunk in stream_backend(
            {"model": "gpt-4o", "messages": []}, "t3"
        ):
            chunks.append(chunk)

        data_lines = [
            c for c in chunks if c.startswith("data:") and "DONE" not in c
        ]
        assert len(data_lines) >= 1
        error_payload = json.loads(
            data_lines[0].removeprefix("data: ").strip()
        )
        assert error_payload["error"]["code"] == 504
        assert "timed out" in error_payload["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_stream_backend_all_rate_limited_sse_includes_retry_after(
        self,
    ) -> None:
        from gateway import stream_backend
        from backend import BackendRateLimit

        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com", "k1"),
                ("b2", "https://api2.example.com", "k2"),
            ]
        )

        async def mock_stream(backend, payload, trace_id):
            yield
            raise BackendRateLimit(f"{backend['name']}: rl", retry_after=30)

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        chunks: list[str] = []
        async for chunk in stream_backend(
            {"model": "gpt-4o", "messages": []}, "t4"
        ):
            chunks.append(chunk)

        data_lines = [
            c for c in chunks if c.startswith("data:") and "DONE" not in c
        ]
        error_payload = json.loads(
            data_lines[0].removeprefix("data: ").strip()
        )
        assert error_payload["error"]["code"] == 429
        assert error_payload["error"]["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_stream_backend_skips_cooldown_backend(self) -> None:
        from gateway import stream_backend

        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com", "k1"),
                ("b2", "https://api2.example.com", "k2"),
            ]
        )
        gw.backend_mgr.cooldowns["b1"] = time_module.time() + 999

        called_backends: list[str] = []

        async def mock_stream(backend, payload, trace_id):
            called_backends.append(backend["name"])
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield ""
            yield "data: [DONE]"

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        chunks: list[str] = []
        async for chunk in stream_backend(
            {"model": "gpt-4o", "messages": []}, "t5"
        ):
            chunks.append(chunk)

        assert called_backends == ["b2"]


# ── chat completions — endpoint streaming ────────────────────


class TestChatCompletionsStreamingEndpoint:
    @pytest.mark.asyncio
    async def test_stream_request_returns_sse_mediatype(self) -> None:
        from config import QUEUE_MAX

        gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)
        gw.backend_mgr = BackendManager()
        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com/v1/chat/completions", "k1"),
                ("b2", "https://api2.example.com/v1/chat/completions", "k2"),
            ]
        )

        async def mock_stream(backend, payload, trace_id):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield ""
            yield "data: [DONE]"

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        w_task, stop_event = _spawn_worker()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "data:" in resp.text

        await _drain_worker(w_task, stop_event)

    @pytest.mark.asyncio
    async def test_stream_request_all_backends_fail(self) -> None:
        from config import QUEUE_MAX
        from backend import BackendTimeout

        gw.request_queue = asyncio.Queue(maxsize=QUEUE_MAX)
        gw.backend_mgr = BackendManager()
        gw.backend_mgr.load(
            [
                ("b1", "https://api1.example.com/v1/chat/completions", "k1"),
                ("b2", "https://api2.example.com/v1/chat/completions", "k2"),
            ]
        )

        async def mock_stream(backend, payload, trace_id):
            yield
            raise BackendTimeout(f"{backend['name']}: timeout")

        gw.backend_mgr.call_stream = mock_stream  # type: ignore[method-assign]

        w_task, stop_event = _spawn_worker()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = resp.text.split("\n")
        data_lines = [
            l.removeprefix("data: ") for l in lines if l.startswith("data:")
        ]
        error_payload = json.loads(data_lines[0])
        assert error_payload["error"]["code"] == 504
        assert "timed out" in error_payload["error"]["message"].lower()
        assert data_lines[-1].strip() == "[DONE]"

        await _drain_worker(w_task, stop_event)

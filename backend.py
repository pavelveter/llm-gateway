from __future__ import annotations

import random
import time
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator

import httpx
from aiolimiter import AsyncLimiter

from config import RPM_LIMIT

log = logging.getLogger("system")

__all__ = [
    "BackendError",
    "BackendTimeout",
    "BackendRateLimit",
    "BackendServerError",
    "BackendNetworkError",
    "BackendManager",
    "aggregate_errors",
]


# ── helpers ─────────────────────────────────────────────────


def _mask_key(key: str) -> str:
    """Mask API key for safe logging: show first 2 and last 2 chars."""
    if len(key) <= 6:
        return "***"
    return key[:2] + "***" + key[-2:]


def _parse_retry_after(value: str | None) -> int | None:
    """Parse Retry-After header: seconds or HTTP-date → seconds."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    return None


# ── domain exceptions ───────────────────────────────────────


class BackendError(Exception):
    """Base error for all backend failures."""


class BackendTimeout(BackendError):
    """Read / connect / write / pool timeout."""


class BackendRateLimit(BackendError):
    """HTTP 429 from upstream."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class BackendServerError(BackendError):
    """HTTP 5xx from upstream."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class BackendNetworkError(BackendError):
    """DNS / connection refused / etc."""


# ── manager ─────────────────────────────────────────────────


class BackendManager:
    """Holds backend list, limiters, cooldowns, latency stats and scoring."""

    _JITTER_MAX = 0.2

    def __init__(self) -> None:
        self.backends: list[dict] = []
        self.limiters: dict[str, AsyncLimiter] = {}
        self.cooldowns: dict[str, float] = {}
        self.failures: defaultdict[str, int] = defaultdict(int)
        self.latency_history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._client: httpx.AsyncClient | None = None
        self.model_aliases: dict[str, str] = {}

    @property
    def loaded(self) -> bool:
        return len(self.backends) > 0

    def get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it on first call."""
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        """Close the shared httpx client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def ping(self, backend: dict) -> dict:
        """HEAD request to check if a backend is reachable."""
        name = backend["name"]
        client = self.get_client()
        start = time.time()
        try:
            resp = await client.head(
                backend["url"],
                headers={"Authorization": f"Bearer {backend['key'].removeprefix('Bearer ')}"},
                timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0),
            )
            elapsed = round(time.time() - start, 3)
            return {"name": name, "status": "ok" if resp.status_code < 500 else "degraded", "http_status": resp.status_code, "latency_ms": round(elapsed * 1000)}
        except Exception as e:
            elapsed = round(time.time() - start, 3)
            return {"name": name, "status": "unreachable", "error": type(e).__name__, "latency_ms": round(elapsed * 1000)}

    def load(self, backends: list[tuple[str, str, str, str | None]]) -> None:
        self.backends = [
            {
                "name": name,
                "url": url,
                "key": key,
            }
            for name, url, key, _model in backends
        ]
        self.limiters = {
            b["name"]: AsyncLimiter(RPM_LIMIT, 60) for b in self.backends
        }
        self.cooldowns = {b["name"]: 0.0 for b in self.backends}
        self.model_aliases = {}
        for name, _url, _key, model in backends:
            if model:
                self.model_aliases[model] = name

        for name, _url, key, model in backends:
            log.info("Loaded backend: %s (key=%s, model=%s)", name, _mask_key(key), model or "*")

        log.info("Loaded backends: %d, model aliases: %d", len(self.backends), len(self.model_aliases))

    def score(self, name: str) -> float:
        """Lower = better. 999999 means unusable. Deterministic."""
        if self.failures[name] >= 3:
            return 999999.0
        if time.time() < self.cooldowns[name]:
            return 999999.0

        hist = self.latency_history[name]
        avg_latency = sum(hist) / len(hist) if hist else 1.0
        return avg_latency + self.failures[name] * 10.0

    def ordered_backends(self, *, jitter: bool = False, model: str | None = None) -> list[dict]:
        """Sort backends by score.

        When called without `jitter` the order is deterministic (good for
        tests).  When `jitter=True` a small random tie-breaker is added so
        that concurrent callers (workers, stream handlers) don't all
        thunder-herd onto the top-ranked backend.

        When `model` is given and model_aliases maps it to a backend name,
        only that backend is returned (if healthy).
        """
        if model and model in self.model_aliases:
            target = self.model_aliases[model]
            for b in self.backends:
                if b["name"] == target:
                    if self.score(target) < 999999.0:
                        return [b]
                    break

        if jitter:
            return sorted(
                self.backends,
                key=lambda b: self._jittered_score(b["name"]),
            )
        return sorted(self.backends, key=lambda b: self.score(b["name"]))

    def _jittered_score(self, name: str) -> float:
        base = self.score(name)
        if base >= 999999.0:
            return base
        return base + random.uniform(0.0, self._JITTER_MAX)

    def is_on_cooldown(self, name: str) -> bool:
        return time.time() < self.cooldowns[name]

    # ── state mutations ─────────────────────────────────────

    def mark_failure(self, name: str, cooldown: float = 15.0) -> None:
        self.failures[name] += 1
        if self.failures[name] >= 3:
            exponent = min(self.failures[name] - 2, 5)
            cooldown = cooldown * (2 ** exponent)
        jitter = random.uniform(-2.0, 2.0)
        actual = max(cooldown + jitter, 1.0)
        self.cooldowns[name] = time.time() + actual

    def mark_success(self, name: str) -> None:
        """Decay failures on success instead of resetting to zero."""
        if self.failures[name] > 0:
            self.failures[name] = max(0, self.failures[name] - 1)

    # ── health ──────────────────────────────────────────────

    def health(self) -> list[dict]:
        return [
            {
                "name": b["name"],
                "failures": self.failures[b["name"]],
                "cooldown_until": self.cooldowns[b["name"]],
            }
            for b in self.backends
        ]

    def metrics(self) -> dict:
        """Aggregate metrics across all backends."""
        all_latencies: list[float] = []
        per_backend: list[dict] = []
        for b in self.backends:
            name = b["name"]
            hist = list(self.latency_history[name])
            all_latencies.extend(hist)
            sorted_hist = sorted(hist)
            n = len(sorted_hist)
            per_backend.append({
                "name": name,
                "failures": self.failures[name],
                "on_cooldown": self.is_on_cooldown(name),
                "latency_count": n,
                "latency_p50": sorted_hist[n // 2] if n else None,
                "latency_p99": sorted_hist[int(n * 0.99)] if n else None,
                "latency_avg": round(sum(hist) / n, 3) if n else None,
            })
        total = len(all_latencies)
        sorted_all = sorted(all_latencies)
        return {
            "backends": per_backend,
            "total_requests": total,
            "latency_p50": sorted_all[total // 2] if total else None,
            "latency_p99": sorted_all[int(total * 0.99)] if total else None,
        }

    # ── call ────────────────────────────────────────────────

    async def call(self, backend: dict, payload: dict, trace_id: str) -> dict:
        """Send request to a single backend.  Raises BackendError subclasses."""
        name: str = backend["name"]

        start = time.time()
        client = self.get_client()

        async with self.limiters[name]:
            log.info("[%s] CONNECT %s", trace_id, name)
            try:
                response = await client.post(
                    backend["url"],
                    headers={
                        "Authorization": f"Bearer {backend['key'].removeprefix('Bearer ')}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TimeoutException as e:
                elapsed = round(time.time() - start, 2)
                log.warning(
                    "[%s] TIMEOUT %s after %.2fs: %s",
                    trace_id, name, elapsed, type(e).__name__,
                )
                raise BackendTimeout(f"{name}: {type(e).__name__}") from e
            except httpx.NetworkError as e:
                elapsed = round(time.time() - start, 2)
                log.warning(
                    "[%s] NETWORK ERROR %s after %.2fs: %s",
                    trace_id, name, elapsed, type(e).__name__,
                )
                raise BackendNetworkError(f"{name}: {type(e).__name__}") from e

        elapsed = round(time.time() - start, 2)

        log.info(
            "[%s] %s status=%d latency=%.2fs",
            trace_id, name, response.status_code, elapsed,
        )

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise BackendRateLimit(
                f"{name}: rate limited (429)",
                status_code=429,
                retry_after=retry_after,
            )
        if response.status_code >= 500:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise BackendServerError(
                f"{name}: upstream {response.status_code}",
                status_code=response.status_code,
                retry_after=retry_after,
            )

        self.latency_history[name].append(elapsed)
        return response.json()

    # ── call stream ────────────────────────────────────────

    async def call_stream(
        self, backend: dict, payload: dict, trace_id: str
    ) -> AsyncIterator[str]:
        """Stream SSE chunks from a single backend.

        Yields raw lines (without trailing \\n) — the caller adds
        the newline.  Raises BackendError subclasses on failure.
        """
        name: str = backend["name"]

        start = time.time()
        client = self.get_client()

        async with self.limiters[name]:
            log.info("[%s] STREAM → %s", trace_id, name)
            try:
                async with client.stream(
                    "POST",
                    backend["url"],
                    headers={
                        "Authorization": f"Bearer {backend['key'].removeprefix('Bearer ')}",
                        "Content-Type": "application/json",
                    },
                    json={**payload, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        elapsed = round(time.time() - start, 2)
                        log.warning(
                            "[%s] STREAM ERROR %s status=%d latency=%.2fs",
                            trace_id, name, response.status_code, elapsed,
                        )
                        body = await response.aread()
                        log.warning(
                            "[%s] STREAM ERROR %s body=%r",
                            trace_id, name, body[:200],
                        )
                        if response.status_code == 429:
                            retry_after = _parse_retry_after(
                                response.headers.get("Retry-After")
                            )
                            raise BackendRateLimit(
                                f"{name}: rate limited (429)",
                                status_code=429,
                                retry_after=retry_after,
                            )
                        if response.status_code >= 500:
                            retry_after = _parse_retry_after(
                                response.headers.get("Retry-After")
                            )
                            raise BackendServerError(
                                f"{name}: upstream {response.status_code}",
                                status_code=response.status_code,
                                retry_after=retry_after,
                            )
                        raise BackendError(
                            f"{name}: HTTP {response.status_code}"
                        )

                    async for line in response.aiter_lines():
                        yield line

                    elapsed = round(time.time() - start, 2)
                    log.info(
                        "[%s] STREAM DONE %s latency=%.2fs",
                        trace_id, name, elapsed,
                    )
                    self.latency_history[name].append(elapsed)

            except httpx.TimeoutException as e:
                elapsed = round(time.time() - start, 2)
                log.warning(
                    "[%s] STREAM TIMEOUT %s after %.2fs: %s",
                    trace_id, name, elapsed, type(e).__name__,
                )
                raise BackendTimeout(f"{name}: {type(e).__name__}") from e
            except httpx.NetworkError as e:
                elapsed = round(time.time() - start, 2)
                log.warning(
                    "[%s] STREAM NETWORK ERROR %s after %.2fs: %s",
                    trace_id, name, elapsed, type(e).__name__,
                )
                raise BackendNetworkError(f"{name}: {type(e).__name__}") from e


def aggregate_errors(
    errors: list[Exception],
) -> tuple[int, str, int | None]:
    """Given all backend errors, determine the best HTTP status code,
    detail message and optional Retry-After seconds to return."""
    if not errors:
        return 503, "No backends available", None

    types = {type(e) for e in errors}

    if types == {BackendRateLimit}:
        retry = max(
            (getattr(e, "retry_after", 0) or 0 for e in errors),
            default=0,
        )
        return 429, "Rate limited on all backends", retry or None

    if types == {BackendTimeout}:
        return 504, "All backends timed out", None

    if types == {BackendServerError}:
        max_code = max(
            (getattr(e, "status_code", 500) for e in errors),
            default=500,
        )
        return 502, f"All backends returned server errors (max {max_code})", None

    if types == {BackendNetworkError}:
        return 502, "All backends unreachable", None

    return 503, "All backends unavailable", None

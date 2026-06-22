import random
import time
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator

import httpx
from aiolimiter import AsyncLimiter

from config import RPM_LIMIT

log = logging.getLogger("system")


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

    # Bound jitter so that small score gaps flip, big gaps don't.
    _JITTER_MAX = 0.2  # seconds of latency-equivalent noise

    def __init__(self) -> None:
        self.backends: list[dict] = []
        self.limiters: dict[str, AsyncLimiter] = {}
        self.cooldowns: dict[str, float] = {}
        self.failures: defaultdict[str, int] = defaultdict(int)
        self.latency_history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )

    # ── lifecycle ───────────────────────────────────────────

    def load(self, backends: list[tuple[str, str, str]]) -> None:
        self.backends = [
            {
                "name": name,
                "url": url,
                "key": key,
            }
            for name, url, key in backends
        ]
        self.limiters = {
            b["name"]: AsyncLimiter(RPM_LIMIT, 60) for b in self.backends
        }
        self.cooldowns = {b["name"]: 0.0 for b in self.backends}

        for name, _url, _key in backends:
            log.info(f"Loaded backend: {name}")

        log.info(f"Loaded backends: {len(self.backends)}")

    # ── scoring ─────────────────────────────────────────────

    def score(self, name: str) -> float:
        """Lower = better. 999999 means unusable. Deterministic."""
        if self.failures[name] >= 3:
            return 999999.0
        if time.time() < self.cooldowns[name]:
            return 999999.0

        hist = self.latency_history[name]
        avg_latency = sum(hist) / len(hist) if hist else 1.0
        return avg_latency + self.failures[name] * 10.0

    def ordered_backends(self, *, jitter: bool = False) -> list[dict]:
        """Sort backends by score.

        When called without `jitter` the order is deterministic (good for
        tests).  When `jitter=True` a small random tie-breaker is added so
        that concurrent callers (workers, stream handlers) don't all
        thunder-herd onto the top-ranked backend.
        """
        if jitter:
            return sorted(
                self.backends,
                key=lambda b: self._jittered_score(b["name"]),
            )
        return sorted(self.backends, key=lambda b: self.score(b["name"]))

    def _jittered_score(self, name: str) -> float:
        base = self.score(name)
        if base >= 999999.0:
            # Unusable backends stay at the end regardless of jitter.
            return base
        return base + random.uniform(0.0, self._JITTER_MAX)

    def is_on_cooldown(self, name: str) -> bool:
        return time.time() < self.cooldowns[name]

    # ── state mutations ─────────────────────────────────────

    def mark_failure(self, name: str, cooldown: float = 15.0) -> None:
        self.failures[name] += 1
        jitter = random.uniform(-2.0, 2.0)
        actual = max(cooldown + jitter, 1.0)
        self.cooldowns[name] = time.time() + actual

    def mark_success(self, name: str) -> None:
        self.failures[name] = 0

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

    # ── call ────────────────────────────────────────────────

    async def call(self, backend: dict, payload: dict, trace_id: str) -> dict:
        """Send request to a single backend.  Raises BackendError subclasses."""
        name: str = backend["name"]

        timeout = httpx.Timeout(connect=5.0, read=40.0, write=10.0, pool=5.0)
        start = time.time()

        async with self.limiters[name]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                log.info(f"[{trace_id}] CONNECT {name}")
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
                        f"[{trace_id}] TIMEOUT {name} after {elapsed}s: {repr(e)}"
                    )
                    raise BackendTimeout(f"{name}: {repr(e)}") from e
                except httpx.NetworkError as e:
                    elapsed = round(time.time() - start, 2)
                    log.warning(
                        f"[{trace_id}] NETWORK ERROR {name} after {elapsed}s: {repr(e)}"
                    )
                    raise BackendNetworkError(f"{name}: {repr(e)}") from e

        elapsed = round(time.time() - start, 2)

        log.info(
            f"[{trace_id}] {name} status={response.status_code} latency={elapsed}s"
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

        # record latency only for successful calls
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

        timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
        start = time.time()

        async with self.limiters[name]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                log.info(f"[{trace_id}] STREAM → {name}")
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
                                f"[{trace_id}] STREAM ERROR {name} "
                                f"status={response.status_code} "
                                f"latency={elapsed}s"
                            )
                            body = await response.aread()
                            log.warning(
                                f"[{trace_id}] STREAM ERROR {name} "
                                f"body={body[:200]!r}"
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
                            f"[{trace_id}] STREAM DONE {name} latency={elapsed}s"
                        )
                        self.latency_history[name].append(elapsed)

                except httpx.TimeoutException as e:
                    elapsed = round(time.time() - start, 2)
                    log.warning(
                        f"[{trace_id}] STREAM TIMEOUT {name} "
                        f"after {elapsed}s: {repr(e)}"
                    )
                    raise BackendTimeout(f"{name}: {repr(e)}") from e
                except httpx.NetworkError as e:
                    elapsed = round(time.time() - start, 2)
                    log.warning(
                        f"[{trace_id}] STREAM NETWORK ERROR {name} "
                        f"after {elapsed}s: {repr(e)}"
                    )
                    raise BackendNetworkError(f"{name}: {repr(e)}") from e


# ── helpers ─────────────────────────────────────────────────


def _parse_retry_after(value: str | None) -> int | None:
    """Parse Retry-After header: seconds or HTTP-date → seconds."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    # HTTP-date — ignored for simplicity (rarely used)
    return None


def aggregate_errors(
    errors: list[Exception],
) -> tuple[int, str, int | None]:
    """Given all backend errors, determine the best HTTP status code,
    detail message and optional Retry-After seconds to return.

    Uses `status_code` / `retry_after` attributes when present on
    BackendRateLimit / BackendServerError instances.
    """
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

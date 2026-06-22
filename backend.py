from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from aiolimiter import AsyncLimiter

from config import CFG
from logger import sys_log


@dataclass
class Backend:
    name: str
    url: str
    key: str

    limiter: AsyncLimiter = field(init=False)
    client: httpx.AsyncClient = field(init=False)
    cooldown_until: float = 0.0
    total_latency: float = 0.0
    success_count: int = 0
    fail_count: int = 0
    rate_limited_until: float = 0.0

    def __post_init__(self):
        self.limiter = AsyncLimiter(CFG.rpm_limit, 60)
        self.client = httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {self.key}"})

    @property
    def healthy(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def score(self) -> float:
        if not self.healthy:
            return float("inf")
        if self.success_count == 0:
            return 0.0
        return self.total_latency / self.success_count

    def record_success(self, latency: float):
        self.total_latency += latency
        self.success_count += 1
        self.cooldown_until = 0.0

    def record_failure(self, is_timeout: bool):
        self.fail_count += 1
        cooldown = CFG.cooldown_timeout if is_timeout else CFG.cooldown_error
        self.cooldown_until = time.time() + cooldown
        sys_log.warning("backend=%s cooldown=%.0fs", self.name, cooldown)

    def record_rate_limited(self, retry_after: float):
        self.rate_limited_until = time.time() + retry_after
        self.cooldown_until = time.time() + retry_after

    async def call(self, payload: dict) -> dict:
        async with self.limiter:
            start = time.time()
            r = await self.client.post(self.url, json=payload)
            latency = time.time() - start
            r.raise_for_status()
            self.record_success(latency)
            return r.json()

    async def call_stream(self, payload: dict) -> AsyncIterator[bytes]:
        async with self.limiter:
            async with self.client.stream("POST", self.url, json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    yield chunk


class BackendManager:
    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def sorted_by_score(self) -> list[Backend]:
        return sorted(self.backends, key=lambda b: b.score)

    async def call(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for b in self.sorted_by_score():
            if not b.healthy:
                continue
            try:
                return await b.call(payload)
            except httpx.TimeoutException:
                b.record_failure(is_timeout=True)
                last_exc = httpx.TimeoutException("timeout")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = float(e.response.headers.get("Retry-After", "60"))
                    b.record_rate_limited(retry_after)
                    last_exc = httpx.HTTPStatusError("rate limited", request=e.request, response=e.response)
                else:
                    b.record_failure(is_timeout=False)
                    last_exc = e
            except Exception as e:
                b.record_failure(is_timeout=False)
                last_exc = e
        raise last_exc or RuntimeError("no backends available")

    async def call_stream(self, payload: dict) -> AsyncIterator[bytes]:
        for b in self.sorted_by_score():
            if not b.healthy:
                continue
            try:
                async for chunk in b.call_stream(payload):
                    yield chunk
                return
            except httpx.TimeoutException:
                b.record_failure(is_timeout=True)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = float(e.response.headers.get("Retry-After", "60"))
                    b.record_rate_limited(retry_after)
                else:
                    b.record_failure(is_timeout=False)
            except Exception:
                b.record_failure(is_timeout=False)
        yield b"data: [ERROR] all backends exhausted\n\n"
        yield b"data: [DONE]\n\n"

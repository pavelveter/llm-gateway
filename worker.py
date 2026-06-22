from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from config import CFG
from backend import BackendManager
from logger import sys_log


class QueueFull(Exception):
    pass


class WorkerManager:
    def __init__(self, manager: BackendManager):
        self.queue: asyncio.Queue[tuple[dict, asyncio.Future]] = asyncio.Queue(maxsize=CFG.queue_max)
        self.manager = manager
        self._workers: list[asyncio.Task] = []

    def start(self):
        for i in range(CFG.workers):
            self._workers.append(asyncio.create_task(self._worker(i)))

    def stop(self):
        for w in self._workers:
            w.cancel()

    async def submit(self, payload: dict) -> Any:
        if self.queue.qsize() >= CFG.queue_max:
            raise QueueFull()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self.queue.put((payload, fut))
        return await fut

    async def _worker(self, idx: int):
        while True:
            payload, fut = await self.queue.get()
            try:
                result = await self.manager.call(payload)
                if not fut.cancelled():
                    fut.set_result(result)
            except Exception as exc:
                sys_log.error("worker=%d error=%s", idx, exc)
                if not fut.cancelled():
                    fut.set_exception(exc)
            finally:
                self.queue.task_done()


def classify_errors(errors: list[Exception]) -> int:
    if not errors:
        return 503
    codes = set()
    for e in errors:
        if isinstance(e, httpx.TimeoutException):
            codes.add(504)
        elif isinstance(e, httpx.HTTPStatusError):
            codes.add(e.response.status_code)
        else:
            codes.add(503)
    if codes == {429}:
        return 429
    if 504 in codes:
        return 504
    if 502 in codes or 503 in codes:
        return 502
    return 503


def get_retry_after(errors: list[Exception]) -> int:
    for e in errors:
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
            try:
                return int(e.response.headers.get("Retry-After", "60"))
            except ValueError:
                return 60
    return 60

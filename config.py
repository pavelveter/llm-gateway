from __future__ import annotations

import os


def _backends() -> list[tuple[str, str]]:
    backends: list[tuple[str, str]] = []
    i = 1
    while True:
        url = os.getenv(f"BACKEND_{i}_URL")
        key = os.getenv(f"BACKEND_{i}_KEY")
        if not url or not key:
            break
        backends.append((url, key))
        i += 1
    return backends


_backend_list = _backends()

BACKEND_URLS: list[str] = [b[0] for b in _backend_list]
BACKEND_KEYS: list[str] = [b[1] for b in _backend_list]
BACKEND_NAMES: list[str] = [f"backend-{i}" for i in range(1, len(_backend_list) + 1)]

rpm_limit = int(os.getenv("LLM_RPM_LIMIT", "38"))
queue_max = int(os.getenv("LLM_QUEUE_MAX", "100"))
workers = int(os.getenv("LLM_WORKERS", "2"))
stream_concurrency = int(os.getenv("LLM_STREAM_CONCURRENCY", "20"))
cooldown_timeout = int(os.getenv("LLM_COOLDOWN_TIMEOUT", "10"))
cooldown_error = int(os.getenv("LLM_COOLDOWN_ERROR", "15"))

log_dir = os.getenv("LLM_LOG_DIR", "logs")


class CFG:
    rpm_limit = rpm_limit
    queue_max = queue_max
    workers = workers
    stream_concurrency = stream_concurrency
    cooldown_timeout = cooldown_timeout
    cooldown_error = cooldown_error
    log_dir = log_dir
    backends = list(zip(BACKEND_NAMES, BACKEND_URLS, BACKEND_KEYS))

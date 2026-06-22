from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from config import CFG


def _make_logger(name: str, filename: str) -> logging.Logger:
    os.makedirs(CFG.log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        os.path.join(CFG.log_dir, filename),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


sys_log = _make_logger("system", "system.log")


class _ChatLogger:
    def __init__(self):
        self._logger = _make_logger("chat", "chat.log")

    def log_request(self, *, model: str, messages: list[dict], backend: str = ""):
        self._logger.info(json.dumps({
            "ts": time.time(),
            "type": "request",
            "model": model,
            "messages": [m.get("content", "")[:200] for m in messages],
            "backend": backend,
        }, ensure_ascii=False))

    def log_response(self, *, model: str, content: str, backend: str = "", latency: float = 0):
        self._logger.info(json.dumps({
            "ts": time.time(),
            "type": "response",
            "model": model,
            "content": content[:500],
            "backend": backend,
            "latency": round(latency, 3),
        }, ensure_ascii=False))


chat_log = _ChatLogger()

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

__all__ = ["setup_logging"]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> tuple[logging.Logger, logging.Logger]:
    """Configure system and chat loggers, return (system_log, chat_log)"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── system logger ──────────────────────────────────────
    system_log = logging.getLogger("system")
    system_log.setLevel(logging.INFO)

    sys_file = RotatingFileHandler(
        LOG_DIR / "system.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    sys_file.setFormatter(formatter)

    sys_stdout = logging.StreamHandler()
    sys_stdout.setFormatter(formatter)

    system_log.addHandler(sys_file)
    system_log.addHandler(sys_stdout)

    # ── chat logger ─────────────────────────────────────────
    chat_log = logging.getLogger("chat")
    chat_log.setLevel(logging.INFO)

    chat_file = RotatingFileHandler(
        LOG_DIR / "chat.log",
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    chat_file.setFormatter(formatter)
    chat_log.addHandler(chat_file)

    return system_log, chat_log

import os
from pathlib import Path

LOG_DIR = Path(os.getenv("LLM_LOG_DIR", "logs"))

RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "38"))
QUEUE_MAX = int(os.getenv("LLM_QUEUE_MAX", "100"))
WORKERS = int(os.getenv("LLM_WORKERS", "2"))
STREAM_CONCURRENCY = max(
    int(os.getenv("LLM_STREAM_CONCURRENCY", "20")), 1
)

CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("LLM_CORS_ORIGINS", "*").split(",")
    if o.strip()
]
MAX_REQUEST_BYTES = int(os.getenv("LLM_MAX_REQUEST_BYTES", str(1024 * 1024)))  # 1 MB
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))  # seconds


def load_backends() -> list[tuple[str, str, str, str | None]]:
    """Load backends from BACKEND_N_URL / BACKEND_N_KEY / BACKEND_N_MODEL env vars.

    Returns a list of (name, url, key, model) tuples.
    """
    backends: list[tuple[str, str, str, str | None]] = []
    index = 1

    while True:
        url_key = f"BACKEND_{index}_URL"
        key_key = f"BACKEND_{index}_KEY"
        model_key = f"BACKEND_{index}_MODEL"

        url = os.getenv(url_key)
        api_key = os.getenv(key_key)
        model = os.getenv(model_key)

        if not url and not api_key:
            break

        if not url:
            raise ValueError(
                f"{url_key} is missing while {key_key} is set"
            )
        if not api_key:
            raise ValueError(
                f"{key_key} is missing while {url_key} is set"
            )

        backends.append((f"backend-{index}", url, api_key, model or None))
        index += 1

    if not backends:
        raise RuntimeError(
            "No backends configured — set BACKEND_1_URL and BACKEND_1_KEY"
        )

    return backends

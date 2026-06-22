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

# Model aliasing: maps model names to backend names.
# Format: MODEL_ALIASES="gpt-4o:backend-1,gpt-3.5-turbo:backend-2"
# If a model isn't in the alias map, all backends are tried (default behavior).
MODEL_ALIASES: dict[str, str] = {}
_aliases_raw = os.getenv("LLM_MODEL_ALIASES", "")
if _aliases_raw:
    for pair in _aliases_raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            model_name, backend_name = pair.split(":", 1)
            MODEL_ALIASES[model_name.strip()] = backend_name.strip()


def load_backends() -> list[tuple[str, str, str]]:
    """Load backends from BACKEND_N_URL / BACKEND_N_KEY env vars.

    Returns a list of (name, url, key) tuples.
    """
    backends: list[tuple[str, str, str]] = []
    index = 1

    while True:
        url_key = f"BACKEND_{index}_URL"
        key_key = f"BACKEND_{index}_KEY"

        url = os.getenv(url_key)
        api_key = os.getenv(key_key)

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

        backends.append((f"backend-{index}", url, api_key))
        index += 1

    if not backends:
        raise RuntimeError(
            "No backends configured — set BACKEND_1_URL and BACKEND_1_KEY"
        )

    return backends

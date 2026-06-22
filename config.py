import os
from pathlib import Path

BASE_DIR = Path("/app")
LOG_DIR = Path(os.getenv("LLM_LOG_DIR", str(BASE_DIR / "logs")))

RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "38"))
QUEUE_MAX = int(os.getenv("LLM_QUEUE_MAX", "100"))
WORKERS = int(os.getenv("LLM_WORKERS", "2"))

# Maximum simultaneously-active streaming completions.
# Streams are *detached* from the worker pool: workers spawn a
# background driver task per StreamJob and immediately move on, so a
# burst of slow streams can't pin the worker pool and starve new
# non-stream requests of dispatcher capacity.  This knob caps how many
# of those background drivers run concurrently against the upstream.
STREAM_CONCURRENCY = max(
    int(os.getenv("LLM_STREAM_CONCURRENCY", "20")), 1
)


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

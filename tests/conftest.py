import os
import tempfile


def pytest_configure(config) -> None:
    """Create a temporary log directory and point LLM_LOG_DIR at it.

    ``gateway.py`` calls ``setup_logging()`` at module level, which
    creates log files in ``LOG_DIR``.  On macOS ``/app/logs`` may not
    be writable, so we redirect logs to a temporary directory.

    Must run *before* any test module that imports ``gateway``.
    """
    tmp = tempfile.mkdtemp(prefix="llm-gateway-test-logs-")
    os.environ.setdefault("LLM_LOG_DIR", tmp)

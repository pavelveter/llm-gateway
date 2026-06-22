import os
import pytest

from config import load_backends, MODEL_ALIASES


class TestLoadBackends:
    def test_loads_single_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_1_URL", "https://api.example.com/v1")
        monkeypatch.setenv("BACKEND_1_KEY", "sk-test")
        monkeypatch.delenv("BACKEND_2_URL", raising=False)
        monkeypatch.delenv("BACKEND_2_KEY", raising=False)

        backends = load_backends()
        assert len(backends) == 1
        assert backends[0] == ("backend-1", "https://api.example.com/v1", "sk-test")

    def test_loads_multiple_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_1_URL", "https://api1.example.com/v1")
        monkeypatch.setenv("BACKEND_1_KEY", "key-1")
        monkeypatch.setenv("BACKEND_2_URL", "https://api2.example.com/v1")
        monkeypatch.setenv("BACKEND_2_KEY", "key-2")
        monkeypatch.delenv("BACKEND_3_URL", raising=False)
        monkeypatch.delenv("BACKEND_3_KEY", raising=False)

        backends = load_backends()
        assert len(backends) == 2
        assert backends[0][0] == "backend-1"
        assert backends[1][0] == "backend-2"

    def test_raises_on_missing_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BACKEND_1_URL", raising=False)
        monkeypatch.setenv("BACKEND_1_KEY", "sk-test")
        monkeypatch.delenv("BACKEND_2_URL", raising=False)
        monkeypatch.delenv("BACKEND_2_KEY", raising=False)

        with pytest.raises(ValueError, match="BACKEND_1_URL is missing"):
            load_backends()

    def test_raises_on_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_1_URL", "https://api.example.com/v1")
        monkeypatch.delenv("BACKEND_1_KEY", raising=False)
        monkeypatch.delenv("BACKEND_2_URL", raising=False)
        monkeypatch.delenv("BACKEND_2_KEY", raising=False)

        with pytest.raises(ValueError, match="BACKEND_1_KEY is missing"):
            load_backends()

    def test_raises_when_no_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BACKEND_1_URL", raising=False)
        monkeypatch.delenv("BACKEND_1_KEY", raising=False)
        monkeypatch.delenv("BACKEND_2_URL", raising=False)
        monkeypatch.delenv("BACKEND_2_KEY", raising=False)

        with pytest.raises(RuntimeError, match="No backends configured"):
            load_backends()

    def test_stops_at_first_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_1_URL", "https://api1.example.com/v1")
        monkeypatch.setenv("BACKEND_1_KEY", "key-1")
        monkeypatch.setenv("BACKEND_3_URL", "https://api3.example.com/v1")
        monkeypatch.setenv("BACKEND_3_KEY", "key-3")
        monkeypatch.delenv("BACKEND_2_URL", raising=False)
        monkeypatch.delenv("BACKEND_2_KEY", raising=False)

        backends = load_backends()
        assert len(backends) == 1


class TestModelAliases:
    def test_empty_by_default(self) -> None:
        assert MODEL_ALIASES == {} or isinstance(MODEL_ALIASES, dict)

    def test_parses_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL_ALIASES", "gpt-4o:backend-1,gpt-3.5:backend-2")
        import importlib
        import config
        importlib.reload(config)
        assert config.MODEL_ALIASES == {"gpt-4o": "backend-1", "gpt-3.5": "backend-2"}
        monkeypatch.delenv("LLM_MODEL_ALIASES")
        importlib.reload(config)

import time
import random
from collections import deque
from unittest.mock import patch

import pytest

from backend import (
    BackendError,
    BackendManager,
    BackendNetworkError,
    BackendRateLimit,
    BackendServerError,
    BackendTimeout,
    _parse_retry_after,
    aggregate_errors,
)


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mgr() -> BackendManager:
    """BackendManager with two loaded backends."""
    m = BackendManager()
    m.load(
        [
            ("backend-1", "https://api1.example.com/v1", "key-aaaa"),
            ("backend-2", "https://api2.example.com/v1", "key-bbbb"),
        ]
    )
    return m


# ── BackendManager lifecycle ─────────────────────────────────


class TestLoad:
    def test_load_creates_backends(self, mgr: BackendManager) -> None:
        assert len(mgr.backends) == 2
        assert mgr.backends[0]["name"] == "backend-1"
        assert mgr.backends[0]["url"] == "https://api1.example.com/v1"
        assert mgr.backends[0]["key"] == "key-aaaa"

    def test_load_sets_cooldowns_to_zero(self, mgr: BackendManager) -> None:
        assert mgr.cooldowns["backend-1"] == 0.0
        assert mgr.cooldowns["backend-2"] == 0.0

    def test_load_creates_limiters(self, mgr: BackendManager) -> None:
        assert "backend-1" in mgr.limiters
        assert "backend-2" in mgr.limiters

    def test_load_resets_failures(self, mgr: BackendManager) -> None:
        assert mgr.failures["backend-1"] == 0
        assert mgr.failures["backend-2"] == 0


# ── scoring ──────────────────────────────────────────────────


class TestScoring:
    def test_fresh_backend_scores_below_999999(self, mgr: BackendManager) -> None:
        assert mgr.score("backend-1") < 999999.0

    def test_backend_with_3_failures_is_unusable(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 3
        assert mgr.score("backend-1") == 999999.0

    def test_backend_on_cooldown_is_unusable(self, mgr: BackendManager) -> None:
        mgr.cooldowns["backend-1"] = time.time() + 60
        assert mgr.score("backend-1") == 999999.0

    def test_latency_affects_score(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([0.5, 0.5, 0.5], maxlen=20)
        mgr.latency_history["backend-2"] = deque([2.0, 2.0, 2.0], maxlen=20)

        # backend-1 should be faster → lower score → first in ordering
        assert mgr.score("backend-1") < mgr.score("backend-2")

    def test_failures_penalize_score(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 2
        mgr.latency_history["backend-1"] = deque([0.5], maxlen=20)

        # score = avg_latency + failures * 10 = 0.5 + 20 = 20.5
        assert mgr.score("backend-1") == pytest.approx(20.5)

    def test_no_latency_defaults_to_one(self, mgr: BackendManager) -> None:
        # empty latency → avg = 1.0
        assert mgr.score("backend-1") == pytest.approx(1.0)


class TestOrderedBackends:
    def test_faster_backend_comes_first(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([0.2], maxlen=20)
        mgr.latency_history["backend-2"] = deque([0.8], maxlen=20)

        ordered = mgr.ordered_backends()
        assert ordered[0]["name"] == "backend-1"
        assert ordered[1]["name"] == "backend-2"

    def test_failed_backend_pushed_to_end(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 3  # unusable → score 999999

        ordered = mgr.ordered_backends()
        assert ordered[0]["name"] == "backend-2"
        assert ordered[1]["name"] == "backend-1"


# ── cooldown + jitter ────────────────────────────────────────


class TestCooldown:
    def test_mark_failure_sets_cooldown(self, mgr: BackendManager) -> None:
        now = time.time()
        mgr.mark_failure("backend-1", cooldown=10.0)
        cooldown_until = mgr.cooldowns["backend-1"]
        # jitter ±2s, so cooldown is in [now+8, now+12] range
        # but clamped to min 1s, so lower bound is now+1
        assert now + 1.0 <= cooldown_until <= now + 12.0 + 0.1

    def test_mark_failure_increments_count(self, mgr: BackendManager) -> None:
        mgr.mark_failure("backend-1")
        assert mgr.failures["backend-1"] == 1
        mgr.mark_failure("backend-1")
        assert mgr.failures["backend-1"] == 2

    def test_is_on_cooldown_during_cooldown(self, mgr: BackendManager) -> None:
        mgr.cooldowns["backend-1"] = time.time() + 999
        assert mgr.is_on_cooldown("backend-1") is True

    def test_is_on_cooldown_after_expiry(self, mgr: BackendManager) -> None:
        mgr.cooldowns["backend-1"] = time.time() - 1
        assert mgr.is_on_cooldown("backend-1") is False

    @patch("backend.time.time", return_value=0.0)
    def test_jitter_produces_varied_cooldowns(
        self, _mock_time, mgr: BackendManager
    ) -> None:
        random.seed(0)
        durations: set[float] = set()
        for _ in range(20):
            mgr.mark_failure("backend-1", cooldown=10.0)
            durations.add(round(mgr.cooldowns["backend-1"], 6))
            mgr.cooldowns["backend-1"] = 0.0
        assert (
            len(durations) > 1
        ), "jitter should produce different cooldown values"

    def test_negative_jitter_clamped_to_one_second(
        self, mgr: BackendManager
    ) -> None:
        now = time.time()
        random.seed(123)
        mgr.mark_failure("backend-1", cooldown=1.0)
        assert mgr.cooldowns["backend-1"] >= now + 1.0


# ── mark_success ─────────────────────────────────────────────


class TestMarkSuccess:
    def test_mark_success_decays_failures(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 5
        mgr.mark_success("backend-1")
        assert mgr.failures["backend-1"] == 4

    def test_mark_success_full_decay_to_zero(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 1
        mgr.mark_success("backend-1")
        assert mgr.failures["backend-1"] == 0

    def test_mark_success_no_op_when_zero(self, mgr: BackendManager) -> None:
        mgr.mark_success("backend-1")
        assert mgr.failures["backend-1"] == 0

    def test_mark_success_does_not_clear_cooldown(self, mgr: BackendManager) -> None:
        mgr.cooldowns["backend-1"] = time.time() + 60
        mgr.mark_success("backend-1")
        assert mgr.is_on_cooldown("backend-1") is True


# ── health ───────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_all_backends(self, mgr: BackendManager) -> None:
        h = mgr.health()
        assert len(h) == 2
        names = {entry["name"] for entry in h}
        assert names == {"backend-1", "backend-2"}

    def test_health_reflects_failures(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 3
        h = mgr.health()
        entry = next(e for e in h if e["name"] == "backend-1")
        assert entry["failures"] == 3


# ── domain exceptions ────────────────────────────────────────


class TestDomainExceptions:
    def test_backend_rate_limit_carries_fields(self) -> None:
        exc = BackendRateLimit(
            "rate limited", status_code=429, retry_after=30
        )
        assert exc.status_code == 429
        assert exc.retry_after == 30
        assert isinstance(exc, BackendError)

    def test_backend_rate_limit_defaults(self) -> None:
        exc = BackendRateLimit("rate limited")
        assert exc.status_code == 429
        assert exc.retry_after is None

    def test_backend_server_error_carries_fields(self) -> None:
        exc = BackendServerError(
            "server error", status_code=503, retry_after=10
        )
        assert exc.status_code == 503
        assert exc.retry_after == 10

    def test_backend_server_error_defaults(self) -> None:
        exc = BackendServerError("server error")
        assert exc.status_code == 500
        assert exc.retry_after is None

    def test_backend_timeout_is_backend_error(self) -> None:
        exc = BackendTimeout("timed out")
        assert isinstance(exc, BackendError)

    def test_backend_network_error_is_backend_error(self) -> None:
        exc = BackendNetworkError("unreachable")
        assert isinstance(exc, BackendError)


# ── _parse_retry_after ───────────────────────────────────────


class TestParseRetryAfter:
    def test_integer_seconds(self) -> None:
        assert _parse_retry_after("30") == 30

    def test_integer_with_whitespace(self) -> None:
        assert _parse_retry_after("  60  ") == 60

    def test_none_value(self) -> None:
        assert _parse_retry_after(None) is None

    def test_empty_string(self) -> None:
        assert _parse_retry_after("") is None

    def test_http_date_ignored(self) -> None:
        assert (
            _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
        )

    def test_zero_seconds(self) -> None:
        assert _parse_retry_after("0") == 0


# ── aggregate_errors ─────────────────────────────────────────


class TestAggregateErrors:
    def test_empty_errors(self) -> None:
        code, detail, retry = aggregate_errors([])
        assert code == 503
        assert detail == "No backends available"
        assert retry is None

    def test_all_rate_limited(self) -> None:
        errors: list[Exception] = [
            BackendRateLimit("rl1", retry_after=30),
            BackendRateLimit("rl2", retry_after=60),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 429
        assert "Rate limited" in detail
        assert retry == 60

    def test_all_rate_limited_some_without_retry(self) -> None:
        errors: list[Exception] = [
            BackendRateLimit("rl1", retry_after=30),
            BackendRateLimit("rl2"),
        ]
        code, _detail, retry = aggregate_errors(errors)
        assert code == 429
        assert retry == 30

    def test_all_timeout(self) -> None:
        errors: list[Exception] = [
            BackendTimeout("t1"),
            BackendTimeout("t2"),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 504
        assert "timed out" in detail
        assert retry is None

    def test_all_server_error(self) -> None:
        errors: list[Exception] = [
            BackendServerError("e1", status_code=500),
            BackendServerError("e2", status_code=503),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 502
        assert "max 503" in detail
        assert retry is None

    def test_all_network_error(self) -> None:
        errors: list[Exception] = [
            BackendNetworkError("dns fail"),
            BackendNetworkError("refused"),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 502
        assert "unreachable" in detail
        assert retry is None

    def test_mixed_errors(self) -> None:
        errors: list[Exception] = [
            BackendRateLimit("rl1"),
            BackendTimeout("t1"),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 503
        assert "unavailable" in detail
        assert retry is None

    def test_rate_limited_with_zero_retry(self) -> None:
        errors: list[Exception] = [
            BackendRateLimit("rl1", retry_after=0),
            BackendRateLimit("rl2"),
        ]
        code, _detail, retry = aggregate_errors(errors)
        assert code == 429
        assert retry is None

    def test_single_rate_limit(self) -> None:
        code, detail, retry = aggregate_errors(
            [BackendRateLimit("rl", status_code=429, retry_after=15)]
        )
        assert code == 429
        assert retry == 15

    def test_single_timeout(self) -> None:
        code, detail, retry = aggregate_errors([BackendTimeout("t")])
        assert code == 504
        assert retry is None

    def test_single_server_error(self) -> None:
        code, detail, retry = aggregate_errors(
            [BackendServerError("e", status_code=503, retry_after=5)]
        )
        assert code == 502
        assert retry is None

    def test_single_network_error(self) -> None:
        code, detail, retry = aggregate_errors([BackendNetworkError("dns")])
        assert code == 502
        assert retry is None

    def test_base_backend_error_mixed_with_rate_limit(self) -> None:
        errors: list[Exception] = [
            BackendRateLimit("rl1"),
            BackendError("unexpected 418"),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 503
        assert retry is None

    def test_multiple_502_categories_still_mixed(self) -> None:
        errors: list[Exception] = [
            BackendServerError("e1"),
            BackendNetworkError("net"),
        ]
        code, detail, retry = aggregate_errors(errors)
        assert code == 503
        assert retry is None


# ── jitter / load-spreading ──────────────────────────────────


class TestJitter:
    def test_default_is_deterministic(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([0.2], maxlen=20)
        mgr.latency_history["backend-2"] = deque([0.8], maxlen=20)
        leaders = {mgr.ordered_backends()[0]["name"] for _ in range(20)}
        assert leaders == {"backend-1"}

    def test_jitter_false_unchanged_default(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([0.5], maxlen=20)
        mgr.latency_history["backend-2"] = deque([0.5], maxlen=20)
        for _ in range(20):
            assert (
                mgr.ordered_backends(jitter=False)[0]["name"] == "backend-1"
            )

    def test_jitter_randomizes_tied_backends(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([1.0], maxlen=20)
        mgr.latency_history["backend-2"] = deque([1.0], maxlen=20)
        leaders: set[str] = set()
        for _ in range(50):
            leaders.add(mgr.ordered_backends(jitter=True)[0]["name"])
        assert leaders == {"backend-1", "backend-2"}, (
            f"jitter didn't randomize the leader; saw only {leaders}"
        )

    def test_jitter_preserves_clear_winner(self, mgr: BackendManager) -> None:
        mgr.latency_history["backend-1"] = deque([0.1], maxlen=20)
        mgr.latency_history["backend-2"] = deque([2.0], maxlen=20)
        for _ in range(20):
            assert mgr.ordered_backends(jitter=True)[0]["name"] == "backend-1"

    def test_jitter_keeps_unusable_last(self, mgr: BackendManager) -> None:
        mgr.failures["backend-1"] = 3
        mgr.cooldowns["backend-2"] = 0.0
        mgr.latency_history["backend-2"] = deque([0.5], maxlen=20)
        for _ in range(20):
            ordered = mgr.ordered_backends(jitter=True)
            assert ordered[0]["name"] == "backend-2"
            assert ordered[1]["name"] == "backend-1"

    def test_jitter_keeps_cooldown_last(self, mgr: BackendManager) -> None:
        mgr.cooldowns["backend-1"] = time.time() + 999
        mgr.latency_history["backend-2"] = deque([0.5], maxlen=20)
        for _ in range(20):
            ordered = mgr.ordered_backends(jitter=True)
            assert ordered[0]["name"] == "backend-2"
            assert ordered[1]["name"] == "backend-1"

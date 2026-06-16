import pytest
from unittest.mock import patch
from src.agent.adaptive_timeout import AdaptiveTimeout
from src.agent import config


def test_t_wait_clamps_to_min():
    at = AdaptiveTimeout()
    at.record_api_latency(0.001)
    assert at.t_wait() >= config.T_WAIT_MIN


def test_t_wait_clamps_to_max():
    at = AdaptiveTimeout()
    at.record_api_latency(999.0)
    assert at.t_wait() <= config.T_WAIT_MAX


def test_rolling_window_size():
    at = AdaptiveTimeout()
    for i in range(15):
        at.record_api_latency(float(i))
    assert len(at._api_window) == 10


def test_ping_api_on_failure_returns_average():
    at = AdaptiveTimeout()
    at.record_api_latency(0.5)
    with patch("requests.get", side_effect=Exception("network error")):
        result = at.ping_api()
    assert result == pytest.approx(0.5)


def test_calibrate_runs_without_raising():
    at = AdaptiveTimeout()
    with patch.object(at, "ping_api", return_value=0.1), \
         patch.object(at, "ping_ollama", return_value=0.2):
        at.calibrate()
    s = at.summary()
    assert "t_wait" in s
    assert "t_behavior" in s

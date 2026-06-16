import os
import tempfile
import time
from unittest.mock import MagicMock, patch

from src.agent.adaptive_timeout import AdaptiveTimeout
from src.agent.behavior import BehaviorManager
from src.agent import config as cfg


def _make_session(tmpdir: str) -> dict:
    cfg.SESSION_PATH = os.path.join(tmpdir, "session.json")
    cfg.ERROR_LOG_PATH = os.path.join(tmpdir, "error.jsonl")
    return {
        "session_id": "test-123",
        "active_prompt": "buy everything",
        "initial_prompt": "buy everything",
        "behavior_change_count": 0,
        "status": "active",
        "cycle": 0,
    }


def test_apply_change_reverts_on_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = _make_session(tmpdir)
        at = AdaptiveTimeout()
        at.record_ollama_latency(0.1)
        bm = BehaviorManager(session, at)
        bm.request_change("sell everything")

        mm = MagicMock()
        il = MagicMock()

        # Make the memory reset take longer than t_behavior
        def slow_reset():
            time.sleep(999)
        mm.reset_all = slow_reset

        # Force very short t_behavior timeout
        with patch.object(at, "t_behavior", return_value=1):
            bm.apply_change(mm, il)

        assert bm.active_prompt == "buy everything"  # reverted


def test_request_change_rejects_while_pending():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = _make_session(tmpdir)
        at = AdaptiveTimeout()
        bm = BehaviorManager(session, at)
        assert bm.request_change("new prompt A") is True
        assert bm.request_change("new prompt B") is False


def test_change_count_increments():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = _make_session(tmpdir)
        at = AdaptiveTimeout()
        bm = BehaviorManager(session, at)
        bm.request_change("new prompt")
        mm = MagicMock()
        il = MagicMock()
        bm.apply_change(mm, il)
        bm.increment_change_count(session)
        assert session["behavior_change_count"] == 1


def test_revert_logs_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        session = _make_session(tmpdir)
        at = AdaptiveTimeout()
        bm = BehaviorManager(session, at)
        bm.revert_to_initial()
        assert os.path.exists(cfg.ERROR_LOG_PATH)
        content = open(cfg.ERROR_LOG_PATH).read()
        assert "Reverted" in content

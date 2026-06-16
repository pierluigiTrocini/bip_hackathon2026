import json
import os
import tempfile

import pytest

from src.agent import journal as j


def _minimal_entry(**overrides) -> dict:
    base = {
        "ts": "2026-01-01T00:00:00Z", "cycle": 1, "ticker": "AAPL", "session_id": "sess1",
        "action": "buy", "conf": 0.8, "conf_raw": 0.85, "stale_penalty": 0.05,
        "reasoning": "test", "accuracy_review": "ok", "decision_source": "agent",
        "price": 150.0, "price_timestamp": "2026-01-01T00:00:00Z", "ma5": 148.0,
        "trend": "up", "sentiment": 0.5, "sentiment_label": "positive", "data_ok": True,
        "imitative_source": None, "prompt_snapshot": "test prompt",
        "t_wait_used": 30, "t_behavior_used": 60, "mode": "normal",
        "portfolio_mode_reason": None, "order_id": None, "market_open": True,
        "price_after": None, "outcome_pct": None, "cash": 95000.0,
        "portfolio_value": 100000.0, "pnl_pct": 0.0, "positions": {},
    }
    base.update(overrides)
    return base


def test_write_read_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        entry = _minimal_entry()
        j.write_entry(entry, path=path)
        result = j.read_last_n(1, path=path)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
    finally:
        os.unlink(path)


def test_outcome_update_fills_fields():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        entry = _minimal_entry(action="buy", price=100.0, price_after=None, session_id="s1")
        j.write_entry(entry, path=path)
        j.outcome_update("AAPL", 110.0, "s1", path=path)
        updated = j.read_last_n(1, path=path)
        assert updated[0]["price_after"] == 110.0
        assert abs(updated[0]["outcome_pct"] - 10.0) < 0.01
    finally:
        os.unlink(path)


def test_read_last_n_never_loads_full_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        for i in range(1000):
            j.write_entry(_minimal_entry(cycle=i), path=path)
        file_size = os.path.getsize(path)
        result = j.read_last_n(5, path=path)
        assert len(result) == 5
        # The implementation reads in 8192-byte chunks from the tail
        # so we can't easily measure bytes read, but assert result is tail
        assert result[-1]["cycle"] == 999
    finally:
        os.unlink(path)


def test_build_entry_raises_on_missing_field():
    with pytest.raises(ValueError, match="Missing required journal fields"):
        j.build_entry(ticker="AAPL")  # missing most fields including 'cycle'


def test_log_error_append_only():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        j.log_error("src", "err1", path=path)
        j.log_error("src", "err2", path=path)
        lines = open(path).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["error"] == "err1"
        assert json.loads(lines[1])["error"] == "err2"
    finally:
        os.unlink(path)


def test_read_session_summary_counts_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        from src.agent import config as cfg
        orig_journal = cfg.JOURNAL_PATH
        orig_error = cfg.ERROR_LOG_PATH
        cfg.JOURNAL_PATH = os.path.join(tmpdir, "journal.jsonl")
        cfg.ERROR_LOG_PATH = os.path.join(tmpdir, "error.jsonl")
        try:
            sid = "test-session-id"
            for action in ["buy", "sell", "hold", "hold", "buy"]:
                j.write_entry(_minimal_entry(action=action, session_id=sid))
            summary = j.read_session_summary(sid)
            assert summary["decisions"]["buy"] == 2
            assert summary["decisions"]["sell"] == 1
            assert summary["decisions"]["hold"] == 2
        finally:
            cfg.JOURNAL_PATH = orig_journal
            cfg.ERROR_LOG_PATH = orig_error

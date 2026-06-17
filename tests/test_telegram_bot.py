"""
Unit tests for the Telegram bot message builders and snapshot persistence.

The critical regression these guard against: numeric values interpolated into
MarkdownV2 messages must have their '.', '+', '-' escaped. An unescaped char
makes Telegram reject the whole message (HTTP 400), which is why /resume,
/portfolio and /nerd used to silently do nothing.
"""
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src import telegram_bot as tb


# ── _esc / _num ────────────────────────────────────────────────────────────────

def test_esc_escapes_reserved_chars():
    assert tb._esc("a.b") == "a\\.b"
    assert tb._esc("+1.24%") == "\\+1\\.24%"
    assert tb._esc("a-b=c") == "a\\-b\\=c"


def test_esc_escapes_backslash_first():
    # backslash must be escaped before the others to avoid double-escaping
    assert tb._esc("a\\b") == "a\\\\b"


def test_num_escapes_decimal_point():
    assert tb._num(102450.0, ",.2f") == "102,450\\.00"


def test_num_escapes_sign_and_percent():
    assert tb._num(0.0124, "+.2%") == "\\+1\\.24%"
    assert tb._num(-0.05, "+.2%") == "\\-5\\.00%"


def test_num_handles_bad_spec_gracefully():
    # Should not raise even if format spec is wrong for the value
    assert isinstance(tb._num("notanumber", ".2f"), str)


# ── helper: detect the regression (unescaped '.' outside code spans) ────────────

def _unescaped_dot_outside_code(text: str) -> bool:
    """True if any literal '.' appears outside backtick code spans without a preceding backslash."""
    in_code = False
    for i, ch in enumerate(text):
        if ch == "`":
            in_code = not in_code
            continue
        if ch == "." and not in_code:
            if i == 0 or text[i - 1] != "\\":
                return True
    return False


# ── build_resume_text ──────────────────────────────────────────────────────────

def test_resume_empty_rows():
    out = tb.build_resume_text(0, [], {}, 0.0, "Contrarian", "14:00 UTC")
    assert "No cycle completed yet" in out


def test_resume_escapes_all_numbers():
    rows = [{
        "ticker": "AAPL", "action": "buy", "conf": 0.82,
        "sentiment_score": 0.45, "price": 213.50,
        "unrealized_pnl_pct": 0.012, "reasoning": "Sentiment acceleration.",
    }]
    pf = {"portfolio_value": 102450.0, "cash": 45200.0}
    out = tb.build_resume_text(12, rows, pf, 0.0124, "Contrarian", "14:32 UTC")
    assert "AAPL" in out
    assert "BUY" in out
    assert not _unescaped_dot_outside_code(out), out


def test_resume_handles_missing_fields():
    rows = [{"ticker": "TSLA", "action": "hold"}]  # minimal row
    out = tb.build_resume_text(1, rows, {}, 0.0, "Value", "10:00 UTC")
    assert "TSLA" in out
    assert not _unescaped_dot_outside_code(out)


# ── build_portfolio_text ─────────────────────────────────────────────────────────

def test_portfolio_escapes_numbers():
    data = {
        "portfolio_value": 102450.0, "cash": 45200.0, "pnl_pct": 0.0124,
        "positions": {"AAPL": {"qty": 5, "market_value": 1067.5, "avg_entry_price": 211.0}},
    }
    live = {"AAPL": {"live": 213.5, "upnl_pct": 0.0118}}
    out = tb.build_portfolio_text(data, live, "14:35 UTC")
    assert "102,450\\.00" in out
    assert not _unescaped_dot_outside_code(out), out
    assert "AAPL" in out


def test_portfolio_no_positions():
    data = {"portfolio_value": 100000.0, "cash": 100000.0, "pnl_pct": 0.0, "positions": {}}
    out = tb.build_portfolio_text(data, {}, "14:35 UTC")
    assert "No open positions" in out
    assert not _unescaped_dot_outside_code(out)


def test_portfolio_position_without_live_price():
    data = {
        "portfolio_value": 100000.0, "cash": 50000.0, "pnl_pct": 0.0,
        "positions": {"MSFT": {"qty": 3, "market_value": 1254.6, "avg_entry_price": 415.0}},
    }
    live = {"MSFT": {"live": None, "upnl_pct": None}}
    out = tb.build_portfolio_text(data, live, "14:35 UTC")
    assert "MSFT" in out
    assert not _unescaped_dot_outside_code(out)


# ── build_nerd_text ──────────────────────────────────────────────────────────────

def test_nerd_no_data():
    out = tb.build_nerd_text(0, [], [], {})
    assert "No cycle rows yet" in out


def test_nerd_snapshot_numbers_escaped_outside_code():
    rows = [{
        "ticker": "AAPL", "trend": "up", "sentiment_score": 0.45,
        "unrealized_pnl_pct": 0.012, "avg_entry_price": 211.0,
        "rsi": 64.3, "bb_pct_b": 0.82,
    }]
    out = tb.build_nerd_text(12, ["AAPL", "TSLA"], rows, {("AAPL", "TSLA"): 0.42})
    assert not _unescaped_dot_outside_code(out), out
    assert "AAPL" in out
    # RSI value should be present (escaped)
    assert "64\\.3" in out


def test_nerd_ncci_matrix_present():
    out = tb.build_nerd_text(5, ["AAPL", "TSLA"], [{"ticker": "AAPL", "trend": "up"}], {("AAPL", "TSLA"): 0.42})
    assert "NCCI matrix" in out
    # matrix value rendered inside code span
    assert "+0.42" in out


def test_nerd_ncci_no_values_message():
    out = tb.build_nerd_text(5, ["AAPL", "TSLA"], [{"ticker": "AAPL", "trend": "up"}], {("AAPL", "TSLA"): 0.0})
    assert "not yet computed" in out


# ── build_modalita_* ─────────────────────────────────────────────────────────────

def test_modalita_list():
    strats = {"contrarian": {"name": "Contrarian"}, "momentum": {"name": "Momentum"}}
    out = tb.build_modalita_list_text(strats, "Contrarian")
    assert "contrarian" in out
    assert "Momentum" in out
    assert not _unescaped_dot_outside_code(out)


def test_modalita_changed():
    out = tb.build_modalita_changed_text("Trend Following")
    assert "Trend Following" in out
    assert not _unescaped_dot_outside_code(out)


# ── _decide_prompt_mode ──────────────────────────────────────────────────────────

def test_decide_prompt_mode_replace_when_empty():
    assert tb._decide_prompt_mode("", "focus on tech") == "replace"


def test_decide_prompt_mode_ignore_when_redundant():
    assert tb._decide_prompt_mode("focus on renewable energy stocks", "renewable energy stocks") == "ignore"


def test_decide_prompt_mode_replace_when_novel():
    assert tb._decide_prompt_mode("focus on renewable energy", "buy defense contractors") == "replace"


# ── snapshot persistence ─────────────────────────────────────────────────────────

def test_snapshot_roundtrip(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        snap = {
            "cycle": 7, "rows": [{"ticker": "AAPL", "action": "buy"}],
            "portfolio": {"cash": 50000.0}, "pnl_pct": 0.01,
            "strategy": "Momentum", "active_prompt": "tech focus",
            "tickers": ["AAPL", "TSLA"], "ncci": [["AAPL", "TSLA", 0.42]],
        }
        tb._save_snapshot(path, snap)
        loaded = tb._load_snapshot(path)
        assert loaded["cycle"] == 7
        assert loaded["strategy"] == "Momentum"
        assert loaded["ncci"] == [["AAPL", "TSLA", 0.42]]
    finally:
        os.unlink(path)


def test_load_snapshot_missing_file():
    assert tb._load_snapshot("/nonexistent/path/snap.json") == {}


def test_notifier_restores_snapshot(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        snap = {
            "cycle": 9, "rows": [{"ticker": "NVDA", "action": "hold"}],
            "portfolio": {"cash": 80000.0, "portfolio_value": 100000.0}, "pnl_pct": 0.02,
            "strategy": "Value", "active_prompt": "balanced", "tickers": ["NVDA"],
            "ncci": [["NVDA", "AMD", 0.31]],
        }
        with open(path, "w") as fh:
            json.dump(snap, fh)
        monkeypatch.setattr(tb.config, "TELEGRAM_SNAPSHOT_PATH", path)
        n = tb.TelegramNotifier(token="DISABLED", chat_id="123")
        assert n._last_cycle == 9
        assert n._last_strategy == "Value"
        assert n._tickers == ["NVDA"]
        assert n._last_ncci.get(("NVDA", "AMD")) == 0.31
    finally:
        os.unlink(path)


def test_strip_md_removes_markup():
    assert tb._strip_md("*bold* _italic_ `code`") == "bold italic code"
    assert tb._strip_md("102,450\\.00") == "102,450.00"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

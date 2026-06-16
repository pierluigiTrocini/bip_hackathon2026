"""
Infrastructure tests — run before every demo.
Usage: uv run python tests/test_connections.py
"""
import json
import os
import sys
import tempfile
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

PASS_SYM = "✓"
FAIL_SYM = "✗"
results: list[tuple[str, bool, str]] = []


def test(label: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((label, True, ""))
                print(f"  {PASS_SYM}  {label}")
            except Exception as exc:
                results.append((label, False, str(exc)))
                print(f"  {FAIL_SYM}  {label}: {exc}")
        return wrapper
    return decorator


# ── Alpaca ────────────────────────────────────────────────────────────────────

@test("Alpaca account")
def test_alpaca_account():
    from src.agent import config
    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key=config.ALPACA_API_KEY, secret_key=config.ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()
    assert float(account.cash) > 0, "Cash must be > 0"


@test("Alpaca clock")
def test_alpaca_clock():
    from src.agent import config
    from alpaca.trading.client import TradingClient
    client = TradingClient(api_key=config.ALPACA_API_KEY, secret_key=config.ALPACA_SECRET_KEY, paper=True)
    clock = client.get_clock()
    assert isinstance(clock.is_open, bool)


@test("Alpaca price AAPL")
def test_alpaca_price_aapl():
    from src.agent.adaptive_timeout import AdaptiveTimeout
    from src.agent.tool_executor import ToolExecutor
    te = ToolExecutor(AdaptiveTimeout())
    result = te.get_price("AAPL")
    assert result.ok, f"get_price failed: {result.error}"
    assert result.data["price"] > 0
    assert "T" in result.data["timestamp"]


@test("Alpaca news AAPL")
def test_alpaca_news_aapl():
    from src.agent.adaptive_timeout import AdaptiveTimeout
    from src.agent.tool_executor import ToolExecutor
    te = ToolExecutor(AdaptiveTimeout())
    result = te.get_news("AAPL")
    # ok even if 0 articles (market may be closed)
    assert isinstance(result.data.get("articles", []), list)


@test("Alpaca place_order dry run")
def test_alpaca_place_order_dry_run():
    from src.agent.broker import Broker
    broker = Broker()
    is_open = broker.is_market_open()
    assert isinstance(is_open, bool)


# ── Ollama ────────────────────────────────────────────────────────────────────

@test("Ollama reachable")
def test_ollama_reachable():
    import requests
    from src.agent import config
    resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
    resp.raise_for_status()
    assert "models" in resp.json()


@test("qwen2.5:3b loaded")
def test_ollama_sentiment_model_loaded():
    import requests
    from src.agent import config
    resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    assert any(config.OLLAMA_SENTIMENT_MODEL in m or m.startswith(config.OLLAMA_SENTIMENT_MODEL) for m in models), \
        f"{config.OLLAMA_SENTIMENT_MODEL} not found. Available: {models}"


@test("gemma4:12b loaded")
def test_ollama_reasoning_model_loaded():
    import requests
    from src.agent import config
    resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    assert any(config.OLLAMA_REASONING_MODEL in m or m.startswith(config.OLLAMA_REASONING_MODEL) for m in models), \
        f"{config.OLLAMA_REASONING_MODEL} not found. Available: {models}"


@test("Sentiment valid JSON")
def test_ollama_sentiment_returns_valid_json():
    from src.agent.sentiment import analyse
    articles = [{"title": "Apple reports record earnings", "summary": "AAPL beats estimates."}]
    result = analyse("AAPL", articles, active_prompt="growth investing", t_behavior=120)
    assert -1.0 <= result["score"] <= 1.0
    assert result["label"] in ("positive", "negative", "neutral")


@test("Reasoning valid decision")
def test_ollama_reasoning_returns_valid_decision():
    from src.agent.reasoner import Reasoner
    r = Reasoner()
    decision = r.decide(
        ticker="AAPL", memory_context="No prior decisions.", price=150.0,
        price_timestamp="2026-01-01T00:00:00Z", ma5=148.0, trend="up",
        sentiment_score=0.3, sentiment_label="positive",
        imitative_hints="", active_prompt="growth investing",
        cash=90000.0, positions={}, mode="normal",
        stale=False, staleness_seconds=0, t_behavior=120,
    )
    assert decision["action"] in ("buy", "sell", "hold")
    assert 0.0 <= decision["confidence"] <= 1.0


# ── Adaptive Timeout ──────────────────────────────────────────────────────────

@test("Adaptive timeout calibrate")
def test_adaptive_timeout_calibrate():
    from src.agent.adaptive_timeout import AdaptiveTimeout
    at = AdaptiveTimeout()
    at.calibrate()
    s = at.summary()
    assert s["t_wait"] > 0
    assert s["t_behavior"] > 0


# ── Journal ───────────────────────────────────────────────────────────────────

@test("Journal write/read")
def test_journal_write_read():
    from src.agent import journal as j
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        entry = j.build_entry(
            ts="2026-01-01T00:00:00Z", cycle=1, ticker="AAPL", session_id="s1",
            action="buy", conf=0.8, conf_raw=0.85, stale_penalty=0.0,
            reasoning="test", accuracy_review="ok", decision_source="agent",
            price=150.0, price_timestamp="2026-01-01T00:00:00Z", ma5=148.0,
            trend="up", sentiment=0.5, sentiment_label="positive", data_ok=True,
            imitative_source=None, prompt_snapshot="test",
            t_wait_used=30, t_behavior_used=60, mode="normal",
            portfolio_mode_reason=None, order_id=None, market_open=True,
            price_after=None, outcome_pct=None, cash=95000.0,
            portfolio_value=100000.0, pnl_pct=0.0, positions={},
        )
        j.write_entry(entry, path=path)
        result = j.read_last_n(1, path=path)
        assert result[0]["ticker"] == "AAPL"
    finally:
        os.unlink(path)


@test("Journal outcome update")
def test_journal_outcome_update():
    from src.agent import journal as j
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        entry = j.build_entry(
            ts="2026-01-01T00:00:00Z", cycle=1, ticker="MSFT", session_id="s2",
            action="buy", conf=0.7, conf_raw=0.7, stale_penalty=0.0,
            reasoning="test", accuracy_review="ok", decision_source="agent",
            price=300.0, price_timestamp="2026-01-01T00:00:00Z", ma5=298.0,
            trend="up", sentiment=0.2, sentiment_label="positive", data_ok=True,
            imitative_source=None, prompt_snapshot="test",
            t_wait_used=30, t_behavior_used=60, mode="normal",
            portfolio_mode_reason=None, order_id=None, market_open=True,
            price_after=None, outcome_pct=None, cash=90000.0,
            portfolio_value=100000.0, pnl_pct=0.0, positions={},
        )
        j.write_entry(entry, path=path)
        j.outcome_update("MSFT", 315.0, "s2", path=path)
        updated = j.read_last_n(1, path=path)
        assert updated[0]["price_after"] == 315.0
        assert abs(updated[0]["outcome_pct"] - 5.0) < 0.01
    finally:
        os.unlink(path)


@test("Error log write")
def test_error_log_write():
    from src.agent import journal as j
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        j.log_error("TestSource", "test error message", path=path)
        with open(path) as fh:
            data = json.loads(fh.read().strip())
        assert data["error"] == "test error message"
    finally:
        os.unlink(path)


# ── Session ───────────────────────────────────────────────────────────────────

@test("Session create and save")
def test_session_create_and_save():
    from src.agent import config as cfg
    from src.agent.session import SessionManager
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.SESSION_PATH = os.path.join(tmpdir, "session.json")
        sm = SessionManager()
        session = sm.create_new("test prompt")
        sid = session["session_id"]
        detected = sm.detect_previous_session()
        assert detected is not None
        assert detected["session_id"] == sid


# ── ToolExecutor ──────────────────────────────────────────────────────────────

@test("ToolExecutor graceful failure")
def test_tool_executor_graceful_on_invalid_ticker():
    from src.agent.adaptive_timeout import AdaptiveTimeout
    from src.agent.tool_executor import ToolExecutor
    te = ToolExecutor(AdaptiveTimeout())
    result = te.get_price("INVALIDXXX123")
    assert result is not None
    # Must not raise — result may be ok=False


# ── Imitative Layer ───────────────────────────────────────────────────────────

@test("Imitative dataset exists")
def test_imitative_dataset_exists():
    from src.agent import config
    assert os.path.exists(config.IMITATIVE_DATASET_PATH), \
        f"Missing: {config.IMITATIVE_DATASET_PATH}"
    with open(config.IMITATIVE_DATASET_PATH) as f:
        data = json.load(f)
    assert "strategies" in data


@test("Imitative filter green")
def test_imitative_filter_green_prompt():
    from src.agent.imitative_layer import ImiativeLayer
    il = ImiativeLayer()
    matched = il.filter_for_prompt("I want to invest in green renewable companies with ESG focus")
    ids = [s["id"] for s in matched]
    assert "green_esg" in ids, f"Expected green_esg, got: {ids}"


@test("Imitative filter defense")
def test_imitative_filter_defense_prompt():
    from src.agent.imitative_layer import ImiativeLayer
    il = ImiativeLayer()
    matched = il.filter_for_prompt("I want to invest in defense and military arms companies")
    ids = [s["id"] for s in matched]
    assert "defense_sector" in ids, f"Expected defense_sector, got: {ids}"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "═" * 50)
    print("  BIP Hackathon 2026 — Infrastructure Tests")
    print("═" * 50)

    for fn in [
        test_alpaca_account, test_alpaca_clock, test_alpaca_price_aapl,
        test_alpaca_news_aapl, test_alpaca_place_order_dry_run,
        test_ollama_reachable, test_ollama_sentiment_model_loaded,
        test_ollama_reasoning_model_loaded, test_ollama_sentiment_returns_valid_json,
        test_ollama_reasoning_returns_valid_decision,
        test_adaptive_timeout_calibrate,
        test_journal_write_read, test_journal_outcome_update, test_error_log_write,
        test_session_create_and_save,
        test_tool_executor_graceful_on_invalid_ticker,
        test_imitative_dataset_exists, test_imitative_filter_green_prompt,
        test_imitative_filter_defense_prompt,
    ]:
        fn()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print("═" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("═" * 50)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()

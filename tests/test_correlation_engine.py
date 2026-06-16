"""
Unit tests for src/agent/correlation_engine.py
Usage: uv run python tests/test_correlation_engine.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

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


def _write_news_log(path: str, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days_ago_ts(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_entry(ticker: str, keywords: list[str], relevance: float = 0.8, ts: str | None = None) -> dict:
    return {
        "ts": ts or _now_ts(),
        "ticker": ticker,
        "cycle": 1,
        "session_id": "s1",
        "source": "Reuters",
        "title": f"{ticker} news",
        "summary": "",
        "url": "",
        "keywords": keywords,
        "sentiment_score": 0.0,
        "relevance_score": relevance,
        "decision_triggered": None,
        "used_in_cycle": 1,
        "ttl_days": 14,
        "compacted": False,
    }


@test("NCCI=1.0 for identical keyword sets")
def test_ncci_identical_keywords():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        kws = ["tech", "AI", "chip", "revenue"]
        _write_news_log(path, [
            _base_entry("AAPL", kws),
            _base_entry("NVDA", kws),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        ncci = ce.get_ncci("AAPL", "NVDA")
        assert abs(ncci - 1.0) < 0.01, f"Expected ~1.0, got {ncci}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("NCCI=0.0 for disjoint keyword sets")
def test_ncci_disjoint_keywords():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "chip", "apple"]),
            _base_entry("XOM", ["oil", "energy", "commodities"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        ncci = ce.get_ncci("AAPL", "XOM")
        assert ncci == 0.0, f"Expected 0.0, got {ncci}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("NCCI partial overlap: 1 shared in 6 total = ~0.167")
def test_ncci_partial_overlap():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "AI", "chip", "energy"]),
            _base_entry("XOM",  ["oil", "energy", "commodities"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        ncci = ce.get_ncci("AAPL", "XOM")
        # intersection={energy}, union={tech,AI,chip,energy,oil,commodities} = 1/6 ≈ 0.167
        assert 0.10 <= ncci <= 0.25, f"Expected ~0.167, got {ncci}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("NCCI is symmetric")
def test_ncci_symmetric():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "AI", "chip"]),
            _base_entry("MSFT", ["tech", "cloud", "chip"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        assert ce.get_ncci("AAPL", "MSFT") == ce.get_ncci("MSFT", "AAPL")
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("decay reduces weight for 14-day-old keyword to ~0.25")
def test_decay_reduces_old_keywords():
    from src.agent.correlation_engine import _decay
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = _decay(old_ts, now, half_life_days=7.0)
    assert abs(d - 0.25) < 0.05, f"Expected ~0.25, got {d}"


@test("decay returns 0.0 for entries older than 30 days")
def test_decay_zeroes_at_30_days():
    from src.agent.correlation_engine import _decay
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = _decay(old_ts, now)
    assert d == 0.0, f"Expected 0.0, got {d}"


@test("rebuild restores matrix from news_log on disk")
def test_rebuild_restores_from_disk():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("TSLA", ["ev", "battery", "energy"]),
            _base_entry("XOM",  ["energy", "oil", "fossil"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        ncci = ce.get_ncci("TSLA", "XOM")
        # shared={energy}, union={ev,battery,energy,oil,fossil} = 1/5 = 0.2
        assert ncci > 0.0, f"Expected ncci > 0, got {ncci}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("build_prompt_section returns '' when no pairs exceed threshold")
def test_build_prompt_section_empty_when_below_threshold():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "AI", "chip"]),
            _base_entry("XOM",  ["oil", "energy", "fossil"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        section = ce.build_prompt_section(["AAPL", "XOM"], {}, threshold=0.99)
        assert section == "", f"Expected empty string, got: {section!r}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("build_prompt_section format contains '↔', NCCI value, shared keywords")
def test_build_prompt_section_format():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "AI", "chip", "revenue"]),
            _base_entry("NVDA", ["tech", "AI", "chip", "gpu"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        section = ce.build_prompt_section(["AAPL", "NVDA"], {}, threshold=0.01)
        assert "↔" in section, f"Missing '↔' in section: {section}"
        assert "AAPL" in section
        assert "NVDA" in section
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("get_correlated_tickers returns list sorted by ncci descending")
def test_get_correlated_tickers_sorted_desc():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        # AAPL shares more with MSFT than with XOM
        _write_news_log(path, [
            _base_entry("AAPL", ["tech", "AI", "chip", "cloud", "software"]),
            _base_entry("MSFT", ["tech", "AI", "chip", "cloud", "azure"]),
            _base_entry("XOM",  ["oil", "energy"]),
        ])
        ce = CorrelationEngine()
        ce.rebuild()
        correlated = ce.get_correlated_tickers("AAPL", threshold=0.0)
        # Must be sorted descending by ncci
        nccivals = [c["ncci"] for c in correlated]
        assert nccivals == sorted(nccivals, reverse=True), f"Not sorted: {nccivals}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("register_dynamic_ticker + rebuild includes new ticker in universe")
def test_dynamic_ticker_included_after_register():
    from src.agent.correlation_engine import CorrelationEngine
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        _write_news_log(path, [])
        ce = CorrelationEngine()
        ce.register_dynamic_ticker("AMZN")
        universe = ce.get_universe()
        assert "AMZN" in universe, f"AMZN not in universe: {universe}"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


def run_all():
    print("\n" + "═" * 50)
    print("  test_correlation_engine.py")
    print("═" * 50)
    for fn in [
        test_ncci_identical_keywords,
        test_ncci_disjoint_keywords,
        test_ncci_partial_overlap,
        test_ncci_symmetric,
        test_decay_reduces_old_keywords,
        test_decay_zeroes_at_30_days,
        test_rebuild_restores_from_disk,
        test_build_prompt_section_empty_when_below_threshold,
        test_build_prompt_section_format,
        test_get_correlated_tickers_sorted_desc,
        test_dynamic_ticker_included_after_register,
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

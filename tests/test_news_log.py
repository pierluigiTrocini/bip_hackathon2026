"""
Unit tests for src/agent/news_log.py
Usage: uv run python tests/test_news_log.py
"""
import json
import os
import sys
import tempfile
import time
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


def _make_articles(n: int = 2) -> list[dict]:
    return [
        {
            "title": f"Test article {i} about AAPL earnings",
            "summary": f"Apple reports record earnings for Q{i}. Revenue up 20%.",
            "source": "Reuters",
            "url": f"https://reuters.com/article/{i}",
        }
        for i in range(n)
    ]


def _make_kw(n: int = 2) -> list[dict]:
    return [{"keywords": ["tech", "earnings", "apple"], "relevance_score": 0.8} for _ in range(n)]


@test("write_articles and get_recent_for_ticker roundtrip")
def test_write_and_read_roundtrip():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        articles = _make_articles(2)
        kw = _make_kw(2)
        nl.write_articles(articles, "AAPL", cycle=1, session_id="s1",
                          sentiment_score=0.5, keywords_and_relevance=kw)
        result = nl.get_recent_for_ticker("AAPL", n=5, session_id="s1")
        assert len(result) == 2, f"Expected 2, got {len(result)}"
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["cycle"] == 1
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("keywords saved on write")
def test_keywords_saved_on_write():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        articles = _make_articles(1)
        kw = [{"keywords": ["tech", "AI", "chip"], "relevance_score": 0.9}]
        nl.write_articles(articles, "NVDA", cycle=2, session_id="s1",
                          sentiment_score=0.3, keywords_and_relevance=kw)
        entries = nl.get_recent_for_ticker("NVDA", n=5)
        assert entries[0]["keywords"] == ["tech", "AI", "chip"]
        assert entries[0]["relevance_score"] == 0.9
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("mark_decision fills decision_triggered field")
def test_mark_decision_fills_field():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        articles = _make_articles(2)
        kw = _make_kw(2)
        nl.write_articles(articles, "MSFT", cycle=3, session_id="s2",
                          sentiment_score=0.2, keywords_and_relevance=kw)
        nl.mark_decision("MSFT", cycle=3, session_id="s2", decision="buy")
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert all(e["decision_triggered"] == "buy" for e in lines if e["ticker"] == "MSFT")
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("ttl_days=14 for relevance_score=0.8")
def test_ttl_from_relevance_high():
    from src.agent.news_log import _compute_ttl
    assert _compute_ttl(0.8) == 14
    assert _compute_ttl(0.7) == 14


@test("ttl_days=3 for relevance_score=0.2")
def test_ttl_from_relevance_low():
    from src.agent.news_log import _compute_ttl
    assert _compute_ttl(0.2) == 3
    assert _compute_ttl(0.0) == 3


@test("compact removes expired entries")
def test_compact_removes_expired():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "ts": old_ts, "ticker": "XOM", "cycle": 1, "session_id": "s1",
            "source": "Bloomberg", "title": "Old news", "summary": "", "url": "",
            "keywords": ["oil"], "sentiment_score": -0.2, "relevance_score": 0.3,
            "decision_triggered": None, "used_in_cycle": 1,
            "ttl_days": 3,  # expired: 20 days > 3
            "compacted": False,
        }
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")
        removed = nl.compact()
        assert removed >= 1, f"Expected at least 1 removed, got {removed}"
        result = nl.get_recent_for_ticker("XOM")
        assert len(result) == 0
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("compact deduplicates entries with >=80% keyword overlap")
def test_compact_deduplicates():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        base = {
            "ts": now, "ticker": "AAPL", "cycle": 1, "session_id": "s1",
            "source": "Reuters", "title": "Apple news", "summary": "", "url": "",
            "sentiment_score": 0.5, "decision_triggered": None, "used_in_cycle": 1,
            "ttl_days": 14, "compacted": False,
        }
        e1 = {**base, "keywords": ["tech", "earnings", "apple", "revenue"], "relevance_score": 0.9}
        e2 = {**base, "keywords": ["tech", "earnings", "apple", "revenue"], "relevance_score": 0.5}
        with open(path, "w") as f:
            f.write(json.dumps(e1) + "\n")
            f.write(json.dumps(e2) + "\n")
        removed = nl.compact()
        assert removed >= 1, f"Expected >=1 removed, got {removed}"
        # The lower-relevance entry should be marked compacted
        with open(path) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        non_compacted = [e for e in entries if not e.get("compacted")]
        assert len(non_compacted) == 1
        assert non_compacted[0]["relevance_score"] == 0.9
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("get_all_keywords_by_ticker respects max_age_days")
def test_get_all_keywords_by_ticker_respects_max_age():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_entry = {
            "ts": now_ts, "ticker": "TSLA", "cycle": 1, "session_id": "s1",
            "source": "x", "title": "x", "summary": "", "url": "",
            "keywords": ["ev", "battery"], "sentiment_score": 0.1, "relevance_score": 0.5,
            "decision_triggered": None, "used_in_cycle": 1, "ttl_days": 7, "compacted": False,
        }
        old_entry = {**new_entry, "ts": old_ts, "keywords": ["old", "stale"]}
        with open(path, "w") as f:
            f.write(json.dumps(new_entry) + "\n")
            f.write(json.dumps(old_entry) + "\n")
        result = nl.get_all_keywords_by_ticker(max_age_days=30)
        tsla_kws = result.get("TSLA", [])
        all_kws = [kw for e in tsla_kws for kw in e["keywords"]]
        assert "ev" in all_kws
        assert "old" not in all_kws, "Old entry should be excluded by max_age_days"
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("extract_keywords_and_relevance returns safe defaults on LLM failure")
def test_extract_keywords_fallback_on_llm_failure():
    import src.agent.news_log as nl
    import src.agent.config as cfg
    original_model = cfg.OLLAMA_SENTIMENT_MODEL
    cfg.OLLAMA_SENTIMENT_MODEL = "nonexistent-model-xyz-999"
    try:
        articles = _make_articles(2)
        result = nl.extract_keywords_and_relevance(articles, "AAPL", t_behavior=5)
        assert len(result) == 2
        assert all(isinstance(r["keywords"], list) for r in result)
        assert all(0.0 <= r["relevance_score"] <= 1.0 for r in result)
    finally:
        cfg.OLLAMA_SENTIMENT_MODEL = original_model


@test("read_for_display sorts by relevance_score descending")
def test_read_for_display_sorts_by_relevance():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        base = {
            "ts": now, "ticker": "AAPL", "cycle": 5, "session_id": "s5",
            "source": "x", "title": "x", "summary": "", "url": "",
            "keywords": [], "sentiment_score": 0.0,
            "decision_triggered": None, "used_in_cycle": 5, "ttl_days": 7, "compacted": False,
        }
        e_low  = {**base, "relevance_score": 0.2, "title": "Low relevance"}
        e_high = {**base, "relevance_score": 0.9, "title": "High relevance"}
        e_mid  = {**base, "relevance_score": 0.5, "title": "Mid relevance"}
        with open(path, "w") as f:
            f.write(json.dumps(e_low)  + "\n")
            f.write(json.dumps(e_high) + "\n")
            f.write(json.dumps(e_mid)  + "\n")
        result = nl.read_for_display("AAPL", cycle=5, session_id="s5", max_articles=3)
        assert result[0]["relevance_score"] >= result[1]["relevance_score"]
        assert result[1]["relevance_score"] >= result[2]["relevance_score"]
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


@test("read_for_display returns only display fields (not summary)")
def test_read_for_display_fields():
    import src.agent.news_log as nl
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    original_path = nl.NEWS_LOG_PATH
    nl.NEWS_LOG_PATH = path
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "ts": now, "ticker": "AAPL", "cycle": 1, "session_id": "s1",
            "source": "WSJ", "title": "Test", "summary": "secret summary",
            "url": "https://x.com", "keywords": [], "sentiment_score": 0.1,
            "relevance_score": 0.7, "decision_triggered": None, "used_in_cycle": 1,
            "ttl_days": 7, "compacted": False,
        }
        with open(path, "w") as f:
            f.write(json.dumps(entry) + "\n")
        result = nl.read_for_display("AAPL", cycle=1, session_id="s1")
        assert len(result) == 1
        assert "summary" not in result[0]
        assert "source" in result[0]
        assert "title" in result[0]
        assert "url" in result[0]
        assert "sentiment_score" in result[0]
        assert "relevance_score" in result[0]
    finally:
        nl.NEWS_LOG_PATH = original_path
        os.unlink(path)


def run_all():
    print("\n" + "═" * 50)
    print("  test_news_log.py")
    print("═" * 50)
    for fn in [
        test_write_and_read_roundtrip,
        test_keywords_saved_on_write,
        test_mark_decision_fills_field,
        test_ttl_from_relevance_high,
        test_ttl_from_relevance_low,
        test_compact_removes_expired,
        test_compact_deduplicates,
        test_get_all_keywords_by_ticker_respects_max_age,
        test_extract_keywords_fallback_on_llm_failure,
        test_read_for_display_sorts_by_relevance,
        test_read_for_display_fields,
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

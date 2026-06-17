"""Tests for news_log.build_news_context_for_prompt() — F1."""
import json
import os
import tempfile
import pytest
from unittest.mock import patch
from src.agent.news_log import build_news_context_for_prompt


def _write_temp_news(entries: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries:
        f.write(json.dumps(e) + "\n")
    f.close()
    return f.name


def _entry(ticker="TSLA", cycle=1, title="Test News", source="bloomberg",
           relevance=0.80, compacted=False) -> dict:
    return {
        "ticker": ticker, "cycle": cycle, "title": title,
        "source": source, "relevance_score": relevance,
        "compacted": compacted, "ts": "2026-06-17T10:00:00Z",
        "sentiment_score": 0.0, "keywords": [],
        "session_id": "s1", "ttl_days": 7,
    }


class TestBuildNewsContextForPrompt:
    def test_header_always_present(self):
        path = _write_temp_news([_entry(cycle=1)])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=1)
        os.unlink(path)
        assert "=== NEWS CONTEXT (TSLA) ===" in result

    def test_current_cycle_articles_included(self):
        path = _write_temp_news([_entry(cycle=5, title="Breaking TSLA news", source="ft")])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=5)
        os.unlink(path)
        assert "Breaking TSLA news" in result
        assert "Current cycle:" in result

    def test_historical_articles_included_if_above_min_relevance(self):
        entries = [
            _entry(cycle=3, title="Old high relevance", relevance=0.80),
            _entry(cycle=5, title="Current news", relevance=0.70),
        ]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, history_cycles=5, min_relevance_historical=0.50
            )
        os.unlink(path)
        assert "Old high relevance" in result

    def test_historical_excluded_below_min_relevance(self):
        entries = [
            _entry(cycle=3, title="Low relevance news", relevance=0.30),
            _entry(cycle=5, title="Current news", relevance=0.70),
        ]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, min_relevance_historical=0.50
            )
        os.unlink(path)
        assert "Low relevance news" not in result

    def test_compacted_entries_excluded(self):
        path = _write_temp_news([_entry(cycle=5, title="Compacted", compacted=True)])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=5)
        os.unlink(path)
        assert "Compacted" not in result

    def test_other_ticker_excluded(self):
        path = _write_temp_news([_entry(ticker="AAPL", cycle=5, title="AAPL news")])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=5)
        os.unlink(path)
        assert "AAPL news" not in result

    def test_no_articles_returns_empty_message(self):
        path = _write_temp_news([])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=5)
        os.unlink(path)
        assert "No news available" in result

    def test_max_articles_respected(self):
        entries = [_entry(cycle=5, title=f"News {i}", relevance=0.9 - i * 0.05)
                   for i in range(10)]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, max_articles=3
            )
        os.unlink(path)
        lines = [l for l in result.splitlines() if "[0." in l]
        assert len(lines) <= 3

    def test_historical_dedup_by_title(self):
        # Same title in current + historical → should appear only in current
        entries = [
            _entry(cycle=5, title="Duplicate title", relevance=0.90),
            _entry(cycle=3, title="Duplicate title", relevance=0.85),
        ]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, history_cycles=5, min_relevance_historical=0.50
            )
        os.unlink(path)
        assert result.count("Duplicate title") == 1

    def test_missing_file_returns_no_news_message(self):
        with patch("src.agent.news_log.NEWS_LOG_PATH", "/nonexistent/path.jsonl"):
            result = build_news_context_for_prompt("TSLA", current_cycle=1)
        assert "No news available" in result

    def test_score_formatted_two_decimals(self):
        path = _write_temp_news([_entry(cycle=1, relevance=0.9)])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=1)
        os.unlink(path)
        assert "[0.90]" in result

    def test_source_in_output(self):
        path = _write_temp_news([_entry(cycle=1, source="reuters", title="Test")])
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt("TSLA", current_cycle=1)
        os.unlink(path)
        assert "reuters" in result

    def test_historical_marker_ciclo(self):
        entries = [
            _entry(cycle=3, title="Historical article", relevance=0.80),
            _entry(cycle=5, title="Current article", relevance=0.70),
        ]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, history_cycles=5, min_relevance_historical=0.50
            )
        os.unlink(path)
        assert "[cycle 3]" in result

    def test_never_raises_on_corrupt_file(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        f.write("NOT JSON\n{broken}\n")
        f.close()
        with patch("src.agent.news_log.NEWS_LOG_PATH", f.name):
            result = build_news_context_for_prompt("TSLA", current_cycle=1)
        os.unlink(f.name)
        assert isinstance(result, str)

    def test_history_cycles_boundary(self):
        # Cycle 1 is outside history_cycles=3 when current=5
        entries = [
            _entry(cycle=1, title="Too old", relevance=0.90),
            _entry(cycle=3, title="In range", relevance=0.90),
            _entry(cycle=5, title="Current", relevance=0.70),
        ]
        path = _write_temp_news(entries)
        with patch("src.agent.news_log.NEWS_LOG_PATH", path):
            result = build_news_context_for_prompt(
                "TSLA", current_cycle=5, history_cycles=3, min_relevance_historical=0.50
            )
        os.unlink(path)
        assert "Too old" not in result
        assert "In range" in result

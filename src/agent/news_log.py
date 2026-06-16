"""
News persistence layer. Every article fetched by tool_executor.get_news() is
written here exactly once, with keywords and relevance extracted in parallel
with the sentiment call in sentiment.py.
"""
import json
import os
import threading
from datetime import datetime, timedelta, timezone

import ollama

from src.agent import config

NEWS_LOG_PATH: str = config.NEWS_LOG_PATH

_write_counter = 0
_counter_lock = threading.Lock()

_KEYWORD_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "keywords":        {"type": "array", "items": {"type": "string"}},
            "relevance_score": {"type": "number"},
        },
        "required": ["keywords", "relevance_score"],
    },
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_ttl(relevance_score: float) -> int:
    if relevance_score >= 0.7:
        return 14
    elif relevance_score >= 0.4:
        return 7
    return 3


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def extract_keywords_and_relevance(
    articles: list[dict],
    ticker: str,
    t_behavior: int,
) -> list[dict]:
    """
    Call qwen2.5:3b ONCE for all articles in a single prompt.
    Returns [{keywords: list[str], relevance_score: float}] — one dict per article.
    On LLM failure: returns [{keywords: [], relevance_score: 0.5}] per article.
    """
    if not articles:
        return []

    fallback = [{"keywords": [], "relevance_score": 0.5} for _ in articles]

    try:
        numbered_parts = []
        for i, a in enumerate(articles, 1):
            title = a.get("title", "")
            summary = (a.get("summary", "") or "")[:200]
            numbered_parts.append(f"[{i}] Title: {title}\n    Text: {summary}")
        numbered_articles_text = "\n\n".join(numbered_parts)

        prompt = (
            f"For each article below, extract 3-8 thematic keywords in English (lowercase) "
            f"and a relevance score from 0.0 to 1.0 for ticker {ticker}.\n\n"
            f"Relevance scoring:\n"
            f"- Direct mention of {ticker} or company name: +0.4\n"
            f"- Earnings, revenue, M&A, product launch, regulatory action: +0.3\n"
            f"- Macroeconomic or sector-wide relevance: +0.2\n"
            f"- Generic market commentary: 0.1\n\n"
            f"Return ONLY a JSON array, one object per article, in the same order:\n"
            f'[{{"keywords": ["keyword1", "keyword2"], "relevance_score": 0.8}}, ...]\n\n'
            f"Articles:\n{numbered_articles_text}"
        )

        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=prompt,
            format=_KEYWORD_SCHEMA,
            options={"temperature": 0.0, "num_predict": 300},
            keep_alive="30s",
        )
        raw = resp.get("response", "[]") if isinstance(resp, dict) else getattr(resp, "response", "[]")
        raw = raw.strip()

        # Find the JSON array
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return fallback

        result = []
        for i, item in enumerate(parsed):
            if i >= len(articles):
                break
            kws = item.get("keywords", [])
            if not isinstance(kws, list):
                kws = []
            kws = [str(k).lower() for k in kws[:8]]
            rel = float(item.get("relevance_score", 0.5))
            rel = max(0.0, min(1.0, rel))
            result.append({"keywords": kws, "relevance_score": rel})

        # Pad if LLM returned fewer items than articles
        while len(result) < len(articles):
            result.append({"keywords": [], "relevance_score": 0.5})

        return result

    except Exception:
        return fallback


def write_articles(
    articles: list[dict],
    ticker: str,
    cycle: int,
    session_id: str,
    sentiment_score: float,
    keywords_and_relevance: list[dict],
) -> None:
    """
    Append one entry per article to NEWS_LOG_PATH.
    Never raises.
    """
    global _write_counter

    try:
        _ensure_dir(NEWS_LOG_PATH)
        now = _now_utc()

        entries = []
        for i, article in enumerate(articles):
            kw_data = keywords_and_relevance[i] if i < len(keywords_and_relevance) else {"keywords": [], "relevance_score": 0.5}
            keywords = kw_data.get("keywords", [])
            relevance_score = float(kw_data.get("relevance_score", 0.5))
            ttl = _compute_ttl(relevance_score)

            entry = {
                "ts":               now,
                "ticker":           ticker,
                "cycle":            cycle,
                "session_id":       session_id,
                "source":           str(article.get("source", "")),
                "title":            str(article.get("title", "")),
                "summary":          str(article.get("summary", "") or "")[:300],
                "url":              str(article.get("url", "") or ""),
                "keywords":         keywords,
                "sentiment_score":  sentiment_score,
                "relevance_score":  relevance_score,
                "decision_triggered": None,
                "used_in_cycle":    cycle,
                "ttl_days":         ttl,
                "compacted":        False,
            }
            entries.append(entry)

        with open(NEWS_LOG_PATH, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        with _counter_lock:
            _write_counter += 1
            should_compact = (_write_counter % config.NEWS_LOG_COMPACT_EVERY == 0)

        if should_compact:
            threading.Thread(
                target=compact,
                kwargs={"max_entries_per_ticker": config.NEWS_LOG_MAX_PER_TICKER},
                daemon=True,
            ).start()

    except Exception as exc:
        try:
            from src.agent import journal as journal_module
            journal_module.log_error(
                source="news_log", error=f"write_articles failed: {exc}",
                ticker=ticker, session_id=session_id,
            )
        except Exception:
            pass


def mark_decision(
    ticker: str,
    cycle: int,
    session_id: str,
    decision: str,
) -> None:
    """
    Find the most recent entries for (ticker, cycle, session_id) where
    decision_triggered is None. Set decision_triggered = decision.
    Atomic rewrite via .tmp + os.replace().
    Never raises.
    """
    try:
        if not os.path.exists(NEWS_LOG_PATH):
            return

        lines = []
        with open(NEWS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        updated = []
        for line in lines:
            try:
                entry = json.loads(line)
                if (
                    entry.get("ticker") == ticker
                    and entry.get("cycle") == cycle
                    and entry.get("session_id") == session_id
                    and entry.get("decision_triggered") is None
                ):
                    entry["decision_triggered"] = decision
                    entry["used_in_cycle"] = cycle
                updated.append(json.dumps(entry, ensure_ascii=False))
            except Exception:
                updated.append(line)

        tmp = NEWS_LOG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for line in updated:
                f.write(line + "\n")
        os.replace(tmp, NEWS_LOG_PATH)

    except Exception:
        pass


def get_recent_for_ticker(
    ticker: str,
    n: int = 3,
    session_id: str | None = None,
) -> list[dict]:
    """
    Return the N most recent non-compacted entries for ticker.
    Never raises.
    """
    try:
        if not os.path.exists(NEWS_LOG_PATH):
            return []

        matches = []
        with open(NEWS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if (
                        entry.get("ticker") == ticker
                        and not entry.get("compacted", False)
                        and (session_id is None or entry.get("session_id") == session_id)
                    ):
                        matches.append(entry)
                except Exception:
                    pass

        return matches[-n:] if len(matches) > n else matches

    except Exception:
        return []


def get_all_keywords_by_ticker(
    max_age_days: int | None = None,
) -> dict[str, list[dict]]:
    """
    Return {ticker: [{keywords, ts, relevance_score}, ...]} for all tickers.
    Excludes compacted entries. Respects max_age_days cutoff.
    Never raises.
    """
    try:
        if not os.path.exists(NEWS_LOG_PATH):
            return {}

        cutoff: datetime | None = None
        if max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        result: dict[str, list[dict]] = {}
        with open(NEWS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("compacted", False):
                        continue
                    if cutoff is not None:
                        ts_str = entry.get("ts", "")
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts < cutoff:
                                continue
                        except Exception:
                            pass
                    ticker = entry.get("ticker", "")
                    if not ticker:
                        continue
                    if ticker not in result:
                        result[ticker] = []
                    result[ticker].append({
                        "keywords":       entry.get("keywords", []),
                        "ts":             entry.get("ts", ""),
                        "relevance_score": entry.get("relevance_score", 0.5),
                    })
                except Exception:
                    pass

        return result

    except Exception:
        return {}


def compact(max_entries_per_ticker: int = 50) -> int:
    """
    Remove expired entries and deduplicate by keyword overlap (>=80%).
    Atomic rewrite via .tmp + os.replace().
    Returns number of entries removed. Never raises.
    """
    try:
        if not os.path.exists(NEWS_LOG_PATH):
            return 0

        now = datetime.now(timezone.utc)
        lines_parsed: list[dict] = []
        raw_lines: list[str] = []

        with open(NEWS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    lines_parsed.append(entry)
                    raw_lines.append(line)
                except Exception:
                    pass

        removed = 0
        kept: list[dict] = []

        # Mark expired
        for entry in lines_parsed:
            if entry.get("compacted", False):
                kept.append(entry)
                continue
            ts_str = entry.get("ts", "")
            ttl = entry.get("ttl_days", 7)
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (now - ts).days >= ttl:
                    removed += 1
                    continue
            except Exception:
                pass
            kept.append(entry)

        # Deduplicate: group by ticker, find high-overlap pairs
        from collections import defaultdict
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for entry in kept:
            if not entry.get("compacted", False):
                by_ticker[entry.get("ticker", "")].append(entry)

        # Build a set of entries to mark compacted
        to_compact_ids: set[int] = set()
        for ticker, entries in by_ticker.items():
            for i in range(len(entries)):
                if id(entries[i]) in to_compact_ids:
                    continue
                kws_i = set(entries[i].get("keywords", []))
                for j in range(i + 1, len(entries)):
                    if id(entries[j]) in to_compact_ids:
                        continue
                    kws_j = set(entries[j].get("keywords", []))
                    if not kws_i and not kws_j:
                        continue
                    union = kws_i | kws_j
                    if not union:
                        continue
                    overlap = len(kws_i & kws_j) / len(union)
                    if overlap >= 0.80:
                        # Keep higher relevance, compact the other
                        rel_i = entries[i].get("relevance_score", 0.0)
                        rel_j = entries[j].get("relevance_score", 0.0)
                        loser = entries[j] if rel_i >= rel_j else entries[i]
                        to_compact_ids.add(id(loser))
                        loser["compacted"] = True
                        removed += 1

        tmp = NEWS_LOG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, NEWS_LOG_PATH)

        return removed

    except Exception:
        return 0


def read_for_display(
    ticker: str,
    cycle: int,
    session_id: str,
    max_articles: int = 3,
) -> list[dict]:
    """
    Return articles to display in terminal at decision time.
    Fields: source, title, url, sentiment_score, relevance_score.
    Sorted by relevance_score descending. Never raises.
    """
    try:
        if not os.path.exists(NEWS_LOG_PATH):
            return []

        matches = []
        with open(NEWS_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if (
                        entry.get("ticker") == ticker
                        and entry.get("cycle") == cycle
                        and entry.get("session_id") == session_id
                        and not entry.get("compacted", False)
                    ):
                        matches.append({
                            "source":          entry.get("source", ""),
                            "title":           entry.get("title", ""),
                            "url":             entry.get("url", ""),
                            "sentiment_score": entry.get("sentiment_score", 0.0),
                            "relevance_score": entry.get("relevance_score", 0.5),
                        })
                except Exception:
                    pass

        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matches[:max_articles]

    except Exception:
        return []

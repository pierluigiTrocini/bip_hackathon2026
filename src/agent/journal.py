import json
import os
from datetime import datetime, timezone

from src.agent import config

_REQUIRED_FIELDS = {
    "ts", "cycle", "ticker", "session_id", "action", "conf", "conf_raw",
    "stale_penalty", "reasoning", "accuracy_review", "decision_source",
    "price", "price_timestamp", "ma5", "trend", "sentiment", "sentiment_label",
    "data_ok", "imitative_source", "prompt_snapshot", "t_wait_used",
    "t_behavior_used", "mode", "portfolio_mode_reason", "order_id",
    "market_open", "price_after", "outcome_pct", "cash", "portfolio_value",
    "pnl_pct", "positions",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_entry(**kwargs) -> dict:
    missing = _REQUIRED_FIELDS - kwargs.keys()
    if missing:
        raise ValueError(f"Missing required journal fields: {missing}")
    return dict(kwargs)


def write_entry(entry: dict, path: str | None = None) -> None:
    if path is None:
        path = config.JOURNAL_PATH
    try:
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def log_error(
    source: str,
    error: str,
    ticker: str | None = None,
    retry_count: int = 0,
    stale_used: bool = False,
    session_id: str = "",
    path: str | None = None,
) -> None:
    if path is None:
        path = config.ERROR_LOG_PATH
    try:
        _ensure_dir(path)
        entry = {
            "ts": _now_utc(),
            "session_id": session_id,
            "source": source,
            "ticker": ticker,
            "error": error,
            "retry_count": retry_count,
            "stale_used": stale_used,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def read_last_n(n: int = 5, path: str | None = None) -> list[dict]:
    if path is None:
        path = config.JOURNAL_PATH
    try:
        if not os.path.exists(path):
            return []
        chunk_size = 8192
        entries: list[dict] = []
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            buf = b""
            pos = file_size
            while pos > 0 and len(entries) < n:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                lines = buf.split(b"\n")
                buf = lines[0]
                for line in reversed(lines[1:]):
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
                        if len(entries) == n:
                            break
            if buf.strip():
                try:
                    entries.append(json.loads(buf.strip()))
                except Exception:
                    pass
        return list(reversed(entries[:n]))
    except Exception:
        return []


def outcome_update(ticker: str, new_price: float, session_id: str, path: str | None = None) -> None:
    if path is None:
        path = config.JOURNAL_PATH
    try:
        if not os.path.exists(path):
            return
        lines: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        target_idx = None
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if (
                entry.get("ticker") == ticker
                and entry.get("session_id") == session_id
                and entry.get("price_after") is None
                and entry.get("action") in ("buy", "sell")
            ):
                target_idx = i
                break
        if target_idx is None:
            return
        entry = json.loads(lines[target_idx].strip())
        old_price = entry.get("price", 0.0)
        entry["price_after"] = new_price
        entry["outcome_pct"] = ((new_price - old_price) / old_price * 100) if old_price else None
        lines[target_idx] = json.dumps(entry) + "\n"
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp_path, path)
    except Exception:
        pass


def write_news_entries(
    articles: list[dict],
    ticker: str,
    cycle: int,
    session_id: str,
    sentiment_score: float,
    decision_triggered: str,
    path: str | None = None,
) -> None:
    if path is None:
        path = config.NEWS_LOG_PATH
    try:
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            for a in articles:
                entry = {
                    "ts": _now_utc(),
                    "ticker": ticker,
                    "cycle": cycle,
                    "session_id": session_id,
                    "source": a.get("source", "alpaca"),
                    "title": a.get("title", ""),
                    "summary": a.get("summary", ""),
                    "url": a.get("url", ""),
                    "keywords": a.get("keywords", a.get("symbols", [])),
                    "sentiment_score": round(sentiment_score, 2),
                    "relevance_score": 1.0,
                    "decision_triggered": decision_triggered,
                    "used_in_cycle": cycle,
                    "ttl_days": 7,
                    "compacted": False,
                }
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def read_session_summary(session_id: str) -> dict:
    decisions: dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
    orders_placed = 0
    autonomous_decisions = 0
    final_pnl_pct = None
    last_portfolio = None
    cycles = 0

    try:
        if os.path.exists(config.JOURNAL_PATH):
            with open(config.JOURNAL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("session_id") != session_id:
                        continue
                    cycles += 1
                    action = entry.get("action", "hold")
                    decisions[action] = decisions.get(action, 0) + 1
                    if entry.get("order_id"):
                        orders_placed += 1
                    if entry.get("decision_source") == "autonomous_timeout":
                        autonomous_decisions += 1
                    final_pnl_pct = entry.get("pnl_pct")
                    last_portfolio = {
                        "cash": entry.get("cash"),
                        "portfolio_value": entry.get("portfolio_value"),
                        "pnl_pct": entry.get("pnl_pct"),
                        "positions": entry.get("positions"),
                    }
    except Exception:
        pass

    errors = 0
    try:
        if os.path.exists(config.ERROR_LOG_PATH):
            with open(config.ERROR_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("session_id") == session_id:
                            errors += 1
                    except Exception:
                        pass
    except Exception:
        pass

    return {
        "session_id": session_id,
        "cycles": cycles,
        "decisions": decisions,
        "orders_placed": orders_placed,
        "autonomous_decisions": autonomous_decisions,
        "final_pnl_pct": final_pnl_pct,
        "errors": errors,
        "last_portfolio": last_portfolio,
    }

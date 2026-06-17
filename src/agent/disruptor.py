import hashlib
import json
import threading
import time
from datetime import datetime, timezone

from src.agent import config


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _title_hash(title: str) -> str:
    return hashlib.md5(title.encode()).hexdigest()[:12]


class MarketDisruptor:
    """
    Background thread that fetches breaking news from additional web sources
    (NewsAPI, Yahoo Finance RSS, Alpaca broader search) and writes them to
    disruptor_news.jsonl with higher priority than the standard news pipeline.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or config.DISRUPTOR_NEWS_PATH
        self._session_id: str = ""
        self._cycle: int = 0
        self._tickers: list[str] = []
        self._seen_hashes: set[str] = set()
        self._running = False
        self._file_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._interval: int = 60

    def start(self, tickers: list[str], session_id: str) -> None:
        self._tickers = list(tickers)
        self._session_id = session_id
        self._seen_hashes = set()
        self._clear_file()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MarketDisruptor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def update_cycle(self, cycle: int) -> None:
        self._cycle = cycle

    def update_tickers(self, tickers: list[str]) -> None:
        self._tickers = list(tickers)

    def _load_seen_hashes(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        h = entry.get("title_hash") or _title_hash(entry.get("title", ""))
                        self._seen_hashes.add(h)
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

    def _clear_file(self) -> None:
        try:
            import os
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with self._file_lock:
                open(self._path, "w").close()
        except Exception:
            pass

    def _run(self) -> None:
        while self._running:
            try:
                self._fetch_and_write_all()
            except Exception:
                pass
            time.sleep(self._interval)

    def _fetch_and_write_all(self) -> None:
        for ticker in list(self._tickers):
            articles = self._fetch_articles(ticker)
            for article in articles:
                title_hash = _title_hash(article.get("title", ""))
                if title_hash in self._seen_hashes:
                    continue
                entry = {
                    "ts": _now_utc(),
                    "ticker": ticker,
                    "cycle": self._cycle,
                    "session_id": self._session_id,
                    "source": article.get("source", "unknown"),
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "keywords": article.get("keywords", []),
                    "title_hash": title_hash,
                    "disruptor": True,
                    "ttl_days": 1,
                    "compacted": False,
                }
                self._append(entry)
                self._seen_hashes.add(title_hash)

    def _fetch_articles(self, ticker: str) -> list[dict]:
        articles: list[dict] = []
        if config.NEWS_API_KEY:
            articles = self._fetch_newsapi(ticker)
        if not articles:
            articles = self._fetch_alpaca_broad(ticker)
        if not articles:
            articles = self._fetch_yahoo_rss(ticker)
        return articles[:5]

    def _fetch_newsapi(self, ticker: str) -> list[dict]:
        try:
            import requests
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": config.NEWS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return [
                {
                    "title": a.get("title", ""),
                    "summary": a.get("description", "") or "",
                    "url": a.get("url", ""),
                    "source": "newsapi",
                    "keywords": [ticker.lower()],
                }
                for a in resp.json().get("articles", [])[:5]
                if a.get("title")
            ]
        except Exception:
            return []

    def _fetch_alpaca_broad(self, ticker: str) -> list[dict]:
        try:
            import requests
            resp = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                },
                params={"symbols": ticker, "limit": 5},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json().get("news", [])
            return [
                {
                    "title": a.get("headline", ""),
                    "summary": a.get("summary", ""),
                    "url": a.get("url", ""),
                    "source": "alpaca_disruptor",
                    "keywords": [ticker.lower()] + [s.lower() for s in a.get("symbols", [])],
                }
                for a in raw[:5]
                if a.get("headline")
            ]
        except Exception:
            return []

    def _fetch_yahoo_rss(self, ticker: str) -> list[dict]:
        try:
            import re
            import requests
            resp = requests.get(
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
            articles = []
            for item in items[:5]:
                title_m = re.search(
                    r"<title><!\[CDATA\[(.*?)\]\]>|<title>(.*?)</title>", item
                )
                desc_m = re.search(
                    r"<description><!\[CDATA\[(.*?)\]\]>|<description>(.*?)</description>", item
                )
                link_m = re.search(r"<link>(.*?)</link>", item)
                if title_m:
                    title = (title_m.group(1) or title_m.group(2) or "").strip()
                    desc = ""
                    if desc_m:
                        desc = (desc_m.group(1) or desc_m.group(2) or "").strip()
                    link = link_m.group(1).strip() if link_m else ""
                    if title:
                        articles.append({
                            "title": title,
                            "summary": desc,
                            "url": link,
                            "source": "yahoo_rss",
                            "keywords": [ticker.lower()],
                        })
            return articles
        except Exception:
            return []

    def _append(self, entry: dict) -> None:
        import os
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with self._file_lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def get_articles(self, ticker: str, max_age_seconds: int = 300) -> list[dict]:
        """Return recent disruptor articles for a ticker (within max_age_seconds)."""
        cutoff = time.time() - max_age_seconds
        articles: list[dict] = []
        with self._file_lock:
            try:
                f = open(self._path, "r", encoding="utf-8")
            except FileNotFoundError:
                return []
            with f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("ticker") != ticker:
                        continue
                    ts_str = entry.get("ts", "")
                    try:
                        ts_epoch = (
                            datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                            .replace(tzinfo=timezone.utc)
                            .timestamp()
                        )
                    except Exception:
                        ts_epoch = 0
                    if ts_epoch >= cutoff:
                        articles.append(entry)
        return articles

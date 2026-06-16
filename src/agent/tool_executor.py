import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agent import config
from src.agent import journal as journal_module
from src.agent.adaptive_timeout import AdaptiveTimeout

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0
_BACKOFF_FACTOR = 2.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]
    stale: bool = False
    staleness_seconds: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=_now_utc)


class ToolExecutor:
    def __init__(self, adaptive_timeout: AdaptiveTimeout) -> None:
        self._at = adaptive_timeout
        self._cache: dict[str, dict[str, ToolResult]] = {}
        self._blacklisted: set[str] = set()
        self._failure_counts: dict[str, int] = {}
        self._session_id: str = ""

    def _get_trading_client(self):
        from alpaca.trading.client import TradingClient
        return TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )

    def _get_data_client(self):
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )

    def _with_retry(self, fn, ticker_key: str | None = None):
        backoff = _INITIAL_BACKOFF
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                start = time.monotonic()
                result = fn()
                self._at.record_api_latency(time.monotonic() - start)
                if ticker_key:
                    self._failure_counts[ticker_key] = 0
                return result
            except Exception as exc:
                last_exc = exc
                journal_module.log_error(
                    source="ToolExecutor",
                    error=str(exc),
                    ticker=ticker_key,
                    retry_count=attempt + 1,
                    session_id=self._session_id,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= _BACKOFF_FACTOR
        if ticker_key:
            self._failure_counts[ticker_key] = self._failure_counts.get(ticker_key, 0) + 1
            if self._failure_counts[ticker_key] >= 3:
                self._blacklisted.add(ticker_key)
                journal_module.log_error(
                    source="ToolExecutor",
                    error=f"Ticker {ticker_key} blacklisted after 3 consecutive failures",
                    ticker=ticker_key,
                    session_id=self._session_id,
                )
        raise last_exc  # type: ignore[misc]

    def _cache_fallback(self, ticker: str, kind: str, error: str) -> ToolResult:
        cached = self._cache.get(ticker, {}).get(kind)
        if cached:
            staleness = int(time.time() - time.mktime(
                time.strptime(cached.timestamp, "%Y-%m-%dT%H:%M:%SZ")
            )) if cached.timestamp else 0
            return ToolResult(
                ok=False, data=cached.data, stale=True,
                staleness_seconds=max(0, staleness), error=error,
            )
        return ToolResult(ok=False, data={}, stale=False, error=error)

    def _store_cache(self, ticker: str, kind: str, result: ToolResult) -> None:
        self._cache.setdefault(ticker, {})[kind] = result

    def get_price(self, ticker: str) -> ToolResult:
        if ticker in self._blacklisted:
            return self._cache_fallback(ticker, "price", f"{ticker} is blacklisted")
        try:
            def fetch():
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestBarRequest
                client = self._get_data_client()
                req = StockLatestBarRequest(symbol_or_symbols=[ticker])
                bars = client.get_stock_latest_bar(req)
                bar = bars[ticker]
                return ToolResult(ok=True, data={
                    "ticker": ticker,
                    "price": float(bar.close),
                    "timestamp": bar.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") if bar.timestamp else _now_utc(),
                    "volume": int(bar.volume) if bar.volume else 0,
                })
            result = self._with_retry(fetch, ticker)
            self._store_cache(ticker, "price", result)
            return result
        except Exception as exc:
            return self._cache_fallback(ticker, "price", str(exc))

    def get_bars(self, ticker: str, limit: int = 5) -> ToolResult:
        if ticker in self._blacklisted:
            return self._cache_fallback(ticker, "bars", f"{ticker} is blacklisted")
        try:
            def fetch():
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                client = self._get_data_client()
                req = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, limit=limit)
                bars_resp = client.get_stock_bars(req)
                bars = bars_resp[ticker] if ticker in bars_resp else []
                closes = [float(b.close) for b in bars]
                ma = sum(closes) / len(closes) if closes else 0.0
                trend = "up" if len(closes) >= 2 and closes[-1] > closes[0] else "down" if len(closes) >= 2 else "flat"
                return ToolResult(ok=True, data={
                    "ticker": ticker,
                    "closes": closes,
                    "ma": round(ma, 2),
                    "trend": trend,
                })
            result = self._with_retry(fetch, ticker)
            self._store_cache(ticker, "bars", result)
            return result
        except Exception as exc:
            return self._cache_fallback(ticker, "bars", str(exc))

    def get_news(self, ticker: str) -> ToolResult:
        if ticker in self._blacklisted:
            return self._cache_fallback(ticker, "news", f"{ticker} is blacklisted")
        try:
            def fetch():
                import requests as req_lib
                url = f"https://data.alpaca.markets/v1beta1/news"
                headers = {
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                }
                resp = req_lib.get(url, headers=headers, params={"symbols": ticker, "limit": 3}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                raw_articles = data.get("news", [])
                articles = [
                    {"title": a.get("headline", ""), "summary": a.get("summary", "")}
                    for a in raw_articles[:3]
                ]
                return ToolResult(ok=True, data={"ticker": ticker, "articles": articles})
            result = self._with_retry(fetch, ticker)
            self._store_cache(ticker, "news", result)
            return result
        except Exception as exc:
            return self._cache_fallback(ticker, "news", str(exc))

    def get_portfolio(self) -> ToolResult:
        try:
            def fetch():
                client = self._get_trading_client()
                account = client.get_account()
                positions_raw = client.get_all_positions()
                portfolio_value = float(account.portfolio_value)
                cash = float(account.cash)
                positions = {
                    p.symbol: {
                        "qty": int(float(p.qty)),
                        "market_value": float(p.market_value),
                        "avg_entry_price": float(p.avg_entry_price),
                    }
                    for p in positions_raw
                }
                pnl_pct = (portfolio_value - 100_000.0) / 100_000.0
                return ToolResult(ok=True, data={
                    "cash": cash,
                    "portfolio_value": portfolio_value,
                    "positions": positions,
                    "pnl_pct": round(pnl_pct, 6),
                })
            result = self._with_retry(fetch, "__portfolio__")
            self._store_cache("__portfolio__", "portfolio", result)
            return result
        except Exception as exc:
            return self._cache_fallback("__portfolio__", "portfolio", str(exc))

    def get_market_news(self, keywords: list[str] | None = None, limit: int = 10) -> ToolResult:
        """Fetch general market news, optionally filtered by keywords."""
        try:
            def fetch():
                import requests as req_lib
                headers = {
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                }
                params: dict = {"limit": limit}
                resp = req_lib.get(
                    "https://data.alpaca.markets/v1beta1/news",
                    headers=headers, params=params, timeout=15,
                )
                resp.raise_for_status()
                raw = resp.json().get("news", [])
                articles = [
                    {
                        "title": a.get("headline", ""),
                        "summary": a.get("summary", ""),
                        "symbols": a.get("symbols", []),
                    }
                    for a in raw[:limit]
                ]
                # keyword filter if provided
                if keywords:
                    kw_lower = [k.lower() for k in keywords]
                    filtered = [
                        a for a in articles
                        if any(k in (a["title"] + a["summary"]).lower() for k in kw_lower)
                    ]
                    articles = filtered if filtered else articles
                return ToolResult(ok=True, data={"articles": articles})
            return self._with_retry(fetch, None)
        except Exception as exc:
            return ToolResult(ok=False, data={"articles": []}, error=str(exc))

    def validate_ticker(self, ticker: str) -> bool:
        """Return True if ticker has a tradeable price on Alpaca."""
        try:
            result = self.get_price(ticker)
            return result.ok and result.data.get("price", 0.0) > 0
        except Exception:
            return False

    def _alpaca_canonical_symbol(self, ticker: str) -> str | None:
        """
        Ask Alpaca for the canonical symbol of a ticker string.
        Returns the symbol as Alpaca knows it, or None if not found.
        Handles format variants (e.g. BRKB → BRK.B).
        """
        try:
            import requests as req_lib
            resp = req_lib.get(
                f"https://paper-api.alpaca.markets/v2/assets/{ticker}",
                headers={
                    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
                },
                timeout=6,
            )
            if resp.status_code == 200:
                asset = resp.json()
                if asset.get("tradable") and asset.get("status") == "active":
                    return str(asset["symbol"])
        except Exception:
            pass
        return None

    def resolve_ticker(self, ticker: str) -> tuple[str | None, bool]:
        """
        Try to find a valid, tradeable Alpaca symbol for *ticker*.

        Resolution order:
          1. Direct price fetch (ticker as-is)
          2. Alpaca asset lookup → get canonical symbol (handles format variants
             like BRKB→BRK.B, case normalisation, etc.)
          3. Price fetch on canonical symbol

        Returns (resolved_symbol, was_remapped):
          - (ticker, False)   if ticker itself is valid
          - (canonical, True) if ticker was remapped to its canonical form
          - (None, False)     if no valid symbol found
        """
        ticker = ticker.upper().strip()

        # Step 1: try as-is
        if self.validate_ticker(ticker):
            return ticker, False

        # Step 2: ask Alpaca for the canonical form
        canonical = self._alpaca_canonical_symbol(ticker)
        if canonical and canonical != ticker:
            if self.validate_ticker(canonical):
                return canonical, True

        return None, False

    def is_market_open(self) -> bool:
        try:
            client = self._get_trading_client()
            clock = client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def unblacklist(self, ticker: str) -> None:
        self._blacklisted.discard(ticker)
        self._failure_counts[ticker] = 0

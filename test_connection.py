"""
Connection test script for the BIP Hackathon 2026 trading agent.
Verifies mandatory (Alpaca) and optional (Anthropic, NewsAPI) integrations.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
SECTION = "\033[94m"
RESET = "\033[0m"


def section(title: str) -> None:
    print(f"\n{SECTION}{'─' * 50}{RESET}")
    print(f"{SECTION}{title}{RESET}")
    print(f"{SECTION}{'─' * 50}{RESET}")


def check(label: str, fn) -> bool:
    try:
        result = fn()
        print(f"  {PASS} {label}: {result}")
        return True
    except Exception as exc:
        print(f"  {FAIL} {label}: {exc}")
        return False


# ── Alpaca credentials ─────────────────────────────────────────────────────────

def _alpaca_key() -> str:
    key = os.getenv("APCA-API-KEY-ID") or os.getenv("APCA_API_KEY_ID")
    if not key:
        raise EnvironmentError("APCA-API-KEY-ID not set in .env")
    return key


def _alpaca_secret() -> str:
    secret = os.getenv("APCA-API-SECRET-KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not secret:
        raise EnvironmentError("APCA-API-SECRET-KEY not set in .env")
    return secret


# ── Tests: Alpaca Paper Trading ────────────────────────────────────────────────

def test_alpaca_account() -> str:
    from alpaca.trading.client import TradingClient

    client = TradingClient(
        api_key=_alpaca_key(),
        secret_key=_alpaca_secret(),
        paper=True,
    )
    account = client.get_account()
    return (
        f"account_id={str(account.id)[:8]}… | "
        f"cash=${float(account.cash):,.2f} | "
        f"portfolio=${float(account.portfolio_value):,.2f}"
    )


def test_alpaca_positions() -> str:
    from alpaca.trading.client import TradingClient

    client = TradingClient(
        api_key=_alpaca_key(),
        secret_key=_alpaca_secret(),
        paper=True,
    )
    positions = client.get_all_positions()
    return f"{len(positions)} open position(s)"


def test_alpaca_market_data() -> str:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestBarRequest

    client = StockHistoricalDataClient(
        api_key=_alpaca_key(),
        secret_key=_alpaca_secret(),
    )
    req = StockLatestBarRequest(symbol_or_symbols=["AAPL", "MSFT"])
    bars = client.get_stock_latest_bar(req)
    results = []
    for symbol, bar in bars.items():
        results.append(f"{symbol}=${bar.close:.2f}")
    return " | ".join(results)


def test_alpaca_news() -> str:
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(
        api_key=_alpaca_key(),
        secret_key=_alpaca_secret(),
    )
    req = NewsRequest(symbols="AAPL", limit=3)
    news = client.get_news(req)
    # NewsSet iterates as (key, value) pairs; articles live in news.data["news"]
    articles = news.data.get("news", []) if hasattr(news, "data") else []
    if not articles:
        return "0 articles returned (may be outside market hours)"
    headline = articles[0].get("headline", "")[:60] if isinstance(articles[0], dict) else articles[0].headline[:60]
    return f"{len(articles)} article(s) — e.g.: \"{headline}…\""


# ── Tests: optional integrations ───────────────────────────────────────────────

def test_anthropic() -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env — skipped")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
    )
    reply = msg.content[0].text.strip()
    return f"model={msg.model} | reply=\"{reply}\""


def test_newsapi() -> str:
    import requests

    api_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise EnvironmentError("NEWS_API_KEY not set in .env — skipped")
    url = "https://newsapi.org/v2/top-headlines"
    resp = requests.get(url, params={"apiKey": api_key, "category": "business", "pageSize": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    total = data.get("totalResults", 0)
    return f"status=ok | totalResults={total}"


def test_polygon() -> str:
    import requests

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise EnvironmentError("POLYGON_API_KEY not set in .env — skipped")
    # Previous close for AAPL — available on the free tier
    url = "https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") not in ("OK", "DELAYED"):
        raise RuntimeError(f"unexpected status from Polygon: {data.get('status')} — {data.get('error', '')}")
    result = data.get("results", [{}])[0]
    return (
        f"AAPL prev close=${result.get('c', '?')} | "
        f"vol={result.get('v', '?'):,} | "
        f"status={data['status']}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nBIP Hackathon 2026 — Connection Test")
    print("=" * 52)

    results: list[bool] = []

    section("Alpaca Paper Trading (MANDATORY)")
    results.append(check("Account balance & status", test_alpaca_account))
    results.append(check("Open positions",           test_alpaca_positions))

    section("Alpaca Market Data (MANDATORY)")
    results.append(check("Latest bar — AAPL, MSFT", test_alpaca_market_data))

    section("Alpaca News Feed (MANDATORY)")
    results.append(check("Recent news — AAPL",      test_alpaca_news))

    section("Anthropic Claude (optional)")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        results.append(check("Claude API ping", test_anthropic))
    else:
        print(f"  {SKIP} ANTHROPIC_API_KEY not set — add it to .env to test")

    section("NewsAPI (optional)")
    newsapi_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        results.append(check("Top business headlines", test_newsapi))
    else:
        print(f"  {SKIP} NEWS_API_KEY not set — add it to .env to test")

    section("Polygon.io Market Data (optional)")
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        results.append(check("Previous close — AAPL", test_polygon))
    else:
        print(f"  {SKIP} POLYGON_API_KEY not set — add it to .env to test")

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'=' * 52}")
    if passed == total:
        print(f"\033[92mAll {total} checks passed. You are ready to start!\033[0m")
    else:
        failed = total - passed
        print(f"\033[91m{failed}/{total} check(s) failed. Fix the errors above before the hackathon.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Discovery phase — 5-step pipeline:

  1. Extract keywords from user prompt
  2. Web search (NewsAPI + Polygon) to find stocks mentioned in real articles
  3. Build enriched context (web headlines + Alpaca market news + ranked candidates)
  4. gemma4:12b proposes 3-5 tickers aligned with the prompt
  5. Validate each ticker is tradeable on Alpaca
"""
import concurrent.futures
import json
import re
from collections import Counter

import requests

import ollama

from src.agent import config
from src.agent import journal as journal_module

_MAX_CANDIDATES = 5
_MIN_CANDIDATES = 2

# ── LLM schema ────────────────────────────────────────────────────────────────

_DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker":     {"type": "string"},
                    "reason":     {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["ticker", "reason", "confidence"],
            },
        },
    },
    "required": ["candidates"],
}

_SYSTEM_PROMPT = (
    "You are a US equity stock-screening assistant.\n"
    "You will receive:\n"
    "  - An investment strategy prompt from the user\n"
    "  - Web search results (news articles mentioning relevant companies)\n"
    "  - A ranked list of stock candidates extracted from those articles\n"
    "  - Recent Alpaca market news\n\n"
    "Your task: select 3–5 US-listed stock tickers that BEST MATCH the strategy.\n\n"
    "Rules:\n"
    "- Only suggest real, actively-traded US stocks (NYSE / NASDAQ).\n"
    "- Ticker: 1–5 uppercase letters.\n"
    "- reason: ONE concise sentence (max 120 chars) explaining alignment with the strategy.\n"
    "- confidence: 0.0–1.0 — how well the stock fits.\n"
    "- Do NOT repeat tickers. Do NOT invent companies.\n"
    "- Order by confidence descending.\n"
    "- Output ONLY valid JSON matching the schema."
)

# ── Step 0: English search-term extraction via LLM ───────────────────────────

_SEARCH_TERMS_SCHEMA = {
    "type": "object",
    "properties": {
        "search_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["search_terms"],
}


def _extract_english_search_terms(prompt: str, t_behavior: int) -> list[str]:
    """
    Use qwen2.5:3b to extract 5-8 English search terms (sector names, company names,
    technologies) from the user prompt regardless of its language.
    Falls back to raw word extraction on failure.
    """
    try:
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,   # qwen2.5:3b — fast
            prompt=(
                "Extract 5-8 English search terms (company names, sector names, technologies) "
                "from the following investment strategy. "
                "Return ONLY valid JSON with key 'search_terms' (array of strings). "
                "Terms must be in English and suitable for searching financial news and stock databases.\n\n"
                f"Strategy: {prompt}"
            ),
            format=_SEARCH_TERMS_SCHEMA,
            options={"temperature": 0.1, "num_predict": 150},
            keep_alive="30s",
        )
        raw = resp.get("response", "{}")
        terms = json.loads(raw).get("search_terms", [])
        terms = [str(t).replace("_", " ").strip() for t in terms if t and len(str(t).strip()) >= 2][:8]
        if terms:
            return terms
    except Exception:
        pass
    # Fallback: raw word extraction (works for English prompts)
    stop = {"that", "with", "this", "from", "have", "invest", "stock", "market",
            "trading", "prefer", "focus", "about", "into", "want", "like"}
    words = re.findall(r"[a-zA-Z]{4,}", prompt.lower())
    return [w for w in words if w not in stop][:8]


# ── Keyword extraction (raw, for Alpaca news filter) ─────────────────────────

def _extract_keywords(prompt: str) -> list[str]:
    """Raw keyword extraction for Alpaca news filter (language-agnostic)."""
    stop = {
        "della", "delle", "degli", "dell", "vuole", "voglio", "sono",
        "that", "with", "this", "from", "have", "invest", "stock", "market",
        "trading", "azioni", "aziende", "settore", "nelle", "negli", "sulle",
        "orientato", "basato", "prefer", "focus", "about", "into", "want",
    }
    words = re.findall(r"[a-zA-Zàèìòùéáíóú]{4,}", prompt.lower())
    return [w for w in words if w not in stop][:10]


# ── Ticker mention extraction from free text ──────────────────────────────────

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
# Common English/Italian words that look like tickers — exclude them
_TICKER_FALSE_POSITIVES = {
    "A", "I", "IT", "AI", "IS", "AT", "BE", "BY", "DO", "GO", "IF", "IN",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "AN", "AS",
    "CEO", "CFO", "IPO", "ETF", "GDP", "USD", "EUR", "API", "USA", "UK",
    "NYSE", "NASDAQ", "SEC", "FED", "ECB", "IMF", "WHO", "UN",
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD",
    "ALL", "NEW", "INC", "LLC", "LTD", "CO", "PLC",
}


def _extract_tickers_from_text(text: str) -> list[str]:
    found = _TICKER_RE.findall(text)
    return [t for t in found if t not in _TICKER_FALSE_POSITIVES and len(t) >= 2]


# ── Step 1b: Web search via NewsAPI ──────────────────────────────────────────

def _search_newsapi(keywords: list[str], max_articles: int = 15) -> list[dict]:
    """Search NewsAPI for articles matching the keywords. Returns article dicts."""
    if not config.NEWS_API_KEY:
        return []
    query = " OR ".join(keywords[:5])
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": max_articles,
                "apiKey": config.NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "source": a.get("source", {}).get("name", ""),
                "title": a.get("title", "") or "",
                "description": (a.get("description", "") or "")[:200],
            }
            for a in articles
        ]
    except Exception as exc:
        journal_module.log_error(source="DiscoveryAgent.newsapi", error=str(exc))
        return []


# ── Step 1b: Company name extraction → ticker resolution via Polygon ──────────

_COMPANY_NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["companies"],
}


def _extract_company_names(
    prompt: str,
    news_articles: list[dict],
    en_terms: list[str],
    t_behavior: int,
) -> list[str]:
    """
    Use qwen2.5:3b to extract US publicly-listed company names from the prompt
    and news articles. Returns up to 15 company name strings.
    """
    # Build a short news digest for the LLM
    news_digest = "\n".join(
        f"- {a['title']}" for a in news_articles[:10]
    ) or "(no news)"

    system_msg = (
        "You are a US equity analyst. "
        "Extract the names of companies that are:\n"
        "1. PUBLICLY TRADED on NYSE or NASDAQ (NOT private companies like Anthropic, OpenAI, SpaceX)\n"
        "2. US-listed common stocks (or major ADRs like NVO, ASML)\n"
        "3. Relevant to the investment strategy\n\n"
        "IMPORTANT: Always include at least 8 well-known sector leaders even if not mentioned "
        "in the headlines. Examples:\n"
        "- AI/Cloud: Nvidia, Microsoft, Alphabet, Amazon, Meta, AMD\n"
        "- Green energy: NextEra Energy, First Solar, Enphase Energy, Vestas\n"
        "- Pharma/Biotech: Eli Lilly, Pfizer, Moderna, Johnson & Johnson, AbbVie, Regeneron\n"
        "- Defense: Lockheed Martin, Raytheon, Northrop Grumman, L3Harris\n\n"
        "Return ONLY valid JSON with key 'companies' (array of company name strings). "
        "Return 8-15 entries. Only real publicly traded companies."
    )
    user_msg = (
        f"Investment strategy: {prompt}\n\n"
        f"Key sectors/themes (in English): {', '.join(en_terms)}\n\n"
        f"Recent news headlines:\n{news_digest}\n\n"
        "List 8-15 publicly-traded US companies aligned with this strategy:"
    )
    try:
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=f"{system_msg}\n\n{user_msg}",
            format=_COMPANY_NAMES_SCHEMA,
            options={"temperature": 0.1, "num_predict": 200},
            keep_alive="30s",
        )
        raw = resp.get("response", "{}")
        names = json.loads(raw).get("companies", [])
        return [str(n).strip() for n in names if n and len(str(n).strip()) >= 2][:15]
    except Exception as exc:
        journal_module.log_error(source="DiscoveryAgent.company_names", error=str(exc))
        return []


def _polygon_sector_tickers(en_terms: list[str], max_results: int = 15) -> list[dict]:
    """
    Single Polygon query per unique sector term (max 2 calls to stay within rate limit).
    Returns top common-stock tickers for each sector keyword.
    """
    if not config.POLYGON_API_KEY:
        return []
    import time as _time
    found: dict[str, str] = {}
    # Use only the first 2 terms to avoid rate-limit (free tier = 5 req/min)
    for kw in en_terms[:2]:
        try:
            resp = requests.get(
                "https://api.polygon.io/v3/reference/tickers",
                params={
                    "search": kw,
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "active": "true",
                    "limit": min(max_results, 10),
                    "apiKey": config.POLYGON_API_KEY,
                },
                timeout=8,
            )
            resp.raise_for_status()
            for item in resp.json().get("results", []):
                ticker = str(item.get("ticker", "")).upper()
                name = str(item.get("name", ""))
                if re.fullmatch(r"[A-Z]{1,5}", ticker):
                    found[ticker] = name
                if len(found) >= max_results:
                    break
            _time.sleep(0.5)   # gentle rate limit guard
        except Exception as exc:
            journal_module.log_error(source="DiscoveryAgent.polygon", error=f"{kw}: {exc}")
    return [{"ticker": t, "name": n} for t, n in found.items()]


# ── Rank candidates from web search ──────────────────────────────────────────

def _rank_web_candidates(
    news_articles: list[dict],
    polygon_tickers: list[dict],
) -> list[dict]:
    """
    Strategy:
    - Polygon results are the PRIMARY source of real tickers (type=CS confirmed)
    - News mentions BOOST the score of polygon tickers
    - Tickers only from news text (not in polygon) are NOT included — too noisy

    Returns sorted list of {ticker, name, mentions, polygon_match}.
    """
    polygon_set: dict[str, str] = {p["ticker"]: p["name"] for p in polygon_tickers}

    # Build short-name index for each polygon ticker (first word of company name)
    # e.g. "Nvidia Corp" → "nvidia"
    ticker_name_tokens: dict[str, list[str]] = {}
    for ticker, name in polygon_set.items():
        tokens = [w.lower() for w in re.split(r"\W+", name) if len(w) >= 4]
        ticker_name_tokens[ticker] = tokens

    # Count mentions: match both ticker symbol AND company name tokens in articles
    mention_counter: Counter = Counter()
    for a in news_articles:
        text_upper = (a.get("title", "") + " " + a.get("description", "")).upper()
        text_lower = text_upper.lower()

        # Ticker-symbol match
        for t in _extract_tickers_from_text(text_upper):
            if t in polygon_set:
                mention_counter[t] += 2   # ticker match is strong signal

        # Company name match
        for ticker, tokens in ticker_name_tokens.items():
            if any(tok in text_lower for tok in tokens):
                mention_counter[ticker] += 1

    ranked: list[dict] = []
    for ticker, name in polygon_set.items():
        ranked.append({
            "ticker": ticker,
            "name": name,
            "mentions": mention_counter.get(ticker, 0),
            "polygon_match": True,
        })

    # Primary sort: news mentions (direct signal of relevance)
    # Secondary: alphabetical (stable tie-break)
    ranked.sort(key=lambda x: (-x["mentions"], x["ticker"]))
    return ranked[:30]


# ── Build LLM prompt ─────────────────────────────────────────────────────────

def _build_llm_prompt(
    user_prompt: str,
    news_articles: list[dict],
    ranked_candidates: list[dict],
    alpaca_headlines: list[str],
    company_names: list[str],
) -> str:
    # Web news block
    web_block = "\n".join(
        f"- [{a['source']}] {a['title']} — {a['description']}"
        for a in news_articles[:12]
    ) or "(no web articles found)"

    # Company names extracted from strategy + news
    companies_block = ", ".join(company_names) if company_names else "(none identified)"

    # Polygon sector candidates (with news-mention boost)
    cand_block = "\n".join(
        f"- {c['ticker']}"
        + (f" ({c['name']})" if c["name"] else "")
        + (f"  news_mentions:{c['mentions']}" if c["mentions"] > 0 else "")
        for c in ranked_candidates[:20]
    ) or "(no Polygon sector candidates)"

    # Alpaca market news
    alpaca_block = "\n".join(f"- {h}" for h in alpaca_headlines[:8]) or "(no Alpaca news)"

    return (
        f"=== INVESTMENT STRATEGY ===\n{user_prompt}\n\n"
        f"=== COMPANIES MENTIONED IN STRATEGY & NEWS ===\n{companies_block}\n"
        "(Note: map these company names to their US stock tickers using your knowledge)\n\n"
        f"=== WEB SEARCH RESULTS (news articles) ===\n{web_block}\n\n"
        f"=== SECTOR TICKERS FROM POLYGON (ranked by news relevance) ===\n{cand_block}\n\n"
        f"=== ALPACA MARKET NEWS ===\n{alpaca_block}\n\n"
        "Based on ALL of the above, select EXACTLY 5 US stock tickers that BEST MATCH "
        "the investment strategy. Prefer companies explicitly mentioned above. "
        "You MUST return 5 tickers — use your knowledge of well-known sector leaders "
        "if fewer are mentioned in the context. "
        "Return ONLY valid JSON."
    )


# ── Main DiscoveryAgent ───────────────────────────────────────────────────────

class DiscoveryAgent:
    def __init__(self) -> None:
        self._session_id: str = ""

    def _fetch_alpaca_headlines(self, tool_executor, keywords: list[str]) -> list[str]:
        result = tool_executor.get_market_news(keywords=keywords, limit=10)
        articles = result.data.get("articles", [])
        lines = []
        for a in articles:
            title = a.get("title", "").strip()
            syms = ", ".join(a.get("symbols", []))
            line = title + (f" [{syms}]" if syms else "")
            lines.append(line)
        return lines

    def _call_llm(self, llm_prompt: str, t_behavior: int) -> list[dict]:
        def _call():
            resp = ollama.generate(
                model=config.OLLAMA_REASONING_MODEL,
                prompt=f"{_SYSTEM_PROMPT}\n\n{llm_prompt}",
                format=_DISCOVERY_SCHEMA,
                options={"temperature": 0.3, "num_predict": 900},
                keep_alive="30s",
            )
            raw = resp.get("response", "{}")
            # Robust parse: if truncated JSON, extract completed candidate objects
            try:
                parsed = json.loads(raw)
                return parsed.get("candidates", [])
            except json.JSONDecodeError:
                # Recover completed candidate objects from truncated output
                candidates = []
                for m in re.finditer(
                    r'\{[^{}]*"ticker"\s*:\s*"([A-Z]{1,5})"[^{}]*"reason"\s*:\s*"([^"]*)"[^{}]*"confidence"\s*:\s*([\d.]+)[^{}]*\}',
                    raw,
                ):
                    try:
                        candidates.append({
                            "ticker": m.group(1),
                            "reason": m.group(2),
                            "confidence": float(m.group(3)),
                        })
                    except Exception:
                        pass
                return candidates

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call)
                return fut.result(timeout=t_behavior)
        except Exception as exc:
            journal_module.log_error(
                source="DiscoveryAgent.llm", error=str(exc),
                session_id=self._session_id,
            )
            return []

    def _validate_candidates(self, candidates: list[dict], tool_executor) -> list[dict]:
        valid: list[dict] = []
        seen: set[str] = set()
        for c in candidates:
            ticker = str(c.get("ticker", "")).upper().strip()
            if not ticker or not re.fullmatch(r"[A-Z]{1,5}", ticker):
                continue
            if ticker in seen or ticker in _TICKER_FALSE_POSITIVES:
                continue
            reason = str(c.get("reason", ""))[:120]
            confidence = max(0.0, min(1.0, float(c.get("confidence", 0.5))))
            if tool_executor.validate_ticker(ticker):
                seen.add(ticker)
                valid.append({"ticker": ticker, "reason": reason, "confidence": confidence})
            if len(valid) >= _MAX_CANDIDATES:
                break
        return sorted(valid, key=lambda x: x["confidence"], reverse=True)

    def discover(
        self,
        prompt: str,
        tool_executor,
        t_behavior: int,
        dashboard=None,
    ) -> list[dict]:
        """
        Full 5-step discovery pipeline. Returns validated candidate list.
        Falls back to config.TICKERS if fewer than _MIN_CANDIDATES survive.
        """

        def _log(msg: str, level: str = "info") -> None:
            if dashboard:
                dashboard.log(msg, level)

        # ── Step 0: extract English search terms via LLM ──────────────────────
        _log(
            f"  [discovery] Estrazione termini di ricerca in inglese "
            f"({config.OLLAMA_SENTIMENT_MODEL})…",
            "info",
        )
        en_terms = _extract_english_search_terms(prompt, t_behavior)
        _log(f"  [discovery] Termini EN: {', '.join(en_terms)}", "info")

        # Raw keywords (any language) for Alpaca news filter
        raw_keywords = _extract_keywords(prompt)

        # ── Step 1a: fetch news + Alpaca headlines + Polygon sector (parallel) ──
        _log("  [discovery] Ricerca news (NewsAPI + Alpaca + Polygon) in corso…", "info")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fut_news    = pool.submit(_search_newsapi, en_terms, 15)
            fut_alpaca  = pool.submit(self._fetch_alpaca_headlines, tool_executor, raw_keywords)
            fut_polygon = pool.submit(_polygon_sector_tickers, en_terms, 15)

            news_articles    = fut_news.result(timeout=20)
            alpaca_headlines = fut_alpaca.result(timeout=15)
            polygon_tickers  = fut_polygon.result(timeout=20)

        _log(
            f"  [discovery] NewsAPI: {len(news_articles)} articoli  "
            f"Alpaca: {len(alpaca_headlines)} headline  "
            f"Polygon (settore): {len(polygon_tickers)} ticker",
            "ok",
        )

        # ── Step 1b: extract company names from prompt + news via LLM ─────────
        _log(
            f"  [discovery] Estrazione nomi aziende ({config.OLLAMA_SENTIMENT_MODEL})…",
            "info",
        )
        company_names = _extract_company_names(prompt, news_articles, en_terms, t_behavior)
        _log(f"  [discovery] Aziende identificate: {', '.join(company_names)}", "info")

        # ── Step 2: build ranked candidates from Polygon sector results ────────
        ranked = _rank_web_candidates(news_articles, polygon_tickers)
        if ranked:
            top_str = ", ".join(
                f"{r['ticker']}(×{r['mentions']})" for r in ranked[:10] if r["mentions"] > 0
            ) or ", ".join(r["ticker"] for r in ranked[:8])
            _log(f"  [discovery] Candidati Polygon rankinizzati: {top_str}", "info")
        else:
            _log("  [discovery] Nessun candidato Polygon.", "warn")

        # ── Step 4: LLM selection ──────────────────────────────────────────────
        _log(
            f"  [discovery] {config.OLLAMA_REASONING_MODEL} seleziona i ticker…",
            "info",
        )
        llm_prompt = _build_llm_prompt(prompt, news_articles, ranked, alpaca_headlines, company_names)
        raw_candidates = self._call_llm(llm_prompt, t_behavior)
        _log(
            f"  [discovery] LLM ha proposto {len(raw_candidates)} candidati: "
            + ", ".join(c.get("ticker", "?") for c in raw_candidates),
            "info",
        )

        # ── Step 5: validate on Alpaca ────────────────────────────────────────
        _log("  [discovery] Validazione ticker su Alpaca…", "info")
        validated = self._validate_candidates(raw_candidates, tool_executor)
        _log(
            f"  [discovery] Ticker validati: {', '.join(c['ticker'] for c in validated)}",
            "ok" if validated else "warn",
        )

        # ── Fallback ──────────────────────────────────────────────────────────
        if len(validated) < _MIN_CANDIDATES:
            journal_module.log_error(
                source="DiscoveryAgent",
                error=f"Only {len(validated)} valid ticker(s) after discovery; "
                      "falling back to config.TICKERS",
                session_id=self._session_id,
            )
            _log(
                "  [discovery] Ticker insufficienti — fallback su config.TICKERS.",
                "warn",
            )
            validated = [
                {
                    "ticker": t,
                    "reason": "Ticker di default (discovery fallback)",
                    "confidence": 0.5,
                }
                for t in config.TICKERS
            ]

        return validated

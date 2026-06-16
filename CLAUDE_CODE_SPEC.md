# BIP Hackathon 2026 — Trading Agent: Claude Code Implementation Spec

> **To Claude Code:** This document is your complete implementation contract. Read it fully before writing any code. Implement every section in the exact order presented. Do not skip sections, do not add dependencies not listed here, do not make architectural decisions not covered here — ask instead. Every constraint is intentional.

---

## 0. Meta-instructions for Claude Code

- Implement **top-to-bottom**: each module depends on the one before it. Do not jump ahead.
- After implementing each module, run its tests before moving to the next.
- If a section says "never" or "always", treat it as a hard constraint, not a suggestion.
- All Python must be compatible with **Python 3.12+**.
- Package manager: **uv** exclusively. Never use pip directly.
- When a function signature is specified, implement it exactly — same name, same parameters, same return type.
- When in doubt about a design decision not covered here, choose the simpler option and add a `# TODO: decision needed` comment.

---

## 1. Project Overview

An autonomous trading agent operating on Alpaca Paper Trading (simulated broker). Runs a continuous loop every N seconds (adaptive timeout) without human intervention. Every decision is motivated by explicit reasoning, recorded in a persistent JSONL journal, and displayed in a rich terminal UI.

### Evaluation axes

| Criterion | Weight | How we cover it |
|---|---|---|
| Demonstrable functionality | 40% | Loop stable 30+ min, orders execute, errors handled, persistent logs |
| Reasoning quality | 35% | No hallucinated data, confidence gate, STALE penalty, self-reflection |
| Originality | 25% | Dual-model, HOT/WARM/COLD memory, imitative layer, adaptive timeout, cambio comportamento |

### Ambition level: **Level 3** (Autonomous Agent)

---

## 2. Repository Layout

Create this exact structure. Do not deviate.

```
bip-hackathon2026/
├── pyproject.toml
├── .env                          # never commit
├── .env.example                  # commit this
├── .gitignore
├── README.md
│
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── config.py             # env + constants
│       ├── adaptive_timeout.py   # Module 1 — latency tracker
│       ├── tool_executor.py      # Module 2 — all external calls
│       ├── journal.py            # Module 3 — JSONL read/write
│       ├── memory_manager.py     # Module 4 — HOT/WARM/COLD
│       ├── imitative_layer.py    # Module 5 — strategy dataset
│       ├── sentiment.py          # Module 6 — qwen2.5:3b
│       ├── reasoner.py           # Module 7 — Gemma4:12b
│       ├── broker.py             # Module 8 — Alpaca orders
│       ├── session.py            # Module 9 — startup/resume
│       ├── behavior.py           # Module 10 — comportamento + fallback
│       └── loop.py               # Module 11 — AgentLoop
│
├── ui/
│   └── dashboard.py              # Module 12 — rich/textual TUI
│
├── main.py                       # entry point
│
├── tests/
│   ├── test_connections.py       # infra connectivity tests
│   ├── test_adaptive_timeout.py
│   ├── test_tool_executor.py
│   ├── test_journal.py
│   ├── test_memory_manager.py
│   ├── test_behavior.py
│   └── test_session.py
│
└── data/
    ├── journal.jsonl             # created at runtime
    ├── error_log.jsonl           # created at runtime
    ├── session.json              # created at runtime
    └── strategies/
        └── imitative_dataset.json  # created at setup
```

---

## 3. Dependencies and Environment

### 3.1 `pyproject.toml`

```toml
[project]
name = "bip-hackathon2026"
version = "0.1.0"
description = "Autonomous trading agent — BIP Hackathon 2026"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "alpaca-py>=0.43.4",
    "anthropic>=0.109.1",
    "python-dotenv>=1.2.2",
    "requests>=2.34.2",
    "ollama>=0.5.0",
    "rich>=13.7.0",
    "textual>=0.60.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agent"]
```

Install: `uv sync`

### 3.2 `.env.example`

```dotenv
# ── Alpaca Paper Trading ─────────────────────────────────────────
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# ── Ollama ───────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REASONING_MODEL=gemma4:12b
OLLAMA_SENTIMENT_MODEL=qwen2.5:3b

# ── Agent behaviour ──────────────────────────────────────────────
TICKERS=AAPL,TSLA,NVDA,MSFT
CONFIDENCE_THRESHOLD_NORMAL=0.65
CONFIDENCE_THRESHOLD_CONSERVATIVE=0.80
MAX_POSITION_PCT_NORMAL=0.10
MAX_POSITION_PCT_CONSERVATIVE=0.05
DRAWDOWN_THRESHOLD=0.05

# ── Adaptive timeout base values (overridden at runtime) ─────────
T_WAIT_MULTIPLIER=3.0
T_BEHAVIOR_MULTIPLIER=5.0
T_WAIT_MIN=15
T_WAIT_MAX=120
T_BEHAVIOR_MIN=20
T_BEHAVIOR_MAX=180

# ── Memory ───────────────────────────────────────────────────────
HOT_WINDOW_SIZE=5
WARM_COMPACTION_TRIGGER=15

# ── Paths ────────────────────────────────────────────────────────
JOURNAL_PATH=data/journal.jsonl
ERROR_LOG_PATH=data/error_log.jsonl
SESSION_PATH=data/session.json
IMITATIVE_DATASET_PATH=data/strategies/imitative_dataset.json
```

### 3.3 Ollama setup

```bash
ollama serve &
ollama pull gemma4:12b
ollama pull qwen2.5:3b
ollama list   # verify both appear
```

### 3.4 `.gitignore`

```gitignore
.env
__pycache__/
*.pyc
.venv/
data/journal.jsonl
data/error_log.jsonl
data/session.json
agent.log
```

---

## 4. `src/agent/config.py`

Load all environment variables here. Every other module imports from config — never from `os.environ` directly.

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER: bool = True  # HARDCODED — never change

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REASONING_MODEL: str = os.getenv("OLLAMA_REASONING_MODEL", "gemma4:12b")
OLLAMA_SENTIMENT_MODEL: str = os.getenv("OLLAMA_SENTIMENT_MODEL", "qwen2.5:3b")

# Agent behaviour
TICKERS: list[str] = os.getenv("TICKERS", "AAPL,TSLA,NVDA,MSFT").split(",")
CONFIDENCE_THRESHOLD_NORMAL: float = float(os.getenv("CONFIDENCE_THRESHOLD_NORMAL", "0.65"))
CONFIDENCE_THRESHOLD_CONSERVATIVE: float = float(os.getenv("CONFIDENCE_THRESHOLD_CONSERVATIVE", "0.80"))
MAX_POSITION_PCT_NORMAL: float = float(os.getenv("MAX_POSITION_PCT_NORMAL", "0.10"))
MAX_POSITION_PCT_CONSERVATIVE: float = float(os.getenv("MAX_POSITION_PCT_CONSERVATIVE", "0.05"))
DRAWDOWN_THRESHOLD: float = float(os.getenv("DRAWDOWN_THRESHOLD", "0.05"))

# Adaptive timeout
T_WAIT_MULTIPLIER: float = float(os.getenv("T_WAIT_MULTIPLIER", "3.0"))
T_BEHAVIOR_MULTIPLIER: float = float(os.getenv("T_BEHAVIOR_MULTIPLIER", "5.0"))
T_WAIT_MIN: int = int(os.getenv("T_WAIT_MIN", "15"))
T_WAIT_MAX: int = int(os.getenv("T_WAIT_MAX", "120"))
T_BEHAVIOR_MIN: int = int(os.getenv("T_BEHAVIOR_MIN", "20"))
T_BEHAVIOR_MAX: int = int(os.getenv("T_BEHAVIOR_MAX", "180"))

# Memory
HOT_WINDOW_SIZE: int = int(os.getenv("HOT_WINDOW_SIZE", "5"))
WARM_COMPACTION_TRIGGER: int = int(os.getenv("WARM_COMPACTION_TRIGGER", "15"))

# Paths
JOURNAL_PATH: str = os.getenv("JOURNAL_PATH", "data/journal.jsonl")
ERROR_LOG_PATH: str = os.getenv("ERROR_LOG_PATH", "data/error_log.jsonl")
SESSION_PATH: str = os.getenv("SESSION_PATH", "data/session.json")
IMITATIVE_DATASET_PATH: str = os.getenv("IMITATIVE_DATASET_PATH", "data/strategies/imitative_dataset.json")
```

---

## 5. Module 1 — `adaptive_timeout.py`

**Responsibility:** Measures and tracks real-time latency of Alpaca API and Ollama. Computes adaptive timeout values `T_wait` and `T_behavior` used throughout the system. Updated after every external call. Never raises.

### 5.1 Interface

```python
class AdaptiveTimeout:
    def record_api_latency(self, latency_seconds: float) -> None: ...
    def record_ollama_latency(self, latency_seconds: float) -> None: ...
    def t_wait(self) -> int:
        """T_wait = clamp(api_avg * T_WAIT_MULTIPLIER, T_WAIT_MIN, T_WAIT_MAX)"""
    def t_behavior(self) -> int:
        """T_behavior = clamp(ollama_avg * T_BEHAVIOR_MULTIPLIER, T_BEHAVIOR_MIN, T_BEHAVIOR_MAX)"""
    def ping_api(self) -> float:
        """Ping Alpaca clock endpoint. Record latency. Return seconds."""
    def ping_ollama(self) -> float:
        """Ping Ollama /api/tags. Record latency. Return seconds."""
    def calibrate(self) -> None:
        """Run ping_api() and ping_ollama() 3 times each. Warm up averages."""
    def summary(self) -> dict:
        """Return {api_avg, ollama_avg, t_wait, t_behavior}"""
```

### 5.2 Implementation notes

- Use a **rolling window of last 10 measurements** per source (API and Ollama separately).
- Average = mean of window. On first call, window has 1 element — that's fine.
- `clamp(value, min, max)` = `max(min_val, min(max_val, value))`.
- `ping_api()` calls `GET https://paper-api.alpaca.markets/v2/clock` with the API key headers. Measures wall-clock time of the full round trip.
- `ping_ollama()` calls `GET {OLLAMA_BASE_URL}/api/tags`. Measures wall-clock time.
- Both ping methods catch all exceptions and return the current average on failure (never raise).
- Thread-safe: use `threading.Lock` on the rolling windows.

### 5.3 Tests — `tests/test_adaptive_timeout.py`

```python
def test_t_wait_clamps_to_min():
    """If API is very fast, T_wait must not go below T_WAIT_MIN."""

def test_t_wait_clamps_to_max():
    """If API is very slow, T_wait must not exceed T_WAIT_MAX."""

def test_rolling_window_size():
    """After 15 records, window contains exactly 10 elements."""

def test_ping_api_on_failure_returns_average():
    """If ping_api raises (bad URL), it returns the current average, not 0."""

def test_calibrate_runs_without_raising():
    """calibrate() completes without raising even if Ollama is unreachable."""
```

---

## 6. Module 2 — `tool_executor.py`

**Responsibility:** Every external call (price, portfolio, news) passes through this layer. Handles retry with exponential backoff, updates `AdaptiveTimeout` after each call, sets `STALE_DATA` flag on definitive failure, logs errors to `error_log.jsonl`. Never raises.

### 6.1 Data types

```python
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone

@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]
    stale: bool = False
    staleness_seconds: int = 0
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
```

### 6.2 Interface

```python
class ToolExecutor:
    def __init__(self, adaptive_timeout: AdaptiveTimeout): ...

    def get_price(self, ticker: str) -> ToolResult:
        """
        Fetch latest bar. Returns ToolResult with:
        data = {
            "ticker": str,
            "price": float,
            "timestamp": str,   # ISO 8601 UTC — from the bar, not wall clock
            "volume": int
        }
        """

    def get_bars(self, ticker: str, limit: int = 5) -> ToolResult:
        """Fetch last N bars. data = {"ticker", "closes": list[float], "ma": float, "trend": str}"""

    def get_news(self, ticker: str) -> ToolResult:
        """
        Fetch news via Alpaca News Feed.
        data = {"ticker": str, "articles": [{"title": str, "summary": str}]}
        Max 3 articles.
        """

    def get_portfolio(self) -> ToolResult:
        """
        data = {
            "cash": float,
            "portfolio_value": float,
            "positions": {symbol: {"qty": int, "market_value": float, "avg_entry_price": float}},
            "pnl_pct": float        # (portfolio_value - 100000) / 100000
        }
        """

    def is_market_open(self) -> bool: ...

    def unblacklist(self, ticker: str) -> None: ...
```

### 6.3 Implementation notes

- `MAX_RETRIES = 3`, `INITIAL_BACKOFF = 1.0s`, `BACKOFF_FACTOR = 2.0`.
- After each successful call, record latency in `AdaptiveTimeout`.
- After each failed call, write to `error_log.jsonl` via `journal.log_error()` (Module 3).
- After 3 consecutive failures on the same ticker, add it to `_blacklisted: set[str]`. Log the blacklisting.
- Cache: `{ticker: {"price": ToolResult, "bars": ToolResult, "news": ToolResult}}`. Cache hit only used when all retries fail.
- `get_portfolio()` uses `__portfolio__` as the cache key.
- `pnl_pct` is computed as `(portfolio_value - 100_000.0) / 100_000.0`. Starting capital is always 100k.
- News endpoint: `GET https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit=3` with headers `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`.

### 6.4 Tests — `tests/test_tool_executor.py`

```python
def test_get_price_returns_tool_result_not_raises():
    """ToolExecutor.get_price never raises even on invalid ticker."""

def test_stale_data_flag_set_on_failure():
    """After all retries fail and cache exists, result.stale == True."""

def test_blacklist_after_three_failures():
    """Ticker is blacklisted after 3 consecutive failures."""

def test_latency_recorded_on_success():
    """AdaptiveTimeout.record_api_latency is called on every successful fetch."""

def test_error_written_to_error_log():
    """On failure, error_log.jsonl receives a new entry."""
```

---

## 7. Module 3 — `journal.py`

**Responsibility:** JSONL writer and reader for both trade decisions and errors. All writes are O(1) append. Reads use tail-reading (O(N), never loads full file). `outcome_update()` fills `price_after` and `outcome_pct` in the previous cycle's entry. Both `journal.jsonl` and `error_log.jsonl` are managed here.

### 7.1 Journal entry schema

Every trade journal entry must contain **exactly** these fields, no more, no less:

```python
{
    # Identity
    "ts":                str,    # ISO 8601 UTC
    "cycle":             int,
    "ticker":            str,
    "session_id":        str,    # UUID of current session

    # Decision
    "action":            str,    # "buy" | "sell" | "hold"
    "conf":              float,
    "conf_raw":          float,
    "stale_penalty":     float,
    "reasoning":         str,    # max 400 chars
    "accuracy_review":   str,    # max 200 chars
    "decision_source":   str,    # "agent" | "user_confirmed" | "user_override" | "autonomous_timeout"

    # Input data (traceable — Rule 1)
    "price":             float,
    "price_timestamp":   str,    # timestamp FROM the ticker JSON object
    "ma5":               float,
    "trend":             str,
    "sentiment":         float,
    "sentiment_label":   str,
    "data_ok":           bool,
    "imitative_source":  str | None,   # e.g. "Buffett:value_investing"
    "prompt_snapshot":   str,          # first 100 chars of active prompt

    # Adaptive timeout used this cycle
    "t_wait_used":       int,
    "t_behavior_used":   int,

    # Behaviour mode
    "mode":              str,    # "normal" | "conservative"
    "portfolio_mode_reason": str | None,

    # Broker
    "order_id":          str | None,
    "market_open":       bool,

    # Outcome (filled at cycle N+1)
    "price_after":       float | None,
    "outcome_pct":       float | None,

    # Portfolio snapshot
    "cash":              float,
    "portfolio_value":   float,
    "pnl_pct":           float,
    "positions":         dict,
}
```

### 7.2 Error log entry schema

```python
{
    "ts":           str,
    "session_id":   str,
    "source":       str,    # "ToolExecutor" | "Reasoner" | "BehaviorManager" | ...
    "ticker":       str | None,
    "error":        str,
    "retry_count":  int,
    "stale_used":   bool,
}
```

### 7.3 Interface

```python
def write_entry(entry: dict) -> None:
    """Append to JOURNAL_PATH. O(1). Never raises."""

def log_error(source: str, error: str, ticker: str | None = None,
              retry_count: int = 0, stale_used: bool = False) -> None:
    """Append to ERROR_LOG_PATH. O(1). Never raises."""

def read_last_n(n: int = 5, path: str = JOURNAL_PATH) -> list[dict]:
    """Read last N entries from tail. O(N). Never loads full file. Never raises."""

def outcome_update(ticker: str, new_price: float, session_id: str) -> None:
    """
    Find the most recent entry for ticker in the current session
    where price_after is None and action in (buy, sell).
    Fill price_after and outcome_pct. Rewrite that single line.
    Never raises.
    """

def build_entry(**kwargs) -> dict:
    """Construct a complete journal entry dict. Validates all required fields are present.
    Raises ValueError if any required field is missing."""

def read_session_summary(session_id: str) -> dict:
    """
    Read journal.jsonl and compute:
    {
        "session_id": str,
        "cycles": int,
        "decisions": {"buy": int, "sell": int, "hold": int},
        "orders_placed": int,
        "autonomous_decisions": int,
        "final_pnl_pct": float | None,
        "errors": int,    # from error_log.jsonl
        "last_portfolio": dict | None,
    }
    """
```

### 7.4 Tests — `tests/test_journal.py`

```python
def test_write_read_roundtrip():
    """write_entry then read_last_n(1) returns the same entry."""

def test_outcome_update_fills_fields():
    """outcome_update correctly computes outcome_pct = (new - old) / old * 100."""

def test_read_last_n_never_loads_full_file():
    """With 1000 entries, read_last_n(5) reads < 10 KB from disk."""

def test_build_entry_raises_on_missing_field():
    """build_entry without 'cycle' raises ValueError."""

def test_log_error_append_only():
    """log_error never overwrites existing entries."""

def test_read_session_summary_counts_correctly():
    """read_session_summary returns correct buy/sell/hold counts."""
```

---

## 8. Module 4 — `memory_manager.py`

**Responsibility:** Three-tier memory per ticker. HOT: last 5 entries (full, in-memory). WARM: entries 6-20 (LLM-compacted summary, regenerated every WARM_COMPACTION_TRIGGER overflows). COLD: everything on disk. Prompt context is constant size regardless of cycle count.

### 8.1 Interface

```python
class MemoryManager:
    def update(self, entry: dict) -> None:
        """Call after every journal write. Pushes to HOT. Triggers WARM compaction on overflow."""

    def build_context(self, ticker: str) -> str:
        """
        Returns a fixed-size string:
        === HISTORICAL SUMMARY ===
        {warm_summary if exists}

        === RECENT DECISIONS (last N) ===
        [ts] ACTION @ $price | conf:X | sentiment:+Y | outcome:Z%
        ...
        """

    def reset_ticker(self, ticker: str) -> None:
        """Clear HOT and WARM for a ticker. Called on behaviour change."""

    def reset_all(self) -> None:
        """Clear all HOT and WARM. Called on full behaviour change."""

    def get_stats(self) -> dict:
        """Return {ticker: {"hot_size": int, "warm_age": int, "total": int}} for all tickers."""
```

### 8.2 Implementation notes

- `_hot: dict[str, list[dict]]` — deque-like, max HOT_WINDOW_SIZE.
- `_warm: dict[str, str]` — LLM-generated summary string.
- `_warm_age: dict[str, int]` — overflows since last compaction.
- Compaction uses `OLLAMA_SENTIMENT_MODEL` (qwen2.5:3b), not Gemma4, to save RAM.
- Compaction fallback: if Ollama unavailable, use `_rule_based_summary()` (compute accuracy % from entries, no LLM).
- `build_context()` returns `"No prior decisions on this ticker."` if both HOT and WARM are empty.

### 8.3 Tests — `tests/test_memory_manager.py`

```python
def test_hot_window_max_size():
    """After HOT_WINDOW_SIZE + 3 updates, HOT has exactly HOT_WINDOW_SIZE entries."""

def test_build_context_constant_size():
    """After 50 updates, build_context() returns roughly the same size string as after 10."""

def test_reset_ticker_clears_hot_and_warm():
    """After reset_ticker, build_context returns 'No prior decisions'."""

def test_compaction_fallback_on_ollama_unavailable():
    """When Ollama is down, compaction uses rule-based summary without raising."""
```

---

## 9. Module 5 — `imitative_layer.py`

**Responsibility:** Loads a static dataset of known investor strategies and public articles. Filters strategies for coherence with the active user prompt. Extracts `imitative_hints` injected into the reasoning prompt each cycle.

### 9.1 Imitative dataset format

Create `data/strategies/imitative_dataset.json` at setup time with this structure:

```json
{
  "strategies": [
    {
      "id": "buffett_value",
      "name": "Warren Buffett — Value Investing",
      "keywords": ["undervalued", "long-term", "fundamentals", "dividend", "moat"],
      "anti_keywords": ["speculation", "short", "crypto", "leverage"],
      "rules": [
        "Buy only when price is significantly below intrinsic value",
        "Hold for the long term — ignore short-term volatility",
        "Prefer companies with durable competitive advantages",
        "Avoid businesses you do not understand"
      ],
      "risk_profile": "conservative",
      "sectors": ["consumer", "finance", "technology"]
    },
    {
      "id": "lynch_growth",
      "name": "Peter Lynch — Growth at Reasonable Price",
      "keywords": ["growth", "earnings", "PEG", "consumer", "retail"],
      "anti_keywords": ["macro", "bonds", "commodities"],
      "rules": [
        "Invest in what you know and understand",
        "Look for companies with PEG ratio below 1",
        "Small-cap growth companies can outperform",
        "Sell when the story changes, not when price drops"
      ],
      "risk_profile": "moderate",
      "sectors": ["consumer", "technology", "healthcare"]
    },
    {
      "id": "simons_quant",
      "name": "Jim Simons — Quantitative Signals",
      "keywords": ["momentum", "trend", "signal", "data", "pattern"],
      "anti_keywords": ["narrative", "opinion", "macro"],
      "rules": [
        "Follow statistical patterns, not narratives",
        "Mean reversion is a valid signal when momentum is absent",
        "Volume confirms price action",
        "Cut losses quickly — do not average down"
      ],
      "risk_profile": "moderate",
      "sectors": ["any"]
    },
    {
      "id": "green_esg",
      "name": "ESG — Green Investing",
      "keywords": ["green", "esg", "sustainable", "renewable", "climate", "environment"],
      "anti_keywords": ["fossil", "arms", "weapons", "tobacco", "gambling"],
      "rules": [
        "Prioritise companies with strong ESG scores",
        "Avoid sectors with high carbon footprint",
        "Renewable energy and clean tech are preferred",
        "Social impact is as important as financial return"
      ],
      "risk_profile": "moderate",
      "sectors": ["energy", "technology", "utilities"]
    },
    {
      "id": "defense_sector",
      "name": "Defense — Sector Investing",
      "keywords": ["defense", "arms", "military", "aerospace", "government contract"],
      "anti_keywords": ["green", "esg", "consumer", "retail"],
      "rules": [
        "Government contracts provide revenue visibility",
        "Geopolitical tension increases sector demand",
        "Prefer primes over sub-contractors for stability",
        "Dividend yield is a key selection criterion"
      ],
      "risk_profile": "conservative",
      "sectors": ["defense", "aerospace", "technology"]
    }
  ]
}
```

### 9.2 Interface

```python
class ImiativeLayer:
    def __init__(self): ...
    # loads dataset from IMITATIVE_DATASET_PATH at init

    def filter_for_prompt(self, prompt: str) -> list[dict]:
        """
        Score each strategy against the prompt using keyword matching.
        Return strategies with score > 0, sorted by score descending.
        Max 2 strategies returned.
        Score = sum(+1 for keyword match) - sum(+2 for anti_keyword match).
        Negative score → strategy excluded.
        """

    def build_hints(self, prompt: str, ticker: str) -> str:
        """
        Return a compact string for injection into the reasoning prompt:

        === IMITATIVE HINTS ===
        [Buffett:value_investing] Buy undervalued, hold long-term, avoid speculation.
        [Lynch:growth] Look for earnings growth, PEG < 1.
        Source: static dataset — not from model memory.
        Active strategy: {matched_strategy_name}

        If no strategy matches, returns empty string.
        """

    def get_active_strategy_id(self, prompt: str) -> str | None:
        """Return the ID of the top-scoring strategy for this prompt, or None."""

    def reload(self) -> None:
        """Reload dataset from disk. Called after behaviour change."""
```

### 9.3 Implementation notes

- Keyword matching is **case-insensitive substring matching** on the full prompt string.
- `build_hints()` always appends `"Source: static dataset — not from model memory."` to make Rule 1 compliance explicit in the journal.
- If `IMITATIVE_DATASET_PATH` does not exist, create it with the default dataset above.

---

## 10. Module 6 — `sentiment.py`

**Responsibility:** Classifies news article sentiment using qwen2.5:3b via Ollama's JSON schema enforcement. Never raises. Returns neutral on failure.

### 10.1 Output schema (enforced via Ollama `format`)

```python
_SCHEMA = {
    "type": "object",
    "properties": {
        "score":     {"type": "number"},      # -1.0 to +1.0
        "label":     {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "rationale": {"type": "string"},      # one sentence
    },
    "required": ["score", "label", "rationale"],
}
```

### 10.2 Interface

```python
def analyse(
    ticker: str,
    articles: list[dict],
    active_prompt: str = "",
    t_behavior: int = 60,
) -> dict:
    """
    Returns:
    {
        "score": float,           # clamped to [-1.0, +1.0]
        "label": str,
        "rationale": str,
        "article_count": int,
        "prompt_filtered": bool,  # True if articles were filtered by prompt coherence
    }

    On failure: returns {"score": 0.0, "label": "neutral", "rationale": "unavailable", ...}
    """
```

### 10.3 Implementation notes

- Filter articles for coherence with `active_prompt` before sending to qwen: remove articles whose content contradicts the active strategy (e.g. exclude green energy articles if prompt is defense-focused).
- Use `keep_alive="30s"` on all Ollama calls.
- `temperature=0.0` for deterministic output.
- `num_predict=200` to cap token output.
- Wrap entire function in try/except — return neutral dict on any failure.

---

## 11. Module 7 — `reasoner.py`

**Responsibility:** Gemma4:12b orchestrator. Receives the fully assembled prompt and returns a structured decision. Applies STALE_DATA penalty deterministically in code. Respects `t_behavior` timeout.

### 11.1 Decision schema (enforced via Ollama `format`)

```python
_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":          {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence":      {"type": "number"},
        "reasoning":       {"type": "string"},       # max 2 sentences
        "accuracy_review": {"type": "string"},       # max 1 sentence
    },
    "required": ["action", "confidence", "reasoning", "accuracy_review"],
}
```

### 11.2 Interface

```python
class Reasoner:
    def decide(
        self,
        ticker: str,
        memory_context: str,
        price: float,
        price_timestamp: str,
        ma5: float,
        trend: str,
        sentiment_score: float,
        sentiment_label: str,
        imitative_hints: str,
        active_prompt: str,
        cash: float,
        positions: dict,
        mode: str,              # "normal" | "conservative"
        stale: bool,
        staleness_seconds: int,
        t_behavior: int,
    ) -> dict:
        """
        Returns:
        {
            "action": str,
            "confidence": float,      # after stale penalty
            "confidence_raw": float,
            "stale_penalty": float,
            "reasoning": str,
            "accuracy_review": str,
        }
        On timeout or error: returns safe hold dict.
        """
```

### 11.3 Implementation notes

**System prompt** (exact text — do not paraphrase):
```
You are a cautious quantitative trading analyst operating on real market data.
You NEVER invent, estimate, or recall prices from memory — you only use the data provided in this prompt.
You NEVER fabricate news or sentiment scores.
When data is stale, uncertain, or confidence is low, you hold — not buy or sell.
Your reasoning must cite specific numbers from the data above.
If the active strategy (imitative hints) conflicts with the data, state the conflict explicitly.
Output only valid JSON matching the schema. Max 2 sentences for reasoning, 1 for accuracy_review.
```

- Wrap the Ollama call in `concurrent.futures.ThreadPoolExecutor` with `timeout=t_behavior`. On timeout, log error and return `_safe_hold()`.
- STALE penalty: `penalty = min(staleness_seconds / 60 * 0.05, 0.40)`. Applied **after** model response.
- `confidence` in output = `max(0.0, confidence_raw - stale_penalty)`.
- `keep_alive="30s"`, `temperature=0.2`, `num_predict=300`.

### 11.4 Safe hold

```python
def _safe_hold(self, reason: str) -> dict:
    return {
        "action": "hold",
        "confidence": 0.0,
        "confidence_raw": 0.0,
        "stale_penalty": 0.0,
        "reasoning": f"Hold forced: {reason}",
        "accuracy_review": "N/A",
    }
```

---

## 12. Module 8 — `broker.py`

**Responsibility:** Executes and manages orders on Alpaca Paper Trading. Position sizing. Market hours check. Never raises.

### 12.1 Interface

```python
class Broker:
    def is_market_open(self) -> bool: ...

    def compute_qty(self, price: float, cash: float, mode: str) -> int:
        """
        mode="normal":       max_pct = MAX_POSITION_PCT_NORMAL (10%)
        mode="conservative": max_pct = MAX_POSITION_PCT_CONSERVATIVE (5%)
        qty = int(cash * max_pct / price)
        Returns max(0, qty).
        """

    def place_order(self, ticker: str, side: str, qty: int) -> dict:
        """
        Returns:
        {
            "ok": bool,
            "order_id": str | None,
            "status": str,
            "reason": str | None,
        }
        Never raises. Catches all APIError and Exception.
        Classifies errors: market_closed | insufficient_funds | rate_limited | broker_unavailable | api_error_N
        """

    def get_open_orders(self) -> list[dict]:
        """Return list of open orders. Empty list on failure."""

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders. Return True on success. Used during graceful stop."""
```

### 12.2 Implementation notes

- `ALPACA_PAPER = True` is always passed to `TradingClient`. Never make it configurable.
- Check `is_market_open()` before every `place_order()` call.
- Log every rejected order to `error_log.jsonl` via `journal.log_error()`.

---

## 13. Module 9 — `session.py`

**Responsibility:** Manages session lifecycle. Detects previous sessions on startup. Generates resoconto (summary) on terminal. Handles resume vs new session. Persists session state to `session.json`.

### 13.1 Session state schema (`data/session.json`)

```json
{
    "session_id": "uuid4",
    "started_at": "ISO8601",
    "last_active_at": "ISO8601",
    "status": "active | paused | completed",
    "cycle": 123,
    "active_prompt": "full prompt text",
    "initial_prompt": "original prompt text — for fallback",
    "active_strategy_id": "buffett_value | null",
    "portfolio_snapshot": {
        "cash": 92000.0,
        "portfolio_value": 95000.0,
        "pnl_pct": -0.05,
        "positions": {}
    },
    "behavior_change_count": 0
}
```

### 13.2 Interface

```python
class SessionManager:
    def detect_previous_session(self) -> dict | None:
        """Read session.json. Return session dict if status == 'active' or 'paused', else None."""

    def print_resoconto(self, session: dict) -> None:
        """
        Print to terminal (using rich) a formatted summary:
        ╔══════════════════════════════════════╗
        ║   SESSIONE PRECEDENTE RILEVATA       ║
        ╠══════════════════════════════════════╣
        ║ ID:           {session_id[:8]}...    ║
        ║ Avviata:      {started_at}           ║
        ║ Cicli:        {cycles}               ║
        ║ Ordini:       {orders_placed}        ║
        ║ P&L:          {pnl_pct:+.2%}         ║
        ║ Decisioni autonome: {autonomous}     ║
        ║ Errori loggati:     {errors}         ║
        ╚══════════════════════════════════════╝
        Vuoi riprendere questa sessione? [s/N]:
        """

    def ask_resume_or_new(self) -> str:
        """Interactive prompt. Returns 'resume' or 'new'."""

    def resume(self, session: dict) -> dict:
        """Load and return the session. Update status to 'active'."""

    def create_new(self, prompt: str) -> dict:
        """Create new session.json with new UUID, prompt, cycle=0."""

    def save(self, session: dict) -> None:
        """Write session.json atomically (write to .tmp then rename)."""

    def mark_paused(self, session: dict) -> None:
        """Set status='paused', save."""

    def mark_completed(self, session: dict) -> None:
        """Set status='completed', save."""
```

### 13.3 Tests — `tests/test_session.py`

```python
def test_detect_previous_session_returns_none_if_no_file():
    """Returns None when session.json does not exist."""

def test_create_new_generates_unique_session_id():
    """Two consecutive create_new() calls return different session_ids."""

def test_save_is_atomic():
    """If save() is interrupted mid-write, old session.json is not corrupted."""

def test_resume_sets_status_active():
    """After resume(), session['status'] == 'active'."""
```

---

## 14. Module 10 — `behavior.py`

**Responsibility:** Manages the active user prompt and behaviour changes. Handles graceful stop of in-flight tasks. Manages fallback to initial prompt on T_behavior timeout.

### 14.1 Interface

```python
class BehaviorManager:
    def __init__(self, session: dict, adaptive_timeout: AdaptiveTimeout): ...

    @property
    def active_prompt(self) -> str: ...

    @property
    def initial_prompt(self) -> str: ...

    def request_change(self, new_prompt: str) -> bool:
        """
        Initiate a behaviour change:
        1. Set _pending_prompt = new_prompt.
        2. Set _change_requested = True (signals loop to graceful stop).
        3. Return True if change accepted, False if another change is already pending.
        """

    def apply_change(
        self,
        memory_manager: MemoryManager,
        imitative_layer: ImiativeLayer,
    ) -> bool:
        """
        Execute the behaviour change within T_behavior timeout:
        1. memory_manager.reset_all()
        2. imitative_layer.reload()
        3. Update active_prompt to _pending_prompt.
        4. Save snapshot to session.json.
        5. If operation exceeds T_behavior → revert to initial_prompt, log fallback.
        Returns True if applied, False if fallback triggered.
        """

    def revert_to_initial(self) -> None:
        """Restore active_prompt = initial_prompt. Log to error_log."""

    @property
    def change_requested(self) -> bool:
        """True if a pending behaviour change is waiting to be applied."""

    def clear_change_request(self) -> None:
        """Reset _change_requested and _pending_prompt."""

    def increment_change_count(self, session: dict) -> None:
        """Increment session['behavior_change_count'] and save."""
```

### 14.2 Tests — `tests/test_behavior.py`

```python
def test_apply_change_reverts_on_timeout():
    """If T_behavior elapses during apply_change, active_prompt == initial_prompt."""

def test_request_change_rejects_while_pending():
    """Second request_change while one is pending returns False."""

def test_change_count_increments():
    """After successful apply_change, session['behavior_change_count'] == 1."""

def test_revert_logs_error():
    """revert_to_initial() writes an entry to error_log.jsonl."""
```

---

## 15. Module 11 — `loop.py`

**Responsibility:** The main autonomous loop. Implements the exact execution flow in Section 17. Runs until stopped. Catches all exceptions at the cycle level — never crashes.

### 15.1 Interface

```python
class AgentLoop:
    def __init__(
        self,
        session: dict,
        adaptive_timeout: AdaptiveTimeout,
        tool_executor: ToolExecutor,
        memory_manager: MemoryManager,
        imitative_layer: ImiativeLayer,
        reasoner: Reasoner,
        broker: Broker,
        behavior_manager: BehaviorManager,
        session_manager: SessionManager,
        dashboard,   # Dashboard instance from Module 12
    ): ...

    def start(self) -> None:
        """Run until KeyboardInterrupt or self.stop() called."""

    def stop(self) -> None:
        """Set _running = False. Current cycle completes before stopping."""

    def _run_cycle(self) -> None:
        """One full cycle. See Section 17 for exact step order."""

    def _handle_behavior_change(self) -> None:
        """Graceful stop + apply_change + reset memory."""

    def _graceful_stop_tasks(self) -> None:
        """Cancel all pending orders via broker.cancel_all_orders(). Wait for confirmation."""
```

### 15.2 Cycle step order (implement exactly)

```
CYCLE N
│
├─ [if behavior_manager.change_requested]
│       _handle_behavior_change()   ← graceful stop, apply or fallback, reset memory
│
├─ [for each ticker in TICKERS, skip if blacklisted]
│   │
│   ├─ 1. OUTCOME UPDATE
│   │      price_result = tool_executor.get_price(ticker)
│   │      if price_result.ok: journal.outcome_update(ticker, price, session_id)
│   │
│   ├─ 2+3. OBSERVE + SENTIMENT (concurrent.futures.ThreadPoolExecutor)
│   │        fut_bars      = pool.submit(tool_executor.get_bars, ticker, 5)
│   │        fut_news      = pool.submit(tool_executor.get_news, ticker)
│   │        fut_portfolio = pool.submit(tool_executor.get_portfolio)
│   │        [after news] sentiment = sentiment.analyse(ticker, articles, active_prompt, t_behavior)
│   │
│   ├─ 4. PORTFOLIO HEALTH CHECK
│   │      pnl_pct from portfolio_result
│   │      mode = "conservative" if pnl_pct < -DRAWDOWN_THRESHOLD else "normal"
│   │
│   ├─ 5. MEMORY CONTEXT
│   │      memory_context = memory_manager.build_context(ticker)
│   │      imitative_hints = imitative_layer.build_hints(active_prompt, ticker)
│   │
│   ├─ 6. THINK (Gemma4:12b)
│   │      decision = reasoner.decide(ticker, memory_context, price, price_timestamp,
│   │                                  ma5, trend, sentiment_score, sentiment_label,
│   │                                  imitative_hints, active_prompt,
│   │                                  cash, positions, mode,
│   │                                  stale, staleness_seconds, t_behavior)
│   │
│   ├─ 7. ACT
│   │      confidence_threshold = CONFIDENCE_THRESHOLD_CONSERVATIVE if mode=="conservative"
│   │                             else CONFIDENCE_THRESHOLD_NORMAL
│   │      if action in (buy, sell) AND confidence >= confidence_threshold:
│   │          qty = broker.compute_qty(price, cash, mode)
│   │          order_result = broker.place_order(ticker, action, qty)
│   │          decision_source = "agent"
│   │      else:
│   │          action = "hold"
│   │          order_result = {"ok": False, "order_id": None}
│   │
│   ├─ 8. RECORD
│   │      entry = journal.build_entry(...)    ← include all schema fields
│   │      journal.write_entry(entry)
│   │      memory_manager.update(entry)
│   │
│   └─ 9. UPDATE DASHBOARD
│          dashboard.update(ticker, entry, portfolio_result.data, t_wait, t_behavior)
│
├─ 10. ADAPTIVE TIMEOUT UPDATE
│       adaptive_timeout.calibrate()   ← runs every 5 cycles, not every cycle
│
├─ 11. NEWS SENTIMENT VETO CHECK
│       if sentiment < -0.7 AND ticker in positions AND action == "hold":
│           veto_triggered = True
│
└─ 12. WAIT with user input window
        if veto_triggered:
            wait_seconds = 2
        else:
            wait_seconds = adaptive_timeout.t_wait()
        result = dashboard.wait_for_user_input(wait_seconds)
        # result: {"source": "confirmed"|"override"|"behavior_change"|"timeout", "data": ...}
        if result["source"] == "behavior_change":
            behavior_manager.request_change(result["data"]["new_prompt"])
        elif result["source"] == "override":
            # apply user override: re-run ACT with user's decision
            # log decision_source = "user_override"
        elif result["source"] == "timeout":
            # log decision_source = "autonomous_timeout"
        session_manager.save(session)
```

---

## 16. Module 12 — `ui/dashboard.py`

**Responsibility:** Rich terminal UI. Shows portfolio state, journal tail, active proposal, countdown timer. Captures user input during WAIT. Uses `rich` library (not `textual`) for maximum stability.

### 16.1 Layout

```
╔══════════════════════════════════════════════════════════════════════╗
║  BIP Trading Agent  │  Sessione: {id[:8]}  │  Ciclo: {N}  │  {ts}  ║
╠══════════════════════════════════════════════════════════════════════╣
║  PORTFOLIO                           ║  TIMEOUT                      ║
║  Cash:     $87,420.00               ║  T_wait:     {N}s              ║
║  Valore:   $95,200.00               ║  T_behavior: {N}s              ║
║  P&L:      -4.80% ↓ [CONSERVATIVE]  ║  API lag:    {N}ms            ║
╠══════════════════════════════════════════════════════════════════════╣
║  PROPOSTA AGENTE                                                      ║
║  AAPL → HOLD (conf: 0.42, motivo: prezzo sotto MA, sentiment -0.31) ║
║  TSLA → BUY  (conf: 0.71, motivo: breakout, sentiment +0.65)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  JOURNAL (ultimi 5)                                                   ║
║  [11:04] AAPL ↑ BUY   0.74  +0.81  outcome:+1.2%                   ║
║  [11:06] TSLA → HOLD  0.38  -0.61  outcome:correct                  ║
║  [11:08] AAPL → HOLD  0.52  [STALE] data unavailable                ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⏳ {countdown}s │ [INVIO] conferma │ [m] modifica │ [c] cambia     ║
║  > _                                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 16.2 Interface

```python
class Dashboard:
    def __init__(self): ...

    def update(
        self,
        ticker: str,
        entry: dict,
        portfolio: dict,
        t_wait: int,
        t_behavior: int,
    ) -> None:
        """Refresh the display with latest data. Thread-safe."""

    def wait_for_user_input(self, timeout_seconds: int) -> dict:
        """
        Show countdown. Accept user input for timeout_seconds.
        Returns one of:
        {"source": "timeout",          "data": {}}
        {"source": "confirmed",        "data": {}}
        {"source": "override",         "data": {"action": str, "ticker": str, "qty": int}}
        {"source": "behavior_change",  "data": {"new_prompt": str}}

        User commands during wait:
          INVIO (empty)  → confirmed
          m              → prompt for manual override (ticker, action, qty)
          c              → prompt for new behaviour prompt
          Ctrl+C         → trigger graceful shutdown
        """

    def print_resoconto(self, summary: dict) -> None:
        """Print session summary in rich format. Called on startup (resume) and shutdown."""

    def print_shutdown_message(self) -> None:
        """Print shutdown message with final P&L and session stats."""
```

### 16.3 Implementation notes

- Use `rich.live.Live` with `refresh_per_second=1` for the countdown.
- Use `rich.table.Table`, `rich.panel.Panel`, `rich.text.Text` for layout.
- Countdown bar uses `rich.progress.Progress`.
- User input is captured via `input()` call inside a separate thread with `concurrent.futures.ThreadPoolExecutor(max_workers=1)`. The main thread waits on the future with `timeout=timeout_seconds`.
- Do **not** use `textual` — it requires an async event loop incompatible with the synchronous loop architecture.
- Color coding: green = positive P&L / buy, red = negative P&L / sell, yellow = hold / stale, blue = autonomous decision, magenta = user override.

---

## 17. `main.py`

```python
"""
BIP Hackathon 2026 — Trading Agent
Entry point.

Usage:
    uv run python main.py

The agent initialises, asks for session resume or new prompt,
then runs autonomously. Press Ctrl+C to stop gracefully.
"""
import logging
import sys
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler("agent.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    from src.agent import config
    from src.agent.adaptive_timeout import AdaptiveTimeout
    from src.agent.tool_executor import ToolExecutor
    from src.agent.journal import read_session_summary
    from src.agent.memory_manager import MemoryManager
    from src.agent.imitative_layer import ImiativeLayer
    from src.agent.sentiment import analyse as sentiment_analyse
    from src.agent.reasoner import Reasoner
    from src.agent.broker import Broker
    from src.agent.session import SessionManager
    from src.agent.behavior import BehaviorManager
    from src.agent.loop import AgentLoop
    from ui.dashboard import Dashboard

    dashboard = Dashboard()

    # ── Step 1: Detect previous session ─────────────────────────────
    session_mgr = SessionManager()
    previous = session_mgr.detect_previous_session()

    if previous:
        summary = read_session_summary(previous["session_id"])
        dashboard.print_resoconto(summary)
        choice = session_mgr.ask_resume_or_new()
        if choice == "resume":
            session = session_mgr.resume(previous)
            prompt = session["active_prompt"]
        else:
            prompt = input("\nInserisci il comportamento del nuovo agente: ").strip()
            session = session_mgr.create_new(prompt)
    else:
        prompt = input("Inserisci il comportamento dell'agente (es. 'orientato a scelte green'): ").strip()
        session = session_mgr.create_new(prompt)

    # ── Step 2: Calibrate adaptive timeout ──────────────────────────
    adaptive_timeout = AdaptiveTimeout()
    logger.info("Calibrating adaptive timeout...")
    adaptive_timeout.calibrate()
    logger.info(f"Timeout calibration: {adaptive_timeout.summary()}")

    # ── Step 3: Initialise modules ──────────────────────────────────
    tool_executor    = ToolExecutor(adaptive_timeout)
    memory_manager   = MemoryManager()
    imitative_layer  = ImiativeLayer()
    reasoner         = Reasoner()
    broker           = Broker()
    behavior_manager = BehaviorManager(session, adaptive_timeout)

    # ── Step 4: Start loop ───────────────────────────────────────────
    loop = AgentLoop(
        session=session,
        adaptive_timeout=adaptive_timeout,
        tool_executor=tool_executor,
        memory_manager=memory_manager,
        imitative_layer=imitative_layer,
        reasoner=reasoner,
        broker=broker,
        behavior_manager=behavior_manager,
        session_manager=session_mgr,
        dashboard=dashboard,
    )

    try:
        loop.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    finally:
        loop.stop()
        broker.cancel_all_orders()
        session_mgr.mark_paused(session)
        summary = read_session_summary(session["session_id"])
        dashboard.print_resoconto(summary)
        dashboard.print_shutdown_message()
        logger.info("Agent stopped. Session saved.")


if __name__ == "__main__":
    main()
```

---

## 18. Infrastructure Tests (`tests/test_connections.py`)

Run before every demo: `uv run python tests/test_connections.py`

Implement the following tests as standalone functions. Each prints `✓ PASS` or `✗ FAIL: reason`.

```python
# ── Alpaca ────────────────────────────────────────────────────────
def test_alpaca_account():
    """Account accessible, cash > 0."""

def test_alpaca_clock():
    """Clock endpoint returns valid is_open boolean."""

def test_alpaca_price_aapl():
    """get_price('AAPL') returns price > 0 with valid ISO timestamp."""

def test_alpaca_news_aapl():
    """News endpoint returns at least 0 articles (market may be closed)."""

def test_alpaca_place_order_dry_run():
    """
    Do NOT actually place an order.
    Instead, verify that the TradingClient initialises correctly
    and is_market_open() returns a boolean without raising.
    """

# ── Ollama ────────────────────────────────────────────────────────
def test_ollama_reachable():
    """GET /api/tags returns 200 and lists models."""

def test_ollama_sentiment_model_loaded():
    """qwen2.5:3b appears in ollama list."""

def test_ollama_reasoning_model_loaded():
    """gemma4:12b appears in ollama list."""

def test_ollama_sentiment_returns_valid_json():
    """sentiment.analyse returns score in [-1, 1] and valid label."""

def test_ollama_reasoning_returns_valid_decision():
    """reasoner.decide returns action in [buy, sell, hold] and confidence in [0, 1]."""

# ── Adaptive Timeout ──────────────────────────────────────────────
def test_adaptive_timeout_calibrate():
    """calibrate() completes and returns non-zero t_wait and t_behavior."""

# ── Journal ───────────────────────────────────────────────────────
def test_journal_write_read():
    """write_entry + read_last_n(1) roundtrip in temp file."""

def test_journal_outcome_update():
    """outcome_update fills price_after and outcome_pct correctly."""

def test_error_log_write():
    """log_error appends to error_log.jsonl without raising."""

# ── Session ───────────────────────────────────────────────────────
def test_session_create_and_save():
    """create_new + save + detect returns same session_id."""

# ── ToolExecutor ──────────────────────────────────────────────────
def test_tool_executor_graceful_on_invalid_ticker():
    """get_price('INVALIDXXX') returns ToolResult without raising."""

# ── Imitative Layer ───────────────────────────────────────────────
def test_imitative_dataset_exists():
    """IMITATIVE_DATASET_PATH exists and parses as valid JSON."""

def test_imitative_filter_green_prompt():
    """filter_for_prompt('green investing') returns green_esg strategy."""

def test_imitative_filter_defense_prompt():
    """filter_for_prompt('arms defense') returns defense_sector strategy."""
```

Run all tests and print a summary table:

```
══════════════════════════════════════════
  BIP Hackathon 2026 — Infrastructure Tests
══════════════════════════════════════════
  ✓  Alpaca account
  ✓  Alpaca clock
  ✓  Alpaca price AAPL
  ✓  Alpaca news AAPL
  ✓  Alpaca place_order dry run
  ✓  Ollama reachable
  ✓  qwen2.5:3b loaded
  ✓  gemma4:12b loaded
  ✓  Sentiment valid JSON
  ✓  Reasoning valid decision
  ✓  Adaptive timeout calibrate
  ✓  Journal write/read
  ✓  Journal outcome update
  ✓  Error log write
  ✓  Session create and save
  ✓  ToolExecutor graceful failure
  ✓  Imitative dataset exists
  ✓  Imitative filter green
  ✓  Imitative filter defense
══════════════════════════════════════════
  Results: 19 passed, 0 failed
══════════════════════════════════════════
```

---

## 19. Hard Rules — Never Violate

These are not guidelines. Violating them costs evaluation points directly.

**R1 — No hallucinated data.**
The text `"Source: static dataset — not from model memory."` must appear in every `imitative_hints` string. Every price passed to Gemma4 must include its `price_timestamp` from the ticker JSON object. The system prompt for Gemma4 must include the phrase "you only use the data provided in this prompt".

**R2 — Autonomous start.**
`main.py` asks for the initial prompt interactively, then starts the loop autonomously. No human intervention is needed to trigger the first cycle after the prompt is entered.

**R3 — ToolExecutor never raises.**
Every method in `ToolExecutor` catches all exceptions. Returns `ToolResult(ok=False, ...)` on failure. The loop never crashes because an API call failed.

**R4 — Journal always written.**
Every cycle, every ticker produces exactly one journal entry. Hold due to STALE, hold due to confidence, hold due to conservative mode — all produce entries. The journal entry for a rejected order is as important as a successful one.

**R5 — Confidence gate is code, not prompt.**
The comparison `confidence >= threshold` is Python code in `loop.py`, not a prompt instruction. The STALE penalty is applied in `reasoner.py` after the model responds. The model cannot bypass either check.

**R6 — Paper trading only.**
`ALPACA_PAPER = True` in `config.py` is a constant, never loaded from env. `TradingClient(..., paper=True)` is hardcoded in `broker.py`.

**R7 — Graceful stop before behaviour change.**
`broker.cancel_all_orders()` must be called and awaited before `behavior_manager.apply_change()`. No pending orders may remain when the behaviour changes.

**R8 — Atomic session writes.**
`session_manager.save()` must write to a `.tmp` file and rename atomically. Never write directly to `session.json` in place.

**R9 — keep_alive=30s on all Ollama calls.**
Both `reasoner.py` and `sentiment.py` must pass `keep_alive="30s"` to every `ollama.chat()` call. This prevents both models from being resident in RAM simultaneously on 16 GB hardware.

**R10 — Adaptive timeout, never fixed.**
The value passed to `dashboard.wait_for_user_input()` must always come from `adaptive_timeout.t_wait()`. The value passed to `reasoner.decide()` as `t_behavior` must always come from `adaptive_timeout.t_behavior()`. No hardcoded sleep values in the loop.

---

## 20. Implementation Order

Implement strictly in this order. Run tests after each step before proceeding.

```
1.  config.py                    → no tests, verify import works
2.  adaptive_timeout.py          → run test_adaptive_timeout.py
3.  journal.py                   → run test_journal.py
4.  tool_executor.py             → run test_tool_executor.py
5.  memory_manager.py            → run test_memory_manager.py
6.  imitative_layer.py           → create imitative_dataset.json, test manually
7.  sentiment.py                 → test via test_connections.py (sentiment tests)
8.  reasoner.py                  → test via test_connections.py (reasoning tests)
9.  broker.py                    → test via test_connections.py (alpaca tests)
10. session.py                   → run test_session.py
11. behavior.py                  → run test_behavior.py
12. ui/dashboard.py              → test manually (run main.py in dry mode)
13. loop.py                      → integrate all modules, run full cycle test
14. main.py                      → end-to-end test: start → 3 cycles → stop → resume
15. tests/test_connections.py    → all 19 tests must pass before demo
```

---

## 21. README.md

Create a `README.md` with:

```markdown
# BIP Hackathon 2026 — Autonomous Trading Agent

## Quick start

# 1. Install dependencies
uv sync

# 2. Pull Ollama models
ollama pull gemma4:12b
ollama pull qwen2.5:3b

# 3. Copy and fill env
cp .env.example .env
# edit .env with your Alpaca Paper Trading keys

# 4. Run infrastructure tests
uv run python tests/test_connections.py

# 5. Start the agent
uv run python main.py

## Architecture

Dual-model pipeline (Gemma4:12b + qwen2.5:3b), HOT/WARM/COLD memory,
adaptive timeout, imitative layer, persistent JSONL journal, rich TUI.
See ARCHITECTURE.md for full specification.

## Stack

- Broker: Alpaca Paper Trading (paper=True, always)
- Reasoning LLM: Gemma4:12b via Ollama
- Sentiment LLM: qwen2.5:3b via Ollama
- Journal: JSONL append-only, outcome tracking
- UI: rich terminal dashboard
- Python 3.12+, uv package manager
```

---

*End of specification. Implement top-to-bottom. Test after each module. All 19 infrastructure tests must pass before the demo.*

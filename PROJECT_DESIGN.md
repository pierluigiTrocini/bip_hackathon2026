# BIP Hackathon 2026 — Trading Agent: Full Design Document

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Infrastructure & Environment](#3-infrastructure--environment)
4. [Module Reference](#4-module-reference)
   - 4.1 [Entry Point — `main.py`](#41-entry-point--mainpy)
   - 4.2 [Configuration — `config.py`](#42-configuration--configpy)
   - 4.3 [Agent Loop — `loop.py`](#43-agent-loop--looppy)
   - 4.4 [Reasoner — `reasoner.py`](#44-reasoner--reasonerpy)
   - 4.5 [Broker — `broker.py`](#45-broker--brokerpy)
   - 4.6 [Tool Executor — `tool_executor.py`](#46-tool-executor--tool_executorpy)
   - 4.7 [Session Manager — `session.py`](#47-session-manager--sessionpy)
   - 4.8 [Behavior Manager — `behavior.py`](#48-behavior-manager--behaviorpy)
   - 4.9 [Behavior Questionnaire — `behavior_questionnaire.py`](#49-behavior-questionnaire--behavior_questionnairepy)
   - 4.10 [Discovery Agent — `discovery.py`](#410-discovery-agent--discoverypy)
   - 4.11 [News Log — `news_log.py`](#411-news-log--news_logpy)
   - 4.12 [Memory Manager — `memory_manager.py`](#412-memory-manager--memory_managerpy)
   - 4.13 [Adaptive Timeout — `adaptive_timeout.py`](#413-adaptive-timeout--adaptive_timeoutpy)
   - 4.14 [Sentiment — `sentiment.py`](#414-sentiment--sentimentpy)
   - 4.15 [Imitative Layer — `imitative_layer.py`](#415-imitative-layer--imitative_layerpy)
   - 4.16 [Strategy Library — `strategy_library.py`](#416-strategy-library--strategy_librarypy)
   - 4.17 [Correlation Engine — `correlation_engine.py`](#417-correlation-engine--correlation_enginepy)
   - 4.18 [Market Disruptor — `disruptor.py`](#418-market-disruptor--disruptorpy)
   - 4.19 [Position Manager — `position_manager.py`](#419-position-manager--position_managerpy)
   - 4.20 [Technical Analyser — `technical_analyser.py`](#420-technical-analyser--technical_analyserpy)
   - 4.21 [User Preference Engine — `user_preference_engine.py`](#421-user-preference-engine--user_preference_enginepy)
   - 4.22 [Journal — `journal.py`](#422-journal--journalpy)
   - 4.23 [LLM Stream — `llm_stream.py`](#423-llm-stream--llm_streampy)
   - 4.24 [Dashboard — `ui/dashboard.py`](#424-dashboard--uidashboardpy)
5. [Feature Deep-Dives](#5-feature-deep-dives)
   - F1: [Unified News Context Layer](#f1-unified-news-context-layer)
   - F2: [Adaptive Position Manager](#f2-adaptive-position-manager)
   - F3: [Technical Indicators](#f3-technical-indicators)
   - F4: [User Preference Engine](#f4-user-preference-engine)
6. [Data Flows](#6-data-flows)
7. [Persistence & Atomicity](#7-persistence--atomicity)
8. [LLM Integration](#8-llm-integration)
9. [Security Constraints](#9-security-constraints)
10. [Test Suite](#10-test-suite)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Project Overview

**BIP Hackathon 2026 — Trading Agent** is an autonomous Python trading system that:

- Runs continuously in a per-ticker cycle loop
- Uses two local Ollama LLMs for reasoning and sentiment/keyword analysis
- Executes paper trades via the Alpaca Markets API (always paper — hardcoded)
- Maintains rich session state, multi-tier memory, and a live Rich TUI
- Adapts its behaviour dynamically based on user prompts, market conditions, and inferred user preferences

The system integrates five major features:
- **F1** — Unified news context (current + historical + disruptor) passed to LLM
- **F2** — Adaptive stop-loss / take-profit per open position
- **F3** — RSI(14) + Bollinger Bands(20) technical signals
- **F4** — User preference engine (explicit + implicit + emotional tone)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  main.py  (entry point)                                                      │
│                                                                               │
│  SessionManager ──→ detect / resume / create session.json                   │
│  AdaptiveTimeout ──→ calibrate API + Ollama latency                         │
│  DiscoveryAgent ──→ NewsAPI + Polygon → LLM → validate on Alpaca            │
│  AgentLoop ──→ per-ticker cycle (see §4.3)                                  │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
              ┌────────────▼────────────────────────────────────────┐
              │  AgentLoop._run_one_cycle()                         │
              │                                                      │
              │  [1] Portfolio fetch (ToolExecutor)                 │
              │  [2] Bars + news fetch                              │
              │  [3] Sentiment (qwen2.5:3b)                        │
              │  [3b] News keywords + relevance scoring             │
              │  [4] Technical signals (RSI, Bollinger Bands)       │
              │  [5] Build unified prompt context                   │
              │       ├─ F1: news_context (current + historical     │
              │       │        + disruptor)                         │
              │       ├─ F2: position_context (stop/take thresholds)│
              │       ├─ F3: technical_signals.prompt_section       │
              │       └─ F4: user_preferences_section               │
              │  [6] LLM decision (gemma4:12b)                      │
              │  [6b] F4 conflict detection + minimum modification  │
              │  [7] Order placement (Broker → Alpaca paper)        │
              │  [8] Journal write + memory update                  │
              │  [9] Dashboard update                               │
              │  [10] Adaptive timeout recalibration (every 5)      │
              └─────────────────────────────────────────────────────┘
```

**Dual-LLM design:**

| Model | Role | Timeout source |
|---|---|---|
| `gemma4:12b` | Trading decisions, strategy reasoning, discovery selection | `T_BEHAVIOR` |
| `qwen2.5:3b` | Sentiment scoring, keyword extraction, questionnaire Q&A | `T_BEHAVIOR` |

**Background threads:**

| Thread | Interval | Purpose |
|---|---|---|
| `MarketDisruptor` | 60 s | Fetch breaking news; push to `disruptor_news.jsonl` |
| `AdaptiveTimeout.calibrate()` | Every 5 cycles | Re-measure API + Ollama latency |

---

## 3. Infrastructure & Environment

### Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally at `http://localhost:11434`
- Models pulled: `gemma4:12b` (reasoning), `qwen2.5:3b` (sentiment)
- Alpaca paper trading account (API key + secret in `.env`)
- Optional: NewsAPI key, Polygon API key (for discovery enrichment)

### `.env` keys

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
NEWS_API_KEY=...          # optional
POLYGON_API_KEY=...       # optional
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REASONING_MODEL=gemma4:12b
OLLAMA_SENTIMENT_MODEL=qwen2.5:3b
```

### Data files (auto-created under `data/`)

| File | Purpose |
|---|---|
| `data/session.json` | Active session state (atomic write) |
| `data/journal.jsonl` | All cycle entries (append-only JSONL) |
| `data/error_log.jsonl` | Error log entries |
| `data/news_log.jsonl` | All fetched articles with keywords + relevance |
| `data/disruptor_news.jsonl` | Breaking news from background disruptor thread |
| `data/strategies/imitative_dataset.json` | Imitative strategy investor profiles |

### Run

```bash
uv run python main.py        # silent loop
uv run python main.py -v     # verbose (LLM output visible)
```

---

## 4. Module Reference

### 4.1 Entry Point — `main.py`

**Responsibilities:** Orchestrates startup, discovery, and loop lifecycle.

**Startup sequence:**

1. Parse `--verbose` flag → set `llm_stream.LOOP_VERBOSE`
2. Detect / resume / create session via `SessionManager`
3. Calibrate `AdaptiveTimeout` (3 API pings + 3 Ollama pings)
4. Instantiate all modules
5. **F4**: extract user preferences from initial prompt
6. **Discovery phase** (or resume from saved tickers): fetch news, rank candidates, validate on Alpaca, confirm with user
7. Start `MarketDisruptor` background thread
8. Instantiate and start `AgentLoop`
9. On `KeyboardInterrupt`: cancel orders, mark session paused, print summary

**Security invariant:** `broker.py` always uses `paper=True` regardless of env config.

---

### 4.2 Configuration — `config.py`

Single source of truth for all numeric constants and paths. Loads from `.env` via `python-dotenv`. Every value has a sensible default. Key groups:

- **Alpaca:** `ALPACA_PAPER = True` (hardcoded, never changed)
- **Ollama:** model names and base URL
- **Agent behaviour:** confidence thresholds, position size limits, drawdown threshold
- **Adaptive timeout:** multipliers and min/max bounds for `T_WAIT` and `T_BEHAVIOR`
- **Memory:** hot window size, warm compaction trigger
- **NCCI:** correlation engine rebuild interval, display threshold, keyword weight, half-life
- **F1–F4:** all feature-specific knobs (see §11)

---

### 4.3 Agent Loop — `loop.py`

**Class:** `AgentLoop`

**State:**
- `_session`, `_cycle`, `_tickers`, `_recent_metrics` (last 20 cycles for auto-switch)
- `_pm: PositionManager | None` (F2)
- `_upe: UserPreferenceEngine | None` (F4)
- `_in_wait_phase`, `_interaction_running` (mutex flags for stdin listener)

**`start()`:** Spawns a `_stdin_listener` daemon thread that captures user keystrokes in between cycles, then calls `_run_one_cycle()` in a loop.

**`_run_one_cycle()`** (core algorithm, per ticker):

```
[1]  fetch portfolio (cash, value, positions, P&L)
[2]  fetch bars (20 closes, MA5, trend) + fetch news articles
[3]  run sentiment (qwen2.5:3b) → score + label
[3b] persist articles to news_log with keywords + relevance (async)
[4]  F3: technical_analyser.analyse(closes, price, ...) → TechnicalSignals
[5]  F1: build_news_context_for_prompt(ticker, cycle, ...)
[5b] F2: update_price, update_thresholds, pre-check stop-loss
[5c] F4: build_prompt_section
[6]  Reasoner.decide(... news_context, position_context, technical_signals,
                     user_preferences_section ...)
[6b] F4: check_conflict + apply_minimum_modification (if conflict: print panel)
[7]  If BUY/SELL + confidence ≥ threshold + market open:
       compute_qty (F4 effective position pct applied)
       place_order via Broker
       F2: on_new_position / on_position_closed
[7b] mark news decision + print decision panel
[8]  journal.write_entry + journal.write_news_entries
     memory_manager.update(entry)
[9]  dashboard.update
[10] NEWS VETO: if sentiment < -0.7 and ticker in positions → set veto_triggered
```

**Post-cycle:**
- Every 5 cycles: `_check_auto_switch()` (strategy auto-switch recommendation)
- `wait_for_user_input(wait_seconds)` — interactive live countdown
- Handle user input: prompt append/change, questionnaire, strategy select, manual override

**`_run_interaction()`** handles all interactive flows:
- `prompt_append`: ask for instruction, append to active_prompt, call `apply_change()`
- `prompt_change`: full prompt replacement
- `questionnaire`: `behavior_questionnaire.generate_questions()` → answers → `synthesize_prompt()`
- `strategy_select`: show strategy menu → call `apply_change()`

**`_handle_behavior_change()`**: calls `behavior_manager.apply_change()` passing `preference_engine` for F4 re-extraction.

---

### 4.4 Reasoner — `reasoner.py`

**Class:** `Reasoner`

**LLM:** `gemma4:12b` (reasoning model)

**`decide()` signature:**

```python
def decide(
    self,
    ticker, memory_context, price, price_timestamp, ma5, trend,
    sentiment_score, sentiment_label, mode, cash, portfolio_value,
    positions, strategy_id="contrarian",
    correlation_section="",
    news_context="",           # F1: unified news block
    position_context="",       # F2: stop-loss / take-profit / P&L
    technical_signals="",      # F3: RSI + Bollinger Bands
    user_preferences_section="", # F4: preferences + derived params
    take_profit_hint="",       # legacy
    historical_news_context="", # legacy
    disruptor_context="",      # legacy
) -> dict
```

**Prompt structure** (7 sections):
1. Strategy system prompt from `strategy_library`
2. Action signal from `_STRATEGY_SIGNALS` mapping `(sentiment_label, trend)` → directive
3. Behaviour description (current active prompt)
4. Market data: price, MA5, trend, cash, portfolio value, positions
5. News block (F1 unified context)
6. Position block (F2 adaptive thresholds)
7. Technical block (F3 RSI + BB)
8. Portfolio context
9. User preferences (F4)
10. Memory context (HOT/WARM/COLD tier)
11. Correlation section (NCCI)

**§8.3 instructions** appended at end of prompt:
- Weight technical signals appropriately
- Override action to SELL if position context shows stop-loss breach
- Respect stated user preferences
- Weight breaking news (HIGH PRIORITY) above historical
- Caption must be a single English sentence (max 160 chars) referencing a concrete data point

**Returns:**
```python
{
  "action": "buy"|"sell"|"hold",
  "confidence": float,          # 0.0–1.0 (after stale penalty)
  "confidence_raw": float,
  "stale_penalty": float,
  "reasoning": str,
  "accuracy_review": str,
  "caption": str,               # short English explanation for TUI
}
```

**`_safe_hold(reason)`**: fallback for LLM failure — returns hold with 0 confidence and `"Forced decision: {reason}"` caption.

---

### 4.5 Broker — `broker.py`

**Class:** `Broker`

**Security:** `ALPACA_PAPER = True` hardcoded — always connects to paper trading.

**`compute_qty(price, cash, mode, action, ticker, positions, portfolio_value, effective_position_pct)`:**
- Computes affordable shares given `effective_position_pct` of portfolio
- BUY: caps at available cash fraction; also enforces per-ticker concentration cap
- SELL: returns full held quantity
- Returns 0 if unaffordable

**`place_order(ticker, action, qty)`:** Submits market order via Alpaca REST API. Returns `{"ok": bool, "order_id": str|None, "reason": str}`.

**`cancel_all_orders()`:** Called on shutdown to cancel any open orders.

---

### 4.6 Tool Executor — `tool_executor.py`

**Class:** `ToolExecutor`

Wraps all external API calls with `Result(ok, data, error)` return type — never raises.

**Methods:**
- `get_portfolio()` → `{cash, portfolio_value, pnl_pct, positions: {ticker: {qty, avg_entry_price, market_value}}}`
- `get_bars(ticker, limit)` → list of OHLCV bars
- `get_news(ticker, limit)` → list of article dicts
- `get_market_clock()` → `{is_open: bool, next_open: str, next_close: str}`
- `resolve_ticker(ticker)` → `(resolved_ticker | None, remapped: bool)` — validates via Alpaca asset lookup

**Blacklist:** tickers that consistently fail are added to `_blacklisted` set and skipped in the loop.

---

### 4.7 Session Manager — `session.py`

**Class:** `SessionManager`

Manages a single `data/session.json` file. Uses atomic writes (`.tmp` + `os.replace()`).

**Session dict schema:**

```json
{
  "session_id": "uuid",
  "started_at": "ISO",
  "last_active_at": "ISO",
  "status": "active|paused|completed",
  "cycle": 0,
  "active_prompt": "...",
  "initial_prompt": "...",
  "active_strategy_id": null,
  "tickers": ["AAPL", "TSLA"],
  "portfolio_snapshot": {...},
  "behavior_change_count": 0,
  // F2
  "user_stop_loss_pct": null,
  "user_take_profit_pct": null,
  // F4
  "pref_sectors": [],
  "pref_excluded_sectors": [],
  "pref_risk_level": "unspecified",
  "pref_ethics": [],
  "pref_time_horizon": "unspecified",
  "pref_emotion": "neutral",
  "pref_emotion_score": 0.0,
  "style_hold_rate": 0.5,
  "style_confirm_rate": 0.5,
  "style_override_count": 0,
  "style_reject_sl_count": 0,
  "style_inferred": "undetected",
  "derived_confidence_delta": 0.0,
  "derived_position_pct_delta": 0.0,
  "derived_mode_bias": "none",
  "wait_choices": [],
  "preference_conflicts": []
}
```

**`detect_previous_session()`:** Returns session dict if status is `active` or `paused`.

**`ask_resume_or_new()`:** Prints the previous session summary table and prompts `y/N`.

---

### 4.8 Behavior Manager — `behavior.py`

**Class:** `BehaviorManager`

**`apply_change(new_prompt, source, preference_engine=None)`:**
1. Sets `session["active_prompt"]` = `new_prompt`
2. Increments `session["behavior_change_count"]`
3. Saves session
4. If `preference_engine` is not None: calls `extract_from_prompt()` + `compute_derived_parameters()` (F4 re-extraction)

---

### 4.9 Behavior Questionnaire — `behavior_questionnaire.py`

Uses `qwen2.5:3b` to generate and synthesize the interactive Q&A questionnaire.

**`generate_questions(context, t_behavior)`:**
- Context: `active_prompt`, tickers, P&L, mode, recent decisions
- LLM generates 3 targeted English questions about risk, sectors, loss reaction, time horizon
- Falls back to 3 hardcoded English questions on LLM failure

**`synthesize_prompt(active_prompt, questions, answers, t_behavior)`:**
- Combines current prompt + Q&A pairs
- LLM writes a new 2–3 sentence English behaviour prompt
- Fallback: appends answers directly to existing prompt

---

### 4.10 Discovery Agent — `discovery.py`

**Class:** `DiscoveryAgent`

5-step pipeline executed before the main loop (or on new session):

1. **Keyword extraction** from user prompt (language-agnostic stop-word filter)
2. **Parallel fetch:** NewsAPI articles + Polygon sector-based tickers (concurrent futures)
3. **Company name extraction** via `qwen2.5:3b` → Polygon symbol lookup
4. **LLM candidate selection** via `gemma4:12b` (proposes 3–5 tickers from enriched context)
5. **Validation loop** (up to 3 rounds):
   - Validate each candidate via `ToolExecutor.resolve_ticker()` (Alpaca asset lookup)
   - Retry with LLM for alternatives when not enough valid tickers

**`_validate_candidates()`** returns a flat list of candidate dicts:
```python
{"ticker": str, "original_ticker": str, "reason": str, "confidence": float, "valid": bool}
```

**Ticker false-positives list:** ~50 common English words that look like tickers (A, IT, AI, US, ...) are excluded from auto-extraction.

---

### 4.11 News Log — `news_log.py`

Append-only JSONL persistence for all fetched articles. Never raises.

**`persist_articles(articles, ticker, cycle, session_id)`:**
- Calls `extract_keywords_and_relevance()` concurrently with the sentiment call
- Writes each article as a JSONL entry with: title, source, url, ticker, cycle, session_id, keywords, relevance_score, TTL

**`build_news_context_for_prompt(ticker, current_cycle, history_cycles, min_relevance_historical, max_articles)`** (F1):
- Reads `news_log.jsonl`, filters by ticker and cycle
- Compacts: current cycle (up to `max_articles`) + historical (remaining slots, score ≥ threshold, no title dedup)
- Returns formatted `=== NEWS CONTEXT (AAPL) ===` block
- Labels: `Current cycle:` + `Historical (last N cycles, score ≥ X):`

**`read_for_display(ticker, cycle, session_id, max_articles)`:**
- Returns recent articles for the TUI decision panel

**`mark_decision(ticker, cycle, session_id, decision)`:**
- Records the final action (buy/sell/hold/veto) on matching article entries

**`compact_news_log()`:**
- Triggered every `NEWS_LOG_COMPACT_EVERY` writes
- Removes articles older than their TTL or beyond `NEWS_LOG_MAX_PER_TICKER` per ticker

---

### 4.12 Memory Manager — `memory_manager.py`

**Class:** `MemoryManager`

Three-tier per-ticker memory: **HOT** (recent `HOT_WINDOW_SIZE` entries), **WARM** (last 15 compacted summaries), **COLD** (global cross-session summary).

**`update(entry)`:**
- Appends to HOT tier; triggers compaction when HOT exceeds `HOT_WINDOW_SIZE`
- On compaction: summarises HOT into WARM entry (confidence, action counts, P&L)
- On WARM overflow (`WARM_COMPACTION_TRIGGER`): merges WARM into COLD

**`get_context(ticker)`:**
- Returns formatted multi-tier memory string passed to `Reasoner.decide()` as `memory_context`

---

### 4.13 Adaptive Timeout — `adaptive_timeout.py`

**Class:** `AdaptiveTimeout`

Measures actual API and Ollama latency to compute safe timeouts.

**`calibrate()`:**
- Pings Alpaca 3 times, pings Ollama 3 times
- Sets `_api_avg_ms` and `_ollama_avg_ms`

**`t_wait()`:** `max(T_WAIT_MIN, min(T_WAIT_MAX, api_avg_ms / 1000 × T_WAIT_MULTIPLIER))`

**`t_behavior()`:** `max(T_BEHAVIOR_MIN, min(T_BEHAVIOR_MAX, ollama_avg_ms / 1000 × T_BEHAVIOR_MULTIPLIER))`

Recalibrates in a background thread every 5 cycles to adapt to changing infrastructure latency.

---

### 4.14 Sentiment — `sentiment.py`

Uses `qwen2.5:3b` to score news/text sentiment.

**`analyse(text, t_behavior)`:**
- Returns `{"score": float (-1 to +1), "label": str}`
- Labels: `very_negative`, `negative`, `neutral`, `positive`, `very_positive`
- Falls back to `{"score": 0.0, "label": "neutral"}` on any error

Called per-ticker in the agent loop step [3].

---

### 4.15 Imitative Layer — `imitative_layer.py`

**Class:** `ImiativeLayer`

Loads `data/strategies/imitative_dataset.json` (5 investor profiles: Buffett, Lynch, Simons, ESG, Defense).

Each profile contains historical behaviour patterns (sector weights, holding periods, signal thresholds). In the loop, the imitative layer selects the best-matching profile and returns it as `imitative_source` to be logged and passed to the reasoner for context.

---

### 4.16 Strategy Library — `strategy_library.py`

Defines 6 trading strategies, each as a dict with:

| Field | Purpose |
|---|---|
| `name` | Display name |
| `description` | One-line English description |
| `best_for` | Market conditions where strategy shines |
| `system_prompt` | Full LLM system prompt injected into `Reasoner.decide()` |

**Strategies:**

| ID | Name | Core thesis |
|---|---|---|
| `contrarian` | Contrarian | Buy extreme fear, sell extreme greed |
| `trend_following` | Trend Following | Ride the trend, exit on reversal |
| `momentum` | Momentum | Ride acceleration, exit before deceleration |
| `value` | Value | Buy below fair value, sell at mean reversion |
| `defensive` | Defensive | Capital preservation above all returns |
| `scalping` | Scalping | Small frequent profits, rapid in/out |

**`recommend_switch(current_id, hold_rate, pnl_trend, avg_sentiment, avg_trend)`:**

Auto-switch logic evaluated every 5 cycles:
- Portfolio declining > 3% → Defensive
- Avg sentiment ≥ 0.30 + uptrend → Momentum
- Avg sentiment ≤ -0.35 + down/flat → Contrarian
- Hold rate > 75% + uptrend → Trend Following

**`_STRATEGY_SIGNALS`:** Per-strategy `(sentiment_label, trend) → action directive` maps injected directly into the LLM prompt as an explicit DIRECTIVE.

---

### 4.17 Correlation Engine — `correlation_engine.py`

**Class:** `CorrelationEngine`

Computes **NCCI (News Co-occurrence Correlation Index)** between ticker pairs using Jaccard similarity on news keywords.

**Algorithm:**
1. Accumulates keyword sets per ticker per cycle from `news_log.py`
2. Every `NCCI_REBUILD_EVERY` cycles: rebuilds the full pairwise NCCI matrix
3. NCCI(A, B) = Jaccard(keywords_A, keywords_B) weighted by time decay (half-life `NCCI_HALF_LIFE_DAYS`)
4. Keywords with weight < `NCCI_KEYWORD_MIN_WEIGHT` are excluded

**`get_ncci(a, b)`:** Returns float 0.0–1.0 (0 = unrelated, 1 = identical keyword set).

**`add_ticker(ticker)`:** Registers ticker for tracking on next rebuild.

Results are printed as a colour-coded matrix in the TUI at end of each cycle.

---

### 4.18 Market Disruptor — `disruptor.py`

**Class:** `MarketDisruptor`

Background thread that monitors breaking news for active tickers.

**`start(tickers, session_id)`:**
- Spawns daemon thread, polls every 60 seconds
- Fetches news for each ticker via `ToolExecutor`
- Deduplicates by content hash (session-scoped: hash file cleared on start)
- Writes new articles to `data/disruptor_news.jsonl` (atomic)
- Articles persist across cycles within a session

**`get_articles(ticker, max_age_seconds)`:**
- Called in the agent loop to get recent disruptor articles
- Returns list of article dicts

**`stop()`:** Sets stop event, joins thread.

---

### 4.19 Position Manager — `position_manager.py`

**Class:** `PositionManager` (F2)

Tracks per-position state and computes adaptive stop-loss / take-profit thresholds.

**`PositionState` dataclass:**
- `ticker`, `entry_price`, `entry_cycle`, `qty`
- `price_history: list[float]` (last `POSITION_HISTORY_CYCLES` prices)
- `thresholds: PositionThresholds`
- `stop_loss_triggered: bool`

**`PositionThresholds` dataclass:**
- `stop_loss_pct` (negative), `take_profit_pct` (positive)
- `stop_source`: `"adaptive"` | `"user"` | `"conservative_override"`
- `take_source`: same
- `volatility_pct`, `sentiment_trend`, `explanation`

**`update_thresholds(ticker, closes, sentiment_scores, mode, user_stop_loss_pct, user_take_profit_pct)`:**

Algorithm (7 steps):
1. Compute cycle-to-cycle volatility from `closes` (std of percentage changes)
2. Raw stop-loss = `−volatility × POSITION_VOLATILITY_MULTIPLIER`
3. Conservative mode: cap at `−POSITION_MIN_STOP_LOSS_PCT`
4. Clamp to `[−POSITION_MAX_STOP_LOSS_PCT, −POSITION_MIN_STOP_LOSS_PCT]`
5. Compute take-profit based on sentiment trend (improving/stable/deteriorating)
6. User override: apply if more restrictive than computed
7. Build explanation string

**`check_stop_loss(ticker, current_price)`:** Returns True if `(current − entry) / entry` < `stop_loss_pct`.

**`build_position_context(ticker, current_price, current_cycle)`:** Returns formatted text block:
```
=== POSITION CONTEXT (AAPL) ===
Entry:           $150.00  (cycle 3, 2 cycles ago)
Current price:   $145.00
Position P&L:    -3.33%  (-$50.00 on 10 shares)
Trend:           $152.00 → $148.00 → $145.00
Active thresholds:
  Stop-loss:    -4.00%  [adaptive]
  Take-profit:  +7.00%  [adaptive]
Distance to stop-loss:    -0.67%  (caution)
Distance to take-profit:  +10.33%  (far)
...explanation...
```

**`load_from_journal(positions, session_id, journal_path)`:** Restores position states from journal on session resume.

---

### 4.20 Technical Analyser — `technical_analyser.py`

**Module:** `technical_analyser` (F3, stdlib only — no pandas/numpy)

**`analyse(closes, current_price, ticker, rsi_period, bb_period, bb_std, rsi_overbought, rsi_oversold, bb_squeeze_pct)`:**

Computes two indicators:

**RSI(14) — Wilder's smoothing:**
- Initial RS = average of first 14 up/down moves
- Subsequent: `avg_gain = (avg_gain × 13 + gain) / 14` (Wilder's method)
- RSI = 100 − 100 / (1 + RS)
- Signals: oversold (< 30), overbought (> 70), neutral

**Bollinger Bands(20) — population std:**
- Middle band = SMA(20)
- Upper/lower = SMA ± 2 × population std (not sample std)
- Signals: above upper band, below lower band, near upper/lower, in middle, squeeze (bandwidth < 1.5%)

**Returns `TechnicalSignals` dataclass:**
- `rsi`, `rsi_signal` ("oversold"|"overbought"|"neutral"|"N/A")
- `bb_upper`, `bb_middle`, `bb_lower`, `bb_signal`
- `prompt_section` — formatted text block injected into LLM prompt:

```
=== TECHNICAL SIGNALS (AAPL) ===
RSI(14):  45.2  → neutral
BB(20):   upper=$155.00  middle=$150.00  lower=$145.00  → in middle
```

---

### 4.21 User Preference Engine — `user_preference_engine.py`

**Class:** `UserPreferenceEngine` (F4)

Infers and tracks user preferences from their natural-language prompts and in-session behaviour.

**`extract_from_prompt(prompt, t_behavior)`:**
- Uses `qwen2.5:3b` to parse: risk level, sectors, excluded sectors, ethics, time horizon
- Detects emotional tone (anxious, optimistic, pessimistic, excited, neutral) and emotion score
- Writes back to session JSON (atomic)

**`compute_derived_parameters()`:**
- From `style_inferred` + risk level + emotion:
  - `derived_confidence_delta`: shift to confidence threshold (e.g., anxious → +0.1)
  - `derived_position_pct_delta`: shift to position size (e.g., low risk → −0.03)
  - `derived_mode_bias`: "conservative" | "aggressive" | "none"

**`get_effective_confidence_threshold(base)`:** `base + derived_confidence_delta`

**`get_effective_position_pct(base)`:** `max(0.02, base + derived_position_pct_delta)`

**`check_conflict(proposed_action, ticker, ticker_pnl_pct, current_pnl_pct, sentiment_score, mode)`:**

Detects 4 conflict types:
1. `buying_while_losing` — BUY while portfolio losing + low/unspecified risk profile
2. `excluded_sector` — BUY in a sector the user excluded
3. `emotional_vs_sentiment` — BUY while user is anxious + negative sentiment
4. `risk_vs_conservative_mode` — BUY with high-risk preference but in conservative mode

Returns conflict dict or None.

**`apply_minimum_modification(conflict, session)`:**
- Type 1: reduces position size by applying conservative mode bias
- Type 2: overrides action to hold
- Type 3: sets conservative mode bias
- Type 4: reduces position size
- Logs conflict to `session["preference_conflicts"]`

**`build_prompt_section()`:**
Returns formatted text injected into LLM prompt:
```
=== USER PREFERENCES ===
Detected style:    aggressive  (hold_rate: 20%, confirm_rate: 80%)
Emotional tone:    optimistic [+0.40]
Risk:              high
Preferred sectors: tech, energy
Excluded sectors:  none
Ethical filters:   none

Adapted parameters:
  Confidence threshold: 0.65 → 0.60  (delta: -0.05)
  Position size:        10.00% → 13.00%  (delta: +0.03)
  Mode bias:            none
```

**Implicit style inference:**
- `style_hold_rate` — fraction of cycles ending in HOLD (updated each cycle)
- `style_confirm_rate` — fraction of user confirmations (updated on each confirm/reject)
- `style_override_count` / `style_reject_sl_count` — counters
- `style_inferred`: `"passive"` (hold_rate > 0.7), `"aggressive"` (confirm_rate > 0.7 + low hold), `"cautious"` (many stop-loss rejections), or `"undetected"`

---

### 4.22 Journal — `journal.py`

Append-only JSONL audit log. All entries written atomically using a write counter + periodic compaction.

**Entry schema:** ts, cycle, ticker, session_id, action, conf, conf_raw, stale_penalty, reasoning, accuracy_review, decision_source, price, price_timestamp, ma5, trend, sentiment, sentiment_label, data_ok, imitative_source, prompt_snapshot, t_wait_used, t_behavior_used, mode, portfolio_mode_reason, order_id, market_open, price_after, outcome_pct, cash, portfolio_value, pnl_pct, positions.

**`log_error(source, error, ticker, session_id)`:** Writes to `error_log.jsonl`.

**`read_session_summary(session_id)`:** Computes aggregate stats (cycles, orders, P&L, decision counts) from journal for the session summary panel.

---

### 4.23 LLM Stream — `llm_stream.py`

Thin wrapper around Ollama with optional streaming.

**`LOOP_VERBOSE: bool`** — when True, streams LLM tokens to console during the loop phase (matching discovery phase verbosity).

Provides `generate(model, prompt, format, options, keep_alive)` that either streams or collects the response silently depending on the flag.

---

### 4.24 Dashboard — `ui/dashboard.py`

**Class:** `Dashboard`

Rich TUI layer. Never raises — all methods wrapped in try/except.

**Live display (`_build_renderable`):**
- Portfolio table (Cash, Value, P&L, mode badge)
- Agent proposal (last action per ticker with confidence)
- Journal tail (last 5 entries with outcomes)
- Countdown bar with key hints (ENTER · a · p · q · s · m)

**Key panels:**

| Method | Panel | Trigger |
|---|---|---|
| `print_cycle_summary()` | `◆ CYCLE N — SUMMARY` | End of each cycle |
| `print_decision_news()` | Per-ticker decision + caption + news | After each ticker decision |
| `print_disruptor_articles()` | `BREAKING — TICKER` | When disruptor articles detected |
| `print_stop_loss_proposal()` | `STOP-LOSS — Confirmation required` | F2 stop-loss trigger |
| `print_preference_conflict()` | `PREFERENCE CONFLICT` | F4 conflict detected |
| `print_correlation_matrix()` | `◆ NCCI — Ticker correlations` | Every cycle with ≥2 tickers |
| `print_discovery_candidates()` | `◆ DISCOVERY — Ticker candidates` | After discovery |
| `print_portfolio_positions()` | `◆ PORTFOLIO — Open positions` | After discovery / on resume |
| `ask_strategy_switch()` | `◆ AUTO STRATEGY SWITCH` | Every 5 cycles if switch recommended |
| `confirm_or_reprompt()` | `Confirm Discovery` | During discovery |
| `print_resoconto()` | `SESSION SUMMARY` | On startup (if resuming) and shutdown |
| `print_shutdown_message()` | `SHUTDOWN` | On exit |

**`wait_for_user_input(timeout_seconds)`:**
- Shows live countdown panel
- Reads one line from stdin with timeout
- Returns `{"source": "confirmed"|"timeout"|"prompt_append"|"prompt_change"|"questionnaire"|"strategy_select"|"override", "data": {...}}`

---

## 5. Feature Deep-Dives

### F1: Unified News Context Layer

**Problem:** Previously, news was passed to the LLM in fragmented ways (take_profit_hint, historical_news_context, disruptor_context as separate parameters). This made the prompt inconsistent and hard to tune.

**Solution:** `news_log.build_news_context_for_prompt()` builds a single, unified `=== NEWS CONTEXT ===` block from three sources:

1. **Current cycle articles** — fetched this cycle, sorted by relevance
2. **Historical articles** (last `NEWS_CONTEXT_HISTORY_CYCLES` cycles, score ≥ `NEWS_CONTEXT_MIN_RELEVANCE`) — deduplicated against current cycle titles
3. **Disruptor articles** — written by `MarketDisruptor` background thread when breaking news detected

Total article count capped at `NEWS_CONTEXT_MAX_ARTICLES` (default 6) with current cycle getting priority.

The block is passed as `news_context=` to `Reasoner.decide()`, completely replacing the old fragmented approach.

---

### F2: Adaptive Position Manager

**Problem:** Fixed stop-loss percentages don't account for each ticker's actual volatility.

**Solution:** `PositionManager` computes adaptive thresholds per position:

```
raw_stop_loss = −volatility_pct/cycle × POSITION_VOLATILITY_MULTIPLIER
```

Where `volatility_pct/cycle` = std(cycle-to-cycle % changes over last 20 closes).

Conservative mode applies a tighter cap (`−POSITION_MIN_STOP_LOSS_PCT = −2.0%`). User can override via session fields `user_stop_loss_pct` / `user_take_profit_pct` (more restrictive wins).

**Stop-loss flow in the loop:**
1. `update_price(ticker, price)` each cycle
2. `update_thresholds(...)` each cycle
3. `check_stop_loss(ticker, price)` — if triggered, show GUI proposal with 20s timeout
4. On confirmation: place SELL order, call `on_position_closed(ticker)`
5. `build_position_context(ticker, price, cycle)` — pass to LLM as `position_context=`

---

### F3: Technical Indicators

**Problem:** LLM decisions were based only on MA5, trend label, and sentiment — no deeper technical signals.

**Solution:** `technical_analyser.analyse()` computes RSI(14) and Bollinger Bands(20) from the same 20 close bars already fetched in step [2]. Uses **stdlib only** (no pandas/numpy) for portability.

**RSI Wilder's smoothing** eliminates the bias of simple average RSI at the start of the series.

**Bollinger Bands population std** (divides by N, not N−1) matches industry-standard charting convention.

Both signals are formatted into a `prompt_section` injected into the LLM prompt.

---

### F4: User Preference Engine

**Problem:** The agent had no awareness of the user's risk appetite, emotional state, or implicit trading style.

**Solution:** A four-layer preference model:

| Layer | Source | Fields |
|---|---|---|
| Explicit | LLM parsing of prompt | risk_level, sectors, ethics, time_horizon |
| Emotional | LLM tone detection | pref_emotion, pref_emotion_score |
| Implicit style | In-session behaviour | hold_rate, confirm_rate, override_count |
| Derived | Computed from above | confidence_delta, position_pct_delta, mode_bias |

The derived parameters shift the agent's confidence threshold and position size automatically. Conflict detection fires when the proposed action contradicts stated preferences, applying minimum modifications (not full veto) to preserve agent autonomy.

---

## 6. Data Flows

### Per-cycle data flow

```
Alpaca API             Ollama (qwen2.5:3b)     Ollama (gemma4:12b)
│                      │                       │
│ get_portfolio()       │                       │
│ get_bars(ticker)      │                       │
│ get_news(ticker)  ───►│ sentiment.analyse()   │
│                       │ news_log.extract_kw() │
│                       │                       │
│                       │                       │ reasoner.decide(
│                       │                       │   news_context,
│                       │                       │   position_context,
│                       │                       │   technical_signals,
│                       │                       │   user_preferences,
│                       │                       │   ...)
│                       │                       │
│ broker.place_order() ◄────────────────────────│ decision["action"]
│
└─► journal.write_entry()
└─► memory_manager.update()
└─► news_log.mark_decision()
└─► dashboard.update()
```

### Session persistence flow

```
session.json  ←──(atomic write)── SessionManager.save()
journal.jsonl ←──(append)────────── journal.write_entry()
news_log.jsonl ←──(append+compact)── news_log.persist_articles()
disruptor_news.jsonl ←──(atomic)──── MarketDisruptor._poll()
```

---

## 7. Persistence & Atomicity

All writes use the pattern:
```python
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f)
os.replace(tmp, path)  # atomic on POSIX and Windows (since Python 3.3)
```

This prevents partial writes from corrupting state files on crashes or `KeyboardInterrupt`.

**JSONL files** (`journal.jsonl`, `news_log.jsonl`, `disruptor_news.jsonl`) use append-only writes. Compaction is triggered periodically and rewrites the file atomically.

---

## 8. LLM Integration

### Model roles

| Model | Used by | Format |
|---|---|---|
| `gemma4:12b` | `Reasoner.decide()`, `DiscoveryAgent._call_llm()` | JSON schema (structured output) |
| `qwen2.5:3b` | `sentiment.analyse()`, `news_log.extract_keywords_and_relevance()`, `behavior_questionnaire.*`, `user_preference_engine.extract_from_prompt()` | JSON schema |

### Timeout strategy

All LLM calls use `keep_alive="30s"` to keep models warm between calls. The `t_behavior` timeout (derived from actual Ollama latency × `T_BEHAVIOR_MULTIPLIER`) is passed through to `ollama.generate()` via `options={"num_predict": N}` limits to prevent runaway generation.

### Structured output

Every LLM call specifies a JSON schema via Ollama's `format=` parameter. This eliminates parsing fragility — the model is forced to output valid JSON matching the schema.

### Verbose mode

When `--verbose` is passed to `main.py`, `llm_stream.LOOP_VERBOSE = True` causes the reasoning LLM's token stream to be printed to console during the loop phase (same as during discovery).

---

## 9. Security Constraints

### Paper trading (CRITICAL)

```python
# broker.py
ALPACA_PAPER: bool = True  # HARDCODED — never change
```

This is a hardcoded constant that is **never read from the environment** and **never passed as a parameter**. The `Broker` class always initialises with `paper=True`. This cannot be changed without modifying the source code.

### No credentials in code

All API keys are read from the `.env` file via `python-dotenv`. The `.gitignore` excludes `.env`.

### Input validation

User-entered tickers are validated via `ToolExecutor.resolve_ticker()` before use. Only tickers confirmed as tradeable on Alpaca are added to the active ticker list.

---

## 10. Test Suite

**126 tests** across 12 test files. Run with:
```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Note: `test_connections.py`, `test_correlation_engine.py`, `test_news_log.py` have a pre-existing fixture error (`label` fixture not found) unrelated to the agent logic — they require live connections.

| File | Tests | Coverage area |
|---|---|---|
| `test_adaptive_timeout.py` | 8 | Timeout calibration and bounds |
| `test_behavior.py` | 6 | Behavior change application |
| `test_journal.py` | 12 | Journal write, read, summary |
| `test_memory_manager.py` | 14 | HOT/WARM/COLD tier logic |
| `test_news_context_layer.py` | 13 | `build_news_context_for_prompt()` |
| `test_position_manager.py` | 18 | All `PositionManager` methods |
| `test_session.py` | 10 | Session create, resume, save |
| `test_technical_analyser.py` | 16 | RSI + Bollinger Bands |
| `test_tool_executor.py` | 8 | ToolExecutor with mocked API |
| `test_user_preference_engine.py` | 21 | All `UserPreferenceEngine` methods |

---

## 11. Configuration Reference

All values read from `.env` (with defaults):

### Agent Behaviour
| Key | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD_NORMAL` | `0.65` | Min confidence to act in normal mode |
| `CONFIDENCE_THRESHOLD_CONSERVATIVE` | `0.80` | Min confidence in conservative mode |
| `MAX_POSITION_PCT_NORMAL` | `0.10` | Max portfolio % per position, normal |
| `MAX_POSITION_PCT_CONSERVATIVE` | `0.05` | Max portfolio % per position, conservative |
| `DRAWDOWN_THRESHOLD` | `0.05` | P&L below this → conservative mode |

### Adaptive Timeout
| Key | Default | Description |
|---|---|---|
| `T_WAIT_MIN` | `15` | Min inter-cycle wait (seconds) |
| `T_WAIT_MAX` | `120` | Max inter-cycle wait |
| `T_BEHAVIOR_MIN` | `20` | Min LLM timeout |
| `T_BEHAVIOR_MAX` | `180` | Max LLM timeout |
| `T_WAIT_MULTIPLIER` | `3.0` | API latency × multiplier = T_WAIT |
| `T_BEHAVIOR_MULTIPLIER` | `5.0` | Ollama latency × multiplier = T_BEHAVIOR |

### F1: News Context
| Key | Default | Description |
|---|---|---|
| `NEWS_CONTEXT_HISTORY_CYCLES` | `5` | Cycles of history to include |
| `NEWS_CONTEXT_MAX_ARTICLES` | `6` | Max total articles in context block |
| `NEWS_CONTEXT_MIN_RELEVANCE` | `0.50` | Min relevance score for historical articles |

### F2: Position Manager
| Key | Default | Description |
|---|---|---|
| `POSITION_MIN_STOP_LOSS_PCT` | `2.0` | Floor: stop-loss cannot be tighter than −2% |
| `POSITION_MAX_STOP_LOSS_PCT` | `8.0` | Ceiling: stop-loss cannot be wider than −8% |
| `POSITION_MIN_TAKE_PROFIT_PCT` | `3.0` | Floor: take-profit minimum |
| `POSITION_MAX_TAKE_PROFIT_PCT` | `15.0` | Ceiling: take-profit maximum |
| `POSITION_VOLATILITY_MULTIPLIER` | `3.0` | volatility × this = raw stop-loss |
| `POSITION_HISTORY_CYCLES` | `8` | Price history buffer size |
| `POSITION_SENTIMENT_TREND_WINDOW` | `3` | Cycles for sentiment trend (improving/stable/deteriorating) |

### F3: Technical Indicators
| Key | Default | Description |
|---|---|---|
| `TECHNICAL_BARS_LOOKBACK` | `20` | Bars to fetch for technical analysis |
| `TECHNICAL_RSI_PERIOD` | `14` | RSI period |
| `TECHNICAL_BB_PERIOD` | `20` | Bollinger Bands period |
| `TECHNICAL_BB_STD` | `2.0` | BB standard deviation multiplier |
| `TECHNICAL_RSI_OVERBOUGHT` | `70.0` | RSI overbought threshold |
| `TECHNICAL_RSI_OVERSOLD` | `30.0` | RSI oversold threshold |
| `TECHNICAL_BB_SQUEEZE_PCT` | `1.5` | BB bandwidth below this = squeeze |

### F4: User Preference Engine
| Key | Default | Description |
|---|---|---|
| `PREFERENCE_UPDATE_EVERY` | `5` | Cycles between implicit style updates |
| `PREFERENCE_WAIT_HISTORY` | `10` | Wait choices kept in history |
| `PREFERENCE_CONFLICT_THRESHOLD` | `0.05` | Portfolio loss % that triggers buying_while_losing conflict |
| `PREFERENCE_EMOTION_WEIGHT` | `0.3` | Weight of emotion in derived params |
| `PREFERENCE_STYLE_WEIGHT` | `0.4` | Weight of implicit style in derived params |

### NCCI (Correlation Engine)
| Key | Default | Description |
|---|---|---|
| `NCCI_REBUILD_EVERY` | `10` | Cycles between matrix rebuilds |
| `NCCI_THRESHOLD_DISPLAY` | `0.20` | Min NCCI to show in white (not grey) |
| `NCCI_KEYWORD_MIN_WEIGHT` | `0.15` | Min keyword relevance weight for NCCI |
| `NCCI_HALF_LIFE_DAYS` | `7.0` | Time decay half-life for NCCI |

---

*Generated 2026-06-17 — BIP Hackathon 2026 Trading Agent*

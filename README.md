# 🤖 BIP Hackathon 2026 — Autonomous Trading Agent

An **autonomous AI trading agent** powered by dual local LLMs, operating on simulated markets with advanced memory management, adaptive timeouts, and persistent decision tracking.

**Level:** 3 (Autonomous Agent)  
**Status:** Active Development  
**Language:** Python 3.12+

---

## 🎯 Project Overview

This project implements a sophisticated autonomous trading system that:

✅ **Runs 24/7** on Alpaca Paper Trading (safe, simulated broker)  
✅ **Dual-LLM reasoning:** Gemma4:12b (decisions) + qwen2.5:3b (sentiment)  
✅ **Adaptive timeouts** based on real-time API latency  
✅ **Three-tier memory** (HOT/WARM/COLD) for scalable decision context  
✅ **Static imitative layer** with 5 known investor strategies  
✅ **Persistent JSONL journal** for full decision traceability  
✅ **Self-correcting** with STALE_DATA penalties and timeout fallbacks  
✅ **Rich TUI dashboard** for real-time monitoring  
✅ **News Co-occurrence Correlation Index (NCCI)** with Jaccard similarity for cross-ticker analysis  
✅ **Dynamic Strategy Library** with customizable trading strategies  
✅ **Behavior Questionnaire System** for adaptive agent personality  
✅ **Advanced Discovery Phase** with ticker cross-search and validation  
✅ **MarketDisruptor Integration** — Breaking news panel with market disruption analysis  

**Evaluation Criteria:**
- Demonstrable Functionality (40%) — stable loop, error handling, persistent logs
- Reasoning Quality (35%) — no hallucinations, confidence gating, self-reflection
- Originality (25%) — dual-model, HOT/WARM/COLD, imitative layer, adaptive timeout, NCCI correlation analysis, MarketDisruptor

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- **Python 3.12+**
- **Ollama** (local LLM server) running on `http://localhost:11434`
- **Alpaca Paper Trading account** with API keys
- **uv** package manager

### 2️⃣ Installation

```bash
# Clone the repo
git clone https://github.com/pierluigiTrocini/bip_hackathon2026.git
cd bip_hackathon2026

# Install dependencies
uv sync

# Pull Ollama models (download ~12GB + 3GB)
ollama pull gemma4:12b
ollama pull qwen2.5:3b

# Verify models are available
ollama list
```

### 3️⃣ Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials:
# - ALPACA_API_KEY = your paper trading key
# - ALPACA_SECRET_KEY = your paper trading secret
```

### 4️⃣ Test Infrastructure

```bash
# Verify all connections work
uv run python test_connection.py
```

Expected output:
```
[PASS] Account balance & status: account_id=PA*** | cash=$100,000.00 | portfolio=$100,000.00
[PASS] Open positions: 0 open position(s)
[PASS] Latest bar — AAPL, MSFT: AAPL=$195.50 | MSFT=$420.30
[PASS] Recent news — AAPL: 3 article(s) — e.g.: "Apple Q3 earnings beat..."
[PASS] Ollama gemma4:12b: model=gemma4:12b | reply="OK"
[PASS] Ollama qwen2.5:3b: model=qwen2.5:3b | reply="OK"

All 6 checks passed. You are ready to start!
```

### 5️⃣ Run the Agent

```bash
uv run python main.py
```

The agent will:
1. Ask if you want to resume a previous session or start new
2. If new, ask for a trading strategy/prompt (e.g., "green investing")
3. Run a **discovery phase** to select and validate tickers with cross-search
4. Begin autonomous trading loop (Ctrl+C to gracefully stop)

---

## 🏗️ Architecture Overview

### **Core Modules** (`src/agent/`)

| Module | Purpose |
|--------|---------|
| **`config.py`** | Load environment variables (single source of truth) |
| **`adaptive_timeout.py`** | Measure API/Ollama latency → compute dynamic timeouts |
| **`tool_executor.py`** | Execute external calls (prices, portfolio, news) with retry + caching |
| **`journal.py`** | Persistent JSONL append-only log of all decisions + outcomes |
| **`memory_manager.py`** | Three-tier memory (HOT/WARM/COLD) per ticker |
| **`imitative_layer.py`** | Load 5 investor strategies → inject into reasoning |
| **`sentiment.py`** | Classify news sentiment using qwen2.5:3b (JSON schema) |
| **`reasoner.py`** | Make buy/sell/hold decisions using Gemma4:12b with penalties |
| **`broker.py`** | Execute orders on Alpaca, manage positions |
| **`session.py`** | Manage session lifecycle (resume/new, UUID, state) |
| **`behavior.py`** | Handle active prompt changes + fallback logic + behavior questionnaire |
| **`loop.py`** | Main autonomous trading loop |
| **`discovery.py`** | Discovery phase with ticker validation and cross-search |
| **`correlation.py`** | News Co-occurrence Correlation Index (NCCI) + Jaccard similarity |
| **`strategy_library.py`** | Dynamic strategy library with customizable trading strategies |
| **`market_disruptor.py`** | Market disruption analysis with breaking news detection |
| **`ui/dashboard.py`** | Rich TUI dashboard for monitoring with strategy display & disruption panel |

### **Decision Flow**

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN LOOP                            │
│  (runs every T_WAIT seconds, adaptive)                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  1. Fetch Market Data          │
          │  - Latest price (AAPL, TSLA...) │
          │  - Moving average (MA5)         │
          │  - Recent news articles         │
          │  - Cross-ticker correlations    │
          │  - Breaking news alerts         │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  2. Market Disruption Check     │
          │  (MarketDisruptor analysis)    │
          │  - Severity scoring            │
          │  - Rapid response mode          │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  3. Sentiment Analysis          │
          │  (qwen2.5:3b)                  │
          │  Score: -1.0 to +1.0           │
          │  + Confidence gate              │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  4. Correlation Analysis        │
          │  (NCCI + Jaccard similarity)   │
          │  Cross-ticker patterns         │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  5. Build Memory Context       │
          │  HOT: last 5 decisions         │
          │  WARM: LLM-compacted summary   │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  6. Reason (Gemma4:12b)        │
          │  - Strategy-based prompts      │
          │  - Imitative hints injected    │
          │  - STALE_DATA penalty applied  │
          │  - Timeout: T_behavior         │
          │  Decision: BUY | SELL | HOLD   │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  7. Order Quantity Calc.       │
          │  - Enhanced position sizing    │
          │  - Strategy signal mapping     │
          │  - Market hours check          │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  8. Place Order (if BUY/SELL)  │
          │  - Position sizing (10% max)   │
          │  - Error handling              │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  9. Journal + Update Memory    │
          │  - Write JSONL entry          │
          │  - Update HOT/WARM            │
          │  - Track outcome (next cycle) │
          └────────────────────────────────┘
```

---

## 🧠 Key Concepts

### **1. Adaptive Timeout**

Measures real-time latency of API calls and Ollama inference, then adjusts wait times dynamically:

```
T_wait = clamp(API_latency_avg × 3.0, min=15s, max=120s)
T_behavior = clamp(Ollama_latency_avg × 5.0, min=20s, max=180s)
```

Updated after every external call. Prevents timeouts on slow networks.

### **2. STALE_DATA Penalty**

If price data is older than a threshold:
```
penalty = min(staleness_seconds / 60 × 0.05, 0.40)
confidence_final = max(0.0, confidence_raw - penalty)
```

Automatically reduces confidence on old data. Hard constraint: if data is very stale, model returns safe HOLD.

### **3. Three-Tier Memory**

Per ticker, maintains decision history at different compression levels:

| Tier | Size | Storage | Refresh |
|------|------|---------|---------|
| **HOT** | Last 5 | Full entries, in-memory | Every decision |
| **WARM** | 6-20 | LLM-compacted 1-line summary | Every 15 overflows |
| **COLD** | 21+ | Raw JSONL journal | Never (archive) |

`build_context()` returns a fixed-size string suitable for LLM injection.

### **4. Imitative Layer**

Loads a static dataset of 5 known investor strategies:

- **Buffett: Value Investing** — undervalued stocks, long-term, fundamentals
- **Lynch: Growth @ Reasonable Price** — earnings growth, small-cap, PEG ratio
- **Simons: Quantitative** — momentum, trends, data-driven signals
- **ESG: Green Investing** — sustainability, climate, renewable energy
- **Defense: Sector Investing** — government contracts, geopolitical hedge

Filters strategies by keyword matching against active prompt, injects top 2 matches into reasoning.

### **5. Dynamic Strategy Library**

Customizable trading strategies with expert-level prompts:
- Strategy encapsulation for various trading methodologies
- Seamless integration into reasoner for system prompt generation
- Strategy switching and signal mapping capabilities
- Dashboard display of current strategy

### **6. News Co-occurrence Correlation Index (NCCI)**

Analyzes cross-ticker correlations using:
- **NCCI Algorithm:** Measures co-occurrence frequency of stocks in news articles
- **Jaccard Similarity:** Calculates set-based similarity between ticker keyword sets
- **Persistent News Logs:** Tracks historical news with timestamps and URLs
- **Cross-ticker Pattern Detection:** Identifies related stocks for diversified trading

### **7. Behavior Questionnaire System**

Enables adaptive agent personality through:
- Interactive questionnaire for trading preference assessment
- Dynamic prompt injection based on behavioral profile
- Contrarian vs. market-following signal mapping
- Session-persistent behavior configuration

### **8. Enhanced Discovery Phase**

Improved ticker selection process:
- Keyword-based cross-search for relevant tickers
- Validation and verification of ticker selections
- News-driven candidate generation
- User confirmation before trading starts

### **9. MarketDisruptor Analysis**

Real-time breaking news detection and market impact assessment:
- **Rapid News Scoring:** Immediate processing of market-moving news
- **Disruption Severity Levels:** Classification from LOW to CRITICAL
- **Dashboard Breaking News Panel:** ⚡ visual alert system with recent major market events
- **Adaptive Response Mode:** Switching to higher sensitivity during high-disruption periods
- **Cross-ticker Impact Mapping:** Understanding how news affects related assets

### **10. Journal Entry**

Every decision is logged as a complete JSONL entry:

```json
{
  "ts": "2026-06-16T15:30:00Z",
  "cycle": 42,
  "ticker": "AAPL",
  "action": "buy",
  "conf": 0.68,
  "conf_raw": 0.70,
  "stale_penalty": 0.02,
  "reasoning": "Price below MA5, positive sentiment.",
  "accuracy_review": "Based on fresh data.",
  "price": 195.50,
  "ma5": 194.20,
  "sentiment": 0.45,
  "sentiment_label": "positive",
  "correlations": [{"ticker": "MSFT", "ncci": 0.82, "jaccard": 0.65}],
  "mode": "normal",
  "strategy": "value_investing",
  "order_id": "order-abc123",
  "price_after": null,
  "outcome_pct": null,
  "data_ok": true,
  "market_open": true,
  "cash": 92000.0,
  "portfolio_value": 95000.0,
  "pnl_pct": -0.05
}
```

Outcome fields (`price_after`, `outcome_pct`) are filled in the **next cycle** when a new price is available.

### **11. Session Persistence**

Saved state in `data/session.json`:

```json
{
  "session_id": "uuid-4-...",
  "status": "active",
  "cycle": 123,
  "active_prompt": "green investing strategy",
  "initial_prompt": "fallback if timeout",
  "active_strategy_id": "green_esg",
  "behavior_profile": {"risk_tolerance": "medium", "approach": "balanced"},
  "portfolio_snapshot": {...},
  "behavior_change_count": 0
}
```

On restart, agent asks: "Resume previous session? [s/N]"

---

## ⚙️ Configuration

Edit `.env` to customize behavior:

```bash
# ── Alpaca Paper Trading ─────────────────────────────────────────
ALPACA_API_KEY=PK123abc...
ALPACA_SECRET_KEY=abc123...

# ── Ollama Local Models ─────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REASONING_MODEL=gemma4:12b
OLLAMA_SENTIMENT_MODEL=qwen2.5:3b

# ── Trading Parameters ──────────────────────────────────────────
TICKERS=AAPL,TSLA,NVDA,MSFT
CONFIDENCE_THRESHOLD_NORMAL=0.65
CONFIDENCE_THRESHOLD_CONSERVATIVE=0.80
MAX_POSITION_PCT_NORMAL=0.10         # max 10% of cash per position
MAX_POSITION_PCT_CONSERVATIVE=0.05   # max 5% conservative mode
DRAWDOWN_THRESHOLD=0.05              # switch to conservative if PnL < -5%

# ── Adaptive Timeout Multipliers ────────────────────────────────
T_WAIT_MULTIPLIER=3.0
T_BEHAVIOR_MULTIPLIER=5.0
T_WAIT_MIN=15
T_WAIT_MAX=120
T_BEHAVIOR_MIN=20
T_BEHAVIOR_MAX=180

# ── Memory Management ───────────────────────────────────────────
HOT_WINDOW_SIZE=5
WARM_COMPACTION_TRIGGER=15

# ── Correlation Analysis ────────────────────────────────────────
NCCI_WINDOW_DAYS=7
JACCARD_MIN_THRESHOLD=0.3

# ── Data Paths ──────────────���───────────────────────────────────
JOURNAL_PATH=data/journal.jsonl
ERROR_LOG_PATH=data/error_log.jsonl
NEWS_LOG_PATH=data/news_log.jsonl
SESSION_PATH=data/session.json
IMITATIVE_DATASET_PATH=data/strategies/imitative_dataset.json
STRATEGY_CONFIG_PATH=data/strategies/strategy_config.json
```

---

## 📊 Output Files

The agent creates these files at runtime:

```
data/
├── journal.jsonl              # All trading decisions (append-only)
├── error_log.jsonl            # All errors and retries
├── news_log.jsonl             # All news articles with URLs (persistent)
├── session.json               # Current session state
└── strategies/
    ├── imitative_dataset.json # 5 investor strategies (created if missing)
    └── strategy_config.json   # Dynamic strategy library configuration
```

**Analyze Results:**

```bash
# See last 10 decisions
tail -10 data/journal.jsonl | jq .

# Count decisions by action
jq '.action' data/journal.jsonl | sort | uniq -c

# Calculate final P&L
tail -1 data/journal.jsonl | jq '.pnl_pct'

# Analyze correlations
jq '.correlations' data/journal.jsonl | jq -s 'add | group_by(.ticker) | map({ticker: .[0].ticker, avg_ncci: (map(.ncci) | add / length)})'

# View recent news
tail -20 data/news_log.jsonl | jq '{ticker, headline, url}'
```

---

## 🧪 Testing

Run infrastructure tests before starting:

```bash
uv run python test_connection.py
```

Checks:
- ✅ Alpaca account & positions
- ✅ Market data feed (AAPL, MSFT latest bars)
- ✅ Alpaca news feed
- ✅ Ollama models available (Gemma4 + Qwen2.5)
- ✅ News persistence and URL validation
- ⚠️ Optional: Anthropic Claude, NewsAPI, Polygon.io

---

## 📁 Repository Structure

```
bip_hackathon2026/
├── README.md                           # This file
├── CLAUDE_CODE_SPEC.md                 # Full 60KB specification
├── pyproject.toml                      # Dependencies
├── .env.example                        # Template (commit this)
├── .gitignore
├── main.py                             # Entry point
├── test_connection.py                  # Infrastructure test
│
├── src/agent/
│   ├── __init__.py
│   ├── config.py                       # Load env vars
│   ├── adaptive_timeout.py             # Module 1: Latency tracking
│   ├── tool_executor.py                # Module 2: External calls
│   ├── journal.py                      # Module 3: JSONL logs
│   ├── memory_manager.py               # Module 4: HOT/WARM/COLD
│   ├── imitative_layer.py              # Module 5: Strategy dataset
│   ├── sentiment.py                    # Module 6: qwen2.5:3b
│   ├── reasoner.py                     # Module 7: Gemma4:12b
│   ├── broker.py                       # Module 8: Alpaca orders
│   ├── session.py                      # Module 9: Session lifecycle
│   ├── behavior.py                     # Module 10: Behavior + questionnaire
│   ├── loop.py                         # Module 11: Main loop
│   ├── discovery.py                    # Module 12: Discovery phase + cross-search
│   ├── correlation.py                  # Module 13: NCCI + Jaccard
│   ├── strategy_library.py             # Module 14: Dynamic strategies
│   ├── market_disruptor.py             # Module 15: Market disruption analysis
│   └── ui/dashboard.py                 # Module 16: Rich TUI + disruption panel
│
├── tests/
│   ├── test_connections.py
│   ├── test_adaptive_timeout.py
│   ├── test_tool_executor.py
│   ├── test_journal.py
│   ├── test_memory_manager.py
│   ├── test_behavior.py
│   ├── test_session.py
│   ├── test_correlation.py
│   ├── test_strategy_library.py
│   └── test_market_disruptor.py
│
├── data/                               # Created at runtime
│   ├── journal.jsonl
│   ├── error_log.jsonl
│   ├── news_log.jsonl
│   ├── session.json
│   └── strategies/
│       ├── imitative_dataset.json
│       └── strategy_config.json
│
└── uv.lock
```

---

## 🛠️ Tech Stack

| Component | Tech |
|-----------|------|
| **Language** | Python 3.12+ |
| **Broker** | Alpaca Paper Trading |
| **Reasoning LLM** | Gemma4:12b via Ollama |
| **Sentiment LLM** | qwen2.5:3b via Ollama |
| **Package Manager** | uv |
| **Data Format** | JSONL (append-only) |
| **UI** | rich + textual |
| **Testing** | pytest |
| **Correlation** | Custom NCCI + Jaccard similarity |
| **Market Disruption** | Real-time breaking news detection |

**Dependencies:**
- `alpaca-py` — trading API
- `ollama` — local LLM client
- `requests` — HTTP calls
- `python-dotenv` — .env loader
- `rich` — terminal formatting
- `textual` — TUI framework

---

## 🎮 Interactive Features

### **Session Management**
```
┌─────────────────────────────────────────────┐
│   PREVIOUS SESSION DETECTED                 │
│                                             │
│ ID:            abc12345...                  │
│ Started:       2026-06-16 14:30:00         │
│ Cycles:        42                          │
│ Orders:        15                          │
│ P&L:           +2.35%                      │
│ Autonomous:    12                          │
│ Errors:        0                           │
│ Strategy:      Value Investing              │
│                                             │
│ Resume this session? [s/N]: s               │
└─────────────────────────────────────────────┘
```

### **Behavior Questionnaire**
Interactive questionnaire that adapts agent trading personality based on user preferences for risk tolerance, market approach, and strategy alignment.

### **Discovery Phase**
Agent analyzes market news for a given strategy, performs cross-ticker search, validates candidates, and asks you to confirm selections before trading starts.

### **Dashboard with MarketDisruptor**
Real-time TUI showing:
- Current portfolio value
- Open positions
- Recent decisions with correlation data
- Current strategy and behavior profile
- **⚡ Breaking News Panel** — Major market events with disruption severity
- Error log
- Adaptive timeout values
- NCCI correlation matrix

---

## 🚨 Error Handling

The agent never crashes. On errors:

1. **API Failure** → Retry with exponential backoff (3 attempts)
2. **Cache hit** → Use stale data if fresh data unavailable (with penalty)
3. **Ollama timeout** → Return safe HOLD decision
4. **Alpaca order rejected** → Log error, skip that ticker
5. **Market closed** → Skip trading (paper market is 24h, but checks anyway)
6. **Invalid ticker from discovery** → Request user validation and re-search

All errors logged to `error_log.jsonl` with timestamp, source, retry count.

---

## 📈 Typical Session

```
1. [INIT] Load previous session or ask for new prompt + behavior profile
2. [CALIBRATE] Measure API + Ollama latency (3 pings each)
3. [QUESTIONNAIRE] Ask behavior preferences (risk, approach, strategy)
4. [DISCOVERY] Analyze news → cross-search tickers → present candidates → confirm
5. [LOOP] Every N seconds:
    - Fetch prices + news + correlations
    - Check for market disruptions (breaking news alerts)
    - Classify sentiment (qwen2.5)
    - Calculate NCCI + Jaccard correlations
    - Build memory context (HOT/WARM)
    - Reason about decision (Gemma4) with strategy injection
    - Calculate optimized order quantity
    - Execute order if BUY/SELL
    - Journal outcome with correlations + disruption flags
6. [SHUTDOWN] Ctrl+C → cancel open orders → save session
```

---

## 🔒 Safety Features

✅ **Paper Trading Only** — All trades are simulated, no real money  
✅ **Position Limits** — Max 10% of cash per position  
✅ **Confidence Gate** — Only trade if model is confident enough  
✅ **Market Hours Check** — Don't trade when markets are closed  
✅ **Staleness Penalty** — Reduce confidence on old data  
✅ **Graceful Degradation** — Fall back to cache or safe HOLD  
✅ **Full Traceability** — Every decision logged with reasoning + correlations  
✅ **URL Validation** — News URLs verified and persisted  
✅ **Cross-ticker Awareness** — NCCI correlation analysis for informed decisions  
✅ **Disruption Detection** — Real-time market disruption alerts with severity scoring  

---

## 📝 Recent Updates (Latest Commits)

### **June 17, 2026 — Today**

#### **MarketDisruptor Integration** 🎯
- Integrated breaking news detection into dashboard with ⚡ visual alert panel
- Added market disruption severity scoring (LOW → CRITICAL)
- Enhanced dashboard article display with formatted disruption indicators
- Improved LLM streaming and output formatting with grey styling

#### **Code Reorganization & Refactoring**
- Reorganized imports and streamlined portfolio position handling in main function
- Refactored LLM streaming and dashboard output formatting
- Consolidated market data fetching pipeline

### **June 17, 2026 — Earlier**
- **Correlation Index Output** — NCCI calculations now visible in dashboard and journals
- **LLM Verbosity Enhanced** — Improved logging and output formatting with grey styling for LLM responses
- **News Log Persistence** — Added missing news log file management to .gitignore
- **News URL Fixes** — Fixed missing URL fields in experimental news logging

### **June 16, 2026**
- **News Co-occurrence Correlation Index (NCCI)** — Implemented sophisticated cross-ticker correlation analysis using Jaccard similarity
- **Better News Retrieval** — Enhanced news persistence and keyword processing
- **Strategy Library Implementation** — Full integration of dynamic strategy library into reasoner
- **Enhanced Order Quantity** — Improved position sizing calculation with strategy signal mapping
- **Discovery Phase Improvements** — Advanced ticker validation and cross-search capabilities
- **Ticker Cross-Search** — Implemented cross-ticker correlation-based search
- **UI Dashboard Enhancements** — Added strategy display and real-time correlation visualization
- **Prompt Injection System** — Work-in-progress behavior questionnaire for adaptive prompts
- **Contrarian Thinking** — Added contrarian vs. market-following signal logic
- **Fallback Optimization** — Refined timeout fallback behavior

---

## 📝 Notes

- **Ollama Setup:** Gemma4 (12B) + Qwen2.5 (3B) require ~15GB total VRAM. Reduce if needed with smaller models.
- **First Run:** Expect 2-3 min for Ollama calibration + discovery phase + behavior questionnaire.
- **Performance:** Typical cycle time is 30-120s (adaptive).
- **Data Privacy:** All LLMs run locally. No data sent to cloud services.
- **Correlation Analysis:** NCCI calculations are incremental and cached for performance.
- **MarketDisruptor:** Breaking news detection runs in parallel to minimize latency impact.

---

## 📖 Full Specification

See **`CLAUDE_CODE_SPEC.md`** for the complete 60KB architecture specification, including:
- Module interfaces
- Data schemas
- Implementation notes
- Test requirements
- Edge cases

---

## 🎓 Learning Resources

- [Alpaca Trading API](https://docs.alpaca.markets/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Gemma Models](https://ai.google.dev/gemma/)
- [Qwen Models](https://huggingface.co/Qwen)

---

## 📄 License

This project is created for the **BIP Hackathon 2026**. See repository for license details.

---

## 👤 Author

**Pierluigi Trocini**, **Elisa Gigliotti** and **Chiara Costantino**  
BIP Hackathon 2026 Submission

---

**Happy Trading! 🚀**

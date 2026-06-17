# Jordan PEC Bot — Telegram Interface

**Bot:** [@jordan_pec_bot](https://t.me/jordan_pec_bot)  
**Library:** `python-telegram-bot >= 21.0`  
**Module:** `src/telegram_bot.py`

---

## Architecture

The bot runs in a **dedicated background thread** (`telegram-bot`) with its own isolated `asyncio` event loop. It starts automatically when the agent launches (`uv run main.py`) and shuts down with it.

```
main.py
  └─ TelegramNotifier.start()
       └─ Thread: _run_bot()
            └─ asyncio event loop
                 ├─ polling Telegram API
                 └─ command handlers
```

The agent loop communicates with the bot through two channels:

| Direction | Method | Purpose |
|---|---|---|
| Agent → Bot | `notify_action()` | Push decision alert per ticker |
| Agent → Bot | `notify_cycle_end()` | Update cached state at end of cycle |
| Bot → Agent | `on_strategy_change(id)` | Change active trading strategy |
| Bot → Agent | `on_prompt_change(mode, text)` | Update the agent's behavioural prompt |

Thread-safety is guaranteed via `threading.Lock` on all shared state.

---

## Configuration

Add to `.env`:

```env
TELEGRAM_BOT_TOKEN=<token from @BotFather>
TELEGRAM_CHAT_ID=<your numeric chat id>
```

Set `TELEGRAM_BOT_TOKEN=DISABLED` to disable the bot without removing the config.  
To retrieve your `CHAT_ID`: send any message to the bot, then call:

```
GET https://api.telegram.org/bot<TOKEN>/getUpdates
```

The `chat.id` field in the last result is your ID.

---

## Automatic Notifications (push)

The bot sends messages automatically — no user action required.

### Per-decision alert (`notify_action`)

Sent every time the agent makes a decision on a ticker.

```
🟢 MSFT → BUY  conf: 87%

_One-line caption from the LLM._

Reasoning: Extreme fear sentiment of -0.80 indicates a
point where mean reversion is expected.

📰 Supporting news:
  • Headline one [reuters]
  • Headline two [benzinga]

⚡ Breaking (disruptor):
  ⚡ Breaking headline [alpaca_disruptor]
```

**Action emojis:**

| Action | Emoji |
|---|---|
| buy | 🟢 |
| sell | 🔴 |
| hold | 🟡 |
| veto | 🚫 |
| wait | ⏳ |

Fields included: ticker, action, confidence %, one-line LLM caption, reasoning (max 300 chars), up to 4 supporting news articles with clickable links, up to 3 disruptor breaking news items.

---

## Commands (pull)

### `/resume`
Last completed cycle summary.

```
📊 Cycle 12 summary — 14:32 UTC
Strategy: Contrarian  |  P&L: 📈 +1.24%
Portfolio: $102,450.00  Cash: $45,200.00

🟢 AAPL → BUY  conf:82%  sent:+0.45  $213.50 (+1.2%)
   _Sentiment acceleration above threshold..._
🟡 TSLA → HOLD  conf:71%  sent:-0.10  $187.20
   _Neutral trend, no contrarian edge._
```

Returns "No cycle completed yet" if the agent has not finished its first cycle.

---

### `/breaking`
Latest news headlines for each monitored ticker (live fetch via NewsAPI).

```
📰 Latest news — 14:33 UTC

AAPL
  • Apple announces new chip roadmap [reuters]
  • ...

TSLA
  • Tesla Q2 delivery figures beat estimates [ft]
  • ...
```

Up to 3 articles per ticker. Article titles are clickable links when a URL is available.

---

### `/insider`
High-priority breaking news from the `MarketDisruptor` background thread (last 60 minutes, deduplicated by title).

```
⚡ Disruptor news — 14:34 UTC

MSFT  ⚡ Microsoft acquires major AI startup [alpaca_disruptor]
   _Brief summary up to 120 characters._
```

Returns "No recent disruptor news" if nothing arrived in the last hour.

---

### `/nerd`
Technical statistics for the current cycle: NCCI correlation matrix and per-ticker snapshot.

```
🔬 Technical stats — cycle 12

NCCI matrix (news correlation):
`          AAPL  TSLA  MSFT`
`AAPL   1.0  +0.42   —`
`TSLA  +0.42   1.0  +0.18`
`MSFT    —   +0.18   1.0`

Technical snapshot:
  AAPL  trend:up  sent:+0.45  P&L:+1.2%  entry:$211.00
  TSLA  trend:flat  sent:-0.10  P&L:—  entry:—
```

NCCI values ≥ 0.20 indicate meaningful news co-occurrence between two tickers.

---

### `/prompt <text>`
Update the agent's behavioural prompt. The LLM (`qwen2.5:3b`) automatically decides whether to **replace**, **append**, or **ignore** the new instruction by comparing it semantically to the current prompt.

| Decision | Condition | Effect |
|---|---|---|
| `replace` | New instruction supersedes the current one | Current prompt overwritten |
| `append` | New instruction adds complementary information | Text appended to current prompt |
| `ignore` | Semantically identical or redundant | No change, user notified |

**Examples:**

```
/prompt focus on renewable energy stocks
→ ✅ Prompt replaced.

/prompt avoid oil and gas
→ ✅ Prompt appended.

/prompt focus on renewable energy
→ ℹ️ Prompt unchanged — instruction already covered by the current prompt.
```

`/prompt` with no arguments shows the current prompt.

**LLM call:** `POST /api/generate` on `OLLAMA_SENTIMENT_MODEL` (`qwen2.5:3b`), 15s timeout, falls back to `replace` on error.

---

### `/modalita [id]`
View or change the active trading strategy.

`/modalita` with no argument lists all available strategies and the current one:

```
Current strategy: Contrarian

Available:
  • contrarian — Contrarian
  • trend_following — Trend Following
  • momentum — Momentum
  • value — Value
  • defensive — Defensive
  • scalping — Scalping

Usage: /modalita <id>
```

`/modalita momentum` switches immediately and calls `on_strategy_change("momentum")` in the agent loop.

---

### `/portfolio`
Live portfolio snapshot fetched in real time from Alpaca Paper Trading.

```
💼 Portfolio — 14:35 UTC
Total value: $102,450.00
Cash: $45,200.00
Session P&L: 📈 +1.24%

Open positions:
  AAPL  5 sh  entry $211.00  live $213.50  MV $1,067.50  P&L 📈 +1.18%
  MSFT  3 sh  entry $415.00  live $418.20  MV $1,254.60  P&L 📈 +0.77%
```

Live price is fetched individually per position for real-time unrealised P&L.

---

## Unknown commands

Any unrecognised command returns the full command list:

```
Available commands:
/resume — last cycle summary
/breaking — latest news per stock
/insider — disruptor (high-priority) news
/nerd — NCCI correlations and technical stats
/prompt <text> — replace agent prompt  |  /prompt +<text> append
/modalita — change strategy
/portfolio — portfolio status
```

---

## Lifecycle

| Event | Behaviour |
|---|---|
| `TELEGRAM_BOT_TOKEN=DISABLED` | Bot skipped entirely, no thread started |
| Agent start | `TelegramNotifier.start()` → daemon thread launched |
| Agent shutdown (Ctrl+C) | `TelegramNotifier.stop()` → polling stopped gracefully |
| Ollama unreachable on `/prompt` | Falls back to `replace` mode silently |
| Alpaca API error on `/portfolio` | Returns "API error: \<message\>" |

---

## Security notes

- The token is stored in `.env` which is listed in `.gitignore` — never committed.
- The bot only responds to the single `TELEGRAM_CHAT_ID` configured at startup; messages from other users are silently ignored by Telegram's routing (commands are processed only for the configured chat).
- `ALPACA_PAPER = True` is hardcoded in `broker.py` — no live trading is possible regardless of bot commands.

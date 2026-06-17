"""
Telegram notifier for BIP Trading Agent.

Runs the bot in a background thread (asyncio event loop isolated from the main thread).
The AgentLoop calls notify_action() and notify_cycle_end() to push data here;
Telegram commands are served from the stored state.

Message building is split into pure, side-effect-free functions (build_* below) so
they can be unit-tested for MarkdownV2 correctness without a live bot. Every dynamic
value is routed through _esc()/_num() because MarkdownV2 reserves '.', '+', '-', '='
and friends — a single unescaped char makes Telegram reject the whole message with
HTTP 400, which is why the numeric-heavy commands used to silently "do nothing".
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.agent import config


if TYPE_CHECKING:
    from src.agent.correlation_engine import CorrelationEngine
    from src.agent.disruptor import MarketDisruptor
    from src.agent.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


_ACTION_EMOJI = {
    "buy":  "🟢",
    "sell": "🔴",
    "hold": "🟡",
    "veto": "🚫",
    "wait": "⏳",
}

# MarkdownV2 reserved characters (Telegram Bot API).
_MD2_RESERVED = r"\_*[]()~`>#+-=|{}.!"


def _esc(text: str) -> str:
    """Escape every MarkdownV2 reserved character. Backslash is escaped first."""
    text = str(text)
    for ch in _MD2_RESERVED:
        text = text.replace(ch, f"\\{ch}")
    return text


def _num(value, spec: str = "") -> str:
    """Format a number and escape it for MarkdownV2 (handles '.', '+', '-', '%')."""
    try:
        return _esc(format(value, spec))
    except (ValueError, TypeError):
        return _esc(str(value))


def _strip_md(text: str) -> str:
    """Best-effort conversion of a MarkdownV2 string back to plain text (fallback path)."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])  # drop the escaping backslash
            i += 2
            continue
        if ch in "*_`":
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")


def _decide_prompt_mode(current: str, new_text: str) -> str:
    """Decide whether to replace, append, or ignore based on token overlap — no LLM call needed."""
    if not current.strip():
        return "replace"

    def _tokens(s: str) -> set[str]:
        return {w.lower().strip(".,!?;:") for w in s.split() if len(w) > 3}

    cur_tokens = _tokens(current)
    new_tokens = _tokens(new_text)
    if not new_tokens:
        return "ignore"

    overlap = len(cur_tokens & new_tokens) / len(new_tokens)
    if overlap >= 0.70:
        return "ignore"   # most words already in current prompt
    if overlap >= 0.30:
        return "append"   # partial overlap — add to existing
    return "replace"      # mostly new content — replace


# ──────────────────────────────────────────────────────────────────────────────
# Pure message builders (testable in isolation — no telegram, no network)
# ──────────────────────────────────────────────────────────────────────────────

def build_resume_text(
    cycle: int,
    rows: list[dict],
    portfolio: dict,
    pnl_pct: float,
    strategy_name: str,
    now_str: str,
) -> str:
    """MarkdownV2 summary of the last completed cycle."""
    if not rows:
        return "No cycle completed yet\\."

    pnl_sym = "📈" if pnl_pct >= 0 else "📉"
    lines = [
        f"*📊 Cycle {_num(cycle)} summary* \\— {_esc(now_str)}",
        f"Strategy: *{_esc(strategy_name)}*  \\|  P\\&L: {pnl_sym} *{_num(pnl_pct, '+.2%')}*",
        f"Portfolio: *${_num(portfolio.get('portfolio_value', 0.0), ',.2f')}*  "
        f"Cash: *${_num(portfolio.get('cash', 0.0), ',.2f')}*",
        "",
    ]
    for r in rows:
        emoji = _ACTION_EMOJI.get(str(r.get("action", "")).lower(), "❓")
        upnl = r.get("unrealized_pnl_pct")
        upnl_str = f" \\({_num(upnl, '+.1%')}\\)" if upnl is not None else ""
        lines.append(
            f"{emoji} *{_esc(r.get('ticker', '?'))}* → {_esc(str(r.get('action', '')).upper())}"
            f"  conf:{_num(r.get('conf', 0.0), '.0%')}  sent:{_num(r.get('sentiment_score', 0.0), '+.2f')}"
            f"  ${_num(r.get('price', 0.0), ',.2f')}{upnl_str}"
        )
        reasoning = (r.get("reasoning") or "")[:120]
        if reasoning:
            lines.append(f"   _{_esc(reasoning)}_")
    return "\n".join(lines)


def build_portfolio_text(data: dict, live_positions: dict, now_str: str) -> str:
    """
    MarkdownV2 live portfolio snapshot.

    live_positions: {symbol: {"live": float|None, "upnl_pct": float|None}} — optional
    per-position live price/unrealised P&L (None when unavailable).
    """
    pf_value  = data.get("portfolio_value", 0.0)
    cash      = data.get("cash", 0.0)
    pnl_pct   = data.get("pnl_pct", 0.0)
    positions = data.get("positions", {})

    pnl_sym = "📈" if pnl_pct >= 0 else "📉"
    lines = [
        f"*💼 Portfolio \\— {_esc(now_str)}*",
        f"Total value: *${_num(pf_value, ',.2f')}*",
        f"Cash: *${_num(cash, ',.2f')}*",
        f"Session P\\&L: {pnl_sym} *{_num(pnl_pct, '+.2%')}*",
        "",
    ]
    if positions:
        lines.append("*Open positions:*")
        for sym, pos in positions.items():
            qty   = pos.get("qty", 0)
            mv    = pos.get("market_value", 0.0)
            entry = pos.get("avg_entry_price", 0.0)
            lp = (live_positions or {}).get(sym, {})
            live = lp.get("live")
            upnl = lp.get("upnl_pct")
            if live is not None:
                price_str = f"${_num(live, ',.2f')}"
            else:
                price_str = "—"
            if upnl is not None:
                upnl_sym = "📈" if upnl >= 0 else "📉"
                upnl_str = f"{upnl_sym} {_num(upnl, '+.2%')}"
            else:
                upnl_str = "—"
            lines.append(
                f"  *{_esc(sym)}*  {_num(qty)} sh  entry ${_num(entry, ',.2f')}  "
                f"live {price_str}  MV ${_num(mv, ',.2f')}  P\\&L {upnl_str}"
            )
    else:
        lines.append("_No open positions\\._")
    return "\n".join(lines)


def build_nerd_text(
    cycle: int,
    tickers: list[str],
    rows: list[dict],
    ncci_matrix: dict[tuple[str, str], float],
) -> str:
    """
    MarkdownV2 technical-stats panel: NCCI correlation matrix + per-ticker snapshot.

    ncci_matrix: {(a, b): value} symmetric off-diagonal NCCI values.
    The matrix block is rendered inside inline-code spans, where '.'/'+'/'-' need NO
    escaping — only the snapshot lines (outside code spans) are escaped.
    """
    lines = [f"*🔬 Technical stats \\— cycle {_num(cycle)}*", ""]

    if len(tickers) >= 2:
        lines.append("*NCCI matrix \\(news correlation\\):*")
        # Header row (inside backticks → safe from MarkdownV2)
        lines.append("`     " + "  ".join(f"{t[:4]:>4}" for t in tickers) + "`")
        has_values = False
        for ta in tickers:
            row_vals = []
            for tb in tickers:
                if ta == tb:
                    row_vals.append(" 1.0")
                else:
                    v = ncci_matrix.get((ta, tb), ncci_matrix.get((tb, ta), 0.0)) or 0.0
                    if v:
                        has_values = True
                    row_vals.append(f"{v:+.2f}" if v else "  —")
            lines.append(f"`{ta[:4]:>4}  {'  '.join(row_vals)}`")
        if not has_values:
            lines.append("_\\(NCCI not yet computed — needs more cycles\\)_")
        lines.append("")

    if rows:
        lines.append("*Technical snapshot:*")
        for r in rows:
            upnl = r.get("unrealized_pnl_pct")
            upnl_str = _num(upnl, "+.1%") if upnl is not None else "—"
            entry = r.get("avg_entry_price")
            entry_str = f"${_num(entry, ',.2f')}" if entry else "—"
            rsi = r.get("rsi")
            rsi_str = _num(rsi, ".1f") if rsi is not None else "—"
            bb = r.get("bb_pct_b")
            bb_str = _num(bb, ".2f") if bb is not None else "—"
            lines.append(
                f"  *{_esc(r.get('ticker', '?'))}*  trend:{_esc(r.get('trend', '—'))}  "
                f"sent:{_num(r.get('sentiment_score', 0.0), '+.2f')}  "
                f"RSI:{rsi_str}  %B:{bb_str}  "
                f"P\\&L:{upnl_str}  entry:{entry_str}"
            )
    else:
        lines.append("_No cycle rows yet\\._")

    return "\n".join(lines)


def build_modalita_list_text(all_strats: dict, current_name: str) -> str:
    names = "\n".join(
        f"  • `{k}` \\— {_esc(v['name'])}" for k, v in all_strats.items()
    )
    return (
        f"*Current strategy:* {_esc(current_name)}\n\n"
        f"*Available:*\n{names}\n\n"
        f"Usage: `/modalita <id>`"
    )


def build_modalita_changed_text(name: str) -> str:
    return (
        f"✅ Strategy change queued → *{_esc(name)}*\n"
        f"_Takes effect at the start of the next cycle\\._"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot persistence (atomic .tmp + os.replace, never raises)
# ──────────────────────────────────────────────────────────────────────────────

def _save_snapshot(path: str, snapshot: dict) -> None:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Telegram snapshot save failed: %s", exc)


def _load_snapshot(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Telegram snapshot load failed: %s", exc)
        return {}


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str | int,
        tool_executor: "ToolExecutor | None" = None,
        disruptor: "MarketDisruptor | None" = None,
        correlation_engine: "CorrelationEngine | None" = None,
    ) -> None:
        self._token = token
        self._chat_id = int(chat_id)
        self._te = tool_executor
        self._disruptor = disruptor
        self._ce = correlation_engine

        # Mutable state shared between loop thread and bot handlers
        self._lock = threading.Lock()
        self._last_cycle_rows: list[dict] = []
        self._last_portfolio: dict = {}
        self._last_cycle: int = 0
        self._last_pnl_pct: float = 0.0
        self._last_strategy: str = "—"
        self._tickers: list[str] = []
        self._active_prompt: str = ""
        self._last_ncci: dict[tuple[str, str], float] = {}

        # Restore last known cycle from disk so /resume and /nerd answer immediately,
        # even before the first cycle of this run completes.
        self._restore_snapshot()

        # Callback injected by AgentLoop to change strategy / prompt.
        # Each returns True if the request was accepted/queued.
        self.on_strategy_change: "callable[[str], bool] | None" = None
        self.on_prompt_change: "callable[[str, str], bool] | None" = None  # (mode, text)

        self._app = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ──────────────────────────────────────────────
    # Called by AgentLoop (from loop thread)
    # ──────────────────────────────────────────────

    def notify_action(
        self,
        ticker: str,
        action: str,
        confidence: float,
        reasoning: str,
        caption: str,
        articles: list[dict],
        disruptor_articles: list[dict],
    ) -> None:
        emoji = _ACTION_EMOJI.get(action.lower(), "❓")
        lines: list[str] = [
            f"{emoji} *{_esc(ticker)}* → *{_esc(action.upper())}*  conf: {_num(confidence, '.0%')}",
            "",
            f"_{_esc(caption)}_" if caption else "",
            "",
            f"*Reasoning:* {_esc(reasoning[:300])}",
        ]

        if articles:
            lines += ["", "*📰 Supporting news:*"]
            for a in articles[:4]:
                title = _esc((a.get("title") or "")[:80])
                src   = _esc(a.get("source") or "")
                url   = a.get("url") or ""
                line  = f"  • [{title}]({url})  \\[{src}\\]" if url else f"  • {title}  \\[{src}\\]"
                lines.append(line)

        if disruptor_articles:
            lines += ["", "*⚡ Breaking \\(disruptor\\):*"]
            for a in disruptor_articles[:3]:
                title = _esc((a.get("title") or "")[:80])
                src   = _esc(a.get("source") or "")
                url   = a.get("url") or ""
                line  = f"  ⚡ [{title}]({url})  \\[{src}\\]" if url else f"  ⚡ {title}  \\[{src}\\]"
                lines.append(line)

        self._send("\n".join(l for l in lines if l is not None))

    def notify_prompt_applied(self, new_prompt: str) -> None:
        self._send(f"✅ *Prompt applied:*\n_{_esc(new_prompt[:300])}_")

    def notify_cycle_end(
        self,
        cycle: int,
        rows: list[dict],
        portfolio: dict,
        pnl_pct: float,
        strategy_name: str,
        active_prompt: str,
        tickers: list[str],
    ) -> None:
        # Snapshot the live NCCI matrix among current tickers so /nerd can show it
        # even after a restart, without waiting for the engine to recompute.
        ncci: dict[tuple[str, str], float] = {}
        if self._ce and len(tickers) >= 2:
            for i, a in enumerate(tickers):
                for b in tickers[i + 1:]:
                    try:
                        v = self._ce.get_ncci(a, b)
                    except Exception:
                        v = 0.0
                    ncci[(a, b)] = v

        with self._lock:
            self._last_cycle = cycle
            self._last_cycle_rows = list(rows)
            self._last_portfolio = dict(portfolio)
            self._last_pnl_pct = pnl_pct
            self._last_strategy = strategy_name
            self._active_prompt = active_prompt
            self._tickers = list(tickers)
            self._last_ncci = ncci

        self._persist_snapshot()

    def set_strategy_display(self, strategy_name: str) -> None:
        """Update the cached strategy name immediately (so /modalita reflects pending change)."""
        with self._lock:
            self._last_strategy = strategy_name

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────

    def _persist_snapshot(self) -> None:
        with self._lock:
            snapshot = {
                "cycle": self._last_cycle,
                "rows": self._last_cycle_rows,
                "portfolio": self._last_portfolio,
                "pnl_pct": self._last_pnl_pct,
                "strategy": self._last_strategy,
                "active_prompt": self._active_prompt,
                "tickers": self._tickers,
                # JSON cannot key on tuples — store NCCI as a list of triples.
                "ncci": [[a, b, v] for (a, b), v in self._last_ncci.items()],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        _save_snapshot(config.TELEGRAM_SNAPSHOT_PATH, snapshot)

    def _restore_snapshot(self) -> None:
        snap = _load_snapshot(config.TELEGRAM_SNAPSHOT_PATH)
        if not snap:
            return
        self._last_cycle = snap.get("cycle", 0)
        self._last_cycle_rows = snap.get("rows", []) or []
        self._last_portfolio = snap.get("portfolio", {}) or {}
        self._last_pnl_pct = snap.get("pnl_pct", 0.0)
        self._last_strategy = snap.get("strategy", "—")
        self._active_prompt = snap.get("active_prompt", "")
        self._tickers = snap.get("tickers", []) or []
        self._last_ncci = {
            (a, b): v for a, b, v in snap.get("ncci", []) if isinstance(v, (int, float))
        }

    # ──────────────────────────────────────────────
    # Bot lifecycle
    # ──────────────────────────────────────────────

    def start(self) -> None:
        if not self._token or self._token == "DISABLED":
            logger.info("TelegramNotifier: no token, bot disabled")
            return
        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="telegram-bot")
        self._thread.start()

    def stop(self) -> None:
        if self._app and self._loop and getattr(self, "_stop_event", None):
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ──────────────────────────────────────────────
    # Internal: send + run bot
    # ──────────────────────────────────────────────

    def _send(self, text: str) -> None:
        """Push a message (fire-and-forget) with a plain-text fallback on parse errors."""
        if not self._app or not self._loop:
            return

        async def _do_send() -> None:
            try:
                await self._app.bot.send_message(
                    chat_id=self._chat_id, text=text,
                    parse_mode="MarkdownV2", disable_web_page_preview=True,
                )
            except Exception as exc:
                logger.warning("Telegram push failed (%s) — retrying as plain text", exc)
                try:
                    await self._app.bot.send_message(
                        chat_id=self._chat_id, text=_strip_md(text),
                        disable_web_page_preview=True,
                    )
                except Exception as exc2:
                    logger.error("Telegram plain-text push also failed: %s", exc2)

        asyncio.run_coroutine_threadsafe(_do_send(), self._loop)

    def _run_bot(self) -> None:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        app = Application.builder().token(self._token).build()
        self._app = app

        async def _reply(update: "Update", text: str) -> None:
            """Reply with MarkdownV2; on any parse/format error, fall back to plain text."""
            try:
                await update.message.reply_text(
                    text, parse_mode="MarkdownV2", disable_web_page_preview=True,
                )
            except Exception as exc:
                logger.warning("Telegram reply failed (%s) — falling back to plain text", exc)
                try:
                    await update.message.reply_text(
                        _strip_md(text), disable_web_page_preview=True,
                    )
                except Exception as exc2:
                    logger.error("Telegram plain-text reply also failed: %s", exc2)

        def _authorized(update: "Update") -> bool:
            chat = update.effective_chat
            return chat is not None and chat.id == self._chat_id

        async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            with self._lock:
                rows  = list(self._last_cycle_rows)
                pf    = dict(self._last_portfolio)
                cycle = self._last_cycle
                pnl   = self._last_pnl_pct
                strat = self._last_strategy
            await _reply(update, build_resume_text(cycle, rows, pf, pnl, strat, _now()))

        async def cmd_breaking(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            with self._lock:
                tickers = list(self._tickers)
            if not tickers or not self._te:
                await _reply(update, "No active tickers\\.")
                return
            lines = [f"*📰 Latest news \\— {_esc(_now())}*", ""]
            for ticker in tickers:
                result = await asyncio.to_thread(self._te.get_news, ticker)
                articles = result.data.get("articles", []) if result.ok else []
                if not articles:
                    continue
                lines.append(f"*{_esc(ticker)}*")
                for a in articles[:3]:
                    title = _esc((a.get("title") or "")[:80])
                    src   = _esc(a.get("source") or "")
                    url   = a.get("url") or ""
                    line  = f"  • [{title}]({url})  \\[{src}\\]" if url else f"  • {title}  \\[{src}\\]"
                    lines.append(line)
                lines.append("")
            await _reply(update, "\n".join(lines) or "No news available\\.")

        async def cmd_insider(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            if not self._disruptor:
                await _reply(update, "Disruptor not active\\.")
                return
            with self._lock:
                tickers = list(self._tickers)
            lines = [f"*⚡ Disruptor news \\— {_esc(_now())}*", ""]
            seen: set[str] = set()
            for ticker in tickers:
                articles = self._disruptor.get_articles(ticker, max_age_seconds=3600)
                for a in articles:
                    title = (a.get("title") or "")
                    if title in seen:
                        continue
                    seen.add(title)
                    src   = _esc(a.get("source") or "")
                    url   = a.get("url") or ""
                    summary = _esc((a.get("summary") or "")[:120])
                    t_esc = _esc(title[:80])
                    line  = f"⚡ [{t_esc}]({url})  \\[{src}\\]" if url else f"⚡ {t_esc}  \\[{src}\\]"
                    lines.append(f"*{_esc(ticker)}*  {line}")
                    if summary:
                        lines.append(f"   _{summary}_")
            if len(lines) <= 2:
                lines.append("No recent disruptor news\\.")
            await _reply(update, "\n".join(lines))

        async def cmd_nerd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            with self._lock:
                tickers = list(self._tickers)
                rows    = list(self._last_cycle_rows)
                cycle   = self._last_cycle
                snap_ncci = dict(self._last_ncci)

            # Fall back to tickers from last cycle rows if notify_cycle_end not called yet
            if not tickers and rows:
                tickers = list({r["ticker"] for r in rows})

            if not rows and not tickers:
                await _reply(
                    update,
                    "No data yet \\— wait for the first cycle to complete\\.",
                )
                return

            # Prefer the live engine; fall back to the persisted NCCI matrix.
            ncci_matrix: dict[tuple[str, str], float] = {}
            for i, a in enumerate(tickers):
                for b in tickers[i + 1:]:
                    v = 0.0
                    if self._ce:
                        try:
                            v = self._ce.get_ncci(a, b)
                        except Exception:
                            v = 0.0
                    if not v:
                        v = snap_ncci.get((a, b), snap_ncci.get((b, a), 0.0))
                    ncci_matrix[(a, b)] = v

            await _reply(update, build_nerd_text(cycle, tickers, rows, ncci_matrix))

        async def cmd_prompt(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            args = ctx.args or []
            if not args:
                with self._lock:
                    cur = _esc(self._active_prompt[:200]) or "_\\(none\\)_"
                await _reply(
                    update,
                    f"*Current prompt:*\n_{cur}_\n\n"
                    f"`/prompt <text>` — auto replace\\/append\n"
                    f"`/prompt +<text>` — force append",
                )
                return
            text = " ".join(args).strip()

            # Explicit + prefix forces append, bypassing the auto-decision
            if text.startswith("+"):
                text = text[1:].strip()
                if not text:
                    await _reply(update, "Please provide a prompt text\\.")
                    return
                mode_flag = "a"
            else:
                with self._lock:
                    current = self._active_prompt
                decision = _decide_prompt_mode(current, text)
                if decision == "ignore":
                    await _reply(update, "ℹ️ Prompt unchanged — instruction already covered\\.")
                    return
                mode_flag = "a" if decision == "append" else "s"

            accepted = True
            if self.on_prompt_change:
                try:
                    res = self.on_prompt_change(mode_flag, text)
                    accepted = True if res is None else bool(res)
                except Exception as exc:
                    logger.error("on_prompt_change failed: %s", exc)
                    accepted = False
            else:
                accepted = False

            if not accepted:
                await _reply(
                    update,
                    "⚠️ Could not queue the prompt change \\(a change may already be pending\\)\\. "
                    "Retry in a moment\\.",
                )
                return

            action = "appended" if mode_flag == "a" else "replaced"
            await _reply(
                update,
                f"✅ Prompt {_esc(action)} \\(queued\\)\\.\n_{_esc(text[:200])}_\n"
                f"_The agent applies it at the next cycle; you'll get a confirmation\\._",
            )

        async def cmd_modalita(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            from src.agent import strategy_library
            all_strats = strategy_library.get_all()
            args = ctx.args or []
            if not args:
                with self._lock:
                    cur = self._last_strategy
                await _reply(update, build_modalita_list_text(all_strats, cur))
                return
            new_id = args[0].lower()
            if new_id not in all_strats:
                ids = ", ".join(f"`{k}`" for k in all_strats)
                await _reply(update, f"Strategy `{_esc(new_id)}` not found\\. Available: {ids}")
                return

            accepted = True
            if self.on_strategy_change:
                try:
                    res = self.on_strategy_change(new_id)
                    accepted = True if res is None else bool(res)
                except Exception as exc:
                    logger.error("on_strategy_change failed: %s", exc)
                    accepted = False
            else:
                accepted = False

            if not accepted:
                await _reply(update, "⚠️ Could not queue the strategy change\\. Retry in a moment\\.")
                return

            name = all_strats[new_id]["name"]
            # Reflect the pending strategy in cached state right away.
            self.set_strategy_display(name)
            await _reply(update, build_modalita_changed_text(name))

        async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            if not self._te:
                await _reply(update, "ToolExecutor not available\\.")
                return

            result = await asyncio.to_thread(self._te.get_portfolio)
            if not result.ok:
                await _reply(update, f"API error: {_esc(str(result.error))}")
                return

            data = result.data
            positions = data.get("positions", {})

            # Fetch live prices per position (off the event loop)
            live_positions: dict = {}
            for sym, pos in positions.items():
                entry = pos.get("avg_entry_price", 0.0)
                price_r = await asyncio.to_thread(self._te.get_price, sym)
                if price_r.ok and entry > 0:
                    live = price_r.data.get("price", entry)
                    live_positions[sym] = {"live": live, "upnl_pct": (live - entry) / entry}
                else:
                    live_positions[sym] = {"live": None, "upnl_pct": None}

            await _reply(update, build_portfolio_text(data, live_positions, _now()))

        async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not _authorized(update):
                return
            cmds = (
                "/resume — last cycle summary\n"
                "/breaking — latest news per stock\n"
                "/insider — disruptor (high-priority) news\n"
                "/nerd — NCCI correlations and technical stats\n"
                "/prompt <text> — replace agent prompt  |  /prompt +<text> append\n"
                "/modalita — change strategy\n"
                "/portfolio — portfolio status"
            )
            await update.message.reply_text(f"Available commands:\n{cmds}")

        app.add_handler(CommandHandler("resume",    cmd_resume))
        app.add_handler(CommandHandler("breaking",  cmd_breaking))
        app.add_handler(CommandHandler("insider",   cmd_insider))
        app.add_handler(CommandHandler("nerd",      cmd_nerd))
        app.add_handler(CommandHandler("prompt",    cmd_prompt))
        app.add_handler(CommandHandler("modalita",  cmd_modalita))
        app.add_handler(CommandHandler("portfolio", cmd_portfolio))
        app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))

        async def _run():
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)
            # keep alive until stop() is called
            stop_event = asyncio.Event()
            self._stop_event = stop_event
            await stop_event.wait()
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

        loop.run_until_complete(_run())

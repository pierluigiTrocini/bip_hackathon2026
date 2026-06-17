"""
Telegram notifier for BIP Trading Agent.

Runs the bot in a background thread (asyncio event loop isolated from the main thread).
The AgentLoop calls notify_action() and notify_cycle_end() to push data here;
Telegram commands are served from the stored state.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.agent.correlation_engine import CorrelationEngine
    from src.agent.disruptor import MarketDisruptor
    from src.agent.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


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

_ACTION_EMOJI = {
    "buy":  "🟢",
    "sell": "🔴",
    "hold": "🟡",
    "veto": "🚫",
    "wait": "⏳",
}


def _esc(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC")


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

        # Callback injected by AgentLoop to change strategy / prompt
        self.on_strategy_change: "callable[[str], None] | None" = None
        self.on_prompt_change: "callable[[str, str], None] | None" = None  # (mode, text)

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
            f"{emoji} *{_esc(ticker)}* → *{_esc(action.upper())}*  conf: {confidence:.0%}",
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
        self._send(
            f"✅ *Prompt applied:*\n_{_esc(new_prompt[:300])}_"
        )

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
        with self._lock:
            self._last_cycle = cycle
            self._last_cycle_rows = list(rows)
            self._last_portfolio = dict(portfolio)
            self._last_pnl_pct = pnl_pct
            self._last_strategy = strategy_name
            self._active_prompt = active_prompt
            self._tickers = list(tickers)

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
        if self._app and self._loop:
            asyncio.run_coroutine_threadsafe(self._app.stop(), self._loop)

    # ──────────────────────────────────────────────
    # Internal: send + run bot
    # ──────────────────────────────────────────────

    def _send(self, text: str) -> None:
        if not self._app or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            ),
            self._loop,
        )

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

        async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            with self._lock:
                rows = list(self._last_cycle_rows)
                pf   = dict(self._last_portfolio)
                cycle = self._last_cycle
                pnl   = self._last_pnl_pct
                strat = self._last_strategy

            if not rows:
                await update.message.reply_text("No cycle completed yet\\.", parse_mode="MarkdownV2")
                return

            pnl_sym = "📈" if pnl >= 0 else "📉"
            lines = [
                f"*📊 Cycle {cycle} summary* \\— {_esc(_now())}",
                f"Strategy: *{_esc(strat)}*  |  P\\&L: {pnl_sym} *{pnl:+.2%}*",
                f"Portfolio: *${pf.get('portfolio_value', 0):,.2f}*  Cash: *${pf.get('cash', 0):,.2f}*",
                "",
            ]
            for r in rows:
                emoji = _ACTION_EMOJI.get(r["action"].lower(), "❓")
                upnl = r.get("unrealized_pnl_pct")
                upnl_str = f" \\({upnl:+.1%}\\)" if upnl is not None else ""
                lines.append(
                    f"{emoji} *{_esc(r['ticker'])}* → {_esc(r['action'].upper())}"
                    f"  conf:{r['conf']:.0%}  sent:{r['sentiment_score']:+.2f}"
                    f"  \\${r['price']:,.2f}{upnl_str}"
                )
                lines.append(f"   _{_esc(r['reasoning'][:120])}_")
            await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2", disable_web_page_preview=True)

        async def cmd_breaking(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            with self._lock:
                tickers = list(self._tickers)
            if not tickers or not self._te:
                await update.message.reply_text("No active tickers\\.", parse_mode="MarkdownV2")
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
            await update.message.reply_text(
                "\n".join(lines) or "No news available\\.",
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

        async def cmd_insider(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not self._disruptor:
                await update.message.reply_text("Disruptor not active\\.", parse_mode="MarkdownV2")
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
            await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2", disable_web_page_preview=True)

        async def cmd_nerd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            with self._lock:
                tickers = list(self._tickers)
                rows    = list(self._last_cycle_rows)
                cycle   = self._last_cycle

            # Fall back to tickers from last cycle rows if notify_cycle_end not called yet
            if not tickers and rows:
                tickers = list({r["ticker"] for r in rows})

            if not rows and not tickers:
                await update.message.reply_text(
                    "No data yet \\— wait for the first cycle to complete\\.",
                    parse_mode="MarkdownV2",
                )
                return

            lines = [f"*🔬 Technical stats \\— cycle {cycle}*", ""]

            # NCCI matrix
            if self._ce and len(tickers) >= 2:
                lines.append("*NCCI matrix \\(news correlation\\):*")
                lines.append("`     " + "  ".join(f"{t[:4]:>4}" for t in tickers) + "`")
                has_values = False
                for ta in tickers:
                    row_vals = []
                    for tb in tickers:
                        if ta == tb:
                            row_vals.append(" 1.0")
                        else:
                            v = self._ce.get_ncci(ta, tb)
                            if v:
                                has_values = True
                            row_vals.append(f"{v:+.2f}" if v else "  —")
                    lines.append(f"`{ta[:4]:>4}  {'  '.join(row_vals)}`")
                if not has_values:
                    lines.append("_\\(NCCI not yet computed — needs more cycles\\)_")
                lines.append("")

            # Per-ticker technical snapshot
            if rows:
                lines.append("*Technical snapshot:*")
                for r in rows:
                    upnl = r.get("unrealized_pnl_pct")
                    upnl_str = f"{upnl:+.1%}" if upnl is not None else "—"
                    entry = r.get("avg_entry_price")
                    entry_str = f"\\${entry:,.2f}" if entry else "—"
                    lines.append(
                        f"  *{_esc(r['ticker'])}*  trend:{_esc(r.get('trend','—'))}  "
                        f"sent:{r.get('sentiment_score', 0.0):+.2f}  "
                        f"P\\&L:{_esc(upnl_str)}  entry:{_esc(entry_str)}"
                    )
            else:
                lines.append("_No cycle rows yet\\._")

            await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2", disable_web_page_preview=True)

        async def cmd_prompt(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            args = ctx.args or []
            if not args:
                with self._lock:
                    cur = _esc(self._active_prompt[:200])
                await update.message.reply_text(
                    f"*Current prompt:*\n_{cur}_\n\n`/prompt <text>` — auto replace\\/append\n`/prompt \\+<text>` — force append",
                    parse_mode="MarkdownV2",
                )
                return
            text = " ".join(args).strip()

            # Explicit + prefix forces append, bypassing the auto-decision
            if text.startswith("+"):
                text = text[1:].strip()
                if not text:
                    await update.message.reply_text("Please provide a prompt text\\.", parse_mode="MarkdownV2")
                    return
                mode_flag = "a"
            else:
                with self._lock:
                    current = self._active_prompt
                decision = _decide_prompt_mode(current, text)
                if decision == "ignore":
                    await update.message.reply_text(
                        "ℹ️ Prompt unchanged — instruction already covered\\.",
                        parse_mode="MarkdownV2",
                    )
                    return
                mode_flag = "a" if decision == "append" else "s"

            if self.on_prompt_change:
                self.on_prompt_change(mode_flag, text)
            action = "appended" if mode_flag == "a" else "replaced"
            await update.message.reply_text(
                f"✅ Prompt {_esc(action)}\\.\n_{_esc(text[:200])}_",
                parse_mode="MarkdownV2",
            )

        async def cmd_modalita(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            from src.agent import strategy_library
            all_strats = strategy_library.get_all()
            args = ctx.args or []
            if not args:
                names = "\n".join(f"  • `{k}` — {_esc(v['name'])}" for k, v in all_strats.items())
                with self._lock:
                    cur = _esc(self._last_strategy)
                await update.message.reply_text(
                    f"*Current strategy:* {cur}\n\n*Available:*\n{names}\n\nUsage: `/modalita <id>`",
                    parse_mode="MarkdownV2",
                )
                return
            new_id = args[0].lower()
            if new_id not in all_strats:
                ids = ", ".join(f"`{k}`" for k in all_strats)
                await update.message.reply_text(
                    f"Strategy `{_esc(new_id)}` not found\\. Available: {ids}",
                    parse_mode="MarkdownV2",
                )
                return
            if self.on_strategy_change:
                self.on_strategy_change(new_id)
            name = _esc(all_strats[new_id]["name"])
            await update.message.reply_text(
                f"✅ Strategy changed → *{name}*",
                parse_mode="MarkdownV2",
            )

        async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not self._te:
                await update.message.reply_text("ToolExecutor not available\\.", parse_mode="MarkdownV2")
                return

            result = await asyncio.to_thread(self._te.get_portfolio)
            if not result.ok:
                await update.message.reply_text(
                    f"API error: {_esc(str(result.error))}",
                    parse_mode="MarkdownV2",
                )
                return

            data      = result.data
            pf_value  = data.get("portfolio_value", 0.0)
            cash      = data.get("cash", 0.0)
            pnl_pct   = data.get("pnl_pct", 0.0)
            positions = data.get("positions", {})

            pnl_sym = "📈" if pnl_pct >= 0 else "📉"
            lines = [
                f"*💼 Portfolio \\— {_esc(_now())}*",
                f"Total value: *\\${pf_value:,.2f}*",
                f"Cash: *\\${cash:,.2f}*",
                f"Session P\\&L: {pnl_sym} *{pnl_pct:+.2%}*",
                "",
            ]
            if positions:
                lines.append("*Open positions:*")
                for sym, pos in positions.items():
                    qty   = pos.get("qty", 0)
                    mv    = pos.get("market_value", 0.0)
                    entry = pos.get("avg_entry_price", 0.0)
                    price_r = await asyncio.to_thread(self._te.get_price, sym)
                    if price_r.ok and entry > 0:
                        live = price_r.data.get("price", entry)
                        upnl_pct = (live - entry) / entry
                        upnl_sym = "📈" if upnl_pct >= 0 else "📉"
                        upnl_str = f"{upnl_sym} {upnl_pct:+.2%}"
                        price_str = f"\\${live:,.2f}"
                    else:
                        upnl_str = "—"
                        price_str = "—"
                    lines.append(
                        f"  *{_esc(sym)}*  {qty} sh  entry \\${entry:,.2f}  live {price_str}"
                        f"  MV \\${mv:,.2f}  P\\&L {upnl_str}"
                    )
            else:
                lines.append("_No open positions\\._")

            await update.message.reply_text(
                "\n".join(lines),
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

        async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            cmds = (
                "/resume — last cycle summary\n"
                "/breaking — latest news per stock\n"
                "/insider — disruptor (high-priority) news\n"
                "/nerd — NCCI correlations and technical stats\n"
                "/prompt <text> — replace agent prompt  |  /prompt +<text> append\n"
                "/modalita — change strategy\n"
                "/portfolio — portfolio status"
            )
            await update.message.reply_text(
                f"Available commands:\n{cmds}",
            )

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

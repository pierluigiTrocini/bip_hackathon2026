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
            f"*Ragionamento:* {_esc(reasoning[:300])}",
        ]

        if articles:
            lines += ["", "*📰 Notizie a supporto:*"]
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
                await update.message.reply_text("Nessun ciclo completato ancora\\.", parse_mode="MarkdownV2")
                return

            pnl_sym = "📈" if pnl >= 0 else "📉"
            lines = [
                f"*📊 Riepilogo ciclo {cycle}* \\— {_esc(_now())}",
                f"Strategia: *{_esc(strat)}*  |  P\\&L: {pnl_sym} *{pnl:+.2%}*",
                f"Portafoglio: *${pf.get('portfolio_value', 0):,.2f}*  Cash: *${pf.get('cash', 0):,.2f}*",
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
                await update.message.reply_text("Nessun ticker attivo\\.", parse_mode="MarkdownV2")
                return
            lines = [f"*📰 Ultime notizie \\— {_esc(_now())}*", ""]
            for ticker in tickers:
                result = self._te.get_news(ticker)
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
                "\n".join(lines) or "Nessuna notizia disponibile\\.",
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

        async def cmd_insider(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not self._disruptor:
                await update.message.reply_text("Disruptor non attivo\\.", parse_mode="MarkdownV2")
                return
            with self._lock:
                tickers = list(self._tickers)
            lines = [f"*⚡ Notizie Disruptor \\— {_esc(_now())}*", ""]
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
                lines.append("Nessuna notizia disruptor recente\\.")
            await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2", disable_web_page_preview=True)

        async def cmd_nerd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            with self._lock:
                tickers = list(self._tickers)
                rows    = list(self._last_cycle_rows)
            if not self._ce or not tickers:
                await update.message.reply_text("Dati tecnici non disponibili\\.", parse_mode="MarkdownV2")
                return
            lines = [f"*🔬 Stats tecniche \\— ciclo {self._last_cycle}*", ""]

            # NCCI matrix
            if len(tickers) >= 2:
                lines.append("*Matrice NCCI \\(correlazione news\\):*")
                lines.append("`          " + "  ".join(f"{t[:4]:>4}" for t in tickers) + "`")
                for ta in tickers:
                    row_vals = []
                    for tb in tickers:
                        if ta == tb:
                            row_vals.append("  1.0")
                        else:
                            v = self._ce.get_ncci(ta, tb)
                            row_vals.append(f"{v:+.2f}" if v else "   —")
                    lines.append(f"`{ta[:4]:>4}  {'  '.join(row_vals)}`")
                lines.append("")

            # Per-ticker technical snapshot
            if rows:
                lines.append("*Snapshot tecnico:*")
                for r in rows:
                    upnl = r.get("unrealized_pnl_pct")
                    upnl_str = f"{upnl:+.1%}" if upnl is not None else "—"
                    entry = r.get("avg_entry_price")
                    entry_str = f"${entry:,.2f}" if entry else "—"
                    lines.append(
                        f"  *{_esc(r['ticker'])}*  trend:{_esc(r['trend'])}  "
                        f"sent:{r['sentiment_score']:+.2f}  "
                        f"P\\&L:{_esc(upnl_str)}  entry:{_esc(entry_str)}"
                    )

            await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2", disable_web_page_preview=True)

        async def cmd_prompt(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            args = ctx.args or []
            if not args:
                with self._lock:
                    cur = _esc(self._active_prompt[:200])
                await update.message.reply_text(
                    f"*Prompt attuale:*\n_{cur}_\n\n"
                    "Uso: `/prompt a <testo>` aggiungi  |  `/prompt s <testo>` sostituisci  |  `/prompt i` ignora",
                    parse_mode="MarkdownV2",
                )
                return
            mode_flag = args[0].lower()
            text = " ".join(args[1:]).strip()
            if mode_flag == "i":
                await update.message.reply_text("✅ Input ignorato\\.", parse_mode="MarkdownV2")
                return
            if mode_flag not in ("a", "s") or not text:
                await update.message.reply_text(
                    "Formato: `/prompt a <testo>` \\(aggiungi\\) o `/prompt s <testo>` \\(sostituisci\\)",
                    parse_mode="MarkdownV2",
                )
                return
            mode_label = "aggiunto al" if mode_flag == "a" else "sostituito il"
            if self.on_prompt_change:
                self.on_prompt_change(mode_flag, text)
            await update.message.reply_text(
                f"✅ Prompt {_esc(mode_label)} prompt attuale\\.\n_{_esc(text[:200])}_",
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
                    f"*Strategia attuale:* {cur}\n\n*Disponibili:*\n{names}\n\nUso: `/modalita <id>`",
                    parse_mode="MarkdownV2",
                )
                return
            new_id = args[0].lower()
            if new_id not in all_strats:
                ids = ", ".join(f"`{k}`" for k in all_strats)
                await update.message.reply_text(
                    f"Strategia `{_esc(new_id)}` non trovata\\. Disponibili: {ids}",
                    parse_mode="MarkdownV2",
                )
                return
            if self.on_strategy_change:
                self.on_strategy_change(new_id)
            name = _esc(all_strats[new_id]["name"])
            await update.message.reply_text(
                f"✅ Strategia cambiata → *{name}*",
                parse_mode="MarkdownV2",
            )

        async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            if not self._te:
                await update.message.reply_text("ToolExecutor non disponibile\\.", parse_mode="MarkdownV2")
                return
            result = self._te.get_portfolio()
            if not result.ok:
                await update.message.reply_text(
                    f"Errore API: {_esc(str(result.error))}",
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
                f"Valore totale: *\\${pf_value:,.2f}*",
                f"Liquidità: *\\${cash:,.2f}*",
                f"P\\&L sessione: {pnl_sym} *{pnl_pct:+.2%}*",
                "",
            ]
            if positions:
                lines.append("*Posizioni aperte:*")
                for sym, pos in positions.items():
                    qty   = pos.get("qty", 0)
                    mv    = pos.get("market_value", 0.0)
                    entry = pos.get("avg_entry_price", 0.0)
                    # fetch live price for real-time P&L
                    price_r = self._te.get_price(sym)
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
                        f"  *{_esc(sym)}*  {qty} az  entry \\${entry:,.2f}  live {price_str}"
                        f"  MV \\${mv:,.2f}  P\\&L {upnl_str}"
                    )
            else:
                lines.append("_Nessuna posizione aperta\\._")

            await update.message.reply_text(
                "\n".join(lines),
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

        async def unknown_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            cmds = (
                "/resume — riepilogo ultimo ciclo\n"
                "/breaking — ultime notizie per ogni stock\n"
                "/insider — notizie disruptor\n"
                "/nerd — correlazioni e statistiche tecniche\n"
                "/prompt — cambia il prompt dell'agente\n"
                "/modalita — cambia strategia\n"
                "/portfolio — stato del portafoglio"
            )
            await update.message.reply_text(
                f"Comandi disponibili:\n{cmds}",
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

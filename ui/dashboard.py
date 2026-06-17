import concurrent.futures
import re
import sys
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_NY_TZ = ZoneInfo("America/New_York")

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from src.agent import config

_console = Console()

_STEP_COLORS = {
    "info":    "white",
    "cycle":   "bold cyan",
    "ok":      "cyan",
    "warn":    "yellow",
    "err":     "bold red",
    "action":  "bold dark_orange",
    "wait":    "dark_orange",
}


def _action_badge(action: str) -> str:
    badges = {
        "buy":  "[bold black on green] ▲ BUY  [/bold black on green]",
        "sell": "[bold white on red] ▼ SELL [/bold white on red]",
        "hold": "[bold white on yellow] ■ HOLD [/bold white on yellow]",
        "veto": "[bold white on magenta] ⊘ VETO [/bold white on magenta]",
    }
    return badges.get(action.lower(), f"[white]{action.upper()}[/white]")


def _now_ts() -> str:
    local = datetime.now().strftime("%H:%M:%S")
    ny = datetime.now(tz=_NY_TZ).strftime("%H:%M:%S")
    return f"{local} IT / {ny} NY"


class Dashboard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proposals: dict[str, dict] = {}
        self._portfolio: dict = {}
        self._journal_tail: list[dict] = []
        self._session_id: str = "--------"
        self._cycle: int = 0
        self._t_wait: int = 0
        self._t_behavior: int = 0
        self._api_lag_ms: float = 0.0
        self._interaction_in_progress: bool = False
        self._current_strategy: str = "contrarian"

    def set_interaction_in_progress(self, value: bool) -> None:
        with self._lock:
            self._interaction_in_progress = value

    def set_current_strategy(self, strategy_id: str) -> None:
        with self._lock:
            self._current_strategy = strategy_id

    def log(self, msg: str, level: str = "info") -> None:
        color = _STEP_COLORS.get(level, "white")
        _console.print(f"[dim]{_now_ts()}[/dim]  [{color}]{msg}[/{color}]")

    def update(self, ticker: str, entry: dict, portfolio: dict, t_wait: int, t_behavior: int) -> None:
        with self._lock:
            self._proposals[ticker] = entry
            self._portfolio = portfolio
            self._t_wait = t_wait
            self._t_behavior = t_behavior
            self._cycle = entry.get("cycle", self._cycle)
            self._session_id = entry.get("session_id", self._session_id)
            self._journal_tail.append(entry)
            if len(self._journal_tail) > 5:
                self._journal_tail = self._journal_tail[-5:]

    def _build_renderable(self, countdown: int) -> Panel:
        with self._lock:
            proposals = dict(self._proposals)
            portfolio = dict(self._portfolio)
            journal_tail = list(self._journal_tail)
            t_wait = self._t_wait
            t_behavior = self._t_behavior
            cycle = self._cycle
            session_id = self._session_id
            strategy = self._current_strategy

        pnl_pct = portfolio.get("pnl_pct", 0.0)
        mode_str = "[red]CONSERVATIVE[/red]" if pnl_pct < -0.05 else "[green]NORMAL[/green]"
        pnl_color = "green" if pnl_pct >= 0 else "red"
        pnl_arrow = "↑" if pnl_pct >= 0 else "↓"

        portfolio_table = Table(show_header=False, box=None, padding=(0, 1))
        portfolio_table.add_row("Cash:", f"${portfolio.get('cash', 0):,.2f}")
        portfolio_table.add_row("Valore:", f"${portfolio.get('portfolio_value', 0):,.2f}")
        portfolio_table.add_row(
            "P&L:",
            f"[{pnl_color}]{pnl_pct:+.2%} {pnl_arrow} [{mode_str}][/{pnl_color}]",
        )

        timeout_table = Table(show_header=False, box=None, padding=(0, 1))
        timeout_table.add_row("T_wait:", f"{t_wait}s")
        timeout_table.add_row("T_behavior:", f"{t_behavior}s")

        proposal_lines: list[str] = []
        for sym, entry in proposals.items():
            action = entry.get("action", "?")
            conf = entry.get("conf", 0.0)
            reasoning = entry.get("reasoning", "")[:70]
            proposal_lines.append(
                f"[bold white]{sym}[/bold white] → {_action_badge(action)} [dim]conf:{conf:.2f}[/dim]  {reasoning}"
            )
        proposals_text = "\n".join(proposal_lines) if proposal_lines else "Waiting for first cycle…"

        journal_lines: list[str] = []
        for e in journal_tail[-5:]:
            ts = e.get("ts", "")[-8:-3] if e.get("ts") else "--:--"
            sym = e.get("ticker", "?")
            act = e.get("action", "?")
            conf = e.get("conf", 0.0)
            sentiment = e.get("sentiment", 0.0)
            outcome = e.get("outcome_pct")
            outcome_col = "bright_green" if outcome and outcome > 0 else "red" if outcome and outcome < 0 else ""
            outcome_str = (
                f"[{outcome_col}]{outcome:+.1f}%[/{outcome_col}]" if outcome is not None
                else "[dim]pending[/dim]"
            )
            stale_str = " [bold yellow]⚠ STALE[/bold yellow]" if e.get("stale_penalty", 0) > 0 else ""
            journal_lines.append(
                f"[dim]{ts}[/dim] [bold white]{sym}[/bold white] {_action_badge(act)} [dim]{conf:.2f}  {sentiment:+.2f}[/dim]  {outcome_str}{stale_str}"
            )
        journal_text = "\n".join(journal_lines) if journal_lines else "No entries yet."

        filled = countdown * 20 // max(t_wait, 1)
        bar = "[dark_orange]" + "█" * filled + "[/dark_orange]" + "[dim]" + "░" * (20 - filled) + "[/dim]"
        countdown_line = (
            f"[bold dark_orange]⏳ {countdown}s[/bold dark_orange]  {bar}  │  "
            "[bold white]INVIO[/bold white] conferma · "
            "[dark_orange]a[/dark_orange] istruzione · "
            "[dark_orange]p[/dark_orange] prompt · "
            "[dark_orange]q[/dark_orange] questionario · "
            "[dark_orange]s[/dark_orange] strategia · "
            "[dark_orange]m[/dark_orange] override"
        )

        header = (
            f"[bold white]BIP Trading Agent[/bold white]  │  "
            f"[dim]Sessione: {str(session_id)[:8]}…  Ciclo: {cycle}[/dim]  │  "
            f"[bold dark_orange]{strategy}[/bold dark_orange]  │  [dim]{_now_ts()}[/dim]"
        )

        body = (
            f"[bold white]PORTFOLIO[/bold white]\n{_table_str(portfolio_table)}\n\n"
            f"[dim]T_wait:{t_wait}s  T_behavior:{t_behavior}s[/dim]\n\n"
            f"[bold white]PROPOSTA AGENTE[/bold white]\n{proposals_text}\n\n"
            f"[bold white]JOURNAL (ultimi 5)[/bold white]\n{journal_text}\n\n"
            f"{countdown_line}\n> "
        )

        return Panel(body, title=header, border_style="dark_orange")

    def wait_for_user_input(self, timeout_seconds: int) -> dict:
        with self._lock:
            _ip = self._interaction_in_progress
        if _ip:
            time.sleep(timeout_seconds)
            return {"source": "timeout", "data": {}}

        def _read_input():
            try:
                return input()
            except (EOFError, KeyboardInterrupt):
                return None

        user_input: str | None = None
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with Live(
                self._build_renderable(timeout_seconds),
                console=_console,
                refresh_per_second=1,
                transient=True,
            ) as live:
                fut = pool.submit(_read_input)
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    remaining = max(0, int(deadline - time.monotonic()))
                    live.update(self._build_renderable(remaining))
                    if fut.done():
                        break
                    time.sleep(0.5)
                try:
                    user_input = fut.result(timeout=0.1)
                except concurrent.futures.TimeoutError:
                    user_input = None
        finally:
            pool.shutdown(wait=False)

        if user_input is None:
            return {"source": "timeout", "data": {}}
        user_input = user_input.strip().lower()
        if user_input == "":
            return {"source": "confirmed", "data": {}}
        if user_input == "a":
            return {"source": "prompt_append", "data": {}}
        if user_input == "p":
            return {"source": "prompt_change", "data": {}}
        if user_input == "q":
            return {"source": "questionnaire", "data": {}}
        if user_input == "s":
            return {"source": "strategy_select", "data": {}}
        if user_input == "m":
            _console.print("[bold yellow]Override manuale[/bold yellow] — inserisci: TICKER SIDE QTY (es: AAPL buy 5) [dim](30s)[/dim]")
            _override_buf: list[str] = []
            def _get_override():
                try:
                    _override_buf.append(input("> ").strip())
                except (EOFError, KeyboardInterrupt):
                    pass
            _ot = threading.Thread(target=_get_override, daemon=True)
            _ot.start()
            _ot.join(timeout=30)
            raw = _override_buf[0].split() if _override_buf else []
            if len(raw) == 3:
                ticker, side, qty_str = raw
                try:
                    return {"source": "override", "data": {"action": side, "ticker": ticker.upper(), "qty": int(qty_str)}}
                except ValueError:
                    pass
            return {"source": "confirmed", "data": {}}
        return {"source": "confirmed", "data": {}}

    def interactive_input(self, prompt_text: str) -> str:
        _console.print(prompt_text)
        try:
            return input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def ask_strategy_switch(self, new_id: str, new_name: str, reason: str, timeout_seconds: int = 15) -> bool:
        """Show an auto-switch recommendation and ask for confirmation. Returns True if accepted."""
        _console.print(
            Panel(
                f"[bold yellow]L'agente suggerisce di cambiare strategia:[/bold yellow]\n\n"
                f"[dim]{reason}[/dim]\n\n"
                f"Nuova strategia: [bold cyan]{new_name}[/bold cyan]\n\n"
                f"[bold]s[/bold] accetta  ·  [bold]n[/bold] rifiuta  ·  "
                f"timeout {timeout_seconds}s → auto-accetta",
                title="[bold yellow]◆ CAMBIO STRATEGIA AUTOMATICO[/bold yellow]",
                border_style="yellow",
            )
        )

        _result: list[str] = []
        def _get_answer():
            try:
                _result.append(input("> ").strip().lower())
            except (EOFError, KeyboardInterrupt):
                pass

        t = threading.Thread(target=_get_answer, daemon=True)
        t.start()
        t.join(timeout=timeout_seconds)

        answer = _result[0] if _result else ""
        if answer == "n":
            _console.print("[dim]Cambio rifiutato — strategia invariata.[/dim]")
            return False
        _console.print(f"[green]✓ Strategia aggiornata a: {new_name}[/green]")
        return True

    def print_discovery_candidates(self, candidates: list[dict]) -> None:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 2),
        )
        table.add_column("#", justify="right", min_width=2, style="dim")
        table.add_column("Ticker", style="bold", min_width=8)
        table.add_column("Stato", min_width=10)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("Motivazione", min_width=55, no_wrap=False)

        valid_count = 0
        invalid_count = 0
        for i, c in enumerate(candidates, 1):
            valid = c.get("valid", True)
            conf = c.get("confidence", 0.0)
            conf_col = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
            original = c.get("original_ticker", c["ticker"])

            if valid:
                valid_count += 1
                remapped = original != c["ticker"]
                ticker_str = (
                    f"[green]{c['ticker']}[/green] [dim](← {original})[/dim]"
                    if remapped
                    else f"[green]{c['ticker']}[/green]"
                )
                status_str = "[green]✓ valido[/green]"
            else:
                invalid_count += 1
                ticker_str = f"[dark_orange]{c['ticker']}[/dark_orange]"
                status_str = "[dark_orange]✗ non trovato[/dark_orange]"
                conf_col = "dark_orange"

            table.add_row(
                str(i),
                ticker_str,
                status_str,
                f"[{conf_col}]{conf:.2f}[/{conf_col}]",
                c.get("reason", ""),
            )

        subtitle = (
            f"[dim]Basato sul tuo prompt e sulle news di mercato — "
            f"[green]{valid_count} validi[/green]"
            + (f"  [dark_orange]{invalid_count} non trovati su Alpaca[/dark_orange]" if invalid_count else "")
            + "[/dim]"
        )
        _console.print(
            Panel(
                table,
                title="[bold magenta]◆ DISCOVERY — Ticker candidati[/bold magenta]",
                subtitle=subtitle,
                border_style="magenta",
                padding=(1, 2),
            )
        )

    def print_portfolio_positions(self, positions: dict) -> None:
        if not positions:
            _console.print(
                Panel(
                    "[dim]Nessuna posizione aperta nel portfolio.[/dim]",
                    title="[bold cyan]◆ PORTFOLIO — Posizioni aperte[/bold cyan]",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Ticker", style="bold", min_width=8)
        table.add_column("Qtà", justify="right", min_width=6)
        table.add_column("Prezzo medio", justify="right", min_width=14)
        table.add_column("Valore mercato", justify="right", min_width=16)

        total_value = 0.0
        for ticker, pos in sorted(positions.items()):
            qty = pos.get("qty", 0)
            avg = pos.get("avg_entry_price", 0.0)
            mktval = pos.get("market_value", 0.0)
            total_value += mktval
            table.add_row(
                f"[cyan]{ticker}[/cyan]",
                str(qty),
                f"${avg:,.2f}",
                f"${mktval:,.2f}",
            )

        subtitle = f"[dim]Valore totale posizioni: [bold]${total_value:,.2f}[/bold][/dim]"
        _console.print(
            Panel(
                table,
                title="[bold cyan]◆ PORTFOLIO — Posizioni aperte[/bold cyan]",
                subtitle=subtitle,
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def confirm_or_reprompt(self, candidates: list[dict], timeout_seconds: int = 30) -> dict:
        default = [c["ticker"] for c in candidates if c.get("valid", True)]
        default_str = ", ".join(default)

        def _build_confirm_panel(remaining: int) -> Panel:
            bar_filled = remaining * 20 // max(timeout_seconds, 1)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            return Panel(
                f"[bold]Ticker proposti:[/bold] [cyan]{default_str}[/cyan]\n\n"
                f"⏳ {remaining}s [{bar}] │ "
                "[bold]INVIO[/bold] conferma · "
                "[bold cyan]testo+INVIO[/bold cyan] nuovo prompt · "
                "[dim]timeout = auto-conferma[/dim]\n> ",
                title="[bold magenta]Conferma Discovery[/bold magenta]",
                border_style="magenta",
            )

        def _read_input():
            try:
                return input()
            except (EOFError, KeyboardInterrupt):
                return None

        user_input: str | None = None
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with Live(
                _build_confirm_panel(timeout_seconds),
                console=_console,
                refresh_per_second=1,
                transient=True,
            ) as live:
                fut = pool.submit(_read_input)
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    remaining = max(0, int(deadline - time.monotonic()))
                    live.update(_build_confirm_panel(remaining))
                    if fut.done():
                        break
                    time.sleep(0.5)
                try:
                    user_input = fut.result(timeout=0.1)
                except concurrent.futures.TimeoutError:
                    user_input = None
        finally:
            pool.shutdown(wait=False)

        if user_input is None:
            _console.print(f"[green]✓ Auto-confermati (timeout):[/green] {default_str}")
            return {"action": "confirm", "tickers": default}

        raw = user_input.strip()
        if not raw:
            _console.print(f"[green]✓ Confermati:[/green] {default_str}")
            return {"action": "confirm", "tickers": default}

        _console.print(f"[yellow]Nuovo prompt ricevuto — riavvio discovery:[/yellow] {raw}")
        return {"action": "reprompt", "new_prompt": raw}

    def print_cycle_summary(
        self,
        cycle: int,
        rows: list[dict],
        pnl_pct: float,
        portfolio_value: float,
        cash: float,
        mode: str,
        wait_seconds: int,
        veto: bool,
        strategy_name: str = "",
    ) -> None:
        table = Table(
            show_header=True,
            header_style="bold white",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Ticker", style="bold white", min_width=6)
        table.add_column("Prezzo", justify="right", min_width=10)
        table.add_column("Trend", min_width=5)
        table.add_column("Sentiment", min_width=16)
        table.add_column("Decisione", min_width=14)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("P&L pos.", justify="right", min_width=9)
        table.add_column("Ordine", min_width=8)
        table.add_column("Perché", min_width=45, no_wrap=False)

        trend_symbol = {"up": "[bright_green]↑[/bright_green]", "down": "[bright_red]↓[/bright_red]", "flat": "[dim]→[/dim]"}
        sentiment_color = {
            "positive":      "bright_green",
            "very_positive": "bold bright_green",
            "negative":      "bright_red",
            "very_negative": "bold bright_red",
            "neutral":       "white",
        }

        for r in rows:
            act = r["action"]
            stale_tag = " [bold yellow]⚠[/bold yellow]" if r.get("stale") else ""
            price_str = f"[white]${r['price']:,.2f}[/white]{stale_tag}"
            trend_s = trend_symbol.get(r["trend"], r["trend"])
            sent_label = r["sentiment_label"]
            sent_score = r["sentiment_score"]
            sent_col = sentiment_color.get(sent_label, "white")
            order_str = "[bright_green]✓ inviato[/bright_green]" if r.get("order_id") else "[dim]—[/dim]"

            upnl = r.get("unrealized_pnl_pct")
            if upnl is not None:
                upnl_col = "bright_green" if upnl >= 0.02 else "bright_red" if upnl <= -0.03 else "white"
                upnl_str = f"[{upnl_col}]{upnl:+.1%}[/{upnl_col}]"
            else:
                upnl_str = "[dim]—[/dim]"

            reasoning_text = (r["reasoning"] or "")
            reasoning_short = "[dim]" + reasoning_text[:78] + ("…" if len(reasoning_text) > 78 else "") + "[/dim]"

            table.add_row(
                r["ticker"],
                price_str,
                trend_s,
                f"[{sent_col}]{sent_label} ({sent_score:+.2f})[/{sent_col}]",
                _action_badge(act),
                f"[white]{r['conf']:.2f}[/white]",
                upnl_str,
                order_str,
                reasoning_short,
            )

        pnl_col = "bright_green" if pnl_pct >= 0 else "bright_red"
        mode_str = "[bold red]● CONSERVATIVE[/bold red]" if mode == "conservative" else "[bright_green]● NORMAL[/bright_green]"
        veto_str = "  [bold red]⊘ NEWS VETO[/bold red]" if veto else ""
        strat_str = f"  [bold dark_orange]{strategy_name}[/bold dark_orange]" if strategy_name else ""
        footer = (
            f"Portfolio: [bold white]${portfolio_value:,.2f}[/bold white]  "
            f"Cash: [white]${cash:,.2f}[/white]  "
            f"P&L: [{pnl_col}]{pnl_pct:+.2%}[/{pnl_col}]  "
            f"{mode_str}{strat_str}{veto_str}\n"
            f"[dim]Prossimo ciclo tra [/dim][bold dark_orange]{wait_seconds}s[/bold dark_orange][dim] — INVIO · a · p · q · s · m[/dim]"
        )

        _console.print(
            Panel(
                table if rows else Text("Nessun ticker elaborato in questo ciclo.", style="dim"),
                title=f"[bold white]◆ CICLO {cycle} — RIEPILOGO[/bold white]",
                subtitle=footer,
                border_style="white",
                padding=(1, 2),
            )
        )

    def print_decision_news(
        self,
        ticker: str,
        action: str,
        confidence: float,
        caption: str,
        articles: list[dict],
    ) -> None:
        """
        Print a Rich Panel immediately after a decision, outside the Live context.
        Uses Console.print() directly — never wraps inside Live.
        """
        border_colors = {
            "buy":  "green",
            "sell": "red",
            "hold": "yellow",
            "veto": "magenta",
        }
        border = border_colors.get(action.lower(), "white")
        badge = _action_badge(action)
        title = f"[bold white]{ticker}[/bold white]  {badge}  [dim]conf: {confidence:.2f}[/dim]"

        lines: list[str] = []

        if caption:
            wrapped = caption[:160]
            lines.append(f"[white]💬 {wrapped}[/white]")
        else:
            lines.append("[dim]Nessuna spiegazione disponibile.[/dim]")

        if articles:
            lines.append("")
            lines.append("[bold white]Notizie a supporto:[/bold white]")
            for i, art in enumerate(articles):
                if i > 0:
                    lines.append("")
                source = str(art.get("source", ""))
                title_art = str(art.get("title", ""))
                url = str(art.get("url", ""))

                src_padded = source[:12].ljust(12)
                title_truncated = title_art[:70] + ("…" if len(title_art) > 70 else "")
                lines.append(f"[bold white]{src_padded}[/bold white]  [white]{title_truncated}[/white]")
                if url:
                    lines.append(f"{'':14}[cyan]{url}[/cyan]")
        else:
            lines.append("")
            lines.append("[dim]Nessuna notizia disponibile per questo ciclo.[/dim]")

        body = "\n".join(lines)
        _console.print(
            Panel(
                body,
                title=title,
                border_style=border,
                padding=(1, 2),
            )
        )

    def print_correlation_matrix(self, tickers: list[str], engine) -> None:
        """
        Print an NCCI correlation matrix for the given tickers.
        engine: CorrelationEngine instance (duck-typed to avoid circular import).
        Skipped when fewer than 2 tickers are provided.
        """
        if len(tickers) < 2:
            return

        # Build symmetric lookup and check if any non-zero value exists
        ncci: dict[tuple[str, str], float] = {}
        has_data = False
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                v = engine.get_ncci(a, b)
                ncci[(a, b)] = v
                ncci[(b, a)] = v
                if v > 0.0:
                    has_data = True

        with self._lock:
            cycle = self._cycle

        if not has_data:
            _console.print(
                Panel(
                    "[dim]Correlazioni non ancora disponibili — news insufficienti per calcolare NCCI.[/dim]",
                    title=f"[bold dim]◆ NCCI — Ciclo {cycle}[/bold dim]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
        )
        table.add_column("", style="bold", min_width=6)
        for t in tickers:
            table.add_column(t, justify="center", min_width=6)

        for row_t in tickers:
            cells: list[str] = [f"[bold]{row_t}[/bold]"]
            for col_t in tickers:
                if row_t == col_t:
                    cells.append("[dim]━━━[/dim]")
                else:
                    v = ncci.get((row_t, col_t), 0.0)
                    if v >= 0.5:
                        cells.append(f"[bold red]{v:.2f}[/bold red]")
                    elif v >= 0.3:
                        cells.append(f"[yellow]{v:.2f}[/yellow]")
                    elif v >= config.NCCI_THRESHOLD_DISPLAY:
                        cells.append(f"[white]{v:.2f}[/white]")
                    else:
                        cells.append(f"[dim]{v:.2f}[/dim]")
            table.add_row(*cells)

        subtitle = (
            f"[dim]"
            f"[bold red]rosso[/bold red] ≥ 0.50  "
            f"[yellow]giallo[/yellow] ≥ 0.30  "
            f"bianco ≥ {config.NCCI_THRESHOLD_DISPLAY:.2f}  "
            f"[dim]grigio[/dim] = trascurabile"
            f"[/dim]"
        )
        _console.print(
            Panel(
                table,
                title=f"[bold dim]◆ NCCI — Correlazioni ticker — Ciclo {cycle}[/bold dim]",
                subtitle=subtitle,
                border_style="dim",
                padding=(1, 2),
            )
        )

    def print_resoconto(self, summary: dict) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        pnl = summary.get("final_pnl_pct")
        pnl_str = f"{pnl:+.2%}" if pnl is not None else "N/A"
        decisions = summary.get("decisions", {})
        rows = [
            ("ID", f"{str(summary.get('session_id',''))[:8]}…"),
            ("Cicli", str(summary.get("cycles", 0))),
            ("Ordini", str(summary.get("orders_placed", 0))),
            ("P&L", pnl_str),
            ("Decisioni autonome", str(summary.get("autonomous_decisions", 0))),
            ("Errori loggati", str(summary.get("errors", 0))),
            ("Buy/Sell/Hold", f"{decisions.get('buy',0)}/{decisions.get('sell',0)}/{decisions.get('hold',0)}"),
        ]
        for label, value in rows:
            table.add_row(f"[bold]{label}:[/bold]", value)
        _console.print(Panel(table, title="[yellow]RESOCONTO SESSIONE[/yellow]", border_style="yellow"))

    def print_shutdown_message(self) -> None:
        _console.print(Panel(
            "[bold green]Agente fermato. Sessione salvata.[/bold green]",
            title="[red]SHUTDOWN[/red]",
            border_style="red",
        ))


def _table_str(table: Table) -> str:
    from io import StringIO
    buf = StringIO()
    c = Console(file=buf, highlight=False)
    c.print(table)
    return buf.getvalue().strip()

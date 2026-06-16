import concurrent.futures
import re
import sys
import threading
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

_console = Console()

_STEP_COLORS = {
    "info":    "cyan",
    "ok":      "green",
    "warn":    "yellow",
    "err":     "red",
    "action":  "bold magenta",
    "wait":    "dim",
}


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


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
            action = entry.get("action", "?").upper()
            conf = entry.get("conf", 0.0)
            reasoning = entry.get("reasoning", "")[:70]
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")
            proposal_lines.append(
                f"[{color}]{sym} → {action}[/{color}] (conf: {conf:.2f}) {reasoning}"
            )
        proposals_text = "\n".join(proposal_lines) if proposal_lines else "Waiting for first cycle…"

        journal_lines: list[str] = []
        for e in journal_tail[-5:]:
            ts = e.get("ts", "")[-8:-3] if e.get("ts") else "--:--"
            sym = e.get("ticker", "?")
            act = e.get("action", "?").upper()
            conf = e.get("conf", 0.0)
            sentiment = e.get("sentiment", 0.0)
            outcome = e.get("outcome_pct")
            outcome_str = f"outcome:{outcome:+.1f}%" if outcome is not None else "[dim]pending[/dim]"
            stale_str = " [yellow][STALE][/yellow]" if e.get("stale_penalty", 0) > 0 else ""
            act_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(act, "white")
            journal_lines.append(
                f"[{ts}] {sym} [{act_color}]{act}[/{act_color}] {conf:.2f}  {sentiment:+.2f}  {outcome_str}{stale_str}"
            )
        journal_text = "\n".join(journal_lines) if journal_lines else "No entries yet."

        bar = "█" * (countdown * 20 // max(t_wait, 1)) + "░" * (20 - countdown * 20 // max(t_wait, 1))
        countdown_line = (
            f"⏳ {countdown}s [{bar}] │ "
            "[bold]INVIO[/bold] conferma · "
            "[bold cyan]a[/bold cyan] istruzione · "
            "[bold cyan]p[/bold cyan] prompt · "
            "[bold cyan]q[/bold cyan] questionario · "
            "[bold cyan]s[/bold cyan] strategia · "
            "[bold yellow]m[/bold yellow] override"
        )

        header = (
            f"[bold]BIP Trading Agent[/bold]  │  "
            f"Sessione: {str(session_id)[:8]}…  │  "
            f"Ciclo: {cycle}  │  "
            f"Strategia: [bold cyan]{strategy}[/bold cyan]  │  {_now_ts()}"
        )

        body = (
            f"[bold cyan]PORTFOLIO[/bold cyan]\n{_table_str(portfolio_table)}\n\n"
            f"[bold cyan]TIMEOUT[/bold cyan]  T_wait:{t_wait}s  T_behavior:{t_behavior}s\n\n"
            f"[bold cyan]PROPOSTA AGENTE[/bold cyan]\n{proposals_text}\n\n"
            f"[bold cyan]JOURNAL (ultimi 5)[/bold cyan]\n{journal_text}\n\n"
            f"{countdown_line}\n> "
        )

        return Panel(body, title=header, border_style="blue")

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
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Prezzo", justify="right", min_width=10)
        table.add_column("Trend", min_width=5)
        table.add_column("Sentiment", min_width=16)
        table.add_column("Decisione", min_width=14)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("P&L pos.", justify="right", min_width=9)
        table.add_column("Ordine", min_width=8)
        table.add_column("Perché", min_width=45, no_wrap=False)

        action_color = {"buy": "green", "sell": "red", "hold": "yellow"}
        trend_symbol = {"up": "↑", "down": "↓", "flat": "→"}
        sentiment_color = {
            "positive": "green", "negative": "red",
            "neutral": "white", "very_negative": "red", "very_positive": "green",
        }

        for r in rows:
            act = r["action"]
            act_col = action_color.get(act, "white")
            stale_tag = " [STALE]" if r.get("stale") else ""
            price_str = f"${r['price']:,.2f}{stale_tag}"
            trend_s = trend_symbol.get(r["trend"], r["trend"])
            sent_label = r["sentiment_label"]
            sent_score = r["sentiment_score"]
            sent_col = sentiment_color.get(sent_label, "white")
            order_str = "✓ inviato" if r.get("order_id") else "—"

            # Unrealized P&L display
            upnl = r.get("unrealized_pnl_pct")
            if upnl is not None:
                upnl_col = "green" if upnl >= 0.02 else "red" if upnl <= -0.03 else "white"
                upnl_str = f"[{upnl_col}]{upnl:+.1%}[/{upnl_col}]"
            else:
                upnl_str = "[dim]—[/dim]"

            # Reasoning — show up to 80 chars in table
            reasoning_text = (r["reasoning"] or "")
            reasoning_short = reasoning_text[:78] + ("…" if len(reasoning_text) > 78 else "")

            table.add_row(
                r["ticker"],
                price_str,
                trend_s,
                f"[{sent_col}]{sent_label} ({sent_score:+.2f})[/{sent_col}]",
                f"[{act_col}]{act.upper()}[/{act_col}]",
                f"{r['conf']:.2f}",
                upnl_str,
                order_str,
                reasoning_short,
            )

        pnl_col = "green" if pnl_pct >= 0 else "red"
        mode_str = "[red]CONSERVATIVE[/red]" if mode == "conservative" else "[green]NORMAL[/green]"
        veto_str = "  [red][NEWS VETO][/red]" if veto else ""
        strat_str = f"  Strategia: [bold cyan]{strategy_name}[/bold cyan]" if strategy_name else ""
        footer = (
            f"Portfolio: [bold]${portfolio_value:,.2f}[/bold]  "
            f"Cash: ${cash:,.2f}  "
            f"P&L: [{pnl_col}]{pnl_pct:+.2%}[/{pnl_col}]  "
            f"Modalità: {mode_str}{strat_str}{veto_str}\n"
            f"[dim]Prossimo ciclo tra {wait_seconds}s — INVIO · [a] istruzione · [p] prompt · [q] questionario · [s] strategia · [m] override[/dim]"
        )

        _console.print(
            Panel(
                table if rows else Text("Nessun ticker elaborato in questo ciclo.", style="dim"),
                title=f"[bold blue]◆ CICLO {cycle} — RIEPILOGO[/bold blue]",
                subtitle=footer,
                border_style="blue",
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

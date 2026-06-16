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

        pnl_pct = portfolio.get("pnl_pct", 0.0)
        mode_str = "[red]CONSERVATIVE[/red]" if pnl_pct < -0.05 else "[green]NORMAL[/green]"
        pnl_color = "green" if pnl_pct >= 0 else "red"
        pnl_arrow = "↑" if pnl_pct >= 0 else "↓"

        # Portfolio section
        portfolio_table = Table(show_header=False, box=None, padding=(0, 1))
        portfolio_table.add_row("Cash:", f"${portfolio.get('cash', 0):,.2f}")
        portfolio_table.add_row("Valore:", f"${portfolio.get('portfolio_value', 0):,.2f}")
        portfolio_table.add_row(
            "P&L:",
            f"[{pnl_color}]{pnl_pct:+.2%} {pnl_arrow} [{mode_str}][/{pnl_color}]",
        )

        # Timeout section
        timeout_table = Table(show_header=False, box=None, padding=(0, 1))
        timeout_table.add_row("T_wait:", f"{t_wait}s")
        timeout_table.add_row("T_behavior:", f"{t_behavior}s")

        # Proposals section
        proposal_lines: list[str] = []
        for sym, entry in proposals.items():
            action = entry.get("action", "?").upper()
            conf = entry.get("conf", 0.0)
            reasoning = entry.get("reasoning", "")[:60]
            sentiment = entry.get("sentiment", 0.0)
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")
            proposal_lines.append(
                f"[{color}]{sym} → {action}[/{color}] (conf: {conf:.2f}, {reasoning})"
            )
        proposals_text = "\n".join(proposal_lines) if proposal_lines else "Waiting for first cycle…"

        # Journal tail
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

        # Countdown bar
        bar = "█" * (countdown * 20 // max(t_wait, 1)) + "░" * (20 - countdown * 20 // max(t_wait, 1))
        countdown_line = f"⏳ {countdown}s [{bar}] │ [INVIO] conferma │ [m] modifica │ [c] cambia"

        header = (
            f"[bold]BIP Trading Agent[/bold]  │  "
            f"Sessione: {str(session_id)[:8]}…  │  Ciclo: {cycle}  │  {_now_ts()}"
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
        result: dict = {"source": "timeout", "data": {}}

        def _read_input():
            try:
                return input()
            except (EOFError, KeyboardInterrupt):
                return None

        with Live(
            self._build_renderable(timeout_seconds),
            console=_console,
            refresh_per_second=1,
            transient=True,
        ) as live:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
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
                    fut.cancel()
                    user_input = None

        if user_input is None:
            return {"source": "timeout", "data": {}}
        user_input = user_input.strip()
        if user_input == "":
            return {"source": "confirmed", "data": {}}
        if user_input.lower() == "m":
            _console.print("Override manuale — inserisci: TICKER SIDE QTY (es: AAPL buy 5)")
            raw = input("> ").strip().split()
            if len(raw) == 3:
                ticker, side, qty_str = raw
                try:
                    return {"source": "override", "data": {"action": side, "ticker": ticker, "qty": int(qty_str)}}
                except ValueError:
                    pass
            return {"source": "confirmed", "data": {}}
        if user_input.lower() == "c":
            _console.print("Inserisci il nuovo comportamento dell'agente:")
            new_prompt = input("> ").strip()
            if new_prompt:
                return {"source": "behavior_change", "data": {"new_prompt": new_prompt}}
            return {"source": "confirmed", "data": {}}
        return {"source": "confirmed", "data": {}}

    def print_discovery_candidates(self, candidates: list[dict]) -> None:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 2),
        )
        table.add_column("#", justify="right", min_width=2, style="dim")
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("Motivazione", min_width=60, no_wrap=False)

        for i, c in enumerate(candidates, 1):
            conf = c.get("confidence", 0.0)
            conf_col = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
            table.add_row(
                str(i),
                c["ticker"],
                f"[{conf_col}]{conf:.2f}[/{conf_col}]",
                c.get("reason", ""),
            )

        _console.print(
            Panel(
                table,
                title="[bold magenta]◆ DISCOVERY — Ticker candidati[/bold magenta]",
                subtitle="[dim]Basato sul tuo prompt e sulle news di mercato[/dim]",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    def confirm_tickers(self, candidates: list[dict]) -> list[str]:
        """
        Ask the user to confirm the proposed ticker list or provide a custom one.
        Returns the final list of ticker strings.
        """
        default = [c["ticker"] for c in candidates]
        default_str = ", ".join(default)
        _console.print(
            f"\n[bold]Ticker selezionati dall'agente:[/bold] [cyan]{default_str}[/cyan]\n"
            f"  [dim]Premi INVIO per confermare, oppure inserisci una lista personalizzata "
            f"(es: AAPL MSFT TSLA):[/dim]"
        )
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""

        if not raw:
            _console.print(f"[green]✓ Confermati:[/green] {default_str}")
            return default

        # Parse space- or comma-separated uppercase tickers
        tokens = re.split(r"[\s,]+", raw.upper())
        custom = [t for t in tokens if re.fullmatch(r"[A-Z]{1,5}", t)]
        if not custom:
            _console.print(f"[yellow]Input non valido — uso la lista proposta: {default_str}[/yellow]")
            return default

        result_str = ", ".join(custom)
        _console.print(f"[green]✓ Lista personalizzata:[/green] {result_str}")
        return custom

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
    ) -> None:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Prezzo", justify="right", min_width=10)
        table.add_column("Trend", min_width=6)
        table.add_column("Sentiment", min_width=16)
        table.add_column("Decisione", min_width=16)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("Ordine", min_width=8)
        table.add_column("Ragionamento (sintesi)", min_width=40, no_wrap=True)

        action_color = {"buy": "green", "sell": "red", "hold": "yellow"}
        trend_symbol = {"up": "↑", "down": "↓", "flat": "→"}
        sentiment_color = {"positive": "green", "negative": "red", "neutral": "white", "very_negative": "red", "very_positive": "green"}

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
            reasoning_short = (r["reasoning"] or "")[:55]
            if len(r["reasoning"] or "") > 55:
                reasoning_short += "…"

            table.add_row(
                r["ticker"],
                price_str,
                trend_s,
                f"[{sent_col}]{sent_label} ({sent_score:+.2f})[/{sent_col}]",
                f"[{act_col}]{act.upper()}[/{act_col}]",
                f"{r['conf']:.2f}",
                order_str,
                reasoning_short,
            )

        pnl_col = "green" if pnl_pct >= 0 else "red"
        mode_str = f"[red]CONSERVATIVE[/red]" if mode == "conservative" else "[green]NORMAL[/green]"
        veto_str = "  [red][NEWS VETO][/red]" if veto else ""
        footer = (
            f"Portfolio: [bold]${portfolio_value:,.2f}[/bold]  "
            f"Cash: ${cash:,.2f}  "
            f"P&L: [{pnl_col}]{pnl_pct:+.2%}[/{pnl_col}]  "
            f"Modalità: {mode_str}{veto_str}\n"
            f"[dim]Prossimo ciclo tra {wait_seconds}s — INVIO conferma · [m] override · [c] cambia comportamento[/dim]"
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

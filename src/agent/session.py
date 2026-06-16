import json
import os
import uuid
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import config
from src.agent import journal as journal_module

_console = Console()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionManager:
    def detect_previous_session(self) -> dict | None:
        try:
            if not os.path.exists(config.SESSION_PATH):
                return None
            with open(config.SESSION_PATH, "r", encoding="utf-8") as f:
                session = json.load(f)
            if session.get("status") in ("active", "paused"):
                return session
            return None
        except Exception:
            return None

    def print_resoconto(self, summary: dict) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        pnl = summary.get("final_pnl_pct")
        pnl_str = f"{pnl:+.2%}" if pnl is not None else "N/A"
        decisions = summary.get("decisions", {})
        rows = [
            ("ID", f"{summary.get('session_id', '')[:8]}…"),
            ("Cicli", str(summary.get("cycles", 0))),
            ("Ordini", str(summary.get("orders_placed", 0))),
            ("P&L", pnl_str),
            ("Decisioni autonome", str(summary.get("autonomous_decisions", 0))),
            ("Errori loggati", str(summary.get("errors", 0))),
            ("Buy/Sell/Hold", f"{decisions.get('buy',0)}/{decisions.get('sell',0)}/{decisions.get('hold',0)}"),
        ]
        for label, value in rows:
            table.add_row(f"[bold]{label}:[/bold]", value)
        panel = Panel(table, title="SESSIONE PRECEDENTE RILEVATA", border_style="yellow")
        _console.print(panel)

    def ask_resume_or_new(self) -> str:
        choice = input("Vuoi riprendere questa sessione? [s/N]: ").strip().lower()
        return "resume" if choice == "s" else "new"

    def resume(self, session: dict) -> dict:
        session["status"] = "active"
        session["last_active_at"] = _now_utc()
        self.save(session)
        return session

    def create_new(self, prompt: str) -> dict:
        os.makedirs(os.path.dirname(config.SESSION_PATH), exist_ok=True)
        session = {
            "session_id": str(uuid.uuid4()),
            "started_at": _now_utc(),
            "last_active_at": _now_utc(),
            "status": "active",
            "cycle": 0,
            "active_prompt": prompt,
            "initial_prompt": prompt,
            "active_strategy_id": None,
            "portfolio_snapshot": {
                "cash": 100_000.0,
                "portfolio_value": 100_000.0,
                "pnl_pct": 0.0,
                "positions": {},
            },
            "behavior_change_count": 0,
        }
        self.save(session)
        return session

    def save(self, session: dict) -> None:
        session["last_active_at"] = _now_utc()
        tmp = config.SESSION_PATH + ".tmp"
        os.makedirs(os.path.dirname(config.SESSION_PATH), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        os.replace(tmp, config.SESSION_PATH)  # R8: atomic write

    def mark_paused(self, session: dict) -> None:
        session["status"] = "paused"
        self.save(session)

    def mark_completed(self, session: dict) -> None:
        session["status"] = "completed"
        self.save(session)

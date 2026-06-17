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
            ("Cycles", str(summary.get("cycles", 0))),
            ("Orders", str(summary.get("orders_placed", 0))),
            ("P&L", pnl_str),
            ("Autonomous decisions", str(summary.get("autonomous_decisions", 0))),
            ("Logged errors", str(summary.get("errors", 0))),
            ("Buy/Sell/Hold", f"{decisions.get('buy',0)}/{decisions.get('sell',0)}/{decisions.get('hold',0)}"),
        ]
        for label, value in rows:
            table.add_row(f"[bold]{label}:[/bold]", value)
        panel = Panel(table, title="PREVIOUS SESSION FOUND", border_style="yellow")
        _console.print(panel)

    def ask_resume_or_new(self) -> str:
        choice = input("Resume this session? [y/N]: ").strip().lower()
        return "resume" if choice == "y" else "new"

    def resume(self, session: dict) -> dict:
        session["status"] = "active"
        session["last_active_at"] = _now_utc()
        # Ensure all F2/F4 fields exist on old sessions
        session.setdefault("user_stop_loss_pct", None)
        session.setdefault("user_take_profit_pct", None)
        session.setdefault("pref_sectors", [])
        session.setdefault("pref_excluded_sectors", [])
        session.setdefault("pref_risk_level", "unspecified")
        session.setdefault("pref_ethics", [])
        session.setdefault("pref_time_horizon", "unspecified")
        session.setdefault("pref_emotion", "neutral")
        session.setdefault("pref_emotion_score", 0.0)
        session.setdefault("style_hold_rate", 0.5)
        session.setdefault("style_confirm_rate", 0.5)
        session.setdefault("style_override_count", 0)
        session.setdefault("style_reject_sl_count", 0)
        session.setdefault("style_inferred", "undetected")
        session.setdefault("derived_confidence_delta", 0.0)
        session.setdefault("derived_position_pct_delta", 0.0)
        session.setdefault("derived_mode_bias", "none")
        session.setdefault("wait_choices", [])
        session.setdefault("preference_conflicts", [])
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
            # F2: user-defined thresholds
            "user_stop_loss_pct":    None,
            "user_take_profit_pct":  None,
            # F4: explicit preferences
            "pref_sectors":          [],
            "pref_excluded_sectors": [],
            "pref_risk_level":       "unspecified",
            "pref_ethics":           [],
            "pref_time_horizon":     "unspecified",
            # F4: emotional tone
            "pref_emotion":          "neutral",
            "pref_emotion_score":    0.0,
            # F4: implicit style
            "style_hold_rate":       0.5,
            "style_confirm_rate":    0.5,
            "style_override_count":  0,
            "style_reject_sl_count": 0,
            "style_inferred":        "undetected",
            # F4: derived parameters
            "derived_confidence_delta":   0.0,
            "derived_position_pct_delta": 0.0,
            "derived_mode_bias":          "none",
            # F4: wait history + conflict log
            "wait_choices":          [],
            "preference_conflicts":  [],
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

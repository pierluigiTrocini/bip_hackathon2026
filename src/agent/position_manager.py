"""
Position Manager — adaptive stop-loss / take-profit, position context.
R1: every public method wraps body in try/except, never raises.
R2: no LLM calls.
"""
import json
import os
from dataclasses import dataclass, field

from src.agent import config
from src.agent import journal as journal_module


@dataclass
class PositionThresholds:
    stop_loss_pct:    float   # negative, e.g. -4.20
    take_profit_pct:  float   # positive, e.g. +7.50
    stop_source:      str     # "adaptive" | "user" | "conservative_override"
    take_source:      str     # "adaptive" | "user" | "conservative_override"
    volatility_pct:   float
    sentiment_trend:  str     # "improving" | "stable" | "deteriorating"
    explanation:      str


@dataclass
class PositionState:
    ticker:              str
    entry_price:         float
    entry_cycle:         int
    qty:                 int
    price_history:       list[float] = field(default_factory=list)
    thresholds:          PositionThresholds = field(default_factory=lambda: PositionThresholds(
        stop_loss_pct=-config.POSITION_MIN_STOP_LOSS_PCT,
        take_profit_pct=config.POSITION_MIN_TAKE_PROFIT_PCT,
        stop_source="adaptive", take_source="adaptive",
        volatility_pct=1.0, sentiment_trend="stable",
        explanation="Default thresholds.",
    ))
    stop_loss_triggered: bool = False


class PositionManager:
    def __init__(self) -> None:
        self._states: dict[str, PositionState] = {}
        self._session_id: str = ""

    # ── Public interface ────────────────────────────────────────────────────────

    def load_from_journal(
        self, positions: dict, session_id: str, journal_path: str
    ) -> None:
        """Restore position states from open portfolio positions + journal history."""
        try:
            self._session_id = session_id
            tickers_to_remove = [t for t in self._states if t not in positions]
            for t in tickers_to_remove:
                del self._states[t]

            if not positions:
                return

            # Read journal for entry cycles and price history
            entries_by_ticker: dict[str, list[dict]] = {}
            try:
                with open(journal_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            t = e.get("ticker", "")
                            if t in positions:
                                entries_by_ticker.setdefault(t, []).append(e)
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

            for ticker, pos in positions.items():
                entry_price = float(pos.get("avg_entry_price", 0.0))
                qty = int(pos.get("qty", 0))
                entry_cycle = 0
                price_history: list[float] = []

                ticker_entries = entries_by_ticker.get(ticker, [])
                # Find most recent buy entry for entry_cycle
                for e in reversed(ticker_entries):
                    if e.get("action") == "buy":
                        entry_cycle = int(e.get("cycle", 0))
                        break

                # Collect price history from entry_cycle onwards
                for e in ticker_entries:
                    if int(e.get("cycle", 0)) >= entry_cycle:
                        p = float(e.get("price", 0.0))
                        if p > 0:
                            price_history.append(p)
                price_history = price_history[-config.POSITION_HISTORY_CYCLES:]

                if ticker not in self._states:
                    self._states[ticker] = PositionState(
                        ticker=ticker,
                        entry_price=entry_price,
                        entry_cycle=entry_cycle,
                        qty=qty,
                        price_history=price_history,
                    )
                else:
                    self._states[ticker].entry_price = entry_price
                    self._states[ticker].qty = qty
        except Exception as exc:
            try:
                journal_module.log_error(
                    source="PositionManager",
                    error=f"load_from_journal failed: {exc}",
                    session_id=self._session_id,
                )
            except Exception:
                pass

    def on_new_position(
        self, ticker: str, entry_price: float, entry_cycle: int, qty: int
    ) -> None:
        """Create PositionState for a newly opened position."""
        try:
            self._states[ticker] = PositionState(
                ticker=ticker,
                entry_price=entry_price,
                entry_cycle=entry_cycle,
                qty=qty,
                price_history=[],
            )
        except Exception:
            pass

    def on_position_closed(self, ticker: str) -> None:
        """Remove position state when a position is sold."""
        try:
            self._states.pop(ticker, None)
        except Exception:
            pass

    def update_price(self, ticker: str, current_price: float) -> None:
        """Append current price to history; create minimal state if missing."""
        try:
            if ticker not in self._states:
                self._states[ticker] = PositionState(
                    ticker=ticker,
                    entry_price=current_price,
                    entry_cycle=0,
                    qty=0,
                    price_history=[],
                )
            state = self._states[ticker]
            state.price_history.append(current_price)
            if len(state.price_history) > config.POSITION_HISTORY_CYCLES:
                state.price_history = state.price_history[-config.POSITION_HISTORY_CYCLES:]
        except Exception:
            pass

    def update_thresholds(
        self,
        ticker: str,
        bars_closes: list[float],
        recent_sentiments: list[float],
        mode: str,
        user_stop_loss_pct: float | None,
        user_take_profit_pct: float | None,
    ) -> PositionThresholds:
        """Compute adaptive stop-loss/take-profit thresholds per spec §5.3."""
        try:
            # Step 1 — Volatility
            if len(bars_closes) >= 2:
                returns = [
                    abs(bars_closes[i] - bars_closes[i - 1]) / bars_closes[i - 1] * 100
                    for i in range(1, len(bars_closes))
                    if bars_closes[i - 1] != 0
                ]
                volatility_pct = sum(returns) / len(returns) if returns else 1.0
            else:
                volatility_pct = 1.0

            # Step 2 — Sentiment trend
            if len(recent_sentiments) < 2:
                sentiment_trend = "stable"
            elif recent_sentiments[-1] > recent_sentiments[0] + 0.15:
                sentiment_trend = "improving"
            elif recent_sentiments[-1] < recent_sentiments[0] - 0.15:
                sentiment_trend = "deteriorating"
            else:
                sentiment_trend = "stable"

            # Step 3 — Adaptive stop-loss
            raw_stop = -(volatility_pct * config.POSITION_VOLATILITY_MULTIPLIER)
            stop_loss_pct = max(
                -config.POSITION_MAX_STOP_LOSS_PCT,
                min(-config.POSITION_MIN_STOP_LOSS_PCT, raw_stop),
            )
            stop_source = "adaptive"

            # Step 4 — Adaptive take-profit (1:1.5 risk/reward)
            base_take = abs(stop_loss_pct) * 1.5
            if sentiment_trend == "improving":
                base_take *= 1.20
            elif sentiment_trend == "deteriorating":
                base_take *= 0.80
            take_profit_pct = max(
                config.POSITION_MIN_TAKE_PROFIT_PCT,
                min(config.POSITION_MAX_TAKE_PROFIT_PCT, base_take),
            )
            take_source = "adaptive"

            # Step 5 — Conservative override
            if mode == "conservative":
                stop_loss_pct   = max(stop_loss_pct, -config.POSITION_MIN_STOP_LOSS_PCT)
                take_profit_pct = min(take_profit_pct, config.POSITION_MIN_TAKE_PROFIT_PCT * 1.5)
                stop_source = take_source = "conservative_override"

            # Step 6 — User override (user wins only if more restrictive)
            if user_stop_loss_pct is not None and user_stop_loss_pct > stop_loss_pct:
                stop_loss_pct = user_stop_loss_pct
                stop_source = "user"
            if user_take_profit_pct is not None and user_take_profit_pct < take_profit_pct:
                take_profit_pct = user_take_profit_pct
                take_source = "user"

            # Step 7 — explanation
            explanation = (
                f"Stop-loss {stop_loss_pct:+.2f}% [{stop_source}]: "
                f"volatility {volatility_pct:.2f}%/cycle × {config.POSITION_VOLATILITY_MULTIPLIER}. "
                f"Take-profit {take_profit_pct:+.2f}% [{take_source}]: sentiment {sentiment_trend}."
            )

            thresholds = PositionThresholds(
                stop_loss_pct=round(stop_loss_pct, 4),
                take_profit_pct=round(take_profit_pct, 4),
                stop_source=stop_source,
                take_source=take_source,
                volatility_pct=round(volatility_pct, 4),
                sentiment_trend=sentiment_trend,
                explanation=explanation,
            )

            if ticker in self._states:
                self._states[ticker].thresholds = thresholds

            return thresholds

        except Exception as exc:
            try:
                journal_module.log_error(
                    source="PositionManager",
                    error=f"update_thresholds failed for {ticker}: {exc}",
                    ticker=ticker, session_id=self._session_id,
                )
            except Exception:
                pass
            default = PositionThresholds(
                stop_loss_pct=-config.POSITION_MIN_STOP_LOSS_PCT,
                take_profit_pct=config.POSITION_MIN_TAKE_PROFIT_PCT,
                stop_source="adaptive", take_source="adaptive",
                volatility_pct=1.0, sentiment_trend="stable",
                explanation="Default thresholds (calculation error).",
            )
            if ticker in self._states:
                self._states[ticker].thresholds = default
            return default

    def check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """True if unrealized P&L% ≤ stop_loss_pct."""
        try:
            state = self._states.get(ticker)
            if not state or state.entry_price <= 0:
                return False
            pnl_pct = (current_price - state.entry_price) / state.entry_price * 100
            return pnl_pct <= state.thresholds.stop_loss_pct
        except Exception:
            return False

    def check_take_profit(self, ticker: str, current_price: float) -> bool:
        """True if unrealized P&L% ≥ take_profit_pct."""
        try:
            state = self._states.get(ticker)
            if not state or state.entry_price <= 0:
                return False
            pnl_pct = (current_price - state.entry_price) / state.entry_price * 100
            return pnl_pct >= state.thresholds.take_profit_pct
        except Exception:
            return False

    def build_position_context(
        self, ticker: str, current_price: float, current_cycle: int
    ) -> str:
        """Build === POSITION CONTEXT === section for Gemma4's prompt."""
        try:
            state = self._states.get(ticker)
            if not state or state.entry_price <= 0:
                return ""

            pnl_pct = (current_price - state.entry_price) / state.entry_price * 100
            abs_pnl = abs(current_price - state.entry_price) * max(state.qty, 1)
            sign = "+" if pnl_pct >= 0 else "-"
            age = current_cycle - state.entry_cycle
            t = state.thresholds

            # Andamento: last min(4, len) values
            hist = state.price_history
            if len(hist) >= 2:
                sample = hist[-min(4, len(hist)):]
                trend_str = "Trend:           " + " → ".join(f"${p:.2f}" for p in sample)
            else:
                trend_str = ""

            dist_stop = pnl_pct - t.stop_loss_pct
            dist_take = t.take_profit_pct - pnl_pct

            if dist_stop > 3.0:
                stop_status = "wide margin"
            elif dist_stop > 0:
                stop_status = "caution"
            else:
                stop_status = "THRESHOLD BREACHED"

            if dist_take > 2.0:
                take_status = "far"
            elif dist_take > 0:
                take_status = "near"
            else:
                take_status = "reached"

            lines = [
                f"=== POSITION CONTEXT ({ticker}) ===",
                f"Entry:           ${state.entry_price:.2f}  (cycle {state.entry_cycle}, {age} cycles ago)",
                f"Current price:   ${current_price:.2f}",
                f"Position P&L:    {pnl_pct:+.2f}%  ({sign}${abs_pnl:.2f} on {state.qty} shares)",
            ]
            if trend_str:
                lines.append(trend_str)
            lines += [
                "",
                "Active thresholds:",
                f"  Stop-loss:    {t.stop_loss_pct:+.2f}%  [{t.stop_source}]",
                f"  Take-profit:  {t.take_profit_pct:+.2f}%  [{t.take_source}]",
                "",
                f"Distance to stop-loss:    {dist_stop:+.2f}%  ({stop_status})",
                f"Distance to take-profit:  {dist_take:+.2f}%  ({take_status})",
                "",
                t.explanation,
            ]
            return "\n".join(lines)

        except Exception:
            return ""

    def get_state(self, ticker: str) -> PositionState | None:
        try:
            return self._states.get(ticker)
        except Exception:
            return None

    def all_tickers(self) -> list[str]:
        try:
            return list(self._states.keys())
        except Exception:
            return []

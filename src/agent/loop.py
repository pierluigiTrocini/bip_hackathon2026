import concurrent.futures
import time
from datetime import datetime, timezone

from src.agent import config
from src.agent import journal as journal_module
from src.agent.adaptive_timeout import AdaptiveTimeout
from src.agent.behavior import BehaviorManager
from src.agent.broker import Broker
from src.agent.imitative_layer import ImiativeLayer
from src.agent.memory_manager import MemoryManager
from src.agent.reasoner import Reasoner
from src.agent.session import SessionManager
from src.agent.tool_executor import ToolExecutor


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AgentLoop:
    def __init__(
        self,
        session: dict,
        adaptive_timeout: AdaptiveTimeout,
        tool_executor: ToolExecutor,
        memory_manager: MemoryManager,
        imitative_layer: ImiativeLayer,
        reasoner: Reasoner,
        broker: Broker,
        behavior_manager: BehaviorManager,
        session_manager: SessionManager,
        dashboard,
    ) -> None:
        self._session = session
        self._at = adaptive_timeout
        self._te = tool_executor
        self._mm = memory_manager
        self._il = imitative_layer
        self._reasoner = reasoner
        self._broker = broker
        self._bm = behavior_manager
        self._sm = session_manager
        self._dashboard = dashboard
        self._running = False
        self._cycle = session.get("cycle", 0)

        # propagate session_id to sub-modules
        sid = session.get("session_id", "")
        self._te._session_id = sid
        self._reasoner._session_id = sid
        self._broker._session_id = sid

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                journal_module.log_error(
                    source="AgentLoop", error=f"Unhandled cycle error: {exc}",
                    session_id=self._session.get("session_id", ""),
                )

    def stop(self) -> None:
        self._running = False

    def _handle_behavior_change(self) -> None:
        self._graceful_stop_tasks()
        applied = self._bm.apply_change(self._mm, self._il)
        if applied:
            self._bm.increment_change_count(self._session)
        self._sm.save(self._session)

    def _graceful_stop_tasks(self) -> None:
        self._broker.cancel_all_orders()

    def _run_cycle(self) -> None:
        if self._bm.change_requested:
            self._handle_behavior_change()

        self._cycle += 1
        self._session["cycle"] = self._cycle
        session_id = self._session.get("session_id", "")
        active_prompt = self._bm.active_prompt
        t_wait = self._at.t_wait()
        t_behavior = self._at.t_behavior()
        market_open = self._te.is_market_open()

        veto_triggered = False
        last_entry: dict = {}

        for ticker in config.TICKERS:
            if ticker in self._te._blacklisted:
                continue
            try:
                # 1. OUTCOME UPDATE
                price_result = self._te.get_price(ticker)
                if price_result.ok:
                    journal_module.outcome_update(
                        ticker, price_result.data["price"], session_id
                    )

                # 2+3. OBSERVE + SENTIMENT (concurrent)
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    fut_bars = pool.submit(self._te.get_bars, ticker, 5)
                    fut_news = pool.submit(self._te.get_news, ticker)
                    fut_portfolio = pool.submit(self._te.get_portfolio)

                    bars_result = fut_bars.result(timeout=30)
                    news_result = fut_news.result(timeout=30)
                    portfolio_result = fut_portfolio.result(timeout=30)

                articles = news_result.data.get("articles", []) if news_result.ok else []
                from src.agent import sentiment as sentiment_module
                sentiment_data = sentiment_module.analyse(ticker, articles, active_prompt, t_behavior)

                # 4. PORTFOLIO HEALTH CHECK
                pnl_pct = portfolio_result.data.get("pnl_pct", 0.0) if portfolio_result.ok else 0.0
                mode = "conservative" if pnl_pct < -config.DRAWDOWN_THRESHOLD else "normal"
                cash = portfolio_result.data.get("cash", 100_000.0) if portfolio_result.ok else 100_000.0
                positions = portfolio_result.data.get("positions", {}) if portfolio_result.ok else {}
                portfolio_value = portfolio_result.data.get("portfolio_value", 100_000.0) if portfolio_result.ok else 100_000.0
                portfolio_mode_reason = (
                    f"drawdown {pnl_pct:.2%}" if mode == "conservative" else None
                )

                # 5. MEMORY CONTEXT
                memory_context = self._mm.build_context(ticker)
                imitative_hints = self._il.build_hints(active_prompt, ticker)
                imitative_source = self._il.get_active_strategy_id(active_prompt)

                # price data
                price = price_result.data.get("price", 0.0) if price_result.ok else 0.0
                price_timestamp = price_result.data.get("timestamp", _now_utc()) if price_result.ok else _now_utc()
                ma5 = bars_result.data.get("ma", 0.0) if bars_result.ok else 0.0
                trend = bars_result.data.get("trend", "flat") if bars_result.ok else "flat"
                stale = not price_result.ok or price_result.stale
                staleness_seconds = price_result.staleness_seconds if not price_result.ok else 0
                data_ok = price_result.ok

                # 6. THINK
                decision = self._reasoner.decide(
                    ticker=ticker,
                    memory_context=memory_context,
                    price=price,
                    price_timestamp=price_timestamp,
                    ma5=ma5,
                    trend=trend,
                    sentiment_score=sentiment_data["score"],
                    sentiment_label=sentiment_data["label"],
                    imitative_hints=imitative_hints,
                    active_prompt=active_prompt,
                    cash=cash,
                    positions=positions,
                    mode=mode,
                    stale=stale,
                    staleness_seconds=staleness_seconds,
                    t_behavior=t_behavior,
                )

                # 7. ACT (R5: confidence gate is code, not prompt)
                threshold = (
                    config.CONFIDENCE_THRESHOLD_CONSERVATIVE if mode == "conservative"
                    else config.CONFIDENCE_THRESHOLD_NORMAL
                )
                action = decision["action"]
                order_result: dict = {"ok": False, "order_id": None}
                decision_source = "agent"

                if action in ("buy", "sell") and decision["confidence"] >= threshold and market_open:
                    qty = self._broker.compute_qty(price, cash, mode)
                    if qty > 0:
                        order_result = self._broker.place_order(ticker, action, qty)
                    else:
                        action = "hold"
                else:
                    action = "hold"
                    decision_source = "autonomous_timeout"

                # 8. RECORD (R4: every cycle writes an entry)
                entry = journal_module.build_entry(
                    ts=_now_utc(),
                    cycle=self._cycle,
                    ticker=ticker,
                    session_id=session_id,
                    action=action,
                    conf=decision["confidence"],
                    conf_raw=decision["confidence_raw"],
                    stale_penalty=decision["stale_penalty"],
                    reasoning=decision["reasoning"],
                    accuracy_review=decision["accuracy_review"],
                    decision_source=decision_source,
                    price=price,
                    price_timestamp=price_timestamp,
                    ma5=ma5,
                    trend=trend,
                    sentiment=sentiment_data["score"],
                    sentiment_label=sentiment_data["label"],
                    data_ok=data_ok,
                    imitative_source=imitative_source,
                    prompt_snapshot=active_prompt[:100],
                    t_wait_used=t_wait,
                    t_behavior_used=t_behavior,
                    mode=mode,
                    portfolio_mode_reason=portfolio_mode_reason,
                    order_id=order_result.get("order_id"),
                    market_open=market_open,
                    price_after=None,
                    outcome_pct=None,
                    cash=cash,
                    portfolio_value=portfolio_value,
                    pnl_pct=pnl_pct,
                    positions=positions,
                )
                journal_module.write_entry(entry)
                self._mm.update(entry)
                last_entry = entry

                # 9. UPDATE DASHBOARD
                self._dashboard.update(ticker, entry, portfolio_result.data if portfolio_result.ok else {}, t_wait, t_behavior)

                # NEWS VETO (step 11 — checked per ticker)
                if sentiment_data["score"] < -0.7 and ticker in positions and action == "hold":
                    veto_triggered = True

            except Exception as exc:
                journal_module.log_error(
                    source="AgentLoop", error=f"Ticker {ticker} cycle error: {exc}",
                    ticker=ticker, session_id=session_id,
                )

        # 10. ADAPTIVE TIMEOUT UPDATE (every 5 cycles)
        if self._cycle % 5 == 0:
            self._at.calibrate()

        # 12. WAIT with user input window (R10: always from adaptive_timeout)
        wait_seconds = 2 if veto_triggered else self._at.t_wait()
        result = self._dashboard.wait_for_user_input(wait_seconds)

        if result["source"] == "behavior_change":
            self._bm.request_change(result["data"]["new_prompt"])
        elif result["source"] == "override" and last_entry:
            ov = result["data"]
            order_result = self._broker.place_order(ov["ticker"], ov["action"], ov["qty"])
            # Log the override in journal
            ov_entry = dict(last_entry)
            ov_entry["action"] = ov["action"]
            ov_entry["decision_source"] = "user_override"
            ov_entry["order_id"] = order_result.get("order_id")
            ov_entry["ts"] = _now_utc()
            journal_module.write_entry(ov_entry)

        self._sm.save(self._session)

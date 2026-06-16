import concurrent.futures
import threading
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
        tickers: list[str] | None = None,
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
        self._tickers: list[str] = tickers if tickers else config.TICKERS

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

    def _run_interaction(self, mode: str, context: dict) -> None:
        """
        Background thread: let the user modify agent behavior without blocking the cycle.
        Calls behavior_manager.request_change() when the user submits — applied next cycle.
        """
        from src.agent.behavior_questionnaire import generate_questions, synthesize_prompt

        d = self._dashboard
        try:
            if mode == "prompt_change":
                d.log("", "info")
                d.log("━━━ MODIFICA COMPORTAMENTO (in background) ━━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                d.log(f"  Prompt attuale: {context['active_prompt'][:80]}", "info")
                new_prompt = d.interactive_input(
                    "\n[bold cyan]Inserisci il nuovo comportamento dell'agente:[/bold cyan]"
                )
                if new_prompt:
                    self._bm.request_change(new_prompt)
                    d.log(f"  Comportamento in coda: {new_prompt[:70]}", "ok")
                else:
                    d.log("  Nessuna modifica inserita.", "info")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

            elif mode == "questionnaire":
                d.log("", "info")
                d.log("━━━ QUESTIONARIO STRATEGIA (in background) ━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                d.log("  Generazione domande in corso…", "info")
                t_behavior = self._at.t_behavior()
                questions = generate_questions(context, t_behavior)
                answers: list[str] = []
                for i, q in enumerate(questions, 1):
                    ans = d.interactive_input(
                        f"\n[bold cyan]{i}/{len(questions)}.[/bold cyan] {q}"
                    )
                    answers.append(ans if ans else "(nessuna risposta)")

                d.log("  Sintesi delle risposte in corso…", "info")
                new_prompt = synthesize_prompt(
                    context["active_prompt"], questions, answers, t_behavior
                )
                self._bm.request_change(new_prompt)
                d.log(f"  Nuovo comportamento in coda: {new_prompt[:70]}", "ok")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        except Exception as exc:
            d.log(f"  Errore interazione background: {exc}", "err")

    def _run_cycle(self) -> None:
        if self._bm.change_requested:
            self._dashboard.log("Applicazione cambio comportamento…", "warn")
            self._handle_behavior_change()
            self._dashboard.log(f"Comportamento aggiornato → {self._bm.active_prompt[:60]}", "ok")

        self._cycle += 1
        self._session["cycle"] = self._cycle
        session_id = self._session.get("session_id", "")
        active_prompt = self._bm.active_prompt
        t_wait = self._at.t_wait()
        t_behavior = self._at.t_behavior()
        market_open = self._te.is_market_open()

        tickers_str = ", ".join(self._tickers)
        mkt_str = "APERTO" if market_open else "CHIUSO"
        self._dashboard.log(
            f"── Ciclo {self._cycle} ─────────────────  "
            f"mercato:{mkt_str}  t_wait:{t_wait}s  t_behavior:{t_behavior}s  "
            f"ticker:[{tickers_str}]",
            "info",
        )

        veto_triggered = False
        last_entry: dict = {}
        cycle_rows: list[dict] = []  # accumulate per-ticker results for end-of-cycle summary

        # Extend ticker list with any positions currently held that are NOT in discovery list
        _initial_portfolio = self._te.get_portfolio()
        _held = set(
            _initial_portfolio.data.get("positions", {}).keys()
        ) if _initial_portfolio.ok else set()
        _extra = [t for t in _held if t not in self._tickers]
        effective_tickers = list(self._tickers) + _extra

        if _extra:
            self._dashboard.log(
                f"  + posizioni aperte non in discovery: [{', '.join(_extra)}] — incluse nel ciclo",
                "warn",
            )

        for ticker in effective_tickers:
            if ticker in self._te._blacklisted:
                self._dashboard.log(f"  {ticker} → blacklistato, skip", "warn")
                continue
            try:
                # 1. OUTCOME UPDATE
                self._dashboard.log(f"  {ticker} → [1] recupero prezzo corrente…", "info")
                price_result = self._te.get_price(ticker)
                if price_result.ok:
                    journal_module.outcome_update(
                        ticker, price_result.data["price"], session_id
                    )
                    stale_tag = " [STALE]" if price_result.stale else ""
                    self._dashboard.log(
                        f"  {ticker} → prezzo ${price_result.data['price']:,.2f}{stale_tag}", "ok"
                    )
                else:
                    self._dashboard.log(f"  {ticker} → prezzo non disponibile (uso cache)", "warn")

                # 2+3. OBSERVE + SENTIMENT (concurrent)
                self._dashboard.log(f"  {ticker} → [2] recupero bars/news/portfolio…", "info")
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    fut_bars = pool.submit(self._te.get_bars, ticker, 5)
                    fut_news = pool.submit(self._te.get_news, ticker)
                    fut_portfolio = pool.submit(self._te.get_portfolio)

                    bars_result = fut_bars.result(timeout=30)
                    news_result = fut_news.result(timeout=30)
                    portfolio_result = fut_portfolio.result(timeout=30)

                articles = news_result.data.get("articles", []) if news_result.ok else []
                article_count = len(articles)
                self._dashboard.log(
                    f"  {ticker} → bars:{'ok' if bars_result.ok else 'err'}  "
                    f"news:{article_count} articoli  "
                    f"portfolio:{'ok' if portfolio_result.ok else 'err'}",
                    "ok" if (bars_result.ok and portfolio_result.ok) else "warn",
                )

                self._dashboard.log(f"  {ticker} → [3] analisi sentiment ({article_count} art.)…", "info")
                from src.agent import sentiment as sentiment_module
                sentiment_data = sentiment_module.analyse(ticker, articles, active_prompt, t_behavior)
                self._dashboard.log(
                    f"  {ticker} → sentiment: {sentiment_data['label']} "
                    f"({sentiment_data['score']:+.2f})",
                    "ok",
                )

                # 4. PORTFOLIO HEALTH CHECK
                pnl_pct = portfolio_result.data.get("pnl_pct", 0.0) if portfolio_result.ok else 0.0
                mode = "conservative" if pnl_pct < -config.DRAWDOWN_THRESHOLD else "normal"
                cash = portfolio_result.data.get("cash", 100_000.0) if portfolio_result.ok else 100_000.0
                positions = portfolio_result.data.get("positions", {}) if portfolio_result.ok else {}
                portfolio_value = portfolio_result.data.get("portfolio_value", 100_000.0) if portfolio_result.ok else 100_000.0
                portfolio_mode_reason = (
                    f"drawdown {pnl_pct:.2%}" if mode == "conservative" else None
                )
                self._dashboard.log(
                    f"  {ticker} → modalità:{mode.upper()}  P&L:{pnl_pct:+.2%}  "
                    f"cash:${cash:,.0f}  portfolio:${portfolio_value:,.0f}",
                    "warn" if mode == "conservative" else "info",
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

                strat_str = f"  strategia:{imitative_source}" if imitative_source else ""
                self._dashboard.log(
                    f"  {ticker} → MA5:${ma5:.2f}  trend:{trend}{strat_str}", "info"
                )

                # 6. THINK
                self._dashboard.log(
                    f"  {ticker} → [6] ragionamento LLM (t_behavior:{t_behavior}s)…", "info"
                )
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

                stale_tag = f"  -stale:{decision['stale_penalty']:.2f}" if decision["stale_penalty"] > 0 else ""
                self._dashboard.log(
                    f"  {ticker} → decisione:{decision['action'].upper()}  "
                    f"conf:{decision['confidence']:.2f} (raw:{decision['confidence_raw']:.2f}{stale_tag})  "
                    f"{decision['reasoning'][:70]}",
                    "ok",
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
                        self._dashboard.log(
                            f"  {ticker} → [7] invio ordine {action.upper()} qty:{qty}…", "action"
                        )
                        order_result = self._broker.place_order(ticker, action, qty)
                        if order_result.get("ok"):
                            self._dashboard.log(
                                f"  {ticker} → ordine accettato (id:{str(order_result.get('order_id','?'))[:8]}…)",
                                "ok",
                            )
                        else:
                            self._dashboard.log(
                                f"  {ticker} → ordine rifiutato: {order_result.get('reason','?')}",
                                "err",
                            )
                    else:
                        action = "hold"
                        self._dashboard.log(f"  {ticker} → qty=0, hold forzato", "warn")
                else:
                    reasons: list[str] = []
                    if action not in ("buy", "sell"):
                        reasons.append(f"azione={action}")
                    elif decision["confidence"] < threshold:
                        reasons.append(f"conf {decision['confidence']:.2f} < soglia {threshold:.2f}")
                    if not market_open:
                        reasons.append("mercato chiuso")
                    action = "hold"
                    decision_source = "autonomous_timeout"
                    self._dashboard.log(
                        f"  {ticker} → HOLD autonomo ({'; '.join(reasons)})", "warn"
                    )

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

                # accumulate row for end-of-cycle summary
                cycle_rows.append({
                    "ticker": ticker,
                    "price": price,
                    "trend": trend,
                    "sentiment_label": sentiment_data["label"],
                    "sentiment_score": sentiment_data["score"],
                    "action": action,
                    "conf": decision["confidence"],
                    "reasoning": decision["reasoning"],
                    "order_id": order_result.get("order_id"),
                    "mode": mode,
                    "stale": stale,
                })

                # 9. UPDATE DASHBOARD
                self._dashboard.update(ticker, entry, portfolio_result.data if portfolio_result.ok else {}, t_wait, t_behavior)

                # NEWS VETO (step 11 — checked per ticker)
                if sentiment_data["score"] < -0.7 and ticker in positions and action == "hold":
                    veto_triggered = True
                    self._dashboard.log(
                        f"  {ticker} → NEWS VETO attivato (sentiment {sentiment_data['score']:.2f})",
                        "err",
                    )

            except Exception as exc:
                self._dashboard.log(f"  {ticker} → ERRORE ciclo: {exc}", "err")
                journal_module.log_error(
                    source="AgentLoop", error=f"Ticker {ticker} cycle error: {exc}",
                    ticker=ticker, session_id=session_id,
                )

        # 10. ADAPTIVE TIMEOUT UPDATE (every 5 cycles)
        if self._cycle % 5 == 0:
            self._dashboard.log("  [10] Ricalibrazione timeout adattivo…", "info")
            self._at.calibrate()
            s = self._at.summary()
            self._dashboard.log(
                f"  → api_avg:{s['api_avg']:.0f}ms  ollama_avg:{s['ollama_avg']:.0f}ms  "
                f"t_wait:{s['t_wait']}s  t_behavior:{s['t_behavior']}s",
                "ok",
            )

        # 12. WAIT with user input window (R10: always from adaptive_timeout)
        wait_seconds = 2 if veto_triggered else self._at.t_wait()
        if veto_triggered:
            self._dashboard.log("  [NEWS VETO] attesa ridotta a 2s", "err")

        self._dashboard.print_cycle_summary(
            cycle=self._cycle,
            rows=cycle_rows,
            pnl_pct=pnl_pct if cycle_rows else 0.0,
            portfolio_value=portfolio_value if cycle_rows else 0.0,
            cash=cash if cycle_rows else 0.0,
            mode=mode if cycle_rows else "normal",
            wait_seconds=wait_seconds,
            veto=veto_triggered,
        )
        result = self._dashboard.wait_for_user_input(wait_seconds)

        if result["source"] in ("prompt_change", "questionnaire"):
            # Non-blocking: spawn background thread, cycle continues immediately
            context = {
                "active_prompt": active_prompt,
                "tickers": effective_tickers,
                "pnl_pct": pnl_pct if cycle_rows else 0.0,
                "mode": mode if cycle_rows else "normal",
                "recent_actions": "  ".join(
                    f"{r['ticker']}:{r['action'].upper()}" for r in cycle_rows[-3:]
                ),
            }
            t = threading.Thread(
                target=self._run_interaction,
                args=(result["source"], context),
                daemon=True,
            )
            t.start()
            mode_label = "prompt" if result["source"] == "prompt_change" else "questionario"
            self._dashboard.log(
                f"  Interazione {mode_label} avviata in background — ciclo successivo in corso…",
                "warn",
            )
        elif result["source"] == "override" and last_entry:
            ov = result["data"]
            self._dashboard.log(
                f"  Override manuale: {ov['ticker']} {ov['action'].upper()} qty:{ov['qty']}", "action"
            )
            order_result = self._broker.place_order(ov["ticker"], ov["action"], ov["qty"])
            ov_entry = dict(last_entry)
            ov_entry["action"] = ov["action"]
            ov_entry["decision_source"] = "user_override"
            ov_entry["order_id"] = order_result.get("order_id")
            ov_entry["ts"] = _now_utc()
            journal_module.write_entry(ov_entry)
            ok_str = "accettato" if order_result.get("ok") else f"rifiutato: {order_result.get('reason','?')}"
            self._dashboard.log(f"  Override → {ok_str}", "ok" if order_result.get("ok") else "err")
        elif result["source"] == "confirmed":
            self._dashboard.log("  → Confermato dall'utente.", "info")

        self._sm.save(self._session)

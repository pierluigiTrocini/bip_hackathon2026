import concurrent.futures
import re
import select
import sys
import threading
import time
from datetime import datetime, timezone

from src.agent import config
from src.agent import journal as journal_module
from src.agent import news_log
from src.agent import strategy_library
from src.agent import technical_analyser
from src.agent.adaptive_timeout import AdaptiveTimeout
from src.agent.behavior import BehaviorManager
from src.agent.broker import Broker
from src.agent.correlation_engine import CorrelationEngine
from src.agent.disruptor import MarketDisruptor
from src.agent.imitative_layer import ImiativeLayer
from src.agent.memory_manager import MemoryManager
from src.agent.position_manager import PositionManager
from src.agent.reasoner import Reasoner
from src.agent.session import SessionManager
from src.agent.tool_executor import ToolExecutor
from src.agent.user_preference_engine import UserPreferenceEngine


_TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
_KNOWN_NON_TICKERS = {
    "US", "CEO", "CFO", "GDP", "IPO", "SEC", "ETF", "AI", "UK", "EU",
    "NY", "FED", "CPI", "Q1", "Q2", "Q3", "Q4", "EPS", "BUY", "SELL",
}


def _extract_mentioned_tickers(article: dict) -> list[str]:
    """Extract potential ticker symbols from article title and summary."""
    text = f"{article.get('title', '')} {article.get('summary', '')}"
    candidates = _TICKER_PATTERN.findall(text)
    return list({c for c in candidates if c not in _KNOWN_NON_TICKERS})



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
        correlation_engine: CorrelationEngine | None = None,
        tickers: list[str] | None = None,
        disruptor: MarketDisruptor | None = None,
        position_manager: PositionManager | None = None,
        preference_engine: UserPreferenceEngine | None = None,
        telegram_notifier=None,
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
        self._correlation_engine = correlation_engine or CorrelationEngine()
        self._disruptor = disruptor
        self._pm = position_manager
        self._upe = preference_engine
        self._telegram = telegram_notifier
        self._running = False
        self._cycle = session.get("cycle", 0)
        self._tickers: list[str] = tickers if tickers else config.TICKERS
        self._interaction_running = False
        self._interaction_lock = threading.Lock()
        self._in_wait_phase = False
        self._pause_requested = False
        self._pending_injection = ""

        _saved_strategy = session.get("current_strategy", strategy_library.DEFAULT_STRATEGY)
        self._current_strategy: str = (
            _saved_strategy if _saved_strategy in strategy_library.get_all()
            else strategy_library.DEFAULT_STRATEGY
        )
        self._recent_metrics: list[dict] = []  # rolling window for auto-switch evaluation

        # propagate session_id to sub-modules
        sid = session.get("session_id", "")
        self._te._session_id = sid
        self._reasoner._session_id = sid
        self._broker._session_id = sid
        if self._pm is not None:
            self._pm._session_id = sid

        self._dashboard.set_current_strategy(self._current_strategy)

    def _stdin_listener(self) -> None:
        while self._running:
            if self._in_wait_phase or self._pause_requested:
                time.sleep(0.1)
                continue
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            except Exception:
                time.sleep(0.2)
                continue
            if ready and not self._in_wait_phase and not self._pause_requested:
                try:
                    line = sys.stdin.readline().strip()
                    if len(line) >= 3:
                        self._pending_injection = line
                        self._pause_requested = True
                except Exception:
                    pass

    def start(self) -> None:
        self._running = True
        self._correlation_engine.rebuild()
        threading.Thread(target=self._stdin_listener, daemon=True).start()
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
        applied = self._bm.apply_change(self._mm, self._il, preference_engine=self._upe)
        if applied:
            self._bm.increment_change_count(self._session)
        self._sm.save(self._session)

    def _graceful_stop_tasks(self) -> None:
        self._broker.cancel_all_orders()

    def _handle_mid_cycle_injection(self, active_prompt: str) -> None:
        d = self._dashboard
        text = self._pending_injection
        self._pending_injection = ""
        d.log("", "info")
        d.log("━━━ PAUSE — INPUT RECEIVED DURING EXECUTION ━━━━━━━━━━━━━━━━━━━━━━", "err")
        d.log(f"  Input received: \"{text[:100]}\"", "warn")
        d.log(f"  Active prompt: {active_prompt[:80]}", "info")
        choice = d.interactive_input(
            "\n[bold cyan](a)[/bold cyan] Append to active prompt  "
            "[bold cyan](s)[/bold cyan] Replace  "
            "[bold cyan](i)[/bold cyan] Ignore"
        )
        choice = (choice or "a").strip().lower()
        if choice == "i":
            d.log("  Input ignored — cycle resumes.", "info")
        elif choice == "s":
            self._bm.request_change(text)
            d.log(f"  New prompt (next cycle): {text[:100]}", "ok")
        else:
            new_prompt = f"{active_prompt}. {text}"
            self._bm.request_change(new_prompt)
            d.log(f"  Instruction added (next cycle): {new_prompt[:100]}", "ok")
        d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

    def _apply_strategy_switch(self, new_id: str) -> None:
        old_name = strategy_library.get(self._current_strategy)["name"]
        new_name = strategy_library.get(new_id)["name"]
        self._current_strategy = new_id
        self._session["current_strategy"] = new_id
        self._dashboard.set_current_strategy(new_id)
        self._sm.save(self._session)
        self._dashboard.log(f"  Strategy: {old_name} → {new_name}", "ok")
        s = strategy_library.get(new_id)
        self._dashboard.log(f"  Description: {s['description']} — best for: {s['best_for']}", "info")

    def _run_interaction(self, mode: str, context: dict) -> None:
        from src.agent.behavior_questionnaire import generate_questions, synthesize_prompt

        d = self._dashboard
        try:
            if mode == "prompt_append":
                d.log("", "info")
                d.log("━━━ ADDITIONAL INSTRUCTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                d.log(f"  Active prompt: {context['active_prompt'][:80]}", "info")
                addition = d.interactive_input(
                    "\n[bold cyan]Enter additional instruction:[/bold cyan]"
                )
                if addition:
                    new_prompt = f"{context['active_prompt']}. {addition}"
                    self._bm.request_change(new_prompt)
                    d.log(f"  Added: {addition[:70]}", "ok")
                    d.log(f"  New prompt: {new_prompt[:100]}", "info")
                else:
                    d.log("  No instruction entered.", "info")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

            elif mode == "prompt_change":
                d.log("", "info")
                d.log("━━━ BEHAVIOUR CHANGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                d.log(f"  Active prompt: {context['active_prompt'][:80]}", "info")
                new_prompt = d.interactive_input(
                    "\n[bold cyan]Enter the new agent behaviour:[/bold cyan]"
                )
                if new_prompt:
                    ok = self._bm.request_change(new_prompt)
                    if ok:
                        d.log(f"  Behaviour queued: {new_prompt[:70]}", "ok")
                    else:
                        d.log("  Change already queued — retry next cycle.", "warn")
                else:
                    d.log("  No change entered.", "info")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

            elif mode == "questionnaire":
                d.log("", "info")
                d.log("━━━ STRATEGY QUESTIONNAIRE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                d.log("  Generating questions…", "info")
                t_behavior = self._at.t_behavior()
                questions = generate_questions(context, t_behavior)
                answers: list[str] = []
                for i, q in enumerate(questions, 1):
                    ans = d.interactive_input(
                        f"\n[bold cyan]{i}/{len(questions)}.[/bold cyan] {q}"
                    )
                    answers.append(ans if ans else "(no answer)")
                d.log("  Synthesising answers…", "info")
                new_prompt = synthesize_prompt(
                    context["active_prompt"], questions, answers, t_behavior
                )
                ok = self._bm.request_change(new_prompt)
                if ok:
                    d.log(f"  New behaviour queued: {new_prompt[:70]}", "ok")
                else:
                    d.log("  Change already queued — questionnaire result discarded.", "warn")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

            elif mode == "strategy_select":
                d.log("", "info")
                d.log("━━━ STRATEGY SELECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "warn")
                all_strats = strategy_library.get_all()
                strat_ids = list(all_strats.keys())
                for i, sid in enumerate(strat_ids, 1):
                    s = all_strats[sid]
                    marker = "  ← ACTIVE" if sid == self._current_strategy else ""
                    d.log(
                        f"  [{i}] [bold]{s['name']}[/bold]: {s['description']}"
                        f"  —  best for: {s['best_for']}{marker}",
                        "ok" if sid == self._current_strategy else "info",
                    )
                choice = d.interactive_input(
                    "\n[bold cyan]Choose strategy (number) or ENTER to cancel:[/bold cyan]"
                )
                if choice.strip().isdigit() and 1 <= int(choice.strip()) <= len(strat_ids):
                    new_id = strat_ids[int(choice.strip()) - 1]
                    if new_id != self._current_strategy:
                        self._apply_strategy_switch(new_id)
                    else:
                        d.log("  Strategy unchanged (same selected).", "info")
                else:
                    d.log("  Selection cancelled — strategy unchanged.", "info")
                d.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        except Exception as exc:
            d.log(f"  Interaction error: {exc}", "err")
        finally:
            with self._interaction_lock:
                self._interaction_running = False
            d.set_interaction_in_progress(False)

    def _check_auto_switch(self) -> None:
        """Evaluate recent metrics and propose a strategy switch if warranted."""
        if len(self._recent_metrics) < 5:
            return
        window = self._recent_metrics[-5:]

        total_decisions = sum(m["total"] for m in window)
        if total_decisions == 0:
            return

        hold_count = sum(m["holds"] for m in window)
        hold_rate = hold_count / total_decisions

        sentiments = [m["avg_sentiment"] for m in window if m["avg_sentiment"] is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        trends = [m["dominant_trend"] for m in window]
        up_count = trends.count("up")
        down_count = trends.count("down")
        if up_count >= 3:
            avg_trend = "up"
        elif down_count >= 3:
            avg_trend = "down"
        else:
            avg_trend = "flat"

        pnl_values = [m["pnl_pct"] for m in window if m["pnl_pct"] is not None]
        pnl_trend = 0.0
        if len(pnl_values) >= 2:
            pnl_trend = pnl_values[-1] - pnl_values[0]

        new_id, reason = strategy_library.recommend_switch(
            self._current_strategy, hold_rate, pnl_trend, avg_sentiment, avg_trend
        )
        if not new_id:
            return

        new_name = strategy_library.get(new_id)["name"]
        self._dashboard.log("", "info")
        self._dashboard.log(
            f"  [AUTO-SWITCH] Agent recommends switching strategy → {new_name}", "warn"
        )
        accepted = self._dashboard.ask_strategy_switch(new_id, new_name, reason)
        if accepted:
            self._apply_strategy_switch(new_id)

    def _run_cycle(self) -> None:
        if self._bm.change_requested:
            self._dashboard.log("Applying behaviour change…", "warn")
            self._handle_behavior_change()
            self._dashboard.log(f"Behaviour updated → {self._bm.active_prompt[:60]}", "ok")

        self._cycle += 1
        self._session["cycle"] = self._cycle
        session_id = self._session.get("session_id", "")
        active_prompt = self._bm.active_prompt
        if self._disruptor:
            self._disruptor.update_cycle(self._cycle)
        t_wait = self._at.t_wait()
        t_behavior = self._at.t_behavior()
        market_open = self._te.is_market_open()

        strategy_name = strategy_library.get(self._current_strategy)["name"]
        tickers_str = ", ".join(self._tickers)
        mkt_str = "OPEN" if market_open else "CLOSED"
        self._dashboard.log(
            f"── Cycle {self._cycle} ─────────────────  "
            f"market:{mkt_str}  strategy:{strategy_name}  "
            f"t_wait:{t_wait}s  t_behavior:{t_behavior}s  ticker:[{tickers_str}]",
            "cycle",
        )

        veto_triggered = False
        last_entry: dict = {}
        cycle_rows: list[dict] = []
        pnl_pct = 0.0
        cash = 100_000.0
        portfolio_value = 100_000.0
        mode = "normal"

        _initial_portfolio = self._te.get_portfolio()
        _held = set(
            _initial_portfolio.data.get("positions", {}).keys()
        ) if _initial_portfolio.ok else set()
        _extra = [t for t in _held if t not in self._tickers]
        effective_tickers = list(self._tickers) + _extra

        if _extra:
            self._dashboard.log(
                f"  + open positions not in discovery: [{', '.join(_extra)}] — included in cycle",
                "warn",
            )

        _cycle_interrupted = False
        for ticker in effective_tickers:
            if self._pause_requested:
                self._in_wait_phase = True
                self._handle_mid_cycle_injection(active_prompt)
                self._in_wait_phase = False
                self._pause_requested = False
                _cycle_interrupted = True
                break

            if ticker in self._te._blacklisted:
                self._dashboard.log(f"  {ticker} → blacklisted, skip", "warn")
                continue
            try:
                # 1. OUTCOME UPDATE
                self._dashboard.log(f"  {ticker} → [1] fetching current price…", "info")
                price_result = self._te.get_price(ticker)
                if price_result.ok:
                    journal_module.outcome_update(
                        ticker, price_result.data["price"], session_id
                    )
                    stale_tag = " [STALE]" if price_result.stale else ""
                    self._dashboard.log(
                        f"  {ticker} → price ${price_result.data['price']:,.2f}{stale_tag}", "ok"
                    )
                else:
                    self._dashboard.log(f"  {ticker} → price unavailable (using cache)", "warn")

                # 2+3. OBSERVE + SENTIMENT (concurrent)
                self._dashboard.log(f"  {ticker} → [2] fetching bars/news…", "info")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    fut_bars = pool.submit(self._te.get_bars, ticker, config.TECHNICAL_BARS_LOOKBACK)
                    fut_news = pool.submit(self._te.get_news, ticker)
                    bars_result = fut_bars.result(timeout=30)
                    news_result = fut_news.result(timeout=30)

                portfolio_result = _initial_portfolio

                regular_articles = news_result.data.get("articles", []) if news_result.ok else []

                # Fetch disruptor articles and merge (disruptor first, regular after; dedup by title)
                disruptor_articles: list[dict] = []
                if self._disruptor:
                    disruptor_articles = self._disruptor.get_articles(ticker, max_age_seconds=3600)
                    for a in disruptor_articles:
                        a["_from_disruptor"] = True
                    if disruptor_articles:
                        self._dashboard.print_disruptor_articles(ticker, disruptor_articles)

                seen_titles: set[str] = {a.get("title", "").lower() for a in disruptor_articles}
                combined_articles = list(disruptor_articles) + [
                    a for a in regular_articles
                    if a.get("title", "").lower() not in seen_titles
                ]
                article_count = len(combined_articles)

                self._dashboard.log(
                    f"  {ticker} → bars:{'ok' if bars_result.ok else 'err'}  "
                    f"news:{len(regular_articles)} + {len(disruptor_articles)} disruptor = {article_count} tot  "
                    f"portfolio:{'ok' if portfolio_result.ok else 'err'}",
                    "ok" if (bars_result.ok and portfolio_result.ok) else "warn",
                )

                self._dashboard.log(f"  {ticker} → [3] sentiment analysis ({article_count} art.)…", "info")
                from src.agent import sentiment as sentiment_module
                sentiment_data, keywords_and_relevance = sentiment_module.analyse(
                    ticker, combined_articles, active_prompt, t_behavior
                )
                self._dashboard.log(
                    f"  {ticker} → sentiment: {sentiment_data['label']} "
                    f"({sentiment_data['score']:+.2f})",
                    "ok",
                )

                # Persist regular articles to news_log (disruptor articles go to their own file)
                non_duped_regular = combined_articles[len(disruptor_articles):]
                kw_for_regular = keywords_and_relevance[len(disruptor_articles):]
                news_log.write_articles(
                    articles=non_duped_regular,
                    ticker=ticker,
                    cycle=self._cycle,
                    session_id=session_id,
                    sentiment_score=sentiment_data["score"],
                    keywords_and_relevance=kw_for_regular,
                )

                # F1: unified news context for LLM prompt
                news_context = news_log.build_news_context_for_prompt(
                    ticker=ticker,
                    current_cycle=self._cycle,
                    history_cycles=config.NEWS_CONTEXT_HISTORY_CYCLES,
                    max_articles=config.NEWS_CONTEXT_MAX_ARTICLES,
                    min_relevance_historical=config.NEWS_CONTEXT_MIN_RELEVANCE,
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
                    f"  {ticker} → mode:{mode.upper()}  P&L:{pnl_pct:+.2%}  "
                    f"cash:${cash:,.0f}  portfolio:${portfolio_value:,.0f}",
                    "warn" if mode == "conservative" else "info",
                )

                # 5. MEMORY CONTEXT
                memory_context = self._mm.build_context(ticker)
                imitative_hints = self._il.build_hints(active_prompt, ticker)
                imitative_source = self._il.get_active_strategy_id(active_prompt)

                # Rebuild correlation matrix every NCCI_REBUILD_EVERY cycles (daemon)
                if self._cycle % config.NCCI_REBUILD_EVERY == 0:
                    threading.Thread(
                        target=self._correlation_engine.rebuild,
                        daemon=True,
                    ).start()

                # Register dynamically mentioned tickers from news
                for article in combined_articles:
                    for mentioned in _extract_mentioned_tickers(article):
                        self._correlation_engine.register_dynamic_ticker(mentioned)

                # Build correlation section for prompt
                correlation_section = self._correlation_engine.build_prompt_section(
                    tickers=effective_tickers,
                    positions=positions,
                    threshold=config.NCCI_THRESHOLD_DISPLAY,
                )

                price = price_result.data.get("price", 0.0) if price_result.ok else 0.0
                price_timestamp = price_result.data.get("timestamp", _now_utc()) if price_result.ok else _now_utc()
                closes = bars_result.data.get("closes", []) if bars_result.ok else []
                ma5 = bars_result.data.get("ma", 0.0) if bars_result.ok else 0.0
                trend = bars_result.data.get("trend", "flat") if bars_result.ok else "flat"
                stale = not price_result.ok or price_result.stale
                staleness_seconds = price_result.staleness_seconds if not price_result.ok else 0
                data_ok = price_result.ok

                # Unrealized P&L for cycle summary column
                unrealized_pnl_pct: float | None = None
                avg_entry_price: float = 0.0
                if ticker in positions and price > 0:
                    pos = positions[ticker]
                    avg_entry_price = pos.get("avg_entry_price", 0.0)
                    qty_held = pos.get("qty", 0)
                    if avg_entry_price > 0 and qty_held > 0:
                        unrealized_pnl_pct = (price - avg_entry_price) / avg_entry_price
                        level = "ok" if unrealized_pnl_pct >= 0.02 else "warn" if unrealized_pnl_pct <= -0.03 else "info"
                        self._dashboard.log(
                            f"  {ticker} → position P&L: {unrealized_pnl_pct:+.1%} "
                            f"(entry ${avg_entry_price:.2f} → now ${price:.2f})",
                            level,
                        )

                # F2: position manager — register/update, compute adaptive thresholds
                position_context = ""
                if self._pm is not None:
                    if ticker in positions and ticker not in self._pm._states:
                        pos = positions[ticker]
                        self._pm.on_new_position(
                            ticker=ticker,
                            entry_price=pos.get("avg_entry_price", price),
                            entry_cycle=self._cycle,
                            qty=int(pos.get("qty", 0)),
                        )
                    elif ticker not in positions and ticker in self._pm._states:
                        self._pm.on_position_closed(ticker)
                    if price > 0:
                        self._pm.update_price(ticker, price)
                    # rolling sentiment list from recent cycle rows
                    recent_sentiments = [
                        r["sentiment_score"] for r in cycle_rows[-config.POSITION_SENTIMENT_TREND_WINDOW:]
                    ]
                    self._pm.update_thresholds(
                        ticker=ticker,
                        bars_closes=closes,
                        recent_sentiments=recent_sentiments,
                        mode=mode,
                        user_stop_loss_pct=self._session.get("user_stop_loss_pct"),
                        user_take_profit_pct=self._session.get("user_take_profit_pct"),
                    )
                    position_context = self._pm.build_position_context(ticker, price, self._cycle)

                # F3: technical analysis
                tech_signals = technical_analyser.analyse(
                    closes=closes,
                    current_price=price,
                    rsi_period=config.TECHNICAL_RSI_PERIOD,
                    bb_period=config.TECHNICAL_BB_PERIOD,
                    bb_std=config.TECHNICAL_BB_STD,
                    rsi_overbought=config.TECHNICAL_RSI_OVERBOUGHT,
                    rsi_oversold=config.TECHNICAL_RSI_OVERSOLD,
                    bb_squeeze_pct=config.TECHNICAL_BB_SQUEEZE_PCT,
                )
                technical_signals_str = tech_signals.prompt_section
                if tech_signals.rsi.valid or tech_signals.bollinger.valid:
                    self._dashboard.log(
                        f"  {ticker} → RSI:{tech_signals.rsi.value:.1f}  "
                        f"BB:%B={tech_signals.bollinger.pct_b:.2f}",
                        "info",
                    )

                # F4: user preferences section
                user_preferences_section = ""
                if self._upe is not None:
                    user_preferences_section = self._upe.build_prompt_section()

                strat_str = f"  strategy:{imitative_source}" if imitative_source else ""
                self._dashboard.log(
                    f"  {ticker} → MA5:${ma5:.2f}  trend:{trend}{strat_str}", "info"
                )

                # F2: stop-loss pre-check — hard stop before LLM reasoning
                if self._pm is not None and ticker in positions and price > 0:
                    if self._pm.check_stop_loss(ticker, price):
                        state = self._pm.get_state(ticker)
                        sl_pct = state.thresholds.stop_loss_pct if state else -5.0
                        explanation = (state.thresholds.explanation if state else
                                       f"Price fell below stop-loss threshold ({sl_pct:.1f}%)")
                        self._dashboard.log(
                            f"  {ticker} → STOP-LOSS triggered ({sl_pct:.1f}%)", "err"
                        )
                        sl_confirm = self._dashboard.print_stop_loss_proposal(
                            ticker=ticker,
                            current_price=price,
                            entry_price=avg_entry_price,
                            pnl_pct=unrealized_pnl_pct or 0.0,
                            qty=int(positions[ticker].get("qty", 0)),
                            explanation=explanation,
                            timeout_seconds=20,
                        )
                        if sl_confirm.get("confirmed"):
                            qty_sl = int(positions[ticker].get("qty", 0))
                            order_result_sl = self._broker.place_order(ticker, "sell", qty_sl)
                            if order_result_sl.get("ok"):
                                self._dashboard.log(
                                    f"  {ticker} → stop-loss executed: sold {qty_sl} shares", "ok"
                                )
                                self._pm.on_position_closed(ticker)
                            else:
                                self._dashboard.log(
                                    f"  {ticker} → stop-loss rejected by broker: "
                                    f"{order_result_sl.get('reason','?')}",
                                    "err",
                                )
                            # Record and continue to next ticker
                            journal_module.log_error(
                                source="AgentLoop",
                                error=f"Stop-loss triggered for {ticker} at {price:.2f}",
                                ticker=ticker,
                                session_id=session_id,
                            )
                            continue

                # 6. THINK
                self._dashboard.log(
                    f"  {ticker} → [6] LLM reasoning [{strategy_name}] (t_behavior:{t_behavior}s)…",
                    "info",
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
                    strategy_id=self._current_strategy,
                    correlation_section=correlation_section,
                    news_context=news_context,
                    position_context=position_context,
                    technical_signals=technical_signals_str,
                    user_preferences_section=user_preferences_section,
                )

                stale_tag = f"  -stale:{decision['stale_penalty']:.2f}" if decision["stale_penalty"] > 0 else ""
                action_color = {"buy": "ok", "sell": "err", "hold": "warn"}
                self._dashboard.log(
                    f"  {ticker} → [{decision['action'].upper()}]  "
                    f"conf:{decision['confidence']:.2f} (raw:{decision['confidence_raw']:.2f}{stale_tag})",
                    action_color.get(decision["action"], "info"),
                )
                # Full reasoning — always shown, not truncated
                self._dashboard.log(
                    f"  {ticker} → REASON: {decision['reasoning']}",
                    "info",
                )

                # 7. ACT
                base_threshold = (
                    config.CONFIDENCE_THRESHOLD_CONSERVATIVE if mode == "conservative"
                    else config.CONFIDENCE_THRESHOLD_NORMAL
                )
                # F4: effective threshold from preference engine
                threshold = (
                    self._upe.get_effective_confidence_threshold(base_threshold)
                    if self._upe is not None else base_threshold
                )
                action = decision["action"]

                # F4: conflict detection post-LLM
                if self._upe is not None and action in ("buy", "sell"):
                    conflict = self._upe.check_conflict(
                        ticker=ticker,
                        proposed_action=action,
                        current_pnl_pct=pnl_pct,
                        ticker_pnl_pct=unrealized_pnl_pct or 0.0,
                        sentiment_score=sentiment_data["score"],
                        mode=mode,
                    )
                    if conflict:
                        self._upe.apply_minimum_modification(conflict, self._session)
                        self._dashboard.print_preference_conflict(ticker, conflict)
                        modified = conflict.get("modified_action", "")
                        if modified and modified != action:
                            self._dashboard.log(
                                f"  {ticker} → action modified by preferences: "
                                f"{action.upper()} → {modified.upper()}",
                                "warn",
                            )
                            action = modified

                order_result: dict = {"ok": False, "order_id": None}
                decision_source = "agent"

                if action in ("buy", "sell") and decision["confidence"] >= threshold and market_open:
                    # F4: effective position size
                    base_pct = (
                        config.MAX_POSITION_PCT_CONSERVATIVE if mode == "conservative"
                        else config.MAX_POSITION_PCT_NORMAL
                    )
                    effective_pct = (
                        self._upe.get_effective_position_pct(base_pct)
                        if self._upe is not None else base_pct
                    )
                    qty = self._broker.compute_qty(
                        price=price,
                        cash=cash,
                        mode=mode,
                        action=action,
                        ticker=ticker,
                        positions=positions,
                        portfolio_value=portfolio_value,
                        effective_position_pct=effective_pct,
                    )
                    self._dashboard.log(
                        f"  {ticker} → qty computed:{qty}  "
                        f"(cash:${cash:,.0f}  portfolio:${portfolio_value:,.0f}  "
                        f"price:${price:.2f}  pos_pct:{effective_pct:.1%})",
                        "info",
                    )
                    if qty > 0:
                        self._dashboard.log(
                            f"  {ticker} → [7] placing order {action.upper()} qty:{qty}…", "action"
                        )
                        order_result = self._broker.place_order(ticker, action, qty)
                        if order_result.get("ok"):
                            self._dashboard.log(
                                f"  {ticker} → order accepted (id:{str(order_result.get('order_id','?'))[:8]}…)",
                                "ok",
                            )
                            # F2: register new position on buy
                            if action == "buy" and self._pm is not None:
                                self._pm.on_new_position(
                                    ticker=ticker,
                                    entry_price=price,
                                    entry_cycle=self._cycle,
                                    qty=qty,
                                )
                            # Show realized P&L on sell; unregister position
                            if action == "sell" and avg_entry_price > 0:
                                realized_pnl = (price - avg_entry_price) / avg_entry_price
                                pnl_level = "ok" if realized_pnl >= 0 else "warn"
                                self._dashboard.log(
                                    f"  {ticker} → SOLD {qty} shares at ${price:.2f}  "
                                    f"(entry: ${avg_entry_price:.2f} | P&L: {realized_pnl:+.1%})",
                                    pnl_level,
                                )
                                if self._pm is not None:
                                    self._pm.on_position_closed(ticker)
                        else:
                            self._dashboard.log(
                                f"  {ticker} → order rejected: {order_result.get('reason','?')}",
                                "err",
                            )
                    else:
                        action = "hold"
                        self._dashboard.log(f"  {ticker} → qty=0, forced hold", "warn")
                else:
                    reasons: list[str] = []
                    if action not in ("buy", "sell"):
                        reasons.append(f"action={action}")
                    elif decision["confidence"] < threshold:
                        reasons.append(f"conf {decision['confidence']:.2f} < threshold {threshold:.2f}")
                    if not market_open:
                        reasons.append("market closed")
                    action = "hold"
                    decision_source = "autonomous_timeout"
                    self._dashboard.log(
                        f"  {ticker} → autonomous HOLD ({'; '.join(reasons)})", "warn"
                    )

                # 7b. MARK NEWS DECISION + PRINT DECISION PANEL
                news_log.mark_decision(
                    ticker=ticker,
                    cycle=self._cycle,
                    session_id=session_id,
                    decision=action,
                )
                display_articles = news_log.read_for_display(
                    ticker=ticker,
                    cycle=self._cycle,
                    session_id=session_id,
                    max_articles=config.NEWS_DISPLAY_MAX_ARTICLES,
                )
                self._dashboard.print_decision_news(
                    ticker=ticker,
                    action=action,
                    confidence=decision["confidence"],
                    caption=decision.get("caption", ""),
                    articles=display_articles,
                )

                if self._telegram:
                    self._telegram.notify_action(
                        ticker=ticker,
                        action=action,
                        confidence=decision["confidence"],
                        reasoning=decision.get("reasoning", ""),
                        caption=decision.get("caption", ""),
                        articles=display_articles,
                        disruptor_articles=disruptor_articles,
                    )

                # 8. RECORD
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
                journal_module.write_news_entries(
                    articles=regular_articles,
                    ticker=ticker,
                    cycle=self._cycle,
                    session_id=session_id,
                    sentiment_score=sentiment_data["score"],
                    decision_triggered=action,
                )
                self._mm.update(entry)
                last_entry = entry

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
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "avg_entry_price": avg_entry_price,
                    "disruptor_used": disruptor_articles,
                })

                # 9. UPDATE DASHBOARD
                self._dashboard.update(
                    ticker, entry, portfolio_result.data if portfolio_result.ok else {},
                    t_wait, t_behavior,
                )

                # NEWS VETO
                if sentiment_data["score"] < -0.7 and ticker in positions and action == "hold":
                    veto_triggered = True
                    self._dashboard.log(
                        f"  {ticker} → NEWS VETO triggered (sentiment {sentiment_data['score']:.2f})",
                        "err",
                    )
                    news_log.mark_decision(
                        ticker=ticker,
                        cycle=self._cycle,
                        session_id=session_id,
                        decision="veto",
                    )

            except Exception as exc:
                self._dashboard.log(f"  {ticker} → cycle ERROR: {exc}", "err")
                journal_module.log_error(
                    source="AgentLoop", error=f"Ticker {ticker} cycle error: {exc}",
                    ticker=ticker, session_id=session_id,
                )

        # 10. ADAPTIVE TIMEOUT UPDATE
        if self._cycle % 5 == 0:
            self._dashboard.log("  [10] Recalibrating adaptive timeout (background)…", "info")
            threading.Thread(target=self._at.calibrate, daemon=True).start()

        # 11. UPDATE RECENT METRICS for auto-switch
        if cycle_rows:
            holds = sum(1 for r in cycle_rows if r["action"] == "hold")
            sentiments = [r["sentiment_score"] for r in cycle_rows]
            trends = [r["trend"] for r in cycle_rows]
            dominant_trend = max(set(trends), key=trends.count) if trends else "flat"
            self._recent_metrics.append({
                "cycle": self._cycle,
                "total": len(cycle_rows),
                "holds": holds,
                "avg_sentiment": sum(sentiments) / len(sentiments) if sentiments else None,
                "dominant_trend": dominant_trend,
                "pnl_pct": pnl_pct,
            })
            if len(self._recent_metrics) > 20:
                self._recent_metrics = self._recent_metrics[-20:]

        if _cycle_interrupted:
            self._dashboard.log("  Cycle interrupted — new behaviour applied at next cycle.", "warn")
            self._sm.save(self._session)
            return

        wait_seconds = 2 if veto_triggered else self._at.t_wait()
        if veto_triggered:
            self._dashboard.log("  [NEWS VETO] wait reduced to 2s", "err")

        if cycle_rows:
            self._dashboard.print_cycle_summary(
                cycle=self._cycle,
                rows=cycle_rows,
                pnl_pct=pnl_pct,
                portfolio_value=portfolio_value,
                cash=cash,
                mode=mode,
                wait_seconds=wait_seconds,
                veto=veto_triggered,
                strategy_name=strategy_name,
            )
            if self._telegram:
                self._telegram.notify_cycle_end(
                    cycle=self._cycle,
                    rows=cycle_rows,
                    portfolio={"portfolio_value": portfolio_value, "cash": cash, "pnl_pct": pnl_pct},
                    pnl_pct=pnl_pct,
                    strategy_name=strategy_name,
                    active_prompt=active_prompt,
                    tickers=effective_tickers,
                )

        if len(effective_tickers) >= 2:
            self._dashboard.print_correlation_matrix(
                tickers=effective_tickers,
                engine=self._correlation_engine,
            )

        # Auto-switch check every 5 cycles (before wait, so user is engaged)
        if self._cycle % 5 == 0:
            self._in_wait_phase = True  # pause stdin_listener during interactive prompt
            self._check_auto_switch()
            self._in_wait_phase = False

        self._in_wait_phase = True
        result = self._dashboard.wait_for_user_input(wait_seconds)
        self._in_wait_phase = False

        interaction_context = {
            "active_prompt": active_prompt,
            "tickers": effective_tickers,
            "pnl_pct": pnl_pct if cycle_rows else 0.0,
            "mode": mode if cycle_rows else "normal",
            "recent_actions": "  ".join(
                f"{r['ticker']}:{r['action'].upper()}" for r in cycle_rows[-3:]
            ),
        }

        if result["source"] in ("prompt_append", "prompt_change", "questionnaire", "strategy_select"):
            self._in_wait_phase = True
            self._interaction_running = True
            self._run_interaction(result["source"], interaction_context)
            self._in_wait_phase = False
        elif result["source"] == "override" and last_entry:
            ov = result["data"]
            self._dashboard.log(
                f"  Manual override: {ov['ticker']} {ov['action'].upper()} qty:{ov['qty']}", "action"
            )
            order_result = self._broker.place_order(ov["ticker"], ov["action"], ov["qty"])
            ov_entry = dict(last_entry)
            ov_entry["action"] = ov["action"]
            ov_entry["decision_source"] = "user_override"
            ov_entry["order_id"] = order_result.get("order_id")
            ov_entry["ts"] = _now_utc()
            journal_module.write_entry(ov_entry)
            ok_str = "accepted" if order_result.get("ok") else f"rejected: {order_result.get('reason','?')}"
            self._dashboard.log(f"  Override → {ok_str}", "ok" if order_result.get("ok") else "err")
        elif result["source"] == "confirmed":
            self._dashboard.log("  → Confirmed by user.", "info")

        self._sm.save(self._session)

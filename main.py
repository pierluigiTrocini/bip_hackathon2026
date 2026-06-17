"""
BIP Hackathon 2026 — Trading Agent
Entry point.

Usage:
    uv run python main.py        # silent loop (LLM output hidden during loop phase)
    uv run python main.py -v     # verbose loop (LLM output visible, same as discovery phase)

The agent initialises, asks for session resume or new prompt,
then runs autonomously. Press Ctrl+C to stop gracefully.
"""
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("agent.log")],
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    _parser = argparse.ArgumentParser(description="BIP Trading Agent 2026")
    _parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show LLM streaming output during the agent loop phase",
    )
    _args = _parser.parse_args()

    from src.agent import llm_stream
    llm_stream.LOOP_VERBOSE = _args.verbose

    from src.agent import config
    from src.agent.adaptive_timeout import AdaptiveTimeout
    from src.agent.tool_executor import ToolExecutor
    from src.agent.journal import read_session_summary
    from src.agent.memory_manager import MemoryManager
    from src.agent.imitative_layer import ImiativeLayer
    from src.agent.reasoner import Reasoner
    from src.agent.broker import Broker
    from src.agent.session import SessionManager
    from src.agent.behavior import BehaviorManager
    from src.agent.discovery import DiscoveryAgent
    from src.agent.correlation_engine import CorrelationEngine
    from src.agent.disruptor import MarketDisruptor
    from src.agent.position_manager import PositionManager
    from src.agent.user_preference_engine import UserPreferenceEngine
    from src.agent.loop import AgentLoop
    from ui.dashboard import Dashboard

    dashboard = Dashboard()
    dashboard.log("BIP Trading Agent 2026 — starting…", "info")

    # Step 1: Detect previous session
    session_mgr = SessionManager()
    dashboard.log("Looking for previous session…", "info")
    previous = session_mgr.detect_previous_session()

    resuming = False
    if previous:
        dashboard.log(
            f"Session found: {str(previous.get('session_id','?'))[:8]}… "
            f"(status: {previous.get('status','?')})",
            "warn",
        )
        summary = read_session_summary(previous["session_id"])
        dashboard.print_resoconto(summary)
        choice = session_mgr.ask_resume_or_new()
        if choice == "resume":
            session = session_mgr.resume(previous)
            prompt = session["active_prompt"]
            resuming = True
            dashboard.log(f"Session resumed. Prompt: {prompt[:60]}", "ok")
        else:
            prompt = input("\nEnter the new agent behaviour: ").strip()
            session = session_mgr.create_new(prompt)
            dashboard.log(f"New session created. Prompt: {prompt[:60]}", "ok")
    else:
        dashboard.log("No previous session found. Starting fresh.", "info")
        prompt = input("Enter the agent behaviour (e.g. 'focus on green investments'): ").strip()
        session = session_mgr.create_new(prompt)
        dashboard.log(f"Session created: {str(session.get('session_id','?'))[:8]}…", "ok")

    # Step 2: Calibrate adaptive timeout
    adaptive_timeout = AdaptiveTimeout()
    dashboard.log("Calibrating adaptive timeout (3 API pings + 3 Ollama pings)…", "info")
    adaptive_timeout.calibrate()
    s = adaptive_timeout.summary()
    dashboard.log(
        f"Calibration done → api:{s['api_avg']:.0f}ms  "
        f"ollama:{s['ollama_avg']:.0f}ms  t_wait:{s['t_wait']}s  t_behavior:{s['t_behavior']}s",
        "ok",
    )

    # Step 3: Initialise modules
    dashboard.log("Initialising modules…", "info")
    tool_executor       = ToolExecutor(adaptive_timeout)
    memory_manager      = MemoryManager()
    imitative_layer     = ImiativeLayer()
    reasoner            = Reasoner()
    broker              = Broker()
    behavior_manager    = BehaviorManager(session, adaptive_timeout)
    discovery_agent     = DiscoveryAgent()
    correlation_engine  = CorrelationEngine()
    disruptor           = MarketDisruptor()
    position_manager    = PositionManager()
    preference_engine   = UserPreferenceEngine(session)
    discovery_agent._session_id = session.get("session_id", "")
    dashboard.log(
        f"Models: {config.OLLAMA_REASONING_MODEL} + {config.OLLAMA_SENTIMENT_MODEL}",
        "ok",
    )

    # F4: extract user preferences from initial/resumed prompt
    t_behavior_init = adaptive_timeout.t_behavior()
    dashboard.log("Extracting user preferences from prompt…", "info")
    preference_engine.extract_from_prompt(prompt, t_behavior_init)
    preference_engine.compute_derived_parameters()
    dashboard.log("User preferences extracted.", "ok")

    # Step 3b: Discovery phase (skip if resuming — use saved tickers)
    if resuming and session.get("tickers"):
        confirmed_tickers: list[str] = session["tickers"]
        dashboard.log(
            f"Resuming session — tickers from previous session: {', '.join(confirmed_tickers)}",
            "ok",
        )
    else:
        dashboard.log("", "info")
        dashboard.log("━━━ DISCOVERY PHASE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        t_behavior = adaptive_timeout.t_behavior()
        discovery_prompt = prompt
        confirmed_tickers: list[str] = []

        while True:
            dashboard.log(f"Prompt: \"{discovery_prompt[:100]}\"", "info")
            dashboard.log("Analysing market news and selecting tickers…", "info")

            candidates = discovery_agent.discover(
                discovery_prompt, tool_executor, t_behavior, dashboard
            )
            dashboard.print_discovery_candidates(candidates)
            # Show existing portfolio positions alongside candidates before asking for confirmation
            _pf = tool_executor.get_portfolio()
            _pos = _pf.data.get("positions", {}) if _pf.ok else {}
            dashboard.print_portfolio_positions(_pos)
            result = dashboard.confirm_or_reprompt(candidates, timeout_seconds=adaptive_timeout.t_wait())

            if result["action"] == "confirm":
                confirmed_tickers = result["tickers"]
                break

            # User provided a new prompt — restart discovery
            discovery_prompt = result["new_prompt"]
            session["active_prompt"] = discovery_prompt
            dashboard.log("", "info")
            dashboard.log("━━━ NEW DISCOVERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        session["tickers"] = confirmed_tickers
        session_mgr.save(session)

        dashboard.log(
            f"Confirmed tickers for this session: {', '.join(confirmed_tickers)}",
            "ok",
        )
        dashboard.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        dashboard.log("", "info")

    # Step 3c: Show portfolio positions on resume (discovery loop handles new sessions)
    if resuming:
        dashboard.log("Fetching open portfolio positions…", "info")
        _portfolio_result = tool_executor.get_portfolio()
        _positions = _portfolio_result.data.get("positions", {}) if _portfolio_result.ok else {}
        dashboard.print_portfolio_positions(_positions)
        # F2: restore position manager state from journal
        position_manager.load_from_journal(
            positions=_positions,
            session_id=session.get("session_id", ""),
            journal_path=config.JOURNAL_PATH,
        )

    # Step 4: Start loop
    disruptor.start(confirmed_tickers, session.get("session_id", ""))
    dashboard.log("MarketDisruptor started in background.", "ok")

    loop = AgentLoop(
        session=session,
        adaptive_timeout=adaptive_timeout,
        tool_executor=tool_executor,
        memory_manager=memory_manager,
        imitative_layer=imitative_layer,
        reasoner=reasoner,
        broker=broker,
        behavior_manager=behavior_manager,
        session_manager=session_mgr,
        dashboard=dashboard,
        correlation_engine=correlation_engine,
        tickers=confirmed_tickers,
        disruptor=disruptor,
        position_manager=position_manager,
        preference_engine=preference_engine,
    )

    dashboard.log("Loop started. Ctrl+C to stop.", "ok")
    try:
        loop.start()
    except KeyboardInterrupt:
        dashboard.log("Shutdown requested by user (Ctrl+C).", "warn")
    finally:
        loop.stop()
        disruptor.stop()
        dashboard.log("Cancelling open orders…", "warn")
        broker.cancel_all_orders()
        session_mgr.mark_paused(session)
        dashboard.log("Session paused.", "ok")
        summary = read_session_summary(session["session_id"])
        dashboard.print_resoconto(summary)
        dashboard.print_shutdown_message()


if __name__ == "__main__":
    main()

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
    from src.agent.loop import AgentLoop
    from ui.dashboard import Dashboard

    dashboard = Dashboard()
    dashboard.log("BIP Trading Agent 2026 — avvio…", "info")

    # Step 1: Detect previous session
    session_mgr = SessionManager()
    dashboard.log("Ricerca sessione precedente…", "info")
    previous = session_mgr.detect_previous_session()

    resuming = False
    if previous:
        dashboard.log(
            f"Sessione trovata: {str(previous.get('session_id','?'))[:8]}… "
            f"(stato: {previous.get('status','?')})",
            "warn",
        )
        summary = read_session_summary(previous["session_id"])
        dashboard.print_resoconto(summary)
        choice = session_mgr.ask_resume_or_new()
        if choice == "resume":
            session = session_mgr.resume(previous)
            prompt = session["active_prompt"]
            resuming = True
            dashboard.log(f"Sessione ripresa. Prompt: {prompt[:60]}", "ok")
        else:
            prompt = input("\nInserisci il comportamento del nuovo agente: ").strip()
            session = session_mgr.create_new(prompt)
            dashboard.log(f"Nuova sessione creata. Prompt: {prompt[:60]}", "ok")
    else:
        dashboard.log("Nessuna sessione precedente. Nuova sessione.", "info")
        prompt = input("Inserisci il comportamento dell'agente (es. 'orientato a scelte green'): ").strip()
        session = session_mgr.create_new(prompt)
        dashboard.log(f"Sessione creata: {str(session.get('session_id','?'))[:8]}…", "ok")

    # Step 2: Calibrate adaptive timeout
    adaptive_timeout = AdaptiveTimeout()
    dashboard.log("Calibrazione timeout adattivo (3 ping API + 3 ping Ollama)…", "info")
    adaptive_timeout.calibrate()
    s = adaptive_timeout.summary()
    dashboard.log(
        f"Calibrazione completata → api:{s['api_avg']:.0f}ms  "
        f"ollama:{s['ollama_avg']:.0f}ms  t_wait:{s['t_wait']}s  t_behavior:{s['t_behavior']}s",
        "ok",
    )

    # Step 3: Initialise modules
    dashboard.log("Inizializzazione moduli…", "info")
    tool_executor       = ToolExecutor(adaptive_timeout)
    memory_manager      = MemoryManager()
    imitative_layer     = ImiativeLayer()
    reasoner            = Reasoner()
    broker              = Broker()
    behavior_manager    = BehaviorManager(session, adaptive_timeout)
    discovery_agent     = DiscoveryAgent()
    correlation_engine  = CorrelationEngine()
    disruptor           = MarketDisruptor()
    discovery_agent._session_id = session.get("session_id", "")
    dashboard.log(
        f"Modelli: {config.OLLAMA_REASONING_MODEL} + {config.OLLAMA_SENTIMENT_MODEL}",
        "ok",
    )

    # Step 3b: Discovery phase (skip if resuming — use saved tickers)
    if resuming and session.get("tickers"):
        confirmed_tickers: list[str] = session["tickers"]
        dashboard.log(
            f"Ripresa sessione — ticker dalla sessione precedente: {', '.join(confirmed_tickers)}",
            "ok",
        )
    else:
        dashboard.log("", "info")
        dashboard.log("━━━ FASE DI DISCOVERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        t_behavior = adaptive_timeout.t_behavior()
        discovery_prompt = prompt
        confirmed_tickers: list[str] = []

        while True:
            dashboard.log(f"Prompt: \"{discovery_prompt[:100]}\"", "info")
            dashboard.log("Analisi news di mercato e selezione ticker in corso…", "info")

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
            dashboard.log("━━━ NUOVA DISCOVERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        session["tickers"] = confirmed_tickers
        session_mgr.save(session)

        dashboard.log(
            f"Ticker confermati per questa sessione: {', '.join(confirmed_tickers)}",
            "ok",
        )
        dashboard.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        dashboard.log("", "info")

    # Step 3c: Show portfolio positions on resume (discovery loop handles new sessions)
    if resuming:
        dashboard.log("Recupero posizioni aperte nel portfolio…", "info")
        _portfolio_result = tool_executor.get_portfolio()
        _positions = _portfolio_result.data.get("positions", {}) if _portfolio_result.ok else {}
        dashboard.print_portfolio_positions(_positions)

    # Step 4: Start loop
    disruptor.start(confirmed_tickers, session.get("session_id", ""))
    dashboard.log("MarketDisruptor avviato in background.", "ok")

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
    )

    dashboard.log("Loop avviato. Ctrl+C per fermare.", "ok")
    try:
        loop.start()
    except KeyboardInterrupt:
        dashboard.log("Shutdown richiesto dall'utente (Ctrl+C).", "warn")
    finally:
        loop.stop()
        disruptor.stop()
        dashboard.log("Cancellazione ordini aperti…", "warn")
        broker.cancel_all_orders()
        session_mgr.mark_paused(session)
        dashboard.log("Sessione sospesa.", "ok")
        summary = read_session_summary(session["session_id"])
        dashboard.print_resoconto(summary)
        dashboard.print_shutdown_message()


if __name__ == "__main__":
    main()

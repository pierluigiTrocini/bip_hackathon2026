"""
BIP Hackathon 2026 — Trading Agent
Entry point.

Usage:
    uv run python main.py

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
    from src.agent.loop import AgentLoop
    from ui.dashboard import Dashboard

    dashboard = Dashboard()
    dashboard.log("BIP Trading Agent 2026 — avvio…", "info")

    # Step 1: Detect previous session
    session_mgr = SessionManager()
    dashboard.log("Ricerca sessione precedente…", "info")
    previous = session_mgr.detect_previous_session()

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
    tool_executor    = ToolExecutor(adaptive_timeout)
    memory_manager   = MemoryManager()
    imitative_layer  = ImiativeLayer()
    reasoner         = Reasoner()
    broker           = Broker()
    behavior_manager = BehaviorManager(session, adaptive_timeout)
    dashboard.log(
        f"Moduli pronti. Ticker: {', '.join(config.TICKERS)}  "
        f"Modelli: {config.OLLAMA_REASONING_MODEL} + {config.OLLAMA_SENTIMENT_MODEL}",
        "ok",
    )

    # Step 4: Start loop
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
    )

    dashboard.log("Loop avviato. Ctrl+C per fermare.", "ok")
    try:
        loop.start()
    except KeyboardInterrupt:
        dashboard.log("Shutdown richiesto dall'utente (Ctrl+C).", "warn")
    finally:
        loop.stop()
        dashboard.log("Cancellazione ordini aperti…", "warn")
        broker.cancel_all_orders()
        session_mgr.mark_paused(session)
        dashboard.log("Sessione sospesa.", "ok")
        summary = read_session_summary(session["session_id"])
        dashboard.print_resoconto(summary)
        dashboard.print_shutdown_message()


if __name__ == "__main__":
    main()

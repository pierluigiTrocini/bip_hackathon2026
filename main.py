"""
BIP Hackathon 2026 — Trading Agent
Entry point.

Usage:
    uv run python main.py

The agent initialises, asks for session resume or new prompt,
then runs autonomously. Press Ctrl+C to stop gracefully.
"""
import logging
import sys

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler("agent.log"),
    ],
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

    # Step 1: Detect previous session
    session_mgr = SessionManager()
    previous = session_mgr.detect_previous_session()

    if previous:
        summary = read_session_summary(previous["session_id"])
        dashboard.print_resoconto(summary)
        choice = session_mgr.ask_resume_or_new()
        if choice == "resume":
            session = session_mgr.resume(previous)
            prompt = session["active_prompt"]
        else:
            prompt = input("\nInserisci il comportamento del nuovo agente: ").strip()
            session = session_mgr.create_new(prompt)
    else:
        prompt = input("Inserisci il comportamento dell'agente (es. 'orientato a scelte green'): ").strip()
        session = session_mgr.create_new(prompt)

    # Step 2: Calibrate adaptive timeout
    adaptive_timeout = AdaptiveTimeout()
    logger.info("Calibrating adaptive timeout...")
    adaptive_timeout.calibrate()
    logger.info(f"Timeout calibration: {adaptive_timeout.summary()}")

    # Step 3: Initialise modules
    tool_executor    = ToolExecutor(adaptive_timeout)
    memory_manager   = MemoryManager()
    imitative_layer  = ImiativeLayer()
    reasoner         = Reasoner()
    broker           = Broker()
    behavior_manager = BehaviorManager(session, adaptive_timeout)

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

    try:
        loop.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    finally:
        loop.stop()
        broker.cancel_all_orders()
        session_mgr.mark_paused(session)
        summary = read_session_summary(session["session_id"])
        dashboard.print_resoconto(summary)
        dashboard.print_shutdown_message()
        logger.info("Agent stopped. Session saved.")


if __name__ == "__main__":
    main()

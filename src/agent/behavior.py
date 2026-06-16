import threading

from src.agent import journal as journal_module
from src.agent.adaptive_timeout import AdaptiveTimeout


class BehaviorManager:
    def __init__(self, session: dict, adaptive_timeout: AdaptiveTimeout) -> None:
        self._session = session
        self._at = adaptive_timeout
        self._active_prompt: str = session.get("active_prompt", "")
        self._initial_prompt: str = session.get("initial_prompt", self._active_prompt)
        self._pending_prompt: str = ""
        self._change_requested: bool = False
        self._change_lock = threading.Lock()

    @property
    def active_prompt(self) -> str:
        return self._active_prompt

    @property
    def initial_prompt(self) -> str:
        return self._initial_prompt

    @property
    def change_requested(self) -> bool:
        return self._change_requested

    def request_change(self, new_prompt: str) -> bool:
        with self._change_lock:
            if self._change_requested:
                return False
            self._pending_prompt = new_prompt
            self._change_requested = True
            return True

    def apply_change(self, memory_manager, imitative_layer) -> bool:
        t_behavior = self._at.t_behavior()
        done = threading.Event()
        exc_holder: list[Exception] = []

        def _do_change():
            try:
                memory_manager.reset_all()
                imitative_layer.reload()
                self._active_prompt = self._pending_prompt
                self._session["active_prompt"] = self._active_prompt
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_do_change, daemon=True)
        t.start()
        completed = done.wait(timeout=t_behavior)

        if completed and not exc_holder:
            self.clear_change_request()
            return True
        else:
            exc_msg = str(exc_holder[0]) if exc_holder else f"timed out after {t_behavior}s"
            journal_module.log_error(
                source="BehaviorManager",
                error=f"apply_change failed: {exc_msg}",
                session_id=self._session.get("session_id", ""),
            )
            self.revert_to_initial()
            self.clear_change_request()
            return False

    def revert_to_initial(self) -> None:
        self._active_prompt = self._initial_prompt
        self._session["active_prompt"] = self._initial_prompt
        journal_module.log_error(
            source="BehaviorManager",
            error="Reverted to initial prompt due to behaviour change failure",
            session_id=self._session.get("session_id", ""),
        )

    def clear_change_request(self) -> None:
        with self._change_lock:
            self._change_requested = False
            self._pending_prompt = ""

    def increment_change_count(self, session: dict) -> None:
        session["behavior_change_count"] = session.get("behavior_change_count", 0) + 1

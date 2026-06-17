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

    def apply_change(self, memory_manager, imitative_layer, preference_engine=None) -> bool:
        try:
            memory_manager.reset_all()
            imitative_layer.reload()
            self._active_prompt = self._pending_prompt
            self._session["active_prompt"] = self._active_prompt
        except Exception as exc:
            journal_module.log_error(
                source="BehaviorManager",
                error=f"apply_change failed: {exc}",
                session_id=self._session.get("session_id", ""),
            )
            self.revert_to_initial()
            self.clear_change_request()
            return False

        self.clear_change_request()

        # Preference extraction is an LLM call — run in background so the main loop never blocks
        if preference_engine is not None:
            t_behavior = self._at.t_behavior()
            prompt_snapshot = self._active_prompt

            def _extract():
                try:
                    preference_engine.extract_from_prompt(prompt_snapshot, t_behavior)
                    preference_engine.compute_derived_parameters()
                except Exception as exc:
                    journal_module.log_error(
                        source="BehaviorManager",
                        error=f"preference extraction failed: {exc}",
                        session_id=self._session.get("session_id", ""),
                    )

            threading.Thread(target=_extract, daemon=True, name="pref-extract").start()

        return True

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

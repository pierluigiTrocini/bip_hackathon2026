import json
import os

from src.agent import config


class ImiativeLayer:
    def __init__(self) -> None:
        self._strategies: list[dict] = []
        self._load()

    def _load(self) -> None:
        path = config.IMITATIVE_DATASET_PATH
        if not os.path.exists(path):
            self._create_default(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._strategies = data.get("strategies", [])
        except Exception:
            self._strategies = []

    def _create_default(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        default = {
            "strategies": [
                {
                    "id": "simons_quant",
                    "name": "Jim Simons — Quantitative Signals",
                    "keywords": ["momentum", "trend", "signal", "data", "pattern"],
                    "anti_keywords": ["narrative", "opinion"],
                    "rules": [
                        "Follow statistical patterns, not narratives",
                        "Volume confirms price action",
                        "Cut losses quickly",
                    ],
                    "risk_profile": "moderate",
                    "sectors": ["any"],
                }
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)

    def filter_for_prompt(self, prompt: str) -> list[dict]:
        prompt_lower = prompt.lower()
        scored: list[tuple[int, dict]] = []
        for s in self._strategies:
            score = sum(1 for kw in s.get("keywords", []) if kw.lower() in prompt_lower)
            score -= sum(2 for akw in s.get("anti_keywords", []) if akw.lower() in prompt_lower)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:2]]

    def build_hints(self, prompt: str, ticker: str) -> str:
        matched = self.filter_for_prompt(prompt)
        if not matched:
            return ""
        lines = ["=== IMITATIVE HINTS ==="]
        for s in matched:
            rules_summary = " ".join(s.get("rules", [])[:2])
            lines.append(f"[{s['name']}] {rules_summary}")
        lines.append("Source: static dataset — not from model memory.")
        lines.append(f"Active strategy: {matched[0]['name']}")
        return "\n".join(lines)

    def get_active_strategy_id(self, prompt: str) -> str | None:
        matched = self.filter_for_prompt(prompt)
        return matched[0]["id"] if matched else None

    def reload(self) -> None:
        self._load()

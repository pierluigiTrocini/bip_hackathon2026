import concurrent.futures
import json

import ollama

from src.agent import config
from src.agent import journal as journal_module

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":          {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence":      {"type": "number"},
        "reasoning":       {"type": "string"},
        "accuracy_review": {"type": "string"},
    },
    "required": ["action", "confidence", "reasoning", "accuracy_review"],
}

_SYSTEM_PROMPT = (
    "You are a cautious quantitative trading analyst operating on real market data. "
    "You NEVER invent, estimate, or recall prices from memory — you only use the data provided in this prompt. "
    "You NEVER fabricate news or sentiment scores. "
    "When data is stale, uncertain, or confidence is low, you hold — not buy or sell. "
    "Your reasoning must cite specific numbers from the data above. "
    "If the active strategy (imitative hints) conflicts with the data, state the conflict explicitly. "
    "Output only valid JSON matching the schema. Max 2 sentences for reasoning, 1 for accuracy_review."
)


class Reasoner:
    def __init__(self) -> None:
        self._session_id: str = ""

    def _safe_hold(self, reason: str) -> dict:
        return {
            "action": "hold",
            "confidence": 0.0,
            "confidence_raw": 0.0,
            "stale_penalty": 0.0,
            "reasoning": f"Hold forced: {reason}",
            "accuracy_review": "N/A",
        }

    def decide(
        self,
        ticker: str,
        memory_context: str,
        price: float,
        price_timestamp: str,
        ma5: float,
        trend: str,
        sentiment_score: float,
        sentiment_label: str,
        imitative_hints: str,
        active_prompt: str,
        cash: float,
        positions: dict,
        mode: str,
        stale: bool,
        staleness_seconds: int,
        t_behavior: int,
    ) -> dict:
        positions_str = ", ".join(
            f"{sym}: {p['qty']} shares @ ${p['avg_entry_price']:.2f}"
            for sym, p in positions.items()
        ) or "none"

        user_prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"=== AGENT BEHAVIOUR ===\n{active_prompt}\n\n"
            f"{imitative_hints}\n\n"
            f"=== MARKET DATA ({ticker}) ===\n"
            f"Price: ${price:.2f} (as of {price_timestamp})\n"
            f"MA5: ${ma5:.2f} | Trend: {trend}\n"
            f"Sentiment: {sentiment_score:+.2f} ({sentiment_label})\n"
            f"Data stale: {stale} (staleness: {staleness_seconds}s)\n\n"
            f"=== PORTFOLIO ===\n"
            f"Cash: ${cash:,.2f} | Mode: {mode}\n"
            f"Positions: {positions_str}\n\n"
            f"=== MEMORY ===\n{memory_context}\n\n"
            f"Decide: buy, sell, or hold {ticker}. Return valid JSON only."
        )

        def _call() -> dict:
            resp = ollama.generate(
                model=config.OLLAMA_REASONING_MODEL,
                prompt=user_prompt,
                format=_DECISION_SCHEMA,
                options={"temperature": 0.2, "num_predict": 300},
                keep_alive="30s",
            )
            raw = resp.get("response", "{}")
            parsed = json.loads(raw)
            action = parsed.get("action", "hold")
            if action not in ("buy", "sell", "hold"):
                action = "hold"
            confidence_raw = float(parsed.get("confidence", 0.0))
            confidence_raw = max(0.0, min(1.0, confidence_raw))
            return {
                "action": action,
                "confidence_raw": confidence_raw,
                "reasoning": str(parsed.get("reasoning", ""))[:400],
                "accuracy_review": str(parsed.get("accuracy_review", ""))[:200],
            }

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call)
                result = fut.result(timeout=t_behavior)
        except concurrent.futures.TimeoutError:
            journal_module.log_error(
                source="Reasoner", error=f"Timeout after {t_behavior}s for {ticker}",
                ticker=ticker, session_id=self._session_id,
            )
            return self._safe_hold(f"LLM timeout after {t_behavior}s")
        except Exception as exc:
            journal_module.log_error(
                source="Reasoner", error=str(exc),
                ticker=ticker, session_id=self._session_id,
            )
            return self._safe_hold(str(exc))

        # Apply stale penalty in code — model cannot bypass this
        penalty = min(staleness_seconds / 60 * 0.05, 0.40)
        confidence = max(0.0, result["confidence_raw"] - penalty)

        return {
            "action": result["action"],
            "confidence": round(confidence, 4),
            "confidence_raw": round(result["confidence_raw"], 4),
            "stale_penalty": round(penalty, 4),
            "reasoning": result["reasoning"],
            "accuracy_review": result["accuracy_review"],
        }

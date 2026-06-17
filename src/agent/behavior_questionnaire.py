"""
LLM-generated questionnaire for agent behavior refinement.
Uses qwen2.5:3b (fast) for question generation and answer synthesis.
"""
import json

import ollama

from src.agent import config

_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["questions"],
}

_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "new_prompt": {"type": "string"},
    },
    "required": ["new_prompt"],
}

_FALLBACK_QUESTIONS = [
    "How much risk are you willing to accept? (low / medium / high)",
    "Do you want to focus on specific sectors or diversify?",
    "How do you want to react to a loss greater than 5%?",
]


def generate_questions(context: dict, t_behavior: int) -> list[str]:
    """
    Generate 3–4 targeted questions in English to refine the agent's strategy.

    context keys: active_prompt, tickers, pnl_pct, mode, recent_actions
    """
    ctx_str = (
        f"Current prompt: {context.get('active_prompt', 'not available')[:120]}\n"
        f"Monitored tickers: {', '.join(context.get('tickers', []))}\n"
        f"Current P&L: {context.get('pnl_pct', 0.0):+.2%}\n"
        f"Mode: {context.get('mode', 'normal')}\n"
        f"Recent decisions: {context.get('recent_actions', 'none')}"
    )
    try:
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=(
                "You are a trading assistant. "
                "Generate exactly 3 short questions (max 15 words each) in English "
                "to help the user refine the agent's strategy. "
                "Questions must be specific to the context below, not generic. "
                "Cover aspects such as: risk tolerance, sector preferences, "
                "reaction to losses, time horizon.\n\n"
                f"Context:\n{ctx_str}\n\n"
                "Reply ONLY with valid JSON according to the schema."
            ),
            format=_QUESTIONS_SCHEMA,
            options={"temperature": 0.7, "num_predict": 200},
            keep_alive="30s",
        )
        raw = resp.response
        questions = json.loads(raw).get("questions", [])
        result = [str(q).strip() for q in questions if q][:4]
        if result:
            return result
    except Exception:
        pass
    return list(_FALLBACK_QUESTIONS)


def synthesize_prompt(
    active_prompt: str,
    questions: list[str],
    answers: list[str],
    t_behavior: int,
) -> str:
    """
    Synthesize Q&A pairs into a new behavior prompt, incorporating the current one.
    """
    qa_str = "\n".join(
        f"D: {q}\nR: {a}" for q, a in zip(questions, answers) if a.strip()
    )
    try:
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=(
                "You are a trading assistant. "
                "Based on the current prompt and the user's answers, "
                "write a new behaviour prompt for the agent. "
                "It must be in English, 2–3 sentences, and incorporate the stated preferences. "
                "Do not lose the original intent if the answers do not contradict it.\n\n"
                f"Current prompt: {active_prompt}\n\n"
                f"Questions and answers:\n{qa_str}\n\n"
                "Reply ONLY with valid JSON."
            ),
            format=_SYNTHESIS_SCHEMA,
            options={"temperature": 0.3, "num_predict": 180},
            keep_alive="30s",
        )
        raw = resp.response
        new_prompt = json.loads(raw).get("new_prompt", "").strip()
        if new_prompt:
            return new_prompt
    except Exception:
        pass
    # Fallback: append answers to current prompt
    return f"{active_prompt}. User update: {'; '.join(a for a in answers if a.strip())}."

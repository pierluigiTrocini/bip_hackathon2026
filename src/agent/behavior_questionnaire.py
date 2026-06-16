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
    "Quanto rischio sei disposto ad accettare? (basso / medio / alto)",
    "Vuoi privilegiare settori specifici o diversificare?",
    "Come vuoi reagire a una perdita superiore al 5%?",
]


def generate_questions(context: dict, t_behavior: int) -> list[str]:
    """
    Generate 3–4 targeted questions in Italian to refine the agent's strategy.

    context keys: active_prompt, tickers, pnl_pct, mode, recent_actions
    """
    ctx_str = (
        f"Prompt attuale: {context.get('active_prompt', 'non disponibile')[:120]}\n"
        f"Ticker monitorati: {', '.join(context.get('tickers', []))}\n"
        f"P&L attuale: {context.get('pnl_pct', 0.0):+.2%}\n"
        f"Modalità: {context.get('mode', 'normal')}\n"
        f"Ultime decisioni: {context.get('recent_actions', 'nessuna')}"
    )
    try:
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=(
                "Sei un assistente di trading. "
                "Genera esattamente 3 domande brevi (max 15 parole ciascuna) in italiano "
                "per aiutare l'utente a raffinare la strategia dell'agente. "
                "Le domande devono essere specifiche al contesto qui sotto, non generiche. "
                "Suggerisci aspetti come: tolleranza al rischio, preferenze settoriali, "
                "reazione alle perdite, orizzonte temporale.\n\n"
                f"Contesto:\n{ctx_str}\n\n"
                "Rispondi SOLO con JSON valido secondo lo schema."
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
                "Sei un assistente di trading. "
                "Basandoti sul prompt attuale e sulle risposte dell'utente, "
                "scrivi un nuovo prompt di comportamento per l'agente. "
                "Deve essere in italiano, 2–3 frasi, e incorporare le preferenze espresse. "
                "Non perdere l'intenzione originale se le risposte non la contraddicono.\n\n"
                f"Prompt attuale: {active_prompt}\n\n"
                f"Domande e risposte:\n{qa_str}\n\n"
                "Rispondi SOLO con JSON valido."
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
    return f"{active_prompt}. Aggiornamento utente: {'; '.join(a for a in answers if a.strip())}."

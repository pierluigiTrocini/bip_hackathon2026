import json

import ollama

from src.agent import config

_SCHEMA = {
    "type": "object",
    "properties": {
        "score":     {"type": "number"},
        "label":     {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "rationale": {"type": "string"},
    },
    "required": ["score", "label", "rationale"],
}

_NEUTRAL = {
    "score": 0.0,
    "label": "neutral",
    "rationale": "unavailable",
    "article_count": 0,
    "prompt_filtered": False,
}


def _filter_articles(articles: list[dict], active_prompt: str) -> tuple[list[dict], bool]:
    if not active_prompt:
        return articles, False
    prompt_lower = active_prompt.lower()
    anti_pairs = [
        (["green", "esg", "renewable", "climate"], ["defense", "arms", "military"]),
        (["defense", "arms", "military"], ["green", "esg", "renewable", "climate"]),
    ]
    for prompt_words, article_words in anti_pairs:
        if any(w in prompt_lower for w in prompt_words):
            filtered = [
                a for a in articles
                if not any(w in (a.get("title", "") + a.get("summary", "")).lower() for w in article_words)
            ]
            return filtered, len(filtered) < len(articles)
    return articles, False


def analyse(
    ticker: str,
    articles: list[dict],
    active_prompt: str = "",
    t_behavior: int = 60,
) -> dict:
    try:
        filtered, was_filtered = _filter_articles(articles, active_prompt)
        if not filtered:
            return {**_NEUTRAL, "article_count": 0, "prompt_filtered": was_filtered}

        articles_text = "\n".join(
            f"- {a.get('title', '')} {a.get('summary', '')}" for a in filtered[:3]
        )
        prompt = (
            f"Analyse the sentiment of these news articles about {ticker}.\n"
            f"Return a JSON with score (-1.0 to +1.0), label (positive/negative/neutral), "
            f"and a one-sentence rationale.\n\nArticles:\n{articles_text}"
        )

        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=prompt,
            format=_SCHEMA,
            options={"temperature": 0.0, "num_predict": 200},
            keep_alive="30s",
        )
        raw = resp.get("response", "{}")
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        label = parsed.get("label", "neutral")
        if label not in ("positive", "negative", "neutral"):
            label = "neutral"
        return {
            "score": score,
            "label": label,
            "rationale": str(parsed.get("rationale", ""))[:200],
            "article_count": len(filtered),
            "prompt_filtered": was_filtered,
        }
    except Exception:
        return {**_NEUTRAL, "article_count": len(articles), "prompt_filtered": False}

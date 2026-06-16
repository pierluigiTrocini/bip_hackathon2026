import concurrent.futures
import json

import ollama

from src.agent import config
from src.agent import news_log

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
) -> tuple[dict, list[dict]]:
    """
    Returns (sentiment_result, keywords_and_relevance).

    sentiment_result: {score, label, rationale, article_count, prompt_filtered}
    keywords_and_relevance: [{keywords: list[str], relevance_score: float}] — one per article

    Both LLM calls (sentiment + keyword extraction) run concurrently.
    """
    fallback_kw = [{"keywords": [], "relevance_score": 0.5} for _ in articles]

    try:
        filtered, was_filtered = _filter_articles(articles, active_prompt)
        if not filtered:
            return (
                {**_NEUTRAL, "article_count": 0, "prompt_filtered": was_filtered},
                fallback_kw,
            )

        articles_text = "\n".join(
            f"- {a.get('title', '')} {a.get('summary', '')}" for a in filtered[:3]
        )
        sentiment_prompt = (
            f"Analyse the sentiment of these news articles about {ticker}.\n"
            f"Return a JSON with score (-1.0 to +1.0), label (positive/negative/neutral), "
            f"and a one-sentence rationale.\n\nArticles:\n{articles_text}"
        )

        def _sentiment_call() -> dict:
            resp = ollama.generate(
                model=config.OLLAMA_SENTIMENT_MODEL,
                prompt=sentiment_prompt,
                format=_SCHEMA,
                options={"temperature": 0.0, "num_predict": 200},
                keep_alive="30s",
            )
            raw = resp.get("response", "{}") if isinstance(resp, dict) else getattr(resp, "response", "{}")
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

        def _keyword_call() -> list[dict]:
            return news_log.extract_keywords_and_relevance(filtered, ticker, t_behavior)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_sentiment = pool.submit(_sentiment_call)
            fut_keywords  = pool.submit(_keyword_call)

            try:
                sentiment_result = fut_sentiment.result(timeout=max(t_behavior, 60))
            except Exception:
                sentiment_result = {**_NEUTRAL, "article_count": len(filtered), "prompt_filtered": was_filtered}

            try:
                kw_result = fut_keywords.result(timeout=max(t_behavior, 60))
            except Exception:
                kw_result = fallback_kw

        # kw_result is indexed over filtered articles; pad to match original articles list
        padded_kw: list[dict] = []
        fi = 0  # index into filtered
        for a in articles:
            if a in filtered:
                padded_kw.append(kw_result[fi] if fi < len(kw_result) else {"keywords": [], "relevance_score": 0.5})
                fi += 1
            else:
                padded_kw.append({"keywords": [], "relevance_score": 0.5})

        return sentiment_result, padded_kw

    except Exception:
        return (
            {**_NEUTRAL, "article_count": len(articles), "prompt_filtered": False},
            fallback_kw,
        )

import concurrent.futures
import json

from src.agent import config
from src.agent import llm_stream
from src.agent import journal as journal_module
from src.agent import strategy_library


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action":          {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence":      {"type": "number"},
        "reasoning":       {"type": "string"},
        "accuracy_review": {"type": "string"},
        "caption":         {"type": "string"},
    },
    "required": ["action", "confidence", "reasoning", "accuracy_review", "caption"],
}

# Per-strategy signal mapping: (sentiment_score, trend) → action recommendation
# These are DIRECTIVES, not descriptions — they tell the model what to DO.
_STRATEGY_SIGNALS: dict[str, dict] = {
    "contrarian": {
        ("fear_extreme", "down"):  "STRONG BUY — extreme fear + downtrend: classic contrarian capitulation entry",
        ("fear_extreme", "other"): "BUY — extreme fear: crowd is maximally wrong, contrarian opportunity",
        ("fear", "any"):           "BUY — market fear/selling: contrarian edge, expect mean reversion",
        ("greed_extreme", "up"):   "STRONG SELL — extreme greed + uptrend: classic contrarian distribution exit",
        ("greed_extreme", "other"):"SELL — extreme greed: crowd is maximally optimistic, contrarian exit",
        ("greed", "any"):          "SELL — market greed/buying: contrarian distribution phase",
        ("neutral", "any"):        "HOLD — neutral sentiment, no contrarian edge",
    },
    "trend_following": {
        ("fear_extreme", "down"):  "SELL — downtrend confirmed + extreme fear: trend broken, exit position",
        ("fear", "down"):          "SELL — downtrend with negative sentiment: trend is your enemy here",
        ("fear", "any"):           "HOLD — sentiment negative but trend unclear, wait for direction",
        ("greed_extreme", "up"):   "BUY — uptrend confirmed + strong sentiment: trend intact, ride it",
        ("greed", "up"):           "BUY — uptrend with positive sentiment: trend is your friend",
        ("greed", "any"):          "HOLD — positive sentiment but trend unclear, wait for breakout",
        ("neutral", "up"):         "BUY — uptrend intact: trend is your friend even without sentiment",
        ("neutral", "down"):       "SELL — downtrend active: exit regardless of neutral sentiment",
        ("neutral", "any"):        "HOLD — no clear trend, preserve capital",
    },
    "momentum": {
        ("fear_extreme", "down"):  "SELL IMMEDIATELY — extreme fear + downtrend: momentum has reversed violently, exit",
        ("fear", "down"):          "SELL — momentum lost: negative sentiment + downtrend, get out",
        ("fear", "any"):           "HOLD — momentum unclear, no entry",
        ("greed_extreme", "up"):   "BUY — peak momentum: strong sentiment + uptrend, ride the acceleration",
        ("greed", "up"):           "BUY — momentum building: buy before the crowd accelerates further",
        ("greed", "any"):          "HOLD — positive but no clear uptrend, momentum unconfirmed",
        ("neutral", "any"):        "HOLD — no momentum signal, stay flat",
    },
    "value": {
        ("fear_extreme", "down"):  "BUY — price likely well below MA5 + extreme fear: irrational selling, value entry",
        ("fear", "any"):           "BUY — market fear punishing price below fair value: mean reversion opportunity",
        ("greed_extreme", "up"):   "SELL — price likely above MA5 + extreme greed: value fully priced, take profit",
        ("greed", "any"):          "SELL — positive sentiment suggests price at or above fair value: distribute",
        ("neutral", "any"):        "HOLD — price near fair value (MA5), no value edge",
    },
    "defensive": {
        ("fear_extreme", "down"):  "SELL — extreme fear + downtrend: if holding, cut losses now",
        ("fear", "down"):          "SELL — negative sentiment + downtrend: capital preservation, exit",
        ("fear", "any"):           "HOLD — uncertainty, stay flat unless position in loss",
        ("greed_extreme", "up"):   "HOLD or SELL — if holding with profit: take it now before reversal",
        ("greed", "any"):          "HOLD — slight positive, not enough confidence for new entry in defensive mode",
        ("neutral", "any"):        "HOLD — defensive mode default: cash is a position",
    },
    "scalping": {
        ("fear_extreme", "down"):  "SELL — extreme fear divergence from prior trend: quick scalp exit",
        ("fear", "any"):           "BUY — short-term fear dip: scalp entry, tight exit target",
        ("greed_extreme", "up"):   "SELL — extreme greed peak: scalp exit, take the small profit now",
        ("greed", "any"):          "SELL if holding — sentiment peaked: scalp profit before reversal",
        ("neutral", "any"):        "HOLD — no scalping signal, flat market",
    },
}


def _get_strategy_signal(strategy_id: str, sentiment_score: float, trend: str) -> str:
    """Map sentiment + trend to a strategy-specific action directive."""
    signals = _STRATEGY_SIGNALS.get(strategy_id, _STRATEGY_SIGNALS["contrarian"])

    if sentiment_score <= -0.4:
        sentiment_bucket = "fear_extreme"
    elif sentiment_score <= -0.15:
        sentiment_bucket = "fear"
    elif sentiment_score >= 0.4:
        sentiment_bucket = "greed_extreme"
    elif sentiment_score >= 0.15:
        sentiment_bucket = "greed"
    else:
        sentiment_bucket = "neutral"

    trend_key = trend if trend in ("up", "down") else "any"

    # Try exact match, then fallback to "any" for trend, then to neutral
    for t in (trend_key, "other", "any"):
        key = (sentiment_bucket, t)
        if key in signals:
            return signals[key]

    return signals.get(("neutral", "any"), "HOLD — no clear signal")


class Reasoner:
    def __init__(self) -> None:
        self._session_id: str = ""

    def _safe_hold(self, reason: str) -> dict:
        return {
            "action":          "hold",
            "confidence":      0.0,
            "confidence_raw":  0.0,
            "stale_penalty":   0.0,
            "reasoning":       f"Hold forced: {reason}",
            "accuracy_review": "N/A",
            "caption":         f"Decisione forzata: {reason[:120]}",
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
        strategy_id: str = "contrarian",
        take_profit_hint: str = "",
        correlation_section: str = "",
        historical_news_context: str = "",
    ) -> dict:
        positions_str = ", ".join(
            f"{sym}: {p['qty']} shares @ ${p['avg_entry_price']:.2f}"
            for sym, p in positions.items()
        ) or "none"

        strategy = strategy_library.get(strategy_id)
        action_signal = _get_strategy_signal(strategy_id, sentiment_score, trend)

        take_profit_section = (
            f"=== POSITION P&L ===\n{take_profit_hint}\n\n"
            if take_profit_hint else ""
        )

        correlation_block = f"\n{correlation_section}\n\n" if correlation_section else ""
        historical_block = (
            f"=== STORICO NEWS ({ticker} — ultimi cicli) ===\n"
            f"{historical_news_context}\n\n"
            if historical_news_context else ""
        )

        user_prompt = (
            f"=== STRATEGY: {strategy['name'].upper()} ===\n"
            f"{strategy['system_prompt']}\n\n"
            f"=== ACTION SIGNAL ===\n"
            f"{action_signal}\n"
            f"Sentiment: {sentiment_score:+.2f} ({sentiment_label}) | Trend: {trend}\n\n"
            f"{take_profit_section}"
            f"=== AGENT BEHAVIOUR ===\n{active_prompt}\n\n"
            f"{imitative_hints}\n\n"
            f"=== MARKET DATA ({ticker}) ===\n"
            f"Price: ${price:.2f} (as of {price_timestamp})\n"
            f"MA5: ${ma5:.2f} | Trend: {trend}\n"
            f"Data stale: {stale} (staleness: {staleness_seconds}s)\n\n"
            f"{historical_block}"
            f"=== PORTFOLIO ===\n"
            f"Cash: ${cash:,.2f} | Mode: {mode}\n"
            f"Positions: {positions_str}\n\n"
            f"=== MEMORY ===\n{memory_context}\n"
            f"{correlation_block}"
            f"YOUR ACTION SIGNAL IS: {action_signal}\n"
            f"Follow the ACTION SIGNAL above. If you deviate, explain why in reasoning.\n"
            f"The 'caption' field must be a single sentence in Italian (max 160 characters) "
            f"explaining the decision to a non-expert user. Reference at least one concrete data point "
            f"(price vs MA, sentiment score, or a specific news topic). "
            f"Do NOT start with the action word (e.g. do not start with 'Ho deciso di comprare').\n"
            f"Return valid JSON only."
        )

        def _call() -> dict:
            raw = llm_stream.generate(
                model=config.OLLAMA_REASONING_MODEL,
                prompt=user_prompt,
                format=_DECISION_SCHEMA,
                options={"temperature": 0.2, "num_predict": 300},
                keep_alive="30s",
                show_output=llm_stream.LOOP_VERBOSE,
            )
            # Strip any trailing content after the JSON object
            raw = raw.strip()
            if raw and raw[0] == "{":
                # Find the end of the first JSON object
                depth = 0
                end = 0
                for i, ch in enumerate(raw):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                raw = raw[:end] if end > 0 else raw

            parsed = json.loads(raw)
            action = parsed.get("action", "hold")
            if action not in ("buy", "sell", "hold"):
                action = "hold"
            confidence_raw = float(parsed.get("confidence", 0.0))
            confidence_raw = max(0.0, min(1.0, confidence_raw))
            return {
                "action":          action,
                "confidence_raw":  confidence_raw,
                "reasoning":       str(parsed.get("reasoning", ""))[:400],
                "accuracy_review": str(parsed.get("accuracy_review", ""))[:200],
                "caption":         str(parsed.get("caption", ""))[:160],
            }

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call)
                result = fut.result(timeout=max(t_behavior, 120))
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

        # Stale penalty applied in code — model cannot bypass this
        penalty = min(staleness_seconds / 60 * 0.05, 0.40)
        confidence = max(0.0, result["confidence_raw"] - penalty)

        return {
            "action":          result["action"],
            "confidence":      round(confidence, 4),
            "confidence_raw":  round(result["confidence_raw"], 4),
            "stale_penalty":   round(penalty, 4),
            "reasoning":       result["reasoning"],
            "accuracy_review": result["accuracy_review"],
            "caption":         result.get("caption", ""),
        }

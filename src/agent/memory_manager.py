from collections import deque
from src.agent import config


class MemoryManager:
    def __init__(self) -> None:
        self._hot: dict[str, deque] = {}
        self._warm: dict[str, str] = {}
        self._warm_age: dict[str, int] = {}

    def _ensure_ticker(self, ticker: str) -> None:
        if ticker not in self._hot:
            self._hot[ticker] = deque(maxlen=config.HOT_WINDOW_SIZE)
            self._warm_age[ticker] = 0

    def update(self, entry: dict) -> None:
        ticker = entry.get("ticker", "UNKNOWN")
        self._ensure_ticker(ticker)
        overflowed = len(self._hot[ticker]) >= config.HOT_WINDOW_SIZE
        self._hot[ticker].append(entry)
        if overflowed:
            self._warm_age[ticker] = self._warm_age.get(ticker, 0) + 1
            if self._warm_age[ticker] >= config.WARM_COMPACTION_TRIGGER:
                self._compact(ticker)

    def _compact(self, ticker: str) -> None:
        entries = list(self._hot[ticker])
        try:
            summary = self._llm_summary(ticker, entries)
        except Exception:
            summary = self._rule_based_summary(ticker, entries)
        self._warm[ticker] = summary
        self._warm_age[ticker] = 0

    def _llm_summary(self, ticker: str, entries: list[dict]) -> str:
        import ollama
        text = "\n".join(
            f"[{e.get('ts', '')}] {e.get('action','?').upper()} @ ${e.get('price',0):.2f} "
            f"conf:{e.get('conf',0):.2f} outcome:{e.get('outcome_pct','?')}%"
            for e in entries
        )
        prompt = (
            f"Summarise these past trading decisions for {ticker} in 2-3 sentences. "
            f"Focus on accuracy trends and patterns:\n{text}"
        )
        resp = ollama.generate(
            model=config.OLLAMA_SENTIMENT_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 150},
            keep_alive="30s",
        )
        return resp.get("response", "").strip()

    def _rule_based_summary(self, ticker: str, entries: list[dict]) -> str:
        total = len(entries)
        if not total:
            return f"No historical data for {ticker}."
        buys = sum(1 for e in entries if e.get("action") == "buy")
        sells = sum(1 for e in entries if e.get("action") == "sell")
        holds = total - buys - sells
        outcomes = [e["outcome_pct"] for e in entries if e.get("outcome_pct") is not None]
        avg_outcome = sum(outcomes) / len(outcomes) if outcomes else 0.0
        return (
            f"{ticker}: {total} decisions — {buys} buy, {sells} sell, {holds} hold. "
            f"Avg outcome: {avg_outcome:+.2f}%."
        )

    def build_context(self, ticker: str) -> str:
        self._ensure_ticker(ticker)
        hot = list(self._hot.get(ticker, []))
        warm = self._warm.get(ticker, "")
        if not hot and not warm:
            return "No prior decisions on this ticker."
        parts: list[str] = []
        if warm:
            parts.append("=== HISTORICAL SUMMARY ===")
            parts.append(warm)
            parts.append("")
        if hot:
            parts.append(f"=== RECENT DECISIONS (last {len(hot)}) ===")
            for e in hot:
                outcome = f"{e.get('outcome_pct'):+.2f}%" if e.get("outcome_pct") is not None else "pending"
                parts.append(
                    f"[{e.get('ts','')}] {e.get('action','?').upper()} @ ${e.get('price',0):.2f} "
                    f"| conf:{e.get('conf',0):.2f} | sentiment:{e.get('sentiment',0):+.2f} "
                    f"| outcome:{outcome}"
                )
        return "\n".join(parts)

    def reset_ticker(self, ticker: str) -> None:
        self._hot.pop(ticker, None)
        self._warm.pop(ticker, None)
        self._warm_age.pop(ticker, None)

    def reset_all(self) -> None:
        self._hot.clear()
        self._warm.clear()
        self._warm_age.clear()

    def get_stats(self) -> dict:
        stats = {}
        for ticker in set(self._hot) | set(self._warm):
            stats[ticker] = {
                "hot_size": len(self._hot.get(ticker, [])),
                "warm_age": self._warm_age.get(ticker, 0),
                "total": len(self._hot.get(ticker, [])),
            }
        return stats

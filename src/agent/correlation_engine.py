"""
Correlation Engine — News Co-occurrence Correlation Index (NCCI).

No LLM calls. All computation is pure Python arithmetic on keyword strings
read from news_log.jsonl. Thread-safe via internal lock.
"""
import threading
from datetime import datetime, timezone

from src.agent import config
from src.agent import news_log


def _decay(ts: str, now: datetime, half_life_days: float = 7.0) -> float:
    """
    Exponential decay. 1.0 at ts=now, 0.5 at 7 days ago, 0.0 after 30 days.
    """
    try:
        age_days = (now - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 86400
    except Exception:
        return 0.0
    if age_days > 30:
        return 0.0
    return 2 ** (-age_days / half_life_days)


class CorrelationEngine:

    def __init__(self) -> None:
        self._matrix: dict[tuple[str, str], float] = {}
        self._shared_kw: dict[tuple[str, str], list[str]] = {}
        self._last_built: datetime | None = None
        self._lock = threading.Lock()
        self._universe: set[str] = set()

    def rebuild(self, max_age_days: int = 30) -> None:
        """
        Recompute the full NCCI matrix from news_log keyword data.
        Called at startup and every NCCI_REBUILD_EVERY cycles (daemon thread).
        """
        try:
            half_life = config.NCCI_HALF_LIFE_DAYS
            min_weight = config.NCCI_KEYWORD_MIN_WEIGHT
            now = datetime.now(timezone.utc)

            raw = news_log.get_all_keywords_by_ticker(max_age_days=max_age_days)

            # Compute weighted keyword sets per ticker
            # weighted_kw[ticker][keyword] = sum(relevance * decay)
            weighted: dict[str, dict[str, float]] = {}
            for ticker, entries in raw.items():
                kw_weights: dict[str, float] = {}
                for entry in entries:
                    d = _decay(entry.get("ts", ""), now, half_life)
                    rel = float(entry.get("relevance_score", 0.5))
                    for kw in entry.get("keywords", []):
                        kw = kw.lower()
                        kw_weights[kw] = kw_weights.get(kw, 0.0) + rel * d
                weighted[ticker] = kw_weights

            # effective_keywords: keywords with weighted score >= min_weight
            effective: dict[str, set[str]] = {
                ticker: {kw for kw, w in kw_weights.items() if w >= min_weight}
                for ticker, kw_weights in weighted.items()
            }

            # Compute NCCI for all pairs
            tickers = list(effective.keys())
            new_matrix: dict[tuple[str, str], float] = {}
            new_shared: dict[tuple[str, str], list[str]] = {}

            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    a, b = tickers[i], tickers[j]
                    set_a = effective.get(a, set())
                    set_b = effective.get(b, set())
                    union = set_a | set_b
                    if not union:
                        ncci = 0.0
                        shared: list[str] = []
                    else:
                        intersection = set_a & set_b
                        ncci = len(intersection) / len(union)
                        shared = sorted(intersection)
                    key = (a, b)
                    new_matrix[key] = round(ncci, 4)
                    new_shared[key] = shared

            with self._lock:
                self._matrix = new_matrix
                self._shared_kw = new_shared
                self._last_built = now
                # Merge discovered tickers into universe
                for t in tickers:
                    self._universe.add(t)

        except Exception:
            pass

    def get_ncci(self, ticker_a: str, ticker_b: str) -> float:
        """
        Return NCCI(A, B). Symmetric: get_ncci(A, B) == get_ncci(B, A).
        """
        with self._lock:
            key = (ticker_a, ticker_b) if (ticker_a, ticker_b) in self._matrix else (ticker_b, ticker_a)
            return self._matrix.get(key, 0.0)

    def _get_shared(self, ticker_a: str, ticker_b: str) -> list[str]:
        with self._lock:
            key = (ticker_a, ticker_b) if (ticker_a, ticker_b) in self._shared_kw else (ticker_b, ticker_a)
            return self._shared_kw.get(key, [])

    def get_correlated_tickers(
        self,
        ticker: str,
        threshold: float = 0.20,
        exclude: list[str] | None = None,
    ) -> list[dict]:
        """
        Return tickers with NCCI >= threshold, sorted by NCCI descending.
        """
        exclude_set = set(exclude or [])
        results = []

        with self._lock:
            all_keys = list(self._matrix.keys())
            matrix_copy = dict(self._matrix)
            shared_copy = dict(self._shared_kw)

        for a, b in all_keys:
            other = None
            if a == ticker:
                other = b
            elif b == ticker:
                other = a
            if other is None or other in exclude_set:
                continue
            key = (a, b)
            ncci = matrix_copy.get(key, 0.0)
            if ncci >= threshold:
                shared = shared_copy.get(key, [])
                results.append({
                    "ticker":          other,
                    "ncci":            ncci,
                    "shared_keywords": shared,
                    "in_portfolio":    False,  # caller sets this if needed
                })

        results.sort(key=lambda x: x["ncci"], reverse=True)
        return results

    def build_prompt_section(
        self,
        tickers: list[str],
        positions: dict,
        threshold: float = 0.20,
    ) -> str:
        """
        Build the correlation matrix section for Gemma4's prompt.
        Only includes pairs where both tickers are in `tickers` and NCCI >= threshold.
        Returns "" if no pairs exceed threshold.
        """
        try:
            ticker_set = set(tickers)
            pairs: list[tuple[str, str, float, list[str]]] = []

            with self._lock:
                for (a, b), ncci in self._matrix.items():
                    if a not in ticker_set or b not in ticker_set:
                        continue
                    if ncci < threshold:
                        continue
                    shared = self._shared_kw.get((a, b), [])
                    pairs.append((a, b, ncci, shared))

            if not pairs:
                return ""

            pairs.sort(key=lambda x: x[2], reverse=True)
            pairs = pairs[:8]

            lines = ["=== CORRELATION MATRIX (NCCI) ==="]
            for a, b, ncci, shared in pairs:
                if ncci >= 0.60:
                    label = "alta"
                elif ncci >= 0.30:
                    label = "media"
                else:
                    label = "bassa"
                kw_str = ", ".join(shared[:4]) if shared else "—"
                lines.append(f"{a} ↔ {b}: {ncci:.2f}  [{kw_str}]  → {label}")

            return "\n".join(lines)

        except Exception:
            return ""

    def register_dynamic_ticker(self, ticker: str) -> None:
        """Add ticker to the universe for tracking. Picked up on next rebuild()."""
        with self._lock:
            self._universe.add(ticker)

    def get_universe(self) -> list[str]:
        """Return all tickers currently tracked."""
        with self._lock:
            return list(self._universe)

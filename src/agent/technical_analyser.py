"""
Technical Indicators — RSI(14) + Bollinger Bands(20).
Standard library only: no numpy, pandas, ta-lib.
R3: zero external dependencies.
R4: RSI uses Wilder's smoothing.
R5: Bollinger uses population std (no Bessel correction).
"""
import math
from dataclasses import dataclass


@dataclass
class RSIResult:
    value:  float
    signal: str
    valid:  bool


@dataclass
class BollingerResult:
    upper:     float
    middle:    float
    lower:     float
    bandwidth: float   # (upper - lower) / middle * 100
    pct_b:     float   # (price - lower) / (upper - lower), clamped [-0.1, 1.1]
    signal:    str
    valid:     bool


@dataclass
class TechnicalSignals:
    rsi:            RSIResult
    bollinger:      BollingerResult
    prompt_section: str   # "" if both invalid


# ── Signal strings (English, R14) ─────────────────────────────────────────────

def _rsi_signal(value: float, overbought: float, oversold: float) -> str:
    if value >= overbought:
        return f"RSI: overbought [{value:.1f}] — possible bearish reversal"
    elif value <= oversold:
        return f"RSI: oversold [{value:.1f}] — possible bullish reversal"
    elif value >= 60:
        return f"RSI: bullish momentum [{value:.1f}]"
    elif value <= 40:
        return f"RSI: bearish momentum [{value:.1f}]"
    else:
        return f"RSI: neutral [{value:.1f}] — no extreme signal"


def _bollinger_signal(pct_b: float, bandwidth: float, squeeze_pct: float) -> str:
    if bandwidth < squeeze_pct:
        return f"BB: squeeze [{bandwidth:.2f}% width] — low volatility, possible breakout incoming"
    if pct_b >= 1.0:
        return f"BB: price above upper band [%B={pct_b:.2f}] — maximum extension, reversal risk"
    elif pct_b <= 0.0:
        return f"BB: price below lower band [%B={pct_b:.2f}] — maximum extension, possible bounce"
    elif pct_b >= 0.80:
        return f"BB: price near upper band [%B={pct_b:.2f}] — resistance zone"
    elif pct_b <= 0.20:
        return f"BB: price near lower band [%B={pct_b:.2f}] — support zone"
    else:
        return f"BB: price in middle band [%B={pct_b:.2f}] — no extreme pressure"


# ── Core computations ──────────────────────────────────────────────────────────

def compute_rsi(
    closes: list[float],
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> RSIResult:
    """RSI with Wilder's smoothing. Never raises."""
    try:
        if len(closes) < period + 1:
            return RSIResult(value=50.0, signal="RSI: insufficient data", valid=False)

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [max(0.0, d) for d in deltas]
        losses = [abs(min(0.0, d)) for d in deltas]

        # Initial averages: simple mean of first `period` values
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder's smoothing for remaining
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - 100.0 / (1.0 + rs)

        rsi_val = max(0.0, min(100.0, rsi_val))
        return RSIResult(
            value=round(rsi_val, 2),
            signal=_rsi_signal(rsi_val, overbought, oversold),
            valid=True,
        )
    except Exception:
        return RSIResult(value=50.0, signal="RSI: computation error", valid=False)


def compute_bollinger(
    closes: list[float],
    current_price: float,
    period: int = 20,
    std_multiplier: float = 2.0,
    squeeze_pct: float = 1.5,
) -> BollingerResult:
    """Bollinger Bands with population std. Never raises."""
    try:
        if len(closes) < period:
            return BollingerResult(
                upper=0.0, middle=0.0, lower=0.0,
                bandwidth=0.0, pct_b=0.5,
                signal="BB: insufficient data", valid=False,
            )

        window = closes[-period:]
        middle = sum(window) / period
        variance = sum((x - middle) ** 2 for x in window) / period  # population std
        std = math.sqrt(variance)
        upper = middle + std_multiplier * std
        lower = middle - std_multiplier * std
        bandwidth = (upper - lower) / middle * 100 if middle != 0 else 0.0
        band_range = upper - lower
        if band_range == 0.0:
            pct_b = 0.5
        else:
            raw_pct_b = (current_price - lower) / band_range
            pct_b = max(-0.1, min(1.1, raw_pct_b))

        return BollingerResult(
            upper=round(upper, 4),
            middle=round(middle, 4),
            lower=round(lower, 4),
            bandwidth=round(bandwidth, 4),
            pct_b=round(pct_b, 4),
            signal=_bollinger_signal(pct_b, bandwidth, squeeze_pct),
            valid=True,
        )
    except Exception:
        return BollingerResult(
            upper=0.0, middle=0.0, lower=0.0,
            bandwidth=0.0, pct_b=0.5,
            signal="BB: computation error", valid=False,
        )


# ── Main entry point ───────────────────────────────────────────────────────────

def analyse(
    closes: list[float],
    current_price: float,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    bb_squeeze_pct: float = 1.5,
) -> TechnicalSignals:
    """
    Compute RSI and Bollinger Bands and build the prompt section.
    Header: '=== TECHNICAL SIGNALS ===' (ticker injected by loop).
    prompt_section = '' if both results are invalid.
    Never raises.
    """
    try:
        rsi = compute_rsi(closes, rsi_period, rsi_overbought, rsi_oversold)
        bb  = compute_bollinger(closes, current_price, bb_period, bb_std, bb_squeeze_pct)

        if not rsi.valid and not bb.valid:
            return TechnicalSignals(rsi=rsi, bollinger=bb, prompt_section="")

        lines = ["=== TECHNICAL SIGNALS ==="]
        if rsi.valid:
            lines.append(f"  {rsi.signal}")
        if bb.valid:
            lines.append(f"  {bb.signal}")
            lines.append(
                f"  BB bands: lower=${bb.lower:.2f}  middle=${bb.middle:.2f}  upper=${bb.upper:.2f}"
                f"  bandwidth={bb.bandwidth:.2f}%"
            )

        return TechnicalSignals(
            rsi=rsi,
            bollinger=bb,
            prompt_section="\n".join(lines),
        )
    except Exception:
        _invalid_rsi = RSIResult(value=50.0, signal="", valid=False)
        _invalid_bb  = BollingerResult(
            upper=0.0, middle=0.0, lower=0.0,
            bandwidth=0.0, pct_b=0.5, signal="", valid=False,
        )
        return TechnicalSignals(rsi=_invalid_rsi, bollinger=_invalid_bb, prompt_section="")

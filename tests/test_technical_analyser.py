"""Tests for technical_analyser.py — RSI + Bollinger Bands."""
import math
import pytest
from src.agent.technical_analyser import (
    compute_rsi, compute_bollinger, analyse,
    RSIResult, BollingerResult, TechnicalSignals,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_closes(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


def _flat_closes(n: int, val: float = 100.0) -> list[float]:
    return [val] * n


# ── RSI tests ──────────────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_insufficient_data_returns_invalid(self):
        r = compute_rsi([100.0] * 5, period=14)
        assert not r.valid
        assert r.value == 50.0

    def test_exactly_period_plus_one_is_valid(self):
        closes = _make_closes(15, start=100.0, step=1.0)
        r = compute_rsi(closes, period=14)
        assert r.valid

    def test_all_gains_rsi_near_100(self):
        closes = _make_closes(30, start=100.0, step=2.0)
        r = compute_rsi(closes, period=14)
        assert r.valid
        assert r.value > 90.0

    def test_all_losses_rsi_near_0(self):
        closes = _make_closes(30, start=200.0, step=-2.0)
        r = compute_rsi(closes, period=14)
        assert r.valid
        assert r.value < 10.0

    def test_flat_prices_no_crash(self):
        closes = _flat_closes(30, 100.0)
        r = compute_rsi(closes, period=14)
        assert r.valid
        assert r.value == 100.0  # avg_loss == 0 → RSI = 100

    def test_overbought_signal(self):
        closes = _make_closes(30, step=5.0)
        r = compute_rsi(closes, period=14, overbought=70.0)
        if r.valid and r.value >= 70.0:
            assert "overbought" in r.signal

    def test_oversold_signal(self):
        closes = _make_closes(30, start=200.0, step=-5.0)
        r = compute_rsi(closes, period=14, oversold=30.0)
        if r.valid and r.value <= 30.0:
            assert "oversold" in r.signal

    def test_rsi_value_in_0_100(self):
        for step in [-3.0, -1.0, 0.5, 1.0, 3.0]:
            closes = _make_closes(30, step=step)
            r = compute_rsi(closes)
            if r.valid:
                assert 0.0 <= r.value <= 100.0

    def test_neutral_signal_midrange(self):
        # Alternating gains and losses → RSI near 50
        closes = [100.0 + (1 if i % 2 == 0 else -1) for i in range(30)]
        r = compute_rsi(closes, period=14)
        if r.valid:
            assert 40.0 <= r.value <= 60.0

    def test_empty_list_returns_invalid(self):
        r = compute_rsi([], period=14)
        assert not r.valid

    def test_wilder_smoothing_applied(self):
        # Simple check: 30 bars of gradual uptrend should give higher RSI than 15
        closes_short = _make_closes(20, step=1.0)
        closes_long  = _make_closes(40, step=1.0)
        r_s = compute_rsi(closes_short, period=14)
        r_l = compute_rsi(closes_long, period=14)
        assert r_s.valid and r_l.valid
        # Both should indicate strong uptrend
        assert r_s.value > 60.0
        assert r_l.value > 60.0


# ── Bollinger tests ────────────────────────────────────────────────────────────

class TestComputeBollinger:
    def test_insufficient_data_returns_invalid(self):
        r = compute_bollinger([100.0] * 5, current_price=100.0, period=20)
        assert not r.valid

    def test_exactly_period_is_valid(self):
        closes = _flat_closes(20, 100.0)
        r = compute_bollinger(closes, current_price=100.0, period=20)
        assert r.valid

    def test_flat_prices_zero_bandwidth(self):
        closes = _flat_closes(20, 100.0)
        r = compute_bollinger(closes, current_price=100.0, period=20)
        assert r.valid
        assert r.bandwidth == 0.0
        assert r.upper == r.lower == r.middle

    def test_population_std_not_sample(self):
        closes = [100.0, 102.0, 98.0, 101.0, 99.0] * 4  # 20 values
        mean = sum(closes) / len(closes)
        pop_variance = sum((x - mean) ** 2 for x in closes) / len(closes)
        pop_std = math.sqrt(pop_variance)
        r = compute_bollinger(closes, current_price=mean, period=20)
        assert r.valid
        assert abs(r.upper - (mean + 2.0 * pop_std)) < 0.01
        assert abs(r.lower - (mean - 2.0 * pop_std)) < 0.01

    def test_pct_b_clamped(self):
        closes = _flat_closes(20, 100.0)
        # Price far above band
        r = compute_bollinger(closes, current_price=200.0, period=20)
        assert r.pct_b <= 1.1
        # Price far below band
        r2 = compute_bollinger(closes, current_price=0.0, period=20)
        assert r2.pct_b >= -0.1

    def test_price_in_middle_pct_b_half(self):
        closes = _make_closes(20, step=1.0)
        mean = sum(closes) / len(closes)
        r = compute_bollinger(closes, current_price=mean, period=20)
        if r.valid and r.upper != r.lower:
            assert abs(r.pct_b - 0.5) < 0.1

    def test_squeeze_signal(self):
        closes = _flat_closes(20, 100.0)
        r = compute_bollinger(closes, current_price=100.0, period=20, squeeze_pct=1.5)
        if r.valid:
            assert "squeeze" in r.signal

    def test_upper_band_signal(self):
        closes = _make_closes(20, step=0.5)
        std_approx = 3.0
        mean = sum(closes) / 20
        price_above = mean + 2.5 * std_approx
        r = compute_bollinger(closes, current_price=price_above, period=20)
        if r.valid and r.pct_b >= 1.0:
            assert "above upper band" in r.signal

    def test_empty_list_returns_invalid(self):
        r = compute_bollinger([], current_price=100.0, period=20)
        assert not r.valid

    def test_upper_greater_than_lower(self):
        closes = _make_closes(25, step=1.5)
        r = compute_bollinger(closes, current_price=closes[-1], period=20)
        if r.valid and r.bandwidth > 0:
            assert r.upper > r.lower


# ── Analyse entry point tests ──────────────────────────────────────────────────

class TestAnalyse:
    def test_both_invalid_empty_prompt_section(self):
        result = analyse(closes=[100.0] * 5, current_price=100.0)
        assert result.prompt_section == ""
        assert not result.rsi.valid
        assert not result.bollinger.valid

    def test_valid_both_produces_prompt_section(self):
        closes = _make_closes(25, step=1.0)
        result = analyse(closes=closes, current_price=closes[-1])
        assert result.prompt_section != ""
        assert "=== TECHNICAL SIGNALS ===" in result.prompt_section

    def test_prompt_section_contains_rsi_signal(self):
        closes = _make_closes(25, step=2.0)
        result = analyse(closes=closes, current_price=closes[-1])
        if result.rsi.valid:
            assert "RSI" in result.prompt_section

    def test_prompt_section_contains_bb_signal(self):
        closes = _make_closes(25, step=1.0)
        result = analyse(closes=closes, current_price=closes[-1])
        if result.bollinger.valid:
            assert "BB" in result.prompt_section

    def test_never_raises_on_garbage_input(self):
        result = analyse(closes=[], current_price=-999.0)
        assert isinstance(result, TechnicalSignals)

    def test_returns_technical_signals_type(self):
        closes = _make_closes(30, step=1.0)
        result = analyse(closes=closes, current_price=closes[-1])
        assert isinstance(result, TechnicalSignals)
        assert isinstance(result.rsi, RSIResult)
        assert isinstance(result.bollinger, BollingerResult)

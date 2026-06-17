"""Tests for position_manager.py."""
import pytest
from src.agent.position_manager import PositionManager, PositionThresholds, PositionState
from src.agent import config


def _make_pm() -> PositionManager:
    pm = PositionManager()
    pm._session_id = "test-session"
    return pm


class TestOnNewPosition:
    def test_creates_state(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", entry_price=200.0, entry_cycle=1, qty=5)
        state = pm.get_state("TSLA")
        assert state is not None
        assert state.entry_price == 200.0
        assert state.qty == 5

    def test_empty_price_history(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        assert pm.get_state("TSLA").price_history == []


class TestUpdatePrice:
    def test_appends_price(self):
        pm = _make_pm()
        pm.on_new_position("AAPL", 150.0, 1, 10)
        pm.update_price("AAPL", 152.0)
        pm.update_price("AAPL", 155.0)
        assert pm.get_state("AAPL").price_history == [152.0, 155.0]

    def test_creates_minimal_state_if_missing(self):
        pm = _make_pm()
        pm.update_price("MSFT", 300.0)
        assert pm.get_state("MSFT") is not None

    def test_trims_to_history_cycles(self):
        pm = _make_pm()
        pm.on_new_position("NVDA", 400.0, 1, 3)
        for i in range(config.POSITION_HISTORY_CYCLES + 5):
            pm.update_price("NVDA", 400.0 + i)
        hist = pm.get_state("NVDA").price_history
        assert len(hist) == config.POSITION_HISTORY_CYCLES


class TestOnPositionClosed:
    def test_removes_state(self):
        pm = _make_pm()
        pm.on_new_position("AMD", 80.0, 1, 10)
        pm.on_position_closed("AMD")
        assert pm.get_state("AMD") is None

    def test_no_crash_on_missing_ticker(self):
        pm = _make_pm()
        pm.on_position_closed("XOM")  # should not raise


class TestCheckStopLoss:
    def test_triggers_when_below_threshold(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        closes = [200.0] * 20
        pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        state = pm.get_state("TSLA")
        # Force a stop-loss at -3%
        state.thresholds.stop_loss_pct = -3.0
        assert pm.check_stop_loss("TSLA", 193.0)   # -3.5% < -3%

    def test_does_not_trigger_above_threshold(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        state = pm.get_state("TSLA")
        state.thresholds.stop_loss_pct = -5.0
        assert not pm.check_stop_loss("TSLA", 198.0)  # -1%

    def test_false_when_ticker_not_tracked(self):
        pm = _make_pm()
        assert not pm.check_stop_loss("UNKNOWN", 100.0)


class TestCheckTakeProfit:
    def test_triggers_when_above_threshold(self):
        pm = _make_pm()
        pm.on_new_position("NEE", 100.0, 1, 20)
        state = pm.get_state("NEE")
        state.thresholds.take_profit_pct = 5.0
        assert pm.check_take_profit("NEE", 106.0)  # +6%

    def test_false_when_ticker_not_tracked(self):
        pm = _make_pm()
        assert not pm.check_take_profit("UNKNOWN", 200.0)


class TestUpdateThresholds:
    def test_returns_thresholds_object(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        closes = [200.0 + i * 0.5 for i in range(20)]
        t = pm.update_thresholds("TSLA", closes, [0.1, 0.2, 0.3], "normal", None, None)
        assert isinstance(t, PositionThresholds)

    def test_stop_loss_is_negative(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        closes = [200.0] * 20
        t = pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        assert t.stop_loss_pct < 0

    def test_take_profit_is_positive(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        closes = [200.0] * 20
        t = pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        assert t.take_profit_pct > 0

    def test_conservative_mode_tightens_thresholds(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        # ~1.5% swings → raw_stop ≈ -4.5%, normal stays -4.5%, conservative clamps to -2.0%
        closes = [200.0, 203.0, 200.0, 203.0, 200.0] * 4
        t_normal = pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        t_cons   = pm.update_thresholds("TSLA", closes, [], "conservative", None, None)
        assert t_cons.stop_loss_pct > t_normal.stop_loss_pct  # less negative

    def test_user_override_more_restrictive(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        closes = [200.0] * 20
        t = pm.update_thresholds("TSLA", closes, [], "normal",
                                  user_stop_loss_pct=-1.5, user_take_profit_pct=4.0)
        assert t.stop_source == "user"
        assert t.stop_loss_pct == -1.5

    def test_bounds_respected(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 5)
        # Extremely volatile closes
        closes = [100.0, 200.0, 50.0, 180.0] * 5
        t = pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        assert t.stop_loss_pct  >= -config.POSITION_MAX_STOP_LOSS_PCT
        assert t.stop_loss_pct  <= -config.POSITION_MIN_STOP_LOSS_PCT
        assert t.take_profit_pct >= config.POSITION_MIN_TAKE_PROFIT_PCT
        assert t.take_profit_pct <= config.POSITION_MAX_TAKE_PROFIT_PCT


class TestBuildPositionContext:
    def test_returns_empty_when_not_tracked(self):
        pm = _make_pm()
        assert pm.build_position_context("AAPL", 150.0, 5) == ""

    def test_contains_required_sections(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 200.0, 1, 10)
        pm.update_price("TSLA", 210.0)
        closes = [200.0 + i for i in range(20)]
        pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        ctx = pm.build_position_context("TSLA", 210.0, 5)
        assert "=== POSITION CONTEXT (TSLA) ===" in ctx
        assert "Stop-loss" in ctx
        assert "Take-profit" in ctx
        assert "P&L" in ctx

    def test_pnl_positive(self):
        pm = _make_pm()
        pm.on_new_position("TSLA", 100.0, 1, 10)
        closes = [100.0] * 20
        pm.update_thresholds("TSLA", closes, [], "normal", None, None)
        ctx = pm.build_position_context("TSLA", 110.0, 5)
        assert "+10.00%" in ctx

    def test_never_raises(self):
        pm = _make_pm()
        # Corrupt state
        pm._states["BAD"] = None  # type: ignore
        result = pm.build_position_context("BAD", 100.0, 1)
        assert isinstance(result, str)

"""Tests for user_preference_engine.py."""
import pytest
from src.agent.user_preference_engine import UserPreferenceEngine
from src.agent import config


def _make_session(**overrides) -> dict:
    base = {
        "session_id":             "test-session",
        "pref_sectors":           [],
        "pref_excluded_sectors":  [],
        "pref_risk_level":        "unspecified",
        "pref_ethics":            [],
        "pref_time_horizon":      "unspecified",
        "pref_emotion":           "neutral",
        "pref_emotion_score":     0.0,
        "style_hold_rate":        0.5,
        "style_confirm_rate":     0.5,
        "style_override_count":   0,
        "style_reject_sl_count":  0,
        "style_inferred":         "undetected",
        "derived_confidence_delta":   0.0,
        "derived_position_pct_delta": 0.0,
        "derived_mode_bias":      "none",
        "wait_choices":           [],
        "preference_conflicts":   [],
        "user_stop_loss_pct":     None,
        "user_take_profit_pct":   None,
    }
    base.update(overrides)
    return base


def _make_upe(**session_overrides) -> UserPreferenceEngine:
    return UserPreferenceEngine(_make_session(**session_overrides))


# ── sector_of ──────────────────────────────────────────────────────────────────

class TestSectorOf:
    def test_known_ticker(self):
        upe = _make_upe()
        assert upe.sector_of("TSLA") == "tech"

    def test_fossil_fuel(self):
        upe = _make_upe()
        assert upe.sector_of("XOM") == "fossil_fuel"

    def test_unknown_ticker_returns_none(self):
        upe = _make_upe()
        assert upe.sector_of("XXXX") is None

    def test_case_insensitive(self):
        upe = _make_upe()
        assert upe.sector_of("tsla") == "tech"


# ── compute_derived_parameters ────────────────────────────────────────────────

class TestComputeDerivedParameters:
    def test_neutral_defaults_zero_delta(self):
        upe = _make_upe()
        dp = upe.compute_derived_parameters()
        assert dp.confidence_delta == 0.0
        assert dp.position_pct_delta == 0.0
        assert dp.mode_bias == "none"

    def test_anxious_sets_conservative_bias(self):
        upe = _make_upe(pref_emotion="anxious", pref_emotion_score=-0.6)
        dp = upe.compute_derived_parameters()
        assert dp.mode_bias == "conservative_bias"

    def test_low_risk_increases_confidence_delta(self):
        upe = _make_upe(pref_risk_level="low")
        dp = upe.compute_derived_parameters()
        assert dp.confidence_delta > 0

    def test_high_risk_aggressive_sets_normal_bias(self):
        upe = _make_upe(pref_risk_level="high", style_inferred="aggressive")
        dp = upe.compute_derived_parameters()
        assert dp.mode_bias == "normal_bias"

    def test_confidence_delta_clamped(self):
        # Force extreme values
        upe = _make_upe(
            pref_emotion_score=1.0,
            pref_risk_level="low",
            style_inferred="cautious",
        )
        dp = upe.compute_derived_parameters()
        assert -0.15 <= dp.confidence_delta <= 0.15

    def test_position_pct_delta_clamped(self):
        upe = _make_upe(pref_risk_level="high", style_inferred="aggressive")
        dp = upe.compute_derived_parameters()
        assert -0.05 <= dp.position_pct_delta <= 0.05

    def test_session_updated(self):
        session = _make_session()
        upe = UserPreferenceEngine(session)
        upe.compute_derived_parameters()
        assert "derived_confidence_delta" in session


# ── get_effective_confidence_threshold ────────────────────────────────────────

class TestGetEffectiveConfidenceThreshold:
    def test_zero_delta_returns_base(self):
        upe = _make_upe(derived_confidence_delta=0.0)
        assert upe.get_effective_confidence_threshold(0.65) == pytest.approx(0.65)

    def test_positive_delta_raises_threshold(self):
        upe = _make_upe(derived_confidence_delta=0.10)
        assert upe.get_effective_confidence_threshold(0.65) == pytest.approx(0.75)

    def test_clamped_at_095(self):
        upe = _make_upe(derived_confidence_delta=0.15)
        assert upe.get_effective_confidence_threshold(0.90) == pytest.approx(0.95)

    def test_clamped_at_050(self):
        upe = _make_upe(derived_confidence_delta=-0.15)
        assert upe.get_effective_confidence_threshold(0.55) == pytest.approx(0.50)


# ── get_effective_position_pct ────────────────────────────────────────────────

class TestGetEffectivePositionPct:
    def test_zero_delta_returns_base(self):
        upe = _make_upe(derived_position_pct_delta=0.0)
        assert upe.get_effective_position_pct(0.10) == pytest.approx(0.10)

    def test_clamped_at_015(self):
        upe = _make_upe(derived_position_pct_delta=0.05)
        assert upe.get_effective_position_pct(0.13) == pytest.approx(0.15)

    def test_clamped_at_002(self):
        upe = _make_upe(derived_position_pct_delta=-0.05)
        assert upe.get_effective_position_pct(0.04) == pytest.approx(0.02)


# ── check_conflict ────────────────────────────────────────────────────────────

class TestCheckConflict:
    def test_no_conflict_default(self):
        upe = _make_upe()
        result = upe.check_conflict("TSLA", "buy", 0.02, 0.03, 0.1, "normal")
        assert result is None

    def test_conflict1_buying_while_losing(self):
        upe = _make_upe(pref_risk_level="low")
        result = upe.check_conflict("TSLA", "buy", -0.10, -0.05, 0.1, "normal")
        assert result is not None
        assert result["type"] == "buying_while_losing"

    def test_conflict2_excluded_sector(self):
        upe = _make_upe(pref_excluded_sectors=["fossil_fuel"])
        result = upe.check_conflict("XOM", "buy", 0.0, 0.0, 0.0, "normal")
        assert result is not None
        assert result["type"] == "excluded_sector"
        assert result["modified_action"] == "hold"

    def test_conflict3_anxious_plus_negative_sentiment(self):
        upe = _make_upe(pref_emotion="anxious")
        result = upe.check_conflict("TSLA", "buy", 0.0, 0.0, -0.50, "normal")
        assert result is not None
        assert result["type"] == "emotional_vs_sentiment"

    def test_conflict4_high_risk_conservative_mode(self):
        upe = _make_upe(pref_risk_level="high")
        result = upe.check_conflict("TSLA", "buy", 0.0, 0.0, 0.1, "conservative")
        assert result is not None
        assert result["type"] == "risk_vs_conservative_mode"

    def test_sell_action_no_conflict(self):
        upe = _make_upe(pref_risk_level="low")
        result = upe.check_conflict("TSLA", "sell", -0.10, -0.05, -0.5, "normal")
        assert result is None

    def test_never_raises(self):
        upe = _make_upe()
        upe._session = None  # type: ignore
        result = upe.check_conflict("X", "buy", 0, 0, 0, "normal")
        assert result is None


# ── apply_minimum_modification ────────────────────────────────────────────────

class TestApplyMinimumModification:
    def test_type1_increases_confidence_delta(self):
        session = _make_session(derived_confidence_delta=0.0)
        upe = UserPreferenceEngine(session)
        conflict = {"type": "buying_while_losing", "modified_action": "buy", "description": "test"}
        upe.apply_minimum_modification(conflict, session)
        assert session["derived_confidence_delta"] > 0

    def test_type1_sets_conservative_bias(self):
        session = _make_session()
        upe = UserPreferenceEngine(session)
        conflict = {"type": "buying_while_losing", "modified_action": "buy", "description": "test"}
        upe.apply_minimum_modification(conflict, session)
        assert session["derived_mode_bias"] == "conservative_bias"

    def test_conflict_logged(self):
        session = _make_session()
        upe = UserPreferenceEngine(session)
        conflict = {"type": "excluded_sector", "modified_action": "hold", "description": "test"}
        upe.apply_minimum_modification(conflict, session)
        assert len(session["preference_conflicts"]) == 1


# ── build_prompt_section ──────────────────────────────────────────────────────

class TestBuildPromptSection:
    def test_empty_on_all_defaults(self):
        upe = _make_upe()
        assert upe.build_prompt_section() == ""

    def test_non_empty_when_risk_set(self):
        upe = _make_upe(pref_risk_level="low")
        upe.compute_derived_parameters()
        section = upe.build_prompt_section()
        assert "=== USER PREFERENCES ===" in section

    def test_contains_emotion(self):
        upe = _make_upe(pref_emotion="anxious", pref_emotion_score=-0.5)
        upe.compute_derived_parameters()
        section = upe.build_prompt_section()
        if section:
            assert "anxious" in section

    def test_excluded_sectors_shown(self):
        upe = _make_upe(pref_excluded_sectors=["fossil_fuel"])
        upe.compute_derived_parameters()
        section = upe.build_prompt_section()
        if section:
            assert "fossil_fuel" in section


# ── record_wait_choice ────────────────────────────────────────────────────────

class TestRecordWaitChoice:
    def test_appends_choice(self):
        session = _make_session()
        upe = UserPreferenceEngine(session)
        upe.record_wait_choice(1, "confirmed", "TSLA", "hold")
        assert len(session["wait_choices"]) == 1

    def test_trims_to_limit(self):
        session = _make_session()
        upe = UserPreferenceEngine(session)
        for i in range(config.PREFERENCE_WAIT_HISTORY + 5):
            upe.record_wait_choice(i, "confirmed", "TSLA", "hold")
        assert len(session["wait_choices"]) == config.PREFERENCE_WAIT_HISTORY

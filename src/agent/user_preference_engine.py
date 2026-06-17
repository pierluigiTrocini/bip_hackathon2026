"""
User Preference Engine — extract explicit + emotional preferences from the user
prompt, infer implicit trading style from journal history, and compute derived
parameters (confidence delta, position size delta, mode bias).
R1: every public method wraps body in try/except.
R12: extract_from_prompt() is NEVER called inside _run_cycle().
R13: _SECTOR_MAP is static and never modified at runtime.
"""
import json
import os
import threading
from dataclasses import dataclass

from src.agent import config
from src.agent import journal as journal_module
from src.agent import llm_stream


_SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech",    "MSFT": "tech",    "GOOGL": "tech",   "NVDA": "tech",
    "AMD":  "tech",    "INTC": "tech",    "META":  "tech",   "AMZN": "tech",
    "NFLX": "tech",    "TSLA": "tech",    "RIVN":  "tech",   "NIO":  "tech",
    "XOM":  "fossil_fuel", "CVX": "fossil_fuel", "BP":  "fossil_fuel",
    "SHEL": "fossil_fuel", "COP": "fossil_fuel",
    "NEE":  "renewable_energy", "ENPH": "renewable_energy",
    "SEDG": "renewable_energy", "FSLR": "renewable_energy",
    "JPM":  "finance", "BAC":  "finance", "GS":   "finance",
    "MS":   "finance", "C":    "finance", "WFC":  "finance",
    "JNJ":  "healthcare", "PFE": "healthcare", "MRK": "healthcare",
    "ABBV": "healthcare", "UNH": "healthcare",
    "LMT":  "defense", "RTX":  "defense", "NOC":  "defense",
    "GD":   "defense", "BA":   "defense",
    "PG":   "consumer","KO":   "consumer","MCD":  "consumer",
    "WMT":  "consumer","TGT":  "consumer",
}

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sectors":          {"type": "array",  "items": {"type": "string"}},
        "excluded_sectors": {"type": "array",  "items": {"type": "string"}},
        "risk_level":       {"type": "string"},
        "ethics":           {"type": "array",  "items": {"type": "string"}},
        "time_horizon":     {"type": "string"},
        "emotion_label":    {"type": "string"},
        "emotion_score":    {"type": "number"},
    },
    "required": ["sectors","excluded_sectors","risk_level","ethics",
                 "time_horizon","emotion_label","emotion_score"],
}

_VALID_RISK_LEVELS   = {"low", "medium", "high", "unspecified"}
_VALID_EMOTIONS      = {"optimistic", "pessimistic", "anxious", "neutral"}
_VALID_HORIZONS      = {"short", "medium", "long", "unspecified"}
_VALID_ETHICS        = {"no_weapons","esg_only","no_fossil_fuel","no_gambling","no_tobacco"}
_VALID_SECTORS       = {
    "tech","finance","healthcare","energy","renewable_energy",
    "consumer","industrial","defense","fossil_fuel",
}


@dataclass
class DerivedParameters:
    confidence_delta:   float   # [-0.15, +0.15]
    position_pct_delta: float   # [-0.05, +0.05]
    mode_bias:          str     # "conservative_bias" | "normal_bias" | "none"


class UserPreferenceEngine:
    def __init__(self, session: dict) -> None:
        self._session = session
        self._lock = threading.Lock()

    # ── Sector lookup ───────────────────────────────────────────────────────────

    def sector_of(self, ticker: str) -> str | None:
        try:
            return _SECTOR_MAP.get(ticker.upper())
        except Exception:
            return None

    # ── Preference extraction (LLM, called outside _run_cycle) ─────────────────

    def extract_from_prompt(self, prompt: str, t_behavior: int) -> None:
        """Call qwen2.5:3b to extract preferences from the user's prompt. Never raises."""
        try:
            extraction_prompt = (
                f'Analyse this trading instruction and extract structured preferences.\n\n'
                f'Instruction: "{prompt}"\n\n'
                f'Return ONLY valid JSON:\n'
                f'{{"sectors": [], "excluded_sectors": [], "risk_level": "", '
                f'"ethics": [], "time_horizon": "", "emotion_label": "", "emotion_score": 0.0}}\n\n'
                f'Vocabulary for sectors/excluded_sectors:\n'
                f'tech, finance, healthcare, energy, renewable_energy, consumer, '
                f'industrial, defense, fossil_fuel\n\n'
                f'Vocabulary for ethics:\nno_weapons, esg_only, no_fossil_fuel, no_gambling, no_tobacco\n\n'
                f'Values for risk_level: low, medium, high, unspecified\n'
                f'Values for time_horizon: short, medium, long, unspecified\n'
                f'Values for emotion_label: optimistic, pessimistic, anxious, neutral'
            )
            raw = llm_stream.generate(
                model=config.OLLAMA_SENTIMENT_MODEL,
                prompt=extraction_prompt,
                format=_EXTRACTION_SCHEMA,
                options={"temperature": 0.0, "num_predict": 200},
                keep_alive="30s",
                show_output=False,
            )
            # Find JSON object
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return
            parsed = json.loads(raw[start:end])

            with self._lock:
                s = self._session
                s["pref_sectors"]          = [x for x in parsed.get("sectors", [])
                                               if x in _VALID_SECTORS]
                s["pref_excluded_sectors"] = [x for x in parsed.get("excluded_sectors", [])
                                               if x in _VALID_SECTORS]
                rl = parsed.get("risk_level", "unspecified")
                s["pref_risk_level"]       = rl if rl in _VALID_RISK_LEVELS else "unspecified"
                s["pref_ethics"]           = [x for x in parsed.get("ethics", [])
                                               if x in _VALID_ETHICS]
                th = parsed.get("time_horizon", "unspecified")
                s["pref_time_horizon"]     = th if th in _VALID_HORIZONS else "unspecified"
                el = parsed.get("emotion_label", "neutral")
                s["pref_emotion"]          = el if el in _VALID_EMOTIONS else "neutral"
                es = float(parsed.get("emotion_score", 0.0))
                s["pref_emotion_score"]    = max(-1.0, min(1.0, es))
        except Exception as exc:
            try:
                journal_module.log_error(
                    source="UserPreferenceEngine",
                    error=f"extract_from_prompt failed: {exc}",
                    session_id=self._session.get("session_id", ""),
                )
            except Exception:
                pass

    # ── Implicit style inference ────────────────────────────────────────────────

    def update_implicit_style(self, session_id: str, journal_path: str) -> None:
        """Read journal and infer implicit trading style. Runs in daemon thread."""
        try:
            if not os.path.exists(journal_path):
                return

            actions: list[str] = []
            with open(journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        if e.get("session_id") == session_id:
                            actions.append(e.get("action", "hold"))
                    except Exception:
                        pass

            if not actions:
                return

            total = len(actions)
            hold_rate    = actions.count("hold") / total
            confirm_rate = 1.0 - (actions.count("sell") / total)  # proxy

            wait_choices = self._session.get("wait_choices", [])
            override_count   = sum(1 for w in wait_choices if w.get("source") == "override")
            reject_sl_count  = int(self._session.get("style_reject_sl_count", 0))

            if hold_rate > 0.70 or override_count == 0:
                inferred = "cautious"
            elif hold_rate < 0.30 or override_count > 3:
                inferred = "aggressive"
            elif hold_rate < 0.50:
                inferred = "moderate"
            else:
                inferred = "undetected"

            with self._lock:
                self._session["style_hold_rate"]      = round(hold_rate, 4)
                self._session["style_confirm_rate"]   = round(confirm_rate, 4)
                self._session["style_override_count"] = override_count
                self._session["style_reject_sl_count"]= reject_sl_count
                self._session["style_inferred"]       = inferred

        except Exception as exc:
            try:
                journal_module.log_error(
                    source="UserPreferenceEngine",
                    error=f"update_implicit_style failed: {exc}",
                    session_id=session_id,
                )
            except Exception:
                pass

    # ── Wait choice tracking ────────────────────────────────────────────────────

    def record_wait_choice(
        self, cycle: int, source: str, ticker: str, agent_action: str
    ) -> None:
        """Append wait choice for implicit style inference."""
        try:
            entry = {
                "cycle": cycle, "source": source,
                "ticker": ticker, "agent_action": agent_action,
            }
            with self._lock:
                choices = self._session.get("wait_choices", [])
                choices.append(entry)
                if len(choices) > config.PREFERENCE_WAIT_HISTORY:
                    choices = choices[-config.PREFERENCE_WAIT_HISTORY:]
                self._session["wait_choices"] = choices
        except Exception:
            pass

    # ── Conflict detection ──────────────────────────────────────────────────────

    def check_conflict(
        self,
        ticker: str,
        proposed_action: str,
        current_pnl_pct: float,
        ticker_pnl_pct: float,
        sentiment_score: float,
        mode: str,
    ) -> dict | None:
        """Return conflict dict if any condition fires, else None."""
        try:
            s = self._session

            # Conflict 1: buying while portfolio is losing + conservative/unspecified risk
            if (proposed_action == "buy"
                    and current_pnl_pct < -config.PREFERENCE_CONFLICT_THRESHOLD
                    and s.get("pref_risk_level") in ("low", "unspecified")):
                return {
                    "type": "buying_while_losing",
                    "modified_action": proposed_action,
                    "description": (
                        f"Portfolio losing ({current_pnl_pct:+.2%}) "
                        f"with risk profile '{s['pref_risk_level']}'"
                    ),
                }

            # Conflict 2: excluded sector
            sector = self.sector_of(ticker)
            excluded = s.get("pref_excluded_sectors", [])
            if proposed_action == "buy" and sector and sector in excluded:
                return {
                    "type": "excluded_sector",
                    "modified_action": "hold",
                    "description": (
                        f"Sector '{sector}' excluded by user preferences"
                    ),
                }

            # Conflict 3: emotional state vs negative sentiment
            if (proposed_action == "buy"
                    and s.get("pref_emotion") in ("anxious", "pessimistic")
                    and sentiment_score < -0.30):
                return {
                    "type": "emotional_vs_sentiment",
                    "modified_action": proposed_action,
                    "description": (
                        f"Emotional tone '{s['pref_emotion']}' "
                        f"with negative sentiment ({sentiment_score:+.2f})"
                    ),
                }

            # Conflict 4: high risk preference in conservative mode
            if (proposed_action == "buy"
                    and mode == "conservative"
                    and s.get("pref_risk_level") == "high"):
                return {
                    "type": "risk_vs_conservative_mode",
                    "modified_action": proposed_action,
                    "description": (
                        f"Risk preference 'high' in conservative mode"
                    ),
                }

            return None
        except Exception:
            return None

    def apply_minimum_modification(self, conflict: dict, session: dict) -> dict:
        """Apply smallest change to resolve conflict. Log to preference_conflicts."""
        try:
            ctype = conflict.get("type", "")

            if ctype == "buying_while_losing":
                delta = session.get("derived_confidence_delta", 0.0) + 0.10
                session["derived_confidence_delta"] = max(-0.15, min(0.15, delta))
                session["derived_mode_bias"] = "conservative_bias"

            elif ctype == "excluded_sector":
                pass  # modified_action already set to "hold" in conflict dict

            elif ctype == "emotional_vs_sentiment":
                delta = session.get("derived_confidence_delta", 0.0) + 0.05
                session["derived_confidence_delta"] = max(-0.15, min(0.15, delta))

            elif ctype == "risk_vs_conservative_mode":
                pass  # keep mode = "conservative", just log

            # Log to preference_conflicts (cap at 50)
            conflicts = session.get("preference_conflicts", [])
            conflicts.append(conflict)
            if len(conflicts) > 50:
                conflicts = conflicts[-50:]
            session["preference_conflicts"] = conflicts

            return conflict
        except Exception:
            return conflict

    # ── Derived parameters ──────────────────────────────────────────────────────

    def compute_derived_parameters(self) -> DerivedParameters:
        """Compute and store derived confidence/position deltas. Never raises."""
        try:
            s = self._session

            # Confidence delta
            delta = 0.0
            delta += -s.get("pref_emotion_score", 0.0) * config.PREFERENCE_EMOTION_WEIGHT
            delta += {
                "low": +0.10, "medium": 0.0, "high": -0.10, "unspecified": 0.0
            }.get(s.get("pref_risk_level", "unspecified"), 0.0)
            style_map = {"cautious": +0.10, "moderate": 0.0, "aggressive": -0.10, "undetected": 0.0}
            delta += style_map.get(s.get("style_inferred", "undetected"), 0.0) * config.PREFERENCE_STYLE_WEIGHT
            derived_conf_delta = max(-0.15, min(0.15, delta))

            # Position size delta
            pos = 0.0
            pos += {"low": -0.03, "medium": 0.0, "high": +0.03, "unspecified": 0.0}.get(
                s.get("pref_risk_level", "unspecified"), 0.0
            )
            pos += {"cautious": -0.02, "moderate": 0.0, "aggressive": +0.02, "undetected": 0.0}.get(
                s.get("style_inferred", "undetected"), 0.0
            )
            derived_pos_delta = max(-0.05, min(0.05, pos))

            # Mode bias
            emotion = s.get("pref_emotion", "neutral")
            risk    = s.get("pref_risk_level", "unspecified")
            style   = s.get("style_inferred", "undetected")
            if emotion in ("anxious", "pessimistic") or risk == "low" or style == "cautious":
                mode_bias = "conservative_bias"
            elif risk == "high" and style == "aggressive":
                mode_bias = "normal_bias"
            else:
                mode_bias = "none"

            with self._lock:
                s["derived_confidence_delta"]   = round(derived_conf_delta, 4)
                s["derived_position_pct_delta"] = round(derived_pos_delta, 4)
                s["derived_mode_bias"]          = mode_bias

            return DerivedParameters(
                confidence_delta=derived_conf_delta,
                position_pct_delta=derived_pos_delta,
                mode_bias=mode_bias,
            )
        except Exception:
            return DerivedParameters(confidence_delta=0.0, position_pct_delta=0.0, mode_bias="none")

    # ── Effective parameter getters ─────────────────────────────────────────────

    def get_effective_confidence_threshold(self, base_threshold: float) -> float:
        """base + derived_confidence_delta, clamped [0.50, 0.95]. Never raises."""
        try:
            delta = float(self._session.get("derived_confidence_delta", 0.0))
            return max(0.50, min(0.95, base_threshold + delta))
        except Exception:
            return base_threshold

    def get_effective_position_pct(self, base_pct: float) -> float:
        """base + derived_position_pct_delta, clamped [0.02, 0.15]. Never raises."""
        try:
            delta = float(self._session.get("derived_position_pct_delta", 0.0))
            return max(0.02, min(0.15, base_pct + delta))
        except Exception:
            return base_pct

    # ── Prompt section builder ──────────────────────────────────────────────────

    def build_prompt_section(self) -> str:
        """Build === USER PREFERENCES === section. Returns '' if no meaningful preferences."""
        try:
            s = self._session
            emotion  = s.get("pref_emotion", "neutral")
            risk     = s.get("pref_risk_level", "unspecified")
            style    = s.get("style_inferred", "undetected")
            sectors  = s.get("pref_sectors", [])
            excluded = s.get("pref_excluded_sectors", [])
            ethics   = s.get("pref_ethics", [])

            if (emotion == "neutral"
                    and risk == "unspecified"
                    and style == "undetected"
                    and not sectors and not excluded and not ethics):
                return ""

            base_conf_normal = config.CONFIDENCE_THRESHOLD_NORMAL
            base_pos_normal  = config.MAX_POSITION_PCT_NORMAL
            eff_conf = self.get_effective_confidence_threshold(base_conf_normal)
            eff_pos  = self.get_effective_position_pct(base_pos_normal)
            conf_delta = float(s.get("derived_confidence_delta", 0.0))
            pos_delta  = float(s.get("derived_position_pct_delta", 0.0))
            mode_bias  = s.get("derived_mode_bias", "none")
            hold_rate  = float(s.get("style_hold_rate", 0.5))
            confirm_rate = float(s.get("style_confirm_rate", 0.5))
            emotion_score = float(s.get("pref_emotion_score", 0.0))

            lines = [
                "=== USER PREFERENCES ===",
                f"Detected style:    {style}  (hold_rate: {hold_rate:.0%}, confirm_rate: {confirm_rate:.0%})",
                f"Emotional tone:    {emotion} [{emotion_score:+.2f}]",
                f"Risk:              {risk}",
                f"Preferred sectors: {', '.join(sectors) if sectors else 'none specified'}",
                f"Excluded sectors:  {', '.join(excluded) if excluded else 'none'}",
                f"Ethical filters:   {', '.join(ethics) if ethics else 'none'}",
                "",
                "Adapted parameters:",
                f"  Confidence threshold: {base_conf_normal:.2f} → {eff_conf:.2f}  (delta: {conf_delta:+.2f})",
                f"  Position size:        {base_pos_normal:.0%} → {eff_pos:.0%}  (delta: {pos_delta:+.2f})",
                f"  Mode bias:            {mode_bias}",
            ]
            return "\n".join(lines)
        except Exception:
            return ""

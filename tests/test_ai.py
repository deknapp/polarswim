"""AI review: credential resolution, prompt construction, offline fallback."""

import pytest

from polarswim import ai

SETS = [
    {"set_id": 1, "n": 6, "stroke": "freestyle", "confidence": 0.65,
     "pace_s": 29.0, "hr_cost": 14.0, "rest_before_s": 0.0, "note": ""},
    {"set_id": 2, "n": 12, "stroke": "undetermined", "confidence": 0.33,
     "pace_s": 32.0, "hr_cost": 42.0, "rest_before_s": 106.0, "note": "repaired"},
]
HEADER = {"start_time": "2026-08-19T17:11", "distance_m": 1394.0,
          "duration_s": 2822.0, "avg_hr": 126, "max_hr": 157, "pool_length_m": 22.86}
PARAMS = {"_global": {"pace_p10": 22.4, "pace_p50": 26.0, "pace_p90": 34.4,
                      "n_obs": 7615}}


def test_prompt_contains_the_derived_facts():
    p = ai.build_prompt(HEADER, SETS, PARAMS)
    assert "freestyle" in p and "undetermined" in p and "repaired" in p
    assert "7615" in p                      # the swimmer's own reference paces


def test_prompt_carries_confidence_so_the_model_can_hedge():
    assert "0.33" in ai.build_prompt(HEADER, SETS, PARAMS)


def test_system_prompt_states_the_labels_are_inferred():
    """Guards against the model presenting an estimate as a measurement."""
    assert "OTHER" in ai.SYSTEM
    assert "inferred" in ai.SYSTEM.lower() and "confidence" in ai.SYSTEM.lower()


def test_offline_review_works_without_a_key():
    out = ai.review_offline(HEADER, SETS, PARAMS)
    assert out.model == "offline"
    assert "18 lengths" in out.text or "lengths" in out.text


def test_offline_review_flags_low_confidence_sets():
    assert "low-confidence" in ai.review_offline(HEADER, SETS, PARAMS).text


def test_review_falls_back_offline_with_no_credential(monkeypatch):
    monkeypatch.setattr(ai, "_load_env_key", lambda: None)
    assert ai.review(HEADER, SETS, PARAMS).model == "offline"


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    assert ai._load_env_key() == "sk-test-value"
    assert ai.available()


def test_key_is_read_from_a_dotenv_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nANTHROPIC_API_KEY="sk-from-file"\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ai, "ENV_FILES", (env,))
    assert ai._load_env_key() == "sk-from-file"


def test_uses_the_current_model_id():
    assert ai.MODEL == "claude-opus-5"

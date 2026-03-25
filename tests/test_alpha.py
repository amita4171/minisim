"""Tests for question-specific alpha signals."""
from src.agents.alpha import _compute_question_alpha, _compute_domain_expertise


def test_rare_events_get_negative_alpha():
    alpha = _compute_question_alpha("Will a major US bank fail in 2026?", "econ")
    assert alpha < 0, "Rare dramatic events should get negative alpha"


def test_institutional_inertia_negative():
    alpha = _compute_question_alpha("Will Congress pass a major regulation bill?", "political")
    assert alpha < 0, "Legislation should get negative alpha (slow)"


def test_continuation_positive():
    alpha = _compute_question_alpha("Will AI investment continue to grow?", "tech")
    assert alpha > 0 or alpha == 0  # continuation or neutral


def test_alpha_bounded():
    questions = [
        "Will the world end tomorrow?",
        "Will a devastating nuclear war happen?",
        "Will Congress pass reform legislation and ban all AI?",
    ]
    for q in questions:
        alpha = _compute_question_alpha(q, "political")
        assert -0.15 <= alpha <= 0.15, f"Alpha {alpha} out of bounds for '{q}'"


def test_domain_expertise_match():
    bonus = _compute_domain_expertise("Macro Economist", "Will the Fed cut rates?", "econ")
    assert bonus > 0.1, "Macro economist should have high expertise on Fed question"


def test_domain_expertise_mismatch():
    bonus = _compute_domain_expertise("Climate Scientist", "Will the Fed cut rates?", "econ")
    assert bonus < 0.1, "Climate scientist shouldn't have high expertise on Fed question"


def test_domain_expertise_bounded():
    bonus = _compute_domain_expertise("Macro Economist", "Will the Fed cut rates?", "econ")
    assert 0 <= bonus <= 0.25

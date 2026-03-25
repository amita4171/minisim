"""Tests for agent archetypes and personality system."""
from src.agents.archetypes import BACKGROUNDS, PERSONALITIES, TEMP_TIERS, _make_name


def test_backgrounds_not_empty():
    assert len(BACKGROUNDS) >= 30


def test_backgrounds_have_required_fields():
    required = {"label", "detail", "econ", "political", "tech", "temp_tier"}
    for bg in BACKGROUNDS:
        assert required.issubset(bg.keys()), f"{bg['label']} missing {required - bg.keys()}"


def test_backgrounds_biases_in_range():
    for bg in BACKGROUNDS:
        for cat in ("econ", "political", "tech"):
            assert 0.0 <= bg[cat] <= 1.0, f"{bg['label']}.{cat} = {bg[cat]} out of range"


def test_temp_tiers_cover_all_backgrounds():
    for bg in BACKGROUNDS:
        assert bg["temp_tier"] in TEMP_TIERS, f"{bg['label']} has unknown tier {bg['temp_tier']}"


def test_personalities_not_empty():
    assert len(PERSONALITIES) >= 5


def test_personalities_have_required_fields():
    for p in PERSONALITIES:
        assert "label" in p
        assert "convergence_rate" in p
        assert "contrarian_factor" in p
        assert 0 <= p["convergence_rate"] <= 1
        assert 0 <= p["contrarian_factor"] <= 1


def test_make_name_returns_two_words():
    name = _make_name(0)
    assert len(name.split()) == 2


def test_make_name_deterministic():
    assert _make_name(0) == _make_name(0)
    assert _make_name(5) == _make_name(5)


def test_make_name_distinct():
    names = {_make_name(i) for i in range(20)}
    assert len(names) == 20

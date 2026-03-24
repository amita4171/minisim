"""Tests for cross-platform question matching and arbitrage detection."""
from src.cross_platform import similarity, _normalize, find_cross_listed


def test_similarity_identical():
    assert similarity("Will the Fed cut rates?", "Will the Fed cut rates?") > 0.95


def test_similarity_similar():
    s = similarity(
        "Will the Federal Reserve cut interest rates in May 2026?",
        "Will the Fed cut rates at the May 2026 FOMC meeting?",
    )
    assert s > 0.4, f"Similar questions should match, got {s}"


def test_similarity_different():
    s = similarity(
        "Will the Fed cut rates?",
        "Will Tesla deliver Optimus robots?",
    )
    assert s < 0.3, f"Different questions should not match, got {s}"


def test_normalize_strips_filler():
    assert "fed" in _normalize("Will the Fed cut rates?")
    assert "will" not in _normalize("Will the Fed cut rates?")


def test_find_cross_listed_groups_matches():
    markets = [
        {"question": "Will Trump visit Russia?", "price": 0.30, "source": "predictit",
         "ticker": "1", "volume": 100, "liquidity_weight": 2.0},
        {"question": "Will Trump visit Russia during his term?", "price": 0.40, "source": "manifold",
         "ticker": "2", "volume": 50, "liquidity_weight": 1.0},
        {"question": "Will Bitcoin hit $200K?", "price": 0.15, "source": "polymarket",
         "ticker": "3", "volume": 200, "liquidity_weight": 3.0},
    ]
    clusters = find_cross_listed(markets, similarity_threshold=0.4)
    # Should find the Trump/Russia match
    matched = [c for c in clusters if "trump" in c["question"].lower()]
    assert len(matched) >= 1
    assert matched[0]["n_platforms"] >= 2
    assert matched[0]["spread"] > 0


def test_find_cross_listed_no_same_platform():
    """Same platform markets should not be matched."""
    markets = [
        {"question": "Will X happen?", "price": 0.30, "source": "kalshi",
         "ticker": "1", "volume": 100, "liquidity_weight": 2.0},
        {"question": "Will X happen?", "price": 0.35, "source": "kalshi",
         "ticker": "2", "volume": 50, "liquidity_weight": 2.0},
    ]
    clusters = find_cross_listed(markets)
    assert len(clusters) == 0

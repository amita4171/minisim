"""
PredictIt API client — regulated US prediction market (CFTC-authorized).
No authentication required for market data.
Base URL: https://www.predictit.org/api/marketdata

PredictIt focuses on US politics: elections, policy, government.
Real-money markets with strong signal on political questions.
"""
from __future__ import annotations

import requests

BASE_URL = "https://www.predictit.org/api/marketdata"
DEFAULT_TIMEOUT = 15


def get_all_markets() -> list[dict]:
    """Fetch all active PredictIt markets in one call."""
    resp = requests.get(f"{BASE_URL}/all/", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("markets", [])


def get_market(market_id: int) -> dict:
    """Fetch a single market by ID."""
    resp = requests.get(f"{BASE_URL}/markets/{market_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_market(m: dict) -> list[dict]:
    """Normalize a PredictIt market into MiniSim format.

    PredictIt markets have multiple contracts (outcomes).
    Returns a list — one entry per binary contract.
    """
    results = []
    market_name = m.get("name", "")
    market_url = m.get("url", "")

    for contract in m.get("contracts", []):
        price = contract.get("lastTradePrice")
        if price is None:
            price = contract.get("bestBuyYesCost", 0.5)

        # Build question from market + contract name
        contract_name = contract.get("name", "")
        if contract_name and contract_name != market_name:
            question = f"{market_name}: {contract_name}?"
        else:
            question = market_name if market_name.endswith("?") else market_name + "?"

        results.append({
            "id": str(contract.get("id", "")),
            "market_id": m.get("id"),
            "question": question,
            "contract_name": contract_name,
            "url": market_url,
            "price": round(float(price), 4) if price else 0.5,
            "best_buy_yes": contract.get("bestBuyYesCost"),
            "best_buy_no": contract.get("bestBuyNoCost"),
            "best_sell_yes": contract.get("bestSellYesCost"),
            "best_sell_no": contract.get("bestSellNoCost"),
            "last_trade_price": price,
            "volume": 0,  # PredictIt doesn't expose volume per contract easily
            "is_resolved": contract.get("status", "") == "Closed",
            "resolution": None,  # PredictIt doesn't clearly expose this
            "source": "predictit",
        })

    return results


def get_active_markets(min_price: float = 0.05, max_price: float = 0.95) -> list[dict]:
    """Get all active PredictIt contracts with meaningful prices."""
    raw = get_all_markets()
    all_contracts = []
    for m in raw:
        contracts = parse_market(m)
        for c in contracts:
            if min_price <= c["price"] <= max_price and len(c["question"]) > 20:
                all_contracts.append(c)
    return all_contracts

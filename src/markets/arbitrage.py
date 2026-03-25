"""
Arbitrage detection and profit computation across prediction market platforms.

Extracted from cross_platform.py for separation of concerns.
"""
from __future__ import annotations


# Platform fee structures (as of March 2026)
PLATFORM_FEES = {
    "kalshi": {
        "maker_fee": 0.00,      # 0% maker
        "taker_fee": 0.01,      # 1% taker (1 cent per contract on $1)
        "withdrawal_fee": 0.00,
        "description": "CFTC-regulated, $1 binary contracts, 1% taker fee",
    },
    "polymarket": {
        "maker_fee": 0.00,      # 0% maker (rebates available)
        "taker_fee": 0.02,      # ~2% taker (varies by market)
        "withdrawal_fee": 0.001, # gas fees for Polygon withdrawal
        "description": "Crypto (Polygon/USDC), 0% maker / ~2% taker",
    },
    "predictit": {
        "maker_fee": 0.00,
        "taker_fee": 0.05,      # 5% on profits (not on trade)
        "profit_fee": 0.10,     # 10% fee on profit withdrawals
        "withdrawal_fee": 0.05, # 5% withdrawal fee
        "description": "CFTC-authorized, 5% profit fee + 10% withdrawal",
    },
    "manifold": {
        "maker_fee": 0.00,
        "taker_fee": 0.00,      # play money, no real fees
        "withdrawal_fee": 0.00,
        "description": "Play money (Mana), no fees, not real-money arbitrage",
    },
}


def compute_arbitrage_profit(
    buy_price: float,
    sell_price: float,
    buy_platform: str,
    sell_platform: str,
    position_size: float = 100.0,
) -> dict:
    """Compute net profit from a cross-platform arbitrage after fees.

    Strategy: Buy YES on the cheap platform, Buy NO (sell YES) on the expensive platform.
    If the question resolves YES, you win on the cheap platform and lose on the expensive.
    If the question resolves NO, the opposite.
    The arbitrage profit comes from the spread exceeding total fees.

    Args:
        buy_price: YES price on cheap platform (e.g., 0.40)
        sell_price: YES price on expensive platform (e.g., 0.55)
        buy_platform: platform to buy YES on
        sell_platform: platform to buy NO on (sell YES equivalent)
        position_size: dollars to deploy per side

    Returns:
        dict with gross_profit, total_fees, net_profit, roi, is_profitable
    """
    buy_fees = PLATFORM_FEES.get(buy_platform, {"taker_fee": 0.02})
    sell_fees = PLATFORM_FEES.get(sell_platform, {"taker_fee": 0.02})

    spread = sell_price - buy_price

    # Cost to buy YES at buy_price on cheap platform
    buy_cost = buy_price + buy_fees.get("taker_fee", 0)

    # Cost to buy NO at (1 - sell_price) on expensive platform
    no_price = 1 - sell_price
    sell_cost = no_price + sell_fees.get("taker_fee", 0)

    # Total cost per "complete set" (buy YES on A + buy NO on B)
    total_cost_per_unit = buy_cost + sell_cost

    # A complete set always pays out $1 regardless of resolution
    # (either YES wins on A or NO wins on B)
    payout_per_unit = 1.0

    # Gross profit per unit
    gross_profit_per_unit = payout_per_unit - total_cost_per_unit

    # Number of units we can buy with position_size per side
    n_units = position_size  # assuming $1 contracts

    gross_profit = gross_profit_per_unit * n_units
    total_fees = (buy_fees.get("taker_fee", 0) + sell_fees.get("taker_fee", 0)) * n_units

    # PredictIt has additional profit fee on winnings
    profit_fee = 0
    for platform, fees in [(buy_platform, buy_fees), (sell_platform, sell_fees)]:
        if "profit_fee" in fees and gross_profit > 0:
            profit_fee += fees["profit_fee"] * gross_profit * 0.5  # apply to winning side

    net_profit = gross_profit - profit_fee
    total_deployed = position_size * 2  # deployed on both sides
    roi = net_profit / total_deployed if total_deployed > 0 else 0

    return {
        "buy_platform": buy_platform,
        "sell_platform": sell_platform,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "spread": round(spread, 4),
        "buy_cost_per_unit": round(buy_cost, 4),
        "sell_cost_per_unit": round(sell_cost, 4),
        "total_cost_per_unit": round(total_cost_per_unit, 4),
        "gross_profit_per_unit": round(gross_profit_per_unit, 4),
        "gross_profit": round(gross_profit, 2),
        "trading_fees": round(total_fees, 2),
        "profit_fees": round(profit_fee, 2),
        "net_profit": round(net_profit, 2),
        "total_deployed": round(total_deployed, 2),
        "roi_pct": round(roi * 100, 2),
        "is_profitable": net_profit > 0,
        "break_even_spread": round(buy_fees.get("taker_fee", 0) + sell_fees.get("taker_fee", 0), 4),
    }


def find_profitable_arbitrage(
    markets: list[dict] | None = None,
    sources: list[str] | None = None,
    position_size: float = 100.0,
) -> list[dict]:
    """Find arbitrage opportunities that are profitable AFTER fees."""
    from src.markets.cross_platform import fetch_all_markets, find_cross_listed

    if markets is None:
        markets = fetch_all_markets(sources=sources)

    cross_listed = find_cross_listed(markets)

    profitable = []
    for cluster in cross_listed:
        prices = cluster["prices"]
        if len(prices) < 2:
            continue

        # Try all platform pairs
        platforms = list(prices.keys())
        for i, p1 in enumerate(platforms):
            for p2 in platforms[i+1:]:
                # Try both directions
                for buy_p, sell_p in [(p1, p2), (p2, p1)]:
                    if prices[buy_p] >= prices[sell_p]:
                        continue  # no spread in this direction

                    result = compute_arbitrage_profit(
                        buy_price=prices[buy_p],
                        sell_price=prices[sell_p],
                        buy_platform=buy_p,
                        sell_platform=sell_p,
                        position_size=position_size,
                    )

                    if result["is_profitable"]:
                        result["question"] = cluster["question"]
                        profitable.append(result)

    profitable.sort(key=lambda a: a["net_profit"], reverse=True)
    return profitable

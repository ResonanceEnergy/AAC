"""Fill / cost model — spread, slippage, commission."""


def apply_costs(
    price: float,
    side: str,
    spread_points: float,
    slippage_points: float,
) -> float:
    """Adjust fill price for spread and slippage.

    Parameters
    ----------
    price : raw price level.
    side : "buy" or "sell".
    spread_points : full spread in price points.
    slippage_points : estimated slippage in price points.
    """
    half_spread = spread_points / 2.0
    if side == "buy":
        return price + half_spread + slippage_points
    else:
        return price - half_spread - slippage_points

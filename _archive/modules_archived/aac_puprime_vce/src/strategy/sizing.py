"""Risk-based position sizing."""


def size_by_risk(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    point_value: float = 1.0,
) -> float:
    """Compute position size (units/lots) so that hitting the stop loses exactly
    equity * risk_pct.

    Parameters
    ----------
    equity : current account equity.
    risk_pct : fraction of equity to risk (e.g. 0.01 = 1%).
    entry : entry price.
    stop : stop-loss price.
    point_value : monetary value per 1.0 price move per 1 unit.

    Returns
    -------
    Position size in units.  Returns 0.0 if stop distance is zero.
    """
    risk_amount = equity * risk_pct
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    return risk_amount / (dist * point_value)

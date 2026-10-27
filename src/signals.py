"""Dual Momentum signals: last trading day of the month, 6-month (126 trading
days) return across 6 risky assets, top-2 positive, 50/50; otherwise BIL.
"""
from __future__ import annotations
import pandas as pd

RISK_ASSETS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD"]
SAFE_ASSET = "BIL"
LOOKBACK_DAYS = 126
TOP_N = 2


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each month present in the data."""
    s = pd.Series(index, index=index)
    return s.groupby([index.year, index.month]).max().values


def build_weights(prices: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS,
                   top_n: int = TOP_N) -> pd.DataFrame:
    """Returns a DataFrame of weights (index=rebalance date, columns=all tickers
    in prices), computed AS OF the rebalance date (signal on that day's close
    -- the position is opened from the next trading day in backtest.py,
    no look-ahead).

    BIL (safe asset) actually started trading only on 2007-05-30 -- before that
    date (and before the lookback history accumulates) we hold PURE CASH (all weights 0),
    not "100% BIL", otherwise the position opens at a NaN price and breaks NAV.
    This handles a real gap in the data, not optimisation of the ticker set.
    """
    rebal_dates = month_end_dates(prices.index)
    all_tickers = list(prices.columns)
    weights = pd.DataFrame(0.0, index=rebal_dates, columns=all_tickers)
    safe_first_valid = prices[SAFE_ASSET].first_valid_index()

    for dt in rebal_dates:
        loc = prices.index.get_loc(dt)
        safe_available = safe_first_valid is not None and dt >= safe_first_valid

        if loc < lookback_days:
            # not enough history for a 6-month return
            if safe_available:
                weights.loc[dt, SAFE_ASSET] = 1.0
            # otherwise -- pure cash (weights are already 0 by default)
            continue

        past = prices.iloc[loc - lookback_days]
        now = prices.loc[dt]
        ret6m = (now[RISK_ASSETS] / past[RISK_ASSETS] - 1.0).dropna()

        positive = ret6m[ret6m > 0].sort_values(ascending=False)
        if positive.empty:
            if safe_available:
                weights.loc[dt, SAFE_ASSET] = 1.0
            # otherwise -- pure cash
        else:
            chosen = positive.index[:top_n]
            w = 1.0 / len(chosen)  # if fewer positives than top_n -- split equally among them
            for tkr in chosen:
                weights.loc[dt, tkr] = w

    return weights

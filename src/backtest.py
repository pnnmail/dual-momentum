"""Backtest engine: positions are held between rebalance dates (weights naturally
drift with prices); on a rebalance date turnover is computed RELATIVE to the
actual (drifted) weights and costs are charged. No
intra-month rebalancing -- only on the last trading day of the month.

IMPORTANT: cash is kept as a SEPARATE explicit balance, not as "zero units" --
otherwise 0*NaN (the price of a ticker not yet trading on that day, e.g. BIL before
2007-05-30) gives NaN and silently zeroes the whole portfolio (found and
fixed on the first run: NAV went to -100%).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COST_BPS_ONE_WAY = 5.0  # spread + slippage, bp per unit of turnover


def run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame,
                  cost_bps: float = COST_BPS_ONE_WAY) -> dict:
    """target_weights: index -- rebalance dates (a subset of prices.index),
    columns -- tickers, weights in each row sum to <= 1 (the remainder is cash).
    Returns a dict with daily NAV (Series) and a turnover log per rebalance."""
    tickers = list(prices.columns)
    rebal_dates = target_weights.index
    rebal_set = set(rebal_dates)

    units = pd.Series(0.0, index=tickers)  # held only where weight > 0
    held_tickers: list[str] = []  # tickers with a non-zero position -- ONLY their prices are used in NAV
    cash = 1.0

    nav = pd.Series(index=prices.index, dtype=float)
    turnover_log = []

    for i, dt in enumerate(prices.index):
        px_today = prices.loc[dt]

        # mark-to-market on the actually held positions (NaN prices of non-participating
        # tickers never get here -- explicit held_tickers list)
        mtm = float(units[held_tickers].mul(px_today[held_tickers]).sum()) if held_tickers else 0.0
        nav.loc[dt] = cash + mtm

        if dt in rebal_set:
            tgt = target_weights.loc[dt].reindex(tickers).fillna(0.0)
            current_value = nav.loc[dt]

            drifted_weights = pd.Series(0.0, index=tickers)
            if current_value > 0 and held_tickers:
                drifted_weights[held_tickers] = units[held_tickers] * px_today[held_tickers] / current_value

            turnover = float((tgt - drifted_weights).abs().sum()) / 2.0
            cost = turnover * (cost_bps / 10000.0)
            value_after_cost = current_value * (1 - cost)

            new_held = list(tgt[tgt > 0].index)
            missing_price = [t for t in new_held if pd.isna(px_today[t])]
            if missing_price:
                print(f"  [backtest] WARNING: no price on {dt.date()} for {missing_price}, weight not invested")
                new_held = [t for t in new_held if t not in missing_price]

            units = pd.Series(0.0, index=tickers)
            for t in new_held:
                units[t] = tgt[t] * value_after_cost / px_today[t]
            # remainder after investing (usually 0 if the weights on held tickers
            # sum to exactly 1; non-zero if something in tgt could not
            # be invested because of missing_price)
            invested_value = float(units[new_held].mul(px_today[new_held]).sum()) if new_held else 0.0
            cash = value_after_cost - invested_value

            held_tickers = new_held
            nav.loc[dt] = value_after_cost
            turnover_log.append({"date": dt, "turnover": turnover, "cost": cost})

    turnover_df = pd.DataFrame(turnover_log).set_index("date") if turnover_log else pd.DataFrame(columns=["turnover", "cost"])
    return {"nav": nav, "turnover_log": turnover_df}


def buy_and_hold(prices: pd.DataFrame, ticker: str) -> pd.Series:
    px = prices[ticker].dropna()
    return px / px.iloc[0]


def fixed_weight_monthly(prices: pd.DataFrame, weights: dict, cost_bps: float = COST_BPS_ONE_WAY) -> dict:
    """A fixed-weight balance (e.g. 60/40), rebalanced on the same month-end
    dates as DM."""
    from signals import month_end_dates
    rebal_dates = month_end_dates(prices.index)
    tgt = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    for tkr, w in weights.items():
        tgt[tkr] = w
    return run_backtest(prices, tgt, cost_bps=cost_bps)

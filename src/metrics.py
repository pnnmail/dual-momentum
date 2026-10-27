"""Backtest metrics: CAGR, Vol, Sharpe, MaxDD, Calmar, Hit Rate, Turnover."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(nav: pd.Series, turnover_log: pd.DataFrame | None = None) -> dict:
    nav = nav.dropna()
    daily_ret = nav.pct_change().dropna()

    n_years = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    # Hit rate -- the share of positive MONTHS (the strategy is monthly by nature)
    monthly_nav = nav.resample("ME").last()
    monthly_ret = monthly_nav.pct_change().dropna()
    hit_rate = (monthly_ret > 0).mean()

    # Annualised turnover -- mean turnover per rebalance * 12 (monthly rebalancing)
    if turnover_log is not None and len(turnover_log) > 0:
        ann_turnover = turnover_log["turnover"].mean() * 12
    else:
        ann_turnover = np.nan

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": calmar,
        "HitRate": hit_rate,
        "Turnover_annual": ann_turnover,
    }


def format_metrics_row(name: str, m: dict) -> dict:
    return {
        "strategy": name,
        "CAGR": f"{m['CAGR']:.2%}",
        "Vol": f"{m['Vol']:.2%}",
        "Sharpe": f"{m['Sharpe']:.2f}",
        "MaxDD": f"{m['MaxDD']:.2%}",
        "Calmar": f"{m['Calmar']:.2f}",
        "HitRate": f"{m['HitRate']:.1%}",
        "Turnover_annual": f"{m['Turnover_annual']:.2f}" if not np.isnan(m["Turnover_annual"]) else "n/a",
    }

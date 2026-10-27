"""Dual Momentum: one command `python run.py` -- downloads data, computes
signals, runs the backtest of DM + 2 baselines, metrics, checks, saves
results/{equity.csv, metrics.csv, summary.txt}."""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from fetch import fetch_prices, TICKERS
from signals import build_weights, RISK_ASSETS, SAFE_ASSET, LOOKBACK_DAYS
from backtest import run_backtest, buy_and_hold, fixed_weight_monthly
from metrics import compute_metrics, format_metrics_row

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def main() -> int:
    prices = fetch_prices()
    prices = prices.dropna(subset=["SPY"])  # SPY is the reference ticker for trading days

    print("\n" + "=" * 90)
    print("MAIN BACKTEST: Dual Momentum vs SPY buy&hold vs 60/40")
    print("=" * 90)

    dm_weights = build_weights(prices, lookback_days=LOOKBACK_DAYS)
    dm_res = run_backtest(prices, dm_weights)

    spy_nav = buy_and_hold(prices, "SPY")
    balanced_res = fixed_weight_monthly(prices, {"SPY": 0.6, "TLT": 0.4})

    metrics_rows = []
    m_dm = compute_metrics(dm_res["nav"], dm_res["turnover_log"])
    m_spy = compute_metrics(spy_nav, None)
    m_bal = compute_metrics(balanced_res["nav"], balanced_res["turnover_log"])
    metrics_rows.append(format_metrics_row("DualMomentum", m_dm))
    metrics_rows.append(format_metrics_row("SPY_buyhold", m_spy))
    metrics_rows.append(format_metrics_row("60_40", m_bal))

    metrics_df = pd.DataFrame(metrics_rows)
    print(metrics_df.to_string(index=False))

    equity_df = pd.DataFrame({
        "DualMomentum": dm_res["nav"],
        "SPY": spy_nav,
        "Balanced_60_40": balanced_res["nav"],
    })
    equity_df.to_csv(os.path.join(RESULTS, "equity.csv"))
    metrics_df.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)

    # ---------------------------------------------------------------
    # Check 1: crash periods -- DM must not lose >20% in 2008/2020/2022
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 1: crash periods (2008, 2020, 2022)")
    print("=" * 90)
    crash_results = []
    for year in [2008, 2020, 2022]:
        yr_nav = dm_res["nav"][dm_res["nav"].index.year == year]
        if yr_nav.empty:
            print(f"  {year}: no data (outside the sample)")
            continue
        yr_mdd = (yr_nav / yr_nav.cummax() - 1).min()
        yr_ret = yr_nav.iloc[-1] / yr_nav.iloc[0] - 1
        status = "OK" if yr_mdd > -0.20 else "FAIL (drawdown >20%)"
        print(f"  {year}: return={yr_ret:+.1%}  intra-year drawdown={yr_mdd:.1%}  [{status}]")
        crash_results.append((year, yr_ret, yr_mdd, status))

    # ---------------------------------------------------------------
    # Check 2: split test 2007-2015 vs 2015-2026, Sharpe must not
    # differ by more than 1.5x
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 2: split test (2007-2015 vs 2015-2026)")
    print("=" * 90)
    split_date = "2015-01-01"
    nav1 = dm_res["nav"][dm_res["nav"].index < split_date]
    nav2 = dm_res["nav"][dm_res["nav"].index >= split_date]
    sh1 = compute_metrics(nav1)["Sharpe"]
    sh2 = compute_metrics(nav2)["Sharpe"]
    ratio = max(abs(sh1), abs(sh2)) / max(min(abs(sh1), abs(sh2)), 1e-9)
    split_status = "OK" if ratio <= 1.5 else "FAIL (Sharpe differs by more than 1.5x)"
    print(f"  2007-2015: Sharpe={sh1:.2f}")
    print(f"  2015-2026: Sharpe={sh2:.2f}")
    print(f"  ratio={ratio:.2f}x  [{split_status}]")

    # ---------------------------------------------------------------
    # Check 3: lookback sensitivity 3/6/9/12 mo (63/126/189/252 days)
    # -- a DIAGNOSTIC, not optimisation (see the spec constraints)
    # ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CHECK 3: lookback sensitivity (3/6/9/12 mo) -- diagnostic only")
    print("=" * 90)
    lookback_days_map = {3: 63, 6: 126, 9: 189, 12: 252}
    sens_sharpes = {}
    for months, days in lookback_days_map.items():
        w = build_weights(prices, lookback_days=days)
        res = run_backtest(prices, w)
        sh = compute_metrics(res["nav"])["Sharpe"]
        sens_sharpes[months] = sh
        print(f"  lookback={months}mo ({days}td): Sharpe={sh:.2f}")
    base_sh = sens_sharpes[6]
    max_dev = max(abs(sh - base_sh) / abs(base_sh) for sh in sens_sharpes.values() if sh != base_sh)
    sens_status = "OK (robust)" if max_dev <= 0.5 else "FAIL (Sharpe jumps >50% -- NOT robust)"
    print(f"  max Sharpe deviation from base (6mo): {max_dev:.1%}  [{sens_status}]")

    # ---------------------------------------------------------------
    # summary.txt
    # ---------------------------------------------------------------
    verdict_lines = []
    verdict_lines.append("DUAL_MOMENTUM -- SUMMARY")
    verdict_lines.append("=" * 60)
    verdict_lines.append("")
    verdict_lines.append("Metrics (2007-01-01 .. 2026-09-20, costs 5bp one-way):")
    verdict_lines.append(metrics_df.to_string(index=False))
    verdict_lines.append("")

    dm_beats_spy = m_dm["Sharpe"] > m_spy["Sharpe"]
    dm_beats_bal = m_dm["Sharpe"] > m_bal["Sharpe"]
    verdict_lines.append(f"DM Sharpe {m_dm['Sharpe']:.2f} vs SPY {m_spy['Sharpe']:.2f} vs 60/40 {m_bal['Sharpe']:.2f}")
    if dm_beats_spy and dm_beats_bal:
        verdict_lines.append("Conclusion: DM beats both baselines on Sharpe.")
    elif not dm_beats_spy:
        verdict_lines.append("Conclusion: DM LOSES to SPY buy&hold on Sharpe. Recorded as is, without tweaking.")
    else:
        verdict_lines.append("Conclusion: mixed result relative to the baselines.")
    verdict_lines.append("")

    verdict_lines.append("Check 1 (crash periods):")
    for year, ret, mdd, status in crash_results:
        verdict_lines.append(f"  {year}: return={ret:+.1%} drawdown={mdd:.1%} [{status}]")
    verdict_lines.append("")
    verdict_lines.append(f"Check 2 (split test): Sharpe 2007-2015={sh1:.2f}, 2015-2026={sh2:.2f}, "
                          f"ratio={ratio:.2f}x [{split_status}]")
    verdict_lines.append("")
    verdict_lines.append("Check 3 (lookback sensitivity, diagnostic):")
    for months, sh in sens_sharpes.items():
        verdict_lines.append(f"  {months}mo: Sharpe={sh:.2f}")
    verdict_lines.append(f"  max deviation from base: {max_dev:.1%} [{sens_status}]")
    verdict_lines.append("")
    verdict_lines.append("Spec constraints respected: lookback/TOP_N/tickers were not optimised "
                          "(sensitivity is a diagnostic, the base configuration lookback=6mo/top2 "
                          "is used in all main calculations), monthly signals only, "
                          "no ML/sentiment/macro overlay.")

    summary_text = "\n".join(verdict_lines)
    with open(os.path.join(RESULTS, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + "=" * 90)
    print(summary_text)
    print("\nFiles saved: results/equity.csv, results/metrics.csv, results/summary.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())

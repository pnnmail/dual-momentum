"""Downloads daily adjusted close for 7 ETFs from Yahoo Finance, cached in parquet."""
from __future__ import annotations
import os
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "BIL"]
START = "2007-01-01"
END = "2026-09-20"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE_PATH = os.path.join(DATA, "prices.parquet")


def fetch_prices(force: bool = False) -> pd.DataFrame:
    """Returns a DataFrame (index=date, columns=tickers) of adjusted close.
    Uses the cache if it exists and a forced refresh was not requested."""
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(CACHE_PATH) and not force:
        prices = pd.read_parquet(CACHE_PATH)
        print(f"[fetch] loaded from cache {CACHE_PATH}: {prices.shape}")
        return prices

    print(f"[fetch] downloading {TICKERS} from {START} to {END} ...")
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close[TICKERS]  # fix the column order
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()

    missing_start = close.apply(lambda s: s.first_valid_index())
    print("[fetch] first available date per ticker:")
    for t in TICKERS:
        print(f"    {t}: {missing_start[t]}")

    close.to_parquet(CACHE_PATH)
    print(f"[fetch] saved to {CACHE_PATH}: {close.shape}")
    return close


if __name__ == "__main__":
    df = fetch_prices(force=True)
    print(df.tail())

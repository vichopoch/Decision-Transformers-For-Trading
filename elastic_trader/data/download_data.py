"""Download OHLCV data for IPSA tickers."""

from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd
import yfinance as yf

from ..utils.data_loader import get_ipsa_tickers


DEFAULT_START = dt.date(2010, 1, 1)


def download_ohlcv(
    tickers: Iterable[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data using ``yfinance``.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols. If ``None`` fetches IPSA tickers.
    start:
        Start date. Defaults to ``2010-01-01``.
    end:
        End date. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Multi-indexed dataframe (date, ticker).
    """
    tickers = list(tickers) if tickers else get_ipsa_tickers()
    start = start or DEFAULT_START
    end = end or dt.date.today()
    data = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns columns like ('Adj Close', 'AAPL'), we want tidy data
    frames = []
    for tkr in tickers:
        df = data[tkr].copy()
        df["Ticker"] = tkr
        frames.append(df)
    combined = pd.concat(frames)
    combined.index.name = "Date"
    return combined.reset_index()

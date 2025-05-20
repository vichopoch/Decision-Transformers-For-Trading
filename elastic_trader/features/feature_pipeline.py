"""Feature engineering for IPSA data."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD


INDICATORS = ["rsi", "macd", "macd_signal"]


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators to the data."""
    df = data.copy()
    close = df["Close"]
    # EVOLVE-START features
    rsi = RSIIndicator(close).rsi()
    macd = MACD(close)
    df["rsi"] = rsi
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    # EVOLVE-END features
    return df

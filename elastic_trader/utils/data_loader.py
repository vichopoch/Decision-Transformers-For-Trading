"""Utility functions for data loading."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def get_ipsa_tickers() -> List[str]:
    """Return the list of S&P IPSA tickers.

    The function attempts to fetch the components from Wikipedia. If the
    network request fails, it falls back to a static list.
    """
    url = "https://en.wikipedia.org/wiki/S%26P_IPSA"
    try:
        tables = pd.read_html(requests.get(url, timeout=10).text)
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].dropna().tolist()
                if tickers:
                    return [t.strip() for t in tickers]
    except Exception as err:  # pragma: no cover - network failure
        logger.warning("Failed to fetch tickers online: %s", err)

    # Fallback list (as of 2024)
    return [
        "ANDERC", "BSANTANDER", "CAP", "CHILE", "CMPC", "COPEC", "ENELCHILE",
        "FALABELLA", "IAM", "ITAUCORP", "PARAUCO", "QUINENCO", "SECURITY",
        "SALFACORP", "SQM-B", "VAPORES",
    ]

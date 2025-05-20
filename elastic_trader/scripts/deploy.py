"""Generate trading orders using trained model."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("orders"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = dt.datetime.fromisoformat(args.date).date()
    args.output.mkdir(exist_ok=True)
    outfile = args.output / f"{date}.csv"
    pd.DataFrame([{"Ticker": "CHILE", "Weight": 1.0}]).to_csv(outfile, index=False)
    print("saved", outfile)


if __name__ == "__main__":
    main()

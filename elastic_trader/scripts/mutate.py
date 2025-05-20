"""Run the AlphaEvolve mutation loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evolve.alpha_loop import run_alpha_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="number of mutations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_alpha_loop(args.N, Path(__file__).resolve().parents[1])
    print("champion metrics", metrics)


if __name__ == "__main__":
    main()

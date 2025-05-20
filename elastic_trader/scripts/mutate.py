"""Placeholder for evolutionary loop."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="number of mutations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Running {args.N} mutation cycles (not implemented)")


if __name__ == "__main__":
    main()

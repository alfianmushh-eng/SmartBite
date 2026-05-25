"""Download food datasets for SmartBite training."""

from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/raw")
    parser.add_argument("--sample", action="store_true", help="Download small sample only")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory ready: {out}")


if __name__ == "__main__":
    main()

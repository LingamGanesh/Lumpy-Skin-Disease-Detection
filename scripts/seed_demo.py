"""Seed LumpyCare with clearly labelled synthetic demo scans."""
from __future__ import annotations

import argparse
from lumpycare.database import init_db, seed_demo_data

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="Number of synthetic scans to create")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    init_db()
    total = seed_demo_data(count=args.count)
    print(f"Database ready. Synthetic demo scans available: {total}")
    print("These records are simulated and must not be interpreted as real farm cases.")

if __name__ == "__main__":
    main()

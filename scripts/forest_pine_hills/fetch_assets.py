#!/usr/bin/env python3
"""Fetch only the CC0 HDR image used by FOREST_PINE_HILLS, with API MD5 checks."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forest_snow.fetch_assets import fetch_assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--res", choices=("1k", "2k", "4k"), default="2k")
    args = parser.parse_args()
    fetch_assets(args.dest, args.res, assets=(("kloppenheim_06_puresky", 0),
                                            ("syferfontein_18d_clear_puresky", 0)))

#!/usr/bin/env python3
"""Backwards-compatible entry point. Prefer the `magi` console script instead."""
from magi_core.cli import main

if __name__ == "__main__":
    main()

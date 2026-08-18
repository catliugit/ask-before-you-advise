#!/usr/bin/env python3
"""CLI shim for slice.kappa_gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slice.kappa_gate import run_cli


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

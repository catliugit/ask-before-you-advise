#!/usr/bin/env python3
"""CLI shim for slice.freeze; the week-four freeze run is intentionally deferred."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slice.freeze import main


if __name__ == "__main__":
    main()

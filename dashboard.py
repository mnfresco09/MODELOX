#!/usr/bin/env python3
"""
MODELOX · AI Parameter Intelligence Dashboard
Usage:  python dashboard.py path/to/file.db
        python dashboard.py                   (interactive prompt)
URL:    http://127.0.0.1:8050
"""
from __future__ import annotations

import os
import sys


def _resolve_db() -> str:
    """Resolve the database path from CLI args or interactive prompt."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        candidate = args[0].strip().strip("'\"")
        if os.path.isfile(candidate) and candidate.endswith(".db"):
            return os.path.abspath(candidate)

    # Interactive prompt
    w = 56
    print(f"\n  ╔{'═'*w}╗")
    print(f"  ║{'MODELOX  AI Parameter Intelligence':^{w}}║")
    print(f"  ╚{'═'*w}╝\n")
    print("  Drag a .db file here and press Enter")
    print("  (or Ctrl+C to cancel)\n")
    try:
        raw = input("  › ").strip().strip("'\"")
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.\n")
        sys.exit(0)

    if os.path.isfile(raw) and raw.endswith(".db"):
        return os.path.abspath(raw)

    print(f"\n  Error: '{raw}' is not a valid .db file.\n")
    sys.exit(1)


if __name__ == "__main__":
    db_path = _resolve_db()

    # Optional --port argument
    port = 8050
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    from visual.optuna_dash.app import run
    run(db_path, port=port, debug=False)

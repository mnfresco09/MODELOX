"""Compat wrapper.

Allows running `python git_push.py` from repo root while keeping the real
implementation in `x/git_push.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    target = repo_root / "x" / "git_push.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

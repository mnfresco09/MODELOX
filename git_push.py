from __future__ import annotations

# Wrapper para mantener compatibilidad con: `python git_push.py`
# La implementación vive en x/git_push.py

from pathlib import Path
import runpy


def main() -> None:
    target = Path(__file__).resolve().parent / "x" / "git_push.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script para actualizar cambios en GitHub automáticamente.
FUERZA la subida de TODO, ignorando incluso el .gitignore.
"""

import subprocess
from datetime import datetime
from pathlib import Path
import sys
import argparse

def run_command(cmd: list[str], check: bool = True) -> tuple[bool, str]:
    """Ejecuta un comando y retorna (éxito, output)."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=str(repo_root),
        )
        return True, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr

def _default_commit_message() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Full System Backup: {timestamp}"

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("message", nargs="*", help="Mensaje de commit")
    parser.add_argument("--force-push", action="store_true", help="Force push a origin/main")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    print("🚀 Forzando subida total del sistema (incluyendo 'data/')...\n")

    # 1. Sincronizar
    run_command(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)

    # 2. Agregar TODO con Fuerza (-f)
    # Esto ignora las reglas del .gitignore para asegurar que 'data' suba.
    print("➕ Agregando archivos (ignorando .gitignore si es necesario)...")
    success, output = run_command(["git", "add", "-A", "-f"], check=False)
    
    if not success:
        print(f"❌ Error al agregar archivos:\n{output}")
        return 1

    # 3. Commit
    commit_msg = " ".join(args.message).strip() if args.message else _default_commit_message()
    print(f"💬 Commit: '{commit_msg}'")
    success, output = run_command(["git", "commit", "-m", commit_msg], check=False)
    
    if not success and "nothing to commit" not in output.lower():
        print(f"❌ Error en commit:\n{output}")
        return 1

    # 4. Push
    print("⬆️  Subiendo a GitHub...")
    push_cmd = ["git", "push", "origin", "main"]
    if args.force_push:
        push_cmd.insert(2, "-f")

    success, output = run_command(push_cmd, check=False)
    if not success:
        print(f"❌ Error al subir:\n{output}")
        return 1

    print("\n✅ ¡Todo subido correctamente, incluyendo carpetas ocultas o ignoradas!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
MODELOX Git Push - Script robusto para subir cambios a GitHub.
Ejecutar desde cualquier ubicación: python github/git_push.py "mensaje"
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path
import sys
import argparse
from typing import Optional, List

# Timeout en segundos para comandos git
GIT_TIMEOUT = 30

# Obtener la raíz del repo (un nivel arriba de github/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: List[str], check: bool = True, timeout: int = GIT_TIMEOUT) -> tuple:
    """Ejecuta un comando y retorna (éxito, output)."""
    cmd_str = ' '.join(cmd)
    print(f"   → {cmd_str}")

    # Create a clean environment for git commands
    clean_env = os.environ.copy()
    if "LD_LIBRARY_PATH" in clean_env:
        del clean_env["LD_LIBRARY_PATH"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=str(REPO_ROOT),  # Siempre ejecutar desde la raíz del repo
            timeout=timeout,
            env=clean_env
        )
        output = result.stdout + result.stderr
        if output.strip():
            lines = output.strip().split('\n')
            if len(lines) > 5:
                print(f"     [{len(lines)} líneas de output]")
            else:
                for line in lines:
                    print(f"     {line}")
        return True, output

    except subprocess.TimeoutExpired:
        print(f"   ⏱️  TIMEOUT después de {timeout}s - comando cancelado")
        return False, f"Timeout después de {timeout} segundos"

    except subprocess.CalledProcessError as e:
        output = (e.stdout or '') + (e.stderr or '')
        print(f"   ✗ Error: {output[:200]}")
        return False, output

def _default_commit_message() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Full update: {timestamp}"

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subir cambios a GitHub de forma segura",
        add_help=True
    )
    parser.add_argument("message", nargs="*", help="Mensaje de commit (opcional)")
    parser.add_argument("--force", "-f", action="store_true", help="Force push")
    parser.add_argument("--no-pull", action="store_true", help="No hacer pull antes")
    parser.add_argument("--amend", action="store_true", help="Enmendar último commit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    print("\n" + "=" * 50)
    print("🚀 MODELOX GIT PUSH")
    print("=" * 50 + "\n")

    # 0) Sincronización previa (sin rebase para evitar problemas)
    if not args.no_pull:
        print("📥 Paso 1: Fetch desde origin...")
        success, output = run_command(["git", "fetch", "origin"], check=False)
        if not success:
            print("⚠️  No se pudo hacer fetch (continuando...)")

    # 1. Verificar estado
    print("\n📊 Paso 2: Verificando estado local...")
    success, output = run_command(["git", "status", "--short"])

    # 2. Agregar cambios
    print("\n➕ Paso 3: Agregando archivos...")
    run_command(["git", "add", "-A"], check=False)

    # 3. Commit
    commit_msg = " ".join(args.message).strip() if args.message else _default_commit_message()

    print("\n💬 Paso 4: Commit...")
    if args.amend:
        success, output = run_command(["git", "commit", "--amend", "-m", commit_msg], check=False)
    else:
        success, output = run_command(["git", "commit", "-m", commit_msg], check=False)

    if not success and "nothing to commit" not in output.lower():
        print("❌ Error en commit")
        return 1

    # 4. Push (con timeout más largo)
    print("\n⬆️  Paso 5: Push a GitHub...")
    push_cmd = ["git", "push", "origin", "main"]
    if args.force:
        push_cmd.insert(2, "--force")

    success, output = run_command(push_cmd, check=False, timeout=120)

    if not success:
        if "large file" in output.lower() or "exceeds" in output.lower():
            print("\n❌ ERROR: Archivos demasiado grandes detectados!")
            print("   Revisa que .gitignore excluya: .venv/, data/, *.csv, *.parquet")
            print("   Puede que necesites limpiar el historial de git.")
        else:
            print("❌ Error al subir")
        return 1

    print("\n" + "=" * 50)
    print("✅ ¡PUSH COMPLETADO CON ÉXITO!")
    print("=" * 50 + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

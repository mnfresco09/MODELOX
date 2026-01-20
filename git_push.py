#!/usr/bin/env python3
"""
Script para actualizar cambios en GitHub automáticamente.
Incluye TODOS los archivos y carpetas del repositorio.
"""

import subprocess
from datetime import datetime
from pathlib import Path
import sys
import argparse
from typing import Optional, List  # <--- CORRECCIÓN 1: Importamos herramientas de compatibilidad

def run_command(cmd: List[str], check: bool = True) -> tuple:
    """Ejecuta un comando y retorna (éxito, output)."""
    try:
        # CORRECCIÓN: Apunta a la carpeta actual donde está este script (root del repo)
        repo_root = Path(__file__).resolve().parent
        
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
    return f"Full update: {timestamp}"

def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("message", nargs="*", help="Mensaje de commit (opcional)")
    parser.add_argument("--force", action="store_true", help="Force push a origin/main")
    parser.add_argument("--no-pull", action="store_true", help="No hacer pull antes de push")
    parser.add_argument("--amend", action="store_true", help="Enmienda el último commit")
    return parser.parse_args(argv)

# CORRECCIÓN 2: Cambiamos "list[str] | None" por "Optional[List[str]]"
def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    print("🚀 Iniciando actualización total del sistema...\n")

    # 0) Sincronización previa
    if not args.no_pull:
        print("🔄 Sincronizando con GitHub...")
        success, output = run_command(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
        if "CONFLICT" in output:
            print("❌ Conflictos detectados. Resuélvelos manualmente.")
            return 1
        print("✅ Sincronización completada")

    # 1. Verificar estado
    print("📊 Verificando cambios...")
    success, output = run_command(["git", "status", "--short"])
    
    if not output.strip():
        print("✅ No hay cambios nuevos detectados por git status.")
        # Quitamos el return para forzar el intento de subida por si acaso
        # return 0 

    # 2. Agregar TODOS los cambios
    print("\n➕ Agregando absolutamente todos los archivos y carpetas...")
    run_command(["git", "reset"], check=False) # Limpia el stage actual
    success, output = run_command(["git", "add", "-A"], check=False) 
    
    if not success:
        print(f"❌ Error al agregar cambios:\n{output}")
        return 1
    print("✅ Todos los archivos agregados al stage")

    # 3. Commit
    commit_msg = " ".join(args.message).strip() if args.message else _default_commit_message()

    if args.amend:
        print(f"\n💬 Haciendo Amend: '{commit_msg}'")
        success, output = run_command(["git", "commit", "--amend", "-m", commit_msg], check=False)
    else:
        print(f"\n💬 Creando Commit: '{commit_msg}'")
        success, output = run_command(["git", "commit", "-m", commit_msg], check=False)
    
    if not success:
        if "nothing to commit" in output.lower():
            print("✅ Nada nuevo que commitear.")
        else:
            print(f"❌ Error en commit:\n{output}")
            return 1

    # 4. Push
    print("\n⬆️  Subiendo a GitHub...")
    push_cmd = ["git", "push", "origin", "main"]
    if args.force:
        push_cmd.insert(2, "-f")

    success, output = run_command(push_cmd, check=False)
    if not success:
        print(f"❌ Error al subir:\n{output}")
        return 1

    print("✅ ¡Push completado con éxito!")
    print("🎉 El repositorio está totalmente actualizado.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
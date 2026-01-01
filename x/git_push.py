#!/usr/bin/env python3
"""
Script para actualizar cambios en GitHub automáticamente.
Uso: python x/git_push.py [mensaje_opcional]
"""


import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> tuple[bool, str]:
    """Ejecuta un comando y retorna (éxito, output)."""
    try:
        # Descubre el root del repo (carpeta que contiene 'x')
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


def main():
    print("🚀 Actualizando repositorio en GitHub...\n")

    # 1. Verificar estado
    print("📊 Verificando cambios...")
    success, output = run_command(["git", "status", "--short"])
    if not success:
        print("❌ Error al verificar estado del repositorio")
        return 1

    if not output.strip():
        print("✅ No hay cambios para commitear")
        return 0

    print(f"📝 Cambios detectados:\n{output}")

    # 2. Agregar todos los cambios (Git respeta .gitignore, data/ no se añadirá)
    print("\n➕ Agregando cambios (data/ excluida por .gitignore)...")

    # Limpia staging previo (por si acaso)
    run_command(["git", "reset"])

    # Stage all: incluye nuevos/modificados/eliminados, pero NO añade ignorados.
    success, output = run_command(["git", "add", "-A"])
    if not success:
        print(f"❌ Error al agregar cambios:\n{output}")
        return 1

    # Si por alguna razón data/ estuviera trackeada, intenta des-stagearla.
    # (no hacemos fallar el script si no existe o no hay nada staged)
    run_command(["git", "restore", "--staged", "--", "data"], check=False)
    run_command(["git", "reset", "--", "data"], check=False)

    print("✅ Cambios agregados")

    # 3. Commit con mensaje personalizado o automático
    if len(sys.argv) > 1:
        commit_msg = " ".join(sys.argv[1:])
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Auto-update: {timestamp}"

    print(f"\n💬 Commit: '{commit_msg}'")
    success, output = run_command(["git", "commit", "-m", commit_msg])
    if not success:
        if "nothing to commit" in output.lower():
            print("✅ No hay cambios nuevos para commitear")
        else:
            print(f"❌ Error en commit:\n{output}")
            return 1
    else:
        print("✅ Commit realizado")

    # 4. Push a origin main
    print("\n⬆️  Subiendo a GitHub (origin main)...")
    success, output = run_command(["git", "push", "origin", "main"])
    if not success:
        print(f"❌ Error al hacer push:\n{output}")
        print("\n💡 Intenta con force push: python x/git_push.py --force")
        return 1

    print("✅ Push completado exitosamente!")
    print(f"\n{output}")
    print("🎉 Repositorio actualizado en GitHub!")
    return 0


if __name__ == "__main__":
    # Manejo de force push si se solicita
    if "--force" in sys.argv:
        print("⚠️  FORCE PUSH activado")
        sys.argv.remove("--force")
        run_command(["git", "push", "-f", "origin", "main"])
        print("✅ Force push completado")
        sys.exit(0)

    sys.exit(main())

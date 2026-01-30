#!/usr/bin/env python3
"""
================================================================================
🔄 MODELOX Git Pull - Actualizar sistema local con cambios de GitHub
================================================================================

Script para sincronizar tu sistema local con los últimos cambios del repositorio
remoto en GitHub. Útil cuando has hecho cambios desde otro dispositivo o cuando
otro colaborador ha subido código.

USO:
    python github/actualizar.py           # Pull normal
    python github/actualizar.py --force   # Descartar cambios locales y actualizar
    python github/actualizar.py --stash   # Guardar cambios locales, actualizar, restaurar
    python github/actualizar.py --check   # Solo verificar si hay actualizaciones

================================================================================
"""

import subprocess
import os
from datetime import datetime
from pathlib import Path
import sys
import argparse
from typing import Optional, List, Tuple

# Colores ANSI
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# Timeout en segundos para comandos git
GIT_TIMEOUT = 60

# Obtener la raíz del repo (un nivel arriba de github/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: List[str], check: bool = True, timeout: int = GIT_TIMEOUT, silent: bool = False) -> Tuple[bool, str]:
    """Ejecuta un comando y retorna (éxito, output)."""
    cmd_str = ' '.join(cmd)
    if not silent:
        print(f"   {Colors.DIM}→ {cmd_str}{Colors.RESET}")

    # Crear entorno limpio para git
    clean_env = os.environ.copy()
    if "LD_LIBRARY_PATH" in clean_env:
        del clean_env["LD_LIBRARY_PATH"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=str(REPO_ROOT),
            timeout=timeout,
            env=clean_env
        )
        output = result.stdout + result.stderr
        if output.strip() and not silent:
            lines = output.strip().split('\n')
            for line in lines[:10]:
                print(f"     {Colors.DIM}{line}{Colors.RESET}")
            if len(lines) > 10:
                print(f"     {Colors.DIM}... ({len(lines) - 10} líneas más){Colors.RESET}")
        return True, output

    except subprocess.TimeoutExpired:
        print(f"   {Colors.YELLOW}⏱️  TIMEOUT después de {timeout}s{Colors.RESET}")
        return False, f"Timeout después de {timeout} segundos"

    except subprocess.CalledProcessError as e:
        output = (e.stdout or '') + (e.stderr or '')
        if not silent:
            print(f"   {Colors.RED}✗ Error: {output[:200]}{Colors.RESET}")
        return False, output


def get_current_branch() -> str:
    """Obtiene el nombre de la rama actual."""
    success, output = run_command(["git", "branch", "--show-current"], silent=True)
    return output.strip() if success else "main"


def has_local_changes() -> bool:
    """Verifica si hay cambios locales sin commitear."""
    success, output = run_command(["git", "status", "--porcelain"], silent=True)
    return bool(output.strip())


def get_commits_behind() -> int:
    """Retorna cuántos commits está detrás del remoto."""
    run_command(["git", "fetch", "origin"], check=False, silent=True)
    branch = get_current_branch()
    success, output = run_command(
        ["git", "rev-list", "--count", f"HEAD..origin/{branch}"],
        check=False,
        silent=True
    )
    try:
        return int(output.strip()) if success else 0
    except ValueError:
        return 0


def get_commits_ahead() -> int:
    """Retorna cuántos commits está adelante del remoto."""
    branch = get_current_branch()
    success, output = run_command(
        ["git", "rev-list", "--count", f"origin/{branch}..HEAD"],
        check=False,
        silent=True
    )
    try:
        return int(output.strip()) if success else 0
    except ValueError:
        return 0


def show_banner():
    """Muestra el banner inicial."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║            🔄 MODELOX - ACTUALIZAR DESDE GITHUB                  ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Actualizar sistema local con cambios de GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python github/actualizar.py           # Pull normal
  python github/actualizar.py --force   # Descartar cambios locales y actualizar
  python github/actualizar.py --stash   # Guardar cambios, actualizar, restaurar
  python github/actualizar.py --check   # Solo verificar actualizaciones
        """
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Descartar cambios locales y forzar actualización")
    parser.add_argument("--stash", "-s", action="store_true",
                        help="Guardar cambios locales (stash), actualizar, y restaurarlos")
    parser.add_argument("--check", "-c", action="store_true",
                        help="Solo verificar si hay actualizaciones disponibles")
    parser.add_argument("--rebase", "-r", action="store_true",
                        help="Usar rebase en lugar de merge")
    
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    
    show_banner()
    
    branch = get_current_branch()
    print(f"{Colors.BLUE}📍 Rama actual: {Colors.BOLD}{branch}{Colors.RESET}\n")
    
    # =========================================================================
    # PASO 1: Fetch para obtener info del remoto
    # =========================================================================
    print(f"{Colors.YELLOW}📡 Paso 1: Conectando con GitHub...{Colors.RESET}")
    success, _ = run_command(["git", "fetch", "origin"], check=False)
    if not success:
        print(f"\n{Colors.RED}❌ No se pudo conectar con GitHub{Colors.RESET}")
        print(f"{Colors.DIM}   Verifica tu conexión a internet y credenciales{Colors.RESET}")
        return 1
    
    # =========================================================================
    # PASO 2: Verificar estado
    # =========================================================================
    print(f"\n{Colors.YELLOW}📊 Paso 2: Analizando estado...{Colors.RESET}")
    
    behind = get_commits_behind()
    ahead = get_commits_ahead()
    has_changes = has_local_changes()
    
    print(f"\n   {Colors.CYAN}Estado:{Colors.RESET}")
    print(f"   • Commits detrás de GitHub: {Colors.GREEN if behind == 0 else Colors.YELLOW}{behind}{Colors.RESET}")
    print(f"   • Commits adelante de GitHub: {Colors.GREEN if ahead == 0 else Colors.YELLOW}{ahead}{Colors.RESET}")
    print(f"   • Cambios locales sin commit: {Colors.RED if has_changes else Colors.GREEN}{'Sí' if has_changes else 'No'}{Colors.RESET}")
    
    # =========================================================================
    # MODO CHECK: Solo mostrar información
    # =========================================================================
    if args.check:
        print(f"\n{Colors.CYAN}{'─' * 50}{Colors.RESET}")
        if behind == 0:
            print(f"{Colors.GREEN}✅ Tu sistema está actualizado{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}📥 Hay {behind} commit(s) nuevos disponibles{Colors.RESET}")
            print(f"{Colors.DIM}   Ejecuta: python github/actualizar.py{Colors.RESET}")
        return 0
    
    # =========================================================================
    # Verificar si hay algo que actualizar
    # =========================================================================
    if behind == 0:
        print(f"\n{Colors.GREEN}✅ Tu sistema ya está actualizado con GitHub{Colors.RESET}")
        return 0
    
    # =========================================================================
    # PASO 3: Manejar cambios locales
    # =========================================================================
    stashed = False
    
    if has_changes:
        print(f"\n{Colors.YELLOW}⚠️  Tienes cambios locales sin commitear{Colors.RESET}")
        
        if args.force:
            print(f"\n{Colors.RED}🗑️  Descartando cambios locales (--force)...{Colors.RESET}")
            run_command(["git", "checkout", "--", "."], check=False)
            run_command(["git", "clean", "-fd"], check=False)
            
        elif args.stash:
            print(f"\n{Colors.BLUE}📦 Guardando cambios locales (stash)...{Colors.RESET}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            success, _ = run_command(["git", "stash", "push", "-m", f"auto_stash_{timestamp}"])
            if success:
                stashed = True
                print(f"   {Colors.GREEN}✓ Cambios guardados temporalmente{Colors.RESET}")
            else:
                print(f"   {Colors.RED}✗ No se pudieron guardar los cambios{Colors.RESET}")
                return 1
        else:
            print(f"\n{Colors.RED}❌ No se puede actualizar con cambios locales pendientes{Colors.RESET}")
            print(f"\n{Colors.DIM}Opciones:{Colors.RESET}")
            print(f"   {Colors.DIM}1. Commitea tus cambios: python github/git_push.py \"mensaje\"{Colors.RESET}")
            print(f"   {Colors.DIM}2. Guarda temporalmente: python github/actualizar.py --stash{Colors.RESET}")
            print(f"   {Colors.DIM}3. Descarta cambios:     python github/actualizar.py --force{Colors.RESET}")
            return 1
    
    # =========================================================================
    # PASO 4: Actualizar (Pull)
    # =========================================================================
    print(f"\n{Colors.YELLOW}📥 Paso 3: Descargando {behind} commit(s) de GitHub...{Colors.RESET}")
    
    if args.rebase:
        success, output = run_command(["git", "pull", "--rebase", "origin", branch])
    else:
        success, output = run_command(["git", "pull", "origin", branch])
    
    if not success:
        print(f"\n{Colors.RED}❌ Error durante la actualización{Colors.RESET}")
        
        # Verificar si hay conflictos
        if "CONFLICT" in output or "conflict" in output:
            print(f"\n{Colors.YELLOW}⚠️  Hay conflictos que resolver manualmente:{Colors.RESET}")
            run_command(["git", "status"], check=False)
            print(f"\n{Colors.DIM}Después de resolver conflictos, ejecuta:{Colors.RESET}")
            print(f"   {Colors.DIM}git add . && git commit{Colors.RESET}")
        
        # Restaurar stash si lo hicimos
        if stashed:
            print(f"\n{Colors.BLUE}📦 Restaurando cambios guardados...{Colors.RESET}")
            run_command(["git", "stash", "pop"], check=False)
        
        return 1
    
    # =========================================================================
    # PASO 5: Restaurar stash si aplica
    # =========================================================================
    if stashed:
        print(f"\n{Colors.BLUE}📦 Restaurando tus cambios locales...{Colors.RESET}")
        success, output = run_command(["git", "stash", "pop"], check=False)
        if not success and "CONFLICT" in output:
            print(f"\n{Colors.YELLOW}⚠️  Conflictos al restaurar tus cambios{Colors.RESET}")
            print(f"{Colors.DIM}   Resuelve los conflictos manualmente{Colors.RESET}")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print(f"""
{Colors.GREEN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                    ✅ ACTUALIZACIÓN COMPLETADA                   ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}

   • Rama:                 {Colors.CYAN}{branch}{Colors.RESET}
   • Commits descargados:  {Colors.GREEN}{behind}{Colors.RESET}
   • Estado:               {Colors.GREEN}Sincronizado con GitHub{Colors.RESET}
""")
    
    # Mostrar últimos commits descargados
    print(f"{Colors.DIM}   Últimos cambios:{Colors.RESET}")
    run_command(["git", "log", "--oneline", f"-{min(behind, 5)}"], check=False)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

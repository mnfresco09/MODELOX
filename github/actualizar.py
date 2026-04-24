#!/usr/bin/env python3
"""
MODELOX - restaurar el sistema local a una version remota elegida.

Comportamiento por defecto:
1. Hace fetch del remoto.
2. Lista las 10 ultimas subidas del remoto con fecha y hora.
3. Pide elegir una version por numero.
4. Deja la rama local exactamente igual que ese commit remoto.
5. Elimina archivos no versionados no ignorados.
6. No toca lo ignorado por .gitignore.

Uso:
    python github/actualizar.py
    python github/actualizar.py --check
    python github/actualizar.py --select 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


GIT_TIMEOUT = 90
REPO_ROOT = Path(__file__).resolve().parent.parent


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


@dataclass(frozen=True)
class RemoteCommit:
    sha: str
    short_sha: str
    date_str: str
    subject: str


def run_command(
    cmd: List[str],
    *,
    timeout: int = GIT_TIMEOUT,
    silent: bool = False,
) -> Tuple[bool, str]:
    """Ejecuta un comando y devuelve (success, output)."""
    if not silent:
        print(f"   {Colors.DIM}-> {' '.join(cmd)}{Colors.RESET}")

    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            timeout=timeout,
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timeout despues de {timeout} segundos"

    output = ((result.stdout or "") + (result.stderr or "")).strip()
    success = result.returncode == 0

    if output and not silent:
        lines = output.splitlines()
        for line in lines[:10]:
            print(f"     {Colors.DIM}{line}{Colors.RESET}")
        if len(lines) > 10:
            print(f"     {Colors.DIM}... ({len(lines) - 10} lineas mas){Colors.RESET}")

    return success, output


def show_banner() -> None:
    print(
        f"\n{Colors.CYAN}{Colors.BOLD}"
        "==============================================================\n"
        "MODELOX - ACTUALIZAR DESDE GITHUB\n"
        "=============================================================="
        f"{Colors.RESET}\n"
    )


def ensure_git_repo() -> bool:
    success, _ = run_command(["git", "rev-parse", "--show-toplevel"], silent=True)
    return success


def get_current_branch() -> Optional[str]:
    success, output = run_command(["git", "branch", "--show-current"], silent=True)
    branch = output.strip() if success else ""
    return branch or None


def get_upstream_ref(branch: str, remote: str) -> str:
    success, output = run_command(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        silent=True,
    )
    upstream = output.strip() if success else ""
    if upstream:
        return upstream
    return f"{remote}/{branch}"


def ref_exists(ref: str) -> bool:
    success, _ = run_command(["git", "rev-parse", "--verify", ref], silent=True)
    return success


def count_commits(rev_range: str) -> int:
    success, output = run_command(["git", "rev-list", "--count", rev_range], silent=True)
    if not success:
        return 0
    try:
        return int(output.strip())
    except ValueError:
        return 0


def has_local_changes() -> bool:
    success, output = run_command(["git", "status", "--porcelain"], silent=True)
    return success and bool(output.strip())


def ignored_items_count() -> int:
    success, output = run_command(["git", "status", "--ignored", "--short"], silent=True)
    if not success or not output:
        return 0
    return sum(1 for line in output.splitlines() if line.startswith("!! "))


def short_commit(ref: str) -> str:
    success, output = run_command(["git", "rev-parse", "--short", ref], silent=True)
    return output.strip() if success else "desconocido"


def get_recent_remote_commits(upstream: str, limit: int = 10) -> List[RemoteCommit]:
    success, output = run_command(
        [
            "git",
            "log",
            upstream,
            f"-{limit}",
            "--date=format:%Y-%m-%d %H:%M:%S",
            "--pretty=format:%H%x1f%h%x1f%cd%x1f%s",
        ],
        silent=True,
    )
    if not success or not output:
        return []

    commits: List[RemoteCommit] = []
    for line in output.splitlines():
        parts = line.split("\x1f", 3)
        if len(parts) != 4:
            continue
        commits.append(
            RemoteCommit(
                sha=parts[0].strip(),
                short_sha=parts[1].strip(),
                date_str=parts[2].strip(),
                subject=parts[3].strip(),
            )
        )
    return commits


def print_recent_remote_commits(commits: List[RemoteCommit]) -> None:
    print(f"\n{Colors.CYAN}Ultimas versiones disponibles:{Colors.RESET}")
    for idx, commit in enumerate(commits, start=1):
        print(
            f"   {Colors.BOLD}{idx:2d}{Colors.RESET}. "
            f"{commit.date_str}  "
            f"{Colors.YELLOW}{commit.short_sha}{Colors.RESET}  "
            f"{commit.subject}"
        )


def choose_commit_index(commits: List[RemoteCommit], preset: Optional[int] = None) -> Optional[int]:
    if not commits:
        return None

    if preset is not None:
        if 1 <= preset <= len(commits):
            return preset - 1
        print(
            f"{Colors.RED}ERROR: --select debe estar entre 1 y {len(commits)}.{Colors.RESET}"
        )
        return None

    while True:
        try:
            raw = input(f"\nSelecciona la version a restaurar (1-{len(commits)}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}Operacion cancelada por el usuario.{Colors.RESET}")
            return None

        if not raw:
            print(f"{Colors.YELLOW}Debes introducir un numero.{Colors.RESET}")
            continue

        try:
            value = int(raw)
        except ValueError:
            print(f"{Colors.YELLOW}Entrada no valida. Usa un numero entero.{Colors.RESET}")
            continue

        if 1 <= value <= len(commits):
            return value - 1

        print(f"{Colors.YELLOW}El numero debe estar entre 1 y {len(commits)}.{Colors.RESET}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restaurar el sistema local a una de las ultimas versiones del remoto."
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Solo mostrar estado y listado de versiones; no aplicar cambios.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remoto a usar (default: origin).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Rama a sincronizar. Por defecto usa la rama actual.",
    )
    parser.add_argument(
        "--select",
        type=int,
        default=None,
        help="Selecciona directamente una version del listado (1 = la mas reciente).",
    )

    # Compatibilidad con el script anterior.
    parser.add_argument("--hard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stash", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rebase", action="store_true", help=argparse.SUPPRESS)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    show_banner()

    if not ensure_git_repo():
        print(f"{Colors.RED}ERROR: {REPO_ROOT} no es un repositorio Git valido.{Colors.RESET}")
        return 1

    current_branch = get_current_branch()
    branch = args.branch or current_branch
    if not branch:
        print(
            f"{Colors.RED}ERROR: no se pudo detectar la rama actual. "
            f"Haz checkout a una rama y vuelve a ejecutar.{Colors.RESET}"
        )
        return 1

    if args.branch and current_branch and args.branch != current_branch:
        print(
            f"{Colors.RED}ERROR: --branch solo puede usarse sobre la rama actualmente "
            f"checkout. Cambia primero a {args.branch} y vuelve a ejecutar.{Colors.RESET}"
        )
        return 1

    remote = args.remote

    print(f"{Colors.BLUE}Repositorio: {Colors.BOLD}{REPO_ROOT}{Colors.RESET}")
    print(f"{Colors.BLUE}Rama local:  {Colors.BOLD}{branch}{Colors.RESET}")
    print(f"{Colors.BLUE}Remoto:      {Colors.BOLD}{remote}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}Paso 1: actualizando referencias remotas...{Colors.RESET}")
    success, output = run_command(["git", "fetch", remote, "--prune"])
    if not success:
        print(f"\n{Colors.RED}ERROR: no se pudo hacer fetch del remoto.{Colors.RESET}")
        if output:
            print(f"{Colors.DIM}{output}{Colors.RESET}")
        return 1

    upstream = get_upstream_ref(branch, remote)
    if not ref_exists(upstream):
        print(f"\n{Colors.RED}ERROR: no existe la referencia remota {upstream}.{Colors.RESET}")
        return 1

    behind = count_commits(f"HEAD..{upstream}")
    ahead = count_commits(f"{upstream}..HEAD")
    dirty = has_local_changes()
    ignored_count = ignored_items_count()
    commits = get_recent_remote_commits(upstream, limit=10)

    print(f"\n{Colors.CYAN}Estado detectado:{Colors.RESET}")
    print(f"   • Upstream remoto:            {Colors.BOLD}{upstream}{Colors.RESET}")
    print(f"   • Commits por detras:         {behind}")
    print(f"   • Commits por delante:        {ahead}")
    print(f"   • Cambios locales detectados: {'si' if dirty else 'no'}")
    print(f"   • Ignorados por .gitignore:   {ignored_count}")

    if not commits:
        print(f"\n{Colors.RED}ERROR: no se pudieron obtener commits recientes de {upstream}.{Colors.RESET}")
        return 1

    print_recent_remote_commits(commits)

    if args.check:
        print(f"\n{Colors.GREEN}Comprobacion terminada. No se aplicaron cambios.{Colors.RESET}")
        return 0

    selected_index = choose_commit_index(commits, preset=args.select)
    if selected_index is None:
        return 1

    selected_commit = commits[selected_index]

    print(
        f"\n{Colors.YELLOW}Paso 2: restaurando rama al commit elegido "
        f"({selected_commit.short_sha})...{Colors.RESET}"
    )
    success, output = run_command(["git", "reset", "--hard", selected_commit.sha])
    if not success:
        print(
            f"\n{Colors.RED}ERROR: fallo el reset contra "
            f"{selected_commit.short_sha}.{Colors.RESET}"
        )
        if output:
            print(f"{Colors.DIM}{output}{Colors.RESET}")
        return 1

    print(
        f"\n{Colors.YELLOW}Paso 3: eliminando archivos no versionados "
        f"(sin tocar lo ignorado)...{Colors.RESET}"
    )
    success, output = run_command(["git", "clean", "-fd"])
    if not success:
        print(f"\n{Colors.RED}ERROR: fallo la limpieza de archivos no versionados.{Colors.RESET}")
        if output:
            print(f"{Colors.DIM}{output}{Colors.RESET}")
        return 1

    final_head = short_commit("HEAD")
    final_clean = not has_local_changes()

    print(
        f"\n{Colors.GREEN}{Colors.BOLD}"
        "==============================================================\n"
        "SINCRONIZACION COMPLETADA\n"
        "=============================================================="
        f"{Colors.RESET}"
    )
    print(f"   • Rama:              {branch}")
    print(f"   • Upstream:          {upstream}")
    print(f"   • Version elegida:   {selected_index + 1}")
    print(f"   • Commit restaurado: {selected_commit.short_sha}")
    print(f"   • Fecha/Hora:        {selected_commit.date_str}")
    print(f"   • Descripcion:       {selected_commit.subject}")
    print(f"   • HEAD local:        {final_head}")
    print(f"   • Estado de trabajo: {'limpio' if final_clean else 'revisar'}")
    print("   • Ignorados:         preservados (.gitignore no se toca)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

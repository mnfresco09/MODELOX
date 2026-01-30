#!/usr/bin/env python3
"""
================================================================================
🧹 LIMPIADOR COMPLETO DEL SISTEMA MODELOX - macOS OPTIMIZADO
================================================================================

Script de limpieza profunda que elimina:
- Caché de Python (__pycache__, .pyc, .pyo)
- Archivos temporales del sistema
- Caché de compilación (Cython, Numba)
- Logs y archivos de depuración
- Caché de pip/conda en el proyecto
- Archivos de checkpoint de Optuna
- Archivos .DS_Store de macOS

⚠️  PRESERVA:
- Carpeta 'resultados/' completa
- Archivos PDF generados
- Archivos de datos originales (data/ohlcv/)
- Código fuente (.py)
- Configuración del proyecto

================================================================================
USO:
    python x/limpiar_sistema.py          # Modo interactivo (pregunta)
    python x/limpiar_sistema.py --force  # Limpieza directa sin preguntar
    python x/limpiar_sistema.py --dry    # Solo muestra qué se eliminaría
================================================================================
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Set
from datetime import datetime

# Colores ANSI para terminal
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# =============================================================================
# CONFIGURACIÓN DE LIMPIEZA
# =============================================================================

# Carpeta raíz del proyecto (un nivel arriba de x/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carpetas y archivos que NUNCA se eliminan
PROTECTED_PATHS = {
    'resultados',           # Resultados de backtests
    'data/ohlcv',           # Datos de mercado originales
    'data',                 # Carpeta de datos general
    '.git',                 # Control de versiones
    '.venv',                # Entorno virtual (si existe)
    'venv',                 # Entorno virtual alternativo
    '.env',                 # Variables de entorno
}

# Extensiones de archivos a preservar siempre
PROTECTED_EXTENSIONS = {
    '.py',          # Código fuente
    '.pyx',         # Cython source
    '.c',           # C source (nuclear_engine)
    '.h',           # Headers
    '.md',          # Documentación
    '.txt',         # Requirements, README
    '.yml', '.yaml', # Config
    '.json',        # Config
    '.toml',        # Config (pyproject.toml)
    '.feather',     # Datos comprimidos
    '.parquet',     # Datos parquet originales
    '.pdf',         # PDFs generados (en cualquier lugar)
    '.xlsx', '.xls', # Excel generados
    '.csv',         # Datos CSV originales
    '.sh',          # Scripts shell
    '.conf',        # Configuración nginx etc
    '.ts', '.tsx',  # TypeScript/React
    '.js', '.jsx',  # JavaScript
    '.css',         # Estilos
    '.html',        # HTML
    '.dockerfile', '.Dockerfile', # Docker
}

# Patrones de archivos/carpetas a ELIMINAR
CLEANUP_PATTERNS = {
    # Python cache
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.hypothesis',
    
    # Cython/Numba cache
    '*.so',              # Compiled extensions (excepto en cp/build si es necesario)
    '*.c.bak',
    '.numba_cache',
    'build/temp.*',
    
    # Jupyter
    '.ipynb_checkpoints',
    '*.ipynb~',
    
    # macOS
    '.DS_Store',
    '._*',               # Resource forks
    '.Spotlight-V100',
    '.Trashes',
    '.fseventsd',
    
    # IDEs
    '.idea',
    '.vscode/settings.json~',
    '*.swp',
    '*.swo',
    '*~',
    
    # Logs y temporales
    '*.log',
    '*.tmp',
    '*.temp',
    '*.bak',
    '*.backup',
    
    # Optuna (solo archivos de sesiones antiguas con storage en disco)
    # NOTA: Con OPTUNA_STORAGE=None (default), Optuna usa RAM y no crea estos archivos
    '*.db',              # SQLite databases de Optuna (solo si usas storage persistente)
    'optuna*.db-journal',
    
    # Otros
    '.coverage',
    'htmlcov',
    '*.egg-info',
    'dist',
    '.eggs',
    'pip-wheel-metadata',
}

# =============================================================================
# FUNCIONES DE LIMPIEZA
# =============================================================================

def is_protected(path: Path) -> bool:
    """Verifica si un path está protegido."""
    path_str = str(path.relative_to(PROJECT_ROOT))
    
    # Verificar carpetas protegidas
    for protected in PROTECTED_PATHS:
        if path_str.startswith(protected) or path_str == protected:
            return True
    
    # Verificar extensiones protegidas
    if path.suffix.lower() in PROTECTED_EXTENSIONS:
        return True
    
    # Verificar si es PDF (protegido en cualquier lugar)
    if path.suffix.lower() == '.pdf':
        return True
    
    return False


def format_size(size_bytes: int) -> str:
    """Formatea tamaño en bytes a formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_directory_size(path: Path) -> int:
    """Calcula el tamaño total de un directorio."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def find_files_to_clean() -> Tuple[List[Path], List[Path], int]:
    """
    Encuentra archivos y carpetas a limpiar.
    Retorna: (archivos, carpetas, tamaño_total)
    """
    files_to_delete: List[Path] = []
    dirs_to_delete: List[Path] = []
    total_size = 0
    
    # Patrones de archivos
    file_patterns = [p for p in CLEANUP_PATTERNS if '*' in p]
    dir_patterns = [p for p in CLEANUP_PATTERNS if '*' not in p and not p.startswith('.')]
    dot_patterns = [p for p in CLEANUP_PATTERNS if p.startswith('.') and '*' not in p]
    
    # Buscar carpetas específicas
    for dir_pattern in dir_patterns:
        for found in PROJECT_ROOT.rglob(dir_pattern):
            if found.is_dir() and not is_protected(found):
                size = get_directory_size(found)
                total_size += size
                dirs_to_delete.append(found)
    
    # Buscar archivos por patrón
    for file_pattern in file_patterns:
        for found in PROJECT_ROOT.rglob(file_pattern):
            if found.is_file() and not is_protected(found):
                try:
                    total_size += found.stat().st_size
                    files_to_delete.append(found)
                except (OSError, PermissionError):
                    pass
    
    # Buscar archivos/carpetas que empiezan con punto
    for dot_pattern in dot_patterns:
        for found in PROJECT_ROOT.rglob(dot_pattern):
            if not is_protected(found):
                if found.is_file():
                    try:
                        total_size += found.stat().st_size
                        files_to_delete.append(found)
                    except (OSError, PermissionError):
                        pass
                elif found.is_dir():
                    size = get_directory_size(found)
                    total_size += size
                    dirs_to_delete.append(found)
    
    # Eliminar duplicados y ordenar
    files_to_delete = sorted(set(files_to_delete))
    dirs_to_delete = sorted(set(dirs_to_delete))
    
    return files_to_delete, dirs_to_delete, total_size


def clean_macos_system_caches() -> int:
    """
    Limpia cachés del sistema macOS relacionadas con desarrollo.
    Retorna tamaño liberado aproximado.
    """
    freed = 0
    home = Path.home()
    
    caches_to_check = [
        home / 'Library/Caches/pip',
        home / 'Library/Caches/torch',
        home / 'Library/Caches/numba',
        home / '.cache/pip',
        home / '.cache/torch',
        home / '.cache/numba',
    ]
    
    for cache_path in caches_to_check:
        if cache_path.exists():
            try:
                size = get_directory_size(cache_path)
                shutil.rmtree(cache_path, ignore_errors=True)
                freed += size
                print(f"  {Colors.GREEN}✓{Colors.RESET} Limpiado: {cache_path.name} ({format_size(size)})")
            except (PermissionError, OSError) as e:
                print(f"  {Colors.YELLOW}⚠{Colors.RESET} No se pudo limpiar {cache_path.name}: {e}")
    
    return freed


def clean_python_memory_caches():
    """Limpia cachés en memoria de Python si están cargados."""
    import gc
    
    # Forzar recolección de basura
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Limpiar caché de módulos importados innecesarios
    modules_to_unload = []
    for name in sys.modules.keys():
        if any(x in name for x in ['matplotlib', 'torch', 'numpy', 'pandas', 'optuna']):
            modules_to_unload.append(name)
    
    # No descargar - podría causar problemas, solo registrar
    print(f"  {Colors.DIM}Módulos en memoria: {len(modules_to_unload)} (liberados con GC){Colors.RESET}")


def purge_macos_memory():
    """Purga memoria inactiva de macOS (requiere sudo para efecto completo)."""
    print(f"\n{Colors.CYAN}🧠 Liberando memoria del sistema...{Colors.RESET}")
    
    try:
        # Intenta purgar memoria (funciona mejor con sudo pero no es obligatorio)
        result = subprocess.run(
            ['purge'], 
            capture_output=True, 
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            print(f"  {Colors.GREEN}✓{Colors.RESET} Memoria inactiva liberada")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.RESET} Purge parcial (ejecuta con sudo para efecto completo)")
    except FileNotFoundError:
        print(f"  {Colors.DIM}Comando 'purge' no disponible{Colors.RESET}")
    except subprocess.TimeoutExpired:
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} Timeout en purge")
    except Exception as e:
        print(f"  {Colors.DIM}No se pudo purgar memoria: {e}{Colors.RESET}")


def show_banner():
    """Muestra banner inicial."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║         🧹 LIMPIADOR DEL SISTEMA MODELOX - macOS                 ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def main():
    parser = argparse.ArgumentParser(
        description='Limpiador completo del sistema MODELOX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python x/limpiar_sistema.py          # Modo interactivo
  python x/limpiar_sistema.py --force  # Sin confirmación
  python x/limpiar_sistema.py --dry    # Solo muestra qué se eliminaría
  python x/limpiar_sistema.py --deep   # Incluye cachés del sistema
        """
    )
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Ejecutar sin pedir confirmación')
    parser.add_argument('--dry', '-d', action='store_true',
                        help='Solo mostrar qué se eliminaría (dry run)')
    parser.add_argument('--deep', action='store_true',
                        help='Incluir cachés del sistema (pip, torch, etc.)')
    parser.add_argument('--memory', '-m', action='store_true',
                        help='Solo liberar memoria RAM (sin borrar archivos)')
    
    args = parser.parse_args()
    
    show_banner()
    
    # Modo solo memoria
    if args.memory:
        print(f"{Colors.YELLOW}📍 Modo: Solo liberación de memoria{Colors.RESET}\n")
        clean_python_memory_caches()
        purge_macos_memory()
        print(f"\n{Colors.GREEN}✅ Memoria liberada{Colors.RESET}")
        return
    
    # Buscar archivos a limpiar
    print(f"{Colors.YELLOW}🔍 Analizando proyecto en: {PROJECT_ROOT}{Colors.RESET}\n")
    
    files, dirs, total_size = find_files_to_clean()
    
    if not files and not dirs:
        print(f"{Colors.GREEN}✨ ¡El proyecto ya está limpio! No hay nada que eliminar.{Colors.RESET}")
        return
    
    # Mostrar resumen
    print(f"{Colors.BOLD}📊 RESUMEN DE LIMPIEZA:{Colors.RESET}")
    print(f"   • Archivos a eliminar: {Colors.CYAN}{len(files)}{Colors.RESET}")
    print(f"   • Carpetas a eliminar: {Colors.CYAN}{len(dirs)}{Colors.RESET}")
    print(f"   • Espacio a liberar:   {Colors.GREEN}{format_size(total_size)}{Colors.RESET}")
    
    # Mostrar carpetas protegidas
    print(f"\n{Colors.DIM}🛡️  Protegido (no se tocará):{Colors.RESET}")
    print(f"   {Colors.DIM}• resultados/ (backtests y reportes){Colors.RESET}")
    print(f"   {Colors.DIM}• data/ohlcv/ (datos de mercado){Colors.RESET}")
    print(f"   {Colors.DIM}• *.pdf, *.xlsx (reportes generados){Colors.RESET}")
    print(f"   {Colors.DIM}• Código fuente (.py, .pyx, etc.){Colors.RESET}")
    
    # Modo dry run
    if args.dry:
        print(f"\n{Colors.MAGENTA}{'─'*60}{Colors.RESET}")
        print(f"{Colors.MAGENTA}📝 MODO DRY RUN - Nada será eliminado{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'─'*60}{Colors.RESET}\n")
        
        if dirs:
            print(f"{Colors.YELLOW}Carpetas:{Colors.RESET}")
            for d in dirs[:20]:  # Limitar a 20 para no saturar
                rel_path = d.relative_to(PROJECT_ROOT)
                size = get_directory_size(d)
                print(f"   {Colors.RED}[-]{Colors.RESET} {rel_path}/ ({format_size(size)})")
            if len(dirs) > 20:
                print(f"   {Colors.DIM}... y {len(dirs) - 20} más{Colors.RESET}")
        
        if files:
            print(f"\n{Colors.YELLOW}Archivos:{Colors.RESET}")
            for f in files[:30]:  # Limitar a 30
                rel_path = f.relative_to(PROJECT_ROOT)
                try:
                    size = f.stat().st_size
                    print(f"   {Colors.RED}[-]{Colors.RESET} {rel_path} ({format_size(size)})")
                except OSError:
                    print(f"   {Colors.RED}[-]{Colors.RESET} {rel_path}")
            if len(files) > 30:
                print(f"   {Colors.DIM}... y {len(files) - 30} más{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}Ejecuta sin --dry para eliminar estos archivos.{Colors.RESET}")
        return
    
    # Confirmación
    if not args.force:
        print(f"\n{Colors.YELLOW}⚠️  Esta acción eliminará {len(files) + len(dirs)} elementos ({format_size(total_size)}){Colors.RESET}")
        response = input(f"\n{Colors.BOLD}¿Continuar? [s/N]: {Colors.RESET}").strip().lower()
        if response not in ('s', 'si', 'sí', 'y', 'yes'):
            print(f"{Colors.DIM}Operación cancelada.{Colors.RESET}")
            return
    
    # Ejecutar limpieza
    print(f"\n{Colors.CYAN}🧹 Limpiando...{Colors.RESET}\n")
    
    deleted_files = 0
    deleted_dirs = 0
    errors = 0
    
    # Eliminar carpetas primero (incluye sus archivos)
    for d in dirs:
        try:
            shutil.rmtree(d)
            deleted_dirs += 1
            rel_path = d.relative_to(PROJECT_ROOT)
            print(f"  {Colors.GREEN}✓{Colors.RESET} {rel_path}/")
        except Exception as e:
            errors += 1
            print(f"  {Colors.RED}✗{Colors.RESET} {d.name}: {e}")
    
    # Eliminar archivos individuales
    for f in files:
        # Verificar que no esté dentro de una carpeta ya eliminada
        if not f.exists():
            continue
        try:
            f.unlink()
            deleted_files += 1
        except Exception as e:
            errors += 1
    
    print(f"\n  {Colors.DIM}Archivos individuales eliminados: {deleted_files}{Colors.RESET}")
    
    # Limpieza profunda opcional
    deep_freed = 0
    if args.deep:
        print(f"\n{Colors.CYAN}🔧 Limpieza profunda de cachés del sistema...{Colors.RESET}")
        deep_freed = clean_macos_system_caches()
    
    # Liberar memoria
    clean_python_memory_caches()
    purge_macos_memory()
    
    # Resumen final
    print(f"""
{Colors.GREEN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                    ✅ LIMPIEZA COMPLETADA                        ║
╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}

   • Carpetas eliminadas:  {Colors.GREEN}{deleted_dirs}{Colors.RESET}
   • Archivos eliminados:  {Colors.GREEN}{deleted_files}{Colors.RESET}
   • Espacio liberado:     {Colors.GREEN}{format_size(total_size + deep_freed)}{Colors.RESET}
   • Errores:              {Colors.YELLOW if errors else Colors.GREEN}{errors}{Colors.RESET}

{Colors.DIM}   📁 resultados/ y PDFs preservados intactos{Colors.RESET}
""")


if __name__ == "__main__":
    main()

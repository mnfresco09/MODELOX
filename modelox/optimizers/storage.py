from __future__ import annotations

import os
import shutil
import glob
from typing import Optional


DATABASE_DIR_NAME = "DATABASE"
RESULTS_DIR_NAME = "resultados"


def get_database_dir(*, base_dir: Optional[str] = None, create: bool = True) -> str:
    """Devuelve la carpeta central de DB y la crea si no existe."""
    root = os.path.abspath(base_dir or os.getcwd())
    db_dir = os.path.join(root, RESULTS_DIR_NAME, DATABASE_DIR_NAME)
    if create:
        os.makedirs(db_dir, exist_ok=True)
    return db_dir


def get_database_file_path(
    db_name: str,
    *,
    base_dir: Optional[str] = None,
    create_dir: bool = True,
    migrate_from_root: bool = True,
) -> str:
    """Ruta absoluta a un .db dentro de DATABASE, con migración opcional.

    Si existe un archivo legacy en la raíz del proyecto y no existe todavía en
    DATABASE, lo mueve automáticamente para mantener todo centralizado.
    """
    db_dir = get_database_dir(base_dir=base_dir, create=create_dir)
    target = os.path.join(db_dir, db_name)

    if migrate_from_root:
        root = os.path.abspath(base_dir or os.getcwd())
        legacy = os.path.join(root, db_name)
        if os.path.exists(legacy):
            # Caso 1: no existe aún en DATABASE -> mover.
            if not os.path.exists(target):
                try:
                    shutil.move(legacy, target)
                except Exception:
                    # Si no se puede mover (permisos/lock), seguimos sin romper ejecución.
                    pass
            else:
                # Caso 2: ya existe en DATABASE -> evitar duplicado en raíz.
                # Si el de raíz está vacío (caso típico por comando manual), eliminarlo.
                try:
                    if os.path.getsize(legacy) == 0:
                        os.remove(legacy)
                except Exception:
                    pass

    return target


def organize_root_database_files(*, base_dir: Optional[str] = None) -> int:
    """Mueve los .db de la raíz del proyecto a DATABASE.

    Retorna cuántos archivos fueron movidos/eliminados en raíz.
    """
    root = os.path.abspath(base_dir or os.getcwd())
    db_dir = get_database_dir(base_dir=root, create=True)

    moved = 0
    for name in os.listdir(root):
        if not name.lower().endswith(".db"):
            continue
        src = os.path.join(root, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(db_dir, name)
        if os.path.exists(dst):
            # Evitar duplicados: si en raíz quedó un .db vacío, se elimina.
            try:
                if os.path.getsize(src) == 0:
                    os.remove(src)
                    moved += 1
            except Exception:
                pass
            continue

        try:
            shutil.move(src, dst)
            moved += 1
        except Exception:
            # Ignorar locks/permisos y continuar con el resto.
            continue

    return moved


def delete_database_file(
    db_name: str,
    *,
    base_dir: Optional[str] = None,
    delete_root_legacy: bool = True,
) -> int:
    """Elimina un .db y sus sidecars (.wal/.shm/.journal).

    Por seguridad elimina en DATABASE/ y, opcionalmente, en raíz del proyecto.
    Retorna cantidad de archivos eliminados.
    """
    root = os.path.abspath(base_dir or os.getcwd())
    db_dir = get_database_dir(base_dir=root, create=True)

    removed = 0

    def _delete_with_sidecars(base_path: str) -> int:
        local_removed = 0
        patterns = [
            base_path,
            f"{base_path}-wal",
            f"{base_path}-shm",
            f"{base_path}-journal",
        ]
        for p in patterns:
            for fpath in glob.glob(p):
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        local_removed += 1
                except Exception:
                    continue
        return local_removed

    # Principal: DATABASE/
    removed += _delete_with_sidecars(os.path.join(db_dir, db_name))

    # Legacy root (por comandos manuales tipo sqlite:///IDx.db)
    if delete_root_legacy:
        removed += _delete_with_sidecars(os.path.join(root, db_name))

    return removed


def delete_strategy_database(
    *,
    strategy_id: Optional[int],
    base_dir: Optional[str] = None,
) -> int:
    """Elimina la DB de una estrategia (IDx.db) y sidecars."""
    try:
        sid = int(strategy_id) if strategy_id is not None else None
    except Exception:
        sid = None

    db_name = f"ID{sid}.db" if sid and sid > 0 else "optuna.db"
    return delete_database_file(db_name, base_dir=base_dir, delete_root_legacy=True)


def resolve_storage_for_strategy(
    *,
    create_database: bool,
    strategy_id: Optional[int] = None,
    reset_existing: bool = True,
) -> Optional[str]:
    """Resuelve el storage SQLite por estrategia (IDx.db) o RAM.

    Si ``reset_existing=True`` (default), elimina automáticamente la DB previa
    de esa estrategia antes de iniciar una nueva ejecución.
    """
    if not create_database:
        return None

    try:
        sid = int(strategy_id) if strategy_id is not None else None
    except Exception:
        sid = None

    db_name = f"ID{sid}.db" if sid and sid > 0 else "optuna.db"

    if reset_existing:
        try:
            delete_database_file(db_name, delete_root_legacy=True)
        except Exception:
            # Nunca bloquear el arranque por problemas de borrado (locks/permisos).
            pass

    db_path = get_database_file_path(db_name)
    return f"sqlite:///{db_path}"

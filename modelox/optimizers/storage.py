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


def build_global_studies(
    strategy_id: Optional[int] = None,
    *,
    base_dir: Optional[str] = None,
) -> int:
    """Crea/actualiza estudios globales en el DB de una estrategia.

    Para cada grupo de estudios del mismo tipo (ej: modelox-breakout-qmc-btc,
    modelox-breakout-qmc-gold, …) crea un estudio consolidado
    (modelox-breakout-qmc-global) que contiene todos sus trials.

    Cada trial copiado recibe en user_attrs:
        __activo        — activo de origen (ej: "btc")
        __source_study  — nombre del estudio origen
        __source_number — número de trial en el estudio origen (para dedup)

    Retorna el número de trials copiados en total.
    """
    try:
        import optuna
        from optuna.trial import FrozenTrial, TrialState
    except ImportError:
        return 0

    try:
        sid = int(strategy_id) if strategy_id is not None else None
    except Exception:
        sid = None

    db_name = f"ID{sid}.db" if sid and sid > 0 else "optuna.db"
    db_path = get_database_file_path(db_name, base_dir=base_dir, create_dir=False)

    if not os.path.exists(db_path):
        return 0

    storage_url = f"sqlite:///{db_path}"

    try:
        all_names = optuna.get_all_study_names(storage=storage_url)
    except Exception:
        return 0

    # Excluir estudios globales ya existentes
    source_studies = [n for n in all_names if not (n.endswith("-global") or n.endswith("_global"))]

    if len(source_studies) < 2:
        return 0

    # ── Agrupar estudios por prefijo (todo menos el activo al final) ──────────
    # Soporta separadores '-' y '_'. Ejemplo:
    #   modelox-breakout-qmc-btc  → prefijo=modelox-breakout-qmc  sep='-'  activo=btc
    groups: dict = {}  # prefix → [(study_name, sep_char, activo_slug)]
    for name in source_studies:
        last_dash  = name.rfind("-")
        last_under = name.rfind("_")
        sep_idx    = max(last_dash, last_under)
        if sep_idx <= 0:
            continue
        sep_char   = name[sep_idx]
        prefix     = name[:sep_idx]
        activo_slug = name[sep_idx + 1:]
        if not prefix or not activo_slug:
            continue
        groups.setdefault(prefix, []).append((name, sep_char, activo_slug))

    total_copied = 0

    for prefix, entries in groups.items():
        if len(entries) < 2:
            continue

        sep_char    = entries[0][1]
        global_name = f"{prefix}{sep_char}global"

        # Crear o cargar estudio global
        try:
            global_study = optuna.create_study(
                study_name=global_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True,
            )
        except Exception:
            continue

        # Rastrear trials ya copiados para evitar duplicados en ejecuciones sucesivas
        existing: set = set()
        for t in global_study.trials:
            src  = t.user_attrs.get("__source_study", "")
            num  = t.user_attrs.get("__source_number", -1)
            if src:
                existing.add((src, int(num)))

        for study_name, _, activo_slug in entries:
            try:
                asset_study = optuna.load_study(study_name=study_name, storage=storage_url)
            except Exception:
                continue

            complete_trials = [t for t in asset_study.trials if t.state == TrialState.COMPLETE]

            # ── Normalizar al rango [1, 10] dentro de cada estudio ───────────
            # Tanto el score (trial.value) como cada métrica numérica en
            # user_attrs["metricas"] y los attrs planos relevantes.
            # Así activos con rangos absolutos muy distintos no distorsionan el global.

            # Attrs planos numéricos a normalizar (excluyendo conteos enteros)
            _FLAT_ATTRS = {"final_score", "expectancy", "max_dd_pct", "trades_per_day"}

            def _build_stats(values):
                """Devuelve (vmin, vmax) o None si no hay suficientes datos."""
                clean = [v for v in values if v is not None]
                if len(clean) < 2:
                    return None
                return (min(clean), max(clean))

            def _norm(v, stats):
                """Normaliza v a [1, 10] dado stats=(vmin, vmax)."""
                if v is None or stats is None:
                    return v
                vmin, vmax = stats
                if vmax == vmin:
                    return 5.5
                return 1.0 + 9.0 * (v - vmin) / (vmax - vmin)

            # Score principal
            _score_stats = _build_stats([t.value for t in complete_trials])

            # Attrs planos
            _flat_stats = {}
            for _ak in _FLAT_ATTRS:
                _flat_stats[_ak] = _build_stats(
                    [t.user_attrs.get(_ak) for t in complete_trials]
                )

            # Claves numéricas dentro del dict "metricas"
            _metric_stats = {}
            for t in complete_trials:
                _m = t.user_attrs.get("metricas", {})
                if not isinstance(_m, dict):
                    continue
                for k, v in _m.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        _metric_stats.setdefault(k, []).append(v)
            _metric_stats = {k: _build_stats(vs) for k, vs in _metric_stats.items()}

            for trial in complete_trials:
                if (study_name, trial.number) in existing:
                    continue

                new_attrs = dict(trial.user_attrs)
                new_attrs["__activo"]        = activo_slug
                new_attrs["__source_study"]  = study_name
                new_attrs["__source_number"] = trial.number
                new_attrs["__score_raw"]     = trial.value  # score original

                # Normalizar attrs planos
                for _ak in _FLAT_ATTRS:
                    if _ak in new_attrs and isinstance(new_attrs[_ak], (int, float)):
                        new_attrs[f"__{_ak}_raw"] = new_attrs[_ak]
                        new_attrs[_ak] = _norm(new_attrs[_ak], _flat_stats.get(_ak))

                # Normalizar dict "metricas"
                _m_orig = new_attrs.get("metricas", {})
                if isinstance(_m_orig, dict) and _m_orig:
                    new_attrs["__metricas_raw"] = _m_orig
                    _m_norm = {}
                    for k, v in _m_orig.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            _m_norm[k] = _norm(v, _metric_stats.get(k))
                        else:
                            _m_norm[k] = v
                    new_attrs["metricas"] = _m_norm

                try:
                    # FrozenTrial exige value XOR values (no ambos a la vez)
                    use_values = trial.values is not None and len(trial.values) > 1
                    norm_value = _norm(trial.value, _score_stats) if not use_values else None
                    copied_trial = FrozenTrial(
                        number              = trial.number,
                        state               = trial.state,
                        value               = norm_value,
                        values              = trial.values if use_values else None,
                        datetime_start      = trial.datetime_start,
                        datetime_complete   = trial.datetime_complete,
                        params              = trial.params,
                        distributions       = trial.distributions,
                        user_attrs          = new_attrs,
                        system_attrs        = trial.system_attrs,
                        intermediate_values = trial.intermediate_values,
                        trial_id            = trial._trial_id,
                    )
                    global_study.add_trial(copied_trial)
                    existing.add((study_name, trial.number))
                    total_copied += 1
                except Exception:
                    pass

    return total_copied


def resolve_storage_for_strategy(
    *,
    create_database: bool,
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    reset_existing: bool = False,
) -> Optional[str]:
    """Resuelve el storage SQLite por estrategia (IDx.db) o RAM.

    El nombre del archivo es siempre IDx.db (sin activo) para mantener
    compatibilidad con optuna-dashboard sqlite:///DATABASE/IDx.db.
    Cada activo genera un estudio distinto dentro del mismo archivo.

    reset_existing=False (default): acumula estudios de distintos activos.
    reset_existing=True: borra el DB completo antes de arrancar (reset manual).
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
            pass

    db_path = get_database_file_path(db_name)
    return f"sqlite:///{db_path}"

"""
================================================================================
MODELOX/STRATEGIES/REGISTRY.PY — SISTEMA DE DESCUBRIMIENTO DE ESTRATEGIAS
================================================================================

PROPÓSITO:
    Gestiona el descubrimiento automático, carga y validación de estrategias 
    en el directorio `modelox/strategies/`.

FUNCIONALIDAD:
    1. Auto-descubrimiento de clases que heredan de `Strategy`.
    2. Validación de interfaz (métodos requeridos, tipos).
    3. Carga selectiva por ID (evita cargar todas las estrategias si no se usan).
    4. Prevención de conflictos de IDs y nombres.

USO PRINCIPAL:
    - `discover_strategies()`: Retorna dict {nombre: clase}
    - `instantiate_strategies(only_id=...)`: Carga e instancia una estrategia.

================================================================================
"""
from __future__ import annotations


import importlib
import importlib.util
import importlib.machinery
import ast
import hashlib
import inspect
import os
import pkgutil
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

from modelox.core.types import Strategy


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    cls: Type[Strategy]


def validate_strategy(cls: Type) -> List[str]:
    """
    Valida que una clase de estrategia cumpla con los requisitos.

    Returns:
        Lista de errores de validación (vacía si no hay errores)
    """
    errors = []

    # Validar name
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        errors.append(f"Estrategia {cls.__name__}: atributo 'name' faltante o vacío")

    # Validar combinacion_id
    combinacion_id = getattr(cls, "combinacion_id", None)
    if not isinstance(combinacion_id, int):
        errors.append(f"Estrategia {cls.__name__}: atributo 'combinacion_id' debe ser int")
    elif combinacion_id <= 0:
        errors.append(f"Estrategia {cls.__name__}: 'combinacion_id' debe ser > 0")

    # Validar métodos requeridos
    required_methods = ["suggest_params", "generate_signals"]
    for method_name in required_methods:
        if not hasattr(cls, method_name):
            errors.append(f"Estrategia {cls.__name__}: método '{method_name}' faltante")
        elif not callable(getattr(cls, method_name)):
            errors.append(f"Estrategia {cls.__name__}: '{method_name}' no es callable")

    # Validar parametros_optuna (recomendado)
    if not hasattr(cls, "parametros_optuna"):
        errors.append(f"Estrategia {cls.__name__}: 'parametros_optuna' no definido (recomendado)")

    return errors


def discover_strategies(*, package: str = "modelox.strategies") -> Dict[str, Type[Strategy]]:
    """
    Auto-discover strategies from `modelox/strategies/*.py`.

    Convention:
    - A strategy is a class with a `name: str` attribute and required methods
            (suggest_params, generate_signals).
    """

    import logging

    logger = logging.getLogger(__name__)
    strategies: Dict[str, Type[Strategy]] = {}
    ids_seen: Dict[int, str] = {}
    files_seen: set[str] = set()

    pkg = importlib.import_module(package)
    pkg_path = getattr(pkg, "__path__", None)
    pkg_dir = None
    if pkg_path:
        try:
            pkg_dir = next(iter(pkg_path))
        except Exception:
            pkg_dir = None

    def _realpath(p: str | None) -> str | None:
        if not isinstance(p, str) or not p:
            return None
        try:
            return os.path.realpath(p)
        except Exception:
            return None

    def _module_file(m: object) -> str | None:
        return _realpath(getattr(m, "__file__", None))

    def _strategy_source_file(cls: Type) -> str | None:
        """Best-effort file path for a strategy class definition."""
        try:
            mod = sys.modules.get(getattr(cls, "__module__", ""))
            if mod is not None:
                mf = _module_file(mod)
                if mf:
                    return mf
        except Exception:
            pass
        try:
            sf = inspect.getsourcefile(cls)
            return _realpath(sf)
        except Exception:
            return None

    def _iter_strategy_module_names() -> List[str]:
        names: List[str] = []
        if pkg_path:
            for mod in pkgutil.iter_modules(pkg_path, package + "."):
                names.append(mod.name)
        return names

    def _iter_strategy_file_paths() -> List[str]:
        paths: List[str] = []
        if not pkg_dir or not os.path.isdir(pkg_dir):
            return paths
        for fname in os.listdir(pkg_dir):
            if not fname.lower().endswith(".py"):
                continue
            if fname in {"__init__.py", "registry.py"}:
                continue
            paths.append(os.path.realpath(os.path.join(pkg_dir, fname)))
        return sorted(paths)

    def _sanitized_module_name(file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        base = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_")
        if not base:
            base = "strategy"
        return f"{package}.{base}"

    def _load_module_from_path(file_path: str):
        # Evitar re-imports del mismo path con distintos nombres
        file_path = os.path.realpath(file_path)
        if file_path in files_seen:
            return None

        mod_name = _sanitized_module_name(file_path)

        # Si ya existe un módulo con ese nombre apuntando al mismo archivo,
        # reutilizarlo para evitar redefinir clases (y falsos duplicados).
        existing = sys.modules.get(mod_name)
        if existing is not None:
            existing_file = _module_file(existing)
            if existing_file and existing_file == file_path:
                files_seen.add(file_path)
                return existing
            # Si colisiona con otro archivo, crear un nombre único y estable.
            suffix = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
            mod_name = f"{mod_name}__fp_{suffix}"

        files_seen.add(file_path)

        try:
            # Forzar SourceFileLoader para soportar extensiones en mayúsculas (ej: .PY)
            # ya que spec_from_file_location sin loader puede devolver None.
            loader = importlib.machinery.SourceFileLoader(mod_name, file_path)
            spec = importlib.util.spec_from_file_location(mod_name, file_path, loader=loader)
            if spec is None or spec.loader is None:
                logger.warning(f"No se pudo crear spec para estrategia: {file_path}")
                return None
            module = importlib.util.module_from_spec(spec)
            # Importante para imports relativos dentro del módulo (from .ESTRATEGIA_BASE import ...)
            module.__package__ = package
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.warning(f"Error importando estrategia desde {file_path}: {e}")
            return None

    def _iter_modules() -> List[object]:
        modules: List[object] = []
        # 1) Ruta estándar (pkgutil)
        for mod_name in _iter_strategy_module_names():
            try:
                m = importlib.import_module(mod_name)
                # Marcar su archivo como ya cargado para evitar el fallback duplicado
                m_file = _module_file(m)
                if m_file:
                    files_seen.add(m_file)
                modules.append(m)
            except Exception as e:
                logger.warning(f"Error importando módulo {mod_name}: {e}")

        # 2) Fallback universal: escanear archivos .py (case-insensitive)
        for path in _iter_strategy_file_paths():
            m = _load_module_from_path(path)
            if m is not None:
                modules.append(m)
        return modules

    for module in _iter_modules():
        if module is None:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Aceptamos solo clases definidas en el propio módulo
            if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
                continue

            name = getattr(obj, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue

            combinacion_id = getattr(obj, "combinacion_id", None)
            if not isinstance(combinacion_id, int) or combinacion_id <= 0:
                continue

            if not all(hasattr(obj, m) for m in ("suggest_params", "generate_signals")):
                continue

            validation_errors = validate_strategy(obj)
            for error in validation_errors:
                logger.warning(error)

            if name in strategies and strategies[name] is not obj:
                # Si es el MISMO archivo importado 2 veces, ignorar el duplicado.
                prev_cls = strategies[name]
                prev_file = _strategy_source_file(prev_cls)
                new_file = _strategy_source_file(obj)
                if prev_file and new_file and prev_file == new_file:
                    continue
                raise ValueError(
                    f"Nombre de estrategia duplicado: '{name}'. "
                    f"Clases: {prev_cls.__name__} y {obj.__name__}."
                )

            if combinacion_id in ids_seen and ids_seen[combinacion_id] != name:
                raise ValueError(
                    f"combinacion_id duplicado: {combinacion_id} en '{name}' y '{ids_seen[combinacion_id]}'"
                )

            ids_seen[combinacion_id] = name
            strategies[name] = obj

    return strategies


def _find_strategy_file_by_id(*, package: str, combinacion_id: int) -> str | None:
    """Best-effort scan of strategy source files to find the file that defines a class with
    `combinacion_id = <target>`.

    This intentionally does NOT import modules (so it can work even if unrelated
    strategies have import-time errors or duplicate names).
    """

    try:
        pkg = importlib.import_module(package)
    except Exception:
        return None

    pkg_path = getattr(pkg, "__path__", None)
    pkg_dir = None
    if pkg_path:
        try:
            pkg_dir = next(iter(pkg_path))
        except Exception:
            pkg_dir = None

    if not pkg_dir or not os.path.isdir(pkg_dir):
        return None

    target = int(combinacion_id)
    for fname in sorted(os.listdir(pkg_dir)):
        if not fname.lower().endswith(".py"):
            continue
        if fname in {"__init__.py", "registry.py"}:
            continue

        path = os.path.realpath(os.path.join(pkg_dir, fname))
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, filename=path)
        except Exception:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "combinacion_id":
                        try:
                            val = ast.literal_eval(stmt.value)
                        except Exception:
                            val = None
                        if isinstance(val, int) and int(val) == target:
                            return path

    return None


def _load_strategy_class_by_id(*, package: str, combinacion_id: int) -> Type[Strategy] | None:
    """Load ONLY the strategy class matching combinacion_id.

    This avoids importing every strategy module (which can fail due to unrelated
    duplicates or import-time issues).
    """

    import logging

    logger = logging.getLogger(__name__)
    cid = int(combinacion_id)

    # Reuse the same robust loader logic as discover_strategies, but in a targeted way.
    files_seen: set[str] = set()

    def _realpath(p: str | None) -> str | None:
        if not isinstance(p, str) or not p:
            return None
        try:
            return os.path.realpath(p)
        except Exception:
            return None

    def _module_file(m: object) -> str | None:
        return _realpath(getattr(m, "__file__", None))

    def _sanitized_module_name(file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        base = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_")
        if not base:
            base = "strategy"
        return f"{package}.{base}"

    def _load_module_from_path(file_path: str):
        file_path = os.path.realpath(file_path)
        if file_path in files_seen:
            return None

        mod_name = _sanitized_module_name(file_path)
        existing = sys.modules.get(mod_name)
        if existing is not None:
            existing_file = _module_file(existing)
            if existing_file and existing_file == file_path:
                files_seen.add(file_path)
                return existing
            suffix = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
            mod_name = f"{mod_name}__fp_{suffix}"

        files_seen.add(file_path)

        try:
            # Forzar SourceFileLoader para soportar extensiones en mayúsculas (ej: .PY)
            loader = importlib.machinery.SourceFileLoader(mod_name, file_path)
            spec = importlib.util.spec_from_file_location(mod_name, file_path, loader=loader)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            module.__package__ = package
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.warning(f"Error importando estrategia desde {file_path}: {e}")
            return None

    # 1) Find file by AST scan (no imports)
    target_file = _find_strategy_file_by_id(package=package, combinacion_id=cid)
    if target_file:
        m = _load_module_from_path(target_file)
        if m is not None:
            for _, obj in inspect.getmembers(m, inspect.isclass):
                if getattr(obj, "__module__", None) != getattr(m, "__name__", None):
                    continue
                if getattr(obj, "combinacion_id", None) == cid and all(
                    hasattr(obj, mm) for mm in ("suggest_params", "generate_signals")
                ):
                    return obj

    # 2) Fallback: try pkgutil modules one by one until match
    try:
        pkg = importlib.import_module(package)
        pkg_path = getattr(pkg, "__path__", None)
    except Exception:
        pkg_path = None

    if pkg_path:
        for mod in pkgutil.iter_modules(pkg_path, package + "."):
            try:
                m = importlib.import_module(mod.name)
                mf = _module_file(m)
                if mf:
                    files_seen.add(mf)
                for _, obj in inspect.getmembers(m, inspect.isclass):
                    if getattr(obj, "__module__", None) != getattr(m, "__name__", None):
                        continue
                    if getattr(obj, "combinacion_id", None) == cid and all(
                        hasattr(obj, mm) for mm in ("suggest_params", "generate_signals")
                    ):
                        return obj
            except Exception:
                continue

    return None


def discover_strategies_by_id(*, package: str = "modelox.strategies") -> Dict[int, Type[Strategy]]:
    """Convenience mapping: combinacion_id -> Strategy class."""

    by_name = discover_strategies(package=package)
    by_id: Dict[int, Type[Strategy]] = {}
    for cls in by_name.values():
        by_id[int(cls.combinacion_id)] = cls
    return by_id


def list_available_strategies(*, package: str = "modelox.strategies") -> List[Tuple[int, str]]:
    """Returns sorted list of (combinacion_id, name)."""

    by_name = discover_strategies(package=package)
    items = [(int(cls.combinacion_id), str(cls.name)) for cls in by_name.values()]
    return sorted(items, key=lambda x: x[0])


def instantiate_strategies(
    *,
    only: Optional[Sequence[str]] = None,
    only_id: Optional[int] = None,
    package: str = "modelox.strategies",
) -> List[Strategy]:
    """Instantiate discovered strategies, optionally filtered by name."""

    if only_id is not None and int(only_id) != 0:
        cid = int(only_id)

        # Targeted load: avoid importing every strategy (unrelated duplicates shouldn't block execution)
        cls = _load_strategy_class_by_id(package=package, combinacion_id=cid)
        if cls is None:
            disponibles = list_available_strategies(package=package)
            raise ValueError(f"Combinación no encontrada: {cid}. Disponibles: {disponibles}")
        return [cls()]

    registry = discover_strategies(package=package)
    names = list(registry.keys())
    if only is not None:
        missing = [n for n in only if n not in registry]
        if missing:
            raise ValueError(f"Estrategias no encontradas: {missing}. Disponibles: {sorted(names)}")
        names = list(only)
    return [registry[n]() for n in names]



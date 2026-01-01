from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

from modelox.core.types import Strategy


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    cls: Type[Strategy]


def discover_strategies(*, package: str = "modelox.strategies") -> Dict[str, Type[Strategy]]:
    """
    Auto-discover strategies from `modelox/strategies/*.py`.

    Convention:
    - A strategy is a class with a `name: str` attribute and required methods
            (suggest_params, generate_signals).
    """

    strategies: Dict[str, Type[Strategy]] = {}
    ids_seen: Dict[int, str] = {}
    pkg = importlib.import_module(package)

    for mod in pkgutil.iter_modules(pkg.__path__, package + "."):
        module = importlib.import_module(mod.name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            name = getattr(obj, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            combinacion_id = getattr(obj, "combinacion_id", None)
            if not isinstance(combinacion_id, int) or combinacion_id <= 0:
                # IDs start at 1; config uses 0 to mean "all".
                continue
            # Heuristic: class must have the expected methods.
            if not all(hasattr(obj, m) for m in ("suggest_params", "generate_signals")):
                continue
            if combinacion_id in ids_seen and ids_seen[combinacion_id] != name:
                raise ValueError(
                    f"combinacion_id duplicado: {combinacion_id} en '{name}' y '{ids_seen[combinacion_id]}'"
                )
            ids_seen[combinacion_id] = name
            strategies[name] = obj  # last one wins if name collides

    return strategies


def discover_strategies_by_id(*, package: str = "modelox.strategies") -> Dict[int, Type[Strategy]]:
    """Convenience mapping: combinacion_id -> Strategy class."""

    by_name = discover_strategies(package=package)
    by_id: Dict[int, Type[Strategy]] = {}
    for cls in by_name.values():
        by_id[int(getattr(cls, "combinacion_id"))] = cls
    return by_id


def list_available_strategies(*, package: str = "modelox.strategies") -> List[Tuple[int, str]]:
    """Returns sorted list of (combinacion_id, name)."""

    by_name = discover_strategies(package=package)
    items = [(int(getattr(cls, "combinacion_id")), str(getattr(cls, "name"))) for cls in by_name.values()]
    return sorted(items, key=lambda x: x[0])


def instantiate_strategies(
    *,
    only: Optional[Sequence[str]] = None,
    only_id: Optional[int] = None,
    package: str = "modelox.strategies",
) -> List[Strategy]:
    """Instantiate discovered strategies, optionally filtered by name."""

    if only_id is not None and int(only_id) != 0:
        reg_by_id = discover_strategies_by_id(package=package)
        cid = int(only_id)
        if cid not in reg_by_id:
            disponibles = list_available_strategies(package=package)
            raise ValueError(f"Combinación no encontrada: {cid}. Disponibles: {disponibles}")
        return [reg_by_id[cid]()]

    registry = discover_strategies(package=package)
    names = list(registry.keys())
    if only is not None:
        missing = [n for n in only if n not in registry]
        if missing:
            raise ValueError(f"Estrategias no encontradas: {missing}. Disponibles: {sorted(names)}")
        names = list(only)
    return [registry[n]() for n in names]



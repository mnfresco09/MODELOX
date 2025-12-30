from __future__ import annotations

import re
import warnings
from typing import Optional

import optuna
from optuna.exceptions import ExperimentalWarning

from modelox.core.optuna_config import OptunaConfig

# Silenciar warnings experimentales de Optuna (multivariate/group en TPESampler)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    activo: Optional[str] = None,
) -> optuna.study.Study:
    """
    Crea un estudio de Optuna para optimización de estrategias.
    
    Características:
    - seed=None: 100% aleatorio, cada ejecución es independiente
    - storage=None: sin persistencia, no se crea ningún .db
    - load_if_exists=False: siempre empieza de cero
    """

    # Sampler recomendado por defecto
    sampler: optuna.samplers.BaseSampler
    if cfg.sampler.lower() == "tpe":
        # seed=None permite aleatoriedad real (no determinista)
        sampler = optuna.samplers.TPESampler(
            seed=cfg.seed,  # None = aleatorio, int = reproducible
            multivariate=True,
            group=True,
        )
    else:
        raise ValueError(f"Sampler no soportado: {cfg.sampler}")

    # Nombre del estudio (solo identificativo, sin persistencia)
    parts = [cfg.study_name_prefix, strategy_name]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))

    # SIEMPRE: storage=None, load_if_exists=False para empezar de cero
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=None,  # Sin persistencia - no .db files
        load_if_exists=False,  # Siempre empezar de cero
    )



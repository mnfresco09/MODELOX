from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptunaConfig:
    """
    Configuración de optimización (Optuna).

    - sampler: por defecto TPE multivariante (mejor búsqueda en espacios mezclados int/float)
    - n_jobs: permite ejecutar trials en paralelo (threads) dentro de un mismo estudio
    - seed: None = aleatorio real, int = reproducible
    - storage: None = sin persistencia (cada sesión empieza de cero)
    """

    seed: Optional[int] = None  # None = 100% aleatorio, no determinista
    n_jobs: int = 1
    storage: Optional[str] = None  # None = sin persistencia, no .db
    study_name_prefix: str = "MODELOX"
    sampler: str = "tpe"  # 'tpe' (por ahora)






from __future__ import annotations

"""
Strategy 7 — Z-Score Reversal + Bollinger Bands

Estrategia 100% determinista basada en:
- Z-Score en extremo (sobreventa/sobrecompra)
- Z-Score revierte durante N velas consecutivas
- Precio toca banda inferior/superior de Bollinger

LÓGICA DE ENTRADA:

  LONG:
    1) Z-Score está en zona extrema negativa (z < -z_threshold)
    2) Z-Score revierte (sube) durante N velas consecutivas (1-3)
    3) Precio está en la banda inferior de Bollinger (close <= bb_lower)
    4) Todas las condiciones se cumplen simultáneamente

  SHORT:
    1) Z-Score está en zona extrema positiva (z > +z_threshold)
    2) Z-Score revierte (baja) durante N velas consecutivas (1-3)
    3) Precio está en la banda superior de Bollinger (close >= bb_upper)
    4) Todas las condiciones se cumplen simultáneamente

LÓGICA DE SALIDA:
  - LONG: precio toca la banda media de Bollinger (close >= bb_basis)
  - SHORT: precio toca la banda media de Bollinger (close <= bb_basis)
  - Stop Loss de emergencia: fijo en % desde entrada (3%-20%)

Combina reversión estadística (Z-Score) con soporte/resistencia dinámico (BB).
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_zscore, cfg_bollinger


class Strategy7ZScoreBollingerReversal:
    """Z-Score Reversal + Bollinger Bands."""

    # Identificación única
    combinacion_id = 132
    name = "ZScore_Bollinger_Reversal"

    # Indicadores para el plot
    __indicators_used = ["zscore", "bb_basis", "bb_upper", "bb_lower"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Espacio de parámetros Optuna
    parametros_optuna = {
        # Z-Score
        "z_length": (10, 60, 5),                   # ventana Z-Score
        "z_threshold": (1.0, 3.0, 0.1),            # umbral extremo (simétrico ±)
        "reversal_bars": (1, 3, 1),                # velas de reversión (1-3)
        # Bollinger Bands
        "bb_period": (15, 30, 5),                  # período BB
        "bb_mult": (1.5, 2.5, 0.1),                # multiplicador BB
        # Stop Loss
        "emergency_sl_pct": (3.0, 20.0, 1.0),      # SL de emergencia (%)
    }

    TIMEOUT_BARS = 200

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_length = trial.suggest_int("z_length", 10, 60, step=5)
        z_threshold = trial.suggest_float("z_threshold", 1.0, 3.0, step=0.1)
        reversal_bars = trial.suggest_int("reversal_bars", 1, 3, step=1)
        bb_period = trial.suggest_int("bb_period", 15, 30, step=5)
        bb_mult = trial.suggest_float("bb_mult", 1.5, 2.5, step=0.1)
        emergency_sl_pct = trial.suggest_float("emergency_sl_pct", 3.0, 20.0, step=1.0)

        return {
            "z_length": int(z_length),
            "z_threshold": float(z_threshold),
            "reversal_bars": int(reversal_bars),
            "bb_period": int(bb_period),
            "bb_mult": float(bb_mult),
            "emergency_sl_pct": float(emergency_sl_pct),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        z_length = int(params.get("z_length", 20))
        z_threshold = float(params.get("z_threshold", 2.0))
        reversal_bars = int(params.get("reversal_bars", 2))
        bb_period = int(params.get("bb_period", 20))
        bb_mult = float(params.get("bb_mult", 2.0))

        # Warm-up robusto
        max_win = max(z_length, bb_period)
        params["__warmup_bars"] = max(max_win + 5, 50)

        # 1) Calcular indicadores
        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
            "bollinger_bands": cfg_bollinger(
                period=bb_period,
                mult=bb_mult,
                col="close",
                out_basis="bb_basis",
                out_upper="bb_upper",
                out_lower="bb_lower",
            ),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) Detectar señales con Numba
        z_arr = df["zscore"].to_numpy().astype(np.float64)
        close_arr = df["close"].to_numpy().astype(np.float64)
        bb_upper_arr = df["bb_upper"].to_numpy().astype(np.float64)
        bb_lower_arr = df["bb_lower"].to_numpy().astype(np.float64)

        signal_long, signal_short = _detect_zscore_bb_reversal(
            z_arr,
            close_arr,
            bb_upper_arr,
            bb_lower_arr,
            float(z_threshold),
            int(reversal_bars),
        )

        return df.with_columns([
            pl.Series("signal_long", signal_long),
            pl.Series("signal_short", signal_short),
        ])

    def decide_exit(
        self,
        df: pl.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        side: str,
        *,
        saldo_apertura: float,
    ) -> Optional[ExitDecision]:
        """Salida cuando precio toca banda media BB + SL de emergencia."""

        n = len(df)
        if entry_idx >= n - 1:
            return None

        is_long = side.upper() == "LONG"
        is_short = side.upper() == "SHORT"
        if not (is_long or is_short):
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="UNKNOWN_SIDE")

        emergency_sl_pct = float(params.get("emergency_sl_pct", 10.0))

        close_arr = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        bb_basis_arr = df["bb_basis"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, len(df) - entry_idx - 1)

        exit_idx, reason_code = _decide_exit_bb_basis(
            entry_idx,
            entry_price,
            close_arr,
            high_arr,
            low_arr,
            bb_basis_arr,
            emergency_sl_pct,
            1 if is_long else -1,
            max_bars,
        )

        reason_map = {
            0: None,
            1: "BB_BASIS_EXIT",        # precio toca banda media
            2: "EMERGENCY_SL",         # stop loss de emergencia
            3: "TIME_EXIT",            # timeout
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])

        fallback_idx = min(entry_idx + max_bars, n - 1)
        return ExitDecision(exit_idx=fallback_idx, reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _detect_zscore_bb_reversal(
    z: np.ndarray,
    close: np.ndarray,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    z_threshold: float,
    reversal_bars: int,
) -> tuple:
    """Detecta señales cuando Z-Score extremo revierte y precio está en banda BB.

    LONG:
      1) z < -z_threshold (extremo negativo)
      2) z sube durante reversal_bars velas consecutivas
      3) close <= bb_lower (precio en banda inferior)

    SHORT:
      1) z > +z_threshold (extremo positivo)
      2) z baja durante reversal_bars velas consecutivas
      3) close >= bb_upper (precio en banda superior)
    """

    n = len(z)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)

    lookback = max(reversal_bars, 3)

    for i in range(lookback, n):
        z_curr = z[i]
        close_curr = close[i]
        bb_u = bb_upper[i]
        bb_l = bb_lower[i]

        if np.isnan(z_curr) or np.isnan(close_curr) or np.isnan(bb_u) or np.isnan(bb_l):
            continue

        # === SEÑAL LONG ===
        # 1) Z-Score en extremo negativo
        if z_curr < -z_threshold:
            # 2) Verificar reversión: z sube durante reversal_bars velas
            reversal_ok = True
            for j in range(1, reversal_bars + 1):
                idx_curr = i - j + 1
                idx_prev = i - j
                if idx_prev < 0 or idx_curr < 0:
                    reversal_ok = False
                    break
                z_c = z[idx_curr]
                z_p = z[idx_prev]
                if np.isnan(z_c) or np.isnan(z_p):
                    reversal_ok = False
                    break
                if z_c <= z_p:  # debe subir estrictamente
                    reversal_ok = False
                    break
            
            # 3) Precio en banda inferior
            if reversal_ok and close_curr <= bb_l:
                signal_long[i] = True

        # === SEÑAL SHORT ===
        # 1) Z-Score en extremo positivo
        if z_curr > z_threshold:
            # 2) Verificar reversión: z baja durante reversal_bars velas
            reversal_ok = True
            for j in range(1, reversal_bars + 1):
                idx_curr = i - j + 1
                idx_prev = i - j
                if idx_prev < 0 or idx_curr < 0:
                    reversal_ok = False
                    break
                z_c = z[idx_curr]
                z_p = z[idx_prev]
                if np.isnan(z_c) or np.isnan(z_p):
                    reversal_ok = False
                    break
                if z_c >= z_p:  # debe bajar estrictamente
                    reversal_ok = False
                    break
            
            # 3) Precio en banda superior
            if reversal_ok and close_curr >= bb_u:
                signal_short[i] = True

    return signal_long, signal_short


@njit(cache=True, fastmath=True)
def _decide_exit_bb_basis(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bb_basis: np.ndarray,
    emergency_sl_pct: float,
    side_flag: int,
    max_bars: int,
) -> tuple:
    """Salida cuando precio toca banda media BB o SL de emergencia.

    LONG: cierra cuando close >= bb_basis (precio sube a media)
    SHORT: cierra cuando close <= bb_basis (precio baja a media)
    """

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    # SL de emergencia
    emergency_factor = emergency_sl_pct / 100.0
    if side_flag == 1:
        emergency_level = entry_price * (1.0 - emergency_factor)
    else:
        emergency_level = entry_price * (1.0 + emergency_factor)

    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        h = high[i]
        l = low[i]
        bb_mid = bb_basis[i]

        if np.isnan(c) or np.isnan(bb_mid):
            continue

        # 1) Emergency SL (prioridad)
        if side_flag == 1:
            if l <= emergency_level:
                return i, 2
        else:
            if h >= emergency_level:
                return i, 2

        # 2) Salida cuando precio toca banda media
        if side_flag == 1:
            # LONG: cierra cuando precio alcanza o supera la banda media
            if c >= bb_mid:
                return i, 1
        else:
            # SHORT: cierra cuando precio alcanza o baja de la banda media
            if c <= bb_mid:
                return i, 1

    # Timeout
    if last_allowed > entry_idx:
        return last_allowed, 3

    return -1, 0

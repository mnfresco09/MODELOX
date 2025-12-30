from __future__ import annotations

"""
Strategy 3 — Z-Score + Momentum Fusion (Exit Type B)

Estrategia 100% determinista basada en:
- Z-Score de precio (ventana configurable, por defecto 20)
- Momentum / ROC de precio (ventana configurable, por defecto 10)

Entradas:
  LONG:
    - Z-Score < -z_threshold  (sobreventa estadística)
    - Momentum actual > Momentum anterior (momento girando al alza)

  SHORT:
    - Z-Score > +z_threshold  (sobrecompra estadística)
    - Momentum actual < Momentum anterior (momento girando a la baja)

Salidas (Exit Type B - trend following):
  LONG:
    - Cerrar cuando Momentum empieza a bajar (mom_t < mom_{t-1})
      O cuando Z-Score toca la banda opuesta (+z_threshold).

  SHORT:
    - Cerrar cuando Momentum empieza a subir (mom_t > mom_{t-1})
      O cuando Z-Score toca la banda opuesta (-z_threshold).

Sin confirmaciones visuales, sin contexto subjetivo: todo se basa
únicamente en los valores de Z-Score y Momentum calculados bar-a-bar.
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_zscore, cfg_roc, cfg_atr


class Strategy3ZScoreMomentumFusion:
    """Z-Score + Momentum Fusion (Exit Type B)."""

    # Identificación única en el sistema
    combinacion_id = 3
    name = "ZScore_Momentum_Fusion_B"

    # Indicadores que queremos ver en el plot (paneles dinámicos)
    __indicators_used = ["zscore", "roc", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Espacio de parámetros Optuna (rango realista, modificable)
    # Formato: nombre -> (min, max, step)
    parametros_optuna = {
        "z_length": (10, 60, 2),       # ventana Z-Score (default ~20)
        "z_threshold": (1.0, 3.0, 0.1),  # |Z| para entrada/salida (default ~2.0)
        "roc_length": (5, 30, 1),      # ventana ROC/Momentum (default ~10)
        "atr_period": (10, 20, 2),     # período ATR para trailing (default ~14)
        "trailing_atr_mult": (1.5, 3.5, 0.1),  # múltiplo ATR para trailing stop
        "emergency_sl_pct": (5.0, 15.0, 0.5),  # stop loss de emergencia en % (5%-15%)
    }

    # Límite de seguridad absoluto
    TIMEOUT_BARS = 200

    # --------------------------------------------------------------
    # Sugerencia de parámetros para Optuna
    # --------------------------------------------------------------
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_length = trial.suggest_int("z_length", 10, 60, step=2)
        z_threshold = trial.suggest_float("z_threshold", 1.0, 3.0, step=0.1)
        roc_length = trial.suggest_int("roc_length", 5, 30, step=1)
        atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
        trailing_atr_mult = trial.suggest_float("trailing_atr_mult", 1.5, 3.5, step=0.1)
        emergency_sl_pct = trial.suggest_float("emergency_sl_pct", 5.0, 15.0, step=0.5)

        return {
            "z_length": int(z_length),
            "z_threshold": float(z_threshold),
            "roc_length": int(roc_length),
            "atr_period": int(atr_period),
            "trailing_atr_mult": float(trailing_atr_mult),
            "emergency_sl_pct": float(emergency_sl_pct),
        }

    # --------------------------------------------------------------
    # Generación de señales deterministas
    # --------------------------------------------------------------
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        z_length = int(params.get("z_length", 20))
        z_threshold = float(params.get("z_threshold", 2.0))
        roc_length = int(params.get("roc_length", 10))
        atr_period = int(params.get("atr_period", 14))

        # Warm-up global robusto: necesitamos al menos la ventana más larga
        max_win = max(z_length, roc_length, atr_period)
        params["__warmup_bars"] = max(max_win + 5, 50)

        # 1) Calcular Z-Score, ROC y ATR mediante IndicadorFactory
        ind_config = {
            "zscore": cfg_zscore(col="close", window=z_length, out="zscore"),
            "roc": cfg_roc(period=roc_length, col="close", out="roc"),
            "atr": cfg_atr(period=atr_period, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) Condiciones de entrada LONG/SHORT (bar-a-bar, 100% determinista)
        z_thr = z_threshold

        # Momento relativo (roc actual vs roc anterior)
        mom = pl.col("roc")
        mom_prev = pl.col("roc").shift(1)

        # Entrada LONG: Z < -z_thr y Momentum subiendo
        cond_long = (
            (pl.col("zscore") < -z_thr)
            & (mom > mom_prev)
        )

        # Entrada SHORT: Z > +z_thr y Momentum bajando
        cond_short = (
            (pl.col("zscore") > z_thr)
            & (mom < mom_prev)
        )

        signal_long = cond_long.fill_null(False)
        signal_short = cond_short.fill_null(False)

        return df.with_columns([
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short"),
        ])

    # --------------------------------------------------------------
    # Gestión de salidas (Trailing Stop + Emergency SL)
    # --------------------------------------------------------------
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
        """Aplica trailing stop basado en ATR y stop loss de emergencia porcentual.

        LONG:
          - Trailing Stop: precio cae por debajo de (max_price_seen - trailing_atr_mult * ATR)
          - Emergency SL: precio cae por debajo de (entry_price * (1 - emergency_sl_pct/100))

        SHORT:
          - Trailing Stop: precio sube por encima de (min_price_seen + trailing_atr_mult * ATR)
          - Emergency SL: precio sube por encima de (entry_price * (1 + emergency_sl_pct/100))
        """

        n = len(df)
        if entry_idx >= n - 1:
            return None

        is_long = side.upper() == "LONG"
        is_short = side.upper() == "SHORT"
        if not (is_long or is_short):
            # Cerrar en la siguiente barra cualquier otro tipo de posición
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="UNKNOWN_SIDE")

        trailing_atr_mult = float(params.get("trailing_atr_mult", 2.0))
        emergency_sl_pct = float(params.get("emergency_sl_pct", 10.0))

        close_arr = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        max_bars = min(self.TIMEOUT_BARS, len(df) - entry_idx - 1)

        exit_idx, reason_code = _decide_exit_trailing_sl(
            entry_idx,
            entry_price,
            close_arr,
            high_arr,
            low_arr,
            atr_arr,
            trailing_atr_mult,
            emergency_sl_pct,
            1 if is_long else -1,
            max_bars,
        )

        reason_map = {
            0: None,
            1: "TRAILING_STOP",       # trailing stop activado
            2: "EMERGENCY_SL",        # stop loss de emergencia
            3: "TIME_EXIT",           # timeout de seguridad
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])

        # Fallback de seguridad
        fallback_idx = min(entry_idx + max_bars, n - 1)
        return ExitDecision(exit_idx=fallback_idx, reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _decide_exit_trailing_sl(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    trailing_atr_mult: float,
    emergency_sl_pct: float,
    side_flag: int,
    max_bars: int,
) -> tuple:
    """Kernel Numba para trailing stop basado en ATR y SL de emergencia.

    side_flag: +1 para LONG, -1 para SHORT.

    Prioridad:
      1) Emergency SL (stop fijo en % desde entrada)
      2) Trailing Stop (basado en ATR desde máximo/mínimo favorable)
      3) TIMEOUT tras max_bars
    """

    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)

    # Calcular niveles de stop de emergencia
    emergency_factor = emergency_sl_pct / 100.0
    if side_flag == 1:
        # LONG: SL de emergencia por debajo de entrada
        emergency_level = entry_price * (1.0 - emergency_factor)
    else:
        # SHORT: SL de emergencia por encima de entrada
        emergency_level = entry_price * (1.0 + emergency_factor)

    # Tracking del mejor precio alcanzado y nivel de trailing stop
    if side_flag == 1:
        best_price = entry_price  # para LONG, seguimos el máximo
        trailing_stop_level = entry_price - trailing_atr_mult * atr[entry_idx] if not np.isnan(atr[entry_idx]) else entry_price * 0.95
    else:
        best_price = entry_price  # para SHORT, seguimos el mínimo
        trailing_stop_level = entry_price + trailing_atr_mult * atr[entry_idx] if not np.isnan(atr[entry_idx]) else entry_price * 1.05

    for i in range(entry_idx + 1, last_allowed + 1):
        c = close[i]
        h = high[i]
        l = low[i]
        atr_val = atr[i]

        if np.isnan(c) or np.isnan(atr_val):
            continue

        # 1) Emergency Stop Loss (prioridad máxima)
        if side_flag == 1:
            # LONG: cerrar si el precio cae por debajo del SL de emergencia
            if l <= emergency_level:
                return i, 2
        else:
            # SHORT: cerrar si el precio sube por encima del SL de emergencia
            if h >= emergency_level:
                return i, 2

        # 2) Actualizar mejor precio alcanzado y trailing stop vela a vela
        if side_flag == 1:
            # LONG: actualizamos máximo y subimos el trailing stop
            if h > best_price:
                best_price = h
                # Trailing stop sube (nunca baja) siguiendo al precio
                new_trailing = best_price - trailing_atr_mult * atr_val
                if new_trailing > trailing_stop_level:
                    trailing_stop_level = new_trailing
        else:
            # SHORT: actualizamos mínimo y bajamos el trailing stop
            if l < best_price:
                best_price = l
                # Trailing stop baja (nunca sube) siguiendo al precio
                new_trailing = best_price + trailing_atr_mult * atr_val
                if new_trailing < trailing_stop_level:
                    trailing_stop_level = new_trailing

        # 3) Trailing Stop: cerrar si el precio toca el nivel de trailing
        if side_flag == 1:
            # LONG: trailing stop por debajo del máximo alcanzado (solo sube, nunca baja)
            if l <= trailing_stop_level:
                return i, 1
        else:
            # SHORT: trailing stop por encima del mínimo alcanzado (solo baja, nunca sube)
            if h >= trailing_stop_level:
                return i, 1

    # 4) TIMEOUT si no se ha salido antes
    if last_allowed > entry_idx:
        return last_allowed, 3

    return -1, 0

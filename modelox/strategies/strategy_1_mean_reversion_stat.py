from __future__ import annotations

"""
Estrategia 1 — Mean Reversion Estadística (BTC · 5m)

Hipótesis:
- En ausencia de tendencia fuerte (ADX bajo), el precio se sobreextiende
  en términos de Z-score y tiende a revertir parcialmente hacia su media.

Indicadores:
- EMA 50 (close)       -> media de referencia
- Z-score(50)         -> (close - media) / std
- ADX(14)             -> régimen (tendencia vs. rango)
- ATR(14) + ATR p90   -> filtro de volatilidad y stop por volatilidad

Entradas:
- LONG:  ADX < adx_max  AND  Z <= -z_entry  AND  rango < max_range_mult * ATR
- SHORT: ADX < adx_max  AND  Z >= +z_entry  AND  rango < max_range_mult * ATR
- Además, se bloquea si ATR > p90(ATR 200) o hay vela tipo "news spike".

Salidas (en este orden de prioridad):
1) Salida estadística principal (edge agotado):
   - LONG:  Z >= -z_exit
   - SHORT: Z <= +z_exit
2) Stop por volatilidad (protección):
   - LONG:  precio <= entrada - atr_stop_mult * ATR_entrada
   - SHORT: precio >= entrada + atr_stop_mult * ATR_entrada
3) Stop estadístico extremo (fail-safe de régimen roto):
   - LONG:  Z <= -z_extreme
   - SHORT: Z >= +z_extreme
4) Salida anticipada por falta de reversión (early exit):
   - Si tras early_exit_bars velas Z sigue demasiado extremo
5) Time stop duro: máximo time_stop_bars velas por trade.

NOTA: El sizing de posición (volatility targeting Size = Capital * Riesgo / ATR)
no se implementa aquí porque el motor de ejecución gestiona el tamaño de
posición de forma global (apalancamiento fijo + QTY_MAX_ACTIVO). Esta 
estrategia se limita a las señales y salidas.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import (
    cfg_zscore,
    cfg_atr,
    cfg_adx_numba,
)


class Strategy1MeanReversionStat:
    """Mean Reversion Estadística basada en Z-score + ADX + ATR."""

    # Identificación
    combinacion_id = 1
    name = "MeanReversion_Z_ADX_ATR"

    # Indicadores que se quieren ver en el plot
    __indicators_used = ["zscore", "adx", "atr", "atr_p90"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Rango de parámetros para Optuna (alrededor de los valores recomendados)
    parametros_optuna = {
        # Z-score de entrada/salida
        "z_entry": (1.8, 2.5, 0.1),      # umbral de entrada (≈2.0)
        "z_exit": (0.3, 0.7, 0.1),       # salida principal (≈0.5)
        "z_extreme": (2.8, 3.4, 0.1),    # fail-safe (≈3.0)
        # Filtros de régimen / volatilidad
        "adx_max": (20.0, 28.0, 1.0),    # máximo ADX para considerar rango (≈25)
        "max_range_mult": (1.3, 1.7, 0.1),  # rango máx relativo al ATR (≈1.5)
        # Stop por volatilidad y gestión temporal
        "atr_stop_mult": (1.8, 2.2, 0.1),  # múltiplo ATR para stop (≈2.0)
        "time_stop_bars": (12, 15, 1),     # time stop duro (12–15 velas)
        "early_exit_bars": (6, 8, 1),      # early exit si no mejora
        "early_target_z": (0.8, 1.2, 0.1), # objetivo de mejora parcial (≈1.0)
    }

    # Límite de seguridad absoluto en caso de que falle la lógica
    TIMEOUT_BARS = 60

    # ---------------------------------------------------------------------
    # Sugerencia de parámetros para Optuna
    # ---------------------------------------------------------------------
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_entry = trial.suggest_float("z_entry", 1.8, 2.5, step=0.1)
        z_exit = trial.suggest_float("z_exit", 0.3, 0.7, step=0.1)
        z_extreme = trial.suggest_float("z_extreme", 2.8, 3.4, step=0.1)

        adx_max = trial.suggest_float("adx_max", 20.0, 28.0, step=1.0)
        max_range_mult = trial.suggest_float("max_range_mult", 1.3, 1.7, step=0.1)

        atr_stop_mult = trial.suggest_float("atr_stop_mult", 1.8, 2.2, step=0.1)
        time_stop_bars = trial.suggest_int("time_stop_bars", 12, 15, step=1)
        early_exit_bars = trial.suggest_int("early_exit_bars", 6, 8, step=1)
        early_target_z = trial.suggest_float("early_target_z", 0.8, 1.2, step=0.1)

        return {
            "z_entry": z_entry,
            "z_exit": z_exit,
            "z_extreme": z_extreme,
            "adx_max": adx_max,
            "max_range_mult": max_range_mult,
            "atr_stop_mult": atr_stop_mult,
            "time_stop_bars": time_stop_bars,
            "early_exit_bars": early_exit_bars,
            "early_target_z": early_target_z,
        }

    # ---------------------------------------------------------------------
    # Generación de señales
    # ---------------------------------------------------------------------
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        # 1) Calcular indicadores base
        ind_config = {
            # Z-score 50 barras sobre close (usa internamente una media móvil)
            "zscore": cfg_zscore(col="close", window=50, out="zscore"),
            # ADX 14 (con +DI/-DI disponibles si se requieren en el futuro)
            "adx": cfg_adx_numba(period=14, out_adx="adx", out_plus_di="plus_di", out_minus_di="minus_di"),
            # ATR 14
            "atr": cfg_atr(period=14, out="atr"),
        }
        df = IndicadorFactory.procesar(df, ind_config)

        # 2) ATR percentil 90 (últimas 200 barras) para filtro de volatilidad
        df = df.with_columns(
            pl.col("atr")
            .rolling_quantile(
                quantile=0.9,
                window_size=200,
                min_periods=1,
                interpolation="nearest",
            )
            .alias("atr_p90")
        )

        # 3) Filtros y condiciones de entrada
        z_entry = float(params["z_entry"])
        adx_max = float(params["adx_max"])
        max_range_mult = float(params["max_range_mult"])

        price_range = pl.col("high") - pl.col("low")

        # Condiciones de "no operar" (hard filters)
        cond_adx_high = pl.col("adx") >= adx_max
        cond_atr_high = pl.col("atr") > pl.col("atr_p90")
        cond_spike = price_range >= max_range_mult * pl.col("atr")

        no_trade = (cond_adx_high | cond_atr_high | cond_spike).fill_null(False)

        # Condiciones base de entrada LONG/SHORT (antes del filtro no_trade)
        cond_long_base = (
            (pl.col("adx") < adx_max)
            & (pl.col("zscore") <= -z_entry)
            & (price_range < max_range_mult * pl.col("atr"))
        )

        cond_short_base = (
            (pl.col("adx") < adx_max)
            & (pl.col("zscore") >= z_entry)
            & (price_range < max_range_mult * pl.col("atr"))
        )

        signal_long = (cond_long_base & (~no_trade)).fill_null(False)
        signal_short = (cond_short_base & (~no_trade)).fill_null(False)

        return df.with_columns([
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short"),
        ])

    # ---------------------------------------------------------------------
    # Gestión de salidas
    # ---------------------------------------------------------------------
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
        """Decide la salida según reglas estadísticas + volatilidad + tiempo."""

        n = len(df)
        if entry_idx >= n - 1:
            return None

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        z_arr = df["zscore"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        is_long = 1 if side.upper() == "LONG" else -1

        z_exit = float(params.get("z_exit", 0.5))
        z_extreme = float(params.get("z_extreme", 3.0))
        atr_stop_mult = float(params.get("atr_stop_mult", 2.0))
        time_stop_bars = int(params.get("time_stop_bars", 13))
        early_exit_bars = int(params.get("early_exit_bars", 7))
        early_target_z = float(params.get("early_target_z", 1.0))

        max_bars = min(self.TIMEOUT_BARS, time_stop_bars + 5)

        exit_idx, reason_code = _decide_exit_numba(
            entry_idx,
            entry_price,
            close,
            high,
            low,
            z_arr,
            atr_arr,
            is_long,
            z_exit,
            z_extreme,
            atr_stop_mult,
            time_stop_bars,
            early_exit_bars,
            early_target_z,
            max_bars,
        )

        reason_map = {
            0: None,
            1: "Z_EXIT",              # salida estadística principal
            2: "ATR_STOP",            # stop por volatilidad
            3: "Z_EXTREME_FAILSAFE",  # ruptura de régimen
            4: "EARLY_EXIT",          # no hay reversión suficiente
            5: "TIME_EXIT",           # time stop duro
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])

        # Fallback de seguridad
        timeout_idx = min(entry_idx + self.TIMEOUT_BARS, n - 1)
        return ExitDecision(exit_idx=timeout_idx, reason="TIME_EXIT")


@njit(cache=True, fastmath=True)
def _decide_exit_numba(
    entry_idx: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    zscore: np.ndarray,
    atr: np.ndarray,
    is_long: int,  # 1 para LONG, -1 para SHORT
    z_exit: float,
    z_extreme: float,
    atr_stop_mult: float,
    time_stop_bars: int,
    early_exit_bars: int,
    early_target_z: float,
    max_bars: int,
) -> Tuple[int, int]:
    """Kernel Numba para aplicar las reglas de salida bar-a-bar.

    Prioridad de reglas (la primera que se cumpla cierra el trade):
    1) Salida estadística principal (Z cerca de 0)
    2) Stop por volatilidad (2x ATR de entrada)
    3) Stop estadístico extremo (Z muy extremo)
    4) Salida anticipada si no hay mejora en early_exit_bars
    5) Time stop duro (time_stop_bars)
    """

    n = len(close)
    if entry_idx >= n - 1:
        return -1, 0

    # ATR en la barra de entrada
    atr_entry = atr[entry_idx] if 0 <= entry_idx < n else np.nan
    if np.isnan(atr_entry) or atr_entry <= 0.0:
        atr_entry = max(entry_price * 0.005, 1e-8)

    # Stop por volatilidad fijo en precio
    if is_long == 1:
        vol_stop = entry_price - atr_stop_mult * atr_entry
    else:
        vol_stop = entry_price + atr_stop_mult * atr_entry

    last_idx = min(n - 1, entry_idx + max_bars)

    for i in range(entry_idx + 1, last_idx + 1):
        z = zscore[i]
        if np.isnan(z):
            z = 0.0

        elapsed = i - entry_idx

        if is_long == 1:
            # 1) Salida estadística principal (Z >= -z_exit)
            if z >= -z_exit:
                return i, 1

            # 2) Stop por volatilidad (precio <= entrada - k * ATR_entrada)
            if low[i] <= vol_stop:
                return i, 2

            # 3) Fail-safe extremo (Z <= -z_extreme)
            if z <= -z_extreme:
                return i, 3

            # 4) Early exit si, tras early_exit_bars, Z sigue demasiado extremo
            if elapsed >= early_exit_bars and z <= -early_target_z:
                return i, 4

        else:  # SHORT
            # 1) Salida estadística principal (Z <= +z_exit)
            if z <= z_exit:
                return i, 1

            # 2) Stop por volatilidad (precio >= entrada + k * ATR_entrada)
            if high[i] >= vol_stop:
                return i, 2

            # 3) Fail-safe extremo (Z >= +z_extreme)
            if z >= z_extreme:
                return i, 3

            # 4) Early exit si, tras early_exit_bars, Z sigue demasiado extremo
            if elapsed >= early_exit_bars and z >= early_target_z:
                return i, 4

        # 5) Time stop duro
        if elapsed >= time_stop_bars:
            return i, 5

    # Si no se ha disparado nada, cerrar por timeout en last_idx
    return last_idx, 5

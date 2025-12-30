from __future__ import annotations

"""
Estrategia 77 (ID 77): Z-Score + RSI Cruce Estricto (Strict Timing)
-------------------------------------------------------------------
Lógica Secuencial Precisa:
1. INICIO: Z-Score rompe el umbral (entra en zona extrema). Esto abre la "Ventana".
2. VENTANA: Dura N velas (param 'window_wait').
3. DISPARADOR (Trigger): Dentro de esa ventana, el RSI debe CRUZAR el nivel.
   - Long: RSI cruza hacia ARRIBA su nivel (recuperación de momentum).
   - Short: RSI cruza hacia ABAJO su nivel.
4. INVALIDACIÓN: Si la ventana cierra sin cruce de RSI, se ignora todo hasta que
   el Z-Score se resetee y vuelva a romper el nivel en el futuro.

Salida: Únicamente Trailing Stop (Maximización de tendencia) + SL Emergencia.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision

class Strategy77ZScoreRsiStrict:
    # Modular indicator declaration for plotting and reporting
    # Update this list with all indicator columns used by this strategy
    __indicators_used = ["zscore", "rsi"]
    combinacion_id = 77
    name = "ZScore_RSI_Strict_Cross"

    # Definición del espacio de búsqueda (Optuna)
    parametros_optuna = {
        # -- Configuración Z-Score (El Setup) --
        "z_period": (20, 30, 5),
        "z_threshold": (1.5, 2.5, 0.2), 
        
        # -- Configuración RSI (El Trigger) --
        "rsi_period": (14, 14, 1),      
        "rsi_threshold": (25, 47, 1),   # Nivel de sobreventa para cruzar
        
        # -- Ventana de Oportunidad Estricta --
        # Cuántas velas dura la oportunidad desde que Z-Score se pone extremo
        "window_wait": (2, 5, 1),       
        
        # -- Gestión de Salida --
        "trailing_pct": (0.005, 0.04, 0.005), # Trailing dinámico
        "sl_pct": (0.04, 0.12, 0.005),        # Stop Fijo de Emergencia
    }

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "z_period": trial.suggest_int("z_period", 20, 60, step=5),
            "z_threshold": trial.suggest_float("z_threshold", 1.8, 3.0, step=0.1),
            "rsi_period": trial.suggest_int("rsi_period", 14, 14),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 25, 35),
            "window_wait": trial.suggest_int("window_wait", 2, 5),
            "trailing_pct": trial.suggest_float("trailing_pct", 0.005, 0.04, step=0.005),
            "sl_pct": trial.suggest_float("sl_pct", 0.01, 0.05, step=0.005),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # For modular reporting/plotting, always set params_reporting with indicators used
        params["__indicators_used"] = self.get_indicators_used()
        """
        Genera señales con lógica de CONFLUENCIA y fuerza la presencia de columnas de indicadores y líneas de referencia.
        """
        # Calcular indicadores principales
        cfg_indicadores = {
            "zscore": {"activo": True, "col": "close", "window": params["z_period"], "out": "zscore"},
            "rsi": {"activo": True, "col": "close", "period": params["rsi_period"], "out": "rsi"}
        }
        df = IndicadorFactory.procesar(df, cfg_indicadores)

        z_thresh = params["z_threshold"]
        rsi_thresh = params["rsi_threshold"]
        window_bars = params["window_wait"]
        rsi_threshold_overbought = 100 - rsi_thresh

        # Detectar eventos de ventana
        z_start_long = ((pl.col("zscore") < -z_thresh) & (pl.col("zscore").shift(1) >= -z_thresh)).fill_null(False)
        z_start_short = ((pl.col("zscore") > z_thresh) & (pl.col("zscore").shift(1) <= z_thresh)).fill_null(False)

        valid_long = (z_start_long.cast(pl.Int8).rolling_max(window_size=window_bars, min_periods=1).fill_null(0) > 0)
        valid_short = (z_start_short.cast(pl.Int8).rolling_max(window_size=window_bars, min_periods=1).fill_null(0) > 0)

        rsi_cross_long = ((pl.col("rsi").shift(1) < rsi_thresh) & (pl.col("rsi") >= rsi_thresh)).fill_null(False)
        rsi_cross_short = ((pl.col("rsi").shift(1) > rsi_threshold_overbought) & (pl.col("rsi") <= rsi_threshold_overbought)).fill_null(False)

        signal_long = valid_long & rsi_cross_long
        signal_short = valid_short & rsi_cross_short

        # Forzar columnas de referencia para el plot modular
        df = df.with_columns([
            pl.col("zscore").alias("zscore"),
            pl.col("rsi").alias("rsi"),
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short")
        ])
        return df

    def decide_exit(
        self, 
        df: pl.DataFrame, 
        params: Dict[str, Any], 
        entry_idx: int, 
        entry_price: float, 
        side: str, 
        *, 
        saldo_apertura: float
    ) -> Optional[ExitDecision]:
        """ Wrapper Python para Salida Numba (Trailing + SL) """
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        direction = 1 if side == "LONG" else -1
        
        max_bars = len(close) # Sin Time Stop lógico, solo límite de array
        
        exit_idx, reason_code = self._decide_exit_numba(
            entry_price, entry_idx, high, low, close, direction,
            params["trailing_pct"], params["sl_pct"], max_bars
        )

        if exit_idx == -1:
            return None

        reason_map = {1: "STOP_LOSS", 2: "TRAILING_STOP"}
        return ExitDecision(exit_idx=exit_idx, reason=reason_map.get(reason_code, "UNKNOWN"))

    @staticmethod
    @njit
    def _decide_exit_numba(
        entry_price: float, 
        entry_idx: int, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        direction: int, 
        trailing_pct: float, 
        sl_pct: float,
        safety_limit: int 
    ) -> Tuple[int, int]:
        """
        Lógica de Salida Numba: SOLO Trailing Stop y Stop Loss Fijo.
        """
        n = len(close)
        end_search = min(entry_idx + safety_limit, n)
        
        peak_price = entry_price
        
        for i in range(entry_idx + 1, end_search):
            curr_high = high[i]
            curr_low = low[i]
            
            if direction == 1: # LONG
                # Actualizar Trailing (Peak High)
                if curr_high > peak_price:
                    peak_price = curr_high
                
                trailing_level = peak_price * (1 - trailing_pct)
                hard_stop_level = entry_price * (1 - sl_pct)
                
                # El stop activo es el MAYOR (más ajustado al precio)
                effective_stop = max(trailing_level, hard_stop_level)

                if curr_low <= effective_stop:
                    reason = 2 if trailing_level > hard_stop_level else 1
                    return (i, reason)

            elif direction == -1: # SHORT
                # Actualizar Trailing (Peak Low)
                if curr_low < peak_price:
                    peak_price = curr_low
                
                trailing_level = peak_price * (1 + trailing_pct)
                hard_stop_level = entry_price * (1 + sl_pct)

                # El stop activo es el MENOR
                effective_stop = min(trailing_level, hard_stop_level)

                if curr_high >= effective_stop:
                    reason = 2 if trailing_level < hard_stop_level else 1
                    return (i, reason)

        return (-1, 0)
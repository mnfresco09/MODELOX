from __future__ import annotations

"""
Estrategia 88 (ID 88): Z-Score Range + MACD Persistent Reversal
-------------------------------------------------------------------
Lógica de Entrada (Persistencia):
1. Z-SCORE RANGO: El precio no debe estar ni "poco barato" ni "catastróficamente barato". 
   Debe mantenerse en un "Sweet Spot" (ej. -2.5 a -1.5) durante N velas.
2. MACD REVERSAL: El MACD debe estar revirtiendo (creciendo en Long) durante esas mismas N velas.
3. FILTRO: MACD debe estar por debajo de 0 (tendencia bajista de fondo) para Longs.

Lógica de Salida:
- Stop Loss de Emergencia (Fijo).
- Trailing Stop Dinámico (Persigue al precio para maximizar ganancia).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import polars as pl
from numba import njit

from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision

class Strategy88ZScoreMACDPersistent:
    """
    Estrategia de Reversión a la Media con Confirmación de Momentum Persistente.
    """
    
    # ========== IDENTIFICACIÓN ==========
    combinacion_id = 88  # ID Único
    name = "ZScore_MACD_Persistent"
    
    # ========== PARÁMETROS OPTUNA ==========
    parametros_optuna = {
        # -- Configuración Z-Score --
        "z_len": (20, 60, 5),
        
        # Rangos LONG (negativos)
        "z_min_long": (-3.0, -2.2, 0.1), # Límite inferior (evitar crash)
        "z_max_long": (-2.0, -1.0, 0.1), # Límite superior (entrada temprana)
        
        # Rangos SHORT (positivos)
        "z_min_short": (1.0, 2.0, 0.1),
        "z_max_short": (2.2, 3.0, 0.1),
        
        # -- Configuración MACD --
        "reversal_candles": (2, 4, 1),   # Cuántas velas debe durar la confirmación
        
        # -- Gestión de Salida --
        "sl_pct": (0.02, 0.10, 0.005),        # Stop de Emergencia (2% a 10%)
        "trailing_pct": (0.01, 0.05, 0.005),  # Trailing Stop (1% a 5%)
    }

    # ========== SUGERIR PARÁMETROS ==========
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        z_len = trial.suggest_int("z_len", 20, 60, step=5)
        
        # Long Rango: Aseguramos que min < max
        z_min_long = trial.suggest_float("z_min_long", -3.0, -2.1, step=0.1)
        z_max_long = trial.suggest_float("z_max_long", -2.0, -1.0, step=0.1)
        if z_min_long >= z_max_long:
            z_min_long = z_max_long - 0.5

        # Short Rango: Aseguramos que min < max
        z_min_short = trial.suggest_float("z_min_short", 1.0, 2.0, step=0.1)
        z_max_short = trial.suggest_float("z_max_short", 2.1, 3.0, step=0.1)
        if z_min_short >= z_max_short:
            z_max_short = z_min_short + 0.5
            
        return {
            "z_len": z_len,
            "z_min_long": z_min_long,
            "z_max_long": z_max_long,
            "z_min_short": z_min_short,
            "z_max_short": z_max_short,
            "reversal_candles": trial.suggest_int("reversal_candles", 2, 4),
            "sl_pct": trial.suggest_float("sl_pct", 0.02, 0.10, step=0.01),
            "trailing_pct": trial.suggest_float("trailing_pct", 0.01, 0.05, step=0.005),
        }

    # ========== GENERAR SEÑALES (POLARS VECTORIZADO) ==========
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Genera señales verificando la persistencia de las condiciones durante N velas.
        """
        # 1. Calcular Indicadores
        cfg = {
            "zscore": {"activo": True, "col": "close", "window": params["z_len"], "out": "z"},
            "macd":   {"activo": True, "col": "close", "fast": 12, "slow": 26, "signal": 9, "out_macd": "macd", "out_signal": "sig", "out_hist": "hist"}
        }
        df = IndicadorFactory.procesar(df, cfg)
        
        N = params["reversal_candles"]
        
        # --- LÓGICA LONG ---
        # A. Condición de Rango Z-Score (Vela a Vela)
        cond_z_long = (pl.col("z") > params["z_min_long"]) & (pl.col("z") < params["z_max_long"])
        
        # B. Condición MACD Creciendo y Negativo (Vela a Vela)
        # Nota: macd > macd.shift(1) implica crecimiento
        cond_macd_long = (pl.col("macd") < 0) & (pl.col("macd") > pl.col("macd").shift(1))
        
        # C. Persistencia (Rolling Sum)
        # Queremos que AMBAS condiciones (Z en rango Y MACD subiendo) hayan sido True
        # durante las últimas N velas consecutivas.
        combined_long = cond_z_long & cond_macd_long
        signal_long = combined_long.rolling_sum(window_size=N).fill_null(0) == N
        
        
        # --- LÓGICA SHORT ---
        # A. Condición de Rango Z-Score
        cond_z_short = (pl.col("z") > params["z_min_short"]) & (pl.col("z") < params["z_max_short"])
        
        # B. Condición MACD Cayendo y Positivo
        cond_macd_short = (pl.col("macd") > 0) & (pl.col("macd") < pl.col("macd").shift(1))
        
        # C. Persistencia
        combined_short = cond_z_short & cond_macd_short
        signal_short = combined_short.rolling_sum(window_size=N).fill_null(0) == N
        
        return df.with_columns([
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short")
        ])

    # ========== SALIDA (NUMBA KERNEL) ==========
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
        """
        Gestiona la salida usando un kernel Numba para máxima velocidad.
        Aplica Stop Loss de Emergencia y Trailing Stop.
        """
        # Preparar arrays numpy para Numba
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        
        direction = 1 if side.upper() == "LONG" else -1
        max_bars = len(close) # Límite técnico
        
        # Llamar al Kernel optimizado
        exit_idx, reason_code = self._exit_logic_numba(
            entry_price, entry_idx, high, low, direction,
            params["trailing_pct"], params["sl_pct"], max_bars
        )

        if exit_idx == -1:
            return None

        # Mapeo de códigos a razones legibles
        reason_map = {1: "STOP_LOSS_EMERGENCY", 2: "TRAILING_STOP"}
        return ExitDecision(exit_idx=exit_idx, reason=reason_map.get(reason_code, "TIME_EXIT"))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _exit_logic_numba(
        entry_price: float, 
        entry_idx: int, 
        high: np.ndarray, 
        low: np.ndarray, 
        direction: int, 
        trailing_pct: float, 
        sl_pct: float,
        safety_limit: int 
    ) -> Tuple[int, int]:
        """
        Kernel Numba para cálculo preciso de Trailing Stop y SL.
        Retorna: (indice_salida, codigo_razon)
        Codigos: 0=No Salida, 1=SL Emergencia, 2=Trailing Stop
        """
        n = len(high)
        end_search = min(entry_idx + safety_limit, n)
        
        # Inicializamos el "pico" extremo alcanzado con el precio de entrada
        extreme_price = entry_price
        
        for i in range(entry_idx + 1, end_search):
            curr_high = high[i]
            curr_low = low[i]
            
            # --- LÓGICA LONG ---
            if direction == 1:
                # 1. Actualizar Trailing: Buscamos el High más alto desde la entrada
                if curr_high > extreme_price:
                    extreme_price = curr_high
                
                # 2. Calcular Niveles
                trailing_price = extreme_price * (1.0 - trailing_pct)
                sl_price = entry_price * (1.0 - sl_pct)
                
                # El stop efectivo es el mayor de los dos (el más cercano al precio)
                effective_stop = trailing_price if trailing_price > sl_price else sl_price
                stop_reason = 2 if trailing_price > sl_price else 1
                
                # 3. Verificar si el precio tocó el stop (Low cruza abajo)
                if curr_low <= effective_stop:
                    return i, stop_reason

            # --- LÓGICA SHORT ---
            else:
                # 1. Actualizar Trailing: Buscamos el Low más bajo desde la entrada
                if curr_low < extreme_price:
                    extreme_price = curr_low
                
                # 2. Calcular Niveles
                trailing_price = extreme_price * (1.0 + trailing_pct)
                sl_price = entry_price * (1.0 + sl_pct)
                
                # El stop efectivo es el menor de los dos (el más cercano al precio)
                effective_stop = trailing_price if trailing_price < sl_price else sl_price
                stop_reason = 2 if trailing_price < sl_price else 1

                # 3. Verificar si el precio tocó el stop (High cruza arriba)
                if curr_high >= effective_stop:
                    return i, stop_reason

        return -1, 0
"""
================================================================================
MODELOX STRATEGY TEMPLATE
================================================================================
Autor: [Tu Nombre]
Fecha: [YYYY-MM-DD]
ID:    [NÚMERO ÚNICO - verificar que no existe en otras estrategias]

DESCRIPCIÓN:
[Descripción breve de la lógica de la estrategia]

LÓGICA DE ENTRADA:
- LONG:  [Condiciones para entrada larga]
- SHORT: [Condiciones para entrada corta]

LÓGICA DE SALIDA:
- LONG:  [Condiciones para cerrar posición larga]
- SHORT: [Condiciones para cerrar posición corta]

INDICADORES UTILIZADOS:
- [Indicador 1]: [Parámetros optimizables]
- [Indicador 2]: [Parámetros optimizables]

NOTAS:
[Cualquier consideración especial]
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

# ============================================================================
# IMPORTS DEL SISTEMA MODELOX
# ============================================================================
from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision

# Helpers para configurar indicadores (opcional, pero recomendado)
from modelox.strategies.indicator_specs import (
    cfg_rsi,
    cfg_macd,
    cfg_stc,
    cfg_adx,
    cfg_zscore,
    cfg_supertrend,
    cfg_ema,
    cfg_mfi,
    cfg_vwma,
    # ... importar los que necesites
)


# ============================================================================
# NUMBA KERNELS (Opcional - para lógica compleja de alta performance)
# ============================================================================
# Usa @njit cuando necesites iterar sobre arrays grandes bar-a-bar.
# Para lógica simple, usa expresiones vectorizadas de Polars directamente.

@njit(cache=True, fastmath=True)
def _detect_signals_kernel(
    indicator1: np.ndarray,
    indicator2: np.ndarray,
    threshold1: float,
    threshold2: float,
) -> tuple:
    """
    Kernel Numba para detección de señales de alta performance.
    
    Args:
        indicator1: Array numpy del indicador 1
        indicator2: Array numpy del indicador 2
        threshold1: Umbral para indicador 1
        threshold2: Umbral para indicador 2
        
    Returns:
        (signal_long, signal_short) - arrays booleanos
    """
    n = len(indicator1)
    signal_long = np.zeros(n, dtype=np.bool_)
    signal_short = np.zeros(n, dtype=np.bool_)
    
    # Necesitamos al menos N barras previas para lookback
    lookback = 3
    
    for i in range(lookback, n):
        # Skip si hay NaN
        if np.isnan(indicator1[i]) or np.isnan(indicator2[i]):
            continue
        
        # ===== EJEMPLO: CONDICIONES LONG =====
        # Reemplazar con tu lógica específica
        cond_long_1 = indicator1[i] < threshold1
        cond_long_2 = indicator2[i] > indicator2[i-1]  # Subiendo
        
        if cond_long_1 and cond_long_2:
            signal_long[i] = True
        
        # ===== EJEMPLO: CONDICIONES SHORT =====
        cond_short_1 = indicator1[i] > threshold2
        cond_short_2 = indicator2[i] < indicator2[i-1]  # Bajando
        
        if cond_short_1 and cond_short_2:
            signal_short[i] = True
    
    return signal_long, signal_short


@njit(cache=True, fastmath=True)
def _find_exit_kernel(
    indicator: np.ndarray,
    entry_idx: int,
    is_long: bool,
    exit_threshold: float,
    max_bars: int,
) -> tuple:
    """
    Kernel Numba para detección de salidas.
    
    Args:
        indicator: Array numpy del indicador de salida
        entry_idx: Índice de la barra de entrada
        is_long: True si es posición LONG, False si SHORT
        exit_threshold: Umbral para salir
        max_bars: Máximo de barras antes de timeout
        
    Returns:
        (exit_idx, reason_code)
        reason_code: 0=no encontrado, 1=threshold, 2=reversal, 3=timeout
    """
    n = len(indicator)
    end_idx = min(entry_idx + max_bars, n)
    
    for i in range(entry_idx + 1, end_idx):
        val = indicator[i]
        
        if np.isnan(val):
            continue
        
        # ===== SALIDA POR UMBRAL =====
        if is_long:
            if val >= exit_threshold:
                return i, 1  # EXIT_THRESHOLD
        else:
            if val <= exit_threshold:
                return i, 1  # EXIT_THRESHOLD
        
        # ===== SALIDA POR REVERSIÓN DE MOMENTUM =====
        # Ejemplo: N barras consecutivas en contra
        if i >= entry_idx + 3:
            v1 = indicator[i - 1]
            v2 = indicator[i - 2]
            
            if not (np.isnan(v1) or np.isnan(v2)):
                if is_long:
                    # LONG: salir si indicador cae 3 barras
                    if (val < v1) and (v1 < v2):
                        return i, 2  # MOMENTUM_REVERSAL
                else:
                    # SHORT: salir si indicador sube 3 barras
                    if (val > v1) and (v1 > v2):
                        return i, 2  # MOMENTUM_REVERSAL
    
    # Timeout
    if end_idx > entry_idx + 1:
        return end_idx - 1, 3  # TIME_EXIT
    
    return -1, 0


# ============================================================================
# CLASE DE ESTRATEGIA
# ============================================================================

class EstrategiaBase:
    """
    [NOMBRE DE LA ESTRATEGIA]
    
    Descripción breve de qué hace la estrategia.
    
    Atributos de clase requeridos:
        combinacion_id: ID único de la estrategia (int > 0)
        name: Nombre corto para reportes/archivos
        parametros_optuna: Diccionario con rangos de parámetros
        TIMEOUT_BARS: Máximo de barras antes de salida forzada
    """
    
    # ========== IDENTIFICACIÓN (REQUERIDO) ==========
    combinacion_id = 9999  # CAMBIAR: ID único (verificar que no existe)
    name = "Mi_Estrategia"  # CAMBIAR: Nombre corto sin espacios

    # ========== INDICADORES USADOS (para plot modular) ==========
    # Ajusta esta lista con todos los nombres de columna de indicadores
    # que genere tu estrategia y quieras ver en el plot.
    # Ejemplos: ["rsi", "stc"], ["dpo", "zscore", "atr"], etc.
    __indicators_used = ["rsi", "stc"]

    @classmethod
    def get_indicators_used(cls):
        """Devuelve la lista de indicadores que usa la estrategia."""
        return cls.__indicators_used
    
    # ========== PARÁMETROS OPTUNA ==========
    # Formato: "nombre": (min, max, step)
    # Tipos soportados: int, float
    parametros_optuna = {
        # === Indicadores ===
        "rsi_period": (7, 21, 1),           # int: período RSI
        "stc_fast": (10, 30, 2),            # int: EMA rápida STC
        "stc_slow": (30, 70, 5),            # int: EMA lenta STC
        
        # === Umbrales de Entrada ===
        "entry_threshold": (0.5, 2.0, 0.1), # float: umbral para entrar
        "confirm_bars": (1, 3, 1),          # int: barras de confirmación
        
        # === Umbrales de Salida ===
        "exit_threshold": (60, 80, 5),      # int: umbral para salir
        "n_reversal": (2, 4, 1),            # int: barras para reversal
    }
    
    # ========== CONFIGURACIÓN ==========
    TIMEOUT_BARS = 300  # Máximo de barras antes de cerrar posición
    
    # ========== MÉTODO: SUGERIR PARÁMETROS (REQUERIDO) ==========
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        Sugiere parámetros para Optuna.
        
        IMPORTANTE: Los nombres deben coincidir con parametros_optuna.
        
        Args:
            trial: Objeto Optuna Trial
            
        Returns:
            Dict con parámetros sugeridos
        """
        # === Parámetros de Indicadores ===
        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        stc_fast = trial.suggest_int("stc_fast", 10, 30, step=2)
        stc_slow = trial.suggest_int("stc_slow", 30, 70, step=5)
        
        # === Validación (asegurar coherencia) ===
        if stc_fast >= stc_slow:
            stc_slow = stc_fast + 10
        
        # === Parámetros de Entrada ===
        entry_threshold = trial.suggest_float("entry_threshold", 0.5, 2.0, step=0.1)
        confirm_bars = trial.suggest_int("confirm_bars", 1, 3)
        
        # === Parámetros de Salida ===
        exit_threshold = trial.suggest_int("exit_threshold", 60, 80, step=5)
        n_reversal = trial.suggest_int("n_reversal", 2, 4)
        
        return {
            "rsi_period": rsi_period,
            "stc_fast": stc_fast,
            "stc_slow": stc_slow,
            "entry_threshold": entry_threshold,
            "confirm_bars": confirm_bars,
            "exit_threshold": exit_threshold,
            "n_reversal": n_reversal,
        }
    
    # ========== MÉTODO: GENERAR SEÑALES (REQUERIDO) ==========
    def generate_signals(
        self, df: pl.DataFrame, params: Dict[str, Any]
    ) -> pl.DataFrame:
        """
        Genera señales de trading (signal_long, signal_short).
        
        IMPORTANTE: 
        - Debe retornar DataFrame con columnas 'signal_long' y 'signal_short'
        - Usar IndicadorFactory para calcular indicadores
        - Preferir expresiones vectorizadas de Polars
        - Usar Numba solo para lógica compleja bar-a-bar
        
        Args:
            df: DataFrame Polars con OHLCV
            params: Parámetros de suggest_params
            
        Returns:
            DataFrame con señales añadidas
        """
        # Para reporting/plot modular: inyectar siempre los indicadores usados
        params["__indicators_used"] = self.get_indicators_used()
        
        # ========================================
        # OPCIÓN A: Usar IndicadorFactory (Recomendado)
        # ========================================
        ind_config = {
            "rsi": {"activo": True, "period": params["rsi_period"], "out": "rsi"},
            "stc": {
                "activo": True,
                "fast": params["stc_fast"],
                "slow": params["stc_slow"],
                "cycle": 10,
                "out": "stc"
            },
            # Añadir más indicadores según necesidad...
        }
        df = IndicadorFactory.procesar(df, ind_config)
        
        # ========================================
        # OPCIÓN B: Usar helpers de indicator_specs
        # ========================================
        # ind_config = {
        #     "rsi": cfg_rsi(period=params["rsi_period"], out="rsi"),
        #     "macd": cfg_macd(fast=12, slow=26, signal=9),
        # }
        # df = IndicadorFactory.procesar(df, ind_config)
        
        # ========================================
        # GENERAR SEÑALES - OPCIÓN 1: Polars Vectorizado
        # ========================================
        # Más legible y mantenible para lógica simple
        
        # Referencias a valores previos
        rsi_prev = pl.col("rsi").shift(1)
        stc_prev = pl.col("stc").shift(1)
        
        # Condiciones de confirmación
        price_rising = (pl.col("close") > pl.col("close").shift(1))
        price_falling = (pl.col("close") < pl.col("close").shift(1))
        
        # LONG: RSI bajo + STC subiendo + precio subiendo
        cond_long = (
            (pl.col("rsi") < 40) &
            (pl.col("stc") < 50) &
            (pl.col("stc") > stc_prev) &
            price_rising
        )
        
        # SHORT: RSI alto + STC bajando + precio bajando
        cond_short = (
            (pl.col("rsi") > 60) &
            (pl.col("stc") > 50) &
            (pl.col("stc") < stc_prev) &
            price_falling
        )
        
        return df.with_columns([
            cond_long.fill_null(False).alias("signal_long"),
            cond_short.fill_null(False).alias("signal_short"),
        ])
        
        # ========================================
        # GENERAR SEÑALES - OPCIÓN 2: Numba Kernel
        # ========================================
        # Usar cuando la lógica es muy compleja para vectorizar
        #
        # rsi_arr = df["rsi"].to_numpy().astype(np.float64)
        # stc_arr = df["stc"].to_numpy().astype(np.float64)
        #
        # signal_long, signal_short = _detect_signals_kernel(
        #     rsi_arr, stc_arr,
        #     threshold1=40.0,
        #     threshold2=60.0,
        # )
        #
        # return df.with_columns([
        #     pl.Series("signal_long", signal_long),
        #     pl.Series("signal_short", signal_short),
        # ])
    
    # ========== MÉTODO: DECIDIR SALIDA (REQUERIDO) ==========
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
        """
        Decide cuándo y por qué cerrar la posición.
        
        IMPORTANTE:
        - Debe retornar ExitDecision o None
        - entry_idx es el índice de la barra de entrada
        - side es "LONG" o "SHORT"
        - saldo_apertura es el balance al entrar
        
        Args:
            df: DataFrame con indicadores calculados
            params: Parámetros de la estrategia
            entry_idx: Índice de entrada
            entry_price: Precio de entrada
            side: "LONG" o "SHORT"
            saldo_apertura: Balance al entrar
            
        Returns:
            ExitDecision con exit_idx y reason, o None si no hay salida
        """
        
        # Verificar que tenemos los indicadores necesarios
        if "rsi" not in df.columns:
            # Fallback: timeout
            timeout_idx = min(entry_idx + self.TIMEOUT_BARS, len(df) - 1)
            return ExitDecision(exit_idx=timeout_idx, reason="TIME_EXIT")
        
        # Extraer arrays para Numba
        rsi_arr = df["rsi"].to_numpy().astype(np.float64)
        is_long = side.upper() == "LONG"
        
        # Determinar umbral de salida
        if is_long:
            exit_threshold = float(params.get("exit_threshold", 70))
        else:
            exit_threshold = 100.0 - float(params.get("exit_threshold", 70))
        
        # ========================================
        # OPCIÓN A: Usar Numba Kernel (Recomendado)
        # ========================================
        exit_idx, reason_code = _find_exit_kernel(
            rsi_arr,
            entry_idx,
            is_long,
            exit_threshold,
            self.TIMEOUT_BARS,
        )
        
        # Mapear códigos a razones
        reason_map = {
            0: None,
            1: "RSI_THRESHOLD",
            2: "MOMENTUM_REVERSAL",
            3: "TIME_EXIT",
        }
        
        if exit_idx > 0 and reason_code > 0:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])
        
        # ========================================
        # OPCIÓN B: Lógica Simple sin Numba
        # ========================================
        # Para estrategias simples, puedes iterar directamente:
        #
        # n = len(df)
        # for i in range(entry_idx + 1, min(entry_idx + self.TIMEOUT_BARS, n)):
        #     rsi = df["rsi"][i]
        #     if is_long and rsi >= exit_threshold:
        #         return ExitDecision(exit_idx=i, reason="RSI_OVERBOUGHT")
        #     if not is_long and rsi <= exit_threshold:
        #         return ExitDecision(exit_idx=i, reason="RSI_OVERSOLD")
        
        # Fallback: timeout
        timeout_idx = min(entry_idx + self.TIMEOUT_BARS, len(df) - 1)
        return ExitDecision(exit_idx=timeout_idx, reason="TIME_EXIT")


# ============================================================================
# CHECKLIST PARA NUEVA ESTRATEGIA
# ============================================================================
"""
✅ 1. IDENTIFICACIÓN
   [ ] combinacion_id es único (verificar en otras estrategias)
   [ ] name es descriptivo y sin espacios especiales
   
✅ 2. PARÁMETROS
   [ ] parametros_optuna define rangos sensatos
   [ ] suggest_params coincide con parametros_optuna
   [ ] Validaciones de coherencia (ej: fast < slow)
   
✅ 3. SEÑALES (generate_signals)
   [ ] Retorna DataFrame con signal_long y signal_short
   [ ] Usa IndicadorFactory para calcular indicadores
   [ ] fill_null(False) para evitar NaN en señales
    [ ] Actualiza __indicators_used y lo inyecta en params["__indicators_used"]
   
✅ 4. SALIDAS (decide_exit)
   [ ] Retorna ExitDecision con exit_idx y reason
   [ ] Tiene fallback de timeout
   [ ] Verifica existencia de indicadores requeridos
   
✅ 5. PRUEBAS
   [ ] Ejecutar con pocos trials (N_TRIALS=5)
   [ ] Verificar que genera señales (no todos 0)
   [ ] Verificar que las salidas funcionan
   [ ] Revisar plots generados

✅ 6. REGISTRO
   [ ] El archivo está en modelox/strategies/
   [ ] Nombre sigue patrón: strategy_XX_nombre.py
   [ ] La clase se descubre automáticamente (verify registry)
"""

# ============================================================================
# INDICADORES DISPONIBLES EN IndicadorFactory
# ============================================================================
"""
Indicadores disponibles (usar en ind_config):

OSCILADORES:
  - "rsi": RSI(period)
  - "stc": Schaff Trend Cycle(fast, slow, cycle, smooth)
  - "mfi": Money Flow Index(period)
  - "adx": Average Directional Index(period)
  - "choppiness_index": CHOP(period)

TREND:
  - "ema": EMA(period, col)
  - "ema200_mtf": EMA 200 Multi-Timeframe(tf_min, period)
  - "supertrend": SuperTrend(period, mult)
  - "linreg_slope": Linear Regression Slope(window)

VOLATILIDAD:
  - "zscore": Z-Score(col, window)
  - "zscore_vwap": Z-Score sobre VWAP(window)
  - "efficiency_ratio": Kaufman Efficiency Ratio(window)

VOLUMEN:
  - "vwma": Volume Weighted MA(window)
  - "vwap_session": VWAP por sesión

ESTRUCTURA:
  - "donchian": Canales Donchian(window) -> out_hi, out_lo
  - "macd": MACD(fast, slow, signal) -> out_macd, out_signal, out_hist
"""

from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA: Z_RSI_ADX (ID 4) - RSI × VOLATILITY × ADX FILTER
# ══════════════════════════════════════════════════════════════════════════════

class StrategyZRsiAdx(EstrategiaBase):
    """
    ESTRATEGIA RSI × VOLATILIDAD × ADX
    
    Componentes:
    - RSI: Índice de Fuerza Relativa (detecta sobrecompra/sobreventa)
    - Volatilidad: Z-Score de volatilidad Garman-Klass (OHLC)
    - ADX: Average Directional Index (filtra mercado en rango vs tendencia)
    
    Lógica de Entrada:
    - LONG: RSI cruza al alza el nivel de sobreventa + Alta volatilidad + ADX bajo (mercado en rango)
    - SHORT: RSI cruza a la baja el nivel de sobrecompra + Alta volatilidad + ADX bajo (mercado en rango)
    
    Salidas:
    - Controladas por el sistema global de exits.py (NO salidas personalizadas)
    """

    combinacion_id = 4
    name = "Z_RSI_ADX"
    SALIDAS_PERSONALIZADAS = False  # Usar sistema global de exits

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        ESPACIO DE BÚSQUEDA DE PARÁMETROS
        
        PARÁMETROS OPTIMIZABLES:
        - RSI PERIODO: Longitud del RSI
        - UMBRAL RSI: Nivel simétrico (ej: 70-30, 65-35)
        - ADX UMBRAL: Nivel máximo de ADX para entrar
        - UMBRAL VOLATILIDAD MINIMA: Mínimo z-score de volatilidad
        - VOLATILIDAD PERIODO: Longitud de ventana para volatilidad
        
        PARÁMETROS FIJOS:
        - z_lookback: 100 (ventana para z-score)
        - z_range: 3.0 (rango de clamp del z-score)
        - adx_smoothing: 14 (suavizado del ADX)
        - di_length: 14 (longitud del DI)
        """
        # PARÁMETROS OPTIMIZABLES
        rsi_periodo = trial.suggest_int("RSI PERIODO", 5, 50, step=1)
        
        # UMBRAL RSI: Optimizamos solo el valor alto, el bajo es simétrico
        rsi_high = trial.suggest_int("rsi_overbought_raw", 60, 85, step=1)
        rsi_low = 100 - rsi_high  # Simétrico
        # Crear string descriptivo "70-30" o "65-35" etc.
        umbral_rsi_display = f"{rsi_high}-{rsi_low}"
        
        adx_umbral = trial.suggest_float("ADX UMBRAL", 15.0, 35.0, step=1.0)
        umbral_vol_minima = trial.suggest_float("UMBRAL VOLATILIDAD MINIMA", 0.5, 2.0, step=0.1)
        volatilidad_periodo = trial.suggest_int("VOLATILIDAD PERIODO", 5, 30, step=1)
        
        # PARÁMETROS FIJOS (valores por defecto razonables)
        z_lookback = 100  # Ventana histórica para calcular z-score
        z_range = 3.0     # Rango de clamp del z-score (±3σ)
        adx_smoothing = 14  # Suavizado estándar del ADX
        di_length = 14      # Longitud estándar del Directional Index
        
        return {
            # Parámetros internos (nombres técnicos para el código)
            "rsi_length": rsi_periodo,
            "rsi_overbought": rsi_high,
            "rsi_oversold": rsi_low,
            "vol_length": volatilidad_periodo,
            "z_lookback": z_lookback,
            "z_range": z_range,
            "vol_threshold": umbral_vol_minima,
            "adx_smoothing": adx_smoothing,
            "di_length": di_length,
            "adx_threshold": adx_umbral,
            
            # Parámetros display (para reporting - nombres amigables)
            "__display_params": {
                "RSI PERIODO": rsi_periodo,
                "UMBRAL RSI": umbral_rsi_display,
                "ADX UMBRAL": adx_umbral,
                "UMBRAL VOLATILIDAD MINIMA": umbral_vol_minima,
                "VOLATILIDAD PERIODO": volatilidad_periodo,
            }
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        GENERADOR DE SEÑALES (POLARS VECTORIAL)
        """
        
        # 1. INICIALIZACIÓN
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        # Extracción de parámetros
        rsi_len = params["rsi_length"]
        rsi_ob = params["rsi_overbought"]
        rsi_os = params["rsi_oversold"]
        
        vol_len = params["vol_length"]
        z_lookback = params["z_lookback"]
        z_range = params["z_range"]
        vol_thresh = params["vol_threshold"]
        
        adx_smooth = params["adx_smoothing"]
        di_len = params["di_length"]
        adx_thresh = params["adx_threshold"]

        # Configuración de Metadata
        warmup = max(rsi_len, z_lookback, adx_smooth, di_len) + 50
        params["__warmup_bars"] = warmup
        params["__indicators_used"] = ["rsi", "z_score", "adx"]
        params["__indicator_bounds"] = {
            "rsi": {"low": rsi_os, "high": rsi_ob, "mid": 50},
            "adx": {"low": 0, "high": 100, "mid": adx_thresh}
        }
        params["__indicator_specs"] = {
            "rsi": {"color": "#00FFFF", "type": "line", "panel": "rsi"},
            "z_score": {"color": "#FF00FF", "type": "line", "panel": "vol"},
            "adx": {"color": "#FFFF00", "type": "line", "panel": "adx"}
        }

        # INICIO LAZY FRAME
        q = df.lazy()

        # ═══════════════════════════════════════════════════════════════════
        # 2. CÁLCULO RSI (usando método de la base)
        # ═══════════════════════════════════════════════════════════════════
        rsi_expr = self.rsi_expr(close=pl.col("close"), length=rsi_len)
        q = q.with_columns([rsi_expr.alias("rsi")])

        # ═══════════════════════════════════════════════════════════════════
        # 3. CÁLCULO ADX
        # ═══════════════════════════════════════════════════════════════════
        # Calcular True Range
        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)
        
        # Directional Movement
        up_move = pl.col("high") - pl.col("high").shift(1)
        down_move = pl.col("low").shift(1) - pl.col("low")
        
        # +DM y -DM
        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
        
        q = q.with_columns([
            tr.alias("tr"),
            plus_dm.alias("plus_dm"),
            minus_dm.alias("minus_dm")
        ])
        
        # Smoothed TR y DM usando EWM (similar a Wilder's smoothing)
        # Wilder's smoothing: alpha = 1/n, lo cual corresponde a com = n-1 en ewm
        com_di = di_len - 1
        
        q = q.with_columns([
            pl.col("tr").ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("atr"),
            pl.col("plus_dm").ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("plus_dm_smooth"),
            pl.col("minus_dm").ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("minus_dm_smooth")
        ])
        
        # +DI y -DI
        q = q.with_columns([
            (100.0 * pl.col("plus_dm_smooth") / pl.col("atr")).fill_null(0).alias("plus_di"),
            (100.0 * pl.col("minus_dm_smooth") / pl.col("atr")).fill_null(0).alias("minus_di")
        ])
        
        # DX (Directional Index)
        di_sum = pl.col("plus_di") + pl.col("minus_di")
        di_diff = (pl.col("plus_di") - pl.col("minus_di")).abs()
        dx = pl.when(di_sum != 0).then(100.0 * di_diff / di_sum).otherwise(0.0)
        
        q = q.with_columns([dx.alias("dx")])
        
        # ADX (smoothed DX)
        com_adx = adx_smooth - 1
        adx_expr = pl.col("dx").ewm_mean(com=com_adx, min_periods=adx_smooth, ignore_nulls=True).fill_null(0)
        
        q = q.with_columns([adx_expr.alias("adx")])

        # ═══════════════════════════════════════════════════════════════════
        # 4. CÁLCULO Z-SCORE VOLATILIDAD (GARMAN-KLASS)
        # ═══════════════════════════════════════════════════════════════════
        # Garman-Klass volatility estimator usando OHLC
        # Formula: sqrt(SMA(0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2))
        
        ln_hl = (pl.col("high") / pl.col("low")).log()
        ln_co = (pl.col("close") / pl.col("open")).log()
        
        gk_component = 0.5 * ln_hl.pow(2) - (2 * np.log(2) - 1) * ln_co.pow(2)
        gk = gk_component.rolling_mean(window_size=vol_len).sqrt()
        
        q = q.with_columns([gk.alias("gk_vol")])
        
        # Z-Score de la volatilidad
        mean_vol = pl.col("gk_vol").rolling_mean(window_size=z_lookback)
        std_vol = pl.col("gk_vol").rolling_std(window_size=z_lookback)
        
        z_score_raw = pl.when(std_vol != 0).then(
            (pl.col("gk_vol") - mean_vol) / std_vol
        ).otherwise(0.0)
        
        # Clamped Z-Score (limitar entre -z_range y +z_range)
        z_clamped = z_score_raw.clip(-z_range, z_range)
        
        # Normalizar a [0, 2] para comparar con threshold
        z_normalized = (z_clamped + z_range) / (2.0 * z_range) * 2.0
        
        q = q.with_columns([
            z_score_raw.alias("z_score_raw"),
            z_normalized.alias("z_score")
        ])

        # ═══════════════════════════════════════════════════════════════════
        # 5. GENERACIÓN DE SEÑALES
        # ═══════════════════════════════════════════════════════════════════
        
        # Filtros
        high_volatility = pl.col("z_score") > vol_thresh
        adx_filter = pl.col("adx") < adx_thresh  # Mercado en rango (ADX bajo)
        
        # Cruces de RSI
        rsi_cross_up = (pl.col("rsi") > rsi_os) & (pl.col("rsi").shift(1) <= rsi_os)
        rsi_cross_down = (pl.col("rsi") < rsi_ob) & (pl.col("rsi").shift(1) >= rsi_ob)
        
        # Señales combinadas
        raw_long = high_volatility & adx_filter & rsi_cross_up
        raw_short = high_volatility & adx_filter & rsi_cross_down
        
        q = q.with_columns([
            self._as_bool(raw_long).alias("signal_long"),
            self._as_bool(raw_short).alias("signal_short")
        ])

        # ═══════════════════════════════════════════════════════════════════
        # 6. RETORNO
        # ═══════════════════════════════════════════════════════════════════
        return self.finalize_signals(
            q, 
            keep_cols=[
                "rsi", 
                "z_score", 
                "adx",
                "plus_di",
                "minus_di"
            ]
        )


# ══════════════════════════════════════════════════════════════════════════════
# NOTAS DE IMPLEMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
"""
ADAPTACIONES DEL CÓDIGO TRADINGVIEW:

1. RSI:
   - Usa el método rsi_expr() de la base (equivalente a Wilder's RSI)
   - Detecta cruces de niveles de sobrecompra/sobreventa

2. VOLATILIDAD (GARMAN-KLASS Z-SCORE):
   - Implementación vectorial de GK estimator: 
     sqrt(SMA(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))
   - Z-Score sobre ventana histórica (z_lookback)
   - Normalización a [0, 2] para facilitar threshold

3. ADX:
   - Implementación completa de ADX usando Wilder's smoothing
   - Calcula True Range, +DM, -DM
   - Smoothing con EWM (com = n-1 para emular Wilder's)
   - +DI, -DI, DX, y finalmente ADX
   - Filtro: ADX < threshold = mercado en rango (consolidación)

4. SALIDAS:
   - SALIDAS_PERSONALIZADAS = False
   - Las salidas TP/SL/Trailing son controladas por exits.py
   - NO se implementa la lógica de distancia fija del código original
   - El sistema global de exits es más flexible y optimizable

5. SEÑALES:
   - LONG: RSI cruza al alza nivel de sobreventa + Alta Vol + ADX bajo
   - SHORT: RSI cruza a la baja nivel de sobrecompra + Alta Vol + ADX bajo
   - Una sola señal por cruce (no repetidas)

6. OPTIMIZACIÓN:
   - Todos los parámetros son optimizables por Optuna
   - Rangos sensatos basados en valores típicos
   - Compatible con el sistema de backtesting MODELOX
"""

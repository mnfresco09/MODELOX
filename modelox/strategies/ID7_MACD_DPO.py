from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA: MACD + DPO (NORMALIZED) - ID 7
# ══════════════════════════════════════════════════════════════════════════════

class StrategyMacdDpoNormalized(EstrategiaBase):
    """
    ESTRATEGIA ID7: MACD + DPO CON NORMALIZACIÓN
    
    Lógica LONG:
      1. MACD Normalizado (-100 a 100) < -Threshold (e.g., -75)
      2. DPO Normalizado (-100 a 100) < -Threshold_DPO (e.g., -80)
      3. TRIGGER: Cruce Bullish de MACD (MACD Line cruza hacia arriba Signal Line)
    
    Lógica SHORT:
      1. MACD Normalizado > Threshold (e.g., 75)
      2. DPO Normalizado > Threshold_DPO (e.g., 80)
      3. TRIGGER: Cruce Bearish de MACD (MACD Line cruza hacia abajo Signal Line)
      
    Normalización:
      - Se usa un periodo de lookback (norm_lookback) para encontrar Min/Max.
      - Valor = ((Raw - Min) / (Max - Min)) * 200 - 100
    """

    combinacion_id = 7
    name = "MACD_DPO_NORM"
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        Definición del espacio de búsqueda para Optuna.
        Restricción: macd_slow >= 2 * macd_fast
        """
        # 1. Sugerir Fast primero
        fast = trial.suggest_int("macd_fast", 6, 15, step=1)
        
        # 2. Calcular límites para Slow (Static Range to avoid Optuna Multivariate issues)
        # fast varies 6-15. Min slow = 12. Max slow usually ~30-60.
        slow = trial.suggest_int("macd_slow", 20, 60, step=1)

        return {
            # --- MACD PARAMS ---
            "macd_fast": fast,
            "macd_slow": slow,
            "macd_signal": trial.suggest_int("macd_signal", 5, 20, step=1),
            
            # --- DPO PARAMS ---
            "dpo_len": trial.suggest_int("dpo_len", 10, 50, step=2),
            
            # --- NORMALIZATION ---
            "norm_lookback": trial.suggest_int("norm_lookback", 100, 200, step=25),
            
            # --- THRESHOLDS (Symmetric) ---
            # Se usa el valor positivo. Para Long se invierte (-val).
            "macd_threshold": trial.suggest_int("macd_threshold", 25, 75, step=5),
            "dpo_threshold": trial.suggest_int("dpo_threshold", 25, 75, step=5),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        
        # 1. INICIALIZACIÓN
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # Parametros
        fast = params["macd_fast"]
        slow = params["macd_slow"]
        sig_len = params["macd_signal"]
        dpo_len = params["dpo_len"]
        norm_lb = params["norm_lookback"]
        
        macd_th = params["macd_threshold"]
        dpo_th = params["dpo_threshold"]

        # Ensure Fast < Slow (swap if needed for consistency)
        if fast >= slow:
            fast, slow = slow, fast

        # Enforce constraint: Slow >= 2 * Fast
        if slow < fast * 2:
            slow = fast * 2
            
        # Update params with actual used values
        params["macd_fast"] = fast
        params["macd_slow"] = slow

        # Configuración de Metadata para plots
        params["__warmup_bars"] = max(slow, dpo_len, norm_lb) + 10
        params["__indicators_used"] = ["norm_macd", "norm_sig", "norm_dpo"]
        
        # Panel grouping:
        # Panel 1: MACD Normalizado + Signal Normalizado (cruces visibles con threshold)
        # Panel 2: Normalized DPO (with thresholds)
        params["__indicator_specs"] = {
            "norm_macd": {"panel": 1, "color": "#2962FF", "name": "MACD Norm"},
            "norm_sig":  {"panel": 1, "color": "#FF6D00", "name": "Signal Norm"},
            "norm_dpo":  {"panel": 2, "color": "#D500F9", "name": "DPO Norm"},
        }

        params["__indicator_bounds"] = {
            "norm_macd": {
                "upper": float(macd_th), 
                "lower": float(-macd_th), 
                "mid": 0
            },
            "norm_dpo": {
                "upper": float(dpo_th), 
                "lower": float(-dpo_th), 
                "mid": 0
            }
        }

        # INICIO LAZY FRAME
        q = df.lazy()

        # 2. CÁLCULO MACD
        # ----------------------------------------------------------------------
        # EMA Fast, EMA Slow
        q = q.with_columns([
            pl.col("close").ewm_mean(span=fast, adjust=False).alias("_ema_fast"),
            pl.col("close").ewm_mean(span=slow, adjust=False).alias("_ema_slow"),
        ])
        
        # MACD Line & Signal Line
        q = q.with_columns([
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
        ])
        
        q = q.with_columns([
            pl.col("macd_line").ewm_mean(span=sig_len, adjust=False).alias("sig_line")
        ])

        # 3. CÁLCULO DPO
        # ----------------------------------------------------------------------
        # DPO = Close - SMA(Close, len)[shifted by len/2 + 1]
        # Valid displaced logic: Comparing current price to a past average.
        displacement = (dpo_len // 2) + 1
        
        q = q.with_columns([
            pl.col("close").rolling_mean(window_size=dpo_len).alias("_sma_dpo")
        ])
        
        q = q.with_columns([
            (pl.col("close") - pl.col("_sma_dpo").shift(displacement)).alias("dpo_raw")
        ])

        # 4. NORMALIZACIÓN (-100 a 100)
        # ----------------------------------------------------------------------
        # Formula: 2 * (Val - Min) / (Max - Min) - 1  --> Scaled to 100: * 100
        # Result: ((Val - Min) / (Max - Min)) * 200 - 100
        
        # Helper para normalizar usando min/max específicos
        def normalize_with_range(col_name: str, min_col: str, max_col: str, alias: str) -> pl.Expr:
            range_val = pl.col(max_col) - pl.col(min_col)
            # Evitar división por cero
            norm = (
                (pl.col(col_name) - pl.col(min_col)) / 
                pl.when(range_val == 0).then(1.0).otherwise(range_val)
            ) * 200.0 - 100.0
            return norm.fill_null(0).alias(alias)

        # A. Calcular rango móvil para MACD
        q = q.with_columns([
            pl.col("macd_line").rolling_min(window_size=norm_lb).alias("_macd_min"),
            pl.col("macd_line").rolling_max(window_size=norm_lb).alias("_macd_max"),
        ])

        # B. Normalizar MACD y Signal usando el MISMO rango (del MACD)
        # Esto preserva los cruces y la escala relativa
        q = q.with_columns([
            normalize_with_range("macd_line", "_macd_min", "_macd_max", "norm_macd"),
            normalize_with_range("sig_line", "_macd_min", "_macd_max", "norm_sig"),
        ])

        # C. Normalizar DPO (su propio rango)
        q = q.with_columns([
            pl.col("dpo_raw").rolling_min(window_size=norm_lb).alias("_dpo_min"),
            pl.col("dpo_raw").rolling_max(window_size=norm_lb).alias("_dpo_max"),
        ])
        
        q = q.with_columns([
            normalize_with_range("dpo_raw", "_dpo_min", "_dpo_max", "norm_dpo"),
        ])

        # Actualizar specs para visualización correcta
        # Panel 1: MACD Normalizado + Signal Normalizado (cruces visibles)
        # Panel 2: DPO Normalizado
        params["__indicator_specs"] = {
            "norm_macd": {"panel": 1, "color": "#2962FF", "name": "MACD Norm"},
            "norm_sig":  {"panel": 1, "color": "#FF6D00", "name": "Signal Norm"},
            "norm_dpo":  {"panel": 2, "color": "#D500F9", "name": "DPO Norm"},
        }

        # Actualizar bounds (líneas de referencia)
        params["__indicator_bounds"] = {
            "norm_macd": {
                "upper": float(macd_th), 
                "lower": float(-macd_th), 
                "mid": 0
            },
             "norm_dpo": {
                "upper": float(dpo_th), 
                "lower": float(-dpo_th), 
                "mid": 0
            }
        }

        # 5. LÓGICA DE CRUCES Y FILTROS
        # ----------------------------------------------------------------------
        # Detectar Cruces MACD vs Signal
        # Bullish Cross: MACD > Signal AND MACD_prev <= Signal_prev
        # Bearish Cross: MACD < Signal AND MACD_prev >= Signal_prev
        
        q = q.with_columns([
            (pl.col("macd_line") > pl.col("sig_line")).alias("macd_bullish"),
            (pl.col("macd_line") < pl.col("sig_line")).alias("macd_bearish"),
        ])
        
        q = q.with_columns([
            (
                pl.col("macd_bullish") & 
                pl.col("macd_bullish").shift(1).fill_null(False).not_()
            ).alias("cross_up"),
            
            (
                pl.col("macd_bearish") & 
                pl.col("macd_bearish").shift(1).fill_null(False).not_()
            ).alias("cross_down"),
        ])

        # 6. SEÑALES FINALES
        # ----------------------------------------------------------------------
        # LONG:
        # - MACD Cros Up
        # - NormMACD < -macd_th
        # - NormDPO < -dpo_th
        
        # SHORT:
        # - MACD Cross Down
        # - NormMACD > macd_th
        # - NormDPO > dpo_th
        
        long_cond = (
            pl.col("cross_up") &
            (pl.col("norm_macd") < -macd_th) &
            (pl.col("norm_dpo") < -dpo_th)
        )
        
        short_cond = (
            pl.col("cross_down") &
            (pl.col("norm_macd") > macd_th) &
            (pl.col("norm_dpo") > dpo_th)
        )

        q = q.with_columns([
            self._as_bool(long_cond).alias("signal_long"),
            self._as_bool(short_cond).alias("signal_short"),
        ])

        # 7. RETORNO
        return self.finalize_signals(
            q, 
            keep_cols=["norm_macd", "norm_sig", "norm_dpo", "macd_line", "sig_line", "dpo_raw"]
        )

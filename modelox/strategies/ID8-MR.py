from __future__ import annotations
from typing import Any, Dict, List
import polars as pl
from .ESTRATEGIA_BASE import EstrategiaBase

class StrategyCandleReversionBoss(EstrategiaBase):
    """
    ESTRATEGIA: CANDLE REVERSION FRACTAL (FINAL BOSS) - FIXED
    ------------------------------------------------
    Detección avanzada de patrones de reversión con geometría parametrizada.
    Corregido error de ambigüedad de variables y optimización de expresiones Polars.
    """

    combinacion_id = 8
    name = "ID8.MEANT REVERSION"
    
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """Define el espacio de búsqueda para Optuna."""
        return {
            "pattern_tf": trial.suggest_categorical("pattern_tf", ["1m", "5m", "15m", "30m", "1h"]),
            "min_wick_ratio": trial.suggest_float("min_wick_ratio", 0.55, 0.85, step=0.05),
            "max_body_ratio": trial.suggest_float("max_body_ratio", 0.10, 0.35, step=0.05),
            "tweezer_precision": trial.suggest_float("tweezer_precision", 0.00005, 0.0003, step=0.00005),
            "z_threshold": trial.suggest_float("z_threshold", 1.8, 3.2, step=0.2),
            "rvol_threshold": trial.suggest_float("rvol_threshold", 1.3, 2.5, step=0.1),
            "z_period": trial.suggest_int("z_period", 30, 100, step=10),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        """Solicita los marcos temporales necesarios."""
        ptf = params.get("pattern_tf", "15m")
        return list(set([ptf, "1h"]))

    def _z_score_expr(self, col_name: str, length: int) -> pl.Expr:
        """Expresión robusta para el Z-Score con protección de división por cero."""
        mean = pl.col(col_name).rolling_mean(length)
        std = pl.col(col_name).rolling_std(length)
        # Evitamos ambigüedad envolviendo col_name en pl.col()
        return (pl.col(col_name) - mean) / pl.when(std == 0).then(0.001).otherwise(std)

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        self._init_params_metadata(params) #
        
        # 1. IDENTIFICACIÓN EXPLÍCITA DE COLUMNAS
        ptf = params.get("pattern_tf", "15m")
        
        # Obtener columnas de forma segura (compatible con DataFrame y LazyFrame)
        # Esto soluciona errores si df es LazyFrame y no tiene el atributo .columns
        if hasattr(df, "collect_schema"):
            cols = df.collect_schema().names()
        elif hasattr(df, "columns"):
            cols = df.columns
        else:
            cols = df.schema.keys()

        # Se verifica la existencia de sufijos para evitar ambigüedades en el join
        suffix = f"_{ptf}"
        
        # Si la columna con sufijo existe (ej. close_15m), se usa. Si no, se usa la base (close).
        c = f"close{suffix}" if f"close{suffix}" in cols else "close"
        o = f"open{suffix}" if f"open{suffix}" in cols else "open"
        h = f"high{suffix}" if f"high{suffix}" in cols else "high"
        low_col = f"low{suffix}" if f"low{suffix}" in cols else "low"
        v = f"volume{suffix}" if f"volume{suffix}" in cols else "volume"
            
        self._require_columns(df, ["timestamp", c, o, h, low_col, v])

        # 2. RECUPERAR PARÁMETROS
        m_wick = params.get("min_wick_ratio", 0.6)
        m_body = params.get("max_body_ratio", 0.2)
        tw_prec = params.get("tweezer_precision", 0.0001)
        z_t = params.get("z_threshold", 2.0)
        rv_t = params.get("rvol_threshold", 1.5)

        q = df.lazy()

        # --- FASE A: CÁLCULOS TÉCNICOS ---
        q = q.with_columns([
            self._z_score_expr(c, params.get("z_period", 50)).alias("z_score"),
            (pl.col(v) / pl.col(v).rolling_mean(20)).alias("rvol"),
            (pl.col(h) - pl.col(low_col)).alias("range_total")
        ])
        
        # SOLUCIÓN AL ERROR: Usar pl.col() dentro de funciones horizontales
        # Se pasa *args en lugar de una lista para compatibilidad con versiones de Polars
        q = q.with_columns([
            (pl.col(c) - pl.col(o)).abs().alias("body_size"),
            (pl.min_horizontal(pl.col(o), pl.col(c)) - pl.col(low_col)).alias("lower_wick"),
            (pl.col(h) - pl.max_horizontal(pl.col(o), pl.col(c))).alias("upper_wick")
        ])

        # --- FASE B: DETECCIÓN DE PATRONES ---
        # Hammer / Pin Bar
        is_hammer = (pl.col("lower_wick") >= pl.col("range_total") * m_wick) & \
                    (pl.col("body_size") <= pl.col("range_total") * m_body)
        
        # Shooting Star
        is_star = (pl.col("upper_wick") >= pl.col("range_total") * m_wick) & \
                  (pl.col("body_size") <= pl.col("range_total") * m_body)

        # Engulfing (Comparación con vela anterior del mismo TF)
        is_engulfing_long = (pl.col(c) > pl.col(o).shift(1)) & \
                            (pl.col(o) < pl.col(c).shift(1)) & \
                            (pl.col(c) > pl.col(o))
                            
        is_engulfing_short = (pl.col(c) < pl.col(o).shift(1)) & \
                             (pl.col(o) > pl.col(c).shift(1)) & \
                             (pl.col(c) < pl.col(o))

        # Tweezers
        is_tweezer_bottom = (pl.col(low_col) - pl.col(low_col).shift(1)).abs() < (pl.col(low_col) * tw_prec)
        is_tweezer_top = (pl.col(h) - pl.col(h).shift(1)).abs() < (pl.col(h) * tw_prec)

        # --- FASE C: LÓGICA DE FILTRADO FINAL ---
        sig_long = (is_hammer | is_engulfing_long | is_tweezer_bottom) & \
                   (pl.col("z_score") < -z_t) & \
                   (pl.col("rvol") > rv_t)
        
        sig_short = (is_star | is_engulfing_short | is_tweezer_top) & \
                    (pl.col("z_score") > z_t) & \
                    (pl.col("rvol") > rv_t)

        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
        ])

        return self.finalize_signals(q, keep_cols=["z_score", "rvol"])

from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA: KINETIC MOMENTUM VALIDATOR (ID 33) - RANGOS INDEPENDIENTES
# ══════════════════════════════════════════════════════════════════════════════

class StrategyKineticMomentumValidator(EstrategiaBase):
    """
    ESTRATEGIA DE CRUCE CON VALIDACIÓN DE EXPANSIÓN (MOMENTUM) - PURE ZLEMA

    Componentes:
    - Fast MA: ZLEMA (Zero-Lag)
    - Slow MA: ZLEMA (Zero-Lag)
    - Optimización: RANGOS INDEPENDIENTES (Fast y Slow libres).
    """

    combinacion_id = 15
    name = "prueba1"
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        CONFIGURACIÓN STANDARD (RANGOS INDEPENDIENTES)
        Se definen rangos separados para Fast y Slow.
        """

        # 1. DEFINICIÓN DE RANGOS INDEPENDIENTES
        # "Mandamelo normal entre rangos todos"
        raw_fast = trial.suggest_int("zlema_fast_len", 70, 200, step=1)
        raw_slow = trial.suggest_int("zlema_slow_len", 400, 800, step=1)

        # 2. VALIDACIÓN LÓGICA (SWAP)
        # Aunque los rangos son libres, para que la lógica de cruce funcione
        # (Fast > Slow = Bullish), la Fast debe ser numéricamente menor.
        # Si el optimizador elige Fast=100 y Slow=50, los intercambiamos.
        if raw_fast < raw_slow:
            fast_len = raw_fast
            slow_len = raw_slow
        else:
            fast_len = raw_slow
            slow_len = raw_fast

        # Caso borde: si son iguales (muy raro), separamos la lenta un poco
        if fast_len == slow_len:
            slow_len += 5

        return {
            "zlema_fast_len": fast_len,
            "zlema_slow_len": slow_len,

            # --- FILTRO DE MOMENTUM ---
            "lookbar": trial.suggest_int("lookbar", 50, 140, step=5),
            "req_dist_pct": trial.suggest_float("req_dist_pct", 0.85, 1.9, step=0.05),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:

        # 1. INICIALIZACIÓN
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # Extracción directa de parámetros
        f_len = params["zlema_fast_len"]
        s_len = params["zlema_slow_len"]
        lookbar = params["lookbar"]
        req_dist_pct = params["req_dist_pct"]

        # Configuración de Metadata
        params["__warmup_bars"] = s_len + 50
        params["__indicators_used"] = ["fast_ma", "slow_ma"]
        params["__indicator_specs"] = {
            "fast_ma": {"color": "#00FFFF", "type": "line"},
            "slow_ma": {"color": "#FF00FF", "type": "line"}
        }

        # INICIO LAZY FRAME
        q = df.lazy()

        # 2. CÁLCULOS DE INDICADORES (ZLEMA PURA)
        # ----------------------------------------------------------------------
        q = q.with_columns(pl.col("close").log().alias("log_close"))

        # Función auxiliar interna para ZLEMA Logarítmica
        def _calc_zlema_expr(col_name: str, length: int) -> pl.Expr:
            # Protección contra longitud 1 o menor
            length = max(2, length)
            lag = int((length - 1) / 2)
            # EMA(Data + (Data - Data(lag)), len)
            de_lagged = pl.col(col_name) + (pl.col(col_name) - pl.col(col_name).shift(lag))
            return de_lagged.ewm_mean(span=length, adjust=False)

        q = q.with_columns([
            _calc_zlema_expr("log_close", f_len).alias("fast_log"),
            _calc_zlema_expr("log_close", s_len).alias("slow_log")
        ])

        # 3. LÓGICA DE CICLOS Y DISTANCIAS
        # ----------------------------------------------------------------------
        q = q.with_columns([
            (pl.col("fast_log") > pl.col("slow_log")).alias("is_bullish"),
            (pl.col("fast_log") - pl.col("slow_log")).alias("raw_diff")
        ])

        # ID de ciclo
        q = q.with_columns([
            (pl.col("is_bullish") != pl.col("is_bullish").shift(1).fill_null(True))
            .cast(pl.Int32).cum_sum().alias("cycle_id")
        ])

        # Métricas intra-ciclo
        log_req = np.log(1 + req_dist_pct / 100.0)

        q = q.with_columns([
            (pl.cum_count("cycle_id").over("cycle_id") - 1).alias("bars_in_cycle"),
            pl.col("raw_diff").abs().first().over("cycle_id").alias("init_dist"),
        ])

        # Evaluación de Meta
        q = q.with_columns([
            (pl.col("init_dist") + log_req).alias("target_dist"),
            pl.col("raw_diff").abs().alias("curr_dist")
        ])

        # 4. GENERACIÓN DE SEÑALES
        # ----------------------------------------------------------------------
        cond_base = (
            (pl.col("bars_in_cycle") <= lookbar) &
            (pl.col("curr_dist") >= pl.col("target_dist"))
        )

        long_cond = pl.col("is_bullish") & cond_base
        short_cond = (~pl.col("is_bullish")) & cond_base

        # Filtro One-Shot
        sig_long = long_cond & (long_cond.cast(pl.Int32).cum_sum().over("cycle_id") == 1)
        sig_short = short_cond & (short_cond.cast(pl.Int32).cum_sum().over("cycle_id") == 1)

        # 5. RETORNO
        # ----------------------------------------------------------------------
        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
            pl.col("fast_log").exp().alias("fast_ma"),
            pl.col("slow_log").exp().alias("slow_ma")
        ])

        return self.finalize_signals(q, keep_cols=["fast_ma", "slow_ma", "cycle_id"])

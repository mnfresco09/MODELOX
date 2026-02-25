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

    combinacion_id = 1
    name = "CRUCE_ZLEMA"
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        CONFIGURACIÓN STANDARD (RANGOS INDEPENDIENTES)
        Se definen rangos separados para Fast y Slow.
        """

        # 1. DEFINICIÓN DE RANGOS INDEPENDIENTES
        # "Mandamelo normal entre rangos todos"
        raw_fast = trial.suggest_int("zlema_fast_len", 14, 70, step=2)
        raw_slow = trial.suggest_int("zlema_slow_len", 100, 250, step=5)

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
            "lookbar": trial.suggest_int("lookbar", 4, 24, step=1),
            "req_dist_abs": trial.suggest_int("req_dist_abs", 0.05, 1.25, step=0.05),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:

        # 1. INICIALIZACIÓN
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close", "high", "low"])

        # Extracción directa de parámetros
        f_len = params["zlema_fast_len"]
        s_len = params["zlema_slow_len"]
        lookbar = params["lookbar"]
        req_dist_abs = params["req_dist_abs"]

        # Configuración de Metadata
        params["__warmup_bars"] = s_len + 120
        params["__indicators_used"] = ["fast_ma", "slow_ma"]
        params["__indicator_specs"] = {
            "fast_ma": {"color": "#00FFFF", "type": "line"},
            "slow_ma": {"color": "#FF00FF", "type": "line"}
        }

        # INICIO LAZY FRAME
        q = df.lazy()

        # 2. CÁLCULOS DE INDICADORES (ZLEMA PURA)
        # ----------------------------------------------------------------------
        # Usamos precio medio (high + low)/2 en lugar de close
        q = q.with_columns(((pl.col("high") + pl.col("low")) / 2).alias("hl2"))

        # Función auxiliar interna para ZLEMA Normal
        def _calc_zlema_expr(col_name: str, length: int) -> pl.Expr:
            # Protección contra longitud 1 o menor
            length = max(2, length)
            lag = int((length - 1) / 2)
            # EMA(Data + (Data - Data(lag)), len)
            de_lagged = pl.col(col_name) + (pl.col(col_name) - pl.col(col_name).shift(lag))
            return de_lagged.ewm_mean(span=length, adjust=False)

        q = q.with_columns([
            _calc_zlema_expr("hl2", f_len).alias("fast_ma"),
            _calc_zlema_expr("hl2", s_len).alias("slow_ma")
        ])

        # 3. LÓGICA DE CICLOS Y DISTANCIAS
        # ----------------------------------------------------------------------
        q = q.with_columns([
            (pl.col("fast_ma") > pl.col("slow_ma")).alias("is_bullish"),
            (pl.col("fast_ma") - pl.col("slow_ma")).alias("raw_diff")
        ])

        # ID de ciclo
        q = q.with_columns([
            (pl.col("is_bullish") != pl.col("is_bullish").shift(1).fill_null(True))
            .cast(pl.Int32).cum_sum().alias("cycle_id")
        ])

        # Métricas intra-ciclo

        q = q.with_columns([
            (pl.cum_count("cycle_id").over("cycle_id") - 1).alias("bars_in_cycle"),
            pl.col("raw_diff").abs().first().over("cycle_id").alias("init_dist"),
        ])

        # Evaluación de Meta
        q = q.with_columns([
            (pl.col("init_dist") + req_dist_abs).alias("target_dist"),
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
            self._as_bool(sig_short).alias("signal_short")
        ])

        return self.finalize_signals(q, keep_cols=["fast_ma", "slow_ma", "cycle_id"])

    def decide_exit(
        self,
        df: pl.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        side: int,
        **kwargs: Any,
    ):
        """Salida personalizada: Cruce inverso de ZLEMA."""
        try:
            fast_ma = df["fast_ma"].to_numpy()
            slow_ma = df["slow_ma"].to_numpy()
            close = df["close"].to_numpy()
        except BaseException:
            return None

        is_long = side == 1 or str(side).upper() == "LONG"
        is_short = side == -1 or str(side).upper() == "SHORT"

        for i in range(entry_idx + 1, len(close)):
            if is_long:
                if fast_ma[i] < slow_ma[i]:
                    return {
                        "exit_idx": i,
                        "exit_price": close[i],
                        "reason": "CRUCE_BAJISTA_ZLEMA"
                    }
            elif is_short:
                if fast_ma[i] > slow_ma[i]:
                    return {
                        "exit_idx": i,
                        "exit_price": close[i],
                        "reason": "CRUCE_ALCISTA_ZLEMA"
                    }
        
        return None

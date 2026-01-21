from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA: KINETIC MOMENTUM VALIDATOR (ASYMMETRIC)
# ══════════════════════════════════════════════════════════════════════════════

class StrategyKineticMomentumValidator(EstrategiaBase):
    """
    ESTRATEGIA DE CRUCE CON VALIDACIÓN DE EXPANSIÓN (MOMENTUM) - ASIMÉTRICA
    
    Permite configuraciones independientes para Long y Short, adaptándose a la
    naturaleza diferente de las caídas (rápidas) vs subidas (lentas).
    """

    combinacion_id = 15
    name = "prueba1"
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        CONFIGURACIÓN ASIMÉTRICA
        Basada en la divergencia observada entre p1/p3 (Bull) y p2 (Bear).
        """
        
        # ─── PARÁMETROS LONG (Basados en p1 y p3: Tendencia Sólida) ───
        # ZLEMA Slow Long: La "constante universal" (465-485)
        zlema_slow_long = trial.suggest_int("zlema_slow_long", 330, 1000, step=10)
        # ZLEMA Fast Long: Lenta para confirmar (80-160)
        zlema_fast_long = trial.suggest_int("zlema_fast_long", 30, 300, step=5)
        # Distancia Long: Menor exigencia, dejar correr (0.5 - 1.2)
        req_dist_long = trial.suggest_float("req_dist_long", 0.2, 3.5, step=0.05)
        # Lookbar Long: Paciencia (100-140)
        lookbar_long = trial.suggest_int("lookbar_long", 20, 150, step=5)

        # ─── PARÁMETROS SHORT (Basados en p2: Reacción Rápida) ───
        # ZLEMA Slow Short: Más reactiva para detectar el cambio a bajista antes
        zlema_slow_short = trial.suggest_int("zlema_slow_short", 330, 1000, step=10)
        # ZLEMA Fast Short: Muy rápida para pillar el "ascensor" (40-80)
        zlema_fast_short = trial.suggest_int("zlema_fast_short", 30, 300, step=5)
        # Distancia Short: Mayor exigencia por volatilidad (1.2 - 2.0)
        req_dist_short = trial.suggest_float("req_dist_short", 0.2, 3.5, step=0.05)
        # Lookbar Short: Menos paciencia, el movimiento debe ser inmediato (40-80)
        lookbar_short = trial.suggest_int("lookbar_short", 20, 150, step=5)

        return {
            "zlema_slow_long": zlema_slow_long,
            "zlema_fast_long": zlema_fast_long,
            "req_dist_long": req_dist_long,
            "lookbar_long": lookbar_long,
            
            "zlema_slow_short": zlema_slow_short,
            "zlema_fast_short": zlema_fast_short,
            "req_dist_short": req_dist_short,
            "lookbar_short": lookbar_short,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # INICIO LAZY FRAME
        q = df.lazy()
        q = q.with_columns(pl.col("close").log().alias("log_close"))

        # Función auxiliar para ZLEMA
        def _calc_zlema_expr(col_name: str, length: int) -> pl.Expr:
            lag = int((length - 1) / 2)
            de_lagged = pl.col(col_name) + (pl.col(col_name) - pl.col(col_name).shift(lag))
            return de_lagged.ewm_mean(span=length, adjust=False)

        # 1. CÁLCULO DE MEDIAS (DOBLE MOTOR)
        # Calculamos 4 medias en total: Par Long y Par Short
        q = q.with_columns([
            # Motor Long
            _calc_zlema_expr("log_close", params["zlema_fast_long"]).alias("fast_L"),
            _calc_zlema_expr("log_close", params["zlema_slow_long"]).alias("slow_L"),
            # Motor Short
            _calc_zlema_expr("log_close", params["zlema_fast_short"]).alias("fast_S"),
            _calc_zlema_expr("log_close", params["zlema_slow_short"]).alias("slow_S"),
        ])

        # 2. LÓGICA LONG (MOTOR 1)
        # -----------------------
        q = q.with_columns([
            (pl.col("fast_L") > pl.col("slow_L")).alias("bull_zone_L"),
            (pl.col("fast_L") - pl.col("slow_L")).alias("diff_L")
        ])
        
        # Ciclo Long
        q = q.with_columns(
            (pl.col("bull_zone_L") != pl.col("bull_zone_L").shift(1).fill_null(True))
            .cast(pl.Int32).cum_sum().alias("cycle_L")
        )

        # Métricas Long
        log_req_L = np.log(1 + params["req_dist_long"] / 100.0)
        q = q.with_columns([
            (pl.cum_count("cycle_L").over("cycle_L") - 1).alias("bars_L"),
            pl.col("diff_L").abs().first().over("cycle_L").alias("init_dist_L")
        ])

        cond_long = (
            pl.col("bull_zone_L") &  # Estamos en zona alcista
            (pl.col("bars_L") <= params["lookbar_long"]) & # Dentro de la ventana de tiempo
            (pl.col("diff_L").abs() >= (pl.col("init_dist_L") + log_req_L)) # Expansión suficiente
        )

        # 3. LÓGICA SHORT (MOTOR 2)
        # -----------------------
        q = q.with_columns([
            (pl.col("fast_S") < pl.col("slow_S")).alias("bear_zone_S"),
            (pl.col("fast_S") - pl.col("slow_S")).alias("diff_S")
        ])

        # Ciclo Short (Independiente)
        q = q.with_columns(
            (pl.col("bear_zone_S") != pl.col("bear_zone_S").shift(1).fill_null(True))
            .cast(pl.Int32).cum_sum().alias("cycle_S")
        )

        # Métricas Short
        log_req_S = np.log(1 + params["req_dist_short"] / 100.0)
        q = q.with_columns([
            (pl.cum_count("cycle_S").over("cycle_S") - 1).alias("bars_S"),
            pl.col("diff_S").abs().first().over("cycle_S").alias("init_dist_S")
        ])

        cond_short = (
            pl.col("bear_zone_S") & # Estamos en zona bajista
            (pl.col("bars_S") <= params["lookbar_short"]) &
            (pl.col("diff_S").abs() >= (pl.col("init_dist_S") + log_req_S))
        )

        # 4. FILTRO ONE-SHOT Y SALIDA
        # ---------------------------
        sig_long = cond_long & (cond_long.cast(pl.Int32).cum_sum().over("cycle_L") == 1)
        sig_short = cond_short & (cond_short.cast(pl.Int32).cum_sum().over("cycle_S") == 1)

        # Visualización: Para graficar, usamos las medias LONG por defecto, 
        # pero internamente operamos con dos juegos.
        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
            pl.col("fast_L").exp().alias("fast_ma"), # Visualización principal
            pl.col("slow_L").exp().alias("slow_ma")
        ])

        return self.finalize_signals(q, keep_cols=["fast_ma", "slow_ma"])
from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID1.PY
================================================================================
ID        : 1
NOMBRE    : ZLEMA — Zero-Lag EMA Cruce + Ciclos + Expansión + Memoria de Ciclo
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
REDISEÑO COMPLETO vs versión original:
  BUGS CORREGIDOS:
    - suggest_int con valores float → suggest_float para dist_pct
    - Distancia absoluta no normalizada → % del precio (HL2)
    - Loop Python en decide_exit → SALIDAS_PERSONALIZADAS=False
    - Swap fast/slow antipatrón → rangos separados sin solapamiento
    - cum_count frágil → rank().over() robusto

  NUEVA FUNCIONALIDAD — MEMORIA DE CICLO:
    Si el ciclo anterior tuvo señal Y fue perdedor → bloquear dirección opuesta
    - Ciclo Long perdedor:  close_final < close_señal → bloquear Short siguiente
    - Ciclo Short perdedor: close_final > close_señal → bloquear Long siguiente
    - Solo evalúa ciclos donde hubo señal real (si no hubo señal → neutro)
    - Bloqueo dura 1 ciclo (el inmediatamente siguiente)

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  ZLEMA (Zero-Lag EMA):
    lag       = (length - 1) // 2
    de_lagged = HL2 + (HL2 - HL2.shift(lag))
    ZLEMA     = EWM(de_lagged, span=length)

  CICLO:
    Definido por cambio de signo de (fast_ma - slow_ma)
    cycle_id incrementa en cada cruce

  CONDICIÓN DE ENTRADA (one-shot por ciclo):
    1. Dentro de lookbar velas desde inicio del ciclo
    2. Expansión: dist_norm_pct ≥ dist_init + dist_pct%
    3. Ciclo anterior en esa dirección NO fue perdedor (memoria)
    4. Solo primera señal válida por ciclo

  MEMORIA DE CICLO:
    Para cada cycle_id con señal:
      - Registrar precio de la vela de señal (close_señal)
      - Registrar precio de la última vela del ciclo (close_final)
      - Long perdedor  = close_final < close_señal → bloquear Short en cycle_id+1
      - Short perdedor = close_final > close_señal → bloquear Long  en cycle_id+1

SALIDAS PERSONALIZADAS (SALIDAS_PERSONALIZADAS=True):
  - Exit Long  : RSI >= 70
  - Exit Short : RSI <= 30
  (SL fijo de exits.py se mantiene como backstop de emergencia)

RANGOS OPTUNA (4 parámetros):
  fast     : int   [14, 70]  step=2
  slow     : int   [100,250] step=5
  lookbar  : int   [3,  10]  step=1
  dist_pct : float [0.1,1.0] step=0.1

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID1ZLEMA(EstrategiaBase):

    combinacion_id: int          = 1
    name: str                    = "ZLEMA"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        4 parámetros Optuna.
        fast máx=70, slow mín=100 → gap mínimo 30, imposible colisión.
        """
        return {
            "fast":     trial.suggest_int  ("fast",     5,  30, step=1),
            "slow":     trial.suggest_int  ("slow",    50, 250, step=5),
            "lookbar":  trial.suggest_int  ("lookbar",   3,  10, step=1),
            "dist_pct": trial.suggest_float("dist_pct", 0.1, 1.0, step=0.1),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) HL2
          B) ZLEMA fast + slow
          C) Diferencial + ciclo (cycle_id)
          D) bars_in_cycle + distancia normalizada
          E) Expansión ok
          F) Condiciones base raw (sin memoria)
          G) One-shot por ciclo
          H) MEMORIA DE CICLO:
             - close_señal: close en la vela de señal
             - close_final: close en la última vela del ciclo
             - Evaluar si fue perdedor
             - Propagar bloqueo al ciclo siguiente
          I) Aplicar bloqueo + flancos + finalize_signals
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "high", "low", "close"])

        fast     = int  (params.get("fast",      30))
        slow     = int  (params.get("slow",     150))
        lookbar  = int  (params.get("lookbar",    5))
        dist_pct = float(params.get("dist_pct",  0.3))

        params["__warmup_bars"]     = slow + fast + 20
        params["__indicators_used"] = ["hl2", "fast_ma", "slow_ma", "dist_norm_pct", "rsi"]
        params["__indicator_specs"] = {
            "hl2":           {"panel": "main", "color": "#AAAAAA", "tipo": "line"},
            "fast_ma":       {"panel": "main", "color": "#00FFFF", "tipo": "line"},
            "slow_ma":       {"panel": "main", "color": "#FF00FF", "tipo": "line"},
            "dist_norm_pct": {"panel": "sub1", "color": "#FFD600", "tipo": "line"},
            "rsi":           {"panel": "sub2", "color": "#26A69A", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "dist_norm_pct": {"lo": 0.0, "hi": None, "mid": dist_pct},
            "rsi":           {"lo": 0.0, "hi": 100.0, "mid": 50.0},
        }

        q = df.lazy()

        # ── FASE A: HL2 ────────────────────────────────────────────────────────
        q = q.with_columns([
            ((pl.col("high") + pl.col("low")) / 2.0).alias("hl2")
        ])

        # ── FASE B: ZLEMA fast + slow ──────────────────────────────────────────
        lag_fast = max(1, (fast - 1) // 2)
        lag_slow = max(1, (slow - 1) // 2)

        q = q.with_columns([
            (pl.col("hl2") + (pl.col("hl2") - pl.col("hl2").shift(lag_fast)))
            .ewm_mean(span=fast, adjust=False)
            .alias("fast_ma"),

            (pl.col("hl2") + (pl.col("hl2") - pl.col("hl2").shift(lag_slow)))
            .ewm_mean(span=slow, adjust=False)
            .alias("slow_ma"),
        ])

        # ── FASE C: Diferencial + Ciclo ───────────────────────────────────────
        q = q.with_columns([
            (pl.col("fast_ma") - pl.col("slow_ma")).alias("_diff")
        ])

        q = q.with_columns([
            (pl.col("_diff") > 0.0).fill_null(False).alias("_is_bullish")
        ])

        q = q.with_columns([
            (
                pl.col("_is_bullish") != pl.col("_is_bullish").shift(1).fill_null(True)
            ).cast(pl.Int32).cum_sum().alias("cycle_id")
        ])

        # ── FASE D: bars_in_cycle + distancia normalizada ──────────────────────
        q = q.with_columns([
            (
                pl.col("cycle_id").rank(method="ordinal").over("cycle_id") - 1
            ).alias("bars_in_cycle")
        ])

        q = q.with_columns([
            pl.when(pl.col("hl2").abs() > 1e-12)
            .then(pl.col("_diff").abs() / pl.col("hl2") * 100.0)
            .otherwise(0.0)
            .alias("dist_norm_pct")
        ])

        # ── RSI (filtro de entrada + salida personalizada) ───────────────────
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=14).alias("rsi")
        ])

        q = q.with_columns([
            pl.col("dist_norm_pct").first().over("cycle_id").alias("_dist_init")
        ])

        # ── FASE E: Expansión ok ───────────────────────────────────────────────
        q = q.with_columns([
            (
                pl.col("dist_norm_pct") >= (pl.col("_dist_init") + dist_pct)
            ).fill_null(False).alias("_expansion_ok")
        ])

        # ── FASE F: Condiciones base raw (sin memoria aún) ────────────────────
        q = q.with_columns([
            (
                pl.col("_is_bullish") &
                (pl.col("bars_in_cycle") <= lookbar) &
                pl.col("_expansion_ok") &
                pl.col("fast_ma").is_not_null()
            ).fill_null(False).alias("_cond_long_raw"),

            (
                (~pl.col("_is_bullish")) &
                (pl.col("bars_in_cycle") <= lookbar) &
                pl.col("_expansion_ok") &
                pl.col("fast_ma").is_not_null()
            ).fill_null(False).alias("_cond_short_raw"),
        ])

        # ── FASE G: One-shot por ciclo ─────────────────────────────────────────
        q = q.with_columns([
            (
                pl.col("_cond_long_raw") &
                (pl.col("_cond_long_raw").cast(pl.Int32).cum_sum().over("cycle_id") == 1)
            ).fill_null(False).alias("_sig_long_raw"),

            (
                pl.col("_cond_short_raw") &
                (pl.col("_cond_short_raw").cast(pl.Int32).cum_sum().over("cycle_id") == 1)
            ).fill_null(False).alias("_sig_short_raw"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE H — MEMORIA DE CICLO
        #
        # Para cada ciclo con señal real:
        #   close_señal = close en la vela donde se emitió la señal
        #   close_final = close en la ÚLTIMA vela del ciclo (antes del cruce)
        #
        # Long perdedor:  close_final < close_señal → bloquear Short en cycle_id+1
        # Short perdedor: close_final > close_señal → bloquear Long  en cycle_id+1
        #
        # Implementación vectorial:
        #   1. close_señal via first() where señal == True over cycle_id
        #      (si no hay señal en el ciclo → null)
        #   2. close_final via last(close) over cycle_id
        #   3. Determinar si fue perdedor por tipo de señal
        #   4. shift(1) del cycle_id para propagar al ciclo siguiente
        # ══════════════════════════════════════════════════════════════════════

        # Precio de cierre en la vela de señal (null si no hubo señal)
        q = q.with_columns([
            pl.when(pl.col("_sig_long_raw"))
            .then(pl.col("close"))
            .otherwise(None)
            .first().over("cycle_id")
            .alias("_close_entrada_long"),

            pl.when(pl.col("_sig_short_raw"))
            .then(pl.col("close"))
            .otherwise(None)
            .first().over("cycle_id")
            .alias("_close_entrada_short"),
        ])

        # Precio de cierre en la última vela del ciclo
        q = q.with_columns([
            pl.col("close").last().over("cycle_id").alias("_close_final")
        ])

        # Determinar si el ciclo fue perdedor (solo para ciclos con señal)
        q = q.with_columns([
            # Long perdedor: entramos Long pero close_final < close_entrada
            pl.when(pl.col("_close_entrada_long").is_not_null())
            .then(pl.col("_close_final") < pl.col("_close_entrada_long"))
            .otherwise(False)
            .alias("_long_perdedor"),

            # Short perdedor: entramos Short pero close_final > close_entrada
            pl.when(pl.col("_close_entrada_short").is_not_null())
            .then(pl.col("_close_final") > pl.col("_close_entrada_short"))
            .otherwise(False)
            .alias("_short_perdedor"),
        ])

        # Tomar un valor por ciclo (first es suficiente, es constante en el ciclo)
        q = q.with_columns([
            pl.col("_long_perdedor") .first().over("cycle_id").alias("_long_perdedor_ciclo"),
            pl.col("_short_perdedor").first().over("cycle_id").alias("_short_perdedor_ciclo"),
        ])

        # Propagar bloqueo al ciclo siguiente via shift sobre cycle_id
        # _bloquear_short en cycle_id N = _long_perdedor en cycle_id N-1
        # _bloquear_long  en cycle_id N = _short_perdedor en cycle_id N-1
        q = q.with_columns([
            pl.col("_long_perdedor_ciclo") .shift(1).fill_null(False).over("cycle_id").alias("_bloquear_long_prev"),
            pl.col("_short_perdedor_ciclo").shift(1).fill_null(False).over("cycle_id").alias("_bloquear_short_prev"),
        ])

        # El bloqueo aplica a todo el ciclo actual — propagar con first()
        # Nota: el shift anterior da el valor del ciclo anterior en la primera vela
        q = q.with_columns([
            # Si cycle_id cambió, la primera vela tiene el valor del ciclo anterior
            pl.when(pl.col("bars_in_cycle") == 0)
            .then(pl.col("_bloquear_long_prev"))
            .otherwise(None)
            .first().over("cycle_id")
            .fill_null(False)
            .alias("_bloquear_long"),

            pl.when(pl.col("bars_in_cycle") == 0)
            .then(pl.col("_bloquear_short_prev"))
            .otherwise(None)
            .first().over("cycle_id")
            .fill_null(False)
            .alias("_bloquear_short"),
        ])

        # ── FASE I: Aplicar bloqueo + flancos + finalize ───────────────────────
        q = q.with_columns([
            (
                pl.col("_sig_long_raw") &
                ~pl.col("_bloquear_long") &
                (pl.col("rsi") <= 70.0)
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_sig_short_raw") &
                ~pl.col("_bloquear_short") &
                (pl.col("rsi") >= 30.0)
            ).fill_null(False).alias("_cond_short"),
        ])

        q = q.with_columns([
            self._as_bool(
                pl.col("_cond_long") &
                ~pl.col("_cond_long").shift(1).fill_null(False)
            ).alias("signal_long"),

            self._as_bool(
                pl.col("_cond_short") &
                ~pl.col("_cond_short").shift(1).fill_null(False) &
                ~pl.col("_cond_long")
            ).alias("signal_short"),

            # Salidas personalizadas por RSI
            self._as_bool(pl.col("rsi") >= 70.0).alias("exit_long"),
            self._as_bool(pl.col("rsi") <= 30.0).alias("exit_short"),
        ])

        return self.finalize_signals(
            q,
            keep_cols=[
                "hl2",
                "fast_ma",
                "slow_ma",
                "dist_norm_pct",
                "rsi",
                "cycle_id",
                "bars_in_cycle",
                "exit_long",
                "exit_short",
            ],
        )
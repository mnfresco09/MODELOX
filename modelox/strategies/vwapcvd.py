from __future__ import annotations
"""
================================================================================
MODELOX / STRATEGIES / VWAP.py
================================================================================
ID        : 4
NOMBRE    : VWAP-CVD - VWAP Distance + CVD Z-Score
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Implementa el indicador VWAP Distance usado como referencia:

    1. EWM-VWAP = EWM(close * volume, halflife=n) / EWM(volume, halflife=n)
    2. Distancia cruda = (close - EWM-VWAP) / EWM-VWAP
    3. Senal suavizada = EWM(distancia cruda, halflife=n)
    4. Clip robusto con estadisticos EWM lentos
    5. Z-score sobre la senal suavizada
    6. CVD normalizado = (taker_buy_volume - taker_sell_volume) / volumen total
    7. Z-score robusto del CVD con la misma logica EWM
    8. Score de entrada = vwap_distance_z + cvd_z

  Si el feed trae volumen total cero para todo el rango, usa peso 1 por vela.
  Esto mantiene el indicador operativo en activos sin volumen informado.

LOGICA DE ENTRADA:
  LONG  : vwap_cvd_z_sum cruza al alza +umbral_inicio_tendencia
  SHORT : vwap_cvd_z_sum cruza a la baja -umbral_inicio_tendencia

PARAMETROS OPTUNA:
  halflife_bars             : int   [5, 100] step=5
  normalization_multiplier  : float [2.0, 5.0] step=0.5
  vwap_clip_sigmas          : float [2.0, 4.0] step=0.5
  umbral_inicio_tendencia   : float [0.5, 3.0] step=0.1

SALIDAS: estandar (motor de exits)
================================================================================
"""

from typing import Any, Dict, List, Optional

import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


VWAP_DEFAULT_HALFLIFE_BARS = 15
VWAP_DEFAULT_NORMALIZATION_MULTIPLIER = 3.0
VWAP_DEFAULT_CLIP_SIGMAS = 2.5
VWAP_DEFAULT_UMBRAL_INICIO_TENDENCIA = 1.0
VWAP_VOLUME_EPSILON = 1e-7
VWAP_SCORE_SCALE_FLOOR = 0.1


def _ewm_mean_expr(expr: pl.Expr, *, half_life: float) -> pl.Expr:
    return expr.ewm_mean(half_life=float(half_life), adjust=False, min_periods=1)


def _ewm_std_expr(expr: pl.Expr, *, half_life: float) -> pl.Expr:
    mean = _ewm_mean_expr(expr, half_life=half_life)
    mean_sq = _ewm_mean_expr(expr * expr, half_life=half_life)
    return (mean_sq - (mean * mean)).clip_min(0.0).sqrt()


def _clip_expr(expr: pl.Expr, lower: float, upper: float) -> pl.Expr:
    return pl.when(expr < lower).then(lower).when(expr > upper).then(upper).otherwise(expr)


class EstrategiaID4VWAP(EstrategiaBase):
    """
    VWAP-CVD - Entrada por suma de z-scores VWAP Distance + CVD.
    """

    combinacion_id: int = 4
    name: str = "VWAP-CVD"
    SALIDAS_PERSONALIZADAS: bool = False
    COLUMNAS_REQUERIDAS: List[str] = ["taker_buy_volume", "taker_sell_volume"]
    timeframe_entry: Optional[str] = None
    timeframe_exit: Optional[str] = None

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "halflife_bars": trial.suggest_int(
                "halflife_bars",
                5,
                100,
                step=5,
            ),
            "normalization_multiplier": trial.suggest_float(
                "normalization_multiplier",
                2.0,
                5.0,
                step=0.5,
            ),
            "vwap_clip_sigmas": trial.suggest_float(
                "vwap_clip_sigmas",
                2.0,
                4.0,
                step=0.5,
            ),
            "umbral_inicio_tendencia": trial.suggest_float(
                "umbral_inicio_tendencia",
                0.5,
                3.0,
                step=0.1,
            ),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Calcular EWM-VWAP
          B) Calcular distancia, senal suavizada y clip robusto
          C) Calcular CVD, senal suavizada y z-score robusto
          D) Sumar z-scores VWAP + CVD
          E) Detectar cruces de la suma contra +/- umbral
          F) finalize_signals
        """

        self._init_params_metadata(params)
        self._require_columns(
            df,
            [
                "timestamp",
                "close",
                "volume",
                "taker_buy_volume",
                "taker_sell_volume",
            ],
        )

        halflife_bars = max(
            1,
            int(params.get("halflife_bars", VWAP_DEFAULT_HALFLIFE_BARS)),
        )
        normalization_multiplier = max(
            0.1,
            float(
                params.get(
                    "normalization_multiplier",
                    VWAP_DEFAULT_NORMALIZATION_MULTIPLIER,
                )
            ),
        )
        vwap_clip_sigmas = max(
            0.1,
            float(
                params.get(
                    "vwap_clip_sigmas",
                    params.get("VWAP_CLIP_SIGMAS", VWAP_DEFAULT_CLIP_SIGMAS),
                )
            ),
        )
        umbral_inicio_tendencia = max(
            0.0,
            float(
                params.get(
                    "umbral_inicio_tendencia",
                    VWAP_DEFAULT_UMBRAL_INICIO_TENDENCIA,
                )
            ),
        )

        normalization_halflife = max(1.0, float(halflife_bars) * normalization_multiplier)
        params["__warmup_bars"] = max(
            halflife_bars * 9,
            int(round(normalization_halflife * 3.0)),
        )
        params["__indicators_used"] = [
            "vwap",
            "vwap_distance_raw",
            "vwap_distance_signal",
            "vwap_distance_z",
            "vwap_distance_score",
            "cvd_delta_raw",
            "cvd_rolling_14",
            "cvd_signal",
            "cvd_z",
            "vwap_cvd_z_sum",
        ]
        params["__indicator_specs"] = {
            "vwap": {"panel": "main", "color": "#00BCD4", "tipo": "line"},
            "vwap_distance_raw": {"panel": "sub1", "color": "#78909C", "tipo": "line"},
            "vwap_distance_signal": {"panel": "sub1", "color": "#F59E0B", "tipo": "line"},
            "vwap_distance_z": {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
            "vwap_distance_score": {"panel": "sub3", "color": "#00E676", "tipo": "line"},
            "cvd_delta_raw": {"panel": "sub1", "color": "#26A69A", "tipo": "histogram"},
            "cvd_rolling_14": {"panel": "sub1", "color": "#4DB6AC", "tipo": "line"},
            "cvd_signal": {"panel": "sub2", "color": "#42A5F5", "tipo": "line"},
            "cvd_z": {"panel": "sub2", "color": "#EF5350", "tipo": "line"},
            "vwap_cvd_z_sum": {"panel": "sub3", "color": "#FFD600", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "vwap_cvd_z_sum": {
                "lo": -umbral_inicio_tendencia,
                "hi": umbral_inicio_tendencia,
                "mid": 0.0,
            },
            "vwap_distance_score": {
                "lo": None,
                "hi": None,
                "mid": 0.0,
            },
            "vwap_distance_z": {"lo": None, "hi": None, "mid": 0.0},
            "vwap_distance_signal": {"lo": None, "hi": None, "mid": 0.0},
            "cvd_delta_raw": {"lo": -1.0, "hi": 1.0, "mid": 0.0},
            "cvd_z": {"lo": None, "hi": None, "mid": 0.0},
        }

        q = df.lazy()

        # FASE A: EWM-VWAP. Si no hay volumen informado en el rango, cada vela
        # pesa 1 para evitar un denominador permanentemente cero.
        q = q.with_columns([
            pl.col("close").cast(pl.Float64).clip_min(0.0).alias("_price"),
            pl.col("volume").cast(pl.Float64).clip_min(0.0).alias("_volume"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_volume").sum() > VWAP_VOLUME_EPSILON)
            .then(pl.col("_volume"))
            .otherwise(1.0)
            .alias("_volume_eff"),
        ])

        q = q.with_columns([
            (pl.col("_price") * pl.col("_volume_eff")).alias("_price_volume"),
        ])

        q = q.with_columns([
            _ewm_mean_expr(pl.col("_price_volume"), half_life=halflife_bars)
            .alias("_ewm_price_volume"),
            _ewm_mean_expr(pl.col("_volume_eff"), half_life=halflife_bars)
            .alias("_ewm_volume"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_ewm_volume") >= VWAP_VOLUME_EPSILON)
            .then(pl.col("_ewm_price_volume") / pl.col("_ewm_volume"))
            .otherwise(0.0)
            .alias("vwap")
        ])

        # FASE B: distancia y senal suavizada del indicador de referencia.
        q = q.with_columns([
            pl.when((pl.col("_price") > 0.0) & (pl.col("vwap") > 0.0))
            .then((pl.col("_price") - pl.col("vwap")) / pl.col("vwap"))
            .otherwise(0.0)
            .alias("vwap_distance_raw")
        ])

        q = q.with_columns([
            _ewm_mean_expr(pl.col("vwap_distance_raw"), half_life=halflife_bars)
            .alias("vwap_distance_signal")
        ])

        q = q.with_columns([
            _ewm_mean_expr(
                pl.col("vwap_distance_signal"),
                half_life=normalization_halflife,
            ).alias("_distance_raw_mean"),
            _ewm_std_expr(
                pl.col("vwap_distance_signal"),
                half_life=normalization_halflife,
            ).alias("_distance_raw_std"),
        ])

        q = q.with_columns([
            (pl.col("_distance_raw_mean") - vwap_clip_sigmas * pl.col("_distance_raw_std"))
            .alias("_clip_lower"),
            (pl.col("_distance_raw_mean") + vwap_clip_sigmas * pl.col("_distance_raw_std"))
            .alias("_clip_upper"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_distance_raw_std") <= 0.0)
            .then(pl.col("vwap_distance_signal"))
            .when(pl.col("vwap_distance_signal") < pl.col("_clip_lower"))
            .then(pl.col("_clip_lower"))
            .when(pl.col("vwap_distance_signal") > pl.col("_clip_upper"))
            .then(pl.col("_clip_upper"))
            .otherwise(pl.col("vwap_distance_signal"))
            .alias("_clipped_signal")
        ])

        # FASE C1: z-score VWAP y score legacy acotado para visualizacion.
        q = q.with_columns([
            _ewm_mean_expr(
                pl.col("_clipped_signal"),
                half_life=normalization_halflife,
            ).alias("_distance_final_mean"),
            _ewm_std_expr(
                pl.col("_clipped_signal"),
                half_life=normalization_halflife,
            ).alias("_distance_final_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_distance_final_std") > 0.0)
            .then(
                (pl.col("vwap_distance_signal") - pl.col("_distance_final_mean"))
                / pl.col("_distance_final_std")
            )
            .otherwise(0.0)
            .alias("vwap_distance_z")
        ])

        q = q.with_columns([
            _ewm_mean_expr(
                pl.col("vwap_distance_z"),
                half_life=normalization_halflife,
            ).alias("_z_mean"),
            _ewm_mean_expr(
                pl.col("vwap_distance_z") * pl.col("vwap_distance_z"),
                half_life=normalization_halflife,
            ).alias("_z_mean_sq"),
        ])

        q = q.with_columns([
            (pl.col("_z_mean_sq").shift(1) - (pl.col("_z_mean").shift(1) * pl.col("_z_mean").shift(1)))
            .clip_min(0.0)
            .sqrt()
            .fill_null(0.0)
            .alias("_z_scale")
        ])

        q = q.with_columns([
            (
                (pl.col("vwap_distance_z") / pl.max_horizontal(pl.col("_z_scale"), pl.lit(VWAP_SCORE_SCALE_FLOOR)))
                .tanh()
                * 3.0
            ).alias("vwap_distance_score")
        ])

        # FASE C2: CVD normalizado por vela, usando el volumen taker del kline.
        q = q.with_columns([
            pl.col("taker_buy_volume").cast(pl.Float64).fill_null(0.0).fill_nan(0.0).clip_min(0.0)
            .alias("_cvd_buy_volume"),
            pl.col("taker_sell_volume").cast(pl.Float64).fill_null(0.0).fill_nan(0.0).clip_min(0.0)
            .alias("_cvd_sell_volume"),
        ])

        q = q.with_columns([
            pl.max_horizontal(
                pl.col("_volume"),
                pl.col("_cvd_buy_volume") + pl.col("_cvd_sell_volume"),
            ).alias("_cvd_total_volume")
        ])

        q = q.with_columns([
            _clip_expr(
                pl.when(pl.col("_cvd_total_volume") > VWAP_VOLUME_EPSILON)
                .then((pl.col("_cvd_buy_volume") - pl.col("_cvd_sell_volume")) / pl.col("_cvd_total_volume"))
                .otherwise(0.0),
                -1.0,
                1.0,
            ).alias("cvd_delta_raw")
        ])

        q = q.with_columns([
            pl.col("cvd_delta_raw")
            .rolling_sum(window_size=14, min_periods=1)
            .alias("cvd_rolling_14"),
            _ewm_mean_expr(pl.col("cvd_delta_raw"), half_life=halflife_bars)
            .alias("cvd_signal"),
        ])

        q = q.with_columns([
            _ewm_mean_expr(
                pl.col("cvd_signal"),
                half_life=normalization_halflife,
            ).alias("_cvd_raw_mean"),
            _ewm_std_expr(
                pl.col("cvd_signal"),
                half_life=normalization_halflife,
            ).alias("_cvd_raw_std"),
        ])

        q = q.with_columns([
            (pl.col("_cvd_raw_mean") - vwap_clip_sigmas * pl.col("_cvd_raw_std"))
            .alias("_cvd_clip_lower"),
            (pl.col("_cvd_raw_mean") + vwap_clip_sigmas * pl.col("_cvd_raw_std"))
            .alias("_cvd_clip_upper"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_cvd_raw_std") <= 0.0)
            .then(pl.col("cvd_signal"))
            .when(pl.col("cvd_signal") < pl.col("_cvd_clip_lower"))
            .then(pl.col("_cvd_clip_lower"))
            .when(pl.col("cvd_signal") > pl.col("_cvd_clip_upper"))
            .then(pl.col("_cvd_clip_upper"))
            .otherwise(pl.col("cvd_signal"))
            .alias("_cvd_clipped_signal")
        ])

        q = q.with_columns([
            _ewm_mean_expr(
                pl.col("_cvd_clipped_signal"),
                half_life=normalization_halflife,
            ).alias("_cvd_final_mean"),
            _ewm_std_expr(
                pl.col("_cvd_clipped_signal"),
                half_life=normalization_halflife,
            ).alias("_cvd_final_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_cvd_final_std") > 0.0)
            .then((pl.col("cvd_signal") - pl.col("_cvd_final_mean")) / pl.col("_cvd_final_std"))
            .otherwise(0.0)
            .alias("cvd_z")
        ])

        # FASE D: score interpretable como suma directa de z-scores.
        q = q.with_columns([
            (pl.col("vwap_distance_z") + pl.col("cvd_z")).alias("vwap_cvd_z_sum")
        ])

        # FASE E: entradas por cruce de la suma VWAP-CVD contra el umbral.
        q = q.with_columns([
            pl.col("vwap_cvd_z_sum").shift(1).alias("_score_prev")
        ])

        q = q.with_columns([
            self._as_bool(
                (pl.col("_score_prev") <= umbral_inicio_tendencia)
                & (pl.col("vwap_cvd_z_sum") > umbral_inicio_tendencia)
                & pl.col("_score_prev").is_not_null()
            ).alias("signal_long"),

            self._as_bool(
                (pl.col("_score_prev") >= -umbral_inicio_tendencia)
                & (pl.col("vwap_cvd_z_sum") < -umbral_inicio_tendencia)
                & pl.col("_score_prev").is_not_null()
            ).alias("signal_short"),
        ])

        return self.finalize_signals(
            q,
            keep_cols=[
                "vwap",
                "vwap_distance_raw",
                "vwap_distance_signal",
                "vwap_distance_z",
                "vwap_distance_score",
                "cvd_delta_raw",
                "cvd_rolling_14",
                "cvd_signal",
                "cvd_z",
                "vwap_cvd_z_sum",
            ],
        )

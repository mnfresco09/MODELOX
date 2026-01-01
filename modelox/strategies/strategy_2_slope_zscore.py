from __future__ import annotations

"""Strategy 2 — SuperGolay Kinematics (Early Momentum)

Objetivo
- Capturar momentum temprano mediante cinemática en log-precio.

Indicador (causal)
- p_t = ln(close)
- Savitzky–Golay causal (ventana solo hacia atrás)
- v_t: 1ª derivada local
- a_t: 2ª derivada local
- Z_v, Z_a: z-score rolling (N)
- sigma_eps: std de residuos de regresión lineal local sobre p_t (misma N)

Score
- score = 0.5*Z_v + 0.4*Z_a - 0.3*sigma_eps

Triggers
- LONG  : score>thr, Z_v>0, Z_a>za_long_min
- SHORT : score<-thr, Z_v<0, Z_a<za_short_max

Nota
- Las salidas (SL/TP intra-vela) son globales y viven en el engine.
"""

from typing import Any, Dict

import polars as pl

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_supergolay


class Strategy2SuperGolayKinematics:
    combinacion_id = 2
    name = "SuperGolay2_EarlyMomentum"

    __indicators_used = ["supergolay", "supergolay_score"]

    # Defaults (align with user spec)
    WINDOW_LENGTH = 21
    POLYORDER = 3
    ZSCORE_WINDOW = 100
    NOISE_WINDOW = 50

    W_V = 0.5
    W_A = 0.4
    W_NOISE = 0.3

    SCORE_THRESHOLD = 1.0
    ZA_CONVEXITY = 0.2

    # Optional: used by ejecutar.py to display which params exist for Optuna.
    # Keep as dict (may be empty) to match the expected interface across strategies.
    parametros_optuna = {}

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # 1) Parámetros del filtro (siempre impar)
            "window_length": int(trial.suggest_int("window_length", 5, 51, step=2)),
            "polyorder": int(trial.suggest_int("polyorder", 2, 4)),

            # 2) Ventanas estadísticas
            "zscore_window": int(trial.suggest_int("zscore_window", 20, 200)),
            "noise_window": int(trial.suggest_int("noise_window", 10, 50)),

            # 3) Ponderación del score
            "w_v": float(trial.suggest_float("w_v", 0.1, 1.0, step=0.05)),
            "w_a": float(trial.suggest_float("w_a", 0.1, 1.0, step=0.05)),
            "w_noise": float(trial.suggest_float("w_noise", 0.0, 1.0, step=0.05)),

            # 4) Umbrales (simétricos)
            "score_threshold": float(trial.suggest_float("score_threshold", 0.5, 2.0, step=0.05)),
            "za_convexity": float(trial.suggest_float("za_convexity", -0.5, 0.5, step=0.05)),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        window_length = int(params.get("window_length", self.WINDOW_LENGTH))
        if window_length < 5:
            window_length = 5
        if window_length % 2 == 0:
            window_length += 1

        polyorder = int(params.get("polyorder", self.POLYORDER))
        if polyorder < 2:
            polyorder = 2
        if polyorder > 4:
            polyorder = 4
        if polyorder >= window_length:
            polyorder = max(2, min(4, window_length - 1))

        zscore_window = int(params.get("zscore_window", self.ZSCORE_WINDOW))
        if zscore_window < 20:
            zscore_window = 20
        noise_window = int(params.get("noise_window", self.NOISE_WINDOW))
        if noise_window < 10:
            noise_window = 10

        score_threshold = float(params.get("score_threshold", self.SCORE_THRESHOLD))
        if score_threshold < 0:
            score_threshold = -score_threshold

        za_convexity = float(params.get("za_convexity", self.ZA_CONVEXITY))

        w_v = float(params.get("w_v", self.W_V))
        w_a = float(params.get("w_a", self.W_A))
        w_noise = float(params.get("w_noise", self.W_NOISE))

        # Keep weights sane; avoid flipping the score unintentionally.
        if w_v < 0:
            w_v = 0.0
        if w_a < 0:
            w_a = 0.0
        if w_noise < 0:
            w_noise = 0.0

        # Warmup: SG window + statistical windows
        params["__warmup_bars"] = max(window_length, zscore_window, noise_window) + 10

        # Expose score thresholds for plot reference lines (±threshold)
        params["supergolay_score_threshold_hi"] = score_threshold
        params["supergolay_score_threshold_lo"] = -score_threshold

        df = IndicadorFactory.procesar(
            df,
            {
                "supergolay": cfg_supergolay(
                    col="close",
                    window=window_length,
                    polyorder=polyorder,
                    zscore_window=zscore_window,
                    noise_window=noise_window,
                    w_v=w_v,
                    w_a=w_a,
                    w_noise=w_noise,
                    out_smooth="supergolay",
                    out_score="supergolay_score",
                    out_zv="supergolay_zv",
                    out_za="supergolay_za",
                    out_noise="supergolay_noise",
                )
            },
        )

        score = pl.col("supergolay_score")
        zv = pl.col("supergolay_zv")
        za = pl.col("supergolay_za")

        # Symmetric trial requirements:
        # - score uses ±score_threshold
        # - convexity uses ±za_convexity around 0:
        #     LONG  requires Z_a > -za_convexity
        #     SHORT requires Z_a < +za_convexity
        signal_long = ((score > score_threshold) & (zv > 0) & (za > (-za_convexity))).fill_null(False)
        signal_short = ((score < (-score_threshold)) & (zv < 0) & (za < (za_convexity))).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

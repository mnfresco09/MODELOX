from __future__ import annotations

"""Strategy 4 — SuperGolay Cross + Z-Score Extremes (Symmetric)

Lógica (simétrica por trial)
- LONG  cuando ocurre cruce AL ALZA: close cruza por encima de la línea `supergolay`
         Y zscore < -z_threshold
- SHORT cuando ocurre cruce A LA BAJA: close cruza por debajo de la línea `supergolay`
         Y zscore > +z_threshold

Notas
- `supergolay` es causal (solo mira hacia atrás).
- z_threshold se fuerza a ser positivo y se aplica simétricamente.
- Las salidas (SL/TP intra-vela) son globales y viven en el engine.
"""

from typing import Any, Dict

import polars as pl

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_supergolay, cfg_zscore


class Strategy4SuperGolayZScoreCross:
    combinacion_id = 4
    name = "SuperGolay4_Cross_ZScoreExtremes"

    __indicators_used = ["supergolay", "zscore"]

    # SuperGolay defaults
    WINDOW_LENGTH = 21
    POLYORDER = 3
    ZSCORE_WINDOW = 100
    NOISE_WINDOW = 50

    # ZScore defaults (of close)
    Z_WINDOW = 100
    Z_THRESHOLD = 2.0

    parametros_optuna = {}

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # Zscore settings
            "z_window": int(trial.suggest_int("z_window", 20, 200)),
            "z_threshold": float(trial.suggest_float("z_threshold", 1.0, 4.0, step=0.1)),

            # Optional: allow tuning the SuperGolay line
            "window_length": int(trial.suggest_int("window_length", 5, 51, step=2)),
            "polyorder": int(trial.suggest_int("polyorder", 2, 4)),
            "zscore_window": int(trial.suggest_int("zscore_window", 20, 200)),
            "noise_window": int(trial.suggest_int("noise_window", 10, 50)),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        # --- SuperGolay params (enforce constraints) ---
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

        sg_zscore_window = int(params.get("zscore_window", self.ZSCORE_WINDOW))
        if sg_zscore_window < 20:
            sg_zscore_window = 20

        noise_window = int(params.get("noise_window", self.NOISE_WINDOW))
        if noise_window < 10:
            noise_window = 10

        # --- ZScore params (symmetry) ---
        z_window = int(params.get("z_window", self.Z_WINDOW))
        if z_window < 2:
            z_window = 2

        z_threshold = float(params.get("z_threshold", self.Z_THRESHOLD))
        if z_threshold < 0:
            z_threshold = -z_threshold

        # Warmup: cover SG + stats + zscore window
        params["__warmup_bars"] = max(window_length, sg_zscore_window, noise_window, z_window) + 10

        # Expose thresholds for plot reference lines (±z_threshold) on zscore panel
        params["z_threshold_upper"] = z_threshold
        params["z_threshold_lower"] = -z_threshold

        df = IndicadorFactory.procesar(
            df,
            {
                "supergolay": cfg_supergolay(
                    col="close",
                    window=window_length,
                    polyorder=polyorder,
                    zscore_window=sg_zscore_window,
                    noise_window=noise_window,
                    out_smooth="supergolay",
                    out_score="supergolay_score",
                    out_zv="supergolay_zv",
                    out_za="supergolay_za",
                    out_noise="supergolay_noise",
                ),
                "zscore": cfg_zscore(col="close", window=z_window, out="zscore"),
            },
        )

        close = pl.col("close")
        line = pl.col("supergolay")
        z = pl.col("zscore")

        # Cross definitions
        cross_up = (close.shift(1) <= line.shift(1)) & (close > line)
        cross_down = (close.shift(1) >= line.shift(1)) & (close < line)

        signal_long = (cross_up & (z < (-z_threshold))).fill_null(False)
        signal_short = (cross_down & (z > (z_threshold))).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

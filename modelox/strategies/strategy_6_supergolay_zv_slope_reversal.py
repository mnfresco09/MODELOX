from __future__ import annotations

"""Strategy 6 — SuperGolay Velocity Z-Score + Slope

Idea ("física" del precio)
- Operar extremos estadísticos de la VELOCIDAD (Z_v) en vez del precio.
- Confirmar giro temprano solo con la pendiente (slope) de regresión.

Lógica (simétrica por trial)
- LONG  si:
    1) supergolay_zv < -zv_threshold
    2) linreg_slope(supergolay) era negativo y empieza a crecer (diff > 0)
- SHORT si:
    1) supergolay_zv > +zv_threshold
    2) linreg_slope(supergolay) era positivo y empieza a decrecer (diff < 0)

Notas
- `supergolay` es causal (solo mira hacia atrás).
- zv_threshold se fuerza a ser positivo y se aplica simétricamente.
- Las salidas (SL/TP intra-vela) son globales y viven en el engine.
"""

from typing import Any, Dict

import polars as pl

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_linreg_slope, cfg_supergolay


class Strategy6SuperGolayZV_SlopeReversal:
    combinacion_id = 6
    name = "SuperGolay6_ZV_Slope"

    __indicators_used = ["supergolay", "supergolay_zv", "linreg_slope"]

    # SuperGolay defaults
    WINDOW_LENGTH = 21
    POLYORDER = 3
    ZSCORE_WINDOW = 100
    NOISE_WINDOW = 50

    # Signal defaults
    ZV_THRESHOLD = 2.0
    SLOPE_WINDOW = 20

    parametros_optuna = {}

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # --- SuperGolay smoothing ---
            "window_length": int(trial.suggest_int("window_length", 5, 51, step=2)),
            "polyorder": int(trial.suggest_int("polyorder", 2, 4)),

            # --- Windows for stats/regression ---
            "zscore_window": int(trial.suggest_int("zscore_window", 20, 200)),
            "slope_window": int(trial.suggest_int("slope_window", 5, 60)),

            # --- Thresholds (symmetric) ---
            "zv_threshold": float(trial.suggest_float("zv_threshold", 1.0, 4.0, step=0.1)),
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

        zscore_window = int(params.get("zscore_window", self.ZSCORE_WINDOW))
        if zscore_window < 20:
            zscore_window = 20

        noise_window = int(params.get("noise_window", self.NOISE_WINDOW))
        if noise_window < 10:
            noise_window = 10

        # --- Slope params ---
        slope_window = int(params.get("slope_window", self.SLOPE_WINDOW))
        if slope_window < 3:
            slope_window = 3

        # --- ZV threshold (symmetric) ---
        zv_threshold = float(params.get("zv_threshold", self.ZV_THRESHOLD))
        if zv_threshold < 0:
            zv_threshold = -zv_threshold

        # Warmup
        params["__warmup_bars"] = max(window_length, zscore_window, noise_window, slope_window) + 10

        # Expose bounds for plotting (±zv_threshold)
        params["supergolay_zv_threshold_hi"] = zv_threshold
        params["supergolay_zv_threshold_lo"] = -zv_threshold

        df = IndicadorFactory.procesar(
            df,
            {
                "supergolay": cfg_supergolay(
                    col="close",
                    window=window_length,
                    polyorder=polyorder,
                    zscore_window=zscore_window,
                    noise_window=noise_window,
                    out_smooth="supergolay",
                    out_score="supergolay_score",
                    out_zv="supergolay_zv",
                    out_za="supergolay_za",
                    out_noise="supergolay_noise",
                ),
                # Slope on the SuperGolay line (price proxy)
                "linreg_slope": cfg_linreg_slope(window=slope_window, col="supergolay", out="linreg_slope"),
            },
        )

        zv = pl.col("supergolay_zv")
        sl = pl.col("linreg_slope")

        # Strictly "z-score + slope": require slope sign and its first difference direction.
        long_turn = (sl.shift(1) < 0) & (sl.diff() > 0)
        short_turn = (sl.shift(1) > 0) & (sl.diff() < 0)

        signal_long = ((zv < (-zv_threshold)) & long_turn).fill_null(False)
        signal_short = ((zv > (zv_threshold)) & short_turn).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

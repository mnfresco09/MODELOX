from __future__ import annotations

"""Strategy 3 — SuperGolay Line + ADX Filter

Entradas
- LONG  cuando close > supergolay y ADX > adx_threshold
- SHORT cuando close < supergolay y ADX > adx_threshold

Notas
- La línea `supergolay` se calcula de forma causal (solo mira hacia atrás).
- Las salidas (SL/TP intra-vela) son globales y viven en el engine.
"""

from typing import Any, Dict

import polars as pl

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_adx, cfg_supergolay


class Strategy3SuperGolayLineADX:
    combinacion_id = 3
    name = "SuperGolay3_LineCross_ADX"

    __indicators_used = ["supergolay", "adx"]

    # SuperGolay defaults (can be extended later if you want)
    WINDOW_LENGTH = 21
    POLYORDER = 3
    ZSCORE_WINDOW = 100
    NOISE_WINDOW = 50

    # ADX defaults
    ADX_PERIOD = 14
    ADX_THRESHOLD = 20.0

    parametros_optuna = {}

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # ADX filter (parametrizable)
            "adx_period": int(trial.suggest_int("adx_period", 7, 25)),
            "adx_threshold": float(trial.suggest_float("adx_threshold", 10.0, 35.0, step=0.5)),

            # Optional: keep SuperGolay stable but allow small tuning if desired
            # (You can remove these from optimization if you truly only want ADX.)
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

        zscore_window = int(params.get("zscore_window", self.ZSCORE_WINDOW))
        if zscore_window < 20:
            zscore_window = 20

        noise_window = int(params.get("noise_window", self.NOISE_WINDOW))
        if noise_window < 10:
            noise_window = 10

        # --- ADX params ---
        adx_period = int(params.get("adx_period", self.ADX_PERIOD))
        if adx_period < 2:
            adx_period = 2

        adx_threshold = float(params.get("adx_threshold", self.ADX_THRESHOLD))
        if adx_threshold < 0:
            adx_threshold = -adx_threshold

        # Warmup: cover SG + stats + ADX
        params["__warmup_bars"] = max(window_length, zscore_window, noise_window, adx_period) + 10

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
                "adx": cfg_adx(period=adx_period, out_adx="adx", out_plus_di="plus_di", out_minus_di="minus_di"),
            },
        )

        close = pl.col("close")
        line = pl.col("supergolay")
        adx = pl.col("adx")

        adx_ok = (adx > adx_threshold)

        # As requested: close above line => long, close below line => short
        signal_long = (adx_ok & (close > line)).fill_null(False)
        signal_short = (adx_ok & (close < line)).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

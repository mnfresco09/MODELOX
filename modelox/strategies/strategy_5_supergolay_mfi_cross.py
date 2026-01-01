from __future__ import annotations

"""Strategy 5 — SuperGolay Cross + MFI Directional Filter

Lógica (MFI centrado en 50)
- LONG  cuando ocurre cruce AL ALZA: close cruza por encima de `supergolay`
                 Y MFI < 50 y MFI está creciendo
- SHORT cuando ocurre cruce A LA BAJA: close cruza por debajo de `supergolay`
                 Y MFI > 50 y MFI está decreciendo

Implementación
- Se usa mfi_c = (MFI - 50) para evaluar signo y pendiente:
    LONG  => mfi_c < 0 y diff(mfi_c) > 0
    SHORT => mfi_c > 0 y diff(mfi_c) < 0

Notas
- `supergolay` es causal (solo mira hacia atrás).
- Las salidas (SL/TP intra-vela) son globales y viven en el engine.
"""

from typing import Any, Dict

import polars as pl

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_mfi, cfg_supergolay


class Strategy5SuperGolayMFICross:
    combinacion_id = 5
    name = "SuperGolay5_Cross_MFI"

    __indicators_used = ["supergolay", "mfi"]

    # SuperGolay defaults
    WINDOW_LENGTH = 21
    POLYORDER = 3
    ZSCORE_WINDOW = 100
    NOISE_WINDOW = 50

    # MFI defaults
    MFI_PERIOD = 14

    parametros_optuna = {}

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # MFI
            "mfi_period": int(trial.suggest_int("mfi_period", 7, 30)),

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

        # --- MFI params ---
        mfi_period = int(params.get("mfi_period", self.MFI_PERIOD))
        if mfi_period < 2:
            mfi_period = 2

        # Warmup: cover SG + stats + MFI
        params["__warmup_bars"] = max(window_length, sg_zscore_window, noise_window, mfi_period) + 10

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
                "mfi": cfg_mfi(period=mfi_period, out="mfi"),
            },
        )

        close = pl.col("close")
        line = pl.col("supergolay")
        mfi = pl.col("mfi")

        # Center MFI around 0 for directional logic
        mfi_c = mfi - 50
        mfi_growing = mfi_c.diff() > 0
        mfi_falling = mfi_c.diff() < 0

        # Cross definitions
        cross_up = (close.shift(1) <= line.shift(1)) & (close > line)
        cross_down = (close.shift(1) >= line.shift(1)) & (close < line)

        signal_long = (cross_up & (mfi_c < 0) & mfi_growing).fill_null(False)
        signal_short = (cross_down & (mfi_c > 0) & mfi_falling).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

from __future__ import annotations

from typing import Any, Dict

import polars as pl


def _add_mfi(df: pl.DataFrame, *, period: int, out: str = "mfi") -> pl.DataFrame:
    high_col = "high"
    low_col = "low"
    close_col = "close"
    volume_col = "volume"

    required = {high_col, low_col, close_col, volume_col}
    if not required.issubset(set(df.columns)):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    period = max(2, int(period))
    tp = (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3.0
    prev_tp = tp.shift(1)

    raw_mf = tp * pl.col(volume_col)
    pos_mf = pl.when(tp > prev_tp).then(raw_mf).otherwise(0.0)
    neg_mf = pl.when(tp < prev_tp).then(raw_mf).otherwise(0.0)

    pos_sum = pos_mf.rolling_sum(window_size=period, min_periods=period)
    neg_sum = neg_mf.rolling_sum(window_size=period, min_periods=period)

    ratio = pos_sum / neg_sum
    mfi = pl.when(neg_sum == 0).then(100.0).otherwise(100.0 - (100.0 / (1.0 + ratio)))
    return df.with_columns(mfi.cast(pl.Float64).alias(out))


def _add_zscore(df: pl.DataFrame, *, col: str, window: int, out: str = "zscore") -> pl.DataFrame:
    window = max(2, int(window))
    if col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))
    mean = pl.col(col).rolling_mean(window_size=window, min_periods=window)
    std = pl.col(col).rolling_std(window_size=window, min_periods=window)
    z = (pl.col(col) - mean) / std
    z = pl.when(std == 0).then(0.0).otherwise(z)
    return df.with_columns(z.cast(pl.Float64).alias(out))


class Strategy9002TestMFIZScore:
    """Estrategia de prueba para evaluar el plot con 2 subpaneles: MFI + Z-Score.

    - MFI con umbrales simétricos: low en [20..40] y high = 100-low (=> [60..80])
    - Z-Score de close con ventana y umbral ±thr

    Señales (solo para tener trades y ver markers):
    - LONG  si MFI cruza arriba de low y Z < -thr
    - SHORT si MFI cruza abajo de high y Z > +thr
    """

    combinacion_id = 9002
    name = "TEST_MFI_ZSCORE"

    parametros_optuna: Dict[str, Any] = {
        "mfi_period": (7, 30, 1),
        "mfi_low": (20, 40, 1),  # simétrico => high = 100-low
        "z_window": (30, 200, 1),
        "z_thr": (1.0, 3.0, 0.1),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        mfi_period = int(trial.suggest_int("mfi_period", 7, 30))
        mfi_low = int(trial.suggest_int("mfi_low", 20, 40))
        mfi_high = 100 - mfi_low

        z_window = int(trial.suggest_int("z_window", 30, 200))
        z_thr = float(trial.suggest_float("z_thr", 1.0, 3.0, step=0.1))

        return {
            "mfi_period": mfi_period,
            "mfi_low": mfi_low,
            "mfi_high": mfi_high,
            "z_window": z_window,
            "z_thr": z_thr,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        mfi_period = max(2, int(params.get("mfi_period", 14)))
        mfi_low = float(params.get("mfi_low", 40))
        mfi_high = float(100.0 - mfi_low)
        params["mfi_high"] = mfi_high

        z_window = max(2, int(params.get("z_window", 100)))
        z_thr = float(params.get("z_thr", 2.0))

        params["__warmup_bars"] = max(mfi_period, z_window) + 10

        # Plot: bounds y specs por indicador (por trial)
        params["__indicator_bounds"] = {
            "mfi": {"hi": mfi_high, "lo": mfi_low, "mid": 50.0},
            "zscore": {"hi": z_thr, "lo": -z_thr, "mid": 0.0},
        }
        params["__indicator_specs"] = {
            "mfi": {
                "panel": "sub",
                "type": "line",
                "name": f"MFI ({mfi_period})",
                "precision": 2,
            },
            "zscore": {
                "panel": "sub",
                "type": "line",
                "name": f"ZScore ({z_window})",
                "precision": 3,
            },
        }

        params["__indicators_used"] = ["mfi", "zscore"]
        df = _add_mfi(df, period=mfi_period, out="mfi")
        df = _add_zscore(df, col="close", window=z_window, out="zscore")

        mfi = pl.col("mfi")
        z = pl.col("zscore")

        cross_up_low = (mfi > mfi_low) & (mfi.shift(1) <= mfi_low)
        cross_dn_high = (mfi < mfi_high) & (mfi.shift(1) >= mfi_high)

        signal_long = (cross_up_low & (z < -z_thr)).fill_null(False)
        signal_short = (cross_dn_high & (z > z_thr)).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

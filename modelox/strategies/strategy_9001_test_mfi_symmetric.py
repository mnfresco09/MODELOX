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


class Strategy9001TestMFISymmetric:
    """Estrategia de prueba: MFI con umbrales simétricos.

    Reglas:
    - Optuna parametriza SOLO `mfi_low` en [20..40]
    - `mfi_high = 100 - mfi_low` (simétrico alrededor de 50)

    Señales:
    - LONG  cuando MFI cruza hacia arriba `mfi_low`
    - SHORT cuando MFI cruza hacia abajo `mfi_high`
    """

    combinacion_id = 9001
    name = "TEST_MFI_SYMM"

    # `ejecutar.py` usa esto para listar indicadores/params del trial.
    # Mantenemos un solo grado de libertad para la simetría:
    # - `mfi_low` en [20..40] => `mfi_high = 100 - mfi_low`
    parametros_optuna: Dict[str, Any] = {
        "mfi_period": (7, 30, 1),
        "mfi_low": (20, 40, 1),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        mfi_period = int(trial.suggest_int("mfi_period", 7, 30))

        # Simetría: low in [20..40] => high in [60..80]
        mfi_low = int(trial.suggest_int("mfi_low", 20, 40))
        mfi_high = 100 - mfi_low

        return {
            "mfi_period": mfi_period,
            "mfi_low": mfi_low,
            "mfi_high": mfi_high,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        mfi_period = int(params.get("mfi_period", 14))
        mfi_period = max(2, mfi_period)

        mfi_low = float(params.get("mfi_low", 40))
        # En runtime recalculamos por seguridad para mantener simetría
        mfi_high = float(100.0 - mfi_low)
        params["mfi_high"] = mfi_high

        # Warmup
        params["__warmup_bars"] = mfi_period + 10

        # Plot: bounds + spec del panel
        params["__indicator_bounds"] = {
            "mfi": {"hi": mfi_high, "lo": mfi_low, "mid": 50.0}
        }
        params["__indicator_specs"] = {
            "mfi": {
                "panel": "sub",
                "type": "line",
                "name": f"MFI ({mfi_period})",
                "precision": 2,
            }
        }

        # Indicadores (inline)
        params["__indicators_used"] = ["mfi"]
        df = _add_mfi(df, period=mfi_period, out="mfi")

        # Señales (cruces)
        mfi = pl.col("mfi")
        cross_up_low = (mfi > mfi_low) & (mfi.shift(1) <= mfi_low)
        cross_dn_high = (mfi < mfi_high) & (mfi.shift(1) >= mfi_high)

        signal_long = cross_up_low.fill_null(False)
        signal_short = cross_dn_high.fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

from __future__ import annotations

"""Strategy 9955 — SuperIndicador (RSI + STOCH + MOM) Cross

INDICADOR
- super_9955: oscilador compuesto (RSI + StochD + Momentum Z), clamp [-cap, +cap]

ENTRADAS (simétricas)
- LONG: super_9955 cruza -level hacia arriba  (prev < -level AND curr > -level)
- SHORT: super_9955 cruza +level hacia abajo (prev > +level AND curr < +level)

NOTA
- Las salidas NO se definen aquí. La salida es global y vive en el engine:
  SL/TP fijos por ATR al momento de entrada (intra-vela).
"""

from typing import Any, Dict

import polars as pl


def _add_rsi(df: pl.DataFrame, *, period: int, out: str = "rsi") -> pl.DataFrame:
    period = max(2, int(period))
    if "close" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

    avg_gain = gain.rolling_mean(window_size=period, min_periods=period)
    avg_loss = loss.rolling_mean(window_size=period, min_periods=period)
    rs = avg_gain / avg_loss
    rsi = pl.when(avg_loss == 0).then(100.0).otherwise(100.0 - (100.0 / (1.0 + rs)))
    return df.with_columns(rsi.cast(pl.Float64).alias(out))


def _add_stoch_d(df: pl.DataFrame, *, period: int, d_period: int = 3, out: str = "stoch_d") -> pl.DataFrame:
    period = max(2, int(period))
    d_period = max(1, int(d_period))
    required = {"high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    lowest_low = pl.col("low").rolling_min(window_size=period, min_periods=period)
    highest_high = pl.col("high").rolling_max(window_size=period, min_periods=period)
    denom = highest_high - lowest_low
    k = pl.when(denom == 0).then(0.0).otherwise((pl.col("close") - lowest_low) / denom * 100.0)
    d = k.rolling_mean(window_size=d_period, min_periods=d_period)
    return df.with_columns(d.cast(pl.Float64).alias(out))


def _add_momentum_z(df: pl.DataFrame, *, len_mom: int, mom_norm: int, out: str = "mom_z") -> pl.DataFrame:
    len_mom = max(1, int(len_mom))
    mom_norm = max(2, int(mom_norm))
    if "close" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out))

    mom = pl.col("close") - pl.col("close").shift(len_mom)
    mean = mom.rolling_mean(window_size=mom_norm, min_periods=mom_norm)
    std = mom.rolling_std(window_size=mom_norm, min_periods=mom_norm)
    z = (mom - mean) / std
    z = pl.when(std == 0).then(0.0).otherwise(z)
    return df.with_columns(z.cast(pl.Float64).alias(out))


class Strategy9955SuperIndicador:
    combinacion_id = 9955
    name = "SuperIndicador9955_Cross_-L+L"

    __indicators_used = ["super_9955"]

    # Defaults (puedes optimizar vía Optuna)
    LEN_RSI = 14
    LEN_STOCH = 14
    LEN_MOM = 10
    MOM_NORM = 100
    AMPLIFICATION = 1.6
    CAP = 3.0

    LEVEL = 2.0

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    parametros_optuna = {
        # Abre rangos aquí si quieres que Optuna los muestre en UI.
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        # Parametrizable por trial
        # Nota: level siempre se fuerza a ser simétrico y <= cap.
        cap = float(trial.suggest_float("cap", 2.0, 4.0, step=0.5))
        level = float(trial.suggest_float("level", 1.0, 4.0, step=0.1))
        if level > cap:
            level = cap

        return {
            "len_rsi": int(trial.suggest_int("len_rsi", 7, 30)),
            "len_stoch": int(trial.suggest_int("len_stoch", 7, 30)),
            "len_mom": int(trial.suggest_int("len_mom", 5, 30)),
            "mom_norm": int(trial.suggest_int("mom_norm", 50, 200, step=10)),
            "amplification": float(trial.suggest_float("amplification", 0.8, 3.0, step=0.1)),
            "cap": cap,
            "level": level,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # ==============================================================
        # INDICADORES (inline, dentro de la estrategia)
        # ==============================================================

        len_rsi = int(params.get("len_rsi", self.LEN_RSI))
        len_stoch = int(params.get("len_stoch", self.LEN_STOCH))
        len_mom = int(params.get("len_mom", self.LEN_MOM))
        mom_norm = int(params.get("mom_norm", self.MOM_NORM))
        amplification = float(params.get("amplification", self.AMPLIFICATION))
        cap = float(params.get("cap", self.CAP))

        level = float(params.get("level", self.LEVEL))
        if level < 0:
            level = -level
        if cap <= 0:
            cap = self.CAP
        if level > cap:
            level = cap

        # Warmup: RSI/Stoch + Momentum normalization
        params["__warmup_bars"] = max(len_rsi + 1, len_stoch + 3, len_mom + mom_norm) + 5

        # Plot reference lines inside the oscillator panel
        params["__indicator_bounds"] = {
            "super_9955": {"hi": float(level), "lo": float(-level), "mid": 0.0}
        }

        params["__indicators_used"] = ["super_9955"]

        df = _add_rsi(df, period=len_rsi, out="rsi_9955")
        df = _add_stoch_d(df, period=len_stoch, d_period=3, out="stoch_d_9955")
        df = _add_momentum_z(df, len_mom=len_mom, mom_norm=mom_norm, out="mom_z_9955")

        # Normalizaciones a escala similar
        rsi_scaled = (pl.col("rsi_9955") - 50.0) / 50.0
        stoch_scaled = (pl.col("stoch_d_9955") - 50.0) / 50.0
        mom_z = pl.col("mom_z_9955")

        super_raw = (rsi_scaled + stoch_scaled + mom_z) / 3.0
        super_amp = super_raw * float(amplification)
        super_clamped = pl.when(super_amp > cap).then(cap).when(super_amp < -cap).then(-cap).otherwise(super_amp)
        df = df.with_columns(super_clamped.cast(pl.Float64).alias("super_9955"))

        s = pl.col("super_9955")

        cross_up_minus = (s.shift(1) < (-level)) & (s > (-level))
        cross_down_plus = (s.shift(1) > level) & (s < level)

        return df.with_columns(
            [
                cross_up_minus.fill_null(False).alias("signal_long"),
                cross_down_plus.fill_null(False).alias("signal_short"),
            ]
        )

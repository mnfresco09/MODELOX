from __future__ import annotations

from typing import Any, Dict

import numpy as np
import polars as pl


def _kalman_1d_estimate(close: np.ndarray, *, k_gain: float) -> np.ndarray:
    """Filtro recursivo 1D (estilo Kalman simple / EMA-like).

    k_est[i] = k_est[i-1] + k_gain * (close[i] - k_est[i-1])

    Nota: es exactamente una EMA con alpha=k_gain, pero lo dejamos explícito.
    """

    n = int(close.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= 0:
        return out

    kg = float(k_gain)
    if not np.isfinite(kg):
        kg = 0.05
    kg = float(np.clip(kg, 0.01, 1.0))

    out[0] = float(close[0])
    for i in range(1, n):
        prev = out[i - 1]
        out[i] = prev + kg * (float(close[i]) - prev)

    return out


class Strategy3KalmanSniper:
    """FRANCOTIRADOR ESTOCÁSTICO: KALMAN-FISHER (M15 idea)

    Implementación basada en tu script Pine.

    Entradas (Pine):
    - is_mean_reverting: chop_idx > chop_thresh
    - is_overextended_up: z_score > z_thresh
    - is_overextended_down: z_score < -z_thresh
    - fisher_turn_down: fisher < fisher_prev and fisher_prev > fisher_extreme
    - fisher_turn_up: fisher > fisher_prev and fisher_prev < -fisher_extreme

    Señales:
    - SHORT: is_mean_reverting AND is_overextended_up AND fisher_turn_down
    - LONG:  is_mean_reverting AND is_overextended_down AND fisher_turn_up

    Importante:
    - NO usamos el `sl_mult` del Pine.
    - Las salidas las gestiona el engine global (ATR/TP/Time) del sistema.
    """

    combinacion_id = 3
    name = "KALMAN SNIPER"

    parametros_optuna: Dict[str, Any] = {
        # Kalman
        "k_gain": (0.01, 1.0, 0.01),
        # Z-Score residual
        "z_len": (20, 120, 1),
        "z_thresh": (1.5, 3.5, 0.1),
        # Choppiness (régimen)
        "chop_len": (10, 40, 1),
        "chop_thresh": (45.0, 65.0, 1.0),
        # Fisher (tanh(normalized RSI))
        "fish_len": (6, 30, 1),
        "fisher_extreme": (0.6, 0.95, 0.05),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        k_gain = float(trial.suggest_float("k_gain", 0.01, 1.0, step=0.01))
        z_len = int(trial.suggest_int("z_len", 20, 120))
        z_thresh = float(trial.suggest_float("z_thresh", 1.5, 3.5, step=0.1))
        chop_len = int(trial.suggest_int("chop_len", 10, 40))
        chop_thresh = float(trial.suggest_float("chop_thresh", 45.0, 65.0, step=1.0))
        fish_len = int(trial.suggest_int("fish_len", 6, 30))
        fisher_extreme = float(trial.suggest_float("fisher_extreme", 0.6, 0.95, step=0.05))

        return {
            "k_gain": k_gain,
            "z_len": z_len,
            "z_thresh": z_thresh,
            "chop_len": chop_len,
            "chop_thresh": chop_thresh,
            "fish_len": fish_len,
            "fisher_extreme": fisher_extreme,
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # -----------------------------
        # Params
        # -----------------------------
        k_gain = float(params.get("k_gain", 0.05))
        z_len = max(2, int(params.get("z_len", 50)))
        z_thresh = float(params.get("z_thresh", 2.5))
        chop_len = max(2, int(params.get("chop_len", 14)))
        chop_thresh = float(params.get("chop_thresh", 55.0))
        fish_len = max(2, int(params.get("fish_len", 9)))
        fisher_extreme = float(params.get("fisher_extreme", 0.8))

        # Warmup: ventanas + margen (ATR14 para choppiness usa TR; RSI usa fish_len)
        params["__warmup_bars"] = max(z_len, chop_len, fish_len, 14) + 25

        # Plot metadata
        params["__indicators_used"] = ["kalman", "z_score", "chop_idx", "fisher"]
        params["__indicator_bounds"] = {
            "z_score": {"hi": z_thresh, "lo": -z_thresh, "mid": 0.0},
            "chop_idx": {"mid": chop_thresh},
            "fisher": {"hi": fisher_extreme, "lo": -fisher_extreme, "mid": 0.0},
        }
        params["__indicator_specs"] = {
            "kalman": {"panel": "overlay", "type": "line", "name": f"Kalman (gain={k_gain:.2f})", "precision": 4},
            "z_score": {"panel": "sub", "type": "line", "name": f"Z-Score ({z_len})", "precision": 3},
            "chop_idx": {"panel": "sub", "type": "line", "name": f"Choppiness ({chop_len})", "precision": 2},
            "fisher": {"panel": "sub", "type": "line", "name": f"Fisher (tanh RSI {fish_len})", "precision": 3},
        }

        # -----------------------------
        # 1) Kalman estimate (numpy, recursive)
        # -----------------------------
        close_np = np.asarray(df["close"].to_numpy(), dtype=np.float64)
        k_est = _kalman_1d_estimate(close_np, k_gain=k_gain)
        df = df.with_columns(pl.Series("kalman", k_est).cast(pl.Float64))

        # -----------------------------
        # 2) Z-Score residual
        # -----------------------------
        residual = (pl.col("close") - pl.col("kalman")).cast(pl.Float64)
        resid_std = residual.rolling_std(window_size=z_len, min_periods=z_len)
        z_score = (residual / resid_std).cast(pl.Float64)
        df = df.with_columns(z_score.alias("z_score"))

        # -----------------------------
        # 3) Choppiness
        # -----------------------------
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            [
                (pl.col("high") - pl.col("low")).abs(),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        ).cast(pl.Float64)

        tr_sum = tr.rolling_sum(window_size=chop_len, min_periods=chop_len)
        hh = pl.col("high").rolling_max(window_size=chop_len, min_periods=chop_len)
        ll = pl.col("low").rolling_min(window_size=chop_len, min_periods=chop_len)
        denom = (hh - ll).cast(pl.Float64)

        # chop = 100 * log10( sum(TR) / (HH-LL) ) / log10(len)
        # evitar 0/0
        ratio = (tr_sum / denom).cast(pl.Float64)
        chop_idx = (
            (pl.when((ratio > 0) & denom.is_finite() & tr_sum.is_finite())
             .then(ratio.log10())
             .otherwise(None))
            / float(np.log10(float(chop_len)))
        ) * 100.0
        df = df.with_columns(chop_idx.cast(pl.Float64).alias("chop_idx"))

        # -----------------------------
        # 4) Fisher (tanh of normalized RSI)
        # -----------------------------
        delta = pl.col("close").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        avg_gain = gain.rolling_mean(window_size=fish_len, min_periods=fish_len)
        avg_loss = loss.rolling_mean(window_size=fish_len, min_periods=fish_len)
        rs = avg_gain / avg_loss
        rsi = pl.when(avg_loss == 0).then(100.0).otherwise(100.0 - (100.0 / (1.0 + rs)))
        norm_rsi = ((rsi - 50.0) / 50.0).cast(pl.Float64)

        # fisher = tanh(norm_rsi) = (exp(2x)-1)/(exp(2x)+1)
        fisher = norm_rsi.tanh().cast(pl.Float64)
        df = df.with_columns(fisher.alias("fisher"))

        # -----------------------------
        # Señales (Pine): régimen + sobreextensión + giro de fisher
        # -----------------------------
        is_mean_reverting = pl.col("chop_idx") > float(chop_thresh)
        is_overextended_up = pl.col("z_score") > float(z_thresh)
        is_overextended_down = pl.col("z_score") < -float(z_thresh)

        fisher_cur = pl.col("fisher")
        fisher_prev = fisher_cur.shift(1)
        fisher_turn_down = (fisher_cur < fisher_prev) & (fisher_prev > float(fisher_extreme))
        fisher_turn_up = (fisher_cur > fisher_prev) & (fisher_prev < -float(fisher_extreme))

        signal_short = (is_mean_reverting & is_overextended_up & fisher_turn_down).fill_null(False)
        signal_long = (is_mean_reverting & is_overextended_down & fisher_turn_up).fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )

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

from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_superindicador_9955


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
        params["__indicators_used"] = self.get_indicators_used()

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

        df = IndicadorFactory.procesar(
            df,
            {
                "superindicador_9955": cfg_superindicador_9955(
                    len_rsi=len_rsi,
                    len_stoch=len_stoch,
                    len_mom=len_mom,
                    mom_norm=mom_norm,
                    amplification=amplification,
                    cap=cap,
                    out="super_9955",
                )
            },
        )

        s = pl.col("super_9955")

        cross_up_minus = (s.shift(1) < (-level)) & (s > (-level))
        cross_down_plus = (s.shift(1) > level) & (s < level)

        return df.with_columns(
            [
                cross_up_minus.fill_null(False).alias("signal_long"),
                cross_down_plus.fill_null(False).alias("signal_short"),
            ]
        )

from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

class StrategyZRsiAdx(EstrategiaBase):
    """
    ESTRATEGIA RSI × VOLATILIDAD × ADX
    LONG : RSI cruza al alza sobreventa + Vol alta + ADX bajo (rango)
    SHORT: RSI cruza a la baja sobrecompra + Vol alta + ADX bajo (rango)
    """

    combinacion_id = 5
    name           = "id5"
    SALIDAS_PERSONALIZADAS = False

    # ─── Parámetros fijos (prioridad baja → no se optimizan) ──────────────────
    ADX_SMOOTHING_FIJO = 14
    DI_LENGTH_FIJO     = 14

    # ══════════════════════════════════════════════════════════════════════════
    # ESPACIO DE BÚSQUEDA
    # ══════════════════════════════════════════════════════════════════════════
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        Nombres claros y agrupados por indicador.
        ADX smoothing y DI length son fijos (prioridad baja).
        RSI es simétrico: sobreventa = 100 - sobrecompra.
        """

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_periodo      = trial.suggest_int("RSI_Periodo",     5, 50, step=1)
        rsi_sobrecompra  = trial.suggest_int("RSI_Sobrecompra", 60, 85, step=1)
        rsi_sobreventa   = 100 - rsi_sobrecompra   # Simétrico automático

        # ── VOLATILIDAD ───────────────────────────────────────────────────────
        vol_periodo      = trial.suggest_int(  "Vol_Periodo",   5, 30,  step=1)
        vol_lookback     = trial.suggest_int(  "Vol_Lookback",  50, 200, step=10)
        vol_clamp        = trial.suggest_float("Vol_Clamp",     1.5, 4.0, step=0.5)
        vol_umbral       = trial.suggest_float("Vol_Umbral",    0.5, 2.0, step=0.1)

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_umbral       = trial.suggest_float("ADX_Umbral",    15.0, 35.0, step=1.0)

        return {
            # RSI
            "rsi_periodo"    : rsi_periodo,
            "rsi_sobrecompra": rsi_sobrecompra,
            "rsi_sobreventa" : rsi_sobreventa,
            # Volatilidad
            "vol_periodo"    : vol_periodo,
            "vol_lookback"   : vol_lookback,
            "vol_clamp"      : vol_clamp,
            "vol_umbral"     : vol_umbral,
            # ADX
            "adx_umbral"     : adx_umbral,
            # Fijos (no en Optuna pero sí en params para generate_signals)
            "adx_smoothing"  : self.ADX_SMOOTHING_FIJO,
            "di_length"      : self.DI_LENGTH_FIJO,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # GENERADOR DE SEÑALES
    # ══════════════════════════════════════════════════════════════════════════
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "open", "high", "low", "close"])

        # ── Extraer parámetros con nombres claros ─────────────────────────────
        rsi_len    = params["rsi_periodo"]
        rsi_ob     = params["rsi_sobrecompra"]
        rsi_os     = params["rsi_sobreventa"]

        vol_len    = params["vol_periodo"]
        z_lookback = params["vol_lookback"]
        z_range    = params["vol_clamp"]
        vol_thresh = params["vol_umbral"]

        adx_smooth = params["adx_smoothing"]
        di_len     = params["di_length"]
        adx_thresh = params["adx_umbral"]

        # ── Metadata ──────────────────────────────────────────────────────────
        warmup = max(rsi_len, z_lookback, adx_smooth, di_len) + 50
        params["__warmup_bars"]       = warmup
        params["__indicators_used"]   = ["rsi", "z_score", "adx"]
        params["__indicator_bounds"]  = {
            "rsi": {"low": rsi_os, "high": rsi_ob, "mid": 50},
            "adx": {"low": 0, "high": 100, "mid": adx_thresh}
        }
        params["__indicator_specs"] = {
            "rsi"    : {"color": "#00FFFF", "type": "line", "panel": "rsi"},
            "z_score": {"color": "#FF00FF", "type": "line", "panel": "vol"},
            "adx"    : {"color": "#FFFF00", "type": "line", "panel": "adx"}
        }

        q = df.lazy()

        # ══════════════════════════════════════════════════════════════════════
        # 1. RSI
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            self.rsi_expr(close=pl.col("close"), length=rsi_len).alias("rsi")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # 2. ADX (Wilder's smoothing via EWM com = n-1)
        # ══════════════════════════════════════════════════════════════════════
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low")  - pl.col("close").shift(1)).abs()
        )
        up_move   = pl.col("high") - pl.col("high").shift(1)
        down_move = pl.col("low").shift(1) - pl.col("low")

        plus_dm  = pl.when((up_move > down_move)   & (up_move   > 0)).then(up_move).otherwise(0.0)
        minus_dm = pl.when((down_move > up_move)   & (down_move > 0)).then(down_move).otherwise(0.0)

        q = q.with_columns([
            tr.alias("tr"),
            plus_dm.alias("plus_dm"),
            minus_dm.alias("minus_dm")
        ])

        com_di = di_len - 1
        q = q.with_columns([
            pl.col("tr")      .ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("atr"),
            pl.col("plus_dm") .ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("plus_dm_s"),
            pl.col("minus_dm").ewm_mean(com=com_di, min_periods=di_len, ignore_nulls=True).alias("minus_dm_s"),
        ])

        q = q.with_columns([
            (100.0 * pl.col("plus_dm_s")  / pl.col("atr")).fill_null(0).alias("plus_di"),
            (100.0 * pl.col("minus_dm_s") / pl.col("atr")).fill_null(0).alias("minus_di"),
        ])

        di_sum  = pl.col("plus_di") + pl.col("minus_di")
        di_diff = (pl.col("plus_di") - pl.col("minus_di")).abs()
        dx      = pl.when(di_sum != 0).then(100.0 * di_diff / di_sum).otherwise(0.0)

        q = q.with_columns([dx.alias("dx")])

        com_adx = adx_smooth - 1
        q = q.with_columns([
            pl.col("dx").ewm_mean(com=com_adx, min_periods=adx_smooth, ignore_nulls=True)
              .fill_null(0).alias("adx")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # 3. Z-SCORE VOLATILIDAD (Garman-Klass)
        # ══════════════════════════════════════════════════════════════════════
        ln_hl = (pl.col("high") / pl.col("low")).log()
        ln_co = (pl.col("close") / pl.col("open")).log()

        gk = (0.5 * ln_hl.pow(2) - (2 * np.log(2) - 1) * ln_co.pow(2)) \
               .rolling_mean(window_size=vol_len).sqrt()

        q = q.with_columns([gk.alias("gk_vol")])

        mean_vol = pl.col("gk_vol").rolling_mean(window_size=z_lookback)
        std_vol  = pl.col("gk_vol").rolling_std(window_size=z_lookback)

        z_raw      = pl.when(std_vol != 0).then((pl.col("gk_vol") - mean_vol) / std_vol).otherwise(0.0)
        z_clamped  = z_raw.clip(-z_range, z_range)
        z_norm     = (z_clamped + z_range) / (2.0 * z_range) * 2.0

        q = q.with_columns([
            z_raw .alias("z_score_raw"),
            z_norm.alias("z_score"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # 4. SEÑALES
        # ══════════════════════════════════════════════════════════════════════
        alta_vol    = pl.col("z_score") > vol_thresh
        rango_adx   = pl.col("adx")    < adx_thresh

        rsi_sube    = (pl.col("rsi") > rsi_os) & (pl.col("rsi").shift(1) <= rsi_os)
        rsi_baja    = (pl.col("rsi") < rsi_ob) & (pl.col("rsi").shift(1) >= rsi_ob)

        raw_long    = alta_vol & rango_adx & rsi_sube
        raw_short   = alta_vol & rango_adx & rsi_baja

        q = q.with_columns([
            self._as_bool(raw_long) .alias("signal_long"),
            self._as_bool(raw_short).alias("signal_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # 5. RETORNO
        # ══════════════════════════════════════════════════════════════════════
        return self.finalize_signals(
            q,
            keep_cols=["rsi", "z_score", "adx", "plus_di", "minus_di"]
        )
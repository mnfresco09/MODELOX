from typing import Any, Dict, Optional
import numpy as np
import polars as pl
from numba import njit
from logic.indicators import IndicadorFactory
from modelox.core.types import ExitDecision
from modelox.strategies.indicator_specs import cfg_atr

# ==============================================================================
# 1. SALIDA ESTRICTA (SL REAL) - MANTENIDA
# ==============================================================================
@njit(cache=True, fastmath=True)
def _decide_exit_strict(
    entry_idx: int,
    entry_atr: float,
    sl_mult: float,
    trail_mult: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    side_flag: int,
    max_bars: int,
) -> tuple:
    n = len(close)
    last_allowed = min(n - 1, entry_idx + max_bars)
    lb = 5
    start = entry_idx - (lb - 1)
    if start < 0: start = 0

    hard_sl = np.nan
    if not np.isnan(entry_atr):
        if side_flag == 1: # LONG
            mn = low[start]
            for j in range(start + 1, entry_idx + 1):
                if low[j] < mn: mn = low[j]
            hard_sl = mn - entry_atr * sl_mult
        else: # SHORT
            mx = high[start]
            for j in range(start + 1, entry_idx + 1):
                if high[j] > mx: mx = high[j]
            hard_sl = mx + entry_atr * sl_mult

    trailing = np.nan
    if not np.isnan(entry_atr):
        if side_flag == 1:
            trailing = close[entry_idx] - entry_atr * trail_mult
        else:
            trailing = close[entry_idx] + entry_atr * trail_mult

    for i in range(entry_idx + 1, last_allowed + 1):
        o, h, l, c, a = open_[i], high[i], low[i], close[i], atr[i]
        if np.isnan(o): continue

        if not np.isnan(a):
            if side_flag == 1:
                dyn = c - a * trail_mult
                if np.isnan(trailing) or dyn > trailing: trailing = dyn
            else:
                dyn = c + a * trail_mult
                if np.isnan(trailing) or dyn < trailing: trailing = dyn

        active_stop = np.nan
        reason_code = 0 

        if side_flag == 1: # LONG
            current_stop = -np.inf
            if not np.isnan(hard_sl):
                current_stop = hard_sl
                reason_code = 1
            if not np.isnan(trailing):
                if trailing > current_stop:
                    current_stop = trailing
                    reason_code = 2
            
            if reason_code > 0:
                if o <= current_stop: return i, o, reason_code # Gap Exit
                if l <= current_stop: return i, current_stop, reason_code # Touch Exit

        else: # SHORT
            current_stop = np.inf
            if not np.isnan(hard_sl):
                current_stop = hard_sl
                reason_code = 1
            if not np.isnan(trailing):
                if trailing < current_stop:
                    current_stop = trailing
                    reason_code = 2
            
            if reason_code > 0:
                if o >= current_stop: return i, o, reason_code
                if h >= current_stop: return i, current_stop, reason_code

    if last_allowed > entry_idx:
        return last_allowed, close[last_allowed], 3
    return -1, np.nan, 0


# ==============================================================================
# 2. ESTRATEGIA: TEMA KINETIC IMPULSE
# ==============================================================================

class Strategy3003TEMAKineticImpulse:
    combinacion_id = 3005
    name = "TEMA_Kinetic_Z_V1"

    __indicators_used = ["tema", "z_velocity", "adx", "atr"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    parametros_optuna = {
        # --- Configuración TEMA (Base) ---
        "tema_period": (10, 50, 2), # Ventana del TEMA
        
        # --- Configuración Kinetic Z (Normalización) ---
        "z_window": (20, 100, 5),   # Ventana histórica para calcular el Z-Score
        "z_trigger": (1.0, 3.0, 0.1), # Umbral de disparo (Sigmas)
        
        # --- Filtro ADX (Fuerza) ---
        "adx_period": (10, 20, 1),
        "adx_min": (15, 30, 1),     # Solo operar si hay tendencia mínima
        
        # --- Gestión de Salida ---
        "sl_multiplier": (1.0, 4.0, 0.1),
        "trail_multiplier": (1.5, 6.0, 0.1),
        "atr_period": (10, 24, 1),
    }

    TIMEOUT_BARS = 260

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        if trial is None:
            return {
                "tema_period": 20,
                "z_window": 50,
                "z_trigger": 1.8,
                "adx_period": 14,
                "adx_min": 20,
                "sl_multiplier": 2.0,
                "trail_multiplier": 3.5,
                "atr_period": 14,
            }
        return {
            "tema_period": trial.suggest_int("tema_period", 10, 50, step=2),
            "z_window": trial.suggest_int("z_window", 20, 100, step=5),
            "z_trigger": trial.suggest_float("z_trigger", 1.0, 3.0, step=0.1),
            "adx_period": trial.suggest_int("adx_period", 10, 20, step=1),
            "adx_min": trial.suggest_int("adx_min", 15, 30, step=1),
            "sl_multiplier": trial.suggest_float("sl_multiplier", 1.0, 4.0, step=0.1),
            "trail_multiplier": trial.suggest_float("trail_multiplier", 1.5, 6.0, step=0.1),
            "atr_period": trial.suggest_int("atr_period", 10, 24, step=1),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        params["__indicators_used"] = self.get_indicators_used()

        if "timestamp" not in df.columns and "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").alias("timestamp"))

        # Params
        tema_len = params.get("tema_period", 20)
        z_win = params.get("z_window", 50)
        z_trig = params.get("z_trigger", 1.8)
        
        adx_len = params.get("adx_period", 14)
        adx_threshold = params.get("adx_min", 20)
        
        params["__warmup_bars"] = max(tema_len*3, z_win, adx_len) + 20

        # 1. Base Indicators: ATR & ADX
        # Usamos IndicadorFactory para ADX y ATR que son estándar
        # Asumimos que tienes cfg_adx disponible, si no, lo calculamos manual abajo por seguridad
        df = IndicadorFactory.procesar(
            df,
            {
                "atr": cfg_atr(period=params.get("atr_period", 14), out="atr"),
            },
        )

        # ----------------------------------------------------------------------
        # 2. CÁLCULO MANUAL: TEMA (Triple Exponential Moving Average)
        # ----------------------------------------------------------------------
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        # Polars ewm_mean es rápido
        
        ema1 = pl.col("close").ewm_mean(span=tema_len, adjust=False)
        # Para ema2 y ema3 necesitamos aplicar ewm sobre la expresión anterior.
        # En Polars lazy, a veces es mejor instanciar columnas intermedias.
        
        df = df.with_columns(ema1.alias("ema1"))
        df = df.with_columns(pl.col("ema1").ewm_mean(span=tema_len, adjust=False).alias("ema2"))
        df = df.with_columns(pl.col("ema2").ewm_mean(span=tema_len, adjust=False).alias("ema3"))
        
        tema_expr = (3 * pl.col("ema1")) - (3 * pl.col("ema2")) + pl.col("ema3")
        df = df.with_columns(tema_expr.alias("tema"))

        # ----------------------------------------------------------------------
        # 3. CÁLCULO KINETIC: Velocity & Z-Score
        # ----------------------------------------------------------------------
        # Velocidad = TEMA actual - TEMA previo
        df = df.with_columns((pl.col("tema") - pl.col("tema").shift(1)).alias("velocity"))
        
        # Z-Score de la Velocidad
        # Z = (Val - Mean) / Std
        v = pl.col("velocity")
        v_mean = v.rolling_mean(z_win)
        v_std = v.rolling_std(z_win)
        
        # Evitar div por cero
        z_score_expr = (v - v_mean) / pl.when(v_std == 0).then(0.00001).otherwise(v_std)
        df = df.with_columns(z_score_expr.fill_null(0).alias("z_velocity"))

        # ----------------------------------------------------------------------
        # 4. CÁLCULO ADX (Manual Rápido si no está en Factory)
        # ----------------------------------------------------------------------
        # True Range ya calculado internamente por ATR normalmente, pero lo rehacemos rápido
        h, l, c_prev = pl.col("high"), pl.col("low"), pl.col("close").shift(1)
        tr = pl.max_horizontal([h-l, (h-c_prev).abs(), (l-c_prev).abs()])
        
        # Directional Movement
        up = h - h.shift(1)
        down = l.shift(1) - l
        pos_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
        neg_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
        
        # Smooth (Wilder's Smoothing is roughly EMA with span = 2*n - 1)
        wilder_span = (adx_len * 2) - 1
        tr_s = tr.ewm_mean(span=wilder_span, adjust=False)
        pos_dm_s = pos_dm.ewm_mean(span=wilder_span, adjust=False)
        neg_dm_s = neg_dm.ewm_mean(span=wilder_span, adjust=False)
        
        pos_di = 100 * pos_dm_s / pl.when(tr_s==0).then(0.001).otherwise(tr_s)
        neg_di = 100 * neg_dm_s / pl.when(tr_s==0).then(0.001).otherwise(tr_s)
        
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 0.001)
        adx_expr = dx.ewm_mean(span=wilder_span, adjust=False)
        
        df = df.with_columns(adx_expr.fill_null(0).alias("adx"))

        # ----------------------------------------------------------------------
        # 5. GENERACIÓN DE SEÑALES
        # ----------------------------------------------------------------------
        z = pl.col("z_velocity")
        adx = pl.col("adx")
        
        # Condiciones
        # 1. Impulso fuerte: Z-Score de velocidad rompe el umbral
        impulse_up = z > z_trig
        impulse_down = z < -z_trig
        
        # 2. Fuerza de tendencia: ADX > min
        trend_ok = adx > adx_threshold

        # Señales de Entrada
        # Long: Impulso alcista fuerte + Tendencia activa
        signal_long = (impulse_up & trend_ok).fill_null(False)
        
        # Short: Impulso bajista fuerte + Tendencia activa
        signal_short = (impulse_down & trend_ok).fill_null(False)

        return df.with_columns([
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short"),
        ])

    def decide_exit(self, df: pl.DataFrame, params: Dict[str, Any], entry_idx: int, entry_price: float, side: str, *, saldo_apertura: float) -> Optional[ExitDecision]:
        n = len(df)
        if entry_idx >= n - 1: return None
        
        sl_mult = float(params.get("sl_multiplier", 2.0))
        trail_mult = float(params.get("trail_multiplier", 3.0))
        if trail_mult <= sl_mult: trail_mult = sl_mult + 0.1 

        if "atr" not in df.columns: return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="NO_ATR")

        open_arr = df["open"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        close_arr = df["close"].to_numpy().astype(np.float64)
        atr_arr = df["atr"].to_numpy().astype(np.float64)

        is_long = side.upper() == "LONG"
        
        exit_idx, exit_price, reason_code = _decide_exit_strict(
            int(entry_idx), float(atr_arr[entry_idx]), sl_mult, trail_mult,
            open_arr, high_arr, low_arr, close_arr, atr_arr,
            1 if is_long else -1, min(self.TIMEOUT_BARS, n - entry_idx - 1)
        )

        reason_map = {0: None, 1: "HARD_SL", 2: "TRAIL_STOP", 3: "TIME_EXIT"}
        
        if exit_idx >= 0 and reason_map.get(reason_code) is not None:
            return ExitDecision(exit_idx=int(exit_idx), reason=reason_map[int(reason_code)], exit_price=float(exit_price))

        return ExitDecision(exit_idx=int(min(entry_idx + self.TIMEOUT_BARS, n - 1)), reason="TIME_EXIT")
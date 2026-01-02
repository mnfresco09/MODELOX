from typing import Any, Dict, List
import polars as pl
from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase

class StrategyTemaADX(EstrategiaBase):
    """
    🚀 ESTRATEGIA: TEMA Trend Velocity + ADX Filter
    -----------------------------------------------
    Una estrategia de seguimiento de tendencia de baja latencia diseñada 
    para timeframes rápidos (5m - 1h).
    
    Concepto:
    1. TEMA (Triple EMA): Reduce el lag significativamente vs SMA/EMA.
    2. ADX: Filtra mercados laterales. Solo opera si hay "fuerza".
    
    Entrada:
    - LONG: Precio cruza sobre TEMA + ADX > Umbral
    - SHORT: Precio cruza bajo TEMA + ADX > Umbral
    """
    
    combinacion_id = 50  # ID Único, asegúrate que no choque con otros
    name = "TEMA_ADX_Trend"

    # Definimos el espacio de búsqueda para Optuna
    parametros_optuna = {
        "tema_period": (10, 50, 5),     # Periodo de la media rápida
        "adx_period": (14, 14, 1),      # Estándar de Wilder suele ser 14
        "adx_threshold": (20, 35, 5),   # Fuerza mínima de tendencia
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            "tema_period": trial.suggest_int("tema_period", 10, 100, step=5),
            "adx_period": trial.suggest_int("adx_period", 14, 28, step=7),
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 40, step=5),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # 1. Recuperar parámetros
        tema_p = params["tema_period"]
        adx_p = params["adx_period"]
        adx_thresh = params["adx_threshold"]

        # 2. Definir Warmup (Crítico para cálculos correctos)
        # Necesitamos suficiente historia para que el ADX y TEMA se estabilicen
        params["__warmup_bars"] = max(tema_p * 3, adx_p * 4)

        # 3. Cálculo de TEMA (Triple Exponential Moving Average)
        # Fórmula: (3 * EMA1) - (3 * EMA2) + EMA3
        # EMA1 = ema(close)
        # EMA2 = ema(EMA1)
        # EMA3 = ema(EMA2)
        
        # Helper para EMA en Polars
        def calc_ema(series, span):
            return series.ewm_mean(span=span, adjust=False)

        ema1 = calc_ema(pl.col("close"), tema_p)
        ema2 = calc_ema(ema1, tema_p)
        ema3 = calc_ema(ema2, tema_p)
        
        tema = (3 * ema1) - (3 * ema2) + ema3
        
        # 4. Cálculo de ADX (Simplificado vectorizado)
        # True Range
        high = pl.col("high")
        low = pl.col("low")
        close = pl.col("close")
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
        
        # Suavizado (Wilder usa una técnica especial, aquí usamos EWM que es muy cercano y rápido)
        # alpha = 1/period para Wilder es aprox span=(2*period)-1 en EWM
        wilder_span = (2 * adx_p) - 1
        
        tr_smooth = calc_ema(tr, wilder_span)
        plus_di = 100 * calc_ema(plus_dm, wilder_span) / tr_smooth
        minus_di = 100 * calc_ema(minus_dm, wilder_span) / tr_smooth
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = calc_ema(dx, wilder_span)

        # 5. Añadir columnas al DataFrame
        df = df.with_columns([
            tema.alias("tema"),
            adx.alias("adx")
        ])

        # 6. Lógica de Señales
        # Cruce de precio sobre TEMA + Filtro ADX
        price = pl.col("close")
        tema_col = pl.col("tema")
        adx_col = pl.col("adx")
        
        # Condición de cruce
        cross_over = (price > tema_col) & (price.shift(1) <= tema_col.shift(1))
        cross_under = (price < tema_col) & (price.shift(1) >= tema_col.shift(1))
        
        # Condición de filtro
        trend_strong = adx_col > adx_thresh
        
        signal_long = cross_over & trend_strong
        signal_short = cross_under & trend_strong

        df = df.with_columns([
            signal_long.fill_null(False).alias("signal_long"),
            signal_short.fill_null(False).alias("signal_short")
        ])

        # 7. Metadata para Reporting (Graficos)
        params["__indicators_used"] = ["tema", "adx"]
        
        # Definir cómo se pintan (TEMA sobre precio, ADX abajo)
        params["__indicator_specs"] = {
            "tema": {"panel": "overlay", "color": "yellow"},
            "adx": {
                "panel": "sub", 
                "bounds": {"hi": adx_thresh, "lo": 0, "mid": 25} # Dibuja la línea del umbral
            }
        }

        return df
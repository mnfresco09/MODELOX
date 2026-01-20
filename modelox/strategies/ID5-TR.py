from __future__ import annotations
from typing import Any, Dict, List
import polars as pl
from .ESTRATEGIA_BASE import EstrategiaBase

class StrategyZlemaFractal(EstrategiaBase):
    """
    ESTRATEGIA: ZLEMA TRIPLE FRACTAL ALIGNMENT (V1)
    ----------------------------------------------
    Lógica de nivel avanzado para reducción de lag:
    1. Dirección Maestra (1h): ZLEMA + Pendiente.
    2. Setup de Momentum (5m): ZLEMA + Pendiente.
    3. Gatillo de Disparo (1m): Cruce de precio sobre ZLEMA.
    4. Filtro de Volatilidad: ATR dinámico para evitar zonas "choppy".
    """

    # Identidad única en el sistema
    combinacion_id = 5
    name = "ID5 TREND FOLLOW"
    
    # Usamos las salidas globales del motor MODELOX
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """Configuración de búsqueda de Optuna."""
        return {
            # Periodos de las ZLEMAs por timeframe
            "zlema_1h_len": trial.suggest_int("zlema_1h_len", 30, 100, step=5),
            "zlema_5m_len": trial.suggest_int("zlema_5m_len", 15, 50, step=5),
            "zlema_1m_len": trial.suggest_int("zlema_1m_len", 5, 25, step=2),
            # Parámetros de volatilidad
            "atr_len": trial.suggest_int("atr_len", 14, 30),
            "atr_mult": trial.suggest_float("atr_mult", 0.8, 1.8, step=0.1),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        """Solicita los datos necesarios al Blender."""
        return ["5m", "1h"]

    def _zlema_expr(self, price_col: str, length: int) -> pl.Expr:
        """Helper para calcular ZLEMA (Zero Lag EMA) de forma vectorial."""
        lag = int((length - 1) / 2)
        # De-lagging: Precio actual + (Precio actual - Precio desplazado por el lag)
        data_delagged = pl.col(price_col) + (pl.col(price_col) - pl.col(price_col).shift(lag))
        return data_delagged.ewm_mean(span=length, adjust=False)

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # 1. Inicialización y Validación
        self._init_params_metadata(params)
        # El Blender inyecta columnas con sufijo _tf (ej: close_1h)
        self._require_columns(df, ["timestamp", "close", "close_5m", "close_1h", "high", "low"])

        # 2. Recuperar Parámetros de Optuna
        z1h_len = params.get("zlema_1h_len", 60)
        z5m_len = params.get("zlema_5m_len", 30)
        z1m_len = params.get("zlema_1m_len", 12)
        atr_len = params.get("atr_len", 20)
        atr_mult = params.get("atr_mult", 1.2)

        # 3. Configurar Metadata de Visualización (Plots)
        params["__warmup_bars"] = max(z1h_len * 60, 300) # Basado en el TF mayor
        params["__indicators_used"] = ["zlema_1m", "zlema_5m", "zlema_1h"]
        params["__indicator_bounds"] = {
            "zlema_1m": {"color": "cyan", "overlay": True},
            "zlema_5m": {"color": "magenta", "overlay": True},
            "zlema_1h": {"color": "yellow", "overlay": True}
        }

        # 4. Procesamiento Vectorial con Polars
        q = df.lazy()

        # --- FASE A: CÁLCULO DE INDICADORES ---
        q = q.with_columns([
            self._zlema_expr("close", z1m_len).alias("zlema_1m"),
            self._zlema_expr("close_5m", z5m_len).alias("zlema_5m"),
            self._zlema_expr("close_1h", z1h_len).alias("zlema_1h"),
        ])

        # Cálculo de Volatilidad (ATR 1m)
        tr = pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ])
        q = q.with_columns([
            tr.rolling_mean(atr_len).alias("atr_1m")
        ])
        q = q.with_columns([
            pl.col("atr_1m").rolling_mean(20).alias("atr_ma")
        ])

        # --- FASE B: LÓGICA DE TENDENCIA Y MOMENTUM ---
        # Tendencia Maestra 1h
        tendencia_1h_up = (pl.col("close_1h") > pl.col("zlema_1h")) & (pl.col("zlema_1h").diff() > 0)
        tendencia_1h_down = (pl.col("close_1h") < pl.col("zlema_1h")) & (pl.col("zlema_1h").diff() < 0)

        # Momentum 5m
        momentum_5m_up = (pl.col("close_5m") > pl.col("zlema_5m")) & (pl.col("zlema_5m").diff() > 0)
        momentum_5m_down = (pl.col("close_5m") < pl.col("zlema_5m")) & (pl.col("zlema_5m").diff() < 0)

        # Filtro de Volatilidad (Veto)
        volatilidad_ok = pl.col("atr_1m") > (pl.col("atr_ma") * atr_mult)

        # --- FASE C: GATILLO DE DISPARO (1m) ---
        # Cruce de precio sobre ZLEMA_1m
        cruce_up = (pl.col("close") > pl.col("zlema_1m")) & (pl.col("close").shift(1) <= pl.col("zlema_1m").shift(1))
        cruce_down = (pl.col("close") < pl.col("zlema_1m")) & (pl.col("close").shift(1) >= pl.col("zlema_1m").shift(1))

        # --- FASE D: GENERACIÓN DE SEÑALES FINALES ---
        sig_long = tendencia_1h_up & momentum_5m_up & volatilidad_ok & cruce_up
        sig_short = tendencia_1h_down & momentum_5m_down & volatilidad_ok & cruce_down

        # Aplicamos la normalización booleana y limpieza
        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
        ])

        # Retornamos el DataFrame con las señales y los indicadores para el reporte
        return self.finalize_signals(q, keep_cols=["zlema_1m", "zlema_5m", "zlema_1h"])
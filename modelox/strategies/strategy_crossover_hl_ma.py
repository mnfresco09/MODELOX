"""
Estrategia: Cruce de Precio con Media Móvil del Punto Medio (HL/2)

Lógica:
1. Calcula el punto medio: (high + low) / 2 para N períodos
2. Normaliza con una media móvil (SMA, EMA o ALMA)
3. LONG: Si el precio cruza la media hacia arriba y close > media
4. SHORT: Si el precio cruza la media hacia abajo y close < media
"""

from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np


class StrategyCrossoverHLMA:
    """
    Estrategia de cruce de precio con media móvil del punto medio (high+low)/2
    """
    
    combinacion_id = 3
    name = "Crossover_HL_MA"
    
    parametros_optuna: Dict[str, Any] = {
        "hl_period": (20, 150, 5),
        "ma_type": ["sma", "ema", "alma"],
    }
    
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda de Optuna
        """
        hl_period = trial.suggest_int("hl_period", 20, 150, step=5)
        ma_type = trial.suggest_categorical("ma_type", ["sma", "ema", "alma"])
        
        return {
            "hl_period": hl_period,
            "ma_type": ma_type,
        }
    
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Genera señales de trading basadas en el cruce del precio con la MA del punto medio
        """
        # 1) Extraer parámetros
        hl_period = int(params.get("hl_period", 50))
        ma_type = params.get("ma_type", "sma")
        
        # Validación
        hl_period = max(5, min(150, hl_period))
        
        # 2) Definir warmup
        warmup = hl_period + 10
        params["__warmup_bars"] = warmup
        
        # 3) Calcular punto medio (HL/2)
        df = df.with_columns([
            ((pl.col("high") + pl.col("low")) / 2.0).alias("hl_mid")
        ])
        
        # 4) Calcular media móvil del punto medio según el tipo
        if ma_type == "sma":
            df = df.with_columns([
                pl.col("hl_mid").rolling_mean(window_size=hl_period).alias("ma_hl")
            ])
        elif ma_type == "ema":
            df = df.with_columns([
                pl.col("hl_mid").ewm_mean(span=hl_period).alias("ma_hl")
            ])
        elif ma_type == "alma":
            # ALMA (Arnaud Legoux Moving Average)
            # Parámetros típicos: offset=0.85, sigma=6
            df = self._calculate_alma(df, "hl_mid", hl_period)
        else:
            # Default a SMA
            df = df.with_columns([
                pl.col("hl_mid").rolling_mean(window_size=hl_period).alias("ma_hl")
            ])
        
        # 5) Detectar cruces
        # Cruce hacia arriba: close anterior < ma anterior Y close actual > ma actual
        # Cruce hacia abajo: close anterior > ma anterior Y close actual < ma actual
        
        df = df.with_columns([
            pl.col("close").shift(1).alias("close_prev"),
            pl.col("ma_hl").shift(1).alias("ma_hl_prev")
        ])
        
        # 6) Generar señales
        df = df.with_columns([
            # LONG: cruce hacia arriba (close cruza por encima de ma_hl)
            (
                (pl.col("close_prev") < pl.col("ma_hl_prev")) &
                (pl.col("close") > pl.col("ma_hl"))
            ).alias("signal_long"),
            
            # SHORT: cruce hacia abajo (close cruza por debajo de ma_hl)
            (
                (pl.col("close_prev") > pl.col("ma_hl_prev")) &
                (pl.col("close") < pl.col("ma_hl"))
            ).alias("signal_short")
        ])
        
        # 7) Rellenar NaN con False para las señales
        df = df.with_columns([
            pl.col("signal_long").fill_null(False),
            pl.col("signal_short").fill_null(False)
        ])
        
        # 8) Definir indicadores para el gráfico
        params["__indicators_used"] = ["hl_mid", "ma_hl"]
        
        # 9) Metadatos adicionales para el plot
        params["__strategy_description"] = f"Crossover HL/2 MA (period={hl_period}, type={ma_type})"
        
        return df
    
    def _calculate_alma(self, df: pl.DataFrame, col_name: str, period: int, offset: float = 0.85, sigma: float = 6.0) -> pl.DataFrame:
        """
        Calcula ALMA (Arnaud Legoux Moving Average) usando Polars
        
        ALMA es una media móvil gaussiana con offset ajustable.
        - offset: controla dónde está el pico de la gaussiana (0.85 = cerca del final)
        - sigma: controla el ancho de la gaussiana
        """
        # Convertir a numpy para el cálculo
        values = df.select(pl.col(col_name)).to_numpy().flatten()
        n = len(values)
        
        # Calcular ALMA
        alma = np.full(n, np.nan)
        
        # Precalcular pesos gaussianos
        m = offset * (period - 1)
        s = period / sigma
        
        weights = np.zeros(period)
        for i in range(period):
            weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        
        # Normalizar pesos
        weights = weights / np.sum(weights)
        
        # Calcular ALMA usando ventana deslizante
        for i in range(period - 1, n):
            window = values[i - period + 1 : i + 1]
            if not np.isnan(window).any():
                alma[i] = np.sum(window * weights)
        
        # Añadir al DataFrame
        df = df.with_columns([
            pl.Series(name="ma_hl", values=alma)
        ])
        
        return df


# =============================================================================
# NOTAS ADICIONALES
# =============================================================================
# 
# Esta estrategia implementa un sistema simple de cruce de medias móviles:
# 
# 1. Punto Medio (HL/2): 
#    - Usa el promedio de high y low en lugar del close
#    - Esto puede reducir el ruido y dar señales más suaves
# 
# 2. Tipos de Media Móvil:
#    - SMA: Simple Moving Average (la más básica)
#    - EMA: Exponential Moving Average (da más peso a datos recientes)
#    - ALMA: Arnaud Legoux MA (gaussiana desplazada, muy suave)
# 
# 3. Lógica de Cruce:
#    - LONG: cuando el precio cruza de abajo hacia arriba la MA
#    - SHORT: cuando el precio cruza de arriba hacia abajo la MA
# 
# 4. Warmup:
#    - Se establece en period + 10 para asegurar que la MA esté estable
# 
# 5. Optimización:
#    - El período varía de 20 a 150 en pasos de 5
#    - El tipo de MA se puede seleccionar entre SMA, EMA y ALMA
# 
# =============================================================================

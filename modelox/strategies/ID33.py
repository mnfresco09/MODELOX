from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

class StrategyGaussianKinetic(EstrategiaBase):
    """
    ESTRATEGIA 1: INVERSIÓN CINÉTICA GAUSSIANA (GAUSSIAN KINETIC REVERSAL)
    
    Concepto:
      Busca reversiones ("agarrar el cuchillo") pero usa la física para confirmar el frenado.
      Combina la Transformada de Fisher (extremos estadísticos) con la Aceleración de ALMA.
    
    Lógica:
      1. Identificar extremo de precio con Fisher Transform (< -2.0 o > 2.0).
      2. No entrar solo por precio ("Falling Knife").
      3. Esperar a que la ACELERACIÓN de la media ALMA cruce cero (Frenado).
         - Long: Precio cae, pero la aceleración pasa de negativa a positiva (se frena la caída).
    
    Referencias:
      - Arnaud Legoux Moving Average (ALMA): Filtro Gaussiano con offset.
      - Fisher Transform: Convierte distribución de precios a campana Gaussiana normalizada.
    """

    combinacion_id = 33# ID único para el registro
    name = "ID3333MEAN REVERSION"
    
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        return {
            # Parámetros ALMA (Física)
            "alma_window": trial.suggest_int("alma_window", 20, 100, step=10),
            "alma_sigma": trial.suggest_int("alma_sigma", 10, 23),
            "alma_offset": trial.suggest_float("alma_offset", 0.75, 0.95, step=0.05),
            
            # Parámetros Fisher (Estadística)
            "fisher_len": trial.suggest_int("fisher_len", 180, 280, step=5),
            "fisher_threshold": trial.suggest_float("fisher_threshold", 1.8, 2.8, step=0.05),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # 1. Validación
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close", "high", "low"])

        # 2. Recuperar Parámetros
        alma_win = params.get("alma_window", 20)
        alma_sigma = params.get("alma_sigma", 6)
        alma_offset = params.get("alma_offset", 0.85)
        
        fisher_len = params.get("fisher_len", 10)
        fisher_thresh = params.get("fisher_threshold", 2.0)

        # 3. Metadata para visualización
        params["__warmup_bars"] = alma_win + 5
        params["__indicators_used"] = ["fisher", "alma_acc"]
        params["__indicator_bounds"] = {
            "fisher": {"panel": 1, "color": "orange", "upper": fisher_thresh, "lower": -fisher_thresh},
            "alma_acc": {"panel": 2, "color": "cyan", "mid": 0.0},
        }

        # 4. Modo Lazy
        q = df.lazy()

        # --- A. CÁLCULO DE ALMA (VECTORIZADO) ---
        # Polars no tiene ALMA nativo. Construimos la suma ponderada dinámicamente.
        # w = exp( - (i - offset)^2 / (2 * sigma^2) )
        
        # 1. Generar pesos en Python (Numpy)
        m = alma_offset * (alma_win - 1)
        s = alma_win / alma_sigma
        indices = np.arange(alma_win)
        weights = np.exp(-((indices - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum() # Normalizar para que sumen 1
        
        # 2. Construir expresión Polars: Sum(Price[t-i] * w[i])
        # Usamos shift() para acceder al pasado vectorialmente.
        alma_expr_parts = []
        for i, w in enumerate(weights):
            if w > 0.0001: # Optimización: ignorar pesos insignificantes
                # shift(i) toma el precio de hace 'i' barras (indices[i] va de 0 a win-1)
                # OJO: indices en ALMA suelen ser 0=más antiguo, pero aquí shift(0) es actual.
                # La fórmula ALMA estándar aplica pesos a la ventana. El peso 'm' (offset) 
                # suele estar cerca del dato reciente.
                # Mapeamos: indices[i] corresponde a shift(alma_win - 1 - i)
                shift_val = alma_win - 1 - i
                alma_expr_parts.append(pl.col("close").shift(shift_val) * w)
        
        alma_expr = sum(alma_expr_parts).alias("alma")
        q = q.with_columns(alma_expr)

        # --- B. CINEMÁTICA (Velocidad y Aceleración) ---
        # Velocidad = Delta ALMA
        # Aceleración = Delta Velocidad
        q = q.with_columns([
            pl.col("alma").diff().alias("alma_vel")
        ])
        q = q.with_columns([
            pl.col("alma_vel").diff().alias("alma_acc")
        ])

        # --- C. TRANSFORMADA DE FISHER ---
        # Normaliza precios a distribución casi Gaussiana para detectar extremos claros.
        # Formula Ehlers simplificada para Polars:
        # 1. Normalizar precio en ventana len a rango [-1, 1]
        
        min_low = pl.col("low").rolling_min(fisher_len)
        max_high = pl.col("high").rolling_max(fisher_len)
        
        # Posición estocástica cruda
        raw_pos = (pl.col("close") - min_low) / (max_high - min_low + 0.00001)
        # Re-escalar a [-1, 1] (aprox)
        norm_val = 2.0 * (raw_pos - 0.5)
        
        # Clamp para evitar log de infinitos (Fisher explota en 1 o -1)
        norm_val_clamped = norm_val.clip(-0.999, 0.999)
        
        # Transformada: 0.5 * ln((1+x)/(1-x))
        fisher_expr = (
            0.5 * ((1.0 + norm_val_clamped) / (1.0 - norm_val_clamped)).log()
        ).alias("fisher_raw")
        
        # Suavizado suave (opcional, Ehlers suele usarlo, aquí usamos raw para rapidez o EMA corta)
        q = q.with_columns(fisher_expr)
        q = q.with_columns(pl.col("fisher_raw").ewm_mean(span=3).alias("fisher"))

        # --- D. LÓGICA DE DISPARO (TRIGGER) ---
        
        # LONG SETUP:
        # 1. Fisher en sobreventa extrema (ej: < -2.0)
        # 2. TRIGGER: Aceleración cruza de Negativa a Positiva (Frenado de la caída)
        
        is_oversold = pl.col("fisher") < -fisher_thresh
        acc_cross_up = (pl.col("alma_acc") > 0) & (pl.col("alma_acc").shift(1) <= 0)
        
        # SHORT SETUP:
        # 1. Fisher en sobrecompra extrema (ej: > 2.0)
        # 2. TRIGGER: Aceleración cruza de Positiva a Negativa (Frenado de subida)
        
        is_overbought = pl.col("fisher") > fisher_thresh
        acc_cross_down = (pl.col("alma_acc") < 0) & (pl.col("alma_acc").shift(1) >= 0)

        # Señales finales
        # Nota: Usamos is_oversold de la vela actual o anterior para dar un poco de margen al setup
        sig_long = is_oversold & acc_cross_up
        sig_short = is_overbought & acc_cross_down

        # --- E. ENSAMBLAJE FINAL ---
        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
        ])

        return self.finalize_signals(q, keep_cols=["alma", "fisher", "alma_acc"]) 
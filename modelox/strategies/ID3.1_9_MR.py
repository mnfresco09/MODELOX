from __future__ import annotations
from typing import Any, Dict
import polars as pl
import numpy as np
from .ESTRATEGIA_BASE import EstrategiaBase

class StrategyGaussianKinetic(EstrategiaBase):
    """
    ESTRATEGIA ID 9: INVERSIÓN CINÉTICA GAUSSIANA (MEJORADA)
    
    CONCEPTO PRINCIPAL:
      ESTA ESTRATEGIA NO ADIVINA, ESPERA CONFIRMACIÓN FÍSICA.
      COMBINA LA ESTADÍSTICA (¿ESTÁ EL PRECIO LEJOS DE LO NORMAL?) CON LA 
      FÍSICA (¿HA DEJADO EL PRECIO DE ACELERAR EN SU CAÍDA?).
    
    PASOS DE LA LÓGICA (SIMPLIFICADO):
      1. ¿ESTÁ BARATO/CARO?: USAMOS FISHER TRANSFORM. SI TOCA -2.0, PREPARAMOS LA COMPRA.
      2. ¿FRENÓ LA CAÍDA?: MIRAMOS LA ACELERACIÓN DE ALMA. ESPERAMOS A QUE CRUCE CERO.
      3. ¿HAY COMPRADORES?: MIRAMOS LA VELA ACTUAL. SI NO ES VERDE, NO ENTRAMOS.
    
    MEJORAS APLICADAS EN ESTA VERSIÓN:
      - MEMORIA: SI EL PRECIO ESTUVO BARATO HACE 2 VELAS, TODAVÍA VALE (EVITA PERDER EL TREN).
      - FILTRO DE COLOR: SOLO COMPRAMOS SI LA VELA CIERRA EN VERDE (CONFIRMACIÓN VISUAL).
    """

    # --- CONFIGURACIÓN DE IDENTIDAD ---
    combinacion_id = 9
    name = "ID9.GAUSSIAN_KINETIC_REVERSION"
    
    # NO NECESITAMOS SALIDAS COMPLEJAS AQUÍ, USAMOS LAS ESTÁNDAR
    SALIDAS_PERSONALIZADAS = False

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        DEFINICIÓN DE PARÁMETROS PARA LA INTELIGENCIA ARTIFICIAL (OPTUNA).
        AQUÍ DEFINIMOS LOS RANGOS QUE LA IA PROBARÁ PARA ENCONTRAR LA MEJOR COMBINACIÓN.
        """
        return {
            # --- PARÁMETROS DE FÍSICA (ALMA) ---
            # VENTANA DE TIEMPO PARA CALCULAR LA CURVA
            "alma_window": trial.suggest_int("alma_window", 40, 65, step=1),
            # SUAVIDAD DE LA CURVA (MÁS ALTO = MÁS SUAVE)
            "alma_sigma": trial.suggest_int("alma_sigma", 7, 15),
            # RESPUESTA (0.85 SIGNIFICA QUE MIRA MÁS AL PRESENTE QUE AL PASADO)
            "alma_offset": trial.suggest_float("alma_offset", 0.80, 0.99, step=0.01),
            
            # --- PARÁMETROS DE ESTADÍSTICA (FISHER) ---
            # CUÁNTAS VELAS MIRAMOS PARA SABER SI ES CARO O BARATO
            "fisher_len": trial.suggest_int("fisher_len", 100, 330, step=2),
            # EL LIMITE PARA DECIR "ESTO ES UN EXTREMO" (EJ: 2.5 DESVIACIONES)
            "fisher_threshold": trial.suggest_float("fisher_threshold", 2.0, 3.0, step=0.05),
        }

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        MOTOR PRINCIPAL: AQUÍ OCURRE LA MAGIA MATEMÁTICA.
        CONVIERTE DATOS DE PRECIO EN SEÑALES DE COMPRA O VENTA.
        """
        
        # ----------------------------------------------------------------
        # 1. VALIDACIÓN Y PREPARACIÓN DE DATOS
        # ----------------------------------------------------------------
        self._init_params_metadata(params)
        # NECESITAMOS PRECIO DE CIERRE, ALTO, BAJO Y APERTURA (PARA EL COLOR DE LA VELA)
        self._require_columns(df, ["timestamp", "close", "high", "low", "open"])

        # RECUPERAMOS LOS VALORES DE LOS PARÁMETROS
        alma_win = params.get("alma_window", 20)
        alma_sigma = params.get("alma_sigma", 6)
        alma_offset = params.get("alma_offset", 0.85)
        
        fisher_len = params.get("fisher_len", 10)
        fisher_thresh = params.get("fisher_threshold", 2.0)

        # CONFIGURACIÓN PARA GRÁFICOS (SI QUEREMOS VERLO DESPUÉS)
        params["__warmup_bars"] = alma_win + 10
        params["__indicators_used"] = ["fisher", "alma_acc"]
        params["__indicator_bounds"] = {
            "fisher": {"panel": 1, "color": "orange", "upper": fisher_thresh, "lower": -fisher_thresh},
            "alma_acc": {"panel": 2, "color": "cyan", "mid": 0.0},
        }

        # INICIAMOS EL MODO RÁPIDO DE POLARS
        q = df.lazy()

        # ----------------------------------------------------------------
        # A. CÁLCULO DE ALMA (LA FÍSICA DEL MOVIMIENTO)
        # ----------------------------------------------------------------
        # ESTO CREA UNA MEDIA MÓVIL GAUSSIANA MUY PRECISA.
        
        # 1. MATEMÁTICA PURA PARA GENERAR LOS PESOS DE LA CURVA
        m = alma_offset * (alma_win - 1)
        s = alma_win / alma_sigma
        indices = np.arange(alma_win)
        weights = np.exp(-((indices - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum() # NORMALIZAR (QUE SUMEN 1)
        
        # 2. APLICAR LOS PESOS AL PRECIO (SUMA PONDERADA)
        alma_expr_parts = []
        for i, w in enumerate(weights):
            if w > 0.0001: # IGNORAMOS PESOS DIMINUTOS PARA IR RÁPIDO
                shift_val = alma_win - 1 - i
                alma_expr_parts.append(pl.col("close").shift(shift_val) * w)
        
        # GUARDAMOS LA COLUMNA "ALMA"
        q = q.with_columns(sum(alma_expr_parts).alias("alma"))

        # ----------------------------------------------------------------
        # B. CINEMÁTICA (VELOCIDAD Y ACELERACIÓN)
        # ----------------------------------------------------------------
        # VELOCIDAD = CUÁNTO CAMBIA LA MEDIA ALMA
        # ACELERACIÓN = CUÁNTO CAMBIA LA VELOCIDAD (AQUÍ ESTÁ EL SECRETO DEL FRENADO)
        
        q = q.with_columns(pl.col("alma").diff().alias("alma_vel"))
        q = q.with_columns(pl.col("alma_vel").diff().alias("alma_acc"))

        # ----------------------------------------------------------------
        # C. TRANSFORMADA DE FISHER (EL DETECTOR DE EXTREMOS)
        # ----------------------------------------------------------------
        # NORMALIZA EL PRECIO PARA DECIRNOS SI ESTÁ ESTADÍSTICAMENTE "RARO" (MUY ALTO O MUY BAJO).
        
        min_low = pl.col("low").rolling_min(fisher_len)
        max_high = pl.col("high").rolling_max(fisher_len)
        
        # CÁLCULO DE LA POSICIÓN RELATIVA
        raw_pos = (pl.col("close") - min_low) / (max_high - min_low + 0.00001)
        norm_val = 2.0 * (raw_pos - 0.5)
        # LIMITAMOS EL VALOR PARA QUE LAS MATEMÁTICAS NO EXPLOTEN
        norm_val_clamped = norm_val.clip(-0.999, 0.999)
        
        # FÓRMULA FINAL DE FISHER
        fisher_expr = (0.5 * ((1.0 + norm_val_clamped) / (1.0 - norm_val_clamped)).log()).alias("fisher_raw")
        
        q = q.with_columns(fisher_expr)
        # SUAVIZAMOS UN POCO LA LÍNEA FINAL
        q = q.with_columns(pl.col("fisher_raw").ewm_mean(span=3).alias("fisher"))

        # ----------------------------------------------------------------
        # D. LÓGICA DE DISPARO (LAS REGLAS DEL JUEGO)
        # ----------------------------------------------------------------
        
        # --- REGLA 1: ZONA EXTREMA (CON MEMORIA) ---
        # ¿ESTUVO EL PRECIO EN ZONA DE COMPRA/VENTA EN LAS ÚLTIMAS 3 VELAS?
        # ESTO NOS DA UN MARGEN DE TIEMPO PARA REACCIONAR.
        was_oversold = (pl.col("fisher") < -fisher_thresh).cast(pl.Int8).rolling_max(3) > 0
        was_overbought = (pl.col("fisher") > fisher_thresh).cast(pl.Int8).rolling_max(3) > 0
        
        # --- REGLA 2: EL GATILLO FÍSICO (CRUCE DE CERO) ---
        # LONG: LA ACELERACIÓN PASA DE NEGATIVA A POSITIVA (FRENAZO DE CAÍDA)
        # SHORT: LA ACELERACIÓN PASA DE POSITIVA A NEGATIVA (FRENAZO DE SUBIDA)
        acc_cross_up = (pl.col("alma_acc") > 0) & (pl.col("alma_acc").shift(1) <= 0)
        acc_cross_down = (pl.col("alma_acc") < 0) & (pl.col("alma_acc").shift(1) >= 0)
        
        # --- REGLA 3: FILTRO DE REALIDAD (COLOR DE LA VELA) ---
        # LONG: EXIGIMOS QUE LA VELA SEA VERDE (CLOSE > OPEN)
        # SHORT: EXIGIMOS QUE LA VELA SEA ROJA (CLOSE < OPEN)
        is_green_candle = pl.col("close") > pl.col("open")
        is_red_candle = pl.col("close") < pl.col("open")

        # ----------------------------------------------------------------
        # E. ENSAMBLAJE FINAL DE LA SEÑAL
        # ----------------------------------------------------------------
        
        # SEÑAL DE COMPRA (LONG) = ESTABA BARATO + FRENÓ LA CAÍDA + VELA VERDE
        sig_long = was_oversold & acc_cross_up & is_green_candle
        
        # SEÑAL DE VENTA (SHORT) = ESTABA CARO + FRENÓ LA SUBIDA + VELA ROJA
        sig_short = was_overbought & acc_cross_down & is_red_candle

        # AÑADIMOS LAS COLUMNAS AL DATAFRAME
        q = q.with_columns([
            self._as_bool(sig_long).alias("signal_long"),
            self._as_bool(sig_short).alias("signal_short"),
        ])

        # DEVOLVEMOS EL RESULTADO LIMPIO
        return self.finalize_signals(q, keep_cols=["alma", "fisher", "alma_acc"])
"""
================================================================================
MODELOX/PERTURBACIONES/AUGMENTER.PY — OHLCV DATA AUGMENTATION ENGINE
================================================================================

PROPÓSITO:
    Muta ligeramente los datos OHLCV respetando restricciones matemáticas
    estrictas de coherencia de velas financieras. Diseñado para inyectar
    un dataset diferente en cada trial de Optuna y así evitar overfitting.

MODELO: Deriva Aleatoria Anclada (Leashed Brownian Motion)
    En mercados cripto 24/7, Close[t] == Open[t+1]. El ruido independiente
    en O/C rompe esta continuidad creando gaps irreales.

    Solución en dos fases:

    FASE 1 — Deriva Acumulada (Macro Drift):
        Desplaza la gráfica entera mediante un multiplicador browniano
        que se acumula pero nunca excede ±MAX_DRIFT_PCT del precio real.
        Preserva la geometría relativa de cada vela (OHLC proporcional).

    FASE 2 — Fricción Microestructural (Mechas y Volumen):
        Sobre los precios macro, expande las mechas con ruido uniforme
        positivo sin tocar el cuerpo (Open/Close). Volumen log-normal.

RESTRICCIONES:
    - Cero Pandas: solo Polars Lazy API (pl.Expr)
    - Cero bucles for/apply: ruido vía map_batches + np.random.default_rng
    - .collect() se ejecuta una única vez al final

================================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


# Umbral mínimo de precio para evitar valores cero/negativos
_PRICE_FLOOR = 1e-8


# =============================================================================
# VALIDACIÓN DE COHERENCIA OHLCV
# =============================================================================

def assert_ohlcv_coherence(df: pl.DataFrame) -> bool:
    """
    Verifica ultrarrápido que un DataFrame OHLCV cumple restricciones.

    Condiciones:
        H ≥ O, H ≥ C, L ≤ O, L ≤ C, V ≥ 0

    Args:
        df: DataFrame con columnas open, high, low, close, volume.

    Returns:
        True si es coherente.

    Raises:
        ValueError: Si alguna fila viola las restricciones.
    """
    checks = df.select(
        (pl.col("high") >= pl.col("open")).all().alias("h_ge_o"),
        (pl.col("high") >= pl.col("close")).all().alias("h_ge_c"),
        (pl.col("low") <= pl.col("open")).all().alias("l_le_o"),
        (pl.col("low") <= pl.col("close")).all().alias("l_le_c"),
        (pl.col("volume") >= 0).all().alias("v_ge_0"),
    )

    row = checks.row(0, named=True)
    violations = [name for name, ok in row.items() if not ok]

    if violations:
        n_bad = df.filter(
            (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
            | (pl.col("volume") < 0)
        ).height
        raise ValueError(
            f"OHLCV incoherente: {violations} — {n_bad} filas violadas de {df.height}"
        )

    return True


# =============================================================================
# AUGMENTER PRINCIPAL
# =============================================================================

class HighFrequencyOHLCVAugmenter:
    """
    Perturbador de alta frecuencia para datos OHLCV.

    Modelo: Deriva Aleatoria Anclada (Leashed Brownian Motion).

    En mercados cripto 24/7:
      - Close[t] == Open[t+1], por lo que NO se puede inyectar
        ruido independiente a O y C sin crear gaps irreales.
      - La solución es un multiplicador browniano acumulativo
        que desplaza TODA la vela proporcionalmente, preservando
        la geometría relativa del cuerpo y las mechas.

    Parámetros:
        max_drift_pct:        Máxima desviación del precio real (±%).
                              Ejemplo: 0.10 = máximo ±10%. (default 0.10)
        drift_step_volatility: Volatilidad del paso browniano por vela.
                              Controla la "velocidad" de la deriva. (default 0.001)
        wick_noise_scale:     Escala de ruido uniforme para las mechas.
                              Solo expande, nunca contrae. (default 0.05)
        volume_noise_scale:   Escala de ruido log-normal para el volumen.
                              (default 0.10)
        seed:                 Semilla para reproducibilidad. None = aleatorio.
    """

    def __init__(
        self,
        max_drift_pct: float = 0.10,
        drift_step_volatility: float = 0.001,
        wick_noise_scale: float = 0.05,
        volume_noise_scale: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        self.max_drift_pct = max_drift_pct
        self.drift_step_volatility = drift_step_volatility
        self.wick_noise_scale = wick_noise_scale
        self.volume_noise_scale = volume_noise_scale
        self.seed = seed

    # ------------------------------------------------------------------
    # API PÚBLICA
    # ------------------------------------------------------------------

    def augment(self, df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Aplica data augmentation OHLCV en dos fases.

        FASE 1 — Deriva Acumulada (Macro Drift):
            1. Z_step ~ N(0, drift_step_volatility)
            2. CumDrift = cum_sum(Z_step)
            3. CumDrift = clip(CumDrift, -max_drift_pct, +max_drift_pct)
            4. M_t = 1.0 + CumDrift_t
            5. O_macro = O × M_t, H_macro = H × M_t, etc.

        FASE 2 — Fricción Microestructural:
            6. UW = H_macro - max(O_macro, C_macro)
            7. LW = min(O_macro, C_macro) - L_macro
            8. UW_new = UW × (1 + |U_uw| × wick_noise_scale)
            9. LW_new = LW × (1 + |U_lw| × wick_noise_scale)
           10. H_final = max(O_macro, C_macro) + UW_new
           11. L_final = min(O_macro, C_macro) - LW_new
           12. V_new = V × exp(Z_v × volume_noise_scale)

        Args:
            df: LazyFrame o DataFrame con columnas OHLCV estándar.

        Returns:
            pl.DataFrame mutado y coherente.
        """
        # Evaluar temporalmente si es LazyFrame para conocer la longitud total (n)
        # Esto asegura que todo el dataset tenga ruido coherente sin cortes por chunks
        df_eval = df.collect() if isinstance(df, pl.LazyFrame) else df
        n = df_eval.height
        lf = df_eval.lazy()

        # RNG para este augmentation
        rng = np.random.default_rng(self.seed)

        # Capturar en locales
        max_drift = self.max_drift_pct
        drift_vol = self.drift_step_volatility
        wick_scale = self.wick_noise_scale
        vol_scale = self.volume_noise_scale

        # Streams independientes
        rng_drift, rng_uuw, rng_ulw, rng_zv = rng.spawn(4)

        # Generar todos los arrays de ruido en NumPy de una sola vez
        # ¡Esto es ultrarrápido y evita problemas de chunks en map_batches!
        steps_arr = rng_drift.normal(0.0, drift_vol, n)
        uw_noise_arr = np.abs(rng_uuw.uniform(-1.0, 1.0, n))
        lw_noise_arr = np.abs(rng_ulw.uniform(-1.0, 1.0, n))
        v_noise_arr = rng_zv.standard_normal(n)

        # Inyectar las series en el LazyFrame
        lf = lf.with_columns(
            pl.Series("_steps", steps_arr, dtype=pl.Float64),
            pl.Series("_u_uw", uw_noise_arr, dtype=pl.Float64),
            pl.Series("_u_lw", lw_noise_arr, dtype=pl.Float64),
            pl.Series("_z_v", v_noise_arr, dtype=pl.Float64),
        )

        # ══════════════════════════════════════════════════════════════
        # FASE 1: DERIVA ACUMULADA (MACRO DRIFT)
        # ══════════════════════════════════════════════════════════════

        # Usar la función .cum_sum() nativa de Polars, la cual sí es segura
        # a través de los chunks de memoria
        lf = lf.with_columns(
            (pl.lit(1.0) + pl.col("_steps").cum_sum().clip(-max_drift, max_drift))
            .alias("_drift_mult")
        )

        # --- Paso 5: Escalar OHLC por M_t ---
        lf = lf.with_columns(
            (pl.col("open") * pl.col("_drift_mult"))
            .clip(lower_bound=_PRICE_FLOOR)
            .alias("open"),
            (pl.col("high") * pl.col("_drift_mult"))
            .clip(lower_bound=_PRICE_FLOOR)
            .alias("high"),
            (pl.col("low") * pl.col("_drift_mult"))
            .clip(lower_bound=_PRICE_FLOOR)
            .alias("low"),
            (pl.col("close") * pl.col("_drift_mult"))
            .clip(lower_bound=_PRICE_FLOOR)
            .alias("close"),
        )

        # ══════════════════════════════════════════════════════════════
        # FASE 2: FRICCIÓN MICROESTRUCTURAL (MECHAS Y VOLUMEN)
        # ══════════════════════════════════════════════════════════════

        # --- Paso 6-7: Extraer mechas macro ---
        #   UW_macro = H_macro - max(O_macro, C_macro)
        #   LW_macro = min(O_macro, C_macro) - L_macro
        lf = lf.with_columns(
            (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
            .clip(lower_bound=0.0)
            .alias("_uw"),
            (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
            .clip(lower_bound=0.0)
            .alias("_lw"),
        )

        # --- Paso 8-9: Expandir mechas con ruido uniforme positivo ---
        #   UW_new = UW × (1 + |U_uw| × wick_noise_scale)
        #   LW_new = LW × (1 + |U_lw| × wick_noise_scale)
        lf = lf.with_columns(
            (pl.col("_uw") * (1.0 + pl.col("_u_uw") * pl.lit(wick_scale)))
            .alias("_uw_new"),
            (pl.col("_lw") * (1.0 + pl.col("_u_lw") * pl.lit(wick_scale)))
            .alias("_lw_new"),
        )

        # --- Paso 10-11: Reconstruir H/L ---
        #   H_final = max(O_macro, C_macro) + UW_new
        #   L_final = min(O_macro, C_macro) - LW_new
        lf = lf.with_columns(
            (pl.max_horizontal(pl.col("open"), pl.col("close")) + pl.col("_uw_new"))
            .alias("high"),
            (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("_lw_new"))
            .clip(lower_bound=_PRICE_FLOOR)
            .alias("low"),
        )

        # --- Paso 12: Volumen log-normal ---
        #   V_new = V × exp(Z_v × volume_noise_scale)
        lf = lf.with_columns(
            (pl.col("volume") * (pl.col("_z_v") * pl.lit(vol_scale)).exp())
            .clip(lower_bound=0.0)
            .alias("volume"),
        )

        # ──────────────────────────────────────────────────────────────
        # LIMPIEZA: Eliminar columnas temporales
        # ──────────────────────────────────────────────────────────────
        _temp_cols = [
            "_steps", "_drift_mult",
            "_u_uw", "_u_lw", "_z_v",
            "_uw", "_lw", "_uw_new", "_lw_new",
        ]
        lf = lf.drop(_temp_cols)

        # ──────────────────────────────────────────────────────────────
        # COLLECT (una sola vez)
        # ──────────────────────────────────────────────────────────────
        return lf.collect()

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HighFrequencyOHLCVAugmenter("
            f"drift={self.max_drift_pct}, step_vol={self.drift_step_volatility}, "
            f"wick={self.wick_noise_scale}, vol={self.volume_noise_scale}, "
            f"seed={self.seed})"
        )

from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID19.PY
================================================================================
ID        : 19
NOMBRE    : ENTROEMA — Entropía Shannon Rolling + Cruce EMA
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  La entropía de Shannon mide el DESORDEN de una distribución.
  Aplicada a retornos logarítmicos:
    - Entropía ALTA  → retornos muy dispersos → mercado ruidoso → no operar
    - Entropía BAJA  → retornos concentrados → mercado predecible → tendencia real

  EDGE: Cuando la entropía está por debajo de su propia media histórica
  (z_entropy < 0), el mercado está en un régimen de baja incertidumbre.
  En ese régimen, el cruce de EMAs tiene mayor probabilidad de ser
  una tendencia real y no ruido aleatorio.

  Análogo al R² de ID2 pero desde la teoría de la información.

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  ENTROPÍA DE SHANNON ROLLING:
    - r_i = ln(close_i / close_{i-1})
    - Discretizar retornos en N_BINS=10 bins sobre la ventana
    - p_k = frecuencia relativa de cada bin
    - H = -Σ p_k * ln(p_k)  (entropía de Shannon en nats)
    - H_norm = H / ln(N_BINS)  (normalizada [0,1]: 0=orden perfecto, 1=ruido máximo)

  FILTRO DE RÉGIMEN (Z-score entropía, dinámico):
    - z_entropy = (H_norm - mean(H_norm, w)) / std(H_norm, w)
    - z_entropy < 0 → entropía por debajo de su media → régimen predecible → OPERAR
    - z_entropy ≥ 0 → entropía por encima de su media → ruido → NO OPERAR
    - Sin parámetro extra: totalmente dinámico y adaptativo

  DIRECCIÓN (cruce EMA):
    - EMA rápida: ema_fast (Optuna)
    - EMA lenta:  ema_slow (Optuna)
    - Cruce alcista: ema_fast cruza ema_slow hacia arriba → Long
    - Cruce bajista: ema_fast cruza ema_slow hacia abajo → Short

  SEÑAL LONG  : z_entropy < 0 AND cruce EMA alcista  [FLANCO]
  SEÑAL SHORT : z_entropy < 0 AND cruce EMA bajista   [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (2 parámetros):
  ema_fast       : int [5,  25] step=5
  ema_slow       : int [35, 60] step=5
  entropy_window : int [10, 60] step=5

PARÁMETROS FIJOS:
  N_BINS    : 10   (bins para discretizar retornos)
  z_entropy : < 0  (dinámico, sin umbral extra)

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - Entropía rolling via map_batches + Numba @njit
  - EMAs via ewm_mean nativo Polars
  - Z-score entropía 100% Polars vectorial
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    def _njit(*args, **kwargs):          # type: ignore[misc]
        def _deco(fn): return fn
        return _deco if (args and callable(args[0])) else _deco
    _NUMBA_AVAILABLE = False

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


# ==============================================================================
# NÚCLEO ENTROPÍA COMPILADO — nivel módulo, @njit(cache=True)
# Calcula entropía de Shannon normalizada sobre ventana rolling
# ==============================================================================
@_njit(cache=True)
def _entropy_numba(
    arr:     np.ndarray,
    w:       int,
    n_bins:  int,
    out_ent: np.ndarray,
) -> None:
    """
    Entropía de Shannon rolling compilada a C via Numba @njit.
    Para cada posición i:
      1. Extraer ventana de w retornos
      2. Discretizar en n_bins bins uniformes
      3. Calcular frecuencias relativas p_k
      4. H = -Σ p_k * ln(p_k)  normalizada por ln(n_bins)
    Modifica out_ent in-place.
    """
    n_tot    = len(arr)
    log_bins = np.log(float(n_bins))

    for i in range(w - 1, n_tot):
        # Extraer ventana
        sub_min =  1e18
        sub_max = -1e18
        for k in range(w):
            v = arr[i - w + 1 + k]
            if v < sub_min: sub_min = v
            if v > sub_max: sub_max = v

        rng = sub_max - sub_min
        if rng < 1e-12:
            # Todos los retornos iguales → entropía 0 (orden perfecto)
            out_ent[i] = 0.0
            continue

        # Contar frecuencias por bin
        counts = np.zeros(n_bins)
        for k in range(w):
            v   = arr[i - w + 1 + k]
            idx = int((v - sub_min) / rng * (n_bins - 1))
            if idx < 0:       idx = 0
            if idx >= n_bins: idx = n_bins - 1
            counts[idx] += 1.0

        # Entropía de Shannon normalizada
        h = 0.0
        for b in range(n_bins):
            p = counts[b] / w
            if p > 1e-12:
                h -= p * np.log(p)

        out_ent[i] = h / log_bins  # normalizada [0, 1]


class EstrategiaID19ENTROEMA(EstrategiaBase):
    """
    ENTROEMA — Entropía de Shannon rolling como filtro de régimen.
    Opera solo cuando el mercado está en régimen de baja incertidumbre
    (z_entropy < 0), confirmando la dirección con cruce de EMAs.
    Inspirado en el rigor estadístico de ID2 pero desde la teoría
    de la información en lugar de la regresión lineal.
    """

    combinacion_id: int          = 19
    name: str                    = "ENTROEMA"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        3 parámetros Optuna.
        Fijos: N_BINS=10, umbral z_entropy < 0 (dinámico)
        Total combinaciones: 5 x 6 x 11 = 330 — controlado.
        """
        return {
            "ema_fast":       trial.suggest_int("ema_fast",       14,  14, step=1),
            "ema_slow":       trial.suggest_int("ema_slow",      50,  50, step=1),
            "entropy_window": trial.suggest_int("entropy_window", 40,  40, step=1),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) Retorno logarítmico
          B) Entropía Shannon rolling via map_batches + Numba JIT
          C) Z-score de entropía (Polars vectorial) → filtro dinámico
          D) EMA rápida + EMA lenta (Polars ewm_mean)
          E) Cruce EMA (cambio de signo del diferencial)
          F) Condiciones base: z_entropy < 0 AND cruce EMA
          G) Flancos
          H) finalize_signals (1 collect)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        ema_fast       = int(params.get("ema_fast",       10))
        ema_slow       = int(params.get("ema_slow",       40))
        entropy_window = int(params.get("entropy_window", 30))
        n_bins         = 10    # FIJO: bins para discretizar retornos

        params["__warmup_bars"]     = entropy_window * 2 + ema_slow + 2
        params["__indicators_used"] = [
            "log_return", "entropy", "z_entropy", "ema_fast_val", "ema_slow_val"
        ]
        params["__indicator_specs"] = {
            "ema_fast_val": {"panel": "main", "color": "#00E676", "tipo": "line"},
            "ema_slow_val": {"panel": "main", "color": "#FF1744", "tipo": "line"},
            "entropy":      {"panel": "sub1", "color": "#FF9800", "tipo": "line"},
            "z_entropy":    {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "entropy":   {"lo": 0.0,  "hi": 1.0,  "mid": 0.5},
            "z_entropy": {"lo": None, "hi": None,  "mid": 0.0},
        }

        # ── FASE A: Retorno logarítmico ────────────────────────────────────────
        q = df.lazy().with_columns([
            (pl.col("close") / pl.col("close").shift(1))
            .log(base=2.718281828)
            .alias("log_return")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — Entropía Shannon rolling via map_batches + Numba JIT
        # Mide el desorden de la distribución de retornos en la ventana
        # Resultado normalizado [0,1]: 0=orden perfecto, 1=ruido máximo
        # ══════════════════════════════════════════════════════════════════════
        window = entropy_window
        bins   = n_bins

        def _entropy_batch(s: pl.Series) -> pl.Series:
            arr     = s.to_numpy(allow_copy=True).astype(np.float64)
            n_tot   = len(arr)
            out_ent = np.full(n_tot, np.nan)
            _entropy_numba(arr, window, bins, out_ent)
            return pl.Series(name="entropy", values=out_ent.tolist())

        q = q.with_columns([
            pl.col("log_return")
            .map_batches(_entropy_batch, return_dtype=pl.Float64)
            .alias("entropy")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Z-score de entropía (filtro dinámico de régimen)
        # z_entropy < 0 → entropía por debajo de su propia media histórica
        # → mercado más predecible que de costumbre → régimen operable
        # Sin umbral fijo: totalmente adaptativo al activo y timeframe
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("entropy").rolling_mean(window_size=window).alias("_ent_mean"),
            pl.col("entropy").rolling_std (window_size=window).alias("_ent_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_ent_std").abs() > 1e-12)
            .then((pl.col("entropy") - pl.col("_ent_mean")) / pl.col("_ent_std"))
            .otherwise(0.0)
            .alias("z_entropy")
        ])

        # Régimen operable: z_entropy < 0
        q = q.with_columns([
            (pl.col("z_entropy") < 0.0)
            .fill_null(False)
            .alias("_regimen_ok")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — EMA rápida + EMA lenta (Polars ewm_mean nativo)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close")
            .ewm_mean(span=ema_fast, adjust=False)
            .alias("ema_fast_val"),

            pl.col("close")
            .ewm_mean(span=ema_slow, adjust=False)
            .alias("ema_slow_val"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — Cruce EMA (cambio de signo del diferencial)
        # Diferencial = ema_fast - ema_slow
        # Cruce alcista: diferencial pasa de negativo a positivo
        # Cruce bajista: diferencial pasa de positivo a negativo
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (pl.col("ema_fast_val") - pl.col("ema_slow_val")).alias("_ema_diff")
        ])

        q = q.with_columns([
            pl.col("_ema_diff").shift(1).fill_null(0.0).alias("_ema_diff_prev")
        ])

        q = q.with_columns([
            (
                (pl.col("_ema_diff_prev") <= 0.0) &
                (pl.col("_ema_diff")      >  0.0)
            ).fill_null(False).alias("_cruce_long"),

            (
                (pl.col("_ema_diff_prev") >= 0.0) &
                (pl.col("_ema_diff")      <  0.0)
            ).fill_null(False).alias("_cruce_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE F — Condiciones base
        # Long:  régimen predecible (z_entropy < 0) AND cruce EMA alcista
        # Short: régimen predecible (z_entropy < 0) AND cruce EMA bajista
        # Mutuamente excluyentes: cruce no puede ser alcista y bajista a la vez
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_regimen_ok") &
                pl.col("_cruce_long")
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_regimen_ok") &
                pl.col("_cruce_short")
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE G: Flancos ────────────────────────────────────────────────────
        q = q.with_columns([
            self._as_bool(
                pl.col("_cond_long") &
                ~pl.col("_cond_long").shift(1).fill_null(False)
            ).alias("signal_long"),

            self._as_bool(
                pl.col("_cond_short") &
                ~pl.col("_cond_short").shift(1).fill_null(False) &
                ~pl.col("_cond_long")
            ).alias("signal_short"),
        ])

        # ── FASE H: finalize_signals ───────────────────────────────────────────
        return self.finalize_signals(
            q,
            keep_cols=[
                "log_return",
                "entropy",
                "z_entropy",
                "ema_fast_val",
                "ema_slow_val",
            ],
        )
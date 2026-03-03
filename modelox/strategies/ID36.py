from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID36.PY
================================================================================
ID        : 36
NOMBRE    : HVNREV — High Volume Node + Z-score Precio Reversión
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
CONCEPTO:
  Las zonas donde se ha operado mucho volumen históricamente actúan como
  soporte y resistencia — el mercado "recuerda" esos niveles. Cuando el
  precio se acerca a una de estas zonas (HVN) y además está estadísticamente
  extendido respecto a ella, la probabilidad de reversión aumenta.

  HIGH VOLUME NODE (HVN):
    En una ventana de N velas, discretizar el rango de precios en bins.
    El bin con mayor volumen acumulado = zona HVN (Point of Control simplificado)
    hvn_price = precio medio del bin con máximo volumen

  PROXIMIDAD LOGARÍTMICA (robusto entre activos):
    dist_log = |ln(close) - ln(hvn_price)| = |ln(close/hvn_price)|
    cerca_hvn = dist_log ≤ proximity_log
    Independiente del nivel de precio — BTC a 60k y altcoin a 0.001
    tienen la misma escala logarítmica.

  Z-SCORE PRECIO RESPECTO A HVN:
    z_hvn = (close - hvn_price) / std(close, hvn_window)
    z_hvn > +z_threshold → precio sobre HVN estadísticamente → Short
    z_hvn < -z_threshold → precio bajo HVN estadísticamente → Long

  SEÑAL:
    Proximidad logarítmica a HVN AND Z-score extendido en dirección opuesta
    → el precio llegó a la zona Y está sobreextendido → reversión hacia HVN

--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  HVN ROLLING (ventana=hvn_window, bins=n_bins) via Numba:
    Para cada vela i, en la ventana [i-hvn_window+1 : i+1]:
      - Calcular rango [min(close), max(close)]
      - Dividir en n_bins bins de igual ancho
      - Acumular volumen por bin
      - hvn_price = precio medio del bin con mayor volumen acumulado

  PROXIMIDAD LOGARÍTMICA:
    dist_log = |ln(close/hvn_price)|
    cerca_hvn = dist_log ≤ proximity_log

  Z-SCORE RELATIVO AL HVN:
    close_std = rolling_std(close, hvn_window)
    z_hvn = (close - hvn_price) / close_std

  SEÑAL LONG  : cerca_hvn AND z_hvn < -z_threshold  [FLANCO]
  SEÑAL SHORT : cerca_hvn AND z_hvn > +z_threshold  [FLANCO]

SALIDAS:
  SALIDAS_PERSONALIZADAS = False

RANGOS OPTUNA (4 parámetros):
  hvn_window    : int   [30,  150] step=10
  n_bins        : int   [10,   30] step=5
  z_threshold   : float [0.5,  2.0] step=0.25
  proximity_log : float [0.002,0.01] step=0.002

PARÁMETROS FIJOS:
  ninguno relevante — todo Optuna o derivado

TIMEFRAME: Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - HVN via map_batches + Numba @njit (loop ventana + bins acumulación)
  - Proximidad logarítmica via expresión Polars vectorial (log nativo)
  - Z-score via rolling_std Polars vectorial
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
# NÚCLEO HVN — High Volume Node rolling via Numba
# ==============================================================================
@_njit(cache=True)
def _hvn_rolling(
    close_arr:  np.ndarray,
    vol_arr:    np.ndarray,
    w:          int,
    n_bins:     int,
    out_hvn:    np.ndarray,
) -> None:
    """
    Para cada vela i calcula el precio HVN en la ventana [i-w+1 : i+1].
    HVN = precio medio del bin con mayor volumen acumulado.
    Modifica out_hvn in-place.
    """
    n_tot = len(close_arr)
    for i in range(w - 1, n_tot):
        # Ventana actual
        c_min = close_arr[i - w + 1]
        c_max = close_arr[i - w + 1]
        for k in range(w):
            c = close_arr[i - w + 1 + k]
            if c < c_min:
                c_min = c
            if c > c_max:
                c_max = c

        rng = c_max - c_min
        if rng < 1e-12:
            # Sin rango — HVN es el precio actual
            out_hvn[i] = close_arr[i]
            continue

        bin_width = rng / float(n_bins)

        # Acumular volumen por bin
        vol_bins = np.zeros(n_bins)
        for k in range(w):
            c = close_arr[i - w + 1 + k]
            v = vol_arr  [i - w + 1 + k]
            b = int((c - c_min) / bin_width)
            if b >= n_bins:
                b = n_bins - 1
            vol_bins[b] += v

        # Bin con mayor volumen
        max_vol = vol_bins[0]
        max_bin = 0
        for b in range(1, n_bins):
            if vol_bins[b] > max_vol:
                max_vol = vol_bins[b]
                max_bin = b

        # Precio medio del bin ganador
        out_hvn[i] = c_min + (max_bin + 0.5) * bin_width


class EstrategiaID36HVNREV(EstrategiaBase):
    """
    HVNREV — High Volume Node + Z-score Precio Reversión.
    Soporte/resistencia dinámica por volumen histórico + reversión
    estadística con proximidad logarítmica (escala-independiente).
    """

    combinacion_id: int          = 36
    name: str                    = "HVNREV"
    SALIDAS_PERSONALIZADAS: bool = False
    timeframe_entry: Optional[str] = None
    timeframe_exit:  Optional[str] = None

    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        4 parámetros Optuna.
        Total combinaciones: 13 x 5 x 7 x 5 = ~2.275 → Optuna muestrea.
        """
        return {
            "hvn_window":    trial.suggest_int  ("hvn_window",    30,  150, step=10),
            "n_bins":        trial.suggest_int  ("n_bins",        10,   30, step=5),
            "z_threshold":   trial.suggest_float("z_threshold",  0.5,  2.0, step=0.25),
            "proximity_log": trial.suggest_float("proximity_log",0.002,0.01, step=0.002),
        }

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) HVN rolling via map_batches + Numba
          B) Proximidad logarítmica: |ln(close/hvn_price)| ≤ proximity_log
          C) Z-score precio relativo al HVN
          D) Condiciones base: cerca_hvn AND z_hvn extendido
          E) Flancos + finalize_signals
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close", "volume"])

        hvn_window    = int  (params.get("hvn_window",    100))
        n_bins        = int  (params.get("n_bins",         20))
        z_threshold   = float(params.get("z_threshold",   1.0))
        proximity_log = float(params.get("proximity_log", 0.005))

        params["__warmup_bars"]     = hvn_window + 4
        params["__indicators_used"] = ["hvn_price", "z_hvn", "dist_log"]
        params["__indicator_specs"] = {
            "hvn_price": {"panel": "main", "color": "#FFD600", "tipo": "line"},
            "z_hvn":     {"panel": "sub1", "color": "#FF9800", "tipo": "line"},
            "dist_log":  {"panel": "sub2", "color": "#00E5FF", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "z_hvn":    {"lo": -z_threshold,   "hi": z_threshold,   "mid": 0.0},
            "dist_log": {"lo": 0.0,             "hi": None,          "mid": proximity_log},
        }

        q = df.lazy()

        # ══════════════════════════════════════════════════════════════════════
        # FASE A — HVN rolling via map_batches + Numba
        # Para cada vela: ventana [i-w+1:i+1] → bin con más volumen → precio
        # ══════════════════════════════════════════════════════════════════════
        w      = hvn_window
        n_b    = n_bins

        def _hvn_batch(s: pl.Series) -> pl.Series:
            arr       = s.to_numpy(allow_copy=True)
            n_tot     = len(arr)

            # Compatibilidad robusta entre versiones de Polars:
            # cada fila de struct puede venir como dict/Row/tuple.
            def _get2(row):
                try:
                    # caso dict-like: {'close':..., 'volume':...}
                    return row["close"], row["volume"]
                except Exception:
                    # fallback posicional
                    return row[0], row[1]

            vals = [_get2(r) for r in arr]
            close_arr = np.array([v[0] for v in vals], dtype=np.float64)
            vol_arr   = np.array([v[1] for v in vals], dtype=np.float64)
            out_hvn   = np.full(n_tot, np.nan)
            _hvn_rolling(close_arr, vol_arr, w, n_b, out_hvn)
            return pl.Series(name="hvn_price", values=out_hvn.tolist())

        q = q.with_columns([
            pl.struct(["close", "volume"])
            .map_batches(_hvn_batch, return_dtype=pl.Float64)
            .alias("hvn_price")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — Proximidad logarítmica al HVN
        # dist_log = |ln(close/hvn_price)|
        # Robusto entre activos — misma escala para BTC y altcoins
        # cerca_hvn = dist_log ≤ proximity_log
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.when(
                (pl.col("hvn_price") > 1e-12) & (pl.col("close") > 1e-12)
            )
            .then(
                (pl.col("close") / pl.col("hvn_price")).log(base=2.718281828)
                .abs()
            )
            .otherwise(999.0)
            .alias("dist_log")
        ])

        q = q.with_columns([
            (pl.col("dist_log") <= proximity_log)
            .fill_null(False)
            .alias("_cerca_hvn")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — Z-score precio relativo al HVN
        # z_hvn = (close - hvn_price) / std(close, hvn_window)
        # Mide cuántas desviaciones está el precio respecto a la zona HVN
        # > +z_threshold → precio sobre HVN estadísticamente → Short
        # < -z_threshold → precio bajo HVN estadísticamente → Long
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("close").rolling_std(window_size=w).alias("_close_std")
        ])

        q = q.with_columns([
            pl.when(pl.col("_close_std").abs() > 1e-12)
            .then(
                (pl.col("close") - pl.col("hvn_price")) / pl.col("_close_std")
            )
            .otherwise(0.0)
            .alias("z_hvn")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Condiciones base
        # Long:  cerca del HVN AND precio estadísticamente bajo respecto a él
        #   → el precio llegó a la zona HVN por debajo → rebote alcista
        # Short: cerca del HVN AND precio estadísticamente alto respecto a él
        #   → el precio llegó a la zona HVN por encima → rebote bajista
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_cerca_hvn") &
                (pl.col("z_hvn") < -z_threshold) &
                pl.col("hvn_price").is_not_null()
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_cerca_hvn") &
                (pl.col("z_hvn") > z_threshold) &
                pl.col("hvn_price").is_not_null()
            ).fill_null(False).alias("_cond_short"),
        ])

        # ── FASE E: Flancos + finalize ─────────────────────────────────────────
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

        # Validación defensiva del esquema antes del collect para evitar KeyError poco claro
        try:
            schema_cols = list(q.schema.keys())
            required = {"timestamp", "signal_long", "signal_short", "hvn_price", "z_hvn", "dist_log"}
            missing = [c for c in required if c not in schema_cols]
            if missing:
                raise ValueError(
                    f"ID36 finalize_signals: faltan columnas antes de collect: {missing}. "
                    f"Schema actual: {schema_cols}"
                )
        except Exception as _e:
            # Re-lanzar con contexto específico de la estrategia
            raise

        return self.finalize_signals(
            q,
            keep_cols=["hvn_price", "z_hvn", "dist_log"],
        )
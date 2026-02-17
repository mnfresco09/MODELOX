"""
================================================================================
VISUAL/GRAFICO.PY — MOTOR GRÁFICO VECTORIZADO (ZERO-LAG)
================================================================================

PROPÓSITO:
    Generación de gráficos interactivos ligeros y dinámicos.
    Usa arquitectura Zero-Lag de alineación (SATA) para sincronización perfecta
    entre velas, indicadores y señales.

FUNCIONALIDAD:
    1. Alineación vectorizada O(log n) de indicadores.
    2. Detección automática de indicadores desde la estrategia.
    3. Normalización centralizada de timestamps (Unix int64).
    4. Serialización ultra-rápida (orjson) para el frontend.

================================================================================
"""

# =============================================================================
# GRAFICO (UNIFICADO)
# =============================================================================
#
# Este archivo es el ÚNICO responsable de la gráfica.
#
# CONTRATO (por trial):
# - Recibe `params` (normalmente `TrialArtifacts.params_reporting`).
# - Si existe `params["__indicators_used"]`, SOLO se dibujan esas columnas.
#   Esto permite que cada trial pinte exactamente los indicadores que calculó.
#
# INDICADORES:
# - Los cálculos de indicadores se hacen dentro de cada estrategia.
# - El plot solo consume columnas ya calculadas (guiado por `params`).
#
# =============================================================================

from __future__ import annotations

import os
import re
from typing import Optional, List, Union, Dict, Any, TYPE_CHECKING

import numpy as np

# Ultra-fast JSON serialization
try:
    import orjson  # type: ignore[reportMissingImports]
    HAS_ORJSON = True

    def _dumps(obj: dict) -> str:
        """Ultra-fast JSON serialization with orjson.
        
        OPT_SERIALIZE_NUMPY: Serialize numpy arrays directly (zero-copy)
        OPT_NON_STR_KEYS: Allow integer keys in dicts
        """
        return orjson.dumps(
            obj,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
        ).decode("utf-8")

    def _dumps_bytes(obj: dict) -> bytes:
        """Return raw bytes for streaming write (avoids decode overhead)."""
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)

except ImportError:
    import json
    HAS_ORJSON = False

    class _NumpyEncoder(json.JSONEncoder):
        """Fallback encoder for numpy types when orjson is unavailable."""
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return super().default(obj)

    def _dumps(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":"), cls=_NumpyEncoder)

    def _dumps_bytes(obj: dict) -> bytes:
        return _dumps(obj).encode("utf-8")

# Polars support
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import pandas as pd

if TYPE_CHECKING:
    DataFrameType = Union[pd.DataFrame, "pl.DataFrame"]


# =============================================================================
# INDICATOR DETECTION CONFIGURATION (Strategy-driven)
# =============================================================================
#
# IMPORTANT:
# - No hardcoded indicator lists, bounds, or colors.
# - The strategy decides what to plot via:
#     params["__indicators_used"]: list[str]
#     params["__indicator_bounds"]: dict[col -> dict[level_name -> value]]
#     params["__indicator_specs"]: dict[col -> {panel,type,color,name,precision,bounds}]
#

_COLOR_PALETTE = [
  "#60a5fa",  # blue
  "#f472b6",  # pink
  "#fb923c",  # orange
  "#22c55e",  # green
  "#a78bfa",  # purple
  "#22d3ee",  # cyan
  "#fbbf24",  # amber
  "#ef4444",  # red
  "#94a3b8",  # gray
]


def _color_for(name: str) -> str:
  try:
    idx = abs(hash(name)) % len(_COLOR_PALETTE)
  except Exception:
    idx = 0
  return _COLOR_PALETTE[idx]


def _get_indicator_specs(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if not params:
    return {}
  specs = params.get("__indicator_specs", None)
  return specs if isinstance(specs, dict) else {}


def _get_indicator_bounds(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if not params:
    return {}
  bounds = params.get("__indicator_bounds", None)
  return bounds if isinstance(bounds, dict) else {}


def _is_overlay_heuristic(series: "pd.Series", price_range: tuple) -> bool:
  """Heurística mejorada: overlay sólo si el indicador claramente está en escala precio.
  
  Indicadores normalizados, z-scores, osciladores, etc. NO son overlay.
  Un overlay real (MA, ALMA, bandas) debe:
    - Tener valores en el mismo orden de magnitud que el precio.
    - Tener un mínimo > 0 para activos como BTC/GOLD (precio siempre positivo).
  """
  try:
    s = series.dropna()
    if s.empty:
      return False
    min_p, max_p = float(price_range[0]), float(price_range[1])
    if not (np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p):
      return False
    ind_min = float(s.min())
    ind_max = float(s.max())
    if not (np.isfinite(ind_min) and np.isfinite(ind_max)):
      return False

    # REGLA 1: Si el indicador puede ser negativo y el precio mínimo es > 100,
    # es muy probable que sea un oscilador/z-score, NO overlay.
    if ind_min < 0 and min_p > 100:
      return False

    # REGLA 2: Si el rango del indicador es pequeño (ej: -3 a +3 para z-score),
    # y el precio está en miles/cientos, no es overlay.
    ind_span = ind_max - ind_min
    if ind_span < 20 and min_p > 50:
      return False

    # REGLA 3: Overlay real debe estar dentro del rango de precio.
    price_span = max_p - min_p
    within_min = (ind_min >= (min_p - 0.15 * price_span))
    within_max = (ind_max <= (max_p + 0.15 * price_span))
    span_ok = ind_span >= 0.1 * price_span and ind_span <= 1.5 * price_span

    return within_min and within_max and span_ok
  except Exception:
    return False


def _detect_indicators(
  df: "pd.DataFrame",
  price_range: tuple,
  params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Detecta qué columnas graficar, 100% guiado por estrategia.

  Si la estrategia proporciona `__indicators_used`, se usan SOLO esas columnas.
  Si no, se infiere de las columnas del DF excluyendo OHLCV/señales.
  """

  skip_cols = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "signal_long",
    "signal_short",
    "_session_day",
    # Columnas internas que NO deben graficarse
    "cycle_id",
    "CYCLE_ID",
    "bar_index",
    "BAR_INDEX",
    "row_nr",
    "ROW_NR",
  }

  # Patrones de indicadores que SIEMPRE van como overlay (en el gráfico de precios)
  # NOTA: Se comparan como PALABRAS COMPLETAS (separadas por _ o inicio/fin de string)
  # para evitar falsos positivos (ej: "ma" no debe matchear "alma_acc")
  OVERLAY_PATTERNS = (
    "ma", "ema", "sma", "zlema", "alma", "wma", "hma", "kama", "dema", "tema",
    "vwma", "vwap", "pivot", "support", "resistance", "band", "upper", "lower",
    "bb_", "boll", "keltner", "donchian", "atr_band", "supertrend",
  )

  # Sufijos/patrones que indican indicadores derivados → NUNCA son overlay
  # (aceleración, velocidad, diferencia, señal, histograma, z-score, ratio, etc.)
  NON_OVERLAY_SUFFIXES = (
    "_acc", "_vel", "_diff", "_roc", "_signal", "_sig", "_hist",
    "_zscore", "_z", "_norm", "_pct", "_ratio", "_slope", "_delta",
    "_momentum", "_mom", "_osc", "_divergence", "_div",
  )
  # Nombres que NUNCA son overlay (osciladores conocidos)
  NON_OVERLAY_NAMES = (
    "fisher", "rsi", "mfi", "cci", "stoch", "macd", "adx", "atr",
    "obv", "cmf", "willr", "dpo", "trix", "roc", "momentum",
    "zscore", "z_score",
  )

  def _is_overlay_by_name(col_lower: str) -> bool:
    """Verifica si el nombre del indicador corresponde a un overlay REAL.
    Usa word-boundary matching: 'alma' matchea 'alma' pero NO 'alma_acc'."""
    # PRIMERO: Si tiene sufijo de derivado, NUNCA es overlay
    if any(col_lower.endswith(suf) for suf in NON_OVERLAY_SUFFIXES):
      return False
    # Si el nombre completo es un oscilador conocido, NUNCA es overlay
    # (también verificar como parte separada por _)
    col_parts = set(col_lower.split('_'))
    if any(name in col_parts for name in NON_OVERLAY_NAMES):
      return False
    # Verificar overlay patterns con word-boundary matching
    for pat in OVERLAY_PATTERNS:
      # Matchear como palabra completa: el patrón debe ser una 'parte' del nombre
      # separada por _ (o ser el nombre completo)
      if pat in col_parts:
        return True
      # También matchear si el nombre EMPIEZA con el patrón seguido de _
      if col_lower.startswith(pat + '_') or col_lower == pat:
        return True
      # Matchear bb_ y similares que tienen _ en el patrón
      if '_' in pat and pat in col_lower:
        return True
    return False

  indicators_used = params.get("__indicators_used", None) if params else None
  if isinstance(indicators_used, list) and indicators_used:
    candidate_cols = [c for c in indicators_used if isinstance(c, str)]
  else:
    candidate_cols = [
      c
      for c in df.columns
      if isinstance(c, str) and c not in skip_cols and not c.startswith("_")
    ]

  # Filtrar columnas que están en skip_cols (case-insensitive)
  skip_lower = {s.lower() for s in skip_cols}
  candidate_cols = [c for c in candidate_cols if c.lower() not in skip_lower]

  specs = _get_indicator_specs(params)
  bounds_map = _get_indicator_bounds(params)

  overlays: List[Dict[str, Any]] = []
  sub_panels: List[Dict[str, Any]] = []

  for col in candidate_cols:
    if col not in df.columns:
      continue
    spec = specs.get(col, {}) if isinstance(specs.get(col, {}), dict) else {}

    panel = spec.get("panel", None)

    # PRIORIDAD 0: Respetar panel info de __indicator_bounds
    # Si la estrategia define panel como "overlay"/"sub" o numérico (1,2,3 = sub-panel)
    if panel not in {"overlay", "sub"}:
      # Verificar si __indicator_bounds tiene info de panel para esta columna
      col_bounds = bounds_map.get(col, None)
      if isinstance(col_bounds, dict) and "panel" in col_bounds:
        bounds_panel = col_bounds["panel"]
        if bounds_panel == "overlay":
          panel = "overlay"
        elif isinstance(bounds_panel, (int, float)) or bounds_panel == "sub":
          panel = "sub"  # Numérico (1, 2, 3...) = sub-panel dedicado

    if panel not in {"overlay", "sub"}:
      # PRIORIDAD 1: Word-boundary matching contra patrones conocidos
      col_lower = col.lower()
      if _is_overlay_by_name(col_lower):
        panel = "overlay"
      else:
        # PRIORIDAD 2: Heurística por valores
        panel = "overlay" if _is_overlay_heuristic(df[col], price_range) else "sub"

    series_type = str(spec.get("type", "line"))
    if series_type not in {"line", "histogram"}:
      series_type = "line"

    color = str(spec.get("color", _color_for(col)))
    name = str(spec.get("name", col.upper()))

    bounds = spec.get("bounds", None)
    if not isinstance(bounds, dict):
      bounds = bounds_map.get(col, None)
    # Fallback: infer bounds/levels from params using generic naming patterns.
    if not isinstance(bounds, dict) and params:
      inferred = _extract_indicator_params_from_optuna(params, col)
      bounds = inferred if isinstance(inferred, dict) and inferred else None
    if not isinstance(bounds, dict):
      bounds = None

    precision = spec.get("precision", None)
    if not isinstance(precision, int):
      precision = 2 if panel == "overlay" else 4

    if panel == "overlay":
      overlays.append({"col": col, "color": color, "type": series_type, "precision": precision})
    else:
      sub_panels.append(
        {
          "col": col,
          "name": name,
          "color": color,
          "type": series_type,
          "bounds": bounds,
          "precision": precision,
        }
      )

  return {"overlays": overlays, "sub_panels": sub_panels}


# =============================================================================
# VECTORIZED DATA PREPARATION (v6.0 - ZERO-LAG ARCHITECTURE)
# =============================================================================
# KEY PRINCIPLE: Single Authoritative Timestamp Array (SATA)
# All data (candles, indicators, markers) MUST reference the same ts_q array.
# This eliminates any possibility of desynchronization.
# =============================================================================

def _normalize_timestamps_to_unix(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert any timestamp format to Unix Epoch seconds (Int64).
    This is the ONLY function that should perform timestamp conversion.
    
    Supports:
    - datetime64[ns], datetime64[s], datetime64[ms], etc.
    - Float timestamps (assumed to be Unix seconds)
    - Already Int64 timestamps
    
    Returns:
        np.ndarray[np.int64] - Unix seconds
    """
    if np.issubdtype(timestamps.dtype, np.datetime64):
        # Convert any datetime64 variant to seconds precision, then to int64
        return timestamps.astype('datetime64[s]').astype(np.int64)
    elif np.issubdtype(timestamps.dtype, np.floating):
        return np.floor(timestamps).astype(np.int64)
    elif np.issubdtype(timestamps.dtype, np.integer):
        return timestamps.astype(np.int64)
    else:
        # Fallback: try to parse as datetime64
        return timestamps.astype('datetime64[s]').astype(np.int64)


def _prepare_ohlcv_vectorized(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    price_precision: int = 2,
) -> tuple:
    """
    Vectorized OHLCV preparation for Lightweight Charts.
    
    RETURNS the authoritative timestamp array (ts_q) that ALL other
    data (indicators, markers) MUST reference for perfect alignment.
    """
    # CRITICAL: Sanitize NaN/Inf values BEFORE any conversion
    opens = np.nan_to_num(opens, nan=0.0, posinf=0.0, neginf=0.0)
    highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
    lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)
    closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    if volumes is not None:
        volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize timestamps to Unix seconds using central function
    timestamps = _normalize_timestamps_to_unix(timestamps)

    # Remove duplicates and ensure strict ordering
    unique_ts, unique_indices = np.unique(timestamps, return_index=True)
    if len(unique_ts) < len(timestamps):
        timestamps = unique_ts
        opens = opens[unique_indices]
        highs = highs[unique_indices]
        lows = lows[unique_indices]
        closes = closes[unique_indices]
        if volumes is not None:
            volumes = volumes[unique_indices]

    # Ensure sorted (should already be, but guarantee it)
    sort_indices = np.argsort(timestamps)
    if not np.all(sort_indices == np.arange(len(timestamps))):
        timestamps = timestamps[sort_indices]
        opens = opens[sort_indices]
        highs = highs[sort_indices]
        lows = lows[sort_indices]
        closes = closes[sort_indices]
        if volumes is not None:
            volumes = volumes[sort_indices]

    # Quantize prices - use float64 to avoid int64 overflow with large synthetic prices
    factor = 10 ** price_precision
    
    # Check if values would overflow int64 (max ~9.2e18)
    max_price = max(np.nanmax(opens), np.nanmax(highs), np.nanmax(lows), np.nanmax(closes))
    max_quantized = max_price * factor
    
    if max_quantized > 9e18 or np.isnan(max_quantized) or np.isinf(max_quantized):
        # Use float64 for very large prices (synthetic data with 100+ years)
        opens = np.round(opens * factor).astype(np.float64)
        highs = np.round(highs * factor).astype(np.float64)
        lows = np.round(lows * factor).astype(np.float64)
        closes = np.round(closes * factor).astype(np.float64)
    else:
        opens = np.round(opens * factor).astype(np.int64)
        highs = np.round(highs * factor).astype(np.int64)
        lows = np.round(lows * factor).astype(np.int64)
        closes = np.round(closes * factor).astype(np.int64)

    if volumes is not None:
        volumes = np.round(volumes).astype(np.int64)

    return timestamps, opens, highs, lows, closes, volumes, factor


class StrictAlignmentMapper:
    """
    ZERO-LAG ALIGNMENT ENGINE (v7.0 - VECTORIZED)
    
    Maps indicator values to the authoritative candle timestamp array.
    Guarantees that indicator[i] corresponds EXACTLY to candle[i].
    
    v7.0 OPTIMIZATION:
    - Uses np.searchsorted for O(log n) vectorized alignment instead of O(n) loop
    - ~100x faster for 1M+ candles (5ms vs 500ms)
    
    Architecture:
    1. ts_q (authoritative) = timestamps from candle data after filtering/dedup
    2. indicator_ts = timestamps from indicator calculation source
    3. This class creates a mapping: indicator_ts -> ts_q indices using binary search
    4. Result: indicator values aligned to ts_q with None for gaps
    """

    def __init__(self, authoritative_timestamps: np.ndarray):
        """
        Initialize with the authoritative timestamp array (ts_q from candles).
        
        Args:
            authoritative_timestamps: np.ndarray[np.int64] - Unix seconds from candle data
        """
        self.ts_q = authoritative_timestamps.astype(np.int64)
        self.n = len(authoritative_timestamps)
        # Keep dict for backwards compatibility with count methods
        self.ts_to_idx = {int(ts): i for i, ts in enumerate(authoritative_timestamps)}

    def align(self, indicator_timestamps: np.ndarray, indicator_values: np.ndarray) -> list:
        """
        VECTORIZED alignment using np.searchsorted.
        
        Args:
            indicator_timestamps: np.ndarray[np.int64] - Timestamps from indicator source
            indicator_values: np.ndarray[np.float64] - Indicator values
            
        Returns:
            list[float|None] - Values aligned to ts_q, with None for gaps/NaN
        """
        if self.n == 0:
            return []

        # Ensure timestamps are Unix seconds (int64)
        if not np.issubdtype(indicator_timestamps.dtype, np.integer):
            indicator_timestamps = _normalize_timestamps_to_unix(indicator_timestamps)
        else:
            indicator_timestamps = indicator_timestamps.astype(np.int64)

        # Initialize result array with NaN (will be converted to None)
        aligned = np.full(self.n, np.nan, dtype=np.float64)

        # VECTORIZED: Use searchsorted to find insertion points
        # searchsorted returns the index where each indicator_ts would be inserted
        indices = np.searchsorted(self.ts_q, indicator_timestamps)

        # Create mask for exact matches (indicator_ts exists in ts_q)
        # indices could be == self.n (past end), so clamp first
        indices_clamped = np.clip(indices, 0, self.n - 1)
        exact_match_mask = (self.ts_q[indices_clamped] == indicator_timestamps)

        # Also need to handle indices that are within bounds
        in_bounds_mask = (indices < self.n)
        valid_match_mask = exact_match_mask & in_bounds_mask

        # Create mask for valid values (not NaN/Inf)
        valid_values_mask = np.isfinite(indicator_values)

        # Combined mask: exact timestamp match AND valid value
        final_mask = valid_match_mask & valid_values_mask

        # Assign values at matched positions
        matched_indices = indices_clamped[final_mask]
        matched_values = indicator_values[final_mask]
        aligned[matched_indices] = matched_values

        # Convert to list with None for NaN
        return [None if np.isnan(v) else float(v) for v in aligned]

    def quantize(self, aligned_values: list, precision: int = 4) -> tuple:
        """
        Quantize aligned values for JSON serialization.
        
        Args:
            aligned_values: list[float|None] from align()
            precision: Decimal places for quantization
            
        Returns:
            (quantized_list, factor)
        """
        factor = 10 ** precision
        quantized = [
            int(round(v * factor)) if v is not None else None
            for v in aligned_values
        ]
        return quantized, factor

    def align_quantized(self, indicator_timestamps: np.ndarray, indicator_values: np.ndarray, precision: int = 4) -> tuple:
        """
        OPTIMIZED: Align and quantize in a single pass (avoids intermediate list).
        
        Returns:
            (quantized_list, factor, valid_count)
        """
        if self.n == 0:
            return [], 1, 0

        factor = 10 ** precision

        # Ensure timestamps are Unix seconds (int64)
        if not np.issubdtype(indicator_timestamps.dtype, np.integer):
            indicator_timestamps = _normalize_timestamps_to_unix(indicator_timestamps)
        else:
            indicator_timestamps = indicator_timestamps.astype(np.int64)

        # Initialize with NaN marker
        aligned = np.full(self.n, np.nan, dtype=np.float64)

        # Vectorized searchsorted alignment
        indices = np.searchsorted(self.ts_q, indicator_timestamps)
        indices_clamped = np.clip(indices, 0, self.n - 1)
        exact_match = (self.ts_q[indices_clamped] == indicator_timestamps) & (indices < self.n)
        valid_vals = np.isfinite(indicator_values)
        final_mask = exact_match & valid_vals

        matched_indices = indices_clamped[final_mask]
        matched_values = indicator_values[final_mask]
        aligned[matched_indices] = matched_values

        # Quantize in numpy (faster than list comprehension)
        valid_count = int(np.sum(np.isfinite(aligned)))

        # Convert to quantized list with None for NaN
        quantized = []
        for v in aligned:
            if np.isnan(v):
                quantized.append(None)
            else:
                quantized.append(int(round(v * factor)))

        return quantized, factor, valid_count

    def count_valid(self, aligned_values: list) -> int:
        """Count non-None values in aligned list."""
        return sum(1 for v in aligned_values if v is not None)


def _prepare_indicator_vectorized_aligned(
    candle_timestamps: np.ndarray,
    indicator_values: np.ndarray,
    precision: int = 4
) -> tuple:
    """
    Vectorized indicator preparation with STRICT 1:1 ALIGNMENT.
    
    DEPRECATED: Use StrictAlignmentMapper for new code.
    Kept for backwards compatibility.
    """
    if len(indicator_values) == 0:
        return None, None, None

    # Ensure arrays have same length
    if len(candle_timestamps) != len(indicator_values):
        min_len = min(len(candle_timestamps), len(indicator_values))
        candle_timestamps = candle_timestamps[:min_len]
        indicator_values = indicator_values[:min_len]

    factor = 10 ** precision

    # Convert values: NaN -> None, valid -> quantized int
    values_list = []
    valid_count = 0

    for val in indicator_values:
        if np.isnan(val) or np.isinf(val):
            values_list.append(None)
        else:
            values_list.append(int(np.round(val * factor)))
            valid_count += 1

    if valid_count == 0:
        return None, None, None

    return candle_timestamps, values_list, factor


def _prepare_indicator_vectorized(
    timestamps: np.ndarray,
    values: np.ndarray,
    precision: int = 4
) -> tuple:
    """
    LEGACY: Vectorized indicator preparation (filters NaN - causes desync).
    DEPRECATED: Use StrictAlignmentMapper instead.
    """
    mask = ~np.isnan(values)
    ts_clean = timestamps[mask]
    vals_clean = values[mask]

    if len(vals_clean) == 0:
        return None, None, None

    factor = 10 ** precision
    vals_quantized = np.round(vals_clean * factor).astype(np.int64)

    return ts_clean, vals_quantized, factor


# =============================================================================
# GLOBAL WARM-UP PERIOD (v6.1)
# =============================================================================
# All indicators must "wake up" at the same candle to ensure visual consistency.
# This eliminates the staggered start caused by different indicator periods.
# =============================================================================

def _detect_max_warmup_period(params: Optional[Dict[str, Any]], min_warmup: int = 1) -> int:
    """
    Detect the maximum warm-up period from all Optuna parameters.
    
    Scans for any key ending with '_period', '_length', '_len', '_window'
    and returns the maximum value found.
    
    Args:
        params: Optuna trial parameters dict
        min_warmup: Minimum warmup period (default: 1)
        
    Returns:
        int: Maximum warm-up period (at least min_warmup)
    
    Example:
        params = {'mfi_period': 14, 'atr_period': 20, 'ema_length': 50}
        -> Returns 50 (ema_length is the max)
    """
    if not params:
      return min_warmup

    # Explicit override from strategy (fully modular):
    # if a strategy sets params["__warmup_bars"], respect it.
    override = params.get("__warmup_bars") if isinstance(params, dict) else None
    try:
      if override is not None:
        ov = int(override)
        if ov > 0:
          return max(min_warmup, ov)
    except (ValueError, TypeError):
      pass

    period_suffixes = ('_period', '_length', '_len', '_window')
    max_period = min_warmup

    for key, value in params.items():
        key_lower = key.lower()
        # Check if key ends with a period-related suffix
        if any(key_lower.endswith(suffix) for suffix in period_suffixes):
            try:
                period_val = int(value)
                if period_val > max_period:
                    max_period = period_val
            except (ValueError, TypeError):
                pass

    return max_period


def _apply_warmup_mask(aligned_values: list, warmup_period: int) -> list:
    """
    Apply warm-up mask to aligned indicator values.
    
    Sets the first `warmup_period` values to None, regardless of their original value.
    This ensures all indicators visually start at the same candle.
    
    Args:
        aligned_values: list[float|None] - Values aligned to ts_q
        warmup_period: int - Number of initial values to mask as None
        
    Returns:
        list[float|None] - Values with warmup period masked
    """
    if warmup_period <= 0:
        return aligned_values

    # Create a copy to avoid modifying the original
    masked = aligned_values.copy()

    # Mask the first `warmup_period` values
    for i in range(min(warmup_period, len(masked))):
        masked[i] = None

    return masked


def _extract_indicator_params_from_optuna(params: Optional[Dict[str, Any]], indicator_name: str) -> Dict[str, Any]:
    """Extrae niveles de referencia desde `params` para un indicador.

    Es deliberadamente GENÉRICO y se basa en patrones de nombres comunes.
    La fuente de verdad sigue siendo la estrategia (params), no la gráfica.
    """

    if not params:
        return {}

    result: Dict[str, Any] = {}
    ind_lower = indicator_name.lower()

    # === PERIOD DETECTION (solo para nombre del panel, opcional) ===
    period_patterns = [
        f"{ind_lower}_period", f"period_{ind_lower}", f"{ind_lower}_length",
        f"length_{ind_lower}", f"{ind_lower}_len", f"{ind_lower}_window",
        f"window_{ind_lower}", "period", "length"
    ]
    for key in period_patterns:
        if key in params and params[key] is not None:
            try:
                result['period'] = int(params[key])
                break
            except (ValueError, TypeError):
                pass

    # === HI/LO DETECTION ===
    overbought_patterns = [
      f"{ind_lower}_overbought",
      f"overbought_{ind_lower}",
      f"{ind_lower}_ob",
      f"{ind_lower}_upper",
      f"upper_{ind_lower}",
      f"{ind_lower}_hi",
      f"hi_{ind_lower}",
      f"{ind_lower}_threshold_hi",
    ]
    oversold_patterns = [
      f"{ind_lower}_oversold",
      f"oversold_{ind_lower}",
      f"{ind_lower}_os",
      f"{ind_lower}_lower",
      f"lower_{ind_lower}",
      f"{ind_lower}_lo",
      f"lo_{ind_lower}",
      f"{ind_lower}_threshold_lo",
    ]

    for key in overbought_patterns:
        if key in params and params[key] is not None:
            try:
                result['hi'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    for key in oversold_patterns:
        if key in params and params[key] is not None:
            try:
                result['lo'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    # === ENTRY/EXIT LEVELS (opcionales) ===
    # Entry Long: when indicator reaches this level, enter long
    entry_long_patterns = [
        f"entry_long_{ind_lower}", f"{ind_lower}_entry_long", "entry_long",
        f"long_entry_{ind_lower}", f"{ind_lower}_long_entry", "entry_level_long"
    ]
    entry_short_patterns = [
        f"entry_short_{ind_lower}", f"{ind_lower}_entry_short", "entry_short",
        f"short_entry_{ind_lower}", f"{ind_lower}_short_entry", "entry_level_short"
    ]
    exit_long_patterns = [
        f"exit_long_{ind_lower}", f"{ind_lower}_exit_long", "exit_long",
        f"long_exit_{ind_lower}", f"{ind_lower}_long_exit", "exit_level_long"
    ]
    exit_short_patterns = [
        f"exit_short_{ind_lower}", f"{ind_lower}_exit_short", "exit_short",
        f"short_exit_{ind_lower}", f"{ind_lower}_short_exit", "exit_level_short"
    ]

    for key in entry_long_patterns:
        if key in params and params[key] is not None:
            try:
                result['entry_long'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    for key in entry_short_patterns:
        if key in params and params[key] is not None:
            try:
                result['entry_short'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    for key in exit_long_patterns:
        if key in params and params[key] is not None:
            try:
                result['exit_long'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    for key in exit_short_patterns:
        if key in params and params[key] is not None:
            try:
                result['exit_short'] = float(params[key])
                break
            except (ValueError, TypeError):
                pass

    # mid opcional: si no viene dado, calcularlo si tenemos hi/lo.
    if 'mid' not in result and 'hi' in result and 'lo' in result:
        try:
            result['mid'] = (float(result['hi']) + float(result['lo'])) / 2.0
        except (ValueError, TypeError):
            pass

    return result



def _generate_dynamic_combo(params: Optional[Dict[str, Any]], strategy_name: str = "") -> str:
    """
    Generate a dynamic combo string summarizing key trial parameters.
    
    Format: STRATEGY | P:14 OB:75 OS:25 | EntryL:20 ExitS:80
    
    This allows identifying winning configurations at a glance in YouTube/TikTok shorts.
    """
    if not params:
        return strategy_name or "TRIAL"

    parts = []

    # Strategy name first
    if strategy_name:
        parts.append(strategy_name.upper())

    # Extract key parameters
    period_keys = [k for k in params.keys() if 'period' in k.lower() or 'length' in k.lower()]
    level_keys = [k for k in params.keys() if any(x in k.lower() for x in ['overbought', 'oversold', 'ob', 'os', 'entry', 'exit', 'threshold'])]

    param_parts = []

    # Periods (P:)
    for key in sorted(period_keys)[:2]:  # Max 2 periods
        val = params[key]
        if val is not None:
            # Extract indicator name from key
            ind_name = key.replace('_period', '').replace('period_', '').replace('_length', '').upper()[:3]
            param_parts.append(f"{ind_name}:{int(val)}")

    # Levels (OB/OS/Entry/Exit)
    for key in sorted(level_keys)[:4]:  # Max 4 levels
        val = params[key]
        if val is not None:
            key_lower = key.lower()
            if 'overbought' in key_lower or '_ob' in key_lower:
                param_parts.append(f"OB:{val:.0f}" if isinstance(val, float) else f"OB:{val}")
            elif 'oversold' in key_lower or '_os' in key_lower:
                param_parts.append(f"OS:{val:.0f}" if isinstance(val, float) else f"OS:{val}")
            elif 'entry_long' in key_lower:
                param_parts.append(f"EL:{val:.0f}" if isinstance(val, float) else f"EL:{val}")
            elif 'entry_short' in key_lower:
                param_parts.append(f"ES:{val:.0f}" if isinstance(val, float) else f"ES:{val}")
            elif 'exit_long' in key_lower:
                param_parts.append(f"XL:{val:.0f}" if isinstance(val, float) else f"XL:{val}")
            elif 'exit_short' in key_lower:
                param_parts.append(f"XS:{val:.0f}" if isinstance(val, float) else f"XS:{val}")

    if param_parts:
        parts.append(' '.join(param_parts[:6]))  # Max 6 params for readability

    return ' | '.join(parts) if parts else strategy_name or "TRIAL"


def _detect_indicators_legacy(
  df_cols: set,
  price_range: tuple,
  params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """LEGACY – mantenido solo por compatibilidad interna.

  La detección real ahora es 100% guiada por estrategia y se hace con la
  versión nueva de `_detect_indicators(df: pd.DataFrame, ...)` definida arriba.
  """
  raise RuntimeError(
    "Legacy _detect_indicators(df_cols, ...) ya no está soportado. "
    "Usa la detección strategy-driven con el DataFrame alineado."
  )


# =============================================================================
# DYNAMIC HTML GENERATOR (v7.0 - STREAMING)
# =============================================================================

def _write_html_streaming(
    filepath: str,
    candle_data: dict,
    indicators: dict,
    trades: dict,
    config: dict
) -> None:
    """
    STREAMING HTML GENERATOR (v7.0)
    
    Writes HTML directly to disk in chunks instead of building a giant string.
    This dramatically reduces RAM usage and improves write performance.
    
    Architecture:
    1. Write static HTML header
    2. Stream JSON data directly using orjson (zero-copy from numpy)
    3. Write static JS/CSS footer
    
    Result: Near-zero RAM overhead for 50MB+ chart files.
    """

    activo = str(config.get("activo", ""))
    combo = str(config.get("combo", ""))
    total_trades = int(config.get("total_trades", 0))
    winrate = float(config.get("winrate", 0))
    pnl_neto = float(config.get("pnl_neto", 0))
    pnl_class = "pos" if pnl_neto >= 0 else "neg"
    score = float(config.get("score", 0))

    with open(filepath, "wb") as f:
        # ============ CHUNK 1: HTML Header + CSS ============
        header = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>MODELOX - Dynamic Chart</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
window.addEventListener('DOMContentLoaded', function() {{
  if (typeof LightweightCharts === 'undefined') {{
    document.body.innerHTML = '<div style="color:#ef4444;padding:40px;font-family:system-ui;text-align:center;"><h2>Error: Could not load chart library</h2><p>CDN may be blocked. Check your internet connection.</p></div>';
  }}
}});
</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:#0b1220;font-family:'SF Pro Display',system-ui,-apple-system,sans-serif;overflow:hidden;touch-action:none}}
.c{{display:flex;flex-direction:column;height:100vh;padding:0}}
.h{{display:flex;justify-content:space-between;align-items:center;padding:4px 8px;background:linear-gradient(180deg,rgba(15,23,42,.98) 0%,rgba(15,23,42,.92) 100%);border-radius:0;border-bottom:1px solid rgba(148,163,184,.1)}}
.h .a{{color:#22d3ee;font-size:16px;font-weight:700}}
.h .t{{color:#94a3b8;font-size:11px;font-weight:500}}
.h .info{{display:flex;gap:16px;align-items:center}}
.h .stat{{display:flex;flex-direction:column;align-items:flex-end}}
.h .stat-label{{color:#64748b;font-size:9px;text-transform:uppercase;letter-spacing:.5px}}
.h .stat-val{{color:#e2e8f0;font-size:12px;font-weight:600}}
.h .stat-val.pos{{color:#22c55e}}
.h .stat-val.neg{{color:#ef4444}}
.p{{flex:1;display:flex;flex-direction:column;min-height:0;gap: 0 !important;}}
.m{{position:relative;min-height:280px;margin-bottom:-1px;}}
.sub{{position:relative;min-height:60px;margin-bottom:-1px;}}
.l{{position:absolute;top:6px;left:10px;z-index:100;background:rgba(15,23,42,.92);color:#e2e8f0;font-size:10px;font-weight:700;padding:3px 10px;border-radius:4px;border:1px solid rgba(148,163,184,.1);letter-spacing:.3px;text-transform:uppercase}}
#tt{{position:fixed;display:none;background:linear-gradient(180deg,rgba(15,23,42,.98) 0%,rgba(10,18,32,.99) 100%);border:1px solid rgba(71,85,105,.6);border-radius:10px;padding:14px 18px;color:#e2e8f0;font-size:12px;z-index:999999;pointer-events:none;min-width:220px;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.6)}}
.tt-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(148,163,184,.15)}}
.tt-type{{font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.5px}}
.tt-type.long{{color:#3b82f6}}
.tt-type.short{{color:#a855f7}}
.tt-badge{{padding:2px 8px;border-radius:4px;font-size:9px;font-weight:600;text-transform:uppercase}}
.tt-badge.win{{background:rgba(34,197,94,.2);color:#22c55e}}
.tt-badge.loss{{background:rgba(239,68,68,.2);color:#ef4444}}
.tt-row{{display:flex;justify-content:space-between;align-items:center;margin:6px 0}}
.tt-label{{color:#94a3b8;font-size:11px}}
.tt-val{{font-weight:600;font-size:12px;font-family:'SF Mono',ui-monospace,monospace}}
.tt-val.pos{{color:#22c55e}}
.tt-val.neg{{color:#ef4444}}
.tt-pnl{{margin-top:10px;padding-top:8px;border-top:1px solid rgba(148,163,184,.15);display:flex;justify-content:space-between;align-items:center}}
.tt-pnl-label{{color:#94a3b8;font-size:11px;font-weight:600}}
.tt-pnl-val{{font-size:16px;font-weight:700;font-family:'SF Mono',ui-monospace,monospace}}
#ohlc{{position:absolute;top:6px;right:10px;z-index:100;display:flex;gap:12px;background:rgba(15,23,42,.92);padding:4px 12px;border-radius:4px;border:1px solid rgba(148,163,184,.1);font-size:11px;font-family:'SF Mono',ui-monospace,monospace}}
.ohlc-item{{display:flex;gap:4px}}
.ohlc-label{{color:#64748b}}
.ohlc-val{{font-weight:600}}
.ohlc-val.up{{color:#22c55e}}
.ohlc-val.down{{color:#ef4444}}
.tv-zoom-container{{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);z-index:10001;display:flex;gap:6px}}
.tv-zoom-btn{{width:32px;height:32px;background:rgba(15,23,42,.85);border:1px solid rgba(148,163,184,.25);border-radius:50%;color:#94a3b8;font-size:16px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .15s;backdrop-filter:blur(4px)}}
.tv-zoom-btn:hover{{background:rgba(30,41,59,.95);color:#e2e8f0;border-color:#60a5fa}}
#scaleMarginHandleTop,#scaleMarginHandleBottom{{display:none}}
.smh{{position:absolute;left:0;right:0;height:14px;z-index:10002;cursor:ns-resize;background:transparent;touch-action:none}}
.smh.top{{top:0}}
.smh.bottom{{bottom:0}}
#globalCrosshair{{position:absolute;top:0;bottom:0;width:1px;background:repeating-linear-gradient(to bottom,rgba(255,255,255,.6) 0px,rgba(255,255,255,.6) 2px,transparent 2px,transparent 4px);pointer-events:none;z-index:9999;display:none}}
#globalCrosshairLabel{{position:absolute;bottom:0;transform:translateX(-50%);background:#1e293b;color:#e2e8f0;font-size:10px;padding:2px 6px;border-radius:2px;white-space:nowrap;pointer-events:none;z-index:10000;display:none}}
.p{{position:relative}}
</style>
</head>
<body>
<div class="c">
<div class="h">
<div style="display:flex;align-items:center;gap:16px">
<span class="a">{activo}</span>
<span class="t">{combo}</span>
</div>
<div class="info">
<div class="stat"><span class="stat-label">Trades</span><span class="stat-val">{total_trades}</span></div>
<div class="stat"><span class="stat-label">Win Rate</span><span class="stat-val">{round(winrate, 1)}%</span></div>
<div class="stat"><span class="stat-label">PnL Neto</span><span class="stat-val {pnl_class}">${round(pnl_neto, 2)}</span></div>
<div class="stat"><span class="stat-label">Score</span><span class="stat-val pos">{round(score, 2)}</span></div>
</div>
</div>
<div class="p" id="ct"><div id="globalCrosshair"></div><div id="globalCrosshairLabel"></div></div>
</div>
<div id="tt"></div>

<script>
(function(){{
'use strict';

try {{

const D='''.encode('utf-8')
        f.write(header)

        # ============ CHUNK 2: Candle Data JSON (streaming) ============
        f.write(_dumps_bytes(candle_data))

        # ============ CHUNK 3: Indicators JSON ============
        f.write(b';\nconst I=')
        f.write(_dumps_bytes(indicators))

        # ============ CHUNK 4: Trades JSON ============
        f.write(b';\nconst T=')
        f.write(_dumps_bytes(trades))

        # ============ CHUNK 5: JavaScript Logic ============
        js_logic = _get_chart_js_logic()
        f.write(js_logic.encode('utf-8'))


def _get_chart_js_logic() -> str:
    """Return the JavaScript chart logic as a string (static, cacheable)."""
    return '''

// ============================================================================
// MODELOX BLOOMBERG TERMINAL v8.0 - ULTRA PROFESSIONAL TRADING INTERFACE
// ============================================================================
// Features:
// - Bloomberg-style dark terminal design
// - Interactive statistics sidebar
// - Equity curve panel with drawdown
// - Trade table with sorting/filtering
// - Professional hotkeys (F=fit, H=home, E=end, T=trades, S=stats)
// - Multi-panel synchronized crosshair
// - Draggable panel margins
// - Export capabilities
// ============================================================================

// Validate data loaded correctly
if (!D || !D.t || D.t.length === 0) {
  console.error('No candle data available');
  document.body.innerHTML = '<div style="color:#fbbf24;padding:40px;font-family:system-ui;text-align:center;background:#0a0e17;height:100vh;display:flex;flex-direction:column;justify-content:center;align-items:center;"><h2 style="font-size:24px;margin-bottom:16px;">⚠ NO DATA AVAILABLE</h2><p style="color:#64748b;">No candle data was generated for this trial.</p></div>';
  return;
}

const dq=(v,f)=>v/f;
const ct=document.getElementById('ct');
const charts=[];
let syncingCharts=false;

// === DRAGGABLE SCALE MARGINS (TOP/BOTTOM) ===
// Permite ajustar rightPriceScale.scaleMargins arrastrando arriba/abajo
// en la zona superior/inferior de cada panel.
const _scaleMarginsByChart = new WeakMap();

function _clamp(v, lo, hi){
  return Math.max(lo, Math.min(hi, v));
}

function _getOrInitMargins(ch){
  if(_scaleMarginsByChart.has(ch)) return _scaleMarginsByChart.get(ch);
  const m = { top: 0.1, bottom: 0.1 };
  _scaleMarginsByChart.set(ch, m);
  return m;
}

function _applyMargins(ch, top, bottom){
  // evitar que se coma el área del chart
  top = _clamp(top, 0.0, 0.45);
  bottom = _clamp(bottom, 0.0, 0.45);
  if(top + bottom > 0.9){
    const excess = (top + bottom) - 0.9;
    // reducir proporcionalmente
    const tShare = top / (top + bottom);
    top = _clamp(top - excess * tShare, 0.0, 0.45);
    bottom = _clamp(bottom - excess * (1 - tShare), 0.0, 0.45);
  }
  _scaleMarginsByChart.set(ch, { top, bottom });
  try{
    ch.applyOptions({ rightPriceScale: { scaleMargins: { top, bottom } } });
  }catch(e){
    // ignore
  }
}

function _attachScaleMarginHandles(panelEl, ch){
  if(!panelEl || !ch) return;

  const hTop = document.createElement('div');
  hTop.className = 'smh top';
  const hBot = document.createElement('div');
  hBot.className = 'smh bottom';
  panelEl.appendChild(hTop);
  panelEl.appendChild(hBot);

  const startDrag = (which, ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    const rect = panelEl.getBoundingClientRect();
    const startY = ev.clientY;
    const startMargins = _getOrInitMargins(ch);
    const startTop = startMargins.top;
    const startBottom = startMargins.bottom;

    const onMove = (e) => {
      const dy = e.clientY - startY;
      const h = Math.max(1, rect.height);
      const df = dy / h;
      if(which === 'top'){
        _applyMargins(ch, startTop + df, startBottom);
      }else{
        _applyMargins(ch, startTop, startBottom - df);
      }
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove, { passive: false });
      window.removeEventListener('pointerup', onUp, { passive: false });
      window.removeEventListener('pointercancel', onUp, { passive: false });
    };
    window.addEventListener('pointermove', onMove, { passive: false });
    window.addEventListener('pointerup', onUp, { passive: false });
    window.addEventListener('pointercancel', onUp, { passive: false });
  };

  hTop.addEventListener('pointerdown', (ev)=>startDrag('top', ev), { passive: false });
  hBot.addEventListener('pointerdown', (ev)=>startDrag('bottom', ev), { passive: false });
}

// Bloquear SOLO zoom por pinch (ctrlKey/metaKey + wheel).
// El scroll normal (sin modificadores) se deja pasar para pan horizontal.
if(ct){
  ct.addEventListener('wheel',(e)=>{
    // Solo bloquear si es zoom (ctrl/meta = pinch en trackpad)
    if(e && (e.ctrlKey || e.metaKey)){
      e.preventDefault();
      e.stopPropagation();
    }
    // El scroll normal (sin modificadores) pasa al chart para pan
  },{passive:false});
  ['gesturestart','gesturechange','gestureend'].forEach((ev)=>{
    ct.addEventListener(ev,(e)=>{ if(e) e.preventDefault(); },{passive:false});
  });
}

const baseOpts={
layout:{background:{type:'solid',color:'#0b1220'},textColor:'#94a3b8',fontSize:10,fontFamily:"'SF Pro Display',system-ui"},
grid:{vertLines:{color:'rgba(148,163,184,.04)'},horzLines:{color:'rgba(148,163,184,.04)'}},
crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{visible:false,labelVisible:false},horzLine:{color:'rgba(255,255,255,.6)',width:1,style:LightweightCharts.LineStyle.SparseDotted,labelBackgroundColor:'#1e293b',labelVisible:true}},
timeScale:{borderColor:'rgba(148,163,184,.1)',timeVisible:true,secondsVisible:false,rightOffset:12,barSpacing:4,minBarSpacing:1,fixLeftEdge:false,fixRightEdge:false,lockVisibleTimeRangeOnResize:true,autoScale:true,visible:false},
rightPriceScale:{borderColor:'rgba(148,163,184,.1)',scaleMargins:{top:.1,bottom:.1},autoScale:true,alignLabels:true,borderVisible:true,entireTextOnly:false},
// ZOOM: desactivado completamente (mouseWheel, pinch, axisDrag todo false).
handleScale:{axisPressedMouseMove:false,mouseWheel:false,pinch:false},
// SCROLL/PAN: mouseWheel=true permite desplazarse horizontalmente con la rueda SIN zoom.
handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},
kineticScroll:{touch:true,mouse:true},
localization:{
  // Force UTC display for consistent time across all browsers
  timeFormatter:(ts)=>{
    const d=new Date(ts*1000);
    const yr=d.getUTCFullYear();
    const mo=String(d.getUTCMonth()+1).padStart(2,'0');
    const dy=String(d.getUTCDate()).padStart(2,'0');
    const hr=String(d.getUTCHours()).padStart(2,'0');
    const mn=String(d.getUTCMinutes()).padStart(2,'0');
    return mo+'/'+dy+'/'+yr+' '+hr+':'+mn;
  },
  dateFormat:'yyyy-MM-dd'
}
};

// === DYNAMIC PANEL CREATION ===
function mkPanel(id,lbl,isMain,heightPct){
try{
const p=document.createElement('div');
p.className=isMain?'m':'sub';
p.id=id;
p.style.flex=isMain?'6':'1.2';
const l=document.createElement('div');
l.className='l';
l.textContent=lbl;
p.appendChild(l);

if(isMain){
  const ohlc=document.createElement('div');
  ohlc.id='ohlc';
  ohlc.innerHTML='<div class="ohlc-item"><span class="ohlc-label">T</span><span class="ohlc-val" id="tv" style="color:#22d3ee">-</span></div><div class="ohlc-item"><span class="ohlc-label">O</span><span class="ohlc-val" id="ov">-</span></div><div class="ohlc-item"><span class="ohlc-label">H</span><span class="ohlc-val" id="hv">-</span></div><div class="ohlc-item"><span class="ohlc-label">L</span><span class="ohlc-val" id="lv">-</span></div><div class="ohlc-item"><span class="ohlc-label">C</span><span class="ohlc-val" id="cv">-</span></div>';
  p.appendChild(ohlc);
  const zoomDiv=document.createElement('div');
  zoomDiv.className='tv-zoom-container';
  zoomDiv.innerHTML='<button class="tv-zoom-btn" id="zoomIn" title="Zoom In">+</button><button class="tv-zoom-btn" id="zoomOut" title="Zoom Out">-</button>';
  p.appendChild(zoomDiv);
}
ct.appendChild(p);

const opts={...baseOpts,width:p.clientWidth,height:p.clientHeight};
// Don't set timeScale visible here - will be set in SINGLE TIMESCALE section
const ch=LightweightCharts.createChart(p,opts);

// init + attach draggable scale margins
_getOrInitMargins(ch);
_attachScaleMarginHandles(p, ch);

charts.push({ch,p,id,label:lbl});
return ch;
}catch(e){
console.error('Failed to create panel:',lbl,e);
return null;
}
}

// === CALCULATE PANEL HEIGHTS ===
const numSubPanels=(I.sub_panels?I.sub_panels.length:0)+(D.vol&&D.vol.length>0?1:0);
const mainFlex=6;
const subFlex=numSubPanels>0?1.2:0;

// === MAIN PRICE CHART (Marker Stability Optimized) ===
try {
  const mc=mkPanel('mc','PRECIO',true);
  if(!mc)throw new Error('Main chart creation failed');
  
  // Configure candlestick series with options that prevent marker recalculation during scroll/zoom
  const cs=mc.addCandlestickSeries({
    upColor:'#22c55e',
    downColor:'#ef4444',
    borderUpColor:'#16a34a',
    borderDownColor:'#dc2626',
    wickUpColor:'#22c55e',
    wickDownColor:'#ef4444',
    priceFormat:{type:'price',precision:2,minMove:0.01},
    // Disable dynamic updates that trigger price scale recalculation
    lastValueVisible:false,
    priceLineVisible:false
  });

  const f=D.f;
  const cData=[];
  
  // Build candle timestamp Set for O(1) marker validation
  const candleTimeSet=new Set();
  
  // Store main series for crosshair sync
  if(charts.length>0) charts[0].series = cs;
  
  if(D.t && D.t.length>0){
    for(let i=0;i<D.t.length;i++){
      const t=D.t[i];
      cData.push({time:t,open:dq(D.o[i],f),high:dq(D.h[i],f),low:dq(D.l[i],f),close:dq(D.c[i],f)});
      candleTimeSet.add(t);
    }
    cs.setData(cData);

    // Stable trade markers on candle chart.
    // COMBINE entry (m) + exit (xm) markers in a single array for the candle series.
    // This avoids issues with duplicate timestamps in separate line series.
    /* try{
      const allMarkers=[];
      // Add entry markers
      if(T.m && T.m.length>0){
        T.m.forEach(m=>allMarkers.push({
          time:m.time,
          position:m.position||'inBar',
          color:m.color,
          shape:m.shape||'circle',
          text:m.text||'',
          size:m.size||2
        }));
      }
      // Add exit markers (use 'aboveBar' position to distinguish from entries)
      if(T.xm && T.xm.length>0){
        T.xm.forEach(m=>allMarkers.push({
          time:m.time,
          position:'aboveBar',
          color:m.color||'#fbbf24',
          shape:'circle',
          text:'',
          size:1
        }));
      }
      if(allMarkers.length>0){
        // Sort by time to ensure proper rendering
        allMarkers.sort((a,b)=>a.time-b.time);
        cs.setMarkers(allMarkers);
        mc.timeScale().subscribeVisibleTimeRangeChange(()=>{
          cs.setMarkers(allMarkers);
        });
      }
    }catch(e){console.warn('Markers error:',e);} */

    // Entry points at real price: colored dots only (NO connecting lines)
    try{
      if(T.ee && T.ee.length>0){
        const ens=mc.addLineSeries({
          color:'rgba(0,0,0,0)',
          lineWidth:1,
          priceLineVisible:false,
          lastValueVisible:false,
          crosshairMarkerVisible:false
        });
        ens.setData(T.ee);

        if(T.em && T.em.length>0){
          const entryMarkers=T.em.map(m=>({
            time:m.time,
            position:m.position||'inBar',
            color:m.color,
            shape:m.shape||'circle',
            text:m.text||'',
            size:m.size||2
          }));
          ens.setMarkers(entryMarkers);
          mc.timeScale().subscribeVisibleTimeRangeChange(()=>{
            ens.setMarkers(entryMarkers);
          });
        }
      }
    }catch(e){console.warn('Entry points error:',e);}    

    // Exit points at real price: white dots only (NO connecting lines)
    try{
      if(T.xe && T.xe.length>0){
        // Use an invisible line series to anchor markers at the exact exit price.
        // The series line is fully transparent so no diagonals can appear.
        const es=mc.addLineSeries({
          color:'rgba(0,0,0,0)',
          lineWidth:1,
          priceLineVisible:false,
          lastValueVisible:false,
          crosshairMarkerVisible:false
        });
        es.setData(T.xe);

        if(T.xm && T.xm.length>0){
          const exitMarkers=T.xm.map(m=>({
            time:m.time,
            position:m.position||'inBar',
            color:m.color||'#ffffff',
            shape:m.shape||'circle',
            text:m.text||'',
            size:m.size||2
          }));
          es.setMarkers(exitMarkers);
          mc.timeScale().subscribeVisibleTimeRangeChange(()=>{
            es.setMarkers(exitMarkers);
          });
        }
      }
    }catch(e){console.warn('Exit points error:',e);}
  }

  // === OVERLAY INDICATORS (on main chart) - NULL-AWARE ===
  if(I.overlays && Array.isArray(I.overlays)){
    I.overlays.forEach(ov=>{
      try{
        if(ov.t&&ov.t.length>0&&ov.v&&ov.v.length>0){
          const ls=mc.addLineSeries({color:ov.color||'#fbbf24',lineWidth:1.5,priceLineVisible:false,lastValueVisible:true,crosshairMarkerVisible:false,lineStyle:0});
          // STRICT ALIGNMENT: Pass null values to library to render gaps
          const ovData = [];
          for (let i = 0; i < ov.t.length; i++) {
            const val = ov.v[i] !== null ? dq(ov.v[i], ov.f) : null;
            ovData.push({ time: ov.t[i], value: val });
          }
          if (ovData.length > 0) ls.setData(ovData);
        }
      }catch(e){console.warn('Overlay error:',e);}
    });
  }
} catch(e) {
  console.error('CRITICAL: Main chart failed:',e);
}

// === VOLUME PANEL ===
try {
  if(D.vol&&D.vol.length>0&&D.t&&D.t.length>0){
    const vc=mkPanel('vc','VOLUMEN',false);
    if(vc){
      vc.priceScale('right').applyOptions({autoScale:true});
      const vs=vc.addHistogramSeries({priceFormat:{type:'volume'},priceLineVisible:false,lastValueVisible:false});
      const vData=D.t.map((t,i)=>({time:t,value:D.vol[i],color:D.c[i]>=D.o[i]?'rgba(34,197,94,.5)':'rgba(239,68,68,.5)'}));
      vs.setData(vData);
      // Store series for crosshair sync
      const volChartIdx = charts.findIndex(c => c.id === 'vc');
      if(volChartIdx >= 0) charts[volChartIdx].series = vs;
    }
  }
} catch(e) {
  console.warn('Volume panel error:',e);
}

// === DYNAMIC SUB-PANELS (auto-generated from detected indicators) ===
if(I.sub_panels && Array.isArray(I.sub_panels)){
  I.sub_panels.forEach((panel,idx)=>{
    try{
      if(!panel.data||!panel.data.t||panel.data.t.length===0)return;
      
      const panelId='sp_'+idx;
      const panelLabel=panel.name.toUpperCase();
      const pc=mkPanel(panelId,panelLabel,false);
      if(!pc)return;
      
      pc.priceScale('right').applyOptions({scaleMargins:{top:.1,bottom:.1},autoScale:true});
      
      // Histogram or Line based on indicator type
      // Build timestamp set for this panel for marker validation
      // Use candle timestamps as reference since indicators are now strictly aligned
      const panelTimeSet=new Set(D.t);
      let mainSeries=null;
      
      // STRICT ALIGNMENT: Pass null values to library to render gaps
      const seriesData = [];
      for (let i = 0; i < panel.data.t.length; i++) {
        const val = panel.data.v[i] !== null ? dq(panel.data.v[i], panel.data.f) : null;
        seriesData.push({ time: panel.data.t[i], value: val });
      }
      if (seriesData.length === 0) return;
      
      if(panel.type==='histogram'){
        const hs=pc.addHistogramSeries({priceLineVisible:false,lastValueVisible:false});
        const hData=seriesData.map(d=>({
          time:d.time,
          value:d.value,
          color:d.value>=0?'rgba(34,197,94,.7)':'rgba(239,68,68,.7)'
        }));
        hs.setData(hData);
        mainSeries=hs;
      }else{
        const ls=pc.addLineSeries({color:panel.color||'#60a5fa',lineWidth:2,priceLineVisible:false,lastValueVisible:true});
        ls.setData(seriesData);
        mainSeries=ls;
        
        // Reference lines for bounded oscillators (DYNAMIC FROM OPTUNA)
        // Use seriesData timestamps for reference lines (now includes all points)
        if(panel.bounds){
          const b=panel.bounds;
          // Overbought line (red dashed)
          if(b.hi!==undefined){
            const hiLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            hiLine.setData(seriesData.map(d=>({time:d.time,value:b.hi})));
          }
          // Oversold line (green dashed)
          if(b.lo!==undefined){
            const loLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            loLine.setData(seriesData.map(d=>({time:d.time,value:b.lo})));
          }
          // Midline (gray dotted)
          if(b.mid!==undefined){
            const midLine=pc.addLineSeries({color:'rgba(148,163,184,.3)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            midLine.setData(seriesData.map(d=>({time:d.time,value:b.mid})));
          }
          // === STRATEGY ENTRY/EXIT LEVELS (Pendulum Visualization) ===
          // Entry Long level (cyan solid)
          if(b.entry_long!==undefined){
            const entryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            entryLongLine.setData(seriesData.map(d=>({time:d.time,value:b.entry_long})));
          }
          // Entry Short level (magenta solid)
          if(b.entry_short!==undefined){
            const entryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            entryShortLine.setData(seriesData.map(d=>({time:d.time,value:b.entry_short})));
          }
          // Exit Long level (orange dotted)
          if(b.exit_long!==undefined){
            const exitLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            exitLongLine.setData(seriesData.map(d=>({time:d.time,value:b.exit_long})));
          }
          // Exit Short level (amber dotted)
          if(b.exit_short!==undefined){
            const exitShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            exitShortLine.setData(seriesData.map(d=>({time:d.time,value:b.exit_short})));
          }
          // === DPO SYMMETRIC TRIGGER LINES ===
          // DPO Long entry level (cyan solid)
          if(b.dpo_long!==undefined){
            const dpoLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoLongLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_long})));
          }
          // DPO Short entry level (magenta solid)
          if(b.dpo_short!==undefined){
            const dpoShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoShortLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_short})));
          }
          // DPO Long exit level (orange dashed)
          if(b.dpo_exit_long!==undefined){
            const dpoExitLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoExitLongLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_exit_long})));
          }
          // DPO Short exit level (amber dashed)
          if(b.dpo_exit_short!==undefined){
            const dpoExitShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoExitShortLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_exit_short})));
          }
          // === ADX THRESHOLD LINE ===
          // ADX minimum threshold (yellow solid)
          if(b.adx_threshold!==undefined){
            const adxThreshLine=pc.addLineSeries({color:'rgba(250,204,21,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            adxThreshLine.setData(seriesData.map(d=>({time:d.time,value:b.adx_threshold})));
          }
          // === RSI ENTRY LEVELS ===
          // RSI LONG entry level (cyan solid)
          if(b.rsi_long!==undefined){
            const rsiLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            rsiLongLine.setData(seriesData.map(d=>({time:d.time,value:b.rsi_long})));
          }
          // RSI SHORT entry level (magenta solid)
          if(b.rsi_short!==undefined){
            const rsiShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            rsiShortLine.setData(seriesData.map(d=>({time:d.time,value:b.rsi_short})));
          }
          // === DPO CYCLE ZONE LEVELS (Strategy 6) ===
          // DPO RSA - Upper High extreme (red solid) - EUPHORIA zone above
          if(b.dpo_rsa!==undefined){
            const dpoRsaLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRsaLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_rsa})));
          }
          // DPO RSM - Upper Mid zone (orange dashed) - DISTRIBUTION zone
          if(b.dpo_rsm!==undefined){
            const dpoRsmLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRsmLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_rsm})));
          }
          // DPO ZERO - Neutral line (white/gray dashed)
          if(b.dpo_zero!==undefined){
            const dpoZeroLine=pc.addLineSeries({color:'rgba(148,163,184,.6)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoZeroLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_zero})));
          }
          // DPO RIM - Lower Mid zone (cyan dashed) - ACCUMULATION zone
          if(b.dpo_rim!==undefined){
            const dpoRimLine=pc.addLineSeries({color:'rgba(34,211,238,.7)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRimLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_rim})));
          }
          // DPO RIB - Lower Low extreme (green solid) - PANIC zone below
          if(b.dpo_rib!==undefined){
            const dpoRibLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            dpoRibLine.setData(seriesData.map(d=>({time:d.time,value:b.dpo_rib})));
          }
          // === MFI THRESHOLD ZONE LEVELS (Strategy 6) ===
          // MFI High zone threshold (magenta solid) - OVERBOUGHT zone above
          if(b.mfi_high!==undefined){
            const mfiHighLine=pc.addLineSeries({color:'rgba(236,72,153,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiHighLine.setData(seriesData.map(d=>({time:d.time,value:b.mfi_high})));
          }
          // MFI Mid equilibrium (white/gray dashed) - 50 line
          if(b.mfi_mid!==undefined){
            const mfiMidLine=pc.addLineSeries({color:'rgba(148,163,184,.6)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiMidLine.setData(seriesData.map(d=>({time:d.time,value:b.mfi_mid})));
          }
          // MFI Low zone threshold (cyan solid) - OVERSOLD zone below
          if(b.mfi_low!==undefined){
            const mfiLowLine=pc.addLineSeries({color:'rgba(34,211,238,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            mfiLowLine.setData(seriesData.map(d=>({time:d.time,value:b.mfi_low})));
          }
          // === Z-SCORE RANGE LEVELS (Strategy 13 v2 Mean Reversion) ===
          // Z-Score LONG range (negative values) - green band
          if(b.z_long_min!==undefined){
            const zLongMinLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMinLine.setData(seriesData.map(d=>({time:d.time,value:b.z_long_min})));
          }
          if(b.z_long_max!==undefined){
            const zLongMaxLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMaxLine.setData(seriesData.map(d=>({time:d.time,value:b.z_long_max})));
          }
          // Z-Score SHORT range (positive values) - red band
          if(b.z_short_min!==undefined){
            const zShortMinLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMinLine.setData(seriesData.map(d=>({time:d.time,value:b.z_short_min})));
          }
          if(b.z_short_max!==undefined){
            const zShortMaxLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMaxLine.setData(seriesData.map(d=>({time:d.time,value:b.z_short_max})));
          }
          // Z-Score entry/TP levels (legacy v1 support)
          if(b.z_entry_long!==undefined){
            const zEntryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryLongLine.setData(seriesData.map(d=>({time:d.time,value:b.z_entry_long})));
          }
          if(b.z_entry_short!==undefined){
            const zEntryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryShortLine.setData(seriesData.map(d=>({time:d.time,value:b.z_entry_short})));
          }
          if(b.z_tp_long!==undefined){
            const zTpLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpLongLine.setData(seriesData.map(d=>({time:d.time,value:b.z_tp_long})));
          }
          if(b.z_tp_short!==undefined){
            const zTpShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpShortLine.setData(seriesData.map(d=>({time:d.time,value:b.z_tp_short})));
          }
          // === Z-SCORE RANGE LEVELS (Strategy 13 v2 Mean Reversion) ===
          // Z-Score LONG range (negative values) - green band
          if(b.z_long_min!==undefined){
            const zLongMinLine=pc.addLineSeries({color:'rgba(34,197,94,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMinLine.setData(seriesData.map(d=>({time:d.time,value:b.z_long_min})));
          }
          if(b.z_long_max!==undefined){
            const zLongMaxLine=pc.addLineSeries({color:'rgba(34,197,94,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zLongMaxLine.setData(seriesData.map(d=>({time:d.time,value:b.z_long_max})));
          }
          // Z-Score SHORT range (positive values) - red band
          if(b.z_short_min!==undefined){
            const zShortMinLine=pc.addLineSeries({color:'rgba(239,68,68,.6)',lineWidth:1.5,lineStyle:2,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMinLine.setData(seriesData.map(d=>({time:d.time,value:b.z_short_min})));
          }
          if(b.z_short_max!==undefined){
            const zShortMaxLine=pc.addLineSeries({color:'rgba(239,68,68,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zShortMaxLine.setData(seriesData.map(d=>({time:d.time,value:b.z_short_max})));
          }
          // Z-Score entry/TP levels (legacy v1 support)
          if(b.z_entry_long!==undefined){
            const zEntryLongLine=pc.addLineSeries({color:'rgba(34,211,238,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryLongLine.setData(seriesData.map(d=>({time:d.time,value:b.z_entry_long})));
          }
          if(b.z_entry_short!==undefined){
            const zEntryShortLine=pc.addLineSeries({color:'rgba(236,72,153,.8)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zEntryShortLine.setData(seriesData.map(d=>({time:d.time,value:b.z_entry_short})));
          }
          if(b.z_tp_long!==undefined){
            const zTpLongLine=pc.addLineSeries({color:'rgba(251,146,60,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpLongLine.setData(seriesData.map(d=>({time:d.time,value:b.z_tp_long})));
          }
          if(b.z_tp_short!==undefined){
            const zTpShortLine=pc.addLineSeries({color:'rgba(251,191,36,.7)',lineWidth:1.5,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            zTpShortLine.setData(seriesData.map(d=>({time:d.time,value:b.z_tp_short})));
          }
          // === EFFICIENCY RATIO THRESHOLD (Strategy 13 ER) ===
          if(b.er_threshold!==undefined){
            const erThreshLine=pc.addLineSeries({color:'rgba(250,204,21,.9)',lineWidth:2,lineStyle:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
            erThreshLine.setData(seriesData.map(d=>({time:d.time,value:b.er_threshold})));
          }
        }
        
        // Zero line for unbounded oscillators (like MACD, zscore)
        if(panel.zero_line){
          const zl=pc.addLineSeries({color:'rgba(148,163,184,.3)',lineWidth:1,lineStyle:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
          zl.setData(seriesData.map(d=>({time:d.time,value:0})));
        }
      }
      
      // Sub-panels ya muestran la información del precio/indicador;
      // REPINTADO ROBUSTO DE MARCADORES EN LOS SUB-PANELES
      // Dibujamos los mismos marcadores de trade también sobre el indicador
      // principal del panel para que nunca desaparezcan al hacer scroll.
      if(mainSeries && T.m && T.m.length>0){
        try{
          const panelMarkers=T.m.map(m=>({
            time:m.time,
            position:m.position||'inBar',
            color:m.color,
            shape:m.shape||'circle',
            text:m.text||'',
            size:m.size||2
          }));
          mainSeries.setMarkers(panelMarkers);
          pc.timeScale().subscribeVisibleTimeRangeChange(()=>{
            mainSeries.setMarkers(panelMarkers);
          });
        }catch(e){console.warn('Sub-panel markers error:',panel.name,e);}
      }
      
      // Store main series for crosshair sync
      const subChartIdx = charts.findIndex(c => c.id === panelId);
      if(subChartIdx >= 0 && mainSeries) charts[subChartIdx].series = mainSeries;
    }catch(e){
      console.warn('Sub-panel error:',panel.name,e);
    }
  });
}

// === SINGLE TIMESCALE: Only LAST panel shows time axis ===
try {
  if(charts.length>1){
    // Hide time axis on all panels except the last one
    for(let i=0;i<charts.length-1;i++){
      if(charts[i].ch)charts[i].ch.timeScale().applyOptions({visible:false});
    }
    // Show time axis only on the last (bottom) panel
    charts[charts.length-1].ch.timeScale().applyOptions({visible:true});
  }else if(charts.length===1){
    charts[0].ch.timeScale().applyOptions({visible:true});
  }
} catch(e) {
  console.warn('TimeScale config error:',e);
}

// === CHART SYNCHRONIZATION ===
// Sync by TIME range (not logical range).
// Logical range breaks when indicator panels have gaps (null-filtered points),
// producing visible marker desync between panels.
try {
  if(charts.length>1){
    const masterTS=charts[0].ch.timeScale();
    charts.forEach(({ch},idx)=>{
      try{
        if(idx===0||!ch)return;
        const slaveTS=ch.timeScale();
        masterTS.subscribeVisibleTimeRangeChange(range=>{
          try{
            if(syncingCharts||!range)return;
            syncingCharts=true;
            slaveTS.setVisibleRange(range);
            syncingCharts=false;
          }catch(e){syncingCharts=false;}
        });
        slaveTS.subscribeVisibleTimeRangeChange(range=>{
          try{
            if(syncingCharts||!range)return;
            syncingCharts=true;
            masterTS.setVisibleRange(range);
            syncingCharts=false;
          }catch(e){syncingCharts=false;}
        });
      }catch(e){console.warn('Sync error:',e);}
    });
  }
} catch(e) {
  console.warn('Sync setup error:',e);
}

// === GLOBAL CROSSHAIR LINE (SPANS ALL PANELS) ===
try {
  const globalLine = document.getElementById('globalCrosshair');
  const globalLabel = document.getElementById('globalCrosshairLabel');
  const container = document.getElementById('ct');
  let lastTime = null;
  
  // Format time for label
  const formatTimeLabel = (ts) => {
    try {
      const d = new Date(ts * 1000);
      const yr = d.getUTCFullYear();
      const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
      const dy = String(d.getUTCDate()).padStart(2, '0');
      const hr = String(d.getUTCHours()).padStart(2, '0');
      const mn = String(d.getUTCMinutes()).padStart(2, '0');
      return mo + '/' + dy + '/' + yr + ' ' + hr + ':' + mn;
    } catch (e) { return ''; }
  };
  
  // Update global crosshair position
  const updateGlobalCrosshair = (param, sourceChart) => {
    if (!param || !param.point || param.point.x === undefined) {
      globalLine.style.display = 'none';
      globalLabel.style.display = 'none';
      lastTime = null;
      return;
    }
    
    // All panels have the same width and are vertically stacked
    // So param.point.x is already the correct X position within any panel
    const xPos = param.point.x;
    
    // Position global line
    globalLine.style.left = xPos + 'px';
    globalLine.style.display = 'block';
    
    // Position and update label
    if (param.time) {
      lastTime = param.time;
      globalLabel.textContent = formatTimeLabel(param.time);
      globalLabel.style.left = xPos + 'px';
      globalLabel.style.display = 'block';
    }
    
    // Sync horizontal crosshair on other charts
    charts.forEach((target) => {
      if (target.ch === sourceChart || !target.ch) return;
      try {
        if (param.time) {
          // Get first series from target chart
          const series = target.series || (target.ch.getSeries && target.ch.getSeries()[0]);
          if (series) {
            target.ch.setCrosshairPosition(0, param.time, series);
          }
        } else {
          target.ch.clearCrosshairPosition();
        }
      } catch (e) {}
    });
  };
  
  // Subscribe to crosshair moves on all charts
  charts.forEach((c) => {
    if (c.ch) {
      c.ch.subscribeCrosshairMove((param) => {
        updateGlobalCrosshair(param, c.ch);
      });
    }
  });
  
  // Hide crosshair when mouse leaves container
  if (container) {
    container.addEventListener('mouseleave', () => {
      globalLine.style.display = 'none';
      globalLabel.style.display = 'none';
      charts.forEach((c) => {
        if (c.ch) c.ch.clearCrosshairPosition();
      });
    });
  }
} catch (e) {
  console.warn('Global crosshair setup error:', e);
}

// === TRADE MAP FOR TOOLTIPS ===
// Map entry AND exit timestamps to trade info
const globalTradeMap={};
const tradeTimesList=[];  // Sorted list for proximity search
if(T.i&&Array.isArray(T.i)){
  T.i.forEach(t=>{
    if(t&&t.time!==undefined){
      globalTradeMap[t.time]=t;
      tradeTimesList.push(t.time);
    }
  });
  tradeTimesList.sort((a,b)=>a-b);
}

// Find nearest trade within tolerance (60 seconds for 1m candles)
const findNearestTrade=(ts,tolerance=60)=>{
  if(tradeTimesList.length===0)return null;
  // Binary search for closest
  let lo=0,hi=tradeTimesList.length-1;
  while(lo<hi){
    const mid=Math.floor((lo+hi)/2);
    if(tradeTimesList[mid]<ts)lo=mid+1;
    else hi=mid;
  }
  // Check lo and lo-1 for closest
  let closest=tradeTimesList[lo];
  if(lo>0&&Math.abs(tradeTimesList[lo-1]-ts)<Math.abs(closest-ts)){
    closest=tradeTimesList[lo-1];
  }
  if(Math.abs(closest-ts)<=tolerance){
    return globalTradeMap[closest];
  }
  return null;
};

// === TOOLTIP SYSTEM (Unified across all panels) ===
try {
  const tt=document.getElementById('tt');
  const candleMap={};
  if(D.t&&D.t.length>0){
    const f=D.f||100;
    for(let i=0;i<D.t.length;i++){
      candleMap[D.t[i]]={time:D.t[i],open:D.o[i]/f,high:D.h[i]/f,low:D.l[i]/f,close:D.c[i]/f};
    }
  }
  
  // Format Unix timestamp to readable date (UTC to avoid timezone confusion)
  const formatTime=(ts)=>{
    try{
      const d=new Date(ts*1000);
      // Use UTC methods to prevent browser timezone conversion
      const yr=d.getUTCFullYear();
      const mo=String(d.getUTCMonth()+1).padStart(2,'0');
      const dy=String(d.getUTCDate()).padStart(2,'0');
      const hr=String(d.getUTCHours()).padStart(2,'0');
      const mn=String(d.getUTCMinutes()).padStart(2,'0');
      return mo+'/'+dy+'/'+yr+' '+hr+':'+mn+' UTC';
    }catch(e){return '-';}
  };

  // Basic HTML escaping (trade fields may contain arbitrary strings)
  const escapeHtml=(s)=>{
    const str=String(s);
    return str.replace(/[&<>"']/g,(m)=>({
      '&':'&amp;',
      '<':'&lt;',
      '>':'&gt;',
      '"':'&quot;',
      "'":'&#39;'
    }[m]||m));
  };
  
  // Handle crosshair move on ALL panels to update unified tooltip
  const handleCrosshair=(param)=>{
    try{
      if(!param.time){
        if(tt)tt.style.display='none';
        return;
      }
      
      // Update time display in header
      const tvEl=document.getElementById('tv');
      if(tvEl)tvEl.textContent=formatTime(param.time);
      
      // Check for trade at this time (exact match or proximity search)
      let tr=globalTradeMap[param.time];
      if(!tr){
        // Try proximity search for nearby trades
        tr=findNearestTrade(param.time,120);  // 2 minute tolerance
      }
      if(tr&&tr.pnl!==undefined&&param.point){
        const isWin=tr.pnl>=0;
        const pnlSign=isWin?'+':'';
        const pnlCls=isWin?'pos':'neg';
        const typeCls=tr.type==='LONG'?'long':'short';
        const badgeCls=isWin?'win':'loss';

        const exitTypeRow=(tr.xs!==undefined&&tr.xs!==null&&String(tr.xs).length>0)
          ? '<div class="tt-row"><span class="tt-label">ExitType</span><span class="tt-val">'+escapeHtml(tr.xs)+'</span></div>'
          : '';
        
        tt.innerHTML='<div class="tt-header"><span class="tt-type '+typeCls+'">'+tr.type+'</span><span class="tt-badge '+badgeCls+'">'+(isWin?'WIN':'LOSS')+'</span></div><div class="tt-row"><span class="tt-label">Entry</span><span class="tt-val">'+(tr.ep?tr.ep.toFixed(2):'-')+'</span></div><div class="tt-row"><span class="tt-label">Exit</span><span class="tt-val">'+(tr.xp?tr.xp.toFixed(2):'-')+'</span></div>'+exitTypeRow+'<div class="tt-row"><span class="tt-label">Qty</span><span class="tt-val">'+(tr.qty?tr.qty.toFixed(4):'-')+'</span></div><div class="tt-row"><span class="tt-label">Comm</span><span class="tt-val">$'+(tr.comm?tr.comm.toFixed(2):'0')+'</span></div><div class="tt-pnl"><span class="tt-pnl-label">PnL Neto</span><span class="tt-pnl-val '+pnlCls+'">'+pnlSign+'$'+(tr.pnl?tr.pnl.toFixed(2):'0')+'</span></div>';
        tt.style.display='block';
        const maxX=window.innerWidth-260;
        const maxY=window.innerHeight-200;
        tt.style.left=Math.min(param.point.x+15,maxX)+'px';
        tt.style.top=Math.min(param.point.y+15,maxY)+'px';
      }else{
        tt.style.display='none';
        // Update OHLC values from candle data
        const candle=candleMap[param.time];
        if(candle){
          const isUp=candle.close>=candle.open;
          const cls=isUp?'up':'down';
          ['ov','hv','lv','cv'].forEach((id,i)=>{
            const el=document.getElementById(id);
            if(el){
              const vals=[candle.open,candle.high,candle.low,candle.close];
              el.textContent=vals[i].toFixed(2);
              el.className='ohlc-val '+cls;
            }
          });
        }
      }
    }catch(e){console.warn('Tooltip error:',e);}
  };
  
  // Subscribe to ALL charts for unified tooltip
  charts.forEach(({ch})=>{
    if(ch)ch.subscribeCrosshairMove(handleCrosshair);
  });
} catch(e) {
  console.warn('Tooltip setup error:',e);
}

// === RESIZE OBSERVER ===
try {
  const ro=new ResizeObserver(()=>{
    charts.forEach(({ch,p})=>{
      if(ch&&p)ch.applyOptions({width:p.clientWidth,height:p.clientHeight});
    });
  });
  charts.forEach(({p})=>{if(p)ro.observe(p);});
} catch(e) {
  console.warn('Resize observer error:',e);
}

// === AUTO-FIT & KEYBOARD SHORTCUTS ===
try {
  setTimeout(()=>{
    // Vista inicial: mostrar desde el PRINCIPIO del dataset.
    if(D.t && D.t.length>0){
      const HOME_BARS=500;
      const toIdx=Math.min(HOME_BARS, D.t.length-1);
      const homeRange={from:D.t[0], to:D.t[toIdx]};
      charts.forEach(({ch})=>{
        if(ch){
          const ts=ch.timeScale();
          ts.setVisibleRange(homeRange);
        }
      });
    }
  },150);
  
  document.addEventListener('keydown',e=>{
    if(e.key==='f'||e.key==='F')charts.forEach(({ch})=>{if(ch)ch.timeScale().fitContent();});
    if(e.key==='r'||e.key==='R')charts.forEach(({ch})=>{if(ch)ch.timeScale().resetTimeScale();});
    // Press 'h' for home (beginning)
    if((e.key==='h'||e.key==='H') && D.t && D.t.length>0){
      const HOME_BARS=500;
      const toIdx=Math.min(HOME_BARS, D.t.length-1);
      const homeRange={from:D.t[0], to:D.t[toIdx]};
      charts.forEach(({ch})=>{if(ch)ch.timeScale().setVisibleRange(homeRange);});
    }
    // Press 'e' for end
    if(e.key==='e'||e.key==='E')charts.forEach(({ch})=>{if(ch)ch.timeScale().scrollToRealTime();});
  });
} catch(e) {
  console.warn('Shortcuts error:',e);
}

// === ZOOM CONTROLS ===
try {
  const zoomInBtn=document.getElementById('zoomIn');
  const zoomOutBtn=document.getElementById('zoomOut');
  
  if(zoomInBtn){
    zoomInBtn.addEventListener('click',()=>{
      charts.forEach(({ch})=>{
        if(ch){
          const ts=ch.timeScale();
          const spacing=(ts.options().barSpacing||8)+2;
          ts.applyOptions({barSpacing:Math.min(spacing,50)});
        }
      });
    });
  }
  if(zoomOutBtn){
    zoomOutBtn.addEventListener('click',()=>{
      charts.forEach(({ch})=>{
        if(ch){
          const ts=ch.timeScale();
          const spacing=(ts.options().barSpacing||8)-2;
          ts.applyOptions({barSpacing:Math.max(spacing,1)});
        }
      });
    });
  }
} catch(e) {
  console.warn('Zoom controls error:',e);
}

} catch(globalError) {
  console.error('Chart initialization failed:', globalError);
  document.body.innerHTML = '<div style="color:#ef4444;padding:40px;font-family:system-ui;text-align:center;"><h2>Chart Error</h2><p>' + globalError.message + '</p><p>Open browser console for details.</p></div>';
}

})();
</script>
</body>
</html>'''


# Legacy function kept for backwards compatibility (uses streaming internally now)
def _generate_hft_html(
    candle_data: dict,
    indicators: dict,
    trades: dict,
    config: dict
) -> str:
    """
    DEPRECATED: Use _write_html_streaming for direct file writing.
    This wrapper builds the HTML in memory for backwards compatibility.
    """
    import tempfile
    import os

    # Create temp file, write with streaming, read back
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.html', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _write_html_streaming(tmp_path, candle_data, indicators, trades, config)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_trades(
    df: DataFrameType,
    df_trades: DataFrameType,
    plot_base: str,
    fecha_inicio_plot: str,
    fecha_fin_plot: str,
    trial_number: int,
    params: dict,
    score: float,
    combo: str,
    metrics: Optional[dict] = None,
    equity_curve: Optional[list] = None,
    saldo_inicial: float = 300.0,
    max_archivos: int = 5,
    activo: Optional[str] = None,
):
    """
    Generate ultra-fast trading chart HTML with dynamic indicator detection.
    
    Automatically detects indicators from DataFrame columns and generates
    appropriate panels (overlays on price, oscillators as sub-panels).
    """

    # ================== EXTRACT NUMPY ARRAYS ==================
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else df.to_pandas().index.values
        opens = df["open"].to_numpy().astype(np.float64)
        highs = df["high"].to_numpy().astype(np.float64)
        lows = df["low"].to_numpy().astype(np.float64)
        closes = df["close"].to_numpy().astype(np.float64)
        volumes = df["volume"].to_numpy().astype(np.float64) if "volume" in df.columns else None
        set(df.columns)
    else:
        df_pd = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        if not isinstance(df_pd.index, pd.DatetimeIndex):
            if "timestamp" in df_pd.columns:
                df_pd = df_pd.set_index("timestamp")
        timestamps = df_pd.index.values
        opens = df_pd["open"].values.astype(np.float64)
        highs = df_pd["high"].values.astype(np.float64)
        lows = df_pd["low"].values.astype(np.float64)
        closes = df_pd["close"].values.astype(np.float64)
        volumes = df_pd["volume"].values.astype(np.float64) if "volume" in df_pd.columns else None
        set(df_pd.columns)

    # ================== DATE FILTERING ==================
    start_pd = pd.to_datetime(fecha_inicio_plot, utc=True)
    end_pd = pd.to_datetime(fecha_fin_plot, utc=True)
    start = np.datetime64(start_pd.tz_localize(None))
    end = np.datetime64(end_pd.tz_localize(None))

    if np.issubdtype(timestamps.dtype, np.datetime64):
        ts_compare = timestamps
    else:
        ts_compare = timestamps.astype('datetime64[ns]')

    # Guardar copia por si el rango configurado no cruza el dataset.
    timestamps_all = timestamps
    opens_all = opens
    highs_all = highs
    lows_all = lows
    closes_all = closes
    volumes_all = volumes

    mask = (ts_compare >= start) & (ts_compare <= end)

    timestamps = timestamps[mask]
    opens = opens[mask]
    highs = highs[mask]
    lows = lows[mask]
    closes = closes[mask]
    if volumes is not None:
      volumes = volumes[mask]

    # Si el rango de plot no tiene datos (muy común al cambiar timeframe),
    # hacemos fallback a todo el dataset disponible para no “dejar de generar”.
    if len(timestamps) == 0:
      timestamps = timestamps_all
      opens = opens_all
      highs = highs_all
      lows = lows_all
      closes = closes_all
      volumes = volumes_all

    # ================== SYNTHETIC DATA: LIMIT PLOT RANGE ==================
    # Para datasets grandes (sintéticos o multi-año en TF cortos), limitar gráfica
    # para evitar overflow numérico y mejorar rendimiento visual.
    # PERO: si el usuario pidió un rango largo (ej: TF 1d, 2017→2025), respetarlo.
    if len(timestamps) >= 2:
        ts_arr = timestamps.astype('datetime64[s]').astype(np.int64)
        first_diff_seconds = abs(ts_arr[1] - ts_arr[0])
        minutes_per_candle = max(1, first_diff_seconds / 60)

        # Calcular cuántas velas cubre el rango de plot solicitado
        range_days = (end_pd - start_pd).total_seconds() / 86400
        candles_for_range = int(range_days * 24 * 60 / minutes_per_candle)

        # Solo recortar si el TF es pequeño (< 12h = 720 min) y hay demasiadas velas
        # Para TF grandes (12h, 1d) NUNCA recortar — el usuario quiere ver todo
        if minutes_per_candle < 720:
            # Velas para 2 meses (60 días)
            candles_2_months = int(60 * 24 * 60 / minutes_per_candle)

            # Si tenemos más velas de las que caben en 2 meses, recortar
            if len(timestamps) > candles_2_months * 1.5:
                # Tomar los últimos 2 meses (donde probablemente hay más trades)
                timestamps = timestamps[-candles_2_months:]
                opens = opens[-candles_2_months:]
                highs = highs[-candles_2_months:]
                lows = lows[-candles_2_months:]
                closes = closes[-candles_2_months:]
                if volumes is not None:
                    volumes = volumes[-candles_2_months:]

    # ================== BANKRUPTCY CUTOFF ==================
    saldo_minimo_operativo = 5.0
    if equity_curve and len(equity_curve) > 0:
        eq_arr = np.array(equity_curve, dtype=np.float64)
        bankruptcy_indices = np.where(eq_arr <= saldo_minimo_operativo)[0]
        if len(bankruptcy_indices) > 0:
            bankruptcy_idx = int(bankruptcy_indices[0])
            if bankruptcy_idx < len(eq_arr) - 1:
                ratio = (bankruptcy_idx + 1) / len(eq_arr)
                candle_cutoff = max(1, min(int(len(timestamps) * ratio), len(timestamps)))
                timestamps = timestamps[:candle_cutoff]
                opens = opens[:candle_cutoff]
                highs = highs[:candle_cutoff]
                lows = lows[:candle_cutoff]
                closes = closes[:candle_cutoff]
                if volumes is not None:
                    volumes = volumes[:candle_cutoff]
                equity_curve = equity_curve[:bankruptcy_idx + 1]

    # ================== PREPARE OHLCV ==================
    ts_q_full, o_q_full, h_q_full, l_q_full, c_q_full, vol_q_full, price_factor = _prepare_ohlcv_vectorized(
        timestamps, opens, highs, lows, closes, volumes
    )

    # ================== GLOBAL WARM-UP PERIOD (v6.1) ==================
    # Detect the maximum period from all *_period params BEFORE slicing data
    # This ensures candles + indicators + markers all start at the same point
    max_warmup = _detect_max_warmup_period(params, min_warmup=1)

    # Para TF grandes (12h, 1d), el warmup en velas puede ser enorme
    # comparado con el dataset total. Limitamos el warmup VISUAL para
    # que la gráfica muestre la mayor parte del rango, sin cortar datos.
    # Los indicadores ya están calculados — solo afecta al recorte visual.
    if len(ts_q_full) >= 2:
        _ts_arr = ts_q_full[:2]
        _candle_diff = abs(int(_ts_arr[1]) - int(_ts_arr[0]))
        _minutes_per_candle = max(1, _candle_diff / 60)
        if _minutes_per_candle >= 720:  # TF >= 12h
            # No recortar más del 15% del dataset por warmup visual
            max_visual_warmup = int(len(ts_q_full) * 0.15)
            if max_warmup > max_visual_warmup:
                max_warmup = max_visual_warmup

    # Clamp warmup to valid range (leave at least 10 candles visible)
    max_warmup = min(max_warmup, len(ts_q_full) - 10)
    max_warmup = max(0, max_warmup)  # Ensure non-negative

    # (debug print removed)

    # SLICE ALL DATA FROM WARMUP POINT - Everything synchronized
    # Start on the *next* candle after warmup so indicator values are based on
    # completed history only (reduces 1-bar visual desync vs trade execution).
    start_idx = min(max_warmup + 1, len(ts_q_full) - 10)
    start_idx = max(0, start_idx)

    ts_q = ts_q_full[start_idx:]
    o_q = o_q_full[start_idx:]
    h_q = h_q_full[start_idx:]
    l_q = l_q_full[start_idx:]
    c_q = c_q_full[start_idx:]
    vol_q = vol_q_full[start_idx:] if vol_q_full is not None else None

    # Get the warmup threshold timestamp for trade marker filtering
    warmup_threshold_ts = int(ts_q[0]) if len(ts_q) > 0 else 0

    candle_data = {
        "t": ts_q.tolist(),
        "o": o_q.tolist(),
        "h": h_q.tolist(),
        "l": l_q.tolist(),
        "c": c_q.tolist(),
        "f": int(price_factor)
    }
    if vol_q is not None:
        candle_data["vol"] = vol_q.tolist()

    # ================== ZERO-LAG INDICATOR ALIGNMENT (v6.0) ==================
    # Architecture: Single Authoritative Timestamp Array (SATA)
    # ts_q is the ONLY source of truth for all timestamp alignment.
    # All indicator values are mapped to ts_q indices using StrictAlignmentMapper.

    # Step 1: Convert source DataFrame to pandas with UTC index
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        df_pd_full = df.to_pandas()
    else:
        df_pd_full = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    if not isinstance(df_pd_full.index, pd.DatetimeIndex):
        if "timestamp" in df_pd_full.columns:
            df_pd_full = df_pd_full.set_index("timestamp")

    # Ensure index is UTC-aware for consistent timestamp conversion
    if df_pd_full.index.tz is None:
        df_pd_full.index = df_pd_full.index.tz_localize("UTC")
    elif str(df_pd_full.index.tz) != 'UTC':
        df_pd_full.index = df_pd_full.index.tz_convert("UTC")

    # Step 2: Extract source timestamps as Unix seconds
    # Use the centralized _normalize_timestamps_to_unix for consistency
    source_ts_raw = df_pd_full.index.tz_localize(None).values  # Remove TZ for datetime64 conversion
    source_timestamps = _normalize_timestamps_to_unix(source_ts_raw)

    # Step 3: Initialize the StrictAlignmentMapper with authoritative timestamps
    # This is the KEY component for zero-lag alignment
    aligner = StrictAlignmentMapper(ts_q)

    # Step 4: Create aligned source mask (which source rows exist in ts_q)
    source_mask = np.array([int(ts) in aligner.ts_to_idx for ts in source_timestamps])
    df_aligned = df_pd_full[source_mask].copy()
    aligned_source_ts = source_timestamps[source_mask]

    # Verify alignment
    alignment_match = len(df_aligned) == len(ts_q)
    # print(f"[PLOT v6.0] Candles: {len(ts_q)}, Aligned Source: {len(df_aligned)}, Perfect Match: {alignment_match}")

    if not alignment_match:
        # Detailed debug for misalignment
        missing_in_source = len(ts_q) - len(df_aligned)
        print(f"[PLOT WARN] Missing {missing_in_source} timestamps in source DataFrame. Using mapping fallback.")

    # Detect indicators (strategy-driven via __indicators_used/__indicator_specs/__indicator_bounds)
    price_range = (float(closes.min()), float(closes.max()))
    detected = _detect_indicators(df_aligned, price_range, params)

    indicators = {"overlays": [], "sub_panels": []}

    # ================== PROCESS OVERLAYS (ZERO-LAG, PRE-SLICED) ==================
    for overlay_cfg in detected["overlays"]:
        col = overlay_cfg["col"]
        if col not in df_aligned.columns:
            continue

        # Extract values and align using the mapper
        vals = df_aligned[col].values.astype(np.float64)
        aligned_vals = aligner.align(aligned_source_ts, vals)

        # Only add if we have valid data
        valid_count = aligner.count_valid(aligned_vals)
        if valid_count > 0:
            precision = int(overlay_cfg.get("precision", 2))
            quantized, factor = aligner.quantize(aligned_vals, precision=precision)

            indicators["overlays"].append({
                "t": ts_q.tolist(),  # AUTHORITATIVE timestamps
                "v": quantized,
                "f": int(factor),
                "color": overlay_cfg["color"]
            })

    # ================== PROCESS SUB-PANELS / OSCILLATORS (ZERO-LAG, PRE-SLICED) ==================
    for panel_cfg in detected["sub_panels"]:
        col = panel_cfg["col"]
        if col not in df_aligned.columns:
            continue

        # Extract values and align using the mapper
        vals = df_aligned[col].values.astype(np.float64)
        aligned_vals = aligner.align(aligned_source_ts, vals)

        # Only add if we have valid data
        valid_count = aligner.count_valid(aligned_vals)
        if valid_count > 0:
            # Use dynamic name from detection (includes period if found in params)
            panel_name = panel_cfg.get("name", col.upper())

            precision = int(panel_cfg.get("precision", 4))
            quantized, factor = aligner.quantize(aligned_vals, precision=precision)

            indicators["sub_panels"].append({
                "name": panel_name,
                "type": panel_cfg["type"],
                "color": panel_cfg["color"],
                "bounds": panel_cfg.get("bounds"),
                "zero_line": bool(panel_cfg.get("bounds", {}) and ("mid" in (panel_cfg.get("bounds") or {}))),
                "data": {
                    "t": ts_q.tolist(),  # AUTHORITATIVE timestamps - ZERO-LAG GUARANTEED
                    "v": quantized,
                    "f": int(factor)
                }
            })

    # ================== TRADE MARKERS (Temporal Snapping + Warmup Filter) ==================
    # Use np.searchsorted to snap trade timestamps to exact candle timestamps
    # This prevents marker disappearance during scroll/zoom
    # WARMUP FILTER: Trades within the warmup period are not displayed
    # m: candle markers (entries; time-only)
    # ee: entry points at exact entry_price (time+value)
    # em: entry markers for the entry-price series
    # i: trade info for tooltips
    # xe: exit points at exact exit_price (time+value)
    # xm: exit markers for the exit series (white dots)
    trades = {"m": [], "ee": [], "em": [], "i": [], "xe": [], "xm": []}
    max_valid_ts = int(ts_q[-1]) if len(ts_q) > 0 else None

    # Build efficient lookup structure for candle timestamps
    candle_ts_set = set(ts_q.tolist()) if len(ts_q) > 0 else set()

    def _snap_to_candle(trade_ts: int, candle_timestamps: np.ndarray) -> int:
        """
        Vectorized temporal snapping using binary search.
        Snaps trade timestamp to the nearest previous/equal candle timestamp.
        
        Uses np.searchsorted with side='right' to find insertion point,
        then subtracts 1 to get the candle at or before the trade time.
        """
        if len(candle_timestamps) == 0:
            return trade_ts

        # Find insertion point (index where trade_ts would be inserted to maintain order)
        idx = np.searchsorted(candle_timestamps, trade_ts, side='right')

        # Clamp to valid range and get the candle at or before trade time
        idx = max(0, min(idx - 1, len(candle_timestamps) - 1))

        return int(candle_timestamps[idx])

    if df_trades is not None:
        if HAS_POLARS and isinstance(df_trades, pl.DataFrame):
            trades_df = df_trades.to_pandas()
        else:
            trades_df = df_trades.copy() if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(df_trades)

        if not trades_df.empty:
            entry_times_dt = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
            exit_times_dt = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

            valid_mask = entry_times_dt.notna()
            trades_df = trades_df[valid_mask].copy()
            entry_times_dt = entry_times_dt[valid_mask]
            exit_times_dt = exit_times_dt[valid_mask]

            if len(trades_df) > 0 and len(ts_q) > 0:
                # Robust epoch seconds (UTC) for tz-aware timestamps
                # NOTE: Polars returns datetime64[us] (microseconds), Pandas may return [ns]
                # Use .apply(lambda x: x.timestamp()) for robust conversion
                entry_timestamps = entry_times_dt.apply(lambda x: int(x.timestamp()) if pd.notna(x) else 0)
                exit_timestamps = exit_times_dt.apply(lambda x: int(x.timestamp()) if pd.notna(x) else 0)

                start_ts = int(start_pd.timestamp())
                end_ts = max_valid_ts if max_valid_ts else int(end_pd.timestamp())
                mask = (entry_timestamps >= start_ts) & (entry_timestamps <= end_ts)

                trades_df = trades_df[mask].copy()
                entry_timestamps = entry_timestamps[mask]
                exit_timestamps = exit_timestamps[mask]

                # Vectorized snapping: snap to the NEAREST candle timestamp.
                # This avoids 1-bar lag when timestamps have minor rounding offsets.
                def _snap_nearest(candle_ts: np.ndarray, trade_ts: np.ndarray) -> np.ndarray:
                  if len(candle_ts) == 0 or len(trade_ts) == 0:
                    return trade_ts
                  idx = np.searchsorted(candle_ts, trade_ts, side='left')
                  idx = np.clip(idx, 0, len(candle_ts) - 1)
                  prev_idx = np.clip(idx - 1, 0, len(candle_ts) - 1)
                  next_ts = candle_ts[idx]
                  prev_ts = candle_ts[prev_idx]
                  choose_prev = (np.abs(trade_ts - prev_ts) <= np.abs(next_ts - trade_ts))
                  return np.where(choose_prev, prev_ts, next_ts)

                snapped_entry_ts = _snap_nearest(ts_q, entry_timestamps)
                snapped_exit_ts = _snap_nearest(ts_q, exit_timestamps)

                # ============================================================
                # VECTORIZED TRADE PROCESSING (v7.0 - NO iterrows)
                # Extract all columns as numpy arrays for 100x faster processing
                # ============================================================
                n_trades = len(trades_df)

                # Extract columns to numpy arrays (zero-copy when possible)
                types_arr = trades_df["type"].values if "type" in trades_df.columns else np.array([""] * n_trades)
                entry_price_arr = trades_df["entry_price"].values.astype(np.float64) if "entry_price" in trades_df.columns else np.zeros(n_trades)
                exit_price_arr = trades_df["exit_price"].values if "exit_price" in trades_df.columns else np.array([None] * n_trades)
                pnl_arr = trades_df["pnl_neto"].values.astype(np.float64) if "pnl_neto" in trades_df.columns else np.zeros(n_trades)

                # Handle optional columns with fallbacks
                if "comision_total" in trades_df.columns:
                    comm_arr = trades_df["comision_total"].values
                elif "comision" in trades_df.columns:
                    comm_arr = trades_df["comision"].values
                else:
                    comm_arr = np.zeros(n_trades)

                if "qty" in trades_df.columns:
                    qty_arr = trades_df["qty"].values
                elif "cantidad" in trades_df.columns:
                    qty_arr = trades_df["cantidad"].values
                elif "size" in trades_df.columns:
                    qty_arr = trades_df["size"].values
                else:
                    qty_arr = np.zeros(n_trades)

                tipo_salida_arr = trades_df["tipo_salida"].values if "tipo_salida" in trades_df.columns else np.array([None] * n_trades)
                exit_ts_valid = exit_timestamps.values if hasattr(exit_timestamps, 'values') else np.array(exit_timestamps)

                # Vectorized masks
                warmup_mask = snapped_entry_ts >= warmup_threshold_ts
                candle_mask = np.isin(snapped_entry_ts, ts_q)
                valid_trade_mask = warmup_mask & candle_mask

                # Process only valid trades
                valid_indices = np.where(valid_trade_mask)[0]

                for i in valid_indices:
                    et = int(snapped_entry_ts[i])
                    trade_type = str(types_arr[i]).upper() if types_arr[i] else ""
                    ep = float(entry_price_arr[i])
                    xp_raw = exit_price_arr[i]
                    xp = float(xp_raw) if pd.notna(xp_raw) else None
                    pnl = float(pnl_arr[i])
                    xt = int(snapped_exit_ts[i]) if pd.notna(exit_ts_valid[i]) else None

                    comm_raw = comm_arr[i]
                    comm = float(comm_raw) if pd.notna(comm_raw) else 0.0
                    qty_raw = qty_arr[i]
                    qty = float(qty_raw) if pd.notna(qty_raw) else 0.0
                    tipo_salida = tipo_salida_arr[i]

                    trade_info = {
                        "type": trade_type,
                        "ep": round(ep, 2),
                        "xp": round(xp, 2) if xp else None,
                        "pnl": round(pnl, 2),
                        "comm": round(comm, 2),
                        "qty": round(qty, 6)
                    }

                    if tipo_salida is not None and pd.notna(tipo_salida):
                        trade_info["xs"] = str(tipo_salida)

                    # DOT-STYLE CENTERED MARKERS (circles, inBar for precision)
                    # Deduplicate: only one entry marker per timestamp
                    entry_color = "#3b82f6" if trade_type == "LONG" else "#a855f7"
                    if not any(x["time"] == et for x in trades["m"]):
                        trades["m"].append({
                            "time": et,
                            "position": "inBar",
                            "color": entry_color,
                            "shape": "circle",
                            "text": "",
                            "size": 2
                        })

                    # Entry points at real price (deduplicated)
                    if not any(x["time"] == et for x in trades["ee"]):
                        trades["ee"].append({"time": et, "value": ep})
                        trades["em"].append({
                            "time": et,
                            "position": "inBar",
                            "color": entry_color,
                            "shape": "circle",
                            "text": "",
                            "size": 2,
                        })

                    trades["i"].append({"time": et, **trade_info})

                    # Exit points at real price (deduplicated - only one marker per exit timestamp)
                    if xt is not None and xp is not None and xt in candle_ts_set:
                        # Check if we already have an exit at this timestamp
                        if not any(x["time"] == xt for x in trades["xm"]):
                            trades["xe"].append({"time": xt, "value": float(xp)})
                            trades["xm"].append({
                                "time": xt,
                                "position": "aboveBar",
                                "color": "#ffffff",
                                "shape": "circle",
                                "text": "",
                                "size": 2,
                            })
                            trades["i"].append({"time": xt, **trade_info})

                trades["m"].sort(key=lambda x: x["time"])
                trades["ee"].sort(key=lambda x: x["time"])
                trades["em"].sort(key=lambda x: x["time"])
                trades["i"].sort(key=lambda x: x["time"])
                trades["xe"].sort(key=lambda x: x["time"])
                trades["xm"].sort(key=lambda x: x["time"])

    # ================== CONFIG ==================
    total_trades = 0
    winrate = 0.0
    pnl_neto = 0.0

    if metrics:
        total_trades = int(metrics.get("total_trades", metrics.get("num_trades", 0)))
        winrate = float(metrics.get("win_rate", metrics.get("winrate", 0))) * 100 if metrics.get("win_rate", metrics.get("winrate", 0)) <= 1 else float(metrics.get("win_rate", metrics.get("winrate", 0)))
        pnl_neto = float(metrics.get("pnl_neto", metrics.get("net_pnl", 0)))
    elif df_trades is not None:
        if HAS_POLARS and isinstance(df_trades, pl.DataFrame):
            total_trades = len(df_trades)
            pnl_col = "pnl_neto" if "pnl_neto" in df_trades.columns else "pnl"
            if pnl_col in df_trades.columns:
                pnl_neto = float(df_trades[pnl_col].sum())
                wins = (df_trades[pnl_col] > 0).sum()
                winrate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            trades_df_for_stats = df_trades if isinstance(df_trades, pd.DataFrame) else pd.DataFrame(df_trades)
            total_trades = len(trades_df_for_stats)
            pnl_col = "pnl_neto" if "pnl_neto" in trades_df_for_stats.columns else "pnl"
            if pnl_col in trades_df_for_stats.columns:
                pnl_neto = float(trades_df_for_stats[pnl_col].sum())
                wins = (trades_df_for_stats[pnl_col] > 0).sum()
                winrate = (wins / total_trades * 100) if total_trades > 0 else 0

    config = {
        "activo": str(activo).upper() if activo else "",
        "combo": _generate_dynamic_combo(params, combo) if params else combo,
        "score": score,
        "trial": trial_number,
        "total_trades": total_trades,
        "winrate": winrate,
        "pnl_neto": pnl_neto
    }

    # ================== SAVE FILE (STREAMING - v7.0) ==================
    os.makedirs(plot_base, exist_ok=True)

    # Sanitize combo name for filename (remove special chars, limit length)
    combo_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", combo or "STRATEGY")[:30]
    filename = f"TRIAL-{trial_number}_SCORE-{score:.2f}_{combo_safe}.html"
    filepath = os.path.join(plot_base, filename)

    # Direct streaming write - zero RAM overhead for large charts
    _write_html_streaming(filepath, candle_data, indicators, trades, config)

    # ================== CLEANUP ==================
    if max_archivos > 0:
        _cleanup_old_plots(plot_base, max_archivos)


def _cleanup_old_plots(plot_base: str, max_archivos: int):
  """Remove old plot files, keeping only the best scores."""
  try:
    all_files = [
      f
      for f in os.listdir(plot_base)
      if f.endswith(".html") and f.startswith("TRIAL-")
    ]

    files_with_scores: list[tuple[str, float]] = []

    for fname in all_files:
      # Match format: TRIAL-{n}_SCORE-{score}_{combo}.html
      match = re.search(r"TRIAL-\d+_SCORE-(-?\d+(?:\.\d+)?)_.*\.html$", fname)
      if not match:
        # Archivos legacy (u otro formato) deben eliminarse primero
        # para garantizar el límite max_archivos.
        files_with_scores.append((fname, float("-inf")))
        continue
      try:
        files_with_scores.append((fname, float(match.group(1))))
      except ValueError:
        files_with_scores.append((fname, float("-inf")))

    files_with_scores.sort(key=lambda x: x[1], reverse=True)

    if len(files_with_scores) > max_archivos:
      for fname, _ in files_with_scores[max_archivos:]:
        old_path = os.path.join(plot_base, fname)
        if os.path.exists(old_path):
          os.remove(old_path)
  except Exception:
    # Best-effort cleanup only
    pass


# =============================================================================
# PLOT REPORTER - Reporter para integración con OptimizationRunner
# =============================================================================
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, Any

logger = logging.getLogger(__name__)


class ReporterProtocol(Protocol):
    """Protocolo base para reporters."""
    def needs_dataframe(self, score: float) -> bool: ...
    def on_trial_end(self, artifacts: Any) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


@dataclass
class PlotReporter:
    """
    Lightweight Charts (TradingView) HTML exporter - OPTIMIZADO.
    
    Genera gráficos HTML interactivos para los top N trials.
    Los gráficos incluyen:
    - Velas OHLC
    - Indicadores calculados
    - Marcadores de entrada/salida
    - Curva de equity
    """

    plot_base: str = "resultados/graficos"
    fecha_inicio_plot: str = "2025-01-01"
    fecha_fin_plot: str = "2025-01-20"
    plot_meses_duracion: int = 2
    max_archivos: int = 5
    saldo_inicial: float = 300.0
    activo: Optional[str] = None
    
    _candidates: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _min_candidate_score: float = field(default=float("-inf"), init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        if score is None:
            return False
        if len(self._candidates) < self.max_archivos:
            return True
        return score > self._min_candidate_score

    def _update_min_score(self):
        if self._candidates:
            self._min_candidate_score = min(c["score"] for c in self._candidates)
        else:
            self._min_candidate_score = float("-inf")

    def on_trial_end(self, artifacts) -> None:
        if artifacts.trial_number == 0 and os.path.exists(self.plot_base):
            try:
                for f in os.listdir(self.plot_base):
                    if f.startswith("TRIAL-") and f.endswith(".html"):
                        os.remove(os.path.join(self.plot_base, f))
            except Exception:
                pass
            self._candidates = []
            self._min_candidate_score = float("-inf")

        score = artifacts.score
        if score is None:
            return

        is_candidate = (
            len(self._candidates) < self.max_archivos or 
            score > self._min_candidate_score
        )
        
        if not is_candidate:
            return
        
        if getattr(artifacts, "df_signals", None) is None:
            return

        params_for_plot = getattr(artifacts, "params_reporting", None) or artifacts.params
        
        candidate = {
            "score": score,
            "trial_number": artifacts.trial_number,
            "strategy_name": artifacts.strategy_name,
            "params": deepcopy(params_for_plot),
            "metrics": deepcopy(artifacts.metrics) if artifacts.metrics else {},
            "equity_curve": list(artifacts.equity_curve) if artifacts.equity_curve else [],
            "df_signals": artifacts.df_signals,
            "trades": artifacts.trades,
            "trial_date_range": getattr(artifacts, "trial_date_range", None),
        }
        
        self._candidates.append(candidate)
        
        if len(self._candidates) > self.max_archivos:
            self._candidates.sort(key=lambda x: x["score"], reverse=True)
            removed = self._candidates.pop()
            del removed
        
        self._update_min_score()

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if not self._candidates:
            return
        
        os.makedirs(self.plot_base, exist_ok=True)
        
        self._candidates.sort(key=lambda x: x["score"], reverse=True)
        
        for candidate in self._candidates[:self.max_archivos]:
            try:
                # Determinar fechas del plot:
                # Si el trial tiene rango propio (USAR_RANGOS_POR_TRIAL),
                # usar los primeros N meses de ese rango.
                # Si no, usar las fechas estáticas de configuración.
                _plot_start = self.fecha_inicio_plot
                _plot_end = self.fecha_fin_plot
                _tdr = candidate.get("trial_date_range")
                if _tdr is not None:
                    from datetime import datetime, timedelta
                    try:
                        _dt_start = datetime.fromisoformat(_tdr[0])
                        _dt_end = _dt_start + timedelta(days=self.plot_meses_duracion * 30)
                        # No exceder el fin del trial
                        _dt_trial_end = datetime.fromisoformat(_tdr[1])
                        if _dt_end > _dt_trial_end:
                            _dt_end = _dt_trial_end
                        _plot_start = _dt_start.strftime("%Y-%m-%d")
                        _plot_end = _dt_end.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                plot_trades(
                    df=candidate["df_signals"],
                    df_trades=candidate["trades"],
                    plot_base=self.plot_base,
                    fecha_inicio_plot=_plot_start,
                    fecha_fin_plot=_plot_end,
                    trial_number=candidate["trial_number"],
                    params=candidate["params"],
                    score=candidate["score"],
                    combo=candidate["strategy_name"],
                    metrics=candidate["metrics"],
                    equity_curve=candidate["equity_curve"],
                    saldo_inicial=self.saldo_inicial,
                    max_archivos=self.max_archivos,
                    activo=self.activo,
                )
            except Exception as e:
                logger.warning(f"Error generando plot para trial {candidate['trial_number']}: {e}")
        
        self._candidates = []
        self._min_candidate_score = float("-inf")
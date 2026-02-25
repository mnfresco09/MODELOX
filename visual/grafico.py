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
  "#4F94CD",  # Steel Blue 3 (Rich Secondary - Standard)
  "#1C86EE",  # Dodger Blue 2 (Rich Primary - Active)
  "#CDAD00",  # Gold 3 (Rich Accent - Highlight)
  "#7B88A1",  # Slate (Neutral)
  "#6CA6CD",  # Sky Blue 3 (Medium)
  "#36648B",  # Steel Blue 4 (Dark)
  "#5F9EA0",  # Cadet Blue (Muted Teal-Blue)
  "#9FB6CD",  # Slate Gray 3 (Light Neutral)
  "#6495ED",  # Cornflower Blue (Soft Blue)
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

    # PRIORIDAD 0: Specs (Direct override)
    panel = spec.get("panel", None)

    # PRIORIDAD 1: Bounds (Si no hay specs, mirar si bounds define panel)
    if panel is None:
      col_bounds = bounds_map.get(col, None)
      if isinstance(col_bounds, dict) and "panel" in col_bounds:
        panel = col_bounds["panel"]

    # PRIORIDAD 2: Heurísticas (Solo si no se definió explícitamente)
    if panel is None:
      col_lower = col.lower()
      if _is_overlay_by_name(col_lower):
        panel = "overlay"
      else:
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
          "panel_id": panel if (isinstance(panel, (int, str)) and panel != "sub") else None,
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
        # Preserve fractional volume for crypto (don't round to int)
        pass

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
    STREAMING HTML GENERATOR (v9.0 - MINIMALIST NEUTRAL)
    
    Writes HTML directly to disk in chunks.
    Style: Minimalist, Neutral, "Nord/Notion" aesthetic.
    """

    activo = str(config.get("activo", ""))
    combo = str(config.get("combo", ""))
    total_trades = int(config.get("total_trades", 0))
    winrate = float(config.get("winrate", 0))
    pnl_neto = float(config.get("pnl_neto", 0))
    pnl_class = "pos" if pnl_neto >= 0 else "neg"
    trial = str(config.get("trial", ""))
    score = float(config.get("score", 0))

    with open(filepath, "wb") as f:
        # ============ CHUNK 1: HTML Header + CSS (Minimalist Style) ============
        header = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>MODELOX | {activo}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script>
window.addEventListener('DOMContentLoaded', function() {{
  if (typeof LightweightCharts === 'undefined') {{
    document.body.innerHTML = '<div style="color:#666;padding:40px;font-family:sans-serif;text-align:center;">Could not load chart library (lightweight-charts)</div>';
  }}
}});
</script>
<style>
/* MINIMALIST NEUTRAL THEME */
:root {{
  --bg-app: #121212;
  --bg-panel: #181818;
  --bg-header: #181818;
  --border-subtle: #2A2A2A;
  --text-main: #C9D1D9;
  --text-muted: #6E7681;
  --accent-blue: #79C0FF;
  --accent-green: #7EE787;
  --accent-red: #FF7B72;
  --accent-gold: #D29922;
}}

*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:var(--bg-app);font-family:'Inter',sans-serif;overflow:hidden;touch-action:none;color:var(--text-main)}}

/* LAYOUT */
.c{{display:flex;flex-direction:column;height:100vh;padding:0}}

/* HEADER - CLEAN & FLAT */
.h{{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:0 24px;
    height: 54px;
    background:var(--bg-header);
    border-bottom:1px solid var(--border-subtle);
}}

.h .left{{display:flex;align-items:center;gap:16px}}
.h .a{{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;color:var(--text-main);letter-spacing:-0.5px}}
.h .t{{font-size:12px;font-weight:400;color:var(--text-muted);background:rgba(255,255,255,0.05);padding:4px 8px;border-radius:4px}}

.h .info{{display:flex;gap:24px;align-items:center}}
.h .stat{{display:flex;align-items:baseline;gap:8px}}
.h .stat-label{{color:var(--text-muted);font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px}}
.h .stat-val{{color:var(--text-main);font-size:14px;font-weight:600;font-family:'IBM Plex Mono',monospace}}
.h .stat-val.pos{{color:var(--accent-green)}}
.h .stat-val.neg{{color:var(--accent-red)}}

/* CHARTS AREA */
.p{{flex:1;display:flex;flex-direction:column;min-height:0;position:relative}}
.m{{position:relative;margin-bottom:0;border-bottom:1px solid var(--border-subtle)}}
.sub{{position:relative;margin-bottom:0;border-bottom:1px solid var(--border-subtle);flex:1 1 0;min-height:60px;overflow:hidden}}

/* PANEL LABELS - SUBTLE */
.l{{
    position:absolute;
    top:12px;
    left:16px;
    z-index:20;
    font-size:10px;
    font-weight:600;
    color:var(--text-muted);
    pointer-events:none;
    text-transform:uppercase;
    letter-spacing:1px;
    opacity:0.7;
}}

/* TOOLTIP - FLOATING & MINIMAL */
#tt{{
    position:fixed;
    display:none;
    background:rgba(24,24,24,0.95);
    border:1px solid var(--border-subtle);
    border-radius:6px;
    padding:12px;
    color:var(--text-main);
    font-size:12px;
    z-index:100;
    pointer-events:none;
    box-shadow:0 8px 24px rgba(0,0,0,0.5);
    backdrop-filter:blur(4px);
    width: 200px;
}}
.tt-row{{display:flex;justify-content:space-between;margin-bottom:4px}}
.tt-label{{color:var(--text-muted)}}
.tt-val{{font-family:'IBM Plex Mono',monospace;font-weight:500}}
.tt-val.pos{{color:var(--accent-green)}}
.tt-val.neg{{color:var(--accent-red)}}
.tt-badge{{padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;text-transform:uppercase}}
.tt-badge.win{{background:rgba(126,231,135,0.1);color:var(--accent-green);border:1px solid rgba(126,231,135,0.2)}}
.tt-badge.loss{{background:rgba(255,123,114,0.1);color:var(--accent-red);border:1px solid rgba(255,123,114,0.2)}}


/* OHLC FLOATING LEGEND */
#ohlc{{
    position:absolute;
    top:12px;
    right:100px;
    z-index:20;
    display:flex;
    gap:16px;
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;
    color:var(--text-muted);
    pointer-events:none;
}}
.ohlc-val{{color:var(--text-main)}}
.ohlc-val.up{{color:var(--accent-green)}}
.ohlc-val.down{{color:var(--accent-red)}}

/* ZOOM CONTROLS */
.tv-zoom-container{{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);z-index:30;display:flex;gap:8px;opacity:0;transition:opacity 0.2s}}
.p:hover .tv-zoom-container{{opacity:1}}
.tv-zoom-btn{{
    width:32px;height:32px;
    background:#2A2A2A;
    color:#fff;
    border:none;
    border-radius:4px;
    cursor:pointer;
    font-size:16px;
    display:flex;align-items:center;justify-content:center;
}}
.tv-zoom-btn:hover{{background:#3A3A3A}}

/* CROSSHAIR */
#globalCrosshair{{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.1);pointer-events:none;z-index:10;display:none}}
#globalCrosshairLabel{{
    position:absolute;
    bottom:0;
    transform:translateX(-50%);
    background:#2A2A2A;
    color:#fff;
    font-size:10px;
    padding:2px 6px;
    border-radius:4px;
    pointer-events:none;
    z-index:30;
    display:none;
    font-family:'IBM Plex Mono',monospace;
}}

.smh{{position:absolute;left:0;right:0;height:10px;z-index:40;cursor:ns-resize}}
.smh.top{{top:0}}
.smh.bottom{{bottom:0}}

</style>
</head>
<body>
<div class="c">
<div class="h">
    <div class="left">
        <span class="a">{activo}</span>
        <span class="t">{combo}</span>
        <span class="t">TRIAL: {trial}</span>
    </div>
    <div class="info">
        <div class="stat"><span class="stat-label">Trades</span><span class="stat-val">{total_trades}</span></div>
        <div class="stat"><span class="stat-label">WinRate</span><span class="stat-val">{round(winrate, 1)}%</span></div>
        <div class="stat"><span class="stat-label">PnL</span><span class="stat-val {pnl_class}">${round(pnl_neto, 2)}</span></div>
        <div class="stat"><span class="stat-label">Score</span><span class="stat-val">{round(score, 2)}</span></div>
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

        # ============ CHUNK 6: Footer (Close Tags) ============
        footer = b"""
} catch(e) {
    console.error(e);
    document.body.innerHTML += '<div style="color:red;margin:20px;background:#333;padding:10px;border-radius:4px;font-family:monospace">JS Error: ' + e.message + '</div>';
}
})();
</script>
</body>
</html>
"""
        f.write(footer)


def _get_chart_js_logic() -> str:
    """Return the JavaScript chart logic as a string (static, cacheable)."""

    return '''

// ============================================================================
// MODELOX CHART v9.0 - NEUTRAL MINIMALIST
// ============================================================================

if (!D || !D.t || D.t.length === 0) {
  document.body.innerHTML = '<div style="display:flex;height:100vh;justify-content:center;align-items:center;color:#666">No data available</div>';
  return;
}

const dq=(v,f)=>v/f;
const ct=document.getElementById('ct');
const charts=[];
let syncingCharts=false;
const _scaleMarginsByChart = new WeakMap();

// HELPERS
function _clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function _findDataAt(s, t) {
    if (!s) return null;
    const d = s.data();
    let l = 0, r = d.length - 1;
    while (l <= r) {
        const m = (l + r) >>> 1;
        const v = d[m];
        if (v.time === t) return v;
        if (v.time < t) l = m + 1;
        else r = m - 1;
    }
    return null;
}
function _getOrInitMargins(ch){
  if(_scaleMarginsByChart.has(ch)) return _scaleMarginsByChart.get(ch);
  const m = { top: 0.1, bottom: 0.1 };
  _scaleMarginsByChart.set(ch, m);
  return m;
}
function _applyMargins(ch, top, bottom){
  top = _clamp(top, 0.05, 0.45);
  bottom = _clamp(bottom, 0.05, 0.45);
  _scaleMarginsByChart.set(ch, { top, bottom });
  try{ ch.applyOptions({ rightPriceScale: { scaleMargins: { top, bottom } } }); }catch(e){}
}
function _attachScaleMarginHandles(panelEl, ch){
  if(!panelEl || !ch) return;
  const hTop = document.createElement('div'); hTop.className = 'smh top';
  const hBot = document.createElement('div'); hBot.className = 'smh bottom';
  panelEl.appendChild(hTop); panelEl.appendChild(hBot);
  
  const startDrag = (which, ev) => {
    ev.preventDefault(); ev.stopPropagation();
    const rect = panelEl.getBoundingClientRect();
    const startY = ev.clientY;
    const startMargins = _getOrInitMargins(ch);
    const onMove = (e) => {
      const dy = e.clientY - startY;
      const df = dy / rect.height;
      if(which === 'top') _applyMargins(ch, startMargins.top + df, startMargins.bottom);
      else _applyMargins(ch, startMargins.top, startMargins.bottom - df);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };
  hTop.addEventListener('pointerdown', (ev)=>startDrag('top', ev));
  hBot.addEventListener('pointerdown', (ev)=>startDrag('bottom', ev));
}

// CHART OPTIONS - NEUTRAL
const baseOpts={
layout:{background:{type:'solid',color:'#121212'},textColor:'#6E7681',fontSize:11,fontFamily:"'Inter',sans-serif"},
grid:{vertLines:{visible:false},horzLines:{color:'#1e1e1e',style:0}},
crosshair:{mode:0,vertLine:{visible:false,labelVisible:false},horzLine:{visible:true,color:'#333',labelBackgroundColor:'#333'}},
timeScale:{
    borderColor:'#2A2A2A',
    rightOffset:5,
    barSpacing:6,
    minBarSpacing:1,
    fixLeftEdge:false,
    fixRightEdge:false,
    lockVisibleTimeRangeOnResize:true,
    visible:false,
    tickMarkFormatter: (time, tickMarkType, locale) => {
        const d = new Date(time * 1000);
        if (d.getUTCMinutes() !== 0) return '';
        const h = d.getUTCHours();
        return h === 0 ? '00' : h.toString();
    }
},
rightPriceScale:{borderColor:'#2A2A2A',scaleMargins:{top:.15,bottom:.15},autoScale:true,alignLabels:true,borderVisible:false},
handleScale:{axisPressedMouseMove:false,mouseWheel:false,pinch:false},
handleScroll:{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},
kineticScroll:{touch:true,mouse:true},
localization:{
  timeFormatter:(ts)=>{
    const d=new Date(ts*1000);
    return d.toISOString().slice(5,16).replace('T',' ');
  },
}
};

function mkPanel(id,lbl,isMain){
  const p=document.createElement('div');
  p.className=isMain?'m':'sub';
  p.id=id;
  p.style.flex=isMain?'3':'1';
  
  const l=document.createElement('div');
  l.className='l';
  l.textContent=lbl;
  p.appendChild(l);

  if(isMain){
    const ohlc=document.createElement('div');
    ohlc.id='ohlc';
    ohlc.innerHTML='<span id="tv" style="margin-right:8px"></span>O <span id="ov" class="ohlc-val">-</span> H <span id="hv" class="ohlc-val">-</span> L <span id="lv" class="ohlc-val">-</span> C <span id="cv" class="ohlc-val">-</span>';
    p.appendChild(ohlc);
    
    // Zoom buttons
    const z=document.createElement('div');
    z.className='tv-zoom-container';
    z.innerHTML='<button class="tv-zoom-btn" id="zoomIn">+</button><button class="tv-zoom-btn" id="zoomOut">-</button>';
    p.appendChild(z);
  }
  
  ct.appendChild(p);

  // === RESIZER LOGIC ===
  if(isMain){
      // Ensure CSS exists
      if(!document.getElementById('resizer-style')){
          const s = document.createElement('style');
          s.id = 'resizer-style';
          s.innerHTML = `.resizer { height: 6px; flex: 0 0 6px; background: #333; cursor: row-resize; width: 100%; transition: background 0.2s; z-index: 10; } .resizer:hover, .resizer.active { background: #2962FF; }`;
          document.head.appendChild(s);
      }

      const resizer = document.createElement('div');
      resizer.className = 'resizer';
      ct.appendChild(resizer);

      let isResizing = false;
      let startY, startHeight;

      resizer.addEventListener('mousedown', (e) => {
          isResizing = true;
          startY = e.clientY;
          startHeight = p.getBoundingClientRect().height;
          resizer.classList.add('active');
          document.body.style.cursor = 'row-resize';
          p.style.flex = 'none'; // Disable flex grow to allow manual sizing
          p.style.height = startHeight + 'px';
      });

      document.addEventListener('mousemove', (e) => {
          if (!isResizing) return;
          const dy = e.clientY - startY;
          let newH = startHeight + dy;
          if(newH < 100) newH = 100; // Min height for Main
          
          // Max height constraint (leave room for sub panels)
          const totalH = ct.clientHeight;
          const subPanelsCount = charts.length - 1; // logical count since 'mc' is main
          const minSubH = 60 * subPanelsCount;
          const maxH = totalH - minSubH - 20; // -20 buffer
          
          if(newH > maxH) newH = maxH;
          
          p.style.height = newH + 'px';
          
          // Force resize main chart immediately
          charts.find(c=>c.id===id).ch.resize(p.clientWidth, newH);
          
          // Force resize others (flex will adjust them, but canvas needs update)
          // We can just rely on the ResizeObserver if we added it, but let's be explicit for smoothness
          requestAnimationFrame(() => {
             charts.forEach(c => {
                 if(c.id !== id) c.ch.resize(c.p.clientWidth, c.p.clientHeight);
             });
          });
      });

      document.addEventListener('mouseup', () => {
          if(isResizing){
            isResizing = false;
            resizer.classList.remove('active');
            document.body.style.cursor = '';
            // Trigger resize for all to be safe
            charts.forEach(c => c.ch.resize(c.p.clientWidth, c.p.clientHeight));
          }
      });
  }

  const opts={...baseOpts,width:p.clientWidth,height:p.clientHeight};
  const ch=LightweightCharts.createChart(p,opts);
  _getOrInitMargins(ch);
  if(!isMain) _attachScaleMarginHandles(p, ch);
  
  charts.push({ch,p,id,label:lbl});
  return ch;
}

// === MAIN CHART ===
const mc=mkPanel('mc','PRICE',true);
// Neutral Candle Colors
const cs=mc.addCandlestickSeries({
    upColor:'#2E8B57', // SeaGreen (Muted)
    downColor:'#CD5C5C', // IndianRed (Muted)
    borderUpColor:'#2E8B57',
    borderDownColor:'#CD5C5C',
    wickUpColor:'#2E8B57',
    wickDownColor:'#CD5C5C',
    priceFormat:{type:'price',precision:2},
});

// VISIBILITY & RESIZE FIX (Tab Open/Restore)
document.addEventListener('visibilitychange', () => {
    if(!document.hidden){
        setTimeout(() => {
            charts.forEach(c => {
                if(c.ch && c.p) c.ch.resize(c.p.clientWidth, c.p.clientHeight);
            });
        }, 100);
    }
});
// Force initial resize check
setTimeout(() => {
    charts.forEach(c => {
        if(c.ch && c.p && c.p.clientHeight < 50) {
             c.p.style.flex = '1 1 0'; // Reset flex if squashed
             c.ch.resize(c.p.clientWidth, c.p.clientHeight);
        }
    });
}, 500);

// RESIZE OBSERVER (Safari/Window Fix)
const ro = new ResizeObserver(entries => {
    window.requestAnimationFrame(() => {
        if(!charts || charts.length===0) return;
        charts.forEach(c => {
            if(c.p && c.ch){
                c.ch.resize(c.p.clientWidth, c.p.clientHeight);
            }
        });
    });
});
if(ct) ro.observe(ct);

const f=D.f;
const cData=D.t.map((t,i)=>({time:t,open:dq(D.o[i],f),high:dq(D.h[i],f),low:dq(D.l[i],f),close:dq(D.c[i],f)}));
cs.setData(cData);
if(charts.length>0) charts[0].series = cs;

// MARKERS & TOOLTIPS (Client-Side Generation)
// Using Scatter LineSeries to place markers at exact prices
const longEntrySeries = mc.addLineSeries({
    color: '#2962FF', 
    lineWidth: 0, 
    pointMarkersVisible: true,
    pointMarkersRadius: 6, 
    crosshairMarkerVisible: false,
    lineVisible: false,
    lastValueVisible: false,
    priceLineVisible: false
});

const shortEntrySeries = mc.addLineSeries({
    color: '#FF6D00',
    lineWidth: 0,
    pointMarkersVisible: true,
    pointMarkersRadius: 6, 
    crosshairMarkerVisible: false,
    lineVisible: false,
    lastValueVisible: false,
    priceLineVisible: false
});

const exitSeries = mc.addLineSeries({
    color: '#FFFFFF',
    lineWidth: 0,
    pointMarkersVisible: true,
    pointMarkersRadius: 5, 
    crosshairMarkerVisible: false,
    lineVisible: false,
    lastValueVisible: false,
    priceLineVisible: false
});

const trailingSeries = mc.addLineSeries({
    color: '#FFFF00', // Yellow
    lineWidth: 0,
    pointMarkersVisible: true,
    pointMarkersRadius: 4, 
    crosshairMarkerVisible: false,
    lineVisible: false,
    lastValueVisible: false,
    priceLineVisible: false
});

const leData = [];
const seData = [];
const exData = [];
const taData = [];
const tooltipMap = new Map();

if(T.list && Array.isArray(T.list)){
    T.list.forEach(tr => {
        // ENTRY
        if(tr.entry_ts !== null) {
            if(tr.type === 'LONG') {
                leData.push({time: tr.entry_ts, value: tr.ep});
            } else {
                seData.push({time: tr.entry_ts, value: tr.ep});
            }
            // Add to tooltip map
            tooltipMap.set(tr.entry_ts, tr);
        }
        
        // EXIT
        if(tr.exit_ts !== null) {
            exData.push({time: tr.exit_ts, value: tr.xp});
            
            // Map exit time too (if different from entry). 
            // Note: If multiple trades close at same candle, last one wins.
            if(!tooltipMap.has(tr.exit_ts)) {
                tooltipMap.set(tr.exit_ts, tr);
            }
        }

        // TRAILING ACTIVATION
        if(tr.ta_ts !== null && tr.ta_p !== null) {
            taData.push({time: tr.ta_ts, value: tr.ta_p});
        }
    });
    
    // Sort required by Lightweight Charts
    leData.sort((a,b) => a.time - b.time);
    seData.sort((a,b) => a.time - b.time);
    exData.sort((a,b) => a.time - b.time);
    taData.sort((a,b) => a.time - b.time);

    longEntrySeries.setData(leData);
    shortEntrySeries.setData(seData);
    exitSeries.setData(exData);
    trailingSeries.setData(taData);
}

// OVERLAYS (MA, Bands)
if(I.overlays && Array.isArray(I.overlays)){
    I.overlays.forEach(ov=>{
        try{
            if(ov.v && ov.v.length>0){
                const ls=mc.addLineSeries({
                    color:ov.color||'#1C86EE', // Rich Dodger Blue
                    lineWidth:1,
                    crosshairMarkerVisible:false,
                    priceLineVisible:false,
                    lastValueVisible:false
                });
                ls.setData(ov.v.map((v,i)=>({time:ov.t[i],value:v!==null?dq(v,ov.f):null})).filter(x=>x.value!==null));
            }
        }catch(e){}
    });
}

// === VOLUME (Separate Panel) ===
if(D.vol && D.vol.length>0){
    const vc = mkPanel('vc', 'VOL', false);
    
    // Configurar márgenes para que ocupe bien su panel
    vc.priceScale('right').applyOptions({
        scaleMargins: { top: 0.1, bottom: 0 },
        borderVisible: false,
        autoScale: true,
    });
    
    const vs = vc.addHistogramSeries({
        color: '#4F94CD', // Rich Steel Blue
        priceFormat: { type: 'volume' },
        priceLineVisible: false,
        lastValueVisible: true, // Mostrar valor actual en eje Y
    });

    vs.setData(D.t.map((t,i)=>({
        time:t, 
        value:D.vol[i], 
        color: D.c[i] >= D.o[i] ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)' // Green/Red muted
    })));
    
    // Registrar para updates
    const vi = charts.findIndex(x=>x.id==='vc');
    if(vi>=0) charts[vi].series = vs;
}

// === INDICATOR PANELS ===
if(I.sub_panels){
    I.sub_panels.forEach((grp,i)=>{
        const pc=mkPanel('sp'+i, grp.title, false);
        grp.series.forEach(ser => {
            let s;
            const data = ser.data.v.map((v,idx)=>({time:ser.data.t[idx],value:v!==null?dq(v,ser.data.f):null})).filter(x=>x.value!==null);
            
            if(ser.type==='histogram'){
                s=pc.addHistogramSeries({color:ser.color,priceLineVisible:false,base:0});
            }else{
                s=pc.addLineSeries({color:ser.color,lineWidth:1,priceLineVisible:false,crosshairMarkerVisible:false});
                
                if(ser.bounds){
                    const b=ser.bounds;
                    
                    // Helper to get color and style for bound
                    const getBoundStyle = (key) => {
                        const k = key.toLowerCase();
                        if(k.includes('upper') || k.includes('hi')) return { c: 'rgba(255, 82, 82, 0.6)', s: 2 }; // Red for Short Entry Zone
                        if(k.includes('lower') || k.includes('lo')) return { c: 'rgba(76, 175, 80, 0.6)', s: 2 }; // Green for Long Entry Zone
                        return { c: 'rgba(255, 255, 255, 0.3)', s: 2 }; // Neutral/Mid
                    };

                    // Iterate explicitly to assign correct colors
                    [
                        {k:'upper', v:b.upper}, 
                        {k:'lower', v:b.lower}, 
                        {k:'mid', v:b.mid}, 
                        {k:'hi', v:b.hi}, 
                        {k:'lo', v:b.lo}
                    ].forEach(item => {
                        if(item.v !== undefined && item.v !== null){
                             const style = getBoundStyle(item.k);
                             const bl=pc.addLineSeries({
                                 color: style.c,
                                 lineWidth: 1,
                                 lineStyle: style.s,
                                 priceLineVisible:false,
                                 lastValueVisible:false,
                                 crosshairMarkerVisible:false
                             });
                             bl.setData(ser.data.t.map(t=>({time:t, value:item.v})));
                        }
                    });
                }
                if(ser.zero_line){
                     const zl=pc.addLineSeries({color:'rgba(255, 255, 255, 0.2)',lineWidth:1,lineStyle:2});
                     zl.setData(ser.data.t.map(t=>({time:t,value:0})));
                }
            }
            s.setData(data);
            
            // === ADD MARKERS TO INDICATOR SERIES ===
            // Filter trades within this series timeframe
            // We use the raw trade list T.list
            if(T && T.list && T.list.length > 0){
                const indMarkers = [];
                const firstTs = ser.data.t[0];
                const lastTs = ser.data.t[ser.data.t.length-1];
                
                // Map timestamps to values for precise placement if needed, 
                // but setMarkers works relative to the checking visible range? 
                // No, setMarkers is for the entire series.
                
                T.list.forEach(tr => {
                    // Entry Marker - Blue (Long) / Orange (Short) Circles
                    if(tr.entry_ts && tr.entry_ts >= firstTs && tr.entry_ts <= lastTs){
                        indMarkers.push({
                            time: tr.entry_ts,
                            position: 'inBar', // Places it on the data point value
                            color: tr.type === 'LONG' ? '#2962FF' : '#FF6D00',
                            shape: 'circle',
                            size: 1
                        });
                    }
                    // Exit Marker - White Circle
                    if(tr.exit_ts && tr.exit_ts >= firstTs && tr.exit_ts <= lastTs){
                        indMarkers.push({
                            time: tr.exit_ts,
                            position: 'inBar',
                            color: '#FFFFFF', 
                            shape: 'circle',
                            size: 1
                        });
                    }
                });
                s.setMarkers(indMarkers);
            }

            const pi = charts.findIndex(x=>x.id==='sp'+i);
            if(pi>=0) charts[pi].series = s;
        });
    });
}

// === SYNC & RESIZE ===
// TimeScale visible only on last chart
if(charts.length>0) charts[charts.length-1].ch.timeScale().applyOptions({visible:true});

// Chart Sync
charts.forEach(c1=>{
    if(!c1.ch)return;
    c1.ch.timeScale().subscribeVisibleTimeRangeChange(r=>{
        if(syncingCharts || !r) return;
        syncingCharts=true;
        charts.forEach(c2=>{
            if(c2!==c1 && c2.ch) c2.ch.timeScale().setVisibleRange(r);
        });
        syncingCharts=false;
    });
});

// Resize
// Resize observer handled earlier (lines ~1400)
// Removed duplicate to prevent conflicts

// Initial Fit - START AT BEGINNING
setTimeout(()=>{
    charts.forEach(c=>{
        if(c.ch) {
            // Scroll to the start (left)
            // Using logical range to show first ~100 bars
            c.ch.timeScale().setVisibleLogicalRange({ from: 0, to: 150 });
        }
    });
}, 100);

// Global Crosshair & Keyboard Nav
const gl=document.getElementById('globalCrosshair');
const gll=document.getElementById('globalCrosshairLabel');
let lastFocusedTime = null;

function updateCrosshair(param, sourceCh){
   if(!param.time || !param.point){
       gl.style.display='none'; gll.style.display='none';
       return;
   }
   
   // TRACK FOCUS
   lastFocusedTime = param.time;

   gl.style.display='block';
   gl.style.left=param.point.x+'px';
   
   // Label
   gll.style.display='block';
   gll.style.left=param.point.x+'px';
   const date = new Date(param.time*1000);
   gll.textContent = date.toISOString().slice(0,16).replace('T',' ');
   
   // Sync others (Time only - vertical line)
   // Sync others (Time only - vertical line + Horizontal Magnet)
   charts.forEach(c=>{
       if(c.ch!==sourceCh && c.ch && c.series){
           const data = _findDataAt(c.series, param.time);
           if(data){
               const val = data.value !== undefined ? data.value : data.close;
               c.ch.setCrosshairPosition(val, param.time, c.series);
           } else {
               c.ch.clearCrosshairPosition();
           }
       }
   });
   
   // Tooltip Update via Map Lookup (Exact Match)
   const trade = tooltipMap.get(param.time);
   const tt=document.getElementById('tt');
   
   if(trade){
       tt.style.display='block';
       tt.style.left=(param.point.x+20)+'px';
       tt.style.top=(param.point.y+20)+'px';
       
       // Keep tooltip inside window
       const rect = tt.getBoundingClientRect();
       if(rect.right > window.innerWidth) tt.style.left = (param.point.x - rect.width - 20) + 'px';
       
       const cls = trade.pnl>=0?'win':'loss';
       const pnlSign = trade.pnl>=0?'+':'';
       
       tt.innerHTML=`
         <div style="font-weight:bold;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
            <span style="color:${trade.type==='LONG'?'#2962FF':'#FF6D00'}">${trade.type}</span>
            <span class="tt-badge ${cls}">${trade.pnl>=0?'WIN':'LOSS'}</span>
         </div>
         <div class="tt-row"><span class="tt-label">Entry</span><span class="tt-val">$${trade.ep.toFixed(2)}</span></div>
         <div class="tt-row"><span class="tt-label">Exit</span><span class="tt-val">$${trade.xp.toFixed(2)}</span></div>
         <div class="tt-row"><span class="tt-label">Qty</span><span class="tt-val">${trade.qty}</span></div>
         <div class="tt-row"><span class="tt-label">Dur</span><span class="tt-val">${trade.dur}</span></div>
         <div class="tt-row"><span class="tt-label">Fees</span><span class="tt-val" style="color:#D29922">$${trade.fees.toFixed(2)}</span></div>
         <div class="tt-row" style="margin-top:8px;border-top:1px solid #333;padding-top:8px">
            <span class="tt-label">Net PnL</span>
            <span class="tt-val ${trade.pnl>=0?'pos':'neg'}">${pnlSign}$${trade.pnl.toFixed(2)}</span>
         </div>
       `;
   }else{
       tt.style.display='none';
       // Update OHLC header
       const candle = cData.find(c=>c.time===param.time);
       if(candle){
           document.getElementById('tv').textContent=gll.textContent;
           document.getElementById('ov').textContent=candle.open.toFixed(2);
           document.getElementById('hv').textContent=candle.high.toFixed(2);
           document.getElementById('lv').textContent=candle.low.toFixed(2);
           const cEl = document.getElementById('cv');
           cEl.textContent=candle.close.toFixed(2);
           cEl.className = candle.close >= candle.open ? 'ohlc-val up' : 'ohlc-val down';
       }
   }
}

charts.forEach(c=>{
    if(c.ch) c.ch.subscribeCrosshairMove(p=>updateCrosshair(p, c.ch));
});

ct.addEventListener('mouseleave', ()=>{
    gl.style.display='none';
    gll.style.display='none';
    charts.forEach(c=>{if(c.ch)c.ch.clearCrosshairPosition();});
});

// Zoom Controls
document.getElementById('zoomIn').onclick = () => {
    charts.forEach(c=>c.ch.timeScale().applyOptions({barSpacing: c.ch.timeScale().options().barSpacing * 1.2}));
};
document.getElementById('zoomOut').onclick = () => {
    charts.forEach(c=>c.ch.timeScale().applyOptions({barSpacing: c.ch.timeScale().options().barSpacing * 0.8}));
};

// KEYBOARD NAVIGATION
document.addEventListener('keydown', e => {
    if(e.key === 'ArrowLeft' || e.key === 'ArrowRight'){
        e.preventDefault();
        
        let idx = -1;
        if(lastFocusedTime !== null) idx = D.t.indexOf(lastFocusedTime);
        if(idx === -1) idx = D.t.length - 1; // Default to end if no focus
        
        if(e.key === 'ArrowLeft') idx = Math.max(0, idx - 1);
        else idx = Math.min(D.t.length - 1, idx + 1);
        
        const t = D.t[idx];
        const candle = cData[idx];
        if(!candle) return;
        
        // Use Main Chart for coords
        const mainCh = charts[0].ch;
        const mainSer = charts[0].series;
        
        const x = mainCh.timeScale().timeToCoordinate(t);
        const y = mainSer.priceToCoordinate(candle.close);
        
        if(x !== null && y !== null){
            updateCrosshair({time: t, point: {x: x, y: y}}, null);
            mainCh.setCrosshairPosition(candle.close, t, mainSer);
            
            // Auto-Scroll if out of view
            const logIdx = mainCh.timeScale().coordinateToLogical(x);
             // Simple fallback: if coordinate is extremely left or right
             if(x < 50 || x > mainCh.timeScale().width() - 50){
                 mainCh.timeScale().scrollToPosition(- (D.t.length - 1 - idx), true);
             }
        }
    }
});

'''





# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_trades(
    df: DataFrameType,
    df_trades: DataFrameType,
    plot_base: str,
    grafica_rango_personalizado: bool,
    grafica_fecha_inicio: str,
    grafica_fecha_fin: str,
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
        # Robust volume detection (case-insensitive)
        vol_col = next((c for c in df.columns if c.lower() in ("volume", "vol", "v")), None)
        volumes = df[vol_col].to_numpy().astype(np.float64) if vol_col else None
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
        
        # Robust volume detection (case-insensitive)
        vol_col = next((c for c in df_pd.columns if c.lower() in ("volume", "vol", "v")), None)
        volumes = df_pd[vol_col].values.astype(np.float64) if vol_col else None

    # ================== DATE FILTERING ==================
    # Sistema binario: rango personalizado o últimos 2 meses automáticos.
    if grafica_rango_personalizado:
        # ---- MODO MANUAL: usar fechas fijas del usuario ----
        start_pd = pd.to_datetime(grafica_fecha_inicio, utc=True)
        end_pd = pd.to_datetime(grafica_fecha_fin, utc=True)
        start = np.datetime64(start_pd.tz_localize(None))
        end = np.datetime64(end_pd.tz_localize(None))

        if np.issubdtype(timestamps.dtype, np.datetime64):
            ts_compare = timestamps
        else:
            ts_compare = timestamps.astype('datetime64[ns]')

        mask = (ts_compare >= start) & (ts_compare <= end)

        ts_filtered = timestamps[mask]
        o_filtered = opens[mask]
        h_filtered = highs[mask]
        l_filtered = lows[mask]
        c_filtered = closes[mask]
        v_filtered = volumes[mask] if volumes is not None else None

        # Fallback: si el rango manual no cruza el dataset, mostrar todo
        if len(ts_filtered) == 0:
            ts_filtered = timestamps
            o_filtered = opens
            h_filtered = highs
            l_filtered = lows
            c_filtered = closes
            v_filtered = volumes

        timestamps = ts_filtered
        opens = o_filtered
        highs = h_filtered
        lows = l_filtered
        closes = c_filtered
        volumes = v_filtered
    else:
        # ---- MODO AUTO: últimos 2 meses (60 días) del trial ----
        ts_unix = _normalize_timestamps_to_unix(timestamps)
        ts_max = int(ts_unix.max())
        ts_min_auto = ts_max - (60 * 86400)  # 60 días en segundos

        # Solo recortar si el trial tiene más de 60 días de datos
        ts_min_dataset = int(ts_unix.min())
        if ts_min_auto > ts_min_dataset:
            mask = ts_unix >= ts_min_auto
            timestamps = timestamps[mask]
            opens = opens[mask]
            highs = highs[mask]
            lows = lows[mask]
            closes = closes[mask]
            if volumes is not None:
                volumes = volumes[mask]
        # Si el trial tiene <= 60 días, no recortamos nada (se grafica todo)

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
      "c": c_q.tolist()
    }
    
    if vol_q is not None:
        candle_data["vol"] = vol_q.tolist()
        
    candle_data["f"] = float(price_factor)

    # ================== PREPARE TRADES ==================
    # ================== PREPARE TRADES (Clean Data for JS) ==================
    trades_data = {"list": []}
    
    if df_trades is not None and len(df_trades) > 0:
        if not isinstance(df_trades, pd.DataFrame):
             try:
                 df_trades = df_trades.to_pandas()
             except:
                 pass

        # Normalize column names if needed
        cols = df_trades.columns
        entry_col = "timestamp_entry" if "timestamp_entry" in cols else "entry_time"
        exit_col = "timestamp_exit" if "timestamp_exit" in cols else "exit_time"
        trail_act_time_col = "trail_act_time"
        trail_act_price_col = "trail_act_price"
        
        if entry_col in cols:
            # Snap timestamps to candles
            for _, tr in df_trades.iterrows():
                try:
                    t_entry = tr.get(entry_col)
                    t_exit = tr.get(exit_col)
                    
                    if pd.isna(t_entry) or pd.isna(t_exit):
                        continue
                        
                    ts_entry_raw = int(t_entry.timestamp()) if hasattr(t_entry, 'timestamp') else int(t_entry)
                    ts_exit_raw = int(t_exit.timestamp()) if hasattr(t_exit, 'timestamp') else int(t_exit)
                    
                    # Snap to nearest candle
                    idx_entry = np.searchsorted(ts_q, ts_entry_raw, side='right') - 1
                    idx_exit = np.searchsorted(ts_q, ts_exit_raw, side='right') - 1
                    
                    # Store both raw and snapped (if valid)
                    # Relax validation: allow marking even if slightly out of bounds if visible
                    entry_valid = 0 <= idx_entry < len(ts_q)
                    exit_valid = 0 <= idx_exit < len(ts_q)
                    
                    # Use snapped time if valid, else ignore for marker purposes
                    ts_entry = int(ts_q[idx_entry]) if entry_valid else None
                    ts_exit = int(ts_q[idx_exit]) if exit_valid else None

                    # Trailing Activation Logic
                    ts_ta = None
                    p_ta = None
                    if trail_act_time_col in tr and trail_act_price_col in tr:
                        t_ta = tr.get(trail_act_time_col)
                        val_ta = tr.get(trail_act_price_col)
                        
                        if pd.notna(t_ta) and pd.notna(val_ta):
                            ts_ta_raw = int(t_ta.timestamp()) if hasattr(t_ta, 'timestamp') else int(t_ta)
                            idx_ta = np.searchsorted(ts_q, ts_ta_raw, side='right') - 1
                            if 0 <= idx_ta < len(ts_q):
                                ts_ta = int(ts_q[idx_ta])
                                p_ta = float(val_ta)
                    
                    if ts_entry is None and ts_exit is None:
                        continue 
                        
                    # Prepare Trade Object
                    # Handle aliases for other columns
                    ep = tr.get("price_entry") if "price_entry" in tr else tr.get("entry_price", 0)
                    xp = tr.get("price_exit") if "price_exit" in tr else tr.get("exit_price", 0)
                    fees = tr.get("commission") if "commission" in tr else tr.get("comision", 0)
                    
                    side = str(tr.get("type", "LONG")).upper()
                    pnl = float(tr.get("pnl_neto", 0))
                    
                    # Duration
                    dur_s = ts_exit_raw - ts_entry_raw # Use raw for duration accuracy
                    if dur_s < 60: dur_str = f"{dur_s}s"
                    elif dur_s < 3600: dur_str = f"{int(dur_s/60)}m"
                    else: dur_str = f"{int(dur_s/3600)}h {int((dur_s%3600)/60)}m"
                    
                    trade_obj = {
                        "type": side,
                        "entry_ts": ts_entry, # Nullable
                        "exit_ts": ts_exit,   # Nullable
                        "ep": float(ep),
                        "xp": float(xp),
                        "qty": float(tr.get("qty", 0)),
                        "pnl": pnl,
                        "fees": float(fees),
                        "dur": dur_str,
                        "win": pnl >= 0,
                        "ta_ts": ts_ta,
                        "ta_p": p_ta
                    }
                    trades_data["list"].append(trade_obj)
                    
                except Exception:
                    continue

    # ================== INDICATORS ==================
    # Detect
    price_range = (float(np.min(l_q))/price_factor, float(np.max(h_q))/price_factor)
    indicators_meta = _detect_indicators(df, price_range, params)
    
    indicators_data = {"overlays": [], "sub_panels": []}
    
    mapper = StrictAlignmentMapper(ts_q)
    
    is_pl = HAS_POLARS and isinstance(df, pl.DataFrame)

    # Extract full-DF timestamps (source) for indicator alignment against ts_q (target)
    if is_pl:
        _ind_ts_raw = df["timestamp"].to_numpy() if "timestamp" in df.columns else None
    else:
        df_pd_ind = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        if isinstance(df_pd_ind.index, pd.DatetimeIndex):
            _ind_ts_raw = df_pd_ind.index.values
        elif "timestamp" in df_pd_ind.columns:
            _ind_ts_raw = df_pd_ind["timestamp"].values
        else:
            _ind_ts_raw = None

    if _ind_ts_raw is not None:
        indicator_ts = _normalize_timestamps_to_unix(_ind_ts_raw)
    else:
        indicator_ts = ts_q  # Fallback: assume df is already sliced to match
    
    def get_col_values(c):
        if is_pl:
            return df[c].to_numpy().astype(np.float64)
        else:
            return df[c].values.astype(np.float64)

    # Overlays
    for ov in indicators_meta["overlays"]:
        col = ov["col"]
        if col in df.columns:
            vals = get_col_values(col)
            q_vals, factor, _ = mapper.align_quantized(indicator_ts, vals, precision=ov["precision"])

            
            indicators_data["overlays"].append({
                "t": ts_q.tolist(),
                "v": q_vals,
                "f": factor,
                "color": ov["color"],
                "name": col,
                "type": ov["type"]
            })

    # Sub-panels
    grouped_panels = {}
    for panel in indicators_meta["sub_panels"]:
        col = panel["col"]
        if col in df.columns:
            vals = get_col_values(col)
            q_vals, factor, _ = mapper.align_quantized(indicator_ts, vals, precision=panel["precision"])
            
            pid = str(panel.get("panel_id")) if panel.get("panel_id") is not None else f"auto_{col}"
            
            if pid not in grouped_panels:
                grouped_panels[pid] = { "title": panel["name"], "series": [] }
            else:
                grouped_panels[pid]["title"] += f" / {panel['name']}"

            grouped_panels[pid]["series"].append({
                "name": panel["name"],
                "color": panel["color"],
                "type": panel["type"],
                "bounds": panel["bounds"],
                "zero_line": (panel["bounds"] and panel["bounds"].get("mid") == 0),
                "data": { "t": ts_q.tolist(), "v": q_vals, "f": factor }
            })
            
    indicators_data["sub_panels"] = list(grouped_panels.values())

    # ================== WRITE HTML ==================
    import os
    if not os.path.exists(plot_base):
        try:
            os.makedirs(plot_base)
        except:
            pass
            
    # Clean filename
    clean_combo = "".join(c for c in str(combo) if c.isalnum() or c in ('_','-'))
    filename = f"TRIAL {trial_number} - {int(score)}.html"
    filepath = os.path.join(plot_base, filename)
    
    # Try using metric get helper or safe get
    def _g(k): 
        return metrics.get(k, 0) if metrics else 0

    _write_html_streaming(
        filepath,
        candle_data,
        indicators_data,
        trades_data,
        config={
            "activo": str(activo or "ASSET"),
            "combo": str(combo),
            "trial": str(trial_number),
            "total_trades": len(df_trades) if df_trades is not None else 0,
            "winrate": _g("winrate"),
            "pnl_neto": _g("pnl_neto"),
            "score": score
        }
    )
    
    return filepath


# =============================================================================
# PLOT REPORTER
# =============================================================================

class PlotReporter:
    """
    Reporter that generates interactive HTML charts for top trials.
    Integrated with the Runner system.

    NOTA: Los gráficos se generan AL FINAL de la optimización (on_strategy_end),
    igual que el Excel. Durante los trials solo se acumulan los artefactos
    de los mejores candidatos en memoria.
    """
    def __init__(
        self,
        plot_base: str,
        grafica_rango_personalizado: bool,
        grafica_fecha_inicio: str,
        grafica_fecha_fin: str,
        max_archivos: int = 5,
        saldo_inicial: float = 1000.0,
        activo: Optional[str] = None,
    ):
        self.plot_base = plot_base
        self.grafica_rango_personalizado = grafica_rango_personalizado
        self.grafica_fecha_inicio = grafica_fecha_inicio
        self.grafica_fecha_fin = grafica_fecha_fin
        self.max_archivos = max_archivos
        self.saldo_inicial = saldo_inicial
        self.activo = activo

        # Candidatos acumulados durante la optimización.
        # Cada entrada: {score, trial, artifacts}
        # Los ficheros HTML NO se generan hasta on_strategy_end.
        self._pending: list = []
        self.min_score = float("-inf")

    def needs_dataframe(self, score: float) -> bool:
        """
        Determina si necesitamos el DataFrame de señales para este trial.
        Devuelve True si:
        1. No hemos llenado el buffer de max_archivos todavía.
        2. O el score supera al peor candidato actual.
        """
        if len(self._pending) < self.max_archivos:
            return True
        return score > self.min_score

    def on_trial_end(self, artifacts) -> None:
        """
        Acumula el artefacto si es candidato. NO escribe nada en disco.
        """
        if not self.needs_dataframe(artifacts.score):
            return

        if artifacts.df_signals is None or len(artifacts.df_signals) == 0:
            return

        # Guardar artefacto en memoria
        self._pending.append({
            "score": artifacts.score,
            "trial": artifacts.trial_number,
            "artifacts": artifacts,
        })

        # Mantener solo los top-N candidatos
        if len(self._pending) > self.max_archivos:
            self._pending.sort(key=lambda x: x["score"], reverse=True)
            self._pending = self._pending[: self.max_archivos]

        # Actualizar umbral mínimo
        self.min_score = min(c["score"] for c in self._pending)

    def on_strategy_end(self, strategy_name: str, study) -> None:
        """
        Genera los ficheros HTML para los mejores candidatos acumulados.
        Se llama UNA SOLA VEZ al terminar la optimización.
        """
        if not self._pending:
            return

        # Ordenar de mejor a peor para escribir en ese orden
        self._pending.sort(key=lambda x: x["score"], reverse=True)

        for entry in self._pending:
            artifacts = entry["artifacts"]
            try:
                plot_trades(
                    df=artifacts.df_signals,
                    df_trades=artifacts.trades,
                    plot_base=self.plot_base,
                    grafica_rango_personalizado=self.grafica_rango_personalizado,
                    grafica_fecha_inicio=self.grafica_fecha_inicio,
                    grafica_fecha_fin=self.grafica_fecha_fin,
                    trial_number=artifacts.trial_number,
                    params=artifacts.params,
                    score=artifacts.score,
                    combo=artifacts.strategy_name,
                    metrics=artifacts.metrics,
                    equity_curve=artifacts.equity_curve,
                    saldo_inicial=self.saldo_inicial,
                    activo=self.activo,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error generando gráfico del trial {artifacts.trial_number}: {e}")

        # Limpiar pendientes
        self._pending.clear()
        self.min_score = float("-inf")


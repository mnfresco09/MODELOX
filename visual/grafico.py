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
  # Paleta institucional formal — inspirada en terminales Bloomberg/Refinitiv
  "#6B9BD2",  # Institutional Blue
  "#7CB4B8",  # Teal Mist
  "#B8A9C9",  # Soft Lavender
  "#D4A574",  # Warm Caramel
  "#8FB8A0",  # Sage Green
  "#C49B9B",  # Dusty Rose
  "#A0B4CC",  # Steel Blue
  "#C4B07B",  # Antique Gold
  "#9BAEB7",  # Mineral Gray
  "#B09FC4",  # Wisteria
  "#7CAFC2",  # Ocean Teal
  "#C9A87C",  # Sand
  "#8DA7BE",  # Glacier Blue
  "#B5C48B",  # Willow Green
  "#C4948E",  # Terra Cotta
  "#A3B5C8",  # Powder Blue
  "#B8B394",  # Khaki Stone
  "#9EADBA",  # Cloud Gray
  "#BAA5B0",  # Mauve Ash
  "#88A8B8",  # Fjord Blue
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
  """Heurística robusta: overlay solo si el indicador está claramente en escala precio.

  Usa percentiles (P2/P98) para evitar que outliers den falsos positivos.
  Un overlay real (MA, ALMA, bandas Bollinger) debe:
    - Tener magnitud similar al precio.
    - No ser negativo cuando el precio es positivo y alto.
    - Tener un rango proporcional al rango de precio.
  """
  try:
    s = series.dropna()
    if len(s) < 5:
      return False
    min_p, max_p = float(price_range[0]), float(price_range[1])
    if not (np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p):
      return False

    # Usar percentiles para robustez ante outliers
    ind_p2  = float(np.percentile(s, 2))
    ind_p98 = float(np.percentile(s, 98))
    if not (np.isfinite(ind_p2) and np.isfinite(ind_p98)):
      return False

    # REGLA 1: Negativo con precio alto → oscilador, no overlay
    if ind_p2 < 0 and min_p > 50:
      return False

    ind_span   = ind_p98 - ind_p2
    price_span = max_p - min_p

    # REGLA 2: Rango demasiado pequeño relativo al precio → no overlay
    # (z-score -3..+3 con BTC en 30k sería ind_span=6, price_span=5000 → ratio 0.001)
    span_ratio = ind_span / price_span if price_span > 0 else 0
    if span_ratio < 0.03:
      return False

    # REGLA 3: Overlay real no debe exceder mucho el rango de precio
    if span_ratio > 3.0:
      return False

    # REGLA 4: Los valores deben estar sustancialmente dentro del rango de precio
    margin = 0.25 * price_span
    within_min = ind_p2  >= min_p - margin
    within_max = ind_p98 <= max_p + margin

    return within_min and within_max
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
    "ichimoku", "tenkan", "kijun", "chikou", "span", "cloud",
    "hull", "lsma", "linreg", "lwma", "smma", "tma", "zlsma",
  )

  # Sufijos/patrones que indican indicadores derivados → NUNCA son overlay
  NON_OVERLAY_SUFFIXES = (
    "_acc", "_vel", "_diff", "_roc", "_signal", "_sig", "_hist",
    "_zscore", "_z", "_norm", "_pct", "_ratio", "_slope", "_delta",
    "_momentum", "_mom", "_osc", "_divergence", "_div", "_pct_rank",
    "_percentile", "_rank", "_score", "_raw",
  )
  # Nombres que NUNCA son overlay (osciladores conocidos)
  NON_OVERLAY_NAMES = (
    "fisher", "rsi", "mfi", "cci", "stoch", "macd", "adx", "atr",
    "obv", "cmf", "willr", "dpo", "trix", "roc", "momentum",
    "zscore", "z_score", "psar", "ppo", "kst", "aroon", "chop",
    "bbw", "bbp", "squeeze", "regime", "trend_strength",
    "hist", "oscillator", "divergence",
  )

  def _normalize_panel(panel_value: Any) -> Any:
    """Normaliza alias de panel sin tocar lógica de cálculo.

    Convenciones soportadas:
    - "main", "price", "overlay" -> "overlay"
    - "sub", "sub1", "sub2", ... -> se mantienen para agrupación
    - int -> se mantiene (panel numérico explícito)
    """
    if panel_value is None:
      return None
    if isinstance(panel_value, str):
      p = panel_value.strip().lower()
      if p in {"main", "price", "overlay"}:
        return "overlay"
      return panel_value
    if isinstance(panel_value, int):
      return panel_value
    return None

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
    panel = _normalize_panel(spec.get("panel", None))

    # PRIORIDAD 1: Bounds (Si no hay specs, mirar si bounds define panel)
    if panel is None:
      col_bounds = bounds_map.get(col, None)
      if isinstance(col_bounds, dict) and "panel" in col_bounds:
        panel = _normalize_panel(col_bounds["panel"])

    # PRIORIDAD 2: Heurísticas (Solo si no se definió explícitamente)
    if panel is None:
      col_lower = col.lower()
      if _is_overlay_by_name(col_lower):
        panel = "overlay"
      else:
        panel = "overlay" if _is_overlay_heuristic(df[col], price_range) else "sub"

    # Compat: las estrategias usan "tipo" históricamente.
    series_type = str(spec.get("type", spec.get("tipo", "line")))
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
# DYNAMIC HTML GENERATOR (v8.0 - TEMPLATE-BASED)
# =============================================================================

_CHART_TEMPLATE: Optional[str] = None


def _load_chart_template() -> str:
    """Load chart_template.html from the same directory as this module (cached)."""
    global _CHART_TEMPLATE
    if _CHART_TEMPLATE is None:
        tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_template.html")
        with open(tpl_path, "r", encoding="utf-8") as f:
            _CHART_TEMPLATE = f.read()
    return _CHART_TEMPLATE


def _write_html_streaming(
    filepath: str,
    candle_data: dict,
    indicators: dict,
    trades: dict,
    config: dict,
    equity_data: Optional[dict] = None,
) -> None:
    """
    Template-based HTML generator (v8.0).
    Reads chart_template.html and injects JSON data via %%INJECT_X%% markers.
    """

    cfg = {
        "activo":        str(config.get("activo", "")),
        "combo":         str(config.get("combo", "")),
        "trial":         str(config.get("trial", "")),
        "total_trades":  int(config.get("total_trades", 0)),
        "winrate":       float(config.get("winrate", 0)),
        "pnl_neto":      float(config.get("pnl_neto", 0)),
        "roi":           float(config.get("roi", 0)),
        "max_dd":        float(config.get("max_dd", 0)),
        "pf":            float(config.get("profit_factor", 0)),
        "expectancy":    float(config.get("expectancy", 0)),
        "score":         float(config.get("score", 0)),
        "avg_win":       float(config.get("avg_win", 0)),
        "avg_loss":      float(config.get("avg_loss", 0)),
    }
    eq = equity_data if equity_data else {"v": [], "t": [], "si": 0}

    MARKERS = [
        "%%INJECT_D%%", "%%INJECT_I%%", "%%INJECT_T%%",
        "%%INJECT_E%%", "%%INJECT_CFG%%",
    ]
    payloads = [candle_data, indicators, trades, eq, cfg]

    template = _load_chart_template()
    with open(filepath, "wb") as f:
        for marker, payload in zip(MARKERS, payloads):
            idx = template.index(marker)
            f.write(template[:idx].encode("utf-8"))
            f.write(_dumps_bytes(payload))
            template = template[idx + len(marker):]
        f.write(template.encode("utf-8"))

# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_trades(
    df: DataFrameType,
    df_trades: DataFrameType,
    plot_base: str,
    grafica_rango_personalizado,  # True = manual | False = 2 meses | "all" = 100%
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
    # Tres modos: "all" = 100% | True = fechas manuales | False = últimos 2 meses
    _modo = str(grafica_rango_personalizado).lower()

    if _modo == "all":
        # ---- MODO ALL: mostrar 100% del rango del trial, sin recorte ----
        pass  # timestamps/opens/etc. ya están completos

    elif _modo == "true":
        # ---- MODO MANUAL: usar fechas fijas del usuario ----
        start_pd = pd.to_datetime(grafica_fecha_inicio, utc=True)
        end_pd = pd.to_datetime(grafica_fecha_fin, utc=True)
        start = np.datetime64(start_pd.tz_convert(None))
        end = np.datetime64(end_pd.tz_convert(None))

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

    # ================== PREPARE EQUITY ==================
    # Instead of blindly spreading the equity array points across the entire
    # chart (which causes diagonal lines during no-trade periods like market regimes),
    # we reconstruct precise equity lines using trade exit timestamps.
    equity_out = None
    if df_trades is not None:
        try:
            records = df_trades.to_dicts() if hasattr(df_trades, "to_dicts") else df_trades.to_dict(orient="records")
            if len(records) > 0 and len(ts_q) > 0:
                eq_ts = [int(ts_q[0])]
                eq_vals = [float(saldo_inicial)]
                for tr in records:
                    t_raw = tr.get("exit_time")
                    if t_raw is None:
                        continue
                    try:
                        if pd.isnull(t_raw):
                            continue
                    except (TypeError, ValueError):
                        pass
                    try:
                        if hasattr(t_raw, "timestamp"):
                            ts = int(t_raw.timestamp())
                        else:
                            ts = int(pd.Timestamp(t_raw).timestamp())
                        # Snap to closest chart timestamp
                        idx = np.searchsorted(ts_q, ts)
                        if idx >= len(ts_q): idx = len(ts_q) - 1
                        eq_ts.append(int(ts_q[idx]))
                        eq_vals.append(float(tr.get("saldo_despues", eq_vals[-1])))
                    except Exception:
                        continue
                
                # Make the line continue visually to the right edge of the chart
                if eq_ts[-1] < int(ts_q[-1]):
                    eq_ts.append(int(ts_q[-1]))
                    eq_vals.append(eq_vals[-1])
                    
                equity_out = {"t": eq_ts, "v": eq_vals, "si": float(saldo_inicial)}
            else:
                # No trades -> Flat equity from start to end
                if len(ts_q) > 0:
                    equity_out = {
                        "t": [int(ts_q[0]), int(ts_q[-1])], 
                        "v": [float(saldo_inicial), float(saldo_inicial)], 
                        "si": float(saldo_inicial)
                    }
        except Exception:
            pass

    # ================== WRITE HTML ==================
    import os
    if not os.path.exists(plot_base):
        try:
            os.makedirs(plot_base)
        except:
            pass

    filename = f"TRIAL {trial_number} - {int(score)}.html"
    filepath = os.path.join(plot_base, filename)

    def _g(k):
        return metrics.get(k, 0) if metrics else 0

    _write_html_streaming(
        filepath,
        candle_data,
        indicators_data,
        trades_data,
        config={
            "activo":        str(activo or "ASSET"),
            "combo":         str(combo),
            "trial":         str(trial_number),
            "total_trades":  len(df_trades) if df_trades is not None else 0,
            "winrate":       _g("winrate") or _g("WINRATE_PCT") or _g("PORC_GANADORAS"),
            "pnl_neto":      _g("pnl_neto") or _g("PNL_NETO"),
            "score":         score,
            "max_dd":        _g("max_dd") or _g("MAX_DD_PCT"),
            "profit_factor": _g("profit_factor") or _g("PROFIT_FACTOR"),
            "roi":           _g("roi") or _g("ROI_PCT"),
            "expectancy":    _g("expectancy") or _g("EXPECTANCY"),
            "avg_win":       _g("avg_win") or _g("AVG_WIN"),
            "avg_loss":      _g("avg_loss") or _g("AVG_LOSS"),
        },
        equity_data=equity_out,
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
        grafica_rango_personalizado,  # True = manual | False = 2 meses | "all" = 100%
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


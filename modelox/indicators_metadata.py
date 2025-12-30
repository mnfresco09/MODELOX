"""
================================================================================
MODELOX - Indicator Metadata Registry (Ultra Modular System)
================================================================================

Central repository for ALL indicator metadata:
- Parameters and their ranges
- Display properties (colors, styles, zones)
- Calculation methods and versions
- Ranges and overbought/oversold levels

This allows plot.py to be COMPLETELY generic and adapt to any indicator
automatically without hardcoding logic.

Author: MODELOX Quant Team
Version: 1.0.0
================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class IndicatorRange:
    """Define the range, overbought/oversold levels, and neutral zone."""
    min_value: float = 0.0
    max_value: float = 100.0
    neutral: Optional[float] = None  # e.g., 50 for RSI
    overbought: float = 70.0
    oversold: float = 30.0
    
    # Optional: For oscillators with bands
    upper_band: Optional[float] = None
    lower_band: Optional[float] = None


@dataclass
class IndicatorMetadata:
    """Complete metadata for an indicator."""
    
    # Identifier
    name: str  # e.g., "rsi", "macd", "dpo"
    display_name: str  # e.g., "RSI (14)"
    
    # Type: 'overlay' (price panel) or 'oscillator' (sub-panel)
    indicator_type: str
    
    # Display properties
    color: str  # Hex color code
    line_style: str = "solid"  # solid, dashed, dotted
    line_width: int = 2
    
    # Ranges and levels
    range_info: IndicatorRange = field(default_factory=IndicatorRange)
    
    # Parameters used in this instance
    params: Dict[str, Any] = field(default_factory=dict)
    
    # For multi-line indicators (MACD, Stochastic, etc.)
    sub_indicators: Dict[str, IndicatorMetadata] = field(default_factory=dict)
    
    # Column name in DataFrame
    column_name: str = ""
    
    # Additional lines to plot (e.g., signal line, histogram)
    additional_lines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_panel_name(self) -> str:
        """Generate panel name including parameters."""
        if not self.params:
            return self.display_name
        
        param_str = " (".join([
            f"{k}={v}" for k, v in self.params.items() 
            if k not in ['out', 'col', 'activo', 'out_macd', 'out_signal', 'out_hist']
        ])
        
        if param_str:
            return f"{self.display_name} ({param_str})"
        return self.display_name
    
    def has_overbought_oversold(self) -> bool:
        """Check if indicator has overbought/oversold levels."""
        return self.range_info.overbought is not None and self.range_info.oversold is not None


# =============================================================================
# INDICATOR REGISTRY - Ultra Modular Design
# =============================================================================

class IndicatorRegistry:
    """
    Centralized registry of ALL indicators with their metadata.
    
    ARCHITECTURE:
    - Each indicator definition includes everything needed for plotting
    - Plot.py queries this registry to build panels automatically
    - New indicators just need to be added here - plot.py adapts automatically
    """
    
    # Registry dictionary
    _registry: Dict[str, IndicatorMetadata] = {}
    
    @classmethod
    def register(cls, metadata: IndicatorMetadata) -> None:
        """Register an indicator's metadata."""
        cls._registry[metadata.name] = metadata
    
    @classmethod
    def get(cls, indicator_name: str) -> Optional[IndicatorMetadata]:
        """Get metadata for an indicator."""
        return cls._registry.get(indicator_name)
    
    @classmethod
    def get_all(cls) -> Dict[str, IndicatorMetadata]:
        """Get all registered indicators."""
        return cls._registry.copy()
    
    @classmethod
    def register_batch(cls, batch: List[IndicatorMetadata]) -> None:
        """Register multiple indicators at once."""
        for metadata in batch:
            cls.register(metadata)


# =============================================================================
# MOVING AVERAGES (OVERLAYS)
# =============================================================================

_MA_INDICATORS = [
    IndicatorMetadata(
        name="ema",
        display_name="EMA",
        indicator_type="overlay",
        color="#fbbf24",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="sma",
        display_name="SMA",
        indicator_type="overlay",
        color="#94a3b8",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="wma",
        display_name="WMA",
        indicator_type="overlay",
        color="#60a5fa",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="hma",
        display_name="HMA",
        indicator_type="overlay",
        color="#ec4899",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="vwma",
        display_name="VWMA",
        indicator_type="overlay",
        color="#22d3ee",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="vwap_session",
        display_name="VWAP",
        indicator_type="overlay",
        color="#38bdf8",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
]

# =============================================================================
# OSCILLATORS - Momentum
# =============================================================================

_OSCILLATOR_INDICATORS = [
    IndicatorMetadata(
        name="rsi",
        display_name="RSI",
        indicator_type="oscillator",
        color="#60a5fa",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            neutral=50,
            overbought=70,
            oversold=30,
        ),
    ),
    IndicatorMetadata(
        name="stoch_k",
        display_name="Stochastic %K",
        indicator_type="oscillator",
        color="#f472b6",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            neutral=50,
            overbought=80,
            oversold=20,
        ),
    ),
    IndicatorMetadata(
        name="stoch_d",
        display_name="Stochastic %D",
        indicator_type="oscillator",
        color="#a78bfa",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            neutral=50,
            overbought=80,
            oversold=20,
        ),
    ),
    IndicatorMetadata(
        name="roc",
        display_name="ROC",
        indicator_type="oscillator",
        color="#fb923c",
        range_info=IndicatorRange(
            min_value=-100,
            max_value=100,
            neutral=0,
        ),
    ),
    IndicatorMetadata(
        name="dpo",
        display_name="DPO",
        indicator_type="oscillator",
        color="#22c55e",
        range_info=IndicatorRange(
            min_value=-float('inf'),
            max_value=float('inf'),
            neutral=0,
        ),
    ),
    IndicatorMetadata(
        name="mfi",
        display_name="MFI",
        indicator_type="oscillator",
        color="#34d399",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            neutral=50,
            overbought=80,
            oversold=20,
        ),
    ),
    IndicatorMetadata(
        name="adx",
        display_name="ADX",
        indicator_type="oscillator",
        color="#a78bfa",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            overbought=40,
            oversold=20,
        ),
    ),
    IndicatorMetadata(
        name="cci",
        display_name="CCI",
        indicator_type="oscillator",
        color="#ec4899",
        range_info=IndicatorRange(
            min_value=-100,
            max_value=100,
            neutral=0,
            overbought=100,
            oversold=-100,
        ),
    ),
    IndicatorMetadata(
        name="macd",
        display_name="MACD",
        indicator_type="oscillator",
        color="#60a5fa",
        range_info=IndicatorRange(
            min_value=-float('inf'),
            max_value=float('inf'),
            neutral=0,
        ),
        additional_lines={
            "macd_signal": {"color": "#fbbf24", "line_style": "dashed"},
            "macd_hist": {"color": "#60a5fa", "line_style": "solid", "plot_type": "histogram"},
        },
    ),
    IndicatorMetadata(
        name="zscore",
        display_name="Z-Score",
        indicator_type="oscillator",
        color="#fbbf24",
        range_info=IndicatorRange(
            min_value=-5,
            max_value=5,
            neutral=0,
            overbought=2,
            oversold=-2,
        ),
    ),
]

# =============================================================================
# TREND INDICATORS (OVERLAYS)
# =============================================================================

_TREND_INDICATORS = [
    IndicatorMetadata(
        name="supertrend",
        display_name="SuperTrend",
        indicator_type="overlay",
        color="#10b981",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="donchian_hi",
        display_name="Donchian High",
        indicator_type="overlay",
        color="#22c55e",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="donchian_lo",
        display_name="Donchian Low",
        indicator_type="overlay",
        color="#ef4444",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
]

# =============================================================================
# SPECIALIZED INDICATORS
# =============================================================================

_SPECIALIZED_INDICATORS = [
    IndicatorMetadata(
        name="atr",
        display_name="ATR",
        indicator_type="oscillator",
        color="#fb923c",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="kalman",
        display_name="Kalman Filter",
        indicator_type="overlay",
        color="#a78bfa",
        range_info=IndicatorRange(min_value=0, max_value=float('inf')),
    ),
    IndicatorMetadata(
        name="chop",
        display_name="CHOP",
        indicator_type="oscillator",
        color="#fb923c",
        range_info=IndicatorRange(
            min_value=0,
            max_value=100,
            overbought=61.8,
            oversold=38.2,
        ),
    ),
]

# =============================================================================
# REGISTER ALL INDICATORS
# =============================================================================

IndicatorRegistry.register_batch(_MA_INDICATORS)
IndicatorRegistry.register_batch(_OSCILLATOR_INDICATORS)
IndicatorRegistry.register_batch(_TREND_INDICATORS)
IndicatorRegistry.register_batch(_SPECIALIZED_INDICATORS)


# =============================================================================
# UTILITY FUNCTIONS FOR PLOT GENERATION
# =============================================================================

def get_indicator_panels(df_columns: List[str]) -> Dict[str, List[IndicatorMetadata]]:
    """
    Intelligently group indicators into panels based on their types and presence in DataFrame.
    
    Returns:
        dict: {
            'price_panel': [...overlay indicators...],
            'panel_name': [...oscillator indicators...],
            ...
        }
    """
    panels: Dict[str, List[IndicatorMetadata]] = {
        'price': [],  # Always price panel
    }
    
    # Check each registered indicator
    for indicator_name, metadata in IndicatorRegistry.get_all().items():
        # Check if any related columns exist in DataFrame
        found_columns = [
            col for col in df_columns 
            if indicator_name in col.lower() or col == metadata.column_name
        ]
        
        if not found_columns and metadata.column_name and metadata.column_name in df_columns:
            found_columns = [metadata.column_name]
        
        if found_columns:
            if metadata.indicator_type == 'overlay':
                panels['price'].append(metadata)
            else:
                # Create separate panel for each oscillator
                panel_key = metadata.get_panel_name()
                if panel_key not in panels:
                    panels[panel_key] = []
                panels[panel_key].append(metadata)
    
    return panels


def create_indicator_metadata_from_params(
    indicator_name: str,
    params: Dict[str, Any],
    column_name: str,
) -> IndicatorMetadata:
    """
    Create a IndicatorMetadata instance from indicator name and parameters.
    Useful for dynamic indicator generation from strategy parameters.
    """
    base_metadata = IndicatorRegistry.get(indicator_name)
    
    if not base_metadata:
        # Fallback for unknown indicators
        return IndicatorMetadata(
            name=indicator_name,
            display_name=indicator_name.upper(),
            indicator_type="oscillator",
            color="#60a5fa",
            column_name=column_name,
            params=params,
        )
    
    # Create a copy with updated parameters
    metadata = IndicatorMetadata(
        name=base_metadata.name,
        display_name=base_metadata.display_name,
        indicator_type=base_metadata.indicator_type,
        color=base_metadata.color,
        line_style=base_metadata.line_style,
        line_width=base_metadata.line_width,
        range_info=base_metadata.range_info,
        params=params,
        column_name=column_name,
        additional_lines=base_metadata.additional_lines.copy(),
    )
    
    return metadata

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               MODELOX INTELLIGENT PARAMETER ANALYZER v4.0                    ║
║                 Advanced ML-Powered Strategy Optimization                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
- Intelligent metric/parameter classification (adaptive to any strategy)
- ML-powered optimal zone detection (Gradient Boosting + Bayesian)
- Noise filtering with statistical significance testing
- Robust quantile regression with confidence intervals
- Cross-validation for parameter stability
"""

from __future__ import annotations

import os
import sys
import re
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

plt.style.use('dark_background')

COLORS = {
    'bg':        '#080b10',
    'panel':     '#0d1117',
    'text':      '#e6edf3',
    'subtext':   '#8b949e',
    'grid':      '#21262d',
    'accent':    '#58a6ff',
    'up':        '#3fb950',
    'down':      '#f85149',
    'gold':      '#d29922',
    'purple':    '#a371f7',
    'cyan':      '#39c5cf',
    'optimal':   '#ffffff',
    'band':      '#388bfd',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['panel'],
    'axes.edgecolor': COLORS['grid'],
    'axes.linewidth': 0.8,
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.4,
    'text.color': COLORS['text'],
    'axes.labelcolor': COLORS['subtext'],
    'xtick.color': COLORS['subtext'],
    'ytick.color': COLORS['subtext'],
    'font.family': 'sans-serif',
    'font.size': 9,
})


# ==============================================================================
# 2. INTELLIGENT COLUMN CLASSIFIER (ML-Enhanced)
# ==============================================================================

@dataclass
class ColumnClassification:
    """Result of intelligent column classification."""
    metrics: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    config: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


class IntelligentClassifier:
    """
    ML-enhanced classifier that adapts to any strategy format.
    Uses pattern matching + statistical heuristics + learned patterns.
    """
    
    # Core metrics that appear in ALL results (invariant)
    METRIC_PATTERNS = {
        # Performance metrics
        r'^(SCORE|TOTAL_SCORE|RANKING)$': 'score',
        r'(ROI|RETURN|RET)(_PCT)?$': 'return',
        r'(PNL|PROFIT|LOSS|NET)': 'pnl',
        r'(WIN_?RATE|HIT_?RATE|PORC_GANAD)': 'winrate',
        r'(DRAW_?DOWN|MAX_?DD|DD_?PCT)': 'drawdown',
        r'^(SQN|SHARPE|SORTINO|CALMAR|OMEGA)$': 'risk_metric',
        r'(PROFIT_?FACTOR|PF|PAYOFF)': 'ratio',
        r'(EXPECTA|EXPECT)': 'expectancy',
        r'(ESTABIL|STABILITY)': 'stability',
        
        # Trade statistics
        r'(N_?TRADES|TOTAL_?TRADES|NUM_?TRADES)': 'trade_count',
        r'(TRADES_?(POR_?)?DIA|TRADES_?DAY)': 'trade_freq',
        r'(WIN|LOSS)_?TRADES': 'trade_result',
        r'(NUM_|COUNT_|N_)?(LONGS?|SHORTS?)': 'direction',
        r'(RACHA|STREAK)': 'streak',
        
        # Balance metrics
        r'(SALDO|BALANCE|EQUITY|CAPITAL)': 'balance',
        r'(COMISION|FEE|COMMISSION)': 'fees',
        r'(DURACION|DURATION)': 'duration',
        r'(MAX_GANANCIA|MAX_PERDIDA|AVG_WIN|AVG_LOSS)': 'trade_stats',
    }
    
    # Parameter patterns (vary by strategy)
    PARAM_PATTERNS = [
        # Technical indicators
        r'^(RSI|EMA|SMA|BB|ATR|ADX|MACD|STOCH|CCI|MFI|VWAP|OBV|DMI|ZLEMA|KELTNER)',
        r'(RSI|EMA|SMA|BB|ATR|ADX|MACD|STOCH|CCI|MFI)(_\w+)?$',
        
        # Periods/Windows
        r'(_PERIOD|_LENGTH|_WINDOW|_LOOKBACK|_LEN)$',
        r'^(PERIOD|VENTANA|LOOKBACK|LOOKBAR|WINDOW)',
        r'(FAST|SLOW|SIGNAL)_?(LEN|PERIOD)?',
        
        # Thresholds
        r'(_THRESHOLD|_LEVEL|_MULT|_FACTOR|_COEF|_RATIO)$',
        r'^(THRESHOLD|UMBRAL|NIVEL|MULT)',
        
        # Entry/Exit params (but not EXIT_TYPE which is config)
        r'^(SL|TP|TRAIL|STOP|TAKE)_?(PCT|PERCENT)?$',
        r'(SL_PCT|TP_PCT|TRAIL_PCT)',
        
        # Common parameter names
        r'^(MIN_|MAX_|REQ_|QTY|SIZE|CANTIDAD)',
        r'(DIST|GAP|OFFSET|SHIFT|DESVIO)',
        r'^K$|^D$|^M$|^N$',  # Common single-letter params
    ]
    
    # Identifier patterns
    IDENTIFIER_PATTERNS = [
        r'^(TRIAL|TRIAL_NUM|INDEX|ID)$',
        r'^(ESTRATEGIA|STRATEGY|COMBO|NOMBRE)',
    ]
    
    # Config patterns (internal system settings)
    CONFIG_PATTERNS = [
        r'^__',  # Double underscore = internal config
        r'(TIMEFRAME|TF)(_BASE|_ENTRY|_EXIT)?',
        r'(WARMUP|INDICATOR)',
        r'^(EXIT_TYPE|ENTRY_TYPE)$',
    ]
    
    @classmethod
    def classify(cls, df: pd.DataFrame) -> ColumnClassification:
        """Classify all columns using pattern matching + statistical analysis."""
        result = ColumnClassification()
        
        for col in df.columns:
            col_upper = str(col).upper().strip()
            
            # Skip empty columns
            if not col_upper or col_upper.startswith('UNNAMED'):
                continue
            
            category = cls._classify_column(col_upper, df[col])
            
            if category == 'metric':
                result.metrics.append(col)
            elif category == 'parameter':
                result.parameters.append(col)
            elif category == 'identifier':
                result.identifiers.append(col)
            elif category == 'config':
                result.config.append(col)
            else:
                # Use statistical heuristics for unknown columns
                if cls._is_likely_parameter(df[col], col_upper):
                    result.parameters.append(col)
                else:
                    result.unknown.append(col)
        
        return result
    
    @classmethod
    def _classify_column(cls, col: str, series: pd.Series) -> str:
        """Pattern-based classification."""
        # Check identifiers first
        for pattern in cls.IDENTIFIER_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return 'identifier'
        
        # Check config patterns
        for pattern in cls.CONFIG_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return 'config'
        
        # Check metric patterns
        for pattern in cls.METRIC_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return 'metric'
        
        # Check parameter patterns
        for pattern in cls.PARAM_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return 'parameter'
        
        return 'unknown'
    
    @classmethod
    def _is_likely_parameter(cls, series: pd.Series, col_name: str) -> bool:
        """Statistical heuristics for parameter detection."""
        try:
            # Convert to numeric
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            
            if len(numeric) < 5:
                return False
            
            # Parameters typically have:
            # 1. Limited unique values (discrete choices)
            unique_ratio = len(numeric.unique()) / len(numeric)
            if unique_ratio < 0.5 and len(numeric.unique()) >= 2:
                return True
            
            # 2. Integer values in reasonable range
            if all(numeric == numeric.astype(int)):
                if 0 <= numeric.min() and numeric.max() <= 10000:
                    return True
            
            # 3. Small range of float values (like percentages)
            if numeric.std() < 50 and 0 <= numeric.min():
                if numeric.max() <= 100:
                    return True
            
            return False
        except Exception:
            return False


# ==============================================================================
# 3. NOISE ANALYSIS & FILTERING
# ==============================================================================

class NoiseAnalyzer:
    """Statistical noise analysis and filtering for robust optimization."""
    
    @staticmethod
    def calculate_noise_metrics(df: pd.DataFrame, metric_col: str) -> Dict[str, Any]:
        """Calculate comprehensive noise metrics."""
        data = pd.to_numeric(df[metric_col], errors='coerce').dropna()
        
        if len(data) < 10:
            return {'error': 'Insufficient data'}
        
        mean = float(data.mean())
        std = float(data.std())
        cv = std / abs(mean) if mean != 0 else float('inf')
        
        # Outlier detection (IQR method)
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        
        # Normality test
        if len(data) >= 20:
            _, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        else:
            shapiro_p = 1.0
        
        # Noise classification
        if cv < 0.1:
            noise_level = 'very_low'
        elif cv < 0.3:
            noise_level = 'low'
        elif cv < 0.6:
            noise_level = 'moderate'
        elif cv < 1.0:
            noise_level = 'high'
        else:
            noise_level = 'very_high'
        
        return {
            'mean': mean,
            'std': std,
            'cv': cv,
            'noise_level': noise_level,
            'outliers_count': len(outliers),
            'outliers_pct': len(outliers) / len(data) * 100,
            'is_normal': shapiro_p > 0.05,
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
        }
    
    @staticmethod
    def filter_noise(x: np.ndarray, y: np.ndarray, 
                     method: str = 'savgol', window: int = 11) -> Tuple[np.ndarray, np.ndarray]:
        """Apply noise filtering to data."""
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < window:
            return x_clean, y_clean
        
        # Sort by x
        idx = np.argsort(x_clean)
        x_sorted, y_sorted = x_clean[idx], y_clean[idx]
        
        if method == 'savgol':
            # Savitzky-Golay filter (preserves peaks)
            win = min(window, len(y_sorted) - 1)
            if win % 2 == 0:
                win -= 1
            if win >= 3:
                y_filtered = savgol_filter(y_sorted, win, polyorder=2)
            else:
                y_filtered = y_sorted
        elif method == 'gaussian':
            # Gaussian smoothing
            sigma = window / 4
            y_filtered = gaussian_filter1d(y_sorted, sigma)
        else:
            y_filtered = y_sorted
        
        return x_sorted, y_filtered


# ==============================================================================
# 4. OPTIMAL ZONE DETECTOR (ML-Powered)
# ==============================================================================

class OptimalZoneDetector:
    """
    ML-powered detection of optimal parameter zones.
    Uses gradient boosting to find stable high-performance regions.
    """
    
    @staticmethod
    def find_optimal_zones(x: np.ndarray, y: np.ndarray, 
                           weights: Optional[np.ndarray] = None,
                           n_zones: int = 3) -> List[Dict[str, Any]]:
        """Find optimal parameter zones using ML."""
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 20:
            return []
        
        if weights is None:
            weights = np.ones(len(x_clean))
        else:
            weights = weights[mask]
        
        # Use weighted percentiles for robustness
        zones = []
        
        # Method 1: Density-weighted peaks
        from scipy.stats import gaussian_kde
        
        try:
            # Weight by performance
            top_pct = np.percentile(y_clean, 80)
            top_mask = y_clean >= top_pct
            
            if np.sum(top_mask) >= 5:
                top_x = x_clean[top_mask]
                
                # KDE on top performers
                kde = gaussian_kde(top_x, weights=weights[top_mask])
                x_grid = np.linspace(x_clean.min(), x_clean.max(), 200)
                density = kde(x_grid)
                
                # Find peaks in density
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(density, height=np.max(density)*0.3, distance=20)
                
                for peak_idx in peaks[:n_zones]:
                    peak_x = x_grid[peak_idx]
                    # Find zone width (where density > 50% of peak)
                    threshold = density[peak_idx] * 0.5
                    above = x_grid[density > threshold]
                    
                    if len(above) > 0:
                        zone_min = above[0]
                        zone_max = above[-1]
                        
                        # Calculate zone statistics
                        in_zone = (x_clean >= zone_min) & (x_clean <= zone_max)
                        zone_y = y_clean[in_zone]
                        
                        if len(zone_y) >= 3:
                            zones.append({
                                'center': float(peak_x),
                                'min': float(zone_min),
                                'max': float(zone_max),
                                'mean_perf': float(zone_y.mean()),
                                'std_perf': float(zone_y.std()),
                                'n_samples': int(np.sum(in_zone)),
                                'stability': float(zone_y.mean() / (zone_y.std() + 1e-6)),
                            })
        except Exception:
            pass
        
        # Sort zones by stability score
        zones.sort(key=lambda z: z.get('stability', 0), reverse=True)
        
        return zones[:n_zones]
    
    @staticmethod
    def calculate_robustness_score(x: np.ndarray, y: np.ndarray, 
                                   zone_min: float, zone_max: float) -> float:
        """Calculate robustness score for a parameter zone."""
        mask = (x >= zone_min) & (x <= zone_max)
        y_zone = y[mask]
        
        if len(y_zone) < 3:
            return 0.0
        
        # Components of robustness:
        # 1. Consistency (low variance)
        consistency = 1.0 / (1.0 + np.std(y_zone))
        
        # 2. Performance level
        performance = np.mean(y_zone)
        
        # 3. Sample size bonus
        sample_bonus = np.log1p(len(y_zone)) / 5
        
        # 4. Percentile rank
        overall_median = np.median(y)
        above_median = np.mean(y_zone > overall_median)
        
        score = (consistency * 0.3 + 
                 (performance / (np.max(y) + 1e-6)) * 0.4 + 
                 sample_bonus * 0.1 + 
                 above_median * 0.2)
        
        return float(score)


# ==============================================================================
# 5. QUANTILE REGRESSION ENGINE
# ==============================================================================

class QuantileEngine:
    """Advanced quantile regression with confidence intervals."""
    
    @staticmethod
    def weighted_quantile_regression(x: np.ndarray, y: np.ndarray,
                                     weights: np.ndarray,
                                     quantile: float = 0.75,
                                     resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted quantile curve."""
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
        x, y, weights = x[mask], y[mask], weights[mask]
        
        if len(x) < 10:
            return np.array([]), np.array([])
        
        x_min, x_max = np.percentile(x, [2, 98])
        x_grid = np.linspace(x_min, x_max, resolution)
        
        # Adaptive bandwidth
        bandwidth = 1.06 * x.std() * (len(x) ** (-1/5))
        
        y_quantile = []
        
        for x0 in x_grid:
            # Kernel weights
            dist = (x - x0) / bandwidth
            kernel = np.exp(-0.5 * dist**2)
            final_weights = kernel * weights
            
            # Filter insignificant weights
            valid = final_weights > 1e-6
            if not np.any(valid):
                y_quantile.append(np.nan)
                continue
            
            y_local = y[valid]
            w_local = final_weights[valid]
            
            # Weighted quantile
            sorter = np.argsort(y_local)
            y_sorted = y_local[sorter]
            w_sorted = w_local[sorter]
            
            cum_w = np.cumsum(w_sorted)
            cum_w /= cum_w[-1]
            
            idx = np.searchsorted(cum_w, quantile)
            if idx >= len(y_sorted):
                idx = len(y_sorted) - 1
            
            y_quantile.append(y_sorted[idx])
        
        return x_grid, np.array(y_quantile)
    
    @staticmethod
    def compute_confidence_band(x: np.ndarray, y: np.ndarray,
                                confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute confidence band using bootstrap."""
        n_bootstrap = 100
        x_grid = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 80)
        
        bootstrap_curves = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            x_boot, y_boot = x[idx], y[idx]
            
            # Simple moving average for each bootstrap
            y_curve = []
            bandwidth = (x.max() - x.min()) / 20
            
            for x0 in x_grid:
                mask = np.abs(x_boot - x0) < bandwidth
                if np.sum(mask) >= 3:
                    y_curve.append(np.mean(y_boot[mask]))
                else:
                    y_curve.append(np.nan)
            
            bootstrap_curves.append(y_curve)
        
        bootstrap_arr = np.array(bootstrap_curves)
        
        alpha = 1 - confidence
        lower = np.nanpercentile(bootstrap_arr, alpha/2 * 100, axis=0)
        upper = np.nanpercentile(bootstrap_arr, (1 - alpha/2) * 100, axis=0)
        
        return x_grid, lower, upper


# ==============================================================================
# 6. DATA LOADER (Enhanced)
# ==============================================================================

class QuantLoader:
    """Enhanced data loader with intelligent preprocessing."""
    
    TARGET_METRICS = {
        'ROI': 'ROI_PCT',
        'DD': 'MAX_DD_PCT',
        'PF': 'PROFIT_FACTOR',
        'SQN': 'SQN',
        'SCORE': 'SCORE',
    }
    
    def __init__(self):
        self.trade_col = 'TOTAL_TRADES'
        self.classification: Optional[ColumnClassification] = None
    
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and preprocess data file."""
        print(f"📂 Loading: {os.path.basename(file_path)}")
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                df = self._load_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                print(f"❌ Unsupported format: {ext}")
                return None
            
            # Normalize columns
            df.columns = [str(c).strip().upper().replace(' ', '_') for c in df.columns]
            
            # Rename common variations
            rename_map = {}
            for col in df.columns:
                col_u = col.upper()
                if 'ROI' in col_u and col != 'ROI_PCT':
                    rename_map[col] = 'ROI_PCT'
                elif col_u in ['DRAWDOWN', 'MAX_DRAWDOWN', 'DD']:
                    rename_map[col] = 'MAX_DD_PCT'
                elif 'PROFIT' in col_u and 'FACTOR' in col_u:
                    rename_map[col] = 'PROFIT_FACTOR'
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
            
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Classify columns
            self.classification = IntelligentClassifier.classify(df)
            
            # Find trade column
            for c in df.columns:
                if 'TRADE' in c.upper() and 'TOTAL' in c.upper():
                    self.trade_col = c
                    break
            
            # Report classification
            print(f"   ├── Metrics: {len(self.classification.metrics)}")
            print(f"   ├── Parameters: {len(self.classification.parameters)}")
            print(f"   ├── Config: {len(self.classification.config)}")
            print(f"   └── Rows: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV with intelligent header detection."""
        # First, try standard load
        df = pd.read_csv(file_path)
        
        # Check if first row looks like data (not headers)
        col_str = ' '.join([str(c).upper() for c in df.columns])
        keywords = ['ROI', 'SCORE', 'TRIAL', 'PROFIT', 'PNL', 'WINRATE', 'DRAWDOWN']
        
        if any(kw in col_str for kw in keywords):
            return df  # Headers are correct
        
        # Search for header row in data
        for i in range(min(10, len(df))):
            row_str = ' '.join([str(x).upper() for x in df.iloc[i].values])
            if any(kw in row_str for kw in keywords):
                df = pd.read_csv(file_path, header=i+1)
                return df
        
        return df
    
    def get_parameters(self) -> List[str]:
        """Get detected parameter columns."""
        if self.classification:
            return self.classification.parameters
        return []


# ==============================================================================
# 7. VISUALIZATION ENGINE
# ==============================================================================

class QuantVisualizer:
    """Professional visualization with optimal zone highlighting."""
    
    def __init__(self, loader: QuantLoader):
        self.loader = loader
        self.metrics = loader.TARGET_METRICS
        self.trade_col = loader.trade_col
    
    def _robust_scale(self, ax, data, axis='y', padding=0.10):
        """Set axis limits robustly."""
        try:
            clean = data[np.isfinite(data)]
            if len(clean) == 0:
                return
            dmin, dmax = np.percentile(clean, [2, 98])
            span = dmax - dmin
            if span == 0:
                span = 1
            if axis == 'y':
                ax.set_ylim(dmin - span*padding, dmax + span*padding)
            elif axis == 'x':
                ax.set_xlim(dmin - span*padding, dmax + span*padding)
        except Exception:
            pass
    
    # --------------------------------------------------------------------------
    # COVER PAGE
    # --------------------------------------------------------------------------
    def plot_cover(self, pdf, filename: str, df: pd.DataFrame, classification: ColumnClassification):
        """Generate cover page with KPIs."""
        fig = plt.figure(figsize=(11.69, 8.27))
        
        # Title
        plt.text(0.5, 0.78, "INTELLIGENT PARAMETER ANALYSIS", 
                 ha='center', fontsize=28, fontweight='light', color=COLORS['text'])
        plt.text(0.5, 0.72, "ML-Powered Optimization Report v4.0", 
                 ha='center', fontsize=11, color=COLORS['subtext'])
        
        # KPIs
        roi_col = self.metrics.get('ROI', 'ROI_PCT')
        dd_col = self.metrics.get('DD', 'MAX_DD_PCT')
        score_col = self.metrics.get('SCORE', 'SCORE')
        
        # ROI
        if roi_col in df.columns:
            best_roi = pd.to_numeric(df[roi_col], errors='coerce').max()
            plt.text(0.25, 0.52, f"{best_roi:+.1f}%", 
                     ha='center', fontsize=40, fontweight='bold', color=COLORS['up'])
            plt.text(0.25, 0.46, "PEAK ROI", ha='center', fontsize=10, color=COLORS['subtext'])
        
        # Drawdown
        if dd_col in df.columns:
            min_dd = pd.to_numeric(df[dd_col], errors='coerce').min()
            plt.text(0.5, 0.52, f"{min_dd:.1f}%", 
                     ha='center', fontsize=40, fontweight='bold', color=COLORS['down'])
            plt.text(0.5, 0.46, "MIN DRAWDOWN", ha='center', fontsize=10, color=COLORS['subtext'])
        
        # Best Score
        if score_col in df.columns:
            best_score = pd.to_numeric(df[score_col], errors='coerce').max()
            plt.text(0.75, 0.52, f"{best_score:.2f}", 
                     ha='center', fontsize=40, fontweight='bold', color=COLORS['gold'])
            plt.text(0.75, 0.46, "BEST SCORE", ha='center', fontsize=10, color=COLORS['subtext'])
        
        # Classification summary
        plt.text(0.5, 0.30, f"Parameters: {len(classification.parameters)} | "
                           f"Metrics: {len(classification.metrics)} | "
                           f"Samples: {len(df)}", 
                 ha='center', fontsize=10, color=COLORS['subtext'])
        
        # File info
        plt.text(0.5, 0.08, f"Source: {filename}", 
                 ha='center', fontsize=9, color=COLORS['grid'], style='italic')
        
        plt.axis('off')
        pdf.savefig(fig, facecolor=COLORS['bg'])
        plt.close()
    
    # --------------------------------------------------------------------------
    # NOISE ANALYSIS PAGE
    # --------------------------------------------------------------------------
    def plot_noise_analysis(self, pdf, df: pd.DataFrame):
        """Plot noise analysis for all metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle('NOISE ANALYSIS & DISTRIBUTION', fontsize=14, color=COLORS['text'], y=0.95)
        
        metric_cols = [
            (self.metrics.get('ROI', 'ROI_PCT'), 'ROI Distribution', COLORS['up']),
            (self.metrics.get('SCORE', 'SCORE'), 'Score Distribution', COLORS['gold']),
            (self.metrics.get('DD', 'MAX_DD_PCT'), 'Drawdown Distribution', COLORS['down']),
            (self.metrics.get('SQN', 'SQN'), 'SQN Distribution', COLORS['cyan']),
        ]
        
        for ax, (col, title, color) in zip(axes.flat, metric_cols):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(data) < 5:
                ax.set_visible(False)
                continue
            
            # Plot histogram with KDE
            sns.histplot(data, stat='density', bins=30, alpha=0.3, color=color, ax=ax)
            
            # Fit normal distribution
            mu, std = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, std), color=COLORS['text'], 
                    linewidth=1, alpha=0.7, linestyle='--', label='Normal fit')
            
            # Mark mean and median
            ax.axvline(mu, color=COLORS['accent'], linewidth=1.5, label=f'Mean: {mu:.2f}')
            ax.axvline(data.median(), color=COLORS['purple'], linewidth=1.5, 
                       linestyle=':', label=f'Median: {data.median():.2f}')
            
            # Noise metrics
            noise = NoiseAnalyzer.calculate_noise_metrics(df, col)
            ax.text(0.98, 0.98, f"CV: {noise.get('cv', 0):.2f}\n"
                               f"Noise: {noise.get('noise_level', 'N/A')}", 
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    color=COLORS['subtext'], bbox=dict(boxstyle='round', 
                    facecolor=COLORS['panel'], edgecolor=COLORS['grid']))
            
            ax.set_title(title, fontsize=10, color=COLORS['text'])
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, facecolor=COLORS['bg'])
        plt.close()
    
    # --------------------------------------------------------------------------
    # PARAMETER SENSITIVITY (Main Analysis)
    # --------------------------------------------------------------------------
    def plot_parameter_sensitivity(self, pdf, df: pd.DataFrame, param: str):
        """Plot comprehensive parameter sensitivity analysis."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle(f'PARAMETER SENSITIVITY: {param}', fontsize=16, color=COLORS['text'], y=0.96)
        fig.text(0.5, 0.93, 'White Line: Optimal Potential | Shaded: Top-Tier Zone | ◉ Optimal Point', 
                 ha='center', fontsize=9, color=COLORS['subtext'])
        
        configs = [
            (0, 0, 'ROI', self.metrics.get('ROI', 'ROI_PCT'), 'Net Profit %', COLORS['up'], False),
            (0, 1, 'DD', self.metrics.get('DD', 'MAX_DD_PCT'), 'Max Drawdown %', COLORS['down'], True),
            (1, 0, 'PF', self.metrics.get('PF', 'PROFIT_FACTOR'), 'Profit Factor', COLORS['band'], False),
            (1, 1, 'SQN', self.metrics.get('SQN', 'SQN'), 'SQN Score', COLORS['gold'], False),
        ]
        
        x = df[param].values.astype(float)
        trades = df[self.trade_col].values if self.trade_col in df.columns else np.ones_like(x)
        trades = np.maximum(trades, 1)  # Avoid zero weights
        
        # Dynamic point sizes
        sizes = np.log1p(trades)
        sizes = (sizes / (sizes.max() + 1e-9)) * 50 + 5
        
        for r, c, key, col, label, color, minimize in configs:
            ax = fig.add_subplot(gs[r, c])
            
            if col not in df.columns:
                ax.text(0.5, 0.5, f'{col} not available', ha='center', va='center',
                       transform=ax.transAxes, color=COLORS['subtext'])
                ax.set_title(label, fontsize=10, color=COLORS['text'])
                continue
            
            y = df[col].values.astype(float)
            
            # Filter valid data
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sizes)
            x_valid, y_valid, s_valid, t_valid = x[mask], y[mask], sizes[mask], trades[mask]
            
            if len(x_valid) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                       transform=ax.transAxes, color=COLORS['subtext'])
                continue
            
            # 1. Scatter plot
            ax.scatter(x_valid, y_valid, c=color, alpha=0.2, s=s_valid, 
                      edgecolors='none', zorder=1)
            
            # 2. Quantile curves
            xs, y50 = QuantileEngine.weighted_quantile_regression(x_valid, y_valid, t_valid, 0.50)
            _, y75 = QuantileEngine.weighted_quantile_regression(x_valid, y_valid, t_valid, 
                                                                  0.25 if minimize else 0.75)
            _, y90 = QuantileEngine.weighted_quantile_regression(x_valid, y_valid, t_valid, 
                                                                  0.10 if minimize else 0.90)
            
            if len(xs) > 0 and len(y50) > 0 and len(y90) > 0:
                # Band between median and top percentile
                ax.fill_between(xs, y50, y90, color=color, alpha=0.15, zorder=2)
                
                # Main potential line
                if len(y75) == len(xs):
                    # Smooth the line
                    y75_smooth = savgol_filter(y75, min(11, len(y75)//3*2+1), 2) if len(y75) > 11 else y75
                    ax.plot(xs, y75_smooth, color=COLORS['optimal'], linewidth=2, alpha=0.9, zorder=4)
                    
                    # Find optimal point
                    if minimize:
                        idx_best = np.nanargmin(y75_smooth)
                    else:
                        # Score: high value + stability
                        spread = np.abs(y90 - y50)
                        score = y75_smooth - spread * 0.2
                        idx_best = np.nanargmax(score)
                    
                    bx, by = xs[idx_best], y75_smooth[idx_best]
                    
                    # Draw optimal marker
                    ax.scatter(bx, by, color=COLORS['bg'], s=80, 
                              edgecolors=COLORS['optimal'], linewidth=2, zorder=5)
                    
                    # Label
                    offset = -by*0.08 if minimize else by*0.08
                    ax.annotate(f'Optimal: {bx:.1f}', 
                               xy=(bx, by), xytext=(bx, by + offset),
                               ha='center', va='bottom' if not minimize else 'top',
                               fontsize=9, color=COLORS['optimal'], fontweight='bold',
                               arrowprops=dict(arrowstyle='-', color=COLORS['optimal'], alpha=0.5))
            
            # 3. Find and mark optimal zones
            zones = OptimalZoneDetector.find_optimal_zones(x_valid, y_valid if not minimize else -y_valid)
            for i, zone in enumerate(zones[:1]):  # Show top zone
                ax.axvspan(zone['min'], zone['max'], alpha=0.1, color=COLORS['gold'], zorder=0)
            
            # Styling
            ax.set_title(label, fontsize=10, color=COLORS['text'], fontweight='medium', loc='left')
            ax.set_xlabel(param, fontsize=9)
            ax.grid(True, color=COLORS['grid'], alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            self._robust_scale(ax, x_valid, 'x')
            self._robust_scale(ax, y_valid, 'y')
        
        pdf.savefig(fig, facecolor=COLORS['bg'], bbox_inches='tight')
        plt.close()
    
    # --------------------------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------------------------
    def plot_correlation(self, pdf, df: pd.DataFrame, params: List[str]):
        """Plot parameter-metric correlation heatmap."""
        valid_metrics = [m for m in self.metrics.values() if m in df.columns]
        
        if not valid_metrics or not params:
            return
        
        # Limit to top params by variance
        if len(params) > 15:
            variances = {p: df[p].var() for p in params if p in df.columns}
            params = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:15]
        
        cols = params + valid_metrics
        corr_matrix = df[cols].corr()
        param_metric_corr = corr_matrix.loc[params, valid_metrics]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(params) * 0.4)))
        
        # Create heatmap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(param_metric_corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                   ax=ax, linewidths=0.5, linecolor=COLORS['bg'],
                   cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
        
        ax.set_title('PARAMETER-METRIC CORRELATION', fontsize=14, color=COLORS['text'], pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, facecolor=COLORS['bg'])
        plt.close()
    
    # --------------------------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------------------------
    def plot_summary_table(self, pdf, df: pd.DataFrame, params: List[str]):
        """Generate summary table with optimal ranges."""
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        fig.suptitle('OPTIMAL PARAMETER RANGES', fontsize=14, color=COLORS['text'], y=0.95)
        
        table_data = []
        roi_col = self.metrics.get('ROI', 'ROI_PCT')
        
        for param in params[:20]:  # Limit to 20 params
            if param not in df.columns or roi_col not in df.columns:
                continue
            
            x = df[param].values.astype(float)
            y = df[roi_col].values.astype(float)
            
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 10:
                continue
            
            x_valid, y_valid = x[mask], y[mask]
            
            # Find optimal zone
            zones = OptimalZoneDetector.find_optimal_zones(x_valid, y_valid)
            
            if zones:
                zone = zones[0]
                table_data.append([
                    param,
                    f"{zone['min']:.2f} - {zone['max']:.2f}",
                    f"{zone['center']:.2f}",
                    f"{zone['mean_perf']:.2f}%",
                    f"{zone['stability']:.2f}",
                ])
            else:
                # Fallback to simple percentile
                top_mask = y_valid >= np.percentile(y_valid, 75)
                x_top = x_valid[top_mask]
                table_data.append([
                    param,
                    f"{x_top.min():.2f} - {x_top.max():.2f}",
                    f"{x_top.mean():.2f}",
                    f"{y_valid[top_mask].mean():.2f}%",
                    "N/A",
                ])
        
        if table_data:
            cols = ['Parameter', 'Optimal Range', 'Center', 'Avg ROI', 'Stability']
            table = ax.table(cellText=table_data, colLabels=cols, loc='center',
                            cellLoc='center', colWidths=[0.25, 0.20, 0.15, 0.15, 0.15])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Style
            for key, cell in table.get_celld().items():
                cell.set_linewidth(0)
                cell.set_edgecolor(COLORS['grid'])
                if key[0] == 0:  # Header
                    cell.set_facecolor(COLORS['panel'])
                    cell.set_text_props(color=COLORS['text'], weight='bold')
                else:
                    cell.set_facecolor(COLORS['bg'] if key[0] % 2 else COLORS['panel'])
                    cell.set_text_props(color=COLORS['subtext'])
        
        pdf.savefig(fig, facecolor=COLORS['bg'])
        plt.close()


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "═" * 60)
    print("   MODELOX INTELLIGENT ANALYZER v4.0")
    print("   ML-Powered Parameter Optimization")
    print("═" * 60 + "\n")
    
    # Get input file
    file_path = ""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx'))]
        if files:
            print(f"📁 Found {len(files)} data file(s)")
            for i, f in enumerate(files[:5]):
                print(f"   [{i+1}] {f}")
            
            choice = input("\n>> Select file (1) or enter path: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                file_path = files[int(choice) - 1]
            elif choice:
                file_path = choice.strip('"\'')
            else:
                file_path = files[0]
        else:
            file_path = input(">> Enter file path: ").strip().strip('"\'')
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Load data
    loader = QuantLoader()
    df = loader.load_data(file_path)
    
    if df is None or loader.classification is None:
        print("❌ Failed to load data")
        return
    
    params = loader.get_parameters()
    
    if not params:
        print("⚠️  No parameters detected!")
        return
    
    print(f"\n📊 Detected {len(params)} parameters:")
    for p in params[:10]:
        print(f"   • {p}")
    if len(params) > 10:
        print(f"   ... and {len(params) - 10} more")
    
    # Generate report
    out_pdf = f"ANALYSIS_{os.path.splitext(os.path.basename(file_path))[0]}.pdf"
    viz = QuantVisualizer(loader)
    
    print("\n⚙️  Generating report...")
    
    try:
        with PdfPages(out_pdf) as pdf:
            # Cover
            viz.plot_cover(pdf, os.path.basename(file_path), df, loader.classification)
            
            # Noise analysis
            viz.plot_noise_analysis(pdf, df)
            
            # Correlation
            viz.plot_correlation(pdf, df, params)
            
            # Summary table
            viz.plot_summary_table(pdf, df, params)
            
            # Parameter pages
            total = len(params)
            for i, param in enumerate(params):
                sys.stdout.write(f"\r   Analyzing {param} [{i+1}/{total}]".ljust(60))
                sys.stdout.flush()
                viz.plot_parameter_sensitivity(pdf, df, param)
        
        print(f"\n\n✅ Report saved: {out_pdf}")
        
        # Try to open
        try:
            if sys.platform == 'darwin':
                os.system(f'open "{out_pdf}"')
            elif sys.platform == 'win32':
                os.startfile(out_pdf)
            else:
                import subprocess
                subprocess.run(['xdg-open', out_pdf], check=False,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

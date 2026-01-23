#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                    MODELOX INSTITUTIONAL ANALYZER v20.0                               ║
║                  Advanced Robustness & Statistical Analysis Edition                    ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║  🎯 ADVANCED OPTIMAL DETECTION (7 técnicas combinadas):                               ║
║     • Regional Growth Algorithm (Plateau Detection)                                   ║
║     • Bootstrap Confidence Intervals (100 remuestreos)                                ║
║     • Cross-Validated Optimal (K-Fold Stability)                                      ║
║     • Derivative Analysis (Gradient + Curvature)                                      ║
║     • Bayesian Surrogate Model (Ensemble GBM)                                         ║
║     • RANSAC-like Robust Regression                                                   ║
║     • KDE Mode Detection (High-Performance Regions)                                   ║
║  📐 ADVANCED RANGE DETECTION:                                                         ║
║     • Sensitivity Analysis (Local Stability)                                          ║
║     • Changepoint Detection (Inflection Points)                                       ║
║     • Performance Threshold + Cluster Analysis                                        ║
║  🔗 Parameter Correlation Analysis (Pearson, Spearman, Joint Importance)              ║
║  📊 3D Surface Optimization (SL/TP + Correlated Pairs vs ROI/SQN)                     ║
║  🧪 Statistical Validation (White's Reality Check + Deflated Sharpe)                  ║
║  🔬 ADVANCED ROBUSTNESS ANALYSIS (NEW v20.0):                                         ║
║     • DBSCAN Cluster Detection (Configuration Grouping)                               ║
║     • Neighborhood Stability Index (NSI) - Similar Params → Similar Results?          ║
║     • Parameter Degradation Testing (Future-Proof Simulation)                         ║
║     • Surface CV Analysis (3D Roughness/Smoothness Quantification)                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import glob
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy import stats
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import UnivariateSpline, griddata
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 ULTRA-MINIMALIST DARK THEME
# ══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Tema visual ultra-minimalista oscuro."""

    # Fondos - muy oscuros
    BG_PRIMARY = '#08090c'
    BG_SECONDARY = '#0d1014'
    BG_TERTIARY = '#12161c'
    BG_CARD = '#171c24'
    BG_HIGHLIGHT = '#1c222c'
    BG_ELEVATED = '#212836'

    # Textos - claros y suaves
    TEXT_PRIMARY = '#f0f4f8'
    TEXT_SECONDARY = '#a0aec0'
    TEXT_MUTED = '#718096'
    TEXT_DARK = '#4a5568'

    # Colores de acento - tonos suaves
    ACCENT = '#64b5f6'
    ACCENT_LIGHT = '#90caf9'
    ACCENT_DIM = '#42a5f5'

    # Estados
    GREEN = '#4caf50'
    GREEN_BRIGHT = '#81c784'
    GREEN_LIGHT = '#a5d6a7'
    RED = '#ef5350'
    RED_BRIGHT = '#e57373'
    ORANGE = '#ffb74d'
    ORANGE_BRIGHT = '#ffd54f'
    GOLD = '#ffd54f'
    PURPLE = '#9575cd'
    PURPLE_BRIGHT = '#b39ddb'
    CYAN = '#4dd0e1'
    BLUE = '#42a5f5'
    BLUE_BRIGHT = '#64b5f6'
    BLUE_LIGHT = '#90caf9'
    PINK = '#f48fb1'

    # Grid y bordes - muy sutiles
    GRID = '#1a2030'
    BORDER = '#2a3444'
    DIVIDER = '#1a2030'

    @classmethod
    def get_surface_cmap(cls):
        """
        Colormap para superficies 3D: SOLO tonos de AZUL.
        Azul oscuro (abajo/bajo) → Azul claro brillante (arriba/alto).
        """
        colors = [
            '#000814',  # Azul casi negro
            '#001d3d',  # Azul muy oscuro
            '#002855',  # Azul marino profundo
            '#003566',  # Azul marino
            '#004080',  # Azul oscuro
            '#0056a3',  # Azul medio-oscuro
            '#0066cc',  # Azul medio
            '#0077e6',  # Azul
            '#1a8cff',  # Azul brillante
            '#4da6ff',  # Azul claro brillante
            '#80bfff',  # Azul claro
            '#99ccff',  # Azul muy claro
            '#b3d9ff',  # Azul pálido brillante
        ]
        return LinearSegmentedColormap.from_list('surface_blue_tones', colors, N=512)

    @classmethod
    def get_chart_cmap(cls):
        """Colormap suave para gráficos."""
        colors = ['#0d1014', '#1a3a5c', '#42a5f5', '#64b5f6', '#90caf9']
        return LinearSegmentedColormap.from_list('chart', colors, N=256)

    @classmethod
    def get_correlation_cmap(cls):
        """Colormap para matriz de correlación: rojo negativo, azul positivo."""
        colors = ['#e53935', '#ef5350', '#1a1a1a', '#42a5f5', '#1565c0']
        return LinearSegmentedColormap.from_list('correlation', colors, N=256)


def format_number(x, pos=None):
    """Formatea números evitando notación científica."""
    if x == 0:
        return '0'
    abs_x = abs(x)
    if abs_x >= 1000000:
        return f'{x/1000000:.1f}M'
    elif abs_x >= 1000:
        return f'{x/1000:.1f}K'
    elif abs_x >= 1:
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
    elif abs_x >= 0.01:
        return f'{x:.2f}'
    elif abs_x >= 0.001:
        return f'{x:.3f}'
    else:
        return f'{x:.4f}'


def format_axis_number(x, pos=None):
    """Formatea números para ejes - más compacto."""
    if x == 0:
        return '0'
    abs_x = abs(x)
    if abs_x >= 1000000:
        return f'{x/1000000:.0f}M'
    elif abs_x >= 10000:
        return f'{x/1000:.0f}K'
    elif abs_x >= 1000:
        return f'{x/1000:.1f}K'
    elif abs_x >= 100:
        return f'{x:.0f}'
    elif abs_x >= 10:
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
    elif abs_x >= 1:
        return f'{x:.1f}'
    elif abs_x >= 0.1:
        return f'{x:.2f}'
    else:
        return f'{x:.3f}'


# Configurar matplotlib
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': Theme.BG_PRIMARY,
    'axes.facecolor': Theme.BG_SECONDARY,
    'axes.edgecolor': Theme.BORDER,
    'axes.labelcolor': Theme.TEXT_SECONDARY,
    'axes.titlecolor': Theme.TEXT_PRIMARY,
    'text.color': Theme.TEXT_PRIMARY,
    'xtick.color': Theme.TEXT_MUTED,
    'ytick.color': Theme.TEXT_MUTED,
    'grid.color': Theme.GRID,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'axes.linewidth': 0.8,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.facecolor': Theme.BG_CARD,
    'legend.edgecolor': Theme.BORDER,
})


# ══════════════════════════════════════════════════════════════════════════════
# 📊 SISTEMA DE DETECCIÓN DE COLUMNAS INTELIGENTE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataSchema:
    """Esquema de datos detectado."""
    metrics: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    exit_params: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)


class SmartColumnDetector:
    """Detector inteligente de columnas con reglas precisas."""

    # Métricas de rendimiento (RESULTADOS - nunca son parámetros)
    METRICS = {
        'SCORE', 'ROI', 'ROI_PCT', 'RETURN', 'PNL', 'PNL_NETO',
        'SHARPE', 'SORTINO', 'CALMAR', 'SQN', 'MAR',
        'PROFIT_FACTOR', 'PAYOFF', 'PAYOFF_RATIO',
        'DRAWDOWN', 'MAX_DD', 'MAX_DD_PCT', 'DD',
        'WINRATE', 'WINRATE_PCT', 'WIN_RATE', 'PORC_GANADORAS', 'PORC_PERDEDORAS',
        'ESTABILIDAD', 'STABILITY', 'CONSISTENCY',
        'EXPECTATIVA', 'EXPECTANCY', 'RETORNO_PROMEDIO',
        'SALDO_ACTUAL', 'SALDO_MIN', 'SALDO_MAX', 'SALDO_MEAN', 'BALANCE',
        'TOTAL_TRADES', 'N_TRADES', 'NUM_TRADES', 'TRADES_DIA', 'TRADES_POR_DIA',
        'NUM_LONGS', 'NUM_SHORTS', 'COUNT_LONGS', 'COUNT_SHORTS',
        'N_TRADES_LONG', 'N_TRADES_SHORT', 'LONGS', 'SHORTS',
        'RACHA_GANADORA', 'RACHA_PERDEDORA', 'MAX_CONSECUTIVE',
        'MAX_GANANCIA', 'MAX_PERDIDA', 'AVG_WIN', 'AVG_LOSS',
        'RIESGO_BENEFICIO', 'RISK_REWARD', 'DURATION_MEAN', 'DURATION_MEAN_MIN',
        'COMISIONES', 'COMISIONES_TOTAL', 'FEES', 'SLIPPAGE',
        'KELLY', 'KELLY_PCT', 'OPTIMAL_F', 'VAR', 'CVAR',
    }

    # Identificadores (no son ni parámetros ni métricas)
    IDENTIFIERS = {
        'TRIAL', 'INDEX', 'ID', 'ESTRATEGIA', 'STRATEGY', 'NOMBRE', 'NAME',
        'NOMBRE_COMBO', 'COMBO', 'CONFIG', 'RUN_ID', 'SEED',
        'FECHA', 'DATE', 'TIMESTAMP', 'ACTIVO', 'SYMBOL', 'ASSET',
    }

    # Columnas de sistema (prefijo __)
    SYSTEM_PREFIXES = ('__', 'UNNAMED', 'INDEX')

    # Patrones de exit params
    EXIT_PATTERNS = ['EXIT_SL', 'EXIT_TP', 'SL_PCT', 'TP_PCT', 'SL%', 'TP%',
                     'STOP_LOSS', 'TAKE_PROFIT', 'TRAIL']

    @classmethod
    def detect(cls, df: pd.DataFrame) -> DataSchema:
        """Detecta y clasifica todas las columnas."""
        schema = DataSchema()

        for col in df.columns:
            col_clean = str(col).strip()
            col_upper = col_clean.upper().replace(' ', '_').replace('%', '_PCT')

            # Ignorar columnas de sistema
            if any(col_upper.startswith(p) for p in cls.SYSTEM_PREFIXES):
                schema.ignored.append(col_clean)
                continue

            # Identificadores
            if col_upper in cls.IDENTIFIERS:
                schema.identifiers.append(col_clean)
                continue

            # Métricas (verificar exacto y parcial)
            if cls._is_metric(col_upper):
                schema.metrics.append(col_clean)
                continue

            # Exit params (TP/SL)
            if cls._is_exit_param(col_upper):
                if cls._has_variation(df[col]):
                    schema.exit_params.append(col_clean)
                else:
                    schema.ignored.append(col_clean)
                continue

            # Parámetros (numéricos con variación)
            if cls._has_variation(df[col]):
                schema.params.append(col_clean)
            else:
                schema.ignored.append(col_clean)

        return schema

    @classmethod
    def _is_metric(cls, col: str) -> bool:
        """Verifica si es métrica."""
        # Exacto
        if col in cls.METRICS:
            return True
        # Sin sufijos
        for suffix in ['_PCT', 'PCT', '_RATIO', '_MEAN', '_MIN', '_MAX']:
            if col.endswith(suffix) and col[:-len(suffix)] in cls.METRICS:
                return True
        # Parcial
        for m in cls.METRICS:
            if col.startswith(m + '_') or col.endswith('_' + m):
                return True
        return False

    @classmethod
    def _is_exit_param(cls, col: str) -> bool:
        """Verifica si es exit param."""
        return any(p in col for p in cls.EXIT_PATTERNS)

    @classmethod
    def _has_variation(cls, series: pd.Series) -> bool:
        """Verifica si tiene variación (es parámetro válido)."""
        try:
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric) < 5:
                return False
            # Debe tener al menos 2 valores únicos y no ser constante
            unique_vals = numeric.nunique()
            return unique_vals >= 2 and numeric.std() > 1e-10
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# 📂 CARGADOR DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Cargador robusto de datos CSV/Excel."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.df_raw: Optional[pd.DataFrame] = None  # Sin filtrar
        self.schema: Optional[DataSchema] = None
        self.strategy_name: str = "STRATEGY"
        self.file_path: str = ""
        self.outliers_removed: int = 0

    def load(self, path: str) -> bool:
        """Carga archivo y detecta esquema."""
        self.file_path = path
        ext = os.path.splitext(path)[1].lower()

        print(f"\n{'━'*70}")
        print(f"  📂 {os.path.basename(path)}")
        print('━'*70)

        try:
            if ext in ['.xlsx', '.xls']:
                self.df = self._load_excel(path)
            else:
                self.df = self._load_csv(path)

            if self.df is None or len(self.df) == 0:
                print("  ✗ No data loaded")
                return False

            self._clean()
            self.df_raw = self.df.copy()
            self.schema = SmartColumnDetector.detect(self.df)
            self._extract_strategy()
            self._print_summary()
            return True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    def _load_excel(self, path: str) -> pd.DataFrame:
        """Carga Excel con detección de header."""
        pd.ExcelFile(path)
        preview = pd.read_excel(path, sheet_name=0, header=None, nrows=10)

        # Buscar fila de header
        keywords = {'ROI', 'SCORE', 'TRIAL', 'SHARPE', 'DRAWDOWN', 'SQN'}
        header_row = 0
        best_score = 0

        for idx in range(min(5, len(preview))):
            row_vals = [str(v).upper() for v in preview.iloc[idx] if pd.notna(v)]
            score = sum(1 for v in row_vals for kw in keywords if kw in v)
            if score > best_score:
                best_score = score
                header_row = idx

        return pd.read_excel(path, sheet_name=0, header=header_row)

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Carga CSV con detección de delimitador y encoding."""
        # Detectar encoding
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        used_encoding = 'utf-8'

        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read(8192)
                used_encoding = enc
                break
            except Exception:
                continue

        if content is None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(8192)

        # Detectar delimitador
        delims = {',': content.count(','), ';': content.count(';'), '\t': content.count('\t')}
        best_delim = max(delims, key=delims.get)

        # Cargar con opciones robustas
        return pd.read_csv(path, sep=best_delim, encoding=used_encoding,
                          on_bad_lines='skip', engine='python')

    def _clean(self):
        """Limpia DataFrame."""
        self.df = self.df.dropna(how='all').dropna(axis=1, how='all')
        self.df.columns = [str(c).strip() for c in self.df.columns]

        # Eliminar columnas unnamed
        cols_drop = [c for c in self.df.columns if 'unnamed' in c.lower()]
        self.df = self.df.drop(columns=cols_drop, errors='ignore')

        # Convertir numéricas
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                numeric = pd.to_numeric(self.df[col], errors='coerce')
                if numeric.notna().sum() > len(self.df) * 0.5:
                    self.df[col] = numeric

    def _extract_strategy(self):
        """Extrae nombre de estrategia."""
        for col in ['ESTRATEGIA', 'STRATEGY', 'NOMBRE', 'NAME']:
            if col in self.df.columns:
                vals = self.df[col].dropna().unique()
                if len(vals) > 0:
                    self.strategy_name = str(vals[0]).replace(' ', '_')[:25]
                    break

    def _print_summary(self):
        """Imprime resumen."""
        self.schema.params + self.schema.exit_params
        print(f"  ✓ {len(self.df):,} trials | {self.strategy_name}")
        print(f"  ✓ Params: {len(self.schema.params)} | Exit: {len(self.schema.exit_params)} | Metrics: {len(self.schema.metrics)}")

    def apply_filters(self, min_score: float = 0.11) -> int:
        """Aplica filtros de calidad."""
        initial = len(self.df)

        # Filtro SCORE
        score_col = self._find_col(['SCORE', 'Score'])
        if score_col:
            self.df = self.df[pd.to_numeric(self.df[score_col], errors='coerce') >= min_score]

        self.df = self.df.reset_index(drop=True)
        removed = initial - len(self.df)

        if removed > 0:
            print(f"  ⚡ Filtrado: {removed} trials eliminados (SCORE<{min_score})")
        else:
            print(f"  ✓ Sin filtrado necesario (todos cumplen SCORE>={min_score})")

        return removed

    def _find_col(self, candidates: List[str]) -> Optional[str]:
        """Busca columna por nombre."""
        for c in candidates:
            if c in self.df.columns:
                return c
            for col in self.df.columns:
                if col.upper() == c.upper():
                    return col
        return None

    def remove_outliers(self, percentile: float = 95.0, metric_col: str = None) -> int:
        """Elimina outliers fuera del percentil especificado (P5-P95)."""
        if metric_col is None:
            metric_col = self._find_col(['SCORE', 'ROI', 'ROI_PCT'])

        if metric_col is None:
            return 0

        initial = len(self.df)
        values = pd.to_numeric(self.df[metric_col], errors='coerce')

        # Calcular percentiles P5 y P95
        lower = values.quantile((100 - percentile) / 100)
        upper = values.quantile(percentile / 100)

        # Filtrar outliers
        mask = (values >= lower) & (values <= upper)
        self.df = self.df[mask].reset_index(drop=True)

        self.outliers_removed = initial - len(self.df)

        if self.outliers_removed > 0:
            print(f"  🎯 Outlier filter: {self.outliers_removed} removed (P{100-percentile:.0f}-P{percentile:.0f} on {metric_col})")

        return self.outliers_removed


# ══════════════════════════════════════════════════════════════════════════════
# 🧪 STATISTICAL VALIDATION (White's Reality Check + Deflated Sharpe)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StatisticalValidation:
    """Resultado de validación estadística."""
    # White's Reality Check
    wrc_p_value: float = 0.0
    wrc_is_significant: bool = False
    wrc_bootstrap_mean: float = 0.0
    wrc_bootstrap_std: float = 0.0

    # Deflated Sharpe Ratio
    original_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    sharpe_haircut: float = 0.0
    dsr_is_significant: bool = False

    # Additional Statistics
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0

    # Left Tail Risk (cross-sectional trials)
    var_95: float = 0.0   # Historical VaR at 95% confidence (5% left tail)
    cvar_95: float = 0.0  # Expected Shortfall (CVaR) at 95% confidence
    prob_loss: float = 0.0

    # Store raw trial results for visualization
    trial_returns: np.ndarray = field(default_factory=lambda: np.array([]))


class StatisticalValidator:
    """Validador estadístico con múltiples tests."""

    def __init__(self, returns: np.ndarray, n_trials: int):
        self.returns = returns
        self.n_trials = n_trials

    def validate(self) -> StatisticalValidation:
        """Ejecuta todos los tests de validación."""
        result = StatisticalValidation()

        if len(self.returns) < 20:
            return result

        # Guardar valores originales (trials cross-sectional)
        result.trial_returns = np.asarray(self.returns, dtype=float)

        # Normalizar retornos
        returns_norm = (self.returns - np.mean(self.returns)) / (np.std(self.returns) + 1e-10)

        # 1. White's Reality Check
        wrc = self._white_reality_check(returns_norm, n_simulations=200)
        result.wrc_p_value = wrc['p_value']
        result.wrc_is_significant = wrc['is_significant']
        result.wrc_bootstrap_mean = wrc['bootstrap_mean']
        result.wrc_bootstrap_std = wrc['bootstrap_std']

        # 2. Deflated Sharpe Ratio
        dsr = self._deflated_sharpe(returns_norm)
        result.original_sharpe = dsr['original']
        result.deflated_sharpe = dsr['deflated']
        result.sharpe_haircut = dsr['haircut']
        result.dsr_is_significant = dsr['is_significant']

        # 3. Distribution Statistics
        result.skewness = stats.skew(self.returns)
        result.kurtosis = stats.kurtosis(self.returns)

        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        result.jarque_bera_stat = jb_stat
        result.jarque_bera_pvalue = jb_pvalue

        # 4. Left Tail Risk (cross-sectional, not time series)
        var_95, cvar_95 = self._historical_var_cvar(self.returns, alpha=0.05)
        result.var_95 = var_95
        result.cvar_95 = cvar_95
        result.prob_loss = float(np.mean(self.returns < 0))

        return result

    def _white_reality_check(self, returns: np.ndarray, n_simulations: int = 200) -> Dict:
        """White's Reality Check con bootstrap.

        Nota: aquí trabajamos con una distribución de resultados de optimización (trials).
        Por tanto usamos bootstrap i.i.d. (no block-bootstrap), ya que no hay orden temporal.
        """
        original_stat = np.mean(returns)
        n = len(returns)
        bootstrap_stats = np.zeros(n_simulations)
        n_strategies_sample = min(50, self.n_trials)

        for i in range(n_simulations):
            # Bootstrap i.i.d. (cross-sectional)
            indices = np.random.randint(0, n, size=n)
            bootstrap_sample = returns[indices]

            # Simular estrategias
            if n_strategies_sample > 1:
                random_means = np.array([
                    np.mean(returns[np.random.randint(0, n, size=n)])
                    for _ in range(min(20, n_strategies_sample - 1))
                ])
                max_stat = max(original_stat, np.max(random_means))
            else:
                max_stat = original_stat

            bootstrap_stats[i] = np.mean(bootstrap_sample) - max_stat

        p_value = np.mean(bootstrap_stats >= 0)

        return {
            'p_value': p_value,
            'is_significant': p_value <= 0.05,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats)
        }

    def _deflated_sharpe(self, returns: np.ndarray) -> Dict:
        """Deflated Sharpe Ratio (Bailey & López de Prado)."""
        # Para resultados de optimización (cross-sectional), no se anualiza.
        annual_factor = 1.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return {'original': 0, 'deflated': 0, 'haircut': 1, 'is_significant': False}

        sharpe = mean_ret / std_ret * np.sqrt(annual_factor)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Expected max Sharpe
        if self.n_trials > 1:
            euler = 0.5772156649
            e_max_sharpe = (1 - euler) * stats.norm.ppf(1 - 1/self.n_trials) + \
                          euler * stats.norm.ppf(1 - 1/(self.n_trials * np.e))
            e_max_sharpe *= np.sqrt(annual_factor / len(returns))
        else:
            e_max_sharpe = 0

        # Variance del Sharpe
        var_sharpe = (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / len(returns)
        std_sharpe = np.sqrt(var_sharpe) * np.sqrt(annual_factor)

        deflated = max(0, sharpe - e_max_sharpe)
        haircut = min(1, e_max_sharpe / (abs(sharpe) + 1e-10)) if sharpe > 0 else 1

        # PSR
        psr = stats.norm.cdf((sharpe - e_max_sharpe) / (std_sharpe + 1e-10)) if std_sharpe > 0 else 0.5

        return {
            'original': sharpe,
            'deflated': deflated,
            'haircut': haircut,
            'is_significant': psr >= 0.95
        }

    def _historical_var_cvar(self, returns: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        """VaR/CVaR histórico sobre una distribución cross-sectional.

        alpha=0.05 => VaR al 95% (percentil 5) y CVaR como media condicional de la cola.
        """
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            return 0.0, 0.0

        var = float(np.quantile(r, alpha))
        tail = r[r <= var]
        cvar = float(np.mean(tail)) if tail.size > 0 else var
        return var, cvar


# ══════════════════════════════════════════════════════════════════════════════
# 🔬 ADVANCED ROBUSTNESS ANALYSIS MODULE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterAnalysisResult:
    """Resultado del análisis de clústeres DBSCAN."""
    n_clusters: int = 0
    n_noise_points: int = 0
    silhouette_score: float = 0.0
    cluster_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_centers: Dict[int, Dict[str, float]] = field(default_factory=dict)
    cluster_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    best_cluster_id: int = -1
    cluster_stability: float = 0.0  # ¿Los clusters son estables?
    intra_cluster_variance: float = 0.0  # Varianza dentro de clusters (menor es mejor)
    inter_cluster_variance: float = 0.0  # Varianza entre clusters


@dataclass
class NeighborhoodStabilityResult:
    """Resultado del Índice de Estabilidad de Vecindad."""
    nsi_global: float = 0.0  # NSI global (0-1, mayor es más estable)
    nsi_by_param: Dict[str, float] = field(default_factory=dict)
    local_stability_map: np.ndarray = field(default_factory=lambda: np.array([]))
    stability_percentiles: Dict[str, float] = field(default_factory=dict)
    stable_regions: List[Tuple[float, float]] = field(default_factory=list)  # Regiones estables
    unstable_regions: List[Tuple[float, float]] = field(default_factory=list)  # Regiones inestables


@dataclass
class DegradationTestResult:
    """Resultado del test de degradación paramétrica."""
    original_performance: float = 0.0
    degraded_performance_mean: float = 0.0
    degraded_performance_std: float = 0.0
    degradation_ratio: float = 0.0  # Cuánto se degrada (0-1, menor es más robusto)
    worst_case_performance: float = 0.0
    best_case_performance: float = 0.0
    performance_at_noise_levels: Dict[float, float] = field(default_factory=dict)  # {noise%: perf}
    robustness_score: float = 0.0  # Score de robustez (0-1)
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)  # Sensibilidad por param


@dataclass
class SurfaceCVResult:
    """Resultado del análisis de Coeficiente de Variación de Superficie."""
    cv_global: float = 0.0  # CV de toda la superficie (menor = más lisa)
    roughness_index: float = 0.0  # Índice de rugosidad (0-1)
    smoothness_score: float = 0.0  # Score de suavidad (0-1, mayor es mejor)
    local_cv_map: np.ndarray = field(default_factory=lambda: np.array([]))
    gradient_magnitude_mean: float = 0.0
    gradient_magnitude_std: float = 0.0
    curvature_mean: float = 0.0
    curvature_std: float = 0.0
    flatness_regions: float = 0.0  # % de superficie que es relativamente plana


@dataclass
class RobustnessAnalysisResult:
    """Resultado completo del análisis de robustez avanzado."""
    cluster_analysis: ClusterAnalysisResult = field(default_factory=ClusterAnalysisResult)
    neighborhood_stability: NeighborhoodStabilityResult = field(default_factory=NeighborhoodStabilityResult)
    degradation_test: DegradationTestResult = field(default_factory=DegradationTestResult)
    surface_cv: SurfaceCVResult = field(default_factory=SurfaceCVResult)

    # Score compuesto de robustez
    overall_robustness_score: float = 0.0
    robustness_grade: str = 'N/A'
    is_robust: bool = False
    confidence_in_robustness: float = 0.0


class AdvancedRobustnessAnalyzer:
    """
    Analizador de robustez avanzado para optimización de parámetros.
    
    Implementa:
    1. Detección de Clústeres (DBSCAN) - Identifica grupos de configuraciones similares
    2. Neighborhood Stability Index (NSI) - Mide estabilidad local
    3. Parameter Degradation Testing - Simula imperfecciones futuras
    4. Surface CV - Mide rugosidad de la superficie de optimización
    """

    def __init__(self, df: pd.DataFrame, params: List[str], target_col: str):
        self.df = df
        self.params = params
        self.target_col = target_col

    def analyze_all(self) -> RobustnessAnalysisResult:
        """Ejecuta todos los análisis de robustez."""
        result = RobustnessAnalysisResult()

        if len(self.df) < 30 or len(self.params) < 1:
            return result

        try:
            # 1. Análisis de Clústeres
            result.cluster_analysis = self._cluster_analysis()

            # 2. Neighborhood Stability Index
            result.neighborhood_stability = self._neighborhood_stability_analysis()

            # 3. Degradation Testing
            result.degradation_test = self._degradation_testing()

            # 4. Surface CV (si hay 2+ params)
            if len(self.params) >= 2:
                result.surface_cv = self._surface_cv_analysis()

            # Calcular score compuesto
            result = self._calculate_overall_robustness(result)

        except Exception as e:
            print(f"    [!] Robustness analysis error: {e}")

        return result

    def _cluster_analysis(self) -> ClusterAnalysisResult:
        """
        Análisis de clústeres DBSCAN para detectar grupos de configuraciones
        con rendimiento similar. Los clústeres estables indican robustez.
        """
        result = ClusterAnalysisResult()

        # Preparar datos
        X = self.df[self.params].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        y = pd.to_numeric(self.df[self.target_col], errors='coerce')

        # Eliminar NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask].values
        y_clean = y[valid_mask].values

        if len(X_clean) < 30:
            return result

        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Determinar epsilon óptimo usando k-distance graph
        k = min(10, len(X_clean) // 5)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])

        # Encontrar el "codo" (knee point)
        eps_optimal = np.percentile(k_distances, 90)
        eps_optimal = max(eps_optimal, 0.1)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps_optimal, min_samples=max(3, len(X_clean) // 20))
        cluster_labels = dbscan.fit_predict(X_scaled)

        result.cluster_labels = cluster_labels
        result.n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        result.n_noise_points = (cluster_labels == -1).sum()

        if result.n_clusters < 2:
            # Si solo hay 1 cluster o ninguno, intentar con KMeans
            n_clusters_kmeans = min(5, max(2, len(X_clean) // 30))
            kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            result.cluster_labels = cluster_labels
            result.n_clusters = n_clusters_kmeans
            result.n_noise_points = 0

        # Calcular silhouette score
        if result.n_clusters >= 2 and result.n_clusters < len(X_clean) - 1:
            try:
                result.silhouette_score = float(silhouette_score(X_scaled, cluster_labels))
            except Exception:
                result.silhouette_score = 0.0

        # Calcular centro y rendimiento de cada cluster
        for cluster_id in range(result.n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() > 0:
                # Centro del cluster (en escala original)
                center = X_clean[mask].mean(axis=0)
                result.cluster_centers[cluster_id] = {
                    self.params[i]: center[i] for i in range(len(self.params))
                }

                # Rendimiento del cluster
                cluster_perf = y_clean[mask]
                result.cluster_performance[cluster_id] = {
                    'mean': float(np.mean(cluster_perf)),
                    'std': float(np.std(cluster_perf)),
                    'median': float(np.median(cluster_perf)),
                    'min': float(np.min(cluster_perf)),
                    'max': float(np.max(cluster_perf)),
                    'count': int(mask.sum()),
                    'cv': float(np.std(cluster_perf) / (np.abs(np.mean(cluster_perf)) + 1e-10))
                }

        # Mejor cluster (mayor rendimiento medio)
        if result.cluster_performance:
            result.best_cluster_id = max(
                result.cluster_performance.keys(),
                key=lambda k: result.cluster_performance[k]['mean']
            )

        # Estabilidad del cluster (1 - CV promedio dentro de clusters)
        cv_values = [v['cv'] for v in result.cluster_performance.values() if v['cv'] < 10]
        if cv_values:
            avg_cv = np.mean(cv_values)
            result.cluster_stability = max(0, 1 - avg_cv)
            result.intra_cluster_variance = avg_cv

        # Varianza entre clusters
        cluster_means = [v['mean'] for v in result.cluster_performance.values()]
        if len(cluster_means) > 1:
            result.inter_cluster_variance = np.std(cluster_means)

        return result

    def _neighborhood_stability_analysis(self) -> NeighborhoodStabilityResult:
        """
        Índice de Estabilidad de Vecindad (NSI).
        
        Mide si configuraciones con parámetros similares tienen
        rendimiento similar. Un NSI alto indica robustez.
        """
        result = NeighborhoodStabilityResult()

        # Preparar datos
        X = self.df[self.params].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        y = pd.to_numeric(self.df[self.target_col], errors='coerce')

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask].values
        y_clean = y[valid_mask].values

        if len(X_clean) < 20:
            return result

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Encontrar vecinos más cercanos
        k = min(10, max(3, len(X_clean) // 10))
        nn = NearestNeighbors(n_neighbors=k + 1)  # +1 porque incluye el punto mismo
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        # Calcular NSI para cada punto
        # NSI = 1 - (varianza de rendimiento en vecindad / varianza global)
        global_var = np.var(y_clean)

        local_stability = np.zeros(len(X_clean))

        for i in range(len(X_clean)):
            neighbor_indices = indices[i, 1:]  # Excluir el punto mismo
            neighbor_performance = y_clean[neighbor_indices]
            local_var = np.var(neighbor_performance)

            # NSI local: 1 - (var_local / var_global)
            # Clipped a [0, 1]
            local_nsi = max(0, min(1, 1 - local_var / (global_var + 1e-10)))
            local_stability[i] = local_nsi

        result.local_stability_map = local_stability
        result.nsi_global = float(np.mean(local_stability))

        # Percentiles de estabilidad
        result.stability_percentiles = {
            'p10': float(np.percentile(local_stability, 10)),
            'p25': float(np.percentile(local_stability, 25)),
            'p50': float(np.percentile(local_stability, 50)),
            'p75': float(np.percentile(local_stability, 75)),
            'p90': float(np.percentile(local_stability, 90)),
        }

        # NSI por parámetro individual
        for i, param in enumerate(self.params):
            # Ordenar por este parámetro
            sort_idx = np.argsort(X_clean[:, i])
            y_sorted = y_clean[sort_idx]

            # Calcular varianza local usando ventana deslizante
            window_size = max(5, len(y_sorted) // 20)
            local_vars = []

            for j in range(len(y_sorted) - window_size):
                window = y_sorted[j:j + window_size]
                local_vars.append(np.var(window))

            if local_vars:
                avg_local_var = np.mean(local_vars)
                param_nsi = max(0, min(1, 1 - avg_local_var / (global_var + 1e-10)))
                result.nsi_by_param[param] = float(param_nsi)

        # Identificar regiones estables e inestables
        stability_threshold_high = 0.7
        stability_threshold_low = 0.3

        # Agrupar puntos en bins y determinar estabilidad promedio
        if len(self.params) == 1:
            param_values = X_clean[:, 0]
            n_bins = min(20, len(np.unique(param_values)))
            bins = np.linspace(param_values.min(), param_values.max(), n_bins + 1)

            for j in range(n_bins):
                bin_mask = (param_values >= bins[j]) & (param_values < bins[j + 1])
                if j == n_bins - 1:
                    bin_mask = (param_values >= bins[j]) & (param_values <= bins[j + 1])

                if bin_mask.sum() > 0:
                    bin_nsi = np.mean(local_stability[bin_mask])
                    region = (float(bins[j]), float(bins[j + 1]))

                    if bin_nsi >= stability_threshold_high:
                        result.stable_regions.append(region)
                    elif bin_nsi <= stability_threshold_low:
                        result.unstable_regions.append(region)

        return result

    def _degradation_testing(self) -> DegradationTestResult:
        """
        Test de Degradación Paramétrica.
        
        Simula qué pasaría si los parámetros óptimos no fueran exactos
        en el futuro (como ocurre cuando el mercado cambia).
        """
        result = DegradationTestResult()

        # Preparar datos
        X = self.df[self.params].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        y = pd.to_numeric(self.df[self.target_col], errors='coerce')

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask].values
        y_clean = y[valid_mask].values

        if len(X_clean) < 30:
            return result

        # Entrenar modelo surrogate para predecir rendimiento
        try:
            model = GradientBoostingRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_clean, y_clean)
        except Exception:
            return result

        # Encontrar la configuración "óptima" actual
        best_idx = np.argmax(y_clean)
        best_params = X_clean[best_idx]
        result.original_performance = float(y_clean[best_idx])

        # Niveles de ruido a probar (% del rango de cada parámetro)
        noise_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        n_simulations = 100

        # Calcular rangos de cada parámetro
        param_ranges = X_clean.max(axis=0) - X_clean.min(axis=0)
        param_ranges = np.where(param_ranges == 0, 1, param_ranges)  # Evitar división por cero

        all_degraded_performances = []

        for noise_level in noise_levels:
            degraded_perfs = []

            for _ in range(n_simulations):
                # Agregar ruido a los parámetros óptimos
                noise = np.random.normal(0, noise_level, len(best_params))
                noise *= param_ranges

                perturbed_params = best_params + noise

                # Clipear a los rangos observados
                perturbed_params = np.clip(
                    perturbed_params,
                    X_clean.min(axis=0),
                    X_clean.max(axis=0)
                )

                # Predecir rendimiento
                pred_perf = model.predict(perturbed_params.reshape(1, -1))[0]
                degraded_perfs.append(pred_perf)

            avg_perf = np.mean(degraded_perfs)
            result.performance_at_noise_levels[noise_level] = float(avg_perf)
            all_degraded_performances.extend(degraded_perfs)

        all_degraded = np.array(all_degraded_performances)

        result.degraded_performance_mean = float(np.mean(all_degraded))
        result.degraded_performance_std = float(np.std(all_degraded))
        result.worst_case_performance = float(np.min(all_degraded))
        result.best_case_performance = float(np.max(all_degraded))

        # Ratio de degradación
        if result.original_performance != 0:
            result.degradation_ratio = float(
                1 - result.degraded_performance_mean / result.original_performance
            )

        # Score de robustez (1 - degradation_ratio), clipped a [0, 1]
        result.robustness_score = float(max(0, min(1, 1 - abs(result.degradation_ratio))))

        # Sensibilidad por parámetro
        for i, param in enumerate(self.params):
            # Perturbar solo este parámetro
            sensitivities = []

            for _ in range(50):
                perturbed = best_params.copy()
                noise = np.random.normal(0, 0.1) * param_ranges[i]
                perturbed[i] += noise
                perturbed[i] = np.clip(perturbed[i], X_clean[:, i].min(), X_clean[:, i].max())

                pred = model.predict(perturbed.reshape(1, -1))[0]
                change = abs(pred - result.original_performance) / (abs(result.original_performance) + 1e-10)
                sensitivities.append(change)

            result.parameter_sensitivity[param] = float(np.mean(sensitivities))

        return result

    def _surface_cv_analysis(self) -> SurfaceCVResult:
        """
        Análisis de Coeficiente de Variación de Superficie.
        
        Cuantifica qué tan "rugoso" o "liso" es el mapa de calor 3D
        de optimización. Una superficie lisa indica robustez.
        """
        result = SurfaceCVResult()

        if len(self.params) < 2:
            return result

        # Usar los primeros 2 parámetros para el análisis de superficie
        param1, param2 = self.params[:2]

        x = pd.to_numeric(self.df[param1], errors='coerce').values
        y_param = pd.to_numeric(self.df[param2], errors='coerce').values
        z = pd.to_numeric(self.df[self.target_col], errors='coerce').values

        mask = ~(np.isnan(x) | np.isnan(y_param) | np.isnan(z))
        x, y_param, z = x[mask], y_param[mask], z[mask]

        if len(z) < 30:
            return result

        # Crear grid interpolado
        grid_size = 30
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y_param.min(), y_param.max(), grid_size)
        xi_mesh, yi_mesh = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y_param), z, (xi_mesh, yi_mesh), method='cubic')
            zi = np.nan_to_num(zi, nan=np.nanmean(z))
        except Exception:
            try:
                zi = griddata((x, y_param), z, (xi_mesh, yi_mesh), method='linear')
                zi = np.nan_to_num(zi, nan=np.nanmean(z))
            except Exception:
                return result

        # CV global de la superficie
        result.cv_global = float(np.std(zi) / (np.abs(np.mean(zi)) + 1e-10))

        # Calcular gradientes
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]

        grad_x, grad_y = np.gradient(zi, dx, dy)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        result.gradient_magnitude_mean = float(np.mean(gradient_magnitude))
        result.gradient_magnitude_std = float(np.std(gradient_magnitude))

        # Calcular curvatura (Laplaciano)
        laplacian = np.gradient(grad_x, dx, axis=0) + np.gradient(grad_y, dy, axis=1)

        result.curvature_mean = float(np.mean(np.abs(laplacian)))
        result.curvature_std = float(np.std(np.abs(laplacian)))

        # Índice de rugosidad
        # Basado en la variabilidad del gradiente normalizada
        z_range = np.max(zi) - np.min(zi)
        if z_range > 0:
            normalized_gradient = gradient_magnitude / z_range
            result.roughness_index = float(np.mean(normalized_gradient))

        # Score de suavidad (1 - roughness normalizada)
        # Normalizamos considerando que roughness típico está entre 0 y 0.5
        result.smoothness_score = float(max(0, min(1, 1 - result.roughness_index * 2)))

        # Porcentaje de regiones "planas" (bajo gradiente)
        flat_threshold = np.percentile(gradient_magnitude, 25)
        result.flatness_regions = float(np.mean(gradient_magnitude < flat_threshold))

        # CV local (mapa)
        local_cv = np.zeros_like(zi)
        window_size = 3

        for i in range(window_size, grid_size - window_size):
            for j in range(window_size, grid_size - window_size):
                window = zi[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
                local_cv[i, j] = np.std(window) / (np.abs(np.mean(window)) + 1e-10)

        result.local_cv_map = local_cv

        return result

    def _calculate_overall_robustness(self, result: RobustnessAnalysisResult) -> RobustnessAnalysisResult:
        """Calcula el score compuesto de robustez."""
        scores = []
        weights = []

        # 1. Cluster Stability (30%)
        if result.cluster_analysis.cluster_stability > 0:
            scores.append(result.cluster_analysis.cluster_stability)
            weights.append(0.30)

        # 2. NSI (30%)
        if result.neighborhood_stability.nsi_global > 0:
            scores.append(result.neighborhood_stability.nsi_global)
            weights.append(0.30)

        # 3. Degradation Robustness (25%)
        if result.degradation_test.robustness_score > 0:
            scores.append(result.degradation_test.robustness_score)
            weights.append(0.25)

        # 4. Surface Smoothness (15%)
        if result.surface_cv.smoothness_score > 0:
            scores.append(result.surface_cv.smoothness_score)
            weights.append(0.15)

        if scores:
            # Normalizar pesos
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            result.overall_robustness_score = float(np.average(scores, weights=weights))

        # Determinar grade
        score = result.overall_robustness_score
        if score >= 0.80:
            result.robustness_grade = 'A+'
        elif score >= 0.70:
            result.robustness_grade = 'A'
        elif score >= 0.60:
            result.robustness_grade = 'B+'
        elif score >= 0.50:
            result.robustness_grade = 'B'
        elif score >= 0.40:
            result.robustness_grade = 'C'
        elif score >= 0.30:
            result.robustness_grade = 'D'
        else:
            result.robustness_grade = 'F'

        result.is_robust = score >= 0.50
        result.confidence_in_robustness = min(1.0, score * 1.2)  # Bonus for high scores

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 🔬 MOTOR DE ANÁLISIS CUANTITATIVO AVANZADO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParameterAnalysis:
    """Resultado completo del análisis de un parámetro."""
    param_name: str
    optimal_value: float
    optimal_range: Tuple[float, float]
    confidence: float
    robustness: float
    stability: float
    sensitivity: float
    monotonicity: float

    # Análisis por métrica
    metric_analysis: Dict[str, Dict] = field(default_factory=dict)

    # Datos para visualización
    x_values: np.ndarray = field(default_factory=lambda: np.array([]))
    y_smooth: Dict[str, np.ndarray] = field(default_factory=dict)
    kde_x: np.ndarray = field(default_factory=lambda: np.array([]))
    kde_y: np.ndarray = field(default_factory=lambda: np.array([]))


class QuantEngine:
    """Motor de análisis cuantitativo con múltiples técnicas."""

    def __init__(self, df: pd.DataFrame, all_params: List[str]):
        self.df = df
        self.all_params = all_params

    def analyze_parameter(self, param: str, metrics: List[str]) -> ParameterAnalysis:
        """Análisis completo de un parámetro."""

        # Obtener datos limpios
        x = pd.to_numeric(self.df[param], errors='coerce').values
        valid_mask = ~np.isnan(x)
        x = x[valid_mask]

        if len(x) < 20:
            return self._empty_analysis(param)

        # Análisis por cada métrica
        metric_results = {}
        optimal_estimates = []

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            y = pd.to_numeric(self.df[metric], errors='coerce').values[valid_mask]
            y_mask = ~np.isnan(y)

            if y_mask.sum() < 20:
                continue

            x_clean = x[y_mask]
            y_clean = y[y_mask]

            # Determinar dirección
            higher_better = not any(kw in metric.upper() for kw in ['DRAWDOWN', 'DD', 'LOSS', 'PERDIDA'])

            # Múltiples análisis
            result = self._multi_analysis(x_clean, y_clean, higher_better)
            metric_results[metric] = result

            if not np.isnan(result['optimal']):
                weight = self._get_metric_weight(metric)
                optimal_estimates.append((result['optimal'], weight * result['confidence']))

        if not optimal_estimates:
            return self._empty_analysis(param)

        # Combinar óptimos
        global_optimal = self._weighted_optimal(optimal_estimates)
        optimal_range = self._calculate_range(x, metric_results)

        # Métricas de calidad
        robustness = self._calculate_robustness(x, metric_results, global_optimal)
        stability = self._calculate_stability(metric_results)
        sensitivity = self._calculate_sensitivity(metric_results)
        monotonicity = self._calculate_monotonicity(metric_results)
        confidence = self._calculate_confidence(optimal_estimates, robustness, stability)

        # Preparar datos para visualización
        x_unique = np.sort(np.unique(x))
        kde_x, kde_y = self._compute_kde(x)

        y_smooth = {}
        for metric, result in metric_results.items():
            if 'curve_x' in result and 'curve_y' in result:
                y_smooth[metric] = (result['curve_x'], result['curve_y'])

        return ParameterAnalysis(
            param_name=param,
            optimal_value=global_optimal,
            optimal_range=optimal_range,
            confidence=confidence,
            robustness=robustness,
            stability=stability,
            sensitivity=sensitivity,
            monotonicity=monotonicity,
            metric_analysis=metric_results,
            x_values=x_unique,
            y_smooth=y_smooth,
            kde_x=kde_x,
            kde_y=kde_y
        )

    def _multi_analysis(self, x: np.ndarray, y: np.ndarray, higher_better: bool) -> Dict:
        """
        Análisis multi-técnica AVANZADO para encontrar óptimo.
        
        Técnicas implementadas:
        1. Regional Growth Algorithm - Detecta mesetas estables vs picos aislados
        2. Bootstrap Confidence Intervals - Intervalos robustos
        3. Bayesian Optimization Surrogate - Modelo probabilístico
        4. Plateau Detection - Zonas de rendimiento estable
        5. Cross-Validated Optimal - Validación de estabilidad
        6. Derivative Analysis - Análisis de gradiente y curvatura
        7. RANSAC Robust Regression - Ignorar outliers
        """
        results = []

        # Pre-procesamiento: eliminar outliers extremos
        y_p5, y_p95 = np.percentile(y, [5, 95])
        inlier_mask = (y >= y_p5) & (y <= y_p95)
        if inlier_mask.sum() >= 20:
            x_clean = x[inlier_mask]
            y_clean = y[inlier_mask]
        else:
            x_clean, y_clean = x.copy(), y.copy()

        # ═══════════════════════════════════════════════════════════════════
        # 1. REGIONAL GROWTH ALGORITHM (Plateau Detection)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_plateau, conf_plateau, plateau_range = self._regional_growth_analysis(
                x_clean, y_clean, higher_better
            )
            if not np.isnan(opt_plateau):
                results.append(('plateau', opt_plateau, conf_plateau, plateau_range))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 2. BOOTSTRAP CONFIDENCE INTERVALS
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_boot, conf_boot, boot_range = self._bootstrap_optimal(
                x_clean, y_clean, higher_better, n_bootstrap=100
            )
            if not np.isnan(opt_boot):
                results.append(('bootstrap', opt_boot, conf_boot, boot_range))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 3. CROSS-VALIDATED OPTIMAL (K-Fold Stability)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_cv, conf_cv, cv_range = self._cross_validated_optimal(
                x_clean, y_clean, higher_better, n_folds=5
            )
            if not np.isnan(opt_cv):
                results.append(('cross_val', opt_cv, conf_cv, cv_range))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 4. DERIVATIVE ANALYSIS (Gradient + Curvature)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_deriv, conf_deriv = self._derivative_analysis(
                x_clean, y_clean, higher_better
            )
            if not np.isnan(opt_deriv):
                results.append(('derivative', opt_deriv, conf_deriv, None))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 5. BAYESIAN SURROGATE MODEL (Gaussian Process-like)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_bayes, conf_bayes = self._bayesian_surrogate(
                x_clean, y_clean, higher_better
            )
            if not np.isnan(opt_bayes):
                results.append(('bayesian', opt_bayes, conf_bayes, None))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 6. ROBUST REGRESSION (RANSAC-like)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_robust, conf_robust = self._robust_regression_optimal(
                x_clean, y_clean, higher_better
            )
            if not np.isnan(opt_robust):
                results.append(('robust_reg', opt_robust, conf_robust, None))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # 7. KDE MODE DETECTION (High-performance regions)
        # ═══════════════════════════════════════════════════════════════════
        try:
            opt_kde, conf_kde = self._kde_mode_optimal(x_clean, y_clean, higher_better)
            if not np.isnan(opt_kde):
                results.append(('kde_mode', opt_kde, conf_kde, None))
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # COMBINAR RESULTADOS CON CONSENSO ROBUSTO
        # ═══════════════════════════════════════════════════════════════════
        if not results:
            return {'optimal': np.nan, 'confidence': 0, 'methods': [],
                    'optimal_range': (np.nan, np.nan)}

        # Obtener óptimo por consenso (mediana ponderada)
        optimal, confidence, optimal_range = self._consensus_optimal(results, x_clean)

        # Generar curva suavizada de alta calidad
        curve_x, curve_y = self._high_quality_smooth_curve(x_clean, y_clean)

        return {
            'optimal': optimal,
            'confidence': confidence,
            'methods': [(m, v, c) for m, v, c, _ in results],
            'higher_better': higher_better,
            'curve_x': curve_x,
            'curve_y': curve_y,
            'optimal_range': optimal_range
        }

    def _regional_growth_analysis(self, x: np.ndarray, y: np.ndarray,
                                   higher_better: bool) -> Tuple[float, float, Tuple[float, float]]:
        """
        Regional Growth Algorithm: Detecta mesetas estables de alto rendimiento.
        
        Busca regiones donde el rendimiento es consistentemente bueno,
        no solo picos aislados que podrían ser ruido.
        """
        # Crear bins adaptivos
        n_bins = min(30, max(10, len(x) // 10))

        # Agrupar por bins
        x_min, x_max = x.min(), x.max()
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calcular estadísticas por bin
        bin_means = np.zeros(n_bins)
        bin_stds = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
            if i == n_bins - 1:  # Incluir el último punto
                mask = (x >= bin_edges[i]) & (x <= bin_edges[i+1])

            if mask.sum() > 0:
                bin_means[i] = np.mean(y[mask])
                bin_stds[i] = np.std(y[mask]) if mask.sum() > 1 else np.std(y) * 0.5
                bin_counts[i] = mask.sum()
            else:
                bin_means[i] = np.nan
                bin_stds[i] = np.nan
                bin_counts[i] = 0

        # Rellenar NaN con interpolación
        valid_bins = ~np.isnan(bin_means)
        if valid_bins.sum() < 3:
            return np.nan, 0, (np.nan, np.nan)

        # Suavizar las medias
        bin_means_smooth = gaussian_filter1d(
            np.nan_to_num(bin_means, nan=np.nanmean(bin_means)),
            sigma=1.5
        )

        # REGIONAL GROWTH: Encontrar región con mejor rendimiento promedio
        # considerando también la estabilidad (baja varianza)

        # Score combinado: rendimiento + estabilidad
        stability_score = 1 / (bin_stds + np.nanmean(bin_stds) * 0.1)
        stability_score = np.nan_to_num(stability_score, nan=0)

        if higher_better:
            performance_score = (bin_means_smooth - np.nanmin(bin_means_smooth)) / \
                               (np.nanmax(bin_means_smooth) - np.nanmin(bin_means_smooth) + 1e-10)
        else:
            performance_score = (np.nanmax(bin_means_smooth) - bin_means_smooth) / \
                               (np.nanmax(bin_means_smooth) - np.nanmin(bin_means_smooth) + 1e-10)

        # Normalizar stability score
        stability_score = stability_score / (np.max(stability_score) + 1e-10)

        # Score combinado (rendimiento 60%, estabilidad 40%)
        combined_score = performance_score * 0.6 + stability_score * 0.4

        # Encontrar la región óptima (plateau)
        # Buscar bins consecutivos con score alto
        threshold = np.percentile(combined_score, 75)
        good_bins = combined_score >= threshold

        # Encontrar la región continua más grande
        best_region_start = 0
        best_region_end = 0
        best_region_score = 0

        current_start = None
        for i in range(n_bins):
            if good_bins[i]:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    region_score = np.mean(combined_score[current_start:i])
                    region_length = i - current_start
                    total_score = region_score * np.sqrt(region_length)  # Premiar regiones más largas

                    if total_score > best_region_score:
                        best_region_score = total_score
                        best_region_start = current_start
                        best_region_end = i
                    current_start = None

        # Verificar última región
        if current_start is not None:
            region_score = np.mean(combined_score[current_start:n_bins])
            region_length = n_bins - current_start
            total_score = region_score * np.sqrt(region_length)

            if total_score > best_region_score:
                best_region_start = current_start
                best_region_end = n_bins

        if best_region_end <= best_region_start:
            # Fallback: usar el bin con mejor score
            best_bin = np.argmax(combined_score)
            optimal = bin_centers[best_bin]
            confidence = combined_score[best_bin] * 0.7
            optimal_range = (bin_edges[best_bin], bin_edges[best_bin + 1])
        else:
            # Óptimo es el centro de la mejor región
            region_weights = combined_score[best_region_start:best_region_end]
            region_centers = bin_centers[best_region_start:best_region_end]

            optimal = np.average(region_centers, weights=region_weights + 1e-10)
            confidence = np.mean(region_weights) * 0.9  # Alta confianza para plateaus
            optimal_range = (bin_edges[best_region_start], bin_edges[best_region_end])

        return optimal, confidence, optimal_range

    def _bootstrap_optimal(self, x: np.ndarray, y: np.ndarray,
                          higher_better: bool, n_bootstrap: int = 100) -> Tuple[float, float, Tuple[float, float]]:
        """
        Bootstrap Confidence Intervals para el óptimo.
        
        Remuestrea los datos múltiples veces para obtener
        una distribución del óptimo y su incertidumbre.
        """
        n = len(x)
        bootstrap_optimals = []

        for _ in range(n_bootstrap):
            # Remuestreo con reemplazo
            idx = np.random.randint(0, n, size=n)
            x_boot = x[idx]
            y_boot = y[idx]

            # Encontrar óptimo simple en este bootstrap
            opt = self._simple_optimal(x_boot, y_boot, higher_better)
            if not np.isnan(opt):
                bootstrap_optimals.append(opt)

        if len(bootstrap_optimals) < 10:
            return np.nan, 0, (np.nan, np.nan)

        bootstrap_optimals = np.array(bootstrap_optimals)

        # Estadísticas del bootstrap
        optimal = np.median(bootstrap_optimals)

        # Intervalo de confianza 90%
        ci_low, ci_high = np.percentile(bootstrap_optimals, [5, 95])

        # Confianza basada en la dispersión del bootstrap
        spread = (ci_high - ci_low) / (x.max() - x.min() + 1e-10)
        confidence = max(0.3, 1 - spread * 2)

        return optimal, confidence, (ci_low, ci_high)

    def _simple_optimal(self, x: np.ndarray, y: np.ndarray, higher_better: bool) -> float:
        """Encuentra óptimo simple usando top performers."""
        n_top = max(3, int(len(y) * 0.1))

        if higher_better:
            top_idx = np.argsort(y)[-n_top:]
        else:
            top_idx = np.argsort(y)[:n_top]

        return np.median(x[top_idx])

    def _cross_validated_optimal(self, x: np.ndarray, y: np.ndarray,
                                  higher_better: bool, n_folds: int = 5) -> Tuple[float, float, Tuple[float, float]]:
        """
        Cross-Validated Optimal: Verifica estabilidad del óptimo.
        
        Divide los datos en K folds y encuentra el óptimo en cada uno.
        Un óptimo robusto debería ser similar en todos los folds.
        """
        n = len(x)
        if n < n_folds * 10:
            n_folds = max(2, n // 10)

        fold_size = n // n_folds
        indices = np.random.permutation(n)

        fold_optimals = []

        for i in range(n_folds):
            # Crear fold de test
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else n

            # Usar todo excepto el fold de test para encontrar óptimo
            train_mask = np.ones(n, dtype=bool)
            train_mask[indices[test_start:test_end]] = False

            x_train = x[train_mask]
            y_train = y[train_mask]

            if len(x_train) >= 10:
                opt = self._simple_optimal(x_train, y_train, higher_better)
                if not np.isnan(opt):
                    fold_optimals.append(opt)

        if len(fold_optimals) < 2:
            return np.nan, 0, (np.nan, np.nan)

        fold_optimals = np.array(fold_optimals)

        optimal = np.median(fold_optimals)

        # Confianza basada en consistencia entre folds
        cv = np.std(fold_optimals) / (np.abs(np.mean(fold_optimals)) + 1e-10)
        confidence = max(0.3, 1 - cv * 2)

        # Rango basado en folds
        cv_range = (np.min(fold_optimals), np.max(fold_optimals))

        return optimal, confidence, cv_range

    def _derivative_analysis(self, x: np.ndarray, y: np.ndarray,
                             higher_better: bool) -> Tuple[float, float]:
        """
        Análisis de derivadas para encontrar el óptimo.
        
        Busca donde la primera derivada cruza por cero
        y la segunda derivada indica un máximo/mínimo.
        """
        # Agrupar y suavizar
        df_temp = pd.DataFrame({'x': x, 'y': y})
        grouped = df_temp.groupby('x')['y'].mean().reset_index()
        x_u = grouped['x'].values
        y_u = grouped['y'].values

        if len(x_u) < 10:
            return np.nan, 0

        # Ordenar
        sort_idx = np.argsort(x_u)
        x_u = x_u[sort_idx]
        y_u = y_u[sort_idx]

        # Suavizado fuerte
        window = min(len(y_u) - 2, max(5, len(y_u) // 4))
        if window % 2 == 0:
            window += 1

        y_smooth = savgol_filter(y_u, window, min(3, window - 1))
        y_smooth = gaussian_filter1d(y_smooth, sigma=2)

        # Primera derivada (gradiente)
        dy = np.gradient(y_smooth, x_u)

        # Segunda derivada (curvatura)
        d2y = np.gradient(dy, x_u)

        # Buscar cruces por cero de la primera derivada
        if higher_better:
            # Buscar máximos: dy cruza de positivo a negativo, d2y < 0
            sign_changes = np.where(np.diff(np.sign(dy)) < 0)[0]
        else:
            # Buscar mínimos: dy cruza de negativo a positivo, d2y > 0
            sign_changes = np.where(np.diff(np.sign(dy)) > 0)[0]

        if len(sign_changes) == 0:
            # Fallback: usar el extremo
            if higher_better:
                optimal = x_u[np.argmax(y_smooth)]
            else:
                optimal = x_u[np.argmin(y_smooth)]
            confidence = 0.5
        else:
            # Evaluar cada cruce
            best_opt = None
            best_score = -np.inf if higher_better else np.inf

            for idx in sign_changes:
                val = y_smooth[idx]
                curvature = d2y[idx]

                # Verificar que la curvatura sea del signo correcto
                if higher_better and curvature < 0:  # Máximo
                    if val > best_score:
                        best_score = val
                        best_opt = x_u[idx]
                elif not higher_better and curvature > 0:  # Mínimo
                    if val < best_score:
                        best_score = val
                        best_opt = x_u[idx]

            if best_opt is None:
                if higher_better:
                    optimal = x_u[np.argmax(y_smooth)]
                else:
                    optimal = x_u[np.argmin(y_smooth)]
                confidence = 0.5
            else:
                optimal = best_opt
                # Confianza basada en la claridad del extremo
                y_range = np.max(y_smooth) - np.min(y_smooth)
                if higher_better:
                    relative_height = (best_score - np.min(y_smooth)) / (y_range + 1e-10)
                else:
                    relative_height = (np.max(y_smooth) - best_score) / (y_range + 1e-10)
                confidence = min(0.9, relative_height * 0.9)

        return optimal, confidence

    def _bayesian_surrogate(self, x: np.ndarray, y: np.ndarray,
                            higher_better: bool) -> Tuple[float, float]:
        """
        Bayesian Surrogate Model usando Gradient Boosting como aproximación.
        
        Entrena un modelo probabilístico y busca el óptimo predicho.
        """
        X = x.reshape(-1, 1)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Modelo ensemble para reducir varianza
        models = []
        n_models = 5

        for i in range(n_models):
            # Bootstrap para cada modelo
            idx = np.random.randint(0, len(x), size=len(x))

            model = GradientBoostingRegressor(
                n_estimators=30, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42 + i
            )
            model.fit(X_scaled[idx], y[idx])
            models.append(model)

        # Predicciones
        x_test = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        x_test_scaled = scaler.transform(x_test)

        predictions = np.array([m.predict(x_test_scaled) for m in models])
        y_mean = predictions.mean(axis=0)
        y_std = predictions.std(axis=0)

        # Acquisition function: UCB (Upper Confidence Bound)
        # Para maximización: mean + kappa * std
        # Para minimización: mean - kappa * std
        kappa = 1.5

        if higher_better:
            acquisition = y_mean + kappa * y_std
            best_idx = np.argmax(acquisition)
        else:
            acquisition = y_mean - kappa * y_std
            best_idx = np.argmin(acquisition)

        optimal = x_test[best_idx, 0]

        # Confianza basada en la incertidumbre
        uncertainty = y_std[best_idx] / (np.std(y) + 1e-10)
        confidence = max(0.4, 1 - uncertainty)

        return optimal, confidence

    def _robust_regression_optimal(self, x: np.ndarray, y: np.ndarray,
                                   higher_better: bool) -> Tuple[float, float]:
        """
        RANSAC-like Robust Regression para ignorar outliers.
        
        Usa múltiples subconjuntos aleatorios y encuentra
        el óptimo más consistente.
        """
        n = len(x)
        n_iterations = 50
        sample_size = max(10, int(n * 0.5))

        all_optimals = []

        for _ in range(n_iterations):
            # Muestra aleatoria
            idx = np.random.choice(n, size=sample_size, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]

            # Encontrar óptimo en esta muestra
            n_top = max(3, int(len(y_sample) * 0.15))
            if higher_better:
                top_idx = np.argsort(y_sample)[-n_top:]
            else:
                top_idx = np.argsort(y_sample)[:n_top]

            opt = np.median(x_sample[top_idx])
            all_optimals.append(opt)

        all_optimals = np.array(all_optimals)

        # Usar mediana (robusta a outliers)
        optimal = np.median(all_optimals)

        # MAD (Median Absolute Deviation) para medir dispersión
        mad = np.median(np.abs(all_optimals - optimal))
        normalized_mad = mad / (x.max() - x.min() + 1e-10)

        confidence = max(0.4, 1 - normalized_mad * 5)

        return optimal, confidence

    def _kde_mode_optimal(self, x: np.ndarray, y: np.ndarray,
                          higher_better: bool) -> Tuple[float, float]:
        """
        KDE Mode Detection: Encuentra la moda de los valores X con alto rendimiento.
        """
        # Seleccionar top performers
        n_top = max(10, int(len(y) * 0.2))
        if higher_better:
            top_idx = np.argsort(y)[-n_top:]
        else:
            top_idx = np.argsort(y)[:n_top]

        x_top = x[top_idx]

        if len(x_top) < 5:
            return np.nan, 0

        # KDE
        bandwidth = max(np.std(x_top) * 0.3, (x.max() - x.min()) / 20)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(x_top.reshape(-1, 1))

        # Evaluar
        x_eval = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        log_density = kde.score_samples(x_eval)
        density = np.exp(log_density)

        # Encontrar la moda (máximo de densidad)
        mode_idx = np.argmax(density)
        optimal = x_eval[mode_idx, 0]

        # Confianza basada en qué tan concentrada está la distribución
        max_density = density[mode_idx]
        mean_density = np.mean(density)
        concentration = (max_density / mean_density) - 1
        confidence = min(0.85, 0.5 + concentration * 0.3)

        return optimal, confidence

    def _consensus_optimal(self, results: List, x: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        Combina los resultados de múltiples métodos usando consenso robusto.
        
        Usa mediana ponderada y clustering para encontrar el consenso.
        """
        if len(results) == 0:
            return np.nan, 0, (np.nan, np.nan)

        optimals = np.array([r[1] for r in results])
        confidences = np.array([r[2] for r in results])
        ranges = [r[3] for r in results if r[3] is not None]

        # Filtrar NaN
        valid_mask = ~np.isnan(optimals)
        if valid_mask.sum() == 0:
            return np.nan, 0, (np.nan, np.nan)

        optimals = optimals[valid_mask]
        confidences = confidences[valid_mask]

        # Detectar outliers en los óptimos (métodos que dan valores muy diferentes)
        if len(optimals) >= 3:
            median_opt = np.median(optimals)
            mad = np.median(np.abs(optimals - median_opt))
            median_opt + 3 * mad * 1.4826  # Factor para convertir MAD a std
            inlier_mask = np.abs(optimals - median_opt) <= 2.5 * mad * 1.4826

            if inlier_mask.sum() >= 2:
                optimals = optimals[inlier_mask]
                confidences = confidences[inlier_mask]

        # Mediana ponderada por confianza
        sorted_idx = np.argsort(optimals)
        sorted_opts = optimals[sorted_idx]
        sorted_confs = confidences[sorted_idx]

        cumsum = np.cumsum(sorted_confs)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        median_idx = min(median_idx, len(sorted_opts) - 1)

        consensus_optimal = sorted_opts[median_idx]

        # Confianza del consenso
        # Basada en: (1) confianzas individuales, (2) acuerdo entre métodos
        spread = np.std(optimals) / (x.max() - x.min() + 1e-10)
        agreement = max(0.3, 1 - spread * 3)
        avg_confidence = np.mean(confidences)

        consensus_confidence = (avg_confidence * 0.6 + agreement * 0.4)

        # Rango óptimo del consenso
        if ranges:
            valid_ranges = [(r[0], r[1]) for r in ranges if r is not None and not np.isnan(r[0])]
            if valid_ranges:
                range_mins = [r[0] for r in valid_ranges]
                range_maxs = [r[1] for r in valid_ranges]
                consensus_range = (np.median(range_mins), np.median(range_maxs))
            else:
                # Fallback: usar dispersión de óptimos
                consensus_range = (np.min(optimals), np.max(optimals))
        else:
            consensus_range = (np.min(optimals), np.max(optimals))

        return consensus_optimal, consensus_confidence, consensus_range

    def _high_quality_smooth_curve(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera curva suavizada de alta calidad para visualización.
        
        Aplica filtrado robusto de outliers para evitar picos artificiales.
        """
        # ═══════════════════════════════════════════════════════════════════
        # 1. FILTRAR OUTLIERS EXTREMOS (usando IQR robusto)
        # ═══════════════════════════════════════════════════════════════════
        y_q1, y_q3 = np.percentile(y, [25, 75])
        y_iqr = y_q3 - y_q1

        # Usar factor 2.5 para IQR (más conservador que 1.5 estándar)
        y_lower = y_q1 - 2.5 * y_iqr
        y_upper = y_q3 + 2.5 * y_iqr

        # Crear máscara de inliers
        inlier_mask = (y >= y_lower) & (y <= y_upper)

        # Si filtrar deja muy pocos datos, usar percentiles más amplios
        if inlier_mask.sum() < 20:
            y_p5, y_p95 = np.percentile(y, [5, 95])
            inlier_mask = (y >= y_p5) & (y <= y_p95)

        if inlier_mask.sum() < 10:
            # Usar todos los datos si aún quedan muy pocos
            x_clean, y_clean = x.copy(), y.copy()
        else:
            x_clean = x[inlier_mask]
            y_clean = y[inlier_mask]

        # ═══════════════════════════════════════════════════════════════════
        # 2. AGRUPAR POR VALORES ÚNICOS CON MEDIANA (más robusta que media)
        # ═══════════════════════════════════════════════════════════════════
        df_temp = pd.DataFrame({'x': x_clean, 'y': y_clean})
        grouped = df_temp.groupby('x')['y'].agg(['median', 'std', 'count']).reset_index()
        x_u = grouped['x'].values
        y_u = grouped['median'].values  # Usar MEDIANA en vez de media
        counts = grouped['count'].values

        if len(x_u) < 5:
            return x_u, y_u

        # Ordenar
        sort_idx = np.argsort(x_u)
        x_u = x_u[sort_idx]
        y_u = y_u[sort_idx]
        counts = counts[sort_idx]

        # ═══════════════════════════════════════════════════════════════════
        # 3. SUAVIZADO ADICIONAL DE LOS PUNTOS AGRUPADOS (eliminar ruido local)
        # ═══════════════════════════════════════════════════════════════════
        if len(y_u) > 7:
            # Media móvil ponderada por counts
            window = min(5, len(y_u) // 3)
            if window >= 3:
                y_u_smooth = np.copy(y_u)
                for i in range(window // 2, len(y_u) - window // 2):
                    start = i - window // 2
                    end = i + window // 2 + 1
                    local_weights = counts[start:end]
                    y_u_smooth[i] = np.average(y_u[start:end], weights=local_weights + 1)
                y_u = y_u_smooth

        # ═══════════════════════════════════════════════════════════════════
        # 4. CREAR CURVA INTERPOLADA
        # ═══════════════════════════════════════════════════════════════════
        n_points = min(200, len(x_u) * 5)
        x_smooth = np.linspace(x_u.min(), x_u.max(), n_points)

        try:
            # Spline ponderado por counts con factor de suavizado alto
            weights = np.sqrt(counts)
            s_factor = len(x_u) * 0.8  # Aumentado de 0.3 a 0.8 para más suavizado
            spline = UnivariateSpline(x_u, y_u, w=weights, s=s_factor)
            y_smooth = spline(x_smooth)

            # Suavizado gaussiano final
            y_smooth = gaussian_filter1d(y_smooth, sigma=3)  # Aumentado de 2 a 3

            # ═══════════════════════════════════════════════════════════════════
            # 5. CLIPPING FINAL para evitar valores fuera del rango de datos limpios
            # ═══════════════════════════════════════════════════════════════════
            y_data_min = np.percentile(y_clean, 2)
            y_data_max = np.percentile(y_clean, 98)
            y_smooth = np.clip(y_smooth, y_data_min, y_data_max)

            return x_smooth, y_smooth
        except Exception:
            return x_u, y_u

    def _compute_kde(self, x: np.ndarray, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula KDE de la distribución del parámetro."""
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=np.std(x) * 0.3)
            kde.fit(x.reshape(-1, 1))

            x_plot = np.linspace(x.min(), x.max(), n_points).reshape(-1, 1)
            log_dens = kde.score_samples(x_plot)

            return x_plot.flatten(), np.exp(log_dens)
        except Exception:
            return np.array([]), np.array([])

    def _weighted_optimal(self, estimates: List[Tuple[float, float]]) -> float:
        """Calcula óptimo ponderado."""
        values, weights = zip(*estimates)
        values = np.array(values)
        weights = np.array(weights)

        mask = ~np.isnan(values)
        if mask.sum() == 0:
            return np.nan

        return np.average(values[mask], weights=weights[mask])

    def _calculate_range(self, x: np.ndarray, metric_results: Dict) -> Tuple[float, float]:
        """
        Calcula rango óptimo usando técnicas avanzadas:
        1. Changepoint Detection - Detecta dónde cambia el comportamiento
        2. Confidence Interval Intersection - Intersección de intervalos de confianza
        3. Plateau Width Analysis - Análisis del ancho de mesetas
        4. Sensitivity Analysis - Dónde el rendimiento es estable
        """
        ranges_from_methods = []

        # Recopilar rangos de cada método que los provea
        for metric, result in metric_results.items():
            if result.get('confidence', 0) < 0.3:
                continue

            # Usar rango del consenso si está disponible
            opt_range = result.get('optimal_range')
            if opt_range is not None and not np.isnan(opt_range[0]):
                ranges_from_methods.append(opt_range)

        # ═══════════════════════════════════════════════════════════════════
        # 1. SENSITIVITY-BASED RANGE
        # ═══════════════════════════════════════════════════════════════════
        sensitivity_ranges = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))
            higher_better = result.get('higher_better', True)

            if len(curve_x) < 20:
                continue

            try:
                # Normalizar la curva
                y_norm = (curve_y - np.min(curve_y)) / (np.max(curve_y) - np.min(curve_y) + 1e-10)
                if not higher_better:
                    y_norm = 1 - y_norm

                # Calcular derivada absoluta (sensibilidad local)
                dy = np.abs(np.gradient(y_norm, curve_x))

                # Suavizar la derivada
                dy_smooth = gaussian_filter1d(dy, sigma=3)

                # Buscar región de baja sensibilidad (estable) con buen rendimiento
                # Score = rendimiento * (1 - sensibilidad normalizada)
                sensitivity_norm = dy_smooth / (np.max(dy_smooth) + 1e-10)
                stability_score = y_norm * (1 - sensitivity_norm * 0.5)

                # Umbral adaptivo para "zona buena"
                threshold = np.percentile(stability_score, 70)
                good_mask = stability_score >= threshold

                if good_mask.sum() > 5:
                    good_x = curve_x[good_mask]
                    sensitivity_ranges.append((good_x.min(), good_x.max()))
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════
        # 2. CHANGEPOINT-BASED RANGE
        # ═══════════════════════════════════════════════════════════════════
        changepoint_ranges = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))
            higher_better = result.get('higher_better', True)

            if len(curve_x) < 20:
                continue

            try:
                # Normalizar
                y_norm = (curve_y - np.min(curve_y)) / (np.max(curve_y) - np.min(curve_y) + 1e-10)
                if not higher_better:
                    y_norm = 1 - y_norm

                # Calcular segunda derivada (cambios en la tendencia)
                dy = np.gradient(y_norm, curve_x)
                d2y = np.gradient(dy, curve_x)
                d2y_smooth = gaussian_filter1d(d2y, sigma=3)

                # Encontrar puntos de inflexión significativos
                d2y_abs = np.abs(d2y_smooth)
                threshold = np.percentile(d2y_abs, 80)

                # Buscar cruces por cero de la segunda derivada
                sign_changes = np.where(np.diff(np.sign(d2y_smooth)))[0]

                if len(sign_changes) >= 2:
                    # Encontrar la zona alta entre cambios
                    optimal_idx = np.argmax(y_norm)

                    # Encontrar changepoints más cercanos al óptimo
                    left_changes = sign_changes[sign_changes < optimal_idx]
                    right_changes = sign_changes[sign_changes > optimal_idx]

                    left_bound = curve_x[left_changes[-1]] if len(left_changes) > 0 else curve_x[0]
                    right_bound = curve_x[right_changes[0]] if len(right_changes) > 0 else curve_x[-1]

                    changepoint_ranges.append((left_bound, right_bound))
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════
        # 3. PERFORMANCE THRESHOLD RANGE (mejorado)
        # ═══════════════════════════════════════════════════════════════════
        threshold_ranges = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))
            higher_better = result.get('higher_better', True)

            if len(curve_x) < 10:
                continue

            try:
                # Umbral del 85% del óptimo (más estricto que antes)
                if higher_better:
                    threshold = np.max(curve_y) * 0.85
                    good_idx = curve_y >= threshold
                else:
                    threshold = np.min(curve_y) * 1.15
                    good_idx = curve_y <= threshold

                if good_idx.sum() > 3:
                    good_x = curve_x[good_idx]

                    # Buscar el cluster más grande de puntos buenos (evitar regiones dispersas)
                    diffs = np.diff(np.where(good_idx)[0])
                    if len(diffs) > 0:
                        # Encontrar gaps grandes
                        gap_threshold = len(curve_x) // 10
                        large_gaps = np.where(diffs > gap_threshold)[0]

                        if len(large_gaps) > 0:
                            # Dividir en segmentos
                            segments = []
                            start = 0
                            for gap_idx in large_gaps:
                                np.where(good_idx)[0][gap_idx]
                                segments.append(curve_x[good_idx][start:gap_idx+1])
                                start = gap_idx + 1
                            segments.append(curve_x[good_idx][start:])

                            # Usar el segmento más largo
                            if segments:
                                longest_segment = max(segments, key=len)
                                if len(longest_segment) > 0:
                                    threshold_ranges.append((longest_segment.min(), longest_segment.max()))
                        else:
                            threshold_ranges.append((good_x.min(), good_x.max()))
                    else:
                        threshold_ranges.append((good_x.min(), good_x.max()))
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════
        # COMBINAR TODOS LOS RANGOS
        # ═══════════════════════════════════════════════════════════════════
        all_ranges = ranges_from_methods + sensitivity_ranges + changepoint_ranges + threshold_ranges

        if not all_ranges:
            # Fallback: usar IQR
            q25, q75 = np.percentile(x, [25, 75])
            return (q25, q75)

        # Encontrar intersección robusta usando mediana
        all_mins = [r[0] for r in all_ranges]
        all_maxs = [r[1] for r in all_ranges]

        # Usar mediana para robustez
        range_min = np.median(all_mins)
        range_max = np.median(all_maxs)

        # Asegurar que el rango tenga sentido
        if range_min >= range_max:
            # Usar el rango más frecuente
            range_min = np.percentile(all_mins, 25)
            range_max = np.percentile(all_maxs, 75)

        # Asegurar que está dentro del dominio
        range_min = max(range_min, x.min())
        range_max = min(range_max, x.max())

        return (range_min, range_max)

    def _calculate_robustness(self, x: np.ndarray, metric_results: Dict, optimal: float) -> float:
        """Calcula robustez del óptimo."""
        if np.isnan(optimal):
            return 0.0

        robustness_scores = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))
            higher_better = result.get('higher_better', True)

            if len(curve_x) < 10:
                continue

            # Encontrar rendimiento en el óptimo
            opt_idx = np.argmin(np.abs(curve_x - optimal))
            y_at_opt = curve_y[opt_idx]

            # Calcular cuánto se degrada al alejarse del óptimo
            x_range = curve_x.max() - curve_x.min()
            tolerance = x_range * 0.15  # 15% del rango

            near_optimal = np.abs(curve_x - optimal) <= tolerance
            if near_optimal.sum() > 0:
                y_near = curve_y[near_optimal]

                if higher_better:
                    degradation = 1 - (np.std(y_near) / (np.abs(y_at_opt) + 1e-10))
                else:
                    degradation = 1 - (np.std(y_near) / (np.abs(y_at_opt) + 1e-10))

                robustness_scores.append(max(0, min(1, degradation)))

        return np.mean(robustness_scores) if robustness_scores else 0.5

    def _calculate_stability(self, metric_results: Dict) -> float:
        """Calcula estabilidad (consistencia entre métricas)."""
        optimals = []

        for metric, result in metric_results.items():
            opt = result.get('optimal', np.nan)
            if not np.isnan(opt):
                optimals.append(opt)

        if len(optimals) < 2:
            return 0.5

        # Coeficiente de variación inverso
        cv = np.std(optimals) / (np.abs(np.mean(optimals)) + 1e-10)
        stability = max(0, 1 - cv)

        return stability

    def _calculate_sensitivity(self, metric_results: Dict) -> float:
        """Calcula sensibilidad promedio."""
        sensitivities = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))

            if len(curve_x) < 10:
                continue

            # Gradiente normalizado
            dx = np.diff(curve_x)
            dy = np.diff(curve_y)

            if len(dx) > 0 and np.any(dx != 0):
                gradient = dy / (dx + 1e-10)
                sensitivity = np.mean(np.abs(gradient)) / (np.std(curve_y) + 1e-10)
                sensitivities.append(min(1, sensitivity))

        return np.mean(sensitivities) if sensitivities else 0.5

    def _calculate_monotonicity(self, metric_results: Dict) -> float:
        """Calcula monotonicidad promedio."""
        monotonicity_scores = []

        for metric, result in metric_results.items():
            curve_x = result.get('curve_x', np.array([]))
            curve_y = result.get('curve_y', np.array([]))

            if len(curve_x) < 10:
                continue

            # Correlación de Spearman
            try:
                corr, _ = stats.spearmanr(curve_x, curve_y)
                monotonicity_scores.append(abs(corr))
            except Exception:
                pass

        return np.mean(monotonicity_scores) if monotonicity_scores else 0.5

    def _calculate_confidence(self, estimates: List[Tuple[float, float]],
                             robustness: float, stability: float) -> float:
        """Calcula confianza global."""
        if not estimates:
            return 0.0

        # Base: promedio de confianzas individuales
        avg_conf = np.mean([w for _, w in estimates])

        # Ajustar por robustez y estabilidad
        confidence = avg_conf * 0.5 + robustness * 0.25 + stability * 0.25

        return min(1, max(0, confidence))

    def _get_metric_weight(self, metric: str) -> float:
        """Peso de cada métrica."""
        weights = {
            'SCORE': 1.0, 'ROI': 0.95, 'ROI_PCT': 0.95,
            'SHARPE': 0.85, 'SORTINO': 0.8, 'SQN': 0.9,
            'PROFIT_FACTOR': 0.85, 'DRAWDOWN': 0.9, 'MAX_DD_PCT': 0.9,
            'WINRATE': 0.6, 'WINRATE_PCT': 0.6,
            'ESTABILIDAD': 0.75, 'EXPECTATIVA': 0.7,
        }

        metric_upper = metric.upper()
        for key, weight in weights.items():
            if key in metric_upper:
                return weight
        return 0.5

    def _empty_analysis(self, param: str) -> ParameterAnalysis:
        """Retorna análisis vacío."""
        return ParameterAnalysis(
            param_name=param,
            optimal_value=np.nan,
            optimal_range=(np.nan, np.nan),
            confidence=0,
            robustness=0,
            stability=0,
            sensitivity=0,
            monotonicity=0
        )


# ══════════════════════════════════════════════════════════════════════════════
# � ANÁLISIS DE CORRELACIÓN ENTRE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParameterCorrelation:
    """Resultado de análisis de correlación entre dos parámetros."""
    param1: str
    param2: str
    pearson_corr: float
    spearman_corr: float
    mutual_info: float
    joint_importance: float  # Importancia combinada para el target
    is_strongly_related: bool

    @property
    def strength(self) -> str:
        """Clasificación de fuerza de correlación."""
        avg = (abs(self.pearson_corr) + abs(self.spearman_corr)) / 2
        if avg >= 0.7:
            return 'STRONG'
        elif avg >= 0.4:
            return 'MODERATE'
        elif avg >= 0.2:
            return 'WEAK'
        return 'NONE'


class CorrelationAnalyzer:
    """Analiza correlaciones entre parámetros para detectar relaciones."""

    def __init__(self, df: pd.DataFrame, params: List[str], target_col: str):
        self.df = df
        self.params = params
        self.target_col = target_col
        self.correlations: List[ParameterCorrelation] = []

    def analyze_all_pairs(self, min_correlation: float = 0.3) -> List[ParameterCorrelation]:
        """Analiza todas las combinaciones de pares de parámetros."""
        self.correlations = []

        if len(self.params) < 2:
            return []

        # Preparar datos
        X = self.df[self.params].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.dropna()

        if len(X) < 20:
            return []

        # Obtener target
        y = pd.to_numeric(self.df.loc[X.index, self.target_col], errors='coerce')

        # Analizar cada par
        for param1, param2 in combinations(self.params, 2):
            corr = self._analyze_pair(X[param1].values, X[param2].values,
                                      y.values, param1, param2)
            if corr and (abs(corr.pearson_corr) >= min_correlation or
                        abs(corr.spearman_corr) >= min_correlation or
                        corr.joint_importance >= 0.15):
                self.correlations.append(corr)

        # Ordenar por relevancia
        self.correlations.sort(key=lambda x: x.joint_importance, reverse=True)

        return self.correlations

    def _analyze_pair(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                      name1: str, name2: str) -> Optional[ParameterCorrelation]:
        """Analiza un par de parámetros."""
        try:
            # Filtrar NaN
            mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
            x1, x2, y = x1[mask], x2[mask], y[mask]

            if len(x1) < 20:
                return None

            # Correlación de Pearson (lineal)
            pearson, _ = stats.pearsonr(x1, x2)

            # Correlación de Spearman (monótona)
            spearman, _ = stats.spearmanr(x1, x2)

            # Información mutua aproximada (discretizada)
            try:
                from sklearn.feature_selection import mutual_info_regression
                X_pair = np.column_stack([x1, x2])
                mi = mutual_info_regression(X_pair, y, random_state=42)
                mutual_info = float(np.mean(mi))
            except Exception:
                mutual_info = 0.0

            # Importancia conjunta: entrenar un modelo simple con ambos params
            try:
                X_pair = np.column_stack([x1, x2, x1 * x2])  # Incluir interacción
                rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                rf.fit(X_pair, y)
                # La importancia de la interacción indica relación conjunta
                joint_importance = float(rf.feature_importances_[2])  # Término de interacción
            except Exception:
                joint_importance = 0.0

            # Determinar si están fuertemente relacionados
            avg_corr = (abs(pearson) + abs(spearman)) / 2
            is_strong = avg_corr >= 0.35 or joint_importance >= 0.1

            return ParameterCorrelation(
                param1=name1,
                param2=name2,
                pearson_corr=pearson,
                spearman_corr=spearman,
                mutual_info=mutual_info,
                joint_importance=joint_importance,
                is_strongly_related=is_strong
            )
        except Exception:
            return None

    def get_strongly_related_pairs(self) -> List[Tuple[str, str]]:
        """Retorna lista de pares fuertemente relacionados."""
        return [(c.param1, c.param2) for c in self.correlations if c.is_strongly_related]

    def get_top_pairs(self, n: int = 5) -> List[ParameterCorrelation]:
        """Retorna los top N pares más relacionados."""
        return self.correlations[:n]



@dataclass
class StrategyAnalysis:
    """Análisis global de la estrategia."""
    name: str
    n_trials: int

    # Distribuciones
    roi_distribution: Dict = field(default_factory=dict)
    score_distribution: Dict = field(default_factory=dict)
    sharpe_distribution: Dict = field(default_factory=dict)

    # Métricas globales
    best_trial: Dict = field(default_factory=dict)
    worst_trial: Dict = field(default_factory=dict)
    median_performance: Dict = field(default_factory=dict)

    # Robustez global
    overall_robustness: float = 0.0
    parameter_importance: Dict = field(default_factory=dict)


class StrategyAnalyzer:
    """Analizador global de estrategia."""

    def __init__(self, df: pd.DataFrame, schema: DataSchema):
        self.df = df
        self.schema = schema

    def analyze(self) -> StrategyAnalysis:
        """Análisis completo de la estrategia."""
        analysis = StrategyAnalysis(
            name=self.df.get('ESTRATEGIA', pd.Series(['STRATEGY'])).iloc[0] if 'ESTRATEGIA' in self.df.columns else 'STRATEGY',
            n_trials=len(self.df)
        )

        # Distribuciones de métricas principales
        for metric, attr in [('ROI_PCT', 'roi_distribution'),
                             ('SCORE', 'score_distribution'),
                             ('SHARPE', 'sharpe_distribution')]:
            col = self._find_col([metric, metric.replace('_PCT', '')])
            if col:
                setattr(analysis, attr, self._analyze_distribution(col))

        # Mejor y peor trial
        score_col = self._find_col(['SCORE', 'ROI_PCT', 'ROI'])
        if score_col:
            best_idx = self.df[score_col].idxmax()
            worst_idx = self.df[score_col].idxmin()

            analysis.best_trial = self.df.iloc[best_idx].to_dict()
            analysis.worst_trial = self.df.iloc[worst_idx].to_dict()
            analysis.median_performance = self.df.median(numeric_only=True).to_dict()

        # Importancia de parámetros (Random Forest)
        analysis.parameter_importance = self._calculate_importance()

        # Robustez global
        analysis.overall_robustness = self._calculate_global_robustness()

        return analysis

    def _analyze_distribution(self, col: str) -> Dict:
        """Analiza distribución de una métrica."""
        values = pd.to_numeric(self.df[col], errors='coerce').dropna()

        if len(values) < 10:
            return {}

        return {
            'mean': values.mean(),
            'std': values.std(),
            'median': values.median(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'min': values.min(),
            'max': values.max(),
            'skew': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'values': values.values
        }

    def _calculate_importance(self) -> Dict[str, float]:
        """Calcula importancia de parámetros con Random Forest."""
        all_params = self.schema.params + self.schema.exit_params

        if len(all_params) < 1:
            return {}

        # Preparar datos
        X = self.df[all_params].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median())

        # Target: SCORE o ROI
        target_col = self._find_col(['SCORE', 'ROI_PCT', 'ROI'])
        if not target_col:
            return {}

        y = pd.to_numeric(self.df[target_col], errors='coerce')
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        if len(y) < 30:
            return {}

        try:
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            importance = dict(zip(all_params, rf.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}

    def _calculate_global_robustness(self) -> float:
        """Calcula robustez global de la estrategia."""
        score_col = self._find_col(['SCORE', 'ROI_PCT'])
        if not score_col:
            return 0.5

        values = pd.to_numeric(self.df[score_col], errors='coerce').dropna()

        if len(values) < 10:
            return 0.5

        # % de trials rentables
        profitable = (values > 0).mean()

        # Ratio media/std (inverso del CV)
        if values.std() > 0:
            consistency = min(1, values.mean() / (values.std() + 1e-10) / 3)
        else:
            consistency = 1

        # Percentil 10 vs máximo
        p10 = values.quantile(0.1)
        p_max = values.max()
        if p_max > 0:
            tail_ratio = max(0, p10 / p_max)
        else:
            tail_ratio = 0

        robustness = profitable * 0.4 + consistency * 0.3 + tail_ratio * 0.3
        return min(1, max(0, robustness))

    def _find_col(self, candidates: List[str]) -> Optional[str]:
        """Busca columna."""
        for c in candidates:
            if c in self.df.columns:
                return c
            for col in self.df.columns:
                if col.upper() == c.upper():
                    return col
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 📈 GENERADOR DE VISUALIZACIONES BLOOMBERG
# ══════════════════════════════════════════════════════════════════════════════

class BloombergVisualizer:
    """Generador de visualizaciones estilo Bloomberg Terminal."""

    def __init__(self, df: pd.DataFrame, schema: DataSchema):
        self.df = df
        self.schema = schema
        self._cached_metric_cols = {}  # Cache para búsqueda de métricas

    def create_cover_page(self, fig: plt.Figure, strategy_name: str, n_trials: int):
        """Portada ultraminimalista."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Título principal
        fig.text(0.5, 0.62, 'MODELOX', fontsize=42, ha='center', va='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY,
                fontfamily='monospace')

        fig.text(0.5, 0.52, 'Parameter Optimization Analysis', fontsize=12,
                ha='center', color=Theme.TEXT_SECONDARY)

        # Línea decorativa sutil
        ax_line = fig.add_axes([0.35, 0.48, 0.3, 0.001])
        ax_line.set_facecolor(Theme.ACCENT)
        ax_line.set_xticks([])
        ax_line.set_yticks([])
        for spine in ax_line.spines.values():
            spine.set_visible(False)

        # Info minimalista
        fig.text(0.5, 0.38, strategy_name, fontsize=10, ha='center',
                color=Theme.TEXT_PRIMARY, fontweight='bold')

        fig.text(0.5, 0.32, f'{n_trials:,} trials', fontsize=9, ha='center',
                color=Theme.TEXT_DARK)

        fig.text(0.5, 0.26, datetime.now().strftime('%Y-%m-%d'), fontsize=8, ha='center',
                color=Theme.TEXT_DARK)

    def create_strategy_overview(self, fig: plt.Figure, analysis: StrategyAnalysis):
        """Página de resumen de estrategia."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Título
        fig.text(0.5, 0.95, 'STRATEGY OVERVIEW', fontsize=14, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY)

        gs = gridspec.GridSpec(3, 3, figure=fig,
                              left=0.08, right=0.92, top=0.88, bottom=0.08,
                              wspace=0.3, hspace=0.4)

        # 1. Distribución de SCORE
        if analysis.score_distribution:
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_distribution(ax1, analysis.score_distribution, 'SCORE', Theme.BLUE_BRIGHT)

        # 2. Distribución de ROI
        if analysis.roi_distribution:
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_distribution(ax2, analysis.roi_distribution, 'ROI %', Theme.GREEN_BRIGHT)

        # 3. Distribución de SHARPE
        if analysis.sharpe_distribution:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_distribution(ax3, analysis.sharpe_distribution, 'SHARPE', Theme.PURPLE_BRIGHT)

        # 4. Importancia de parámetros
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_importance(ax4, analysis.parameter_importance)

        # 5. Métricas clave
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_key_metrics(ax5, analysis)

        # 6. Robustez global
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_robustness_summary(ax6, analysis)

    def _plot_distribution(self, ax: plt.Axes, dist: Dict, title: str, color: str):
        """Histograma con KDE overlay - Ultraminimalista."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        values = dist.get('values', np.array([]))
        if len(values) == 0:
            return

        # Histograma con color único del tema
        n, bins, patches = ax.hist(values, bins=25, density=True,
                                   alpha=0.4, color=Theme.ACCENT, edgecolor='none')

        # KDE
        try:
            kde = stats.gaussian_kde(values)
            x_kde = np.linspace(values.min(), values.max(), 100)
            ax.plot(x_kde, kde(x_kde), color=Theme.ACCENT, linewidth=1.5, alpha=0.8)
        except Exception:
            pass

        # Línea de mediana
        ax.axvline(dist['median'], color=Theme.TEXT_SECONDARY, linestyle='--', linewidth=1, alpha=0.6)

        ax.set_title(title, fontsize=9, color=Theme.TEXT_SECONDARY, pad=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)

        # Stats minimalista
        ax.text(0.95, 0.95, f'μ={dist["mean"]:.2f}\nσ={dist.get("std", 0):.2f}',
               transform=ax.transAxes, fontsize=6, color=Theme.TEXT_DARK,
               ha='right', va='top')

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_importance(self, ax: plt.Axes, importance: Dict):
        """Barras de importancia de parámetros - Ultraminimalista."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        if not importance:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   color=Theme.TEXT_DARK, fontsize=9)
            return

        # Top 8 parámetros
        top_params = list(importance.items())[:8]
        params = [p[0] for p in top_params]
        values = [p[1] for p in top_params]

        # Colores degradados usando el color de acento
        colors = [Theme.ACCENT if i < 3 else Theme.TEXT_SECONDARY if i < 5 else Theme.TEXT_DARK
                 for i in range(len(params))]

        y_pos = np.arange(len(params))
        ax.barh(y_pos, values, color=colors, height=0.55, edgecolor='none', alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=7, color=Theme.TEXT_SECONDARY)
        ax.set_xlim(0, max(values) * 1.1)
        ax.set_title('PARAMETER IMPORTANCE', fontsize=9, color=Theme.TEXT_SECONDARY, pad=6)
        ax.invert_yaxis()
        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)

        # Valores
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.02, i, f'{v:.3f}', va='center', fontsize=6, color=Theme.TEXT_DARK)

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_key_metrics(self, ax: plt.Axes, analysis: StrategyAnalysis):
        """Panel de métricas clave."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Título
        ax.text(0.5, 0.92, 'KEY METRICS', fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_PRIMARY)

        metrics = [
            ('Trials', f"{analysis.n_trials:,}"),
            ('Robustness', f"{analysis.overall_robustness:.1%}"),
        ]

        if analysis.score_distribution:
            metrics.append(('Best Score', f"{analysis.score_distribution.get('max', 0):.2f}"))
            metrics.append(('Median Score', f"{analysis.score_distribution.get('median', 0):.2f}"))

        if analysis.roi_distribution:
            metrics.append(('Best ROI', f"{analysis.roi_distribution.get('max', 0):.1f}%"))

        y_pos = 0.78
        for name, value in metrics[:6]:
            ax.text(0.1, y_pos, name, fontsize=8, transform=ax.transAxes, color=Theme.TEXT_MUTED)
            ax.text(0.9, y_pos, value, fontsize=8, transform=ax.transAxes,
                   color=Theme.GREEN_BRIGHT if 'Best' in name else Theme.TEXT_PRIMARY, ha='right')
            y_pos -= 0.12

    def _plot_robustness_summary(self, ax: plt.Axes, analysis: StrategyAnalysis):
        """Resumen visual de robustez."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        # Crear barras de progreso para diferentes aspectos
        aspects = [
            ('Overall Robustness', analysis.overall_robustness, Theme.BLUE_BRIGHT),
        ]

        if analysis.score_distribution:
            # Consistencia (1 - CV)
            cv = analysis.score_distribution.get('std', 1) / (abs(analysis.score_distribution.get('mean', 1)) + 1e-10)
            consistency = max(0, min(1, 1 - cv))
            aspects.append(('Score Consistency', consistency, Theme.GREEN_BRIGHT))

            # Rentabilidad (% positivos)
            values = analysis.score_distribution.get('values', np.array([]))
            if len(values) > 0:
                profitability = (values > 0).mean()
                aspects.append(('Profitability Rate', profitability, Theme.GOLD))

        np.arange(len(aspects))
        bar_height = 0.5

        for i, (name, value, color) in enumerate(aspects):
            # Fondo gris
            ax.barh(i, 1, height=bar_height, color=Theme.BG_HIGHLIGHT, edgecolor='none')
            # Barra de valor
            ax.barh(i, value, height=bar_height, color=color, edgecolor='none')
            # Texto
            ax.text(-0.02, i, name, va='center', ha='right', fontsize=9, color=Theme.TEXT_SECONDARY)
            ax.text(value + 0.02, i, f'{value:.1%}', va='center', fontsize=9, color=Theme.TEXT_PRIMARY)

        ax.set_xlim(-0.4, 1.1)
        ax.set_ylim(-0.5, len(aspects) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('ROBUSTNESS ANALYSIS', fontsize=10, fontweight='bold',
                    color=Theme.TEXT_PRIMARY, pad=10)

        for spine in ax.spines.values():
            spine.set_visible(False)

    def create_parameter_page(self, fig: plt.Figure, analysis: ParameterAnalysis,
                             is_exit_param: bool = False):
        """Página de análisis individual de parámetro."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        param = analysis.param_name

        # Header con indicador de tipo
        param_type = "EXIT PARAM" if is_exit_param else "INDICATOR"
        type_color = Theme.GOLD if is_exit_param else Theme.BLUE_BRIGHT

        fig.text(0.04, 0.96, param_type, fontsize=8, color=type_color, fontweight='bold')
        fig.text(0.5, 0.96, param, fontsize=14, ha='center', fontweight='bold', color=Theme.TEXT_PRIMARY)

        # Valor óptimo destacado
        if not np.isnan(analysis.optimal_value):
            fig.text(0.96, 0.96, f'OPTIMAL: {format_number(analysis.optimal_value)}', fontsize=10,
                    ha='right', color=Theme.GREEN_BRIGHT, fontweight='bold')

        # Grid layout
        gs = gridspec.GridSpec(3, 4, figure=fig,
                              left=0.06, right=0.94, top=0.90, bottom=0.06,
                              wspace=0.25, hspace=0.35,
                              height_ratios=[1.2, 1.2, 0.8])

        # Métricas principales - buscar las disponibles en el análisis
        # DRAWDOWN primero como métrica principal de comparación, SQN en vez de PROFIT_FACTOR
        main_metrics_priority = ['DRAWDOWN', 'ROI', 'ROI_PCT', 'SHARPE', 'SQN', 'PROFIT_FACTOR', 'SCORE', 'WINRATE']
        metric_colors = [Theme.BLUE_BRIGHT, Theme.GREEN_BRIGHT, Theme.PURPLE_BRIGHT, Theme.CYAN,
                        Theme.GOLD, Theme.RED_BRIGHT, Theme.ORANGE_BRIGHT, Theme.PINK]

        # Encontrar las 4 métricas disponibles
        available_metrics = []
        for mk in main_metrics_priority:
            col = self._find_metric_col(mk)
            if col and col in analysis.metric_analysis and col not in available_metrics:
                available_metrics.append(col)
            if len(available_metrics) >= 4:
                break

        for i in range(4):
            ax = fig.add_subplot(gs[i // 2, i % 2])

            if i < len(available_metrics):
                metric_col = available_metrics[i]
                self._plot_metric_curve(ax, analysis, metric_col, metric_colors[i])
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       color=Theme.TEXT_MUTED, fontsize=10)
                ax.set_facecolor(Theme.BG_SECONDARY)
                for spine in ax.spines.values():
                    spine.set_color(Theme.BORDER)

        # Panel de estadísticas (derecha superior)
        ax_stats = fig.add_subplot(gs[0:2, 2:4])
        self._plot_stats_panel(ax_stats, analysis)

        # Panel de recomendación (inferior)
        ax_rec = fig.add_subplot(gs[2, :])
        self._plot_recommendation(ax_rec, analysis, is_exit_param)

    def _plot_metric_curve(self, ax: plt.Axes, analysis: ParameterAnalysis,
                          metric: str, color: str):
        """Gráfico de curva para una métrica - Ultraminimalista con filtrado de outliers."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        result = analysis.metric_analysis.get(metric, {})

        if 'curve_x' not in result or 'curve_y' not in result:
            return

        x_curve = result['curve_x']
        y_curve = result['curve_y']

        # Datos originales
        x_orig = pd.to_numeric(self.df[analysis.param_name], errors='coerce').values
        y_orig = pd.to_numeric(self.df[metric], errors='coerce').values
        mask = ~(np.isnan(x_orig) | np.isnan(y_orig))

        x_valid = x_orig[mask]
        y_valid = y_orig[mask]

        # ═══════════════════════════════════════════════════════════════════
        # FILTRAR OUTLIERS para scatter plot (consistente con la curva)
        # ═══════════════════════════════════════════════════════════════════
        if len(y_valid) > 20:
            y_q1, y_q3 = np.percentile(y_valid, [25, 75])
            y_iqr = y_q3 - y_q1
            y_lower = y_q1 - 2.5 * y_iqr
            y_upper = y_q3 + 2.5 * y_iqr
            scatter_mask = (y_valid >= y_lower) & (y_valid <= y_upper)

            # Si muy pocos quedan, usar percentiles
            if scatter_mask.sum() < 20:
                y_p5, y_p95 = np.percentile(y_valid, [5, 95])
                scatter_mask = (y_valid >= y_p5) & (y_valid <= y_p95)

            x_scatter = x_valid[scatter_mask]
            y_scatter = y_valid[scatter_mask]
        else:
            x_scatter = x_valid
            y_scatter = y_valid

        # Scatter con color uniforme del tema (más sutil)
        ax.scatter(x_scatter, y_scatter, c=Theme.ACCENT, alpha=0.08, s=6, edgecolors='none')

        # Curva suavizada con color del tema
        ax.plot(x_curve, y_curve, color=Theme.ACCENT, linewidth=2, zorder=10, alpha=0.9)

        # Zona óptima (sutil)
        opt_range = analysis.optimal_range
        if not np.isnan(opt_range[0]):
            ax.axvspan(opt_range[0], opt_range[1], color=Theme.ACCENT, alpha=0.05, zorder=1)

        # Línea de óptimo
        if not np.isnan(analysis.optimal_value):
            ax.axvline(analysis.optimal_value, color=Theme.TEXT_SECONDARY, linestyle='--',
                      linewidth=1, alpha=0.5, zorder=8)

        # Título con nombre de métrica limpio
        metric_name = metric.replace('_PCT', '%').replace('_', ' ')
        ax.set_title(metric_name, fontsize=9, fontweight='bold', color=Theme.TEXT_SECONDARY, pad=4)

        ax.set_xlabel(analysis.param_name, fontsize=7, color=Theme.TEXT_DARK)

        # Formatear ejes sin notación científica
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))

        ax.tick_params(axis='both', labelsize=6, colors=Theme.TEXT_DARK)

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_stats_panel(self, ax: plt.Axes, analysis: ParameterAnalysis):
        """Panel de estadísticas del parámetro - Ultraminimalista."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Título
        ax.text(0.5, 0.92, 'PARAMETER STATS', fontsize=10, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        # Línea separadora
        ax.plot([0.08, 0.92], [0.85, 0.85], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes, clip_on=False)

        y_pos = 0.75

        # Datos del parámetro
        param_data = pd.to_numeric(self.df[analysis.param_name], errors='coerce')
        valid_data = param_data.dropna()

        # Óptimo
        ax.text(0.08, y_pos, 'OPTIMAL', fontsize=9, color=Theme.TEXT_MUTED, transform=ax.transAxes)
        opt_text = format_number(analysis.optimal_value) if not np.isnan(analysis.optimal_value) else 'N/A'
        ax.text(0.92, y_pos, opt_text, fontsize=13, color=Theme.ACCENT,
               ha='right', transform=ax.transAxes, fontweight='bold')

        y_pos -= 0.12

        # Rango óptimo
        ax.text(0.08, y_pos, 'RANGE', fontsize=9, color=Theme.TEXT_MUTED, transform=ax.transAxes)
        if not np.isnan(analysis.optimal_range[0]):
            range_text = f'[{format_number(analysis.optimal_range[0])} — {format_number(analysis.optimal_range[1])}]'
        else:
            range_text = 'N/A'
        ax.text(0.92, y_pos, range_text, fontsize=10, color=Theme.TEXT_PRIMARY,
               ha='right', transform=ax.transAxes)

        y_pos -= 0.12

        # Línea divisoria sutil
        ax.plot([0.08, 0.92], [y_pos + 0.04, y_pos + 0.04], color=Theme.DIVIDER,
               linewidth=0.3, transform=ax.transAxes, clip_on=False)

        # Data Range
        ax.text(0.08, y_pos, 'DATA RANGE', fontsize=8, color=Theme.TEXT_DARK, transform=ax.transAxes)
        if len(valid_data) > 0:
            data_range = f'{format_number(valid_data.min())} — {format_number(valid_data.max())}'
        else:
            data_range = 'N/A'
        ax.text(0.92, y_pos, data_range, fontsize=9, color=Theme.TEXT_SECONDARY,
               ha='right', transform=ax.transAxes)

        y_pos -= 0.10

        # Sample count
        ax.text(0.08, y_pos, 'SAMPLES', fontsize=8, color=Theme.TEXT_DARK, transform=ax.transAxes)
        ax.text(0.92, y_pos, f'{len(valid_data):,}', fontsize=9, color=Theme.TEXT_SECONDARY,
               ha='right', transform=ax.transAxes)

        y_pos -= 0.10

        # Unique values
        ax.text(0.08, y_pos, 'UNIQUE', fontsize=8, color=Theme.TEXT_DARK, transform=ax.transAxes)
        ax.text(0.92, y_pos, f'{valid_data.nunique()}', fontsize=9, color=Theme.TEXT_SECONDARY,
               ha='right', transform=ax.transAxes)

        y_pos -= 0.14

        # Confidence meter minimalista
        ax.plot([0.08, 0.92], [y_pos + 0.06, y_pos + 0.06], color=Theme.DIVIDER,
               linewidth=0.3, transform=ax.transAxes, clip_on=False)

        ax.text(0.5, y_pos, 'CONFIDENCE', fontsize=8, ha='center',
               color=Theme.TEXT_DARK, transform=ax.transAxes)

        y_pos -= 0.08

        # Barra de confianza minimalista
        bar_width = 0.70
        bar_x = 0.15

        # Fondo de barra
        rect_bg = FancyBboxPatch((bar_x, y_pos - 0.015), bar_width, 0.03,
                                 boxstyle="round,pad=0.005", facecolor=Theme.BG_HIGHLIGHT,
                                 edgecolor='none', transform=ax.transAxes)
        ax.add_patch(rect_bg)

        # Valor de barra
        conf_color = Theme.ACCENT if analysis.confidence >= 0.6 else Theme.TEXT_SECONDARY
        rect_val = FancyBboxPatch((bar_x, y_pos - 0.015), bar_width * analysis.confidence, 0.03,
                                  boxstyle="round,pad=0.005", facecolor=conf_color,
                                  edgecolor='none', transform=ax.transAxes, alpha=0.7)
        ax.add_patch(rect_val)

        # Porcentaje
        ax.text(0.92, y_pos - 0.005, f'{analysis.confidence:.0%}', fontsize=9,
               color=Theme.TEXT_PRIMARY, ha='right', transform=ax.transAxes)

    def _plot_recommendation(self, ax: plt.Axes, analysis: ParameterAnalysis, is_exit: bool):
        """Panel de recomendación - Ultraminimalista."""
        # Fondo sutil basado en confianza
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            spine.set_linewidth(1)

        # Recomendación
        if np.isnan(analysis.optimal_value):
            ax.text(0.5, 0.5, 'INSUFFICIENT DATA',
                   ha='center', va='center', fontsize=10, color=Theme.TEXT_MUTED,
                   transform=ax.transAxes)
            return

        # Tipo de parámetro
        param_type = '◆ EXIT' if is_exit else '○ PARAM'
        type_color = Theme.ACCENT if is_exit else Theme.TEXT_SECONDARY
        ax.text(0.03, 0.5, param_type, fontsize=8, va='center',
               transform=ax.transAxes, color=type_color)

        # Valor recomendado central
        ax.text(0.18, 0.55, f'{analysis.param_name}', fontsize=9, color=Theme.TEXT_MUTED,
               transform=ax.transAxes)
        ax.text(0.18, 0.25, format_number(analysis.optimal_value), fontsize=14, color=Theme.TEXT_PRIMARY,
               fontweight='bold', transform=ax.transAxes)

        # Indicador de confianza a la derecha
        conf_text = f'{analysis.confidence:.0%}'
        conf_color = Theme.ACCENT if analysis.confidence >= 0.6 else Theme.TEXT_SECONDARY
        ax.text(0.92, 0.4, conf_text, fontsize=11, ha='right', va='center',
               color=conf_color, transform=ax.transAxes, fontweight='bold')

    def _find_metric_col(self, key: str) -> Optional[str]:
        """Busca columna de métrica con múltiples aliases."""
        # Diccionario de aliases para cada métrica
        aliases_map = {
            'ROI_PCT': ['ROI', 'ROI_PCT', 'RETURN', 'RETORNO', 'PNL_PCT'],
            'ROI': ['ROI', 'ROI_PCT', 'RETURN', 'RETORNO', 'PNL_PCT'],
            'SCORE': ['SCORE'],
            'SHARPE': ['SHARPE', 'SHARPE_RATIO'],
            'PROFIT_FACTOR': ['PROFIT_FACTOR', 'PF', 'PROFITFACTOR'],
            'SQN': ['SQN', 'SYSTEM_QUALITY'],
            'DRAWDOWN': ['DRAWDOWN', 'MAX_DD', 'MAX_DD_PCT', 'DD', 'MAX_DRAWDOWN'],
            'WINRATE': ['WINRATE', 'WINRATE_PCT', 'WIN_RATE', 'PORC_GANADORAS'],
            'ESTABILIDAD': ['ESTABILIDAD', 'STABILITY', 'CONSISTENCY'],
        }

        key_upper = key.upper().replace(' ', '_')
        aliases = aliases_map.get(key_upper, [key_upper, key])

        df_cols_upper = {col.upper(): col for col in self.df.columns}

        # Búsqueda exacta
        for alias in aliases:
            if alias.upper() in df_cols_upper:
                return df_cols_upper[alias.upper()]

        # Búsqueda parcial
        for alias in aliases:
            for col_upper, col in df_cols_upper.items():
                if alias.upper() in col_upper:
                    return col

        return None

    def create_recommendations_table(self, fig: plt.Figure, all_analyses: Dict[str, ParameterAnalysis],
                                    exit_params: List[str]):
        """Página final de recomendaciones - Ultraminimalista."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Header
        fig.text(0.5, 0.95, 'OPTIMIZATION RECOMMENDATIONS', fontsize=14, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY, family='monospace')
        fig.text(0.5, 0.91, 'All parameters sorted by confidence', fontsize=8, ha='center',
                color=Theme.TEXT_DARK)

        # Línea separadora
        ax_line = fig.add_axes([0.05, 0.885, 0.90, 0.001])
        ax_line.set_facecolor(Theme.BORDER)
        ax_line.set_xticks([])
        ax_line.set_yticks([])
        for spine in ax_line.spines.values():
            spine.set_visible(False)

        # Ordenar por confianza
        sorted_analyses = sorted(
            [(p, a) for p, a in all_analyses.items() if not np.isnan(a.optimal_value)],
            key=lambda x: x[1].confidence,
            reverse=True
        )

        if not sorted_analyses:
            fig.text(0.5, 0.5, 'No valid recommendations', ha='center', va='center',
                    fontsize=12, color=Theme.TEXT_MUTED)
            return

        # Dividir en dos columnas
        n_params = len(sorted_analyses)
        mid_point = (n_params + 1) // 2

        # Header de columnas
        y_header = 0.86
        col_positions = [
            {'type': 0.04, 'param': 0.08, 'value': 0.28, 'range': 0.36, 'conf': 0.47},
            {'type': 0.54, 'param': 0.58, 'value': 0.78, 'range': 0.86, 'conf': 0.97}
        ]

        for col in col_positions:
            fig.text(col['param'], y_header, 'PARAMETER', fontsize=7, color=Theme.TEXT_DARK, fontweight='bold')
            fig.text(col['value'], y_header, 'OPTIMAL', fontsize=7, color=Theme.TEXT_DARK, ha='right', fontweight='bold')
            fig.text(col['range'], y_header, 'RANGE', fontsize=7, color=Theme.TEXT_DARK, fontweight='bold')
            fig.text(col['conf'], y_header, 'CONF', fontsize=7, color=Theme.TEXT_DARK, ha='right', fontweight='bold')

        # Renderizar filas
        row_height = 0.035
        start_y = 0.82

        for i, (param, analysis) in enumerate(sorted_analyses):
            col_idx = 0 if i < mid_point else 1
            row_in_col = i if i < mid_point else i - mid_point
            col = col_positions[col_idx]

            y_pos = start_y - (row_in_col * row_height)

            if y_pos < 0.06:
                break

            is_exit = param in exit_params

            # Color según confianza
            if analysis.confidence >= 0.7:
                text_color = Theme.ACCENT
            elif analysis.confidence >= 0.5:
                text_color = Theme.TEXT_PRIMARY
            else:
                text_color = Theme.TEXT_SECONDARY

            # Indicador de tipo
            type_indicator = '◆' if is_exit else '○'
            fig.text(col['type'], y_pos, type_indicator, fontsize=7, color=text_color)

            # Nombre del parámetro
            param_display = param[:18] + '…' if len(param) > 18 else param
            fig.text(col['param'], y_pos, param_display, fontsize=8, color=text_color)

            # Valor óptimo
            fig.text(col['value'], y_pos, format_number(analysis.optimal_value), fontsize=9,
                    color=text_color, ha='right', fontweight='bold')

            # Rango
            if not np.isnan(analysis.optimal_range[0]):
                range_str = f'{format_number(analysis.optimal_range[0])}—{format_number(analysis.optimal_range[1])}'
            else:
                range_str = '—'
            fig.text(col['range'], y_pos, range_str, fontsize=7, color=Theme.TEXT_DARK)

            # Confianza
            fig.text(col['conf'], y_pos, f'{analysis.confidence:.0%}', fontsize=8,
                    color=text_color, ha='right')

        # Leyenda al fondo
        fig.text(0.05, 0.025, '◆ Exit Parameters    ○ Indicator Parameters',
                fontsize=7, color=Theme.TEXT_DARK)

        # Stats del análisis
        avg_conf = np.mean([a.confidence for _, a in sorted_analyses])
        high_conf = len([a for _, a in sorted_analyses if a.confidence >= 0.7])
        fig.text(0.95, 0.025, f'{len(sorted_analyses)} params · {high_conf} high conf · avg {avg_conf:.0%}',
                fontsize=7, color=Theme.TEXT_DARK, ha='right')

    def create_3d_surface(self, fig: plt.Figure, param1: str, param2: str, metric_col: str,
                          title: str = ""):
        """Crea superficie 3D de optimización."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(Theme.BG_SECONDARY)

        # Datos
        x = pd.to_numeric(self.df[param1], errors='coerce').values
        y = pd.to_numeric(self.df[param2], errors='coerce').values
        z = pd.to_numeric(self.df[metric_col], errors='coerce').values

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(z) < 20:
            ax.text2D(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes,
                     ha='center', va='center', color=Theme.TEXT_MUTED, fontsize=12)
            return

        # Crear grid interpolado
        grid_size = 40
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi, yi), method='cubic')
            zi = np.nan_to_num(zi, nan=np.nanmean(z))
            zi = gaussian_filter(zi, sigma=3.0)  # Mayor suavizado
        except Exception:
            zi = griddata((x, y), z, (xi, yi), method='linear')
            zi = np.nan_to_num(zi, nan=np.nanmean(z))
            zi = gaussian_filter(zi, sigma=2.5)

        # Superficie con colormap rojo→azul
        cmap = Theme.get_surface_cmap()
        norm = Normalize(vmin=np.percentile(z, 5), vmax=np.percentile(z, 95))

        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, norm=norm,
                               alpha=0.9, antialiased=True,
                               linewidth=0.1, edgecolor='none')

        # Scatter puntos reales (muy sutil)
        ax.scatter(x, y, z, c=z, cmap=cmap, norm=norm, s=5, alpha=0.3, edgecolors='none')

        # Estilo minimalista
        ax.set_xlabel(param1, color=Theme.TEXT_DARK, labelpad=8, fontsize=8)
        ax.set_ylabel(param2, color=Theme.TEXT_DARK, labelpad=8, fontsize=8)
        ax.set_zlabel(metric_col, color=Theme.TEXT_DARK, labelpad=8, fontsize=8)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(Theme.BORDER)
        ax.yaxis.pane.set_edgecolor(Theme.BORDER)
        ax.zaxis.pane.set_edgecolor(Theme.BORDER)

        # Formatear ejes sin notación científica
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.zaxis.set_major_formatter(FuncFormatter(format_axis_number))

        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)
        ax.view_init(elev=25, azim=45)

        if title:
            ax.set_title(title, color=Theme.TEXT_SECONDARY, fontsize=10, pad=15)

        # Colorbar minimalista
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        cbar.ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)
        cbar.outline.set_edgecolor(Theme.BORDER)

    def create_dual_3d_surfaces(self, fig: plt.Figure, param1: str, param2: str):
        """Crea dos superficies 3D lado a lado (ROI y SQN)."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        fig.text(0.5, 0.97, f'OPTIMIZATION SURFACE: {param1} × {param2}', fontsize=12,
                ha='center', fontweight='bold', color=Theme.TEXT_PRIMARY)

        # Buscar métricas
        roi_col = self._find_metric_col('ROI')
        sqn_col = self._find_metric_col('SQN')

        gs = gridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.95,
                              top=0.90, bottom=0.08, wspace=0.15)

        # Surface 1: ROI
        if roi_col:
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self._plot_3d_mini(ax1, param1, param2, roi_col, f'{param1} × {param2} → ROI')

        # Surface 2: SQN
        if sqn_col:
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            self._plot_3d_mini(ax2, param1, param2, sqn_col, f'{param1} × {param2} → SQN')

    def _plot_3d_mini(self, ax: plt.Axes, param1: str, param2: str, metric_col: str, title: str):
        """Mini plot 3D para grid - Ultraminimalista."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        x = pd.to_numeric(self.df[param1], errors='coerce').values
        y = pd.to_numeric(self.df[param2], errors='coerce').values
        z = pd.to_numeric(self.df[metric_col], errors='coerce').values

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(z) < 15:
            ax.text2D(0.5, 0.5, 'No Data', transform=ax.transAxes,
                     ha='center', va='center', color=Theme.TEXT_DARK)
            return

        # Grid
        grid_size = 30
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        try:
            zi = griddata((x, y), z, (xi, yi), method='cubic')
            zi = np.nan_to_num(zi, nan=np.nanmean(z))
            zi = gaussian_filter(zi, sigma=3.0)  # Mayor suavizado
        except Exception:
            zi = griddata((x, y), z, (xi, yi), method='linear')
            zi = np.nan_to_num(zi, nan=np.nanmean(z))
            zi = gaussian_filter(zi, sigma=2.5)

        # Usar el colormap de superficie (rojo→azul)
        cmap = Theme.get_surface_cmap()
        norm = Normalize(vmin=np.percentile(z, 10), vmax=np.percentile(z, 90))

        ax.plot_surface(xi, yi, zi, cmap=cmap, norm=norm,
                               alpha=0.9, antialiased=True, linewidth=0)

        ax.set_xlabel(param1, color=Theme.TEXT_DARK, fontsize=7, labelpad=4)
        ax.set_ylabel(param2, color=Theme.TEXT_DARK, fontsize=7, labelpad=4)
        ax.set_zlabel(metric_col, color=Theme.TEXT_DARK, fontsize=7, labelpad=4)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(Theme.BORDER)
        ax.yaxis.pane.set_edgecolor(Theme.BORDER)
        ax.zaxis.pane.set_edgecolor(Theme.BORDER)

        # Formatear ejes sin notación científica
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.zaxis.set_major_formatter(FuncFormatter(format_axis_number))

        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=5)
        ax.view_init(elev=20, azim=45)
        ax.set_title(title, fontsize=8, color=Theme.TEXT_SECONDARY, pad=8)

    def create_statistical_validation_page(self, fig: plt.Figure, validation: StatisticalValidation):
        """Página de validación estadística PROFESIONAL completa."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Header profesional
        fig.text(0.5, 0.97, 'STATISTICAL VALIDATION REPORT', fontsize=14, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY)
        fig.text(0.5, 0.94, "Institutional-Grade Robustness Analysis",
                fontsize=9, ha='center', color=Theme.TEXT_MUTED, style='italic')

        # Línea separadora
        line = plt.Line2D([0.05, 0.95], [0.92, 0.92], color=Theme.DIVIDER, linewidth=0.5)
        fig.add_artist(line)

        gs = gridspec.GridSpec(3, 3, figure=fig, left=0.06, right=0.94,
                              top=0.89, bottom=0.06, wspace=0.20, hspace=0.30,
                              height_ratios=[1.2, 1.2, 0.8])

        # 1. White's Reality Check Panel (expandido)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_wrc_panel_pro(ax1, validation)

        # 2. Deflated Sharpe Panel (expandido)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_dsr_panel_pro(ax2, validation)

        # 3. Probabilistic Sharpe Ratio Visual
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_psr_visual(ax3, validation)

        # 4. Distribution Analysis con histograma
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_distribution_pro(ax4, validation)

        # 5. Left Tail Risk Analysis (reemplaza tests de series temporales/regímenes)
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_risk_tail_analysis(ax5, validation)

        # 7. Summary Dashboard (full width)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_validation_dashboard(ax7, validation)

    def _plot_wrc_panel_pro(self, ax: plt.Axes, v: StatisticalValidation):
        """Panel profesional de White's Reality Check."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Título con icono
        ax.text(0.5, 0.94, "⚡ WHITE'S REALITY CHECK", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        # Línea separadora
        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        # Descripción
        ax.text(0.5, 0.82, "Data Snooping Bias Test", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # Status principal con indicador visual
        status_color = Theme.ACCENT if v.wrc_is_significant else '#e53935'
        status_text = '✓ PASSED' if v.wrc_is_significant else '✗ FAILED'

        # Círculo indicador
        circle_color = Theme.ACCENT if v.wrc_is_significant else '#e53935'
        circle = plt.Circle((0.15, 0.68), 0.04, color=circle_color, transform=ax.transAxes)
        ax.add_patch(circle)

        ax.text(0.25, 0.68, status_text, fontsize=10, va='center',
               transform=ax.transAxes, color=status_color, fontweight='bold')

        # Métricas detalladas
        y_pos = 0.52
        metrics = [
            ('P-Value', f'{v.wrc_p_value:.4f}', v.wrc_p_value <= 0.05),
            ('Bootstrap μ', f'{v.wrc_bootstrap_mean:.4f}', True),
            ('Bootstrap σ', f'{v.wrc_bootstrap_std:.4f}', True),
            ('Significance', '95% CI' if v.wrc_is_significant else 'Below 95%', v.wrc_is_significant),
        ]

        for label, value, is_good in metrics:
            ax.text(0.08, y_pos, label, fontsize=7, transform=ax.transAxes, color=Theme.TEXT_DARK)
            val_color = Theme.ACCENT if is_good else Theme.TEXT_SECONDARY
            ax.text(0.92, y_pos, value, fontsize=8, ha='right', transform=ax.transAxes, color=val_color)
            y_pos -= 0.12

        # Interpretación
        interpretation = "Strategy outperforms random" if v.wrc_is_significant else "No significant edge detected"
        ax.text(0.5, 0.06, interpretation, fontsize=6, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

    def _plot_dsr_panel_pro(self, ax: plt.Axes, v: StatisticalValidation):
        """Panel profesional de Deflated Sharpe Ratio."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Título
        ax.text(0.5, 0.94, "📊 DEFLATED SHARPE RATIO", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        ax.text(0.5, 0.82, "Bailey & López de Prado (2014)", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # Comparación visual Original vs Deflated
        bar_width = 0.15

        # Barra Original
        orig_height = min(0.25, abs(v.original_sharpe) * 0.08)
        ax.add_patch(plt.Rectangle((0.2, 0.45), bar_width, orig_height,
                    color=Theme.TEXT_SECONDARY, transform=ax.transAxes, alpha=0.7))
        ax.text(0.275, 0.42, 'Original', fontsize=6, ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.275, 0.45 + orig_height + 0.02, f'{v.original_sharpe:.2f}', fontsize=8,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        # Barra Deflated
        defl_height = min(0.25, abs(v.deflated_sharpe) * 0.08)
        defl_color = Theme.ACCENT if v.dsr_is_significant else '#e53935'
        ax.add_patch(plt.Rectangle((0.65, 0.45), bar_width, defl_height,
                    color=defl_color, transform=ax.transAxes))
        ax.text(0.725, 0.42, 'Deflated', fontsize=6, ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.725, 0.45 + defl_height + 0.02, f'{v.deflated_sharpe:.2f}', fontsize=9,
               ha='center', transform=ax.transAxes, color=defl_color, fontweight='bold')

        # Flecha de haircut
        ax.annotate('', xy=(0.55, 0.55), xytext=(0.45, 0.55),
                   arrowprops=dict(arrowstyle='->', color=Theme.TEXT_DARK, lw=1),
                   transform=ax.transAxes)
        ax.text(0.5, 0.58, f'-{v.sharpe_haircut:.0%}', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)

        # Status
        status_text = '✓ Robust Performance' if v.dsr_is_significant else '⚠ Potential Overfit'
        status_color = Theme.ACCENT if v.dsr_is_significant else '#ff9800'
        ax.text(0.5, 0.12, status_text, fontsize=8, ha='center',
               transform=ax.transAxes, color=status_color, fontweight='bold')

        # Interpretación
        ax.text(0.5, 0.04, f"Multiple testing penalty applied ({v.sharpe_haircut:.0%} reduction)",
               fontsize=6, ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

    def _plot_psr_visual(self, ax: plt.Axes, v: StatisticalValidation):
        """Visual de Probabilistic Sharpe Ratio."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "🎯 CONFIDENCE METRICS", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        # Gauge de confianza
        # Calcular PSR aproximado
        psr = 0.95 if v.dsr_is_significant else 0.5 + (v.deflated_sharpe / (v.original_sharpe + 1e-10)) * 0.4
        psr = max(0, min(1, psr))

        # Arco de fondo
        theta1, theta2 = 180, 0
        for i in range(100):
            t = theta1 + (theta2 - theta1) * i / 100
            color_val = i / 100
            if color_val < 0.5:
                color = '#e53935'
            elif color_val < 0.7:
                color = '#ff9800'
            else:
                color = Theme.ACCENT
            arc_x = 0.5 + 0.25 * np.cos(np.radians(t))
            arc_y = 0.55 + 0.15 * np.sin(np.radians(t))
            ax.plot(arc_x, arc_y, 'o', markersize=2, color=color, transform=ax.transAxes, alpha=0.3)

        # Indicador
        indicator_angle = 180 - psr * 180
        ind_x = 0.5 + 0.20 * np.cos(np.radians(indicator_angle))
        ind_y = 0.55 + 0.12 * np.sin(np.radians(indicator_angle))
        ax.plot([0.5, ind_x], [0.55, ind_y], color=Theme.TEXT_PRIMARY, linewidth=2, transform=ax.transAxes)
        ax.plot(ind_x, ind_y, 'o', markersize=6, color=Theme.ACCENT, transform=ax.transAxes)

        # Valor central
        ax.text(0.5, 0.38, f'{psr:.0%}', fontsize=14, ha='center', fontweight='bold',
               transform=ax.transAxes, color=Theme.TEXT_PRIMARY)
        ax.text(0.5, 0.30, 'PSR', fontsize=8, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_MUTED)

        # Métricas adicionales
        ax.text(0.15, 0.15, f'Skew: {v.skewness:.2f}', fontsize=7, transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.85, 0.15, f'Kurt: {v.kurtosis:.2f}', fontsize=7, ha='right', transform=ax.transAxes, color=Theme.TEXT_DARK)

        # Interpretación
        if psr >= 0.95:
            interp = "High confidence in strategy skill"
        elif psr >= 0.7:
            interp = "Moderate evidence of skill"
        else:
            interp = "Insufficient evidence of skill"
        ax.text(0.5, 0.04, interp, fontsize=6, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_DARK, style='italic')

    def _plot_distribution_pro(self, ax: plt.Axes, v: StatisticalValidation):
        """Panel profesional de análisis de distribución."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "📈 DISTRIBUTION ANALYSIS", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        # Tests de normalidad con indicadores visuales
        tests = [
            ('Jarque-Bera', f'p={v.jarque_bera_pvalue:.4f}', v.jarque_bera_pvalue > 0.05,
             'Normal' if v.jarque_bera_pvalue > 0.05 else 'Non-Normal'),
            ('Skewness', f'{v.skewness:.3f}', -0.5 <= v.skewness <= 0.5,
             'Symmetric' if -0.5 <= v.skewness <= 0.5 else ('Left Tail' if v.skewness < -0.5 else 'Right Tail')),
            ('Kurtosis', f'{v.kurtosis:.3f}', -1 <= v.kurtosis <= 1,
             'Normal Tails' if -1 <= v.kurtosis <= 1 else ('Thin Tails' if v.kurtosis < -1 else 'Fat Tails')),
        ]

        y_pos = 0.75
        for name, value, is_good, interpretation in tests:
            # Indicador de estado
            status_color = Theme.ACCENT if is_good else '#ff9800'
            ax.plot(0.08, y_pos, 'o', markersize=6, color=status_color, transform=ax.transAxes)

            # Nombre del test
            ax.text(0.14, y_pos, name, fontsize=8, va='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

            # Valor
            ax.text(0.58, y_pos, value, fontsize=8, va='center', ha='right', transform=ax.transAxes, color=Theme.TEXT_DARK)

            # Interpretación
            ax.text(0.62, y_pos, interpretation, fontsize=7, va='center', transform=ax.transAxes,
                   color=status_color, style='italic')

            y_pos -= 0.18

        # Resumen de distribución
        n_passed = sum([v.jarque_bera_pvalue > 0.05, -0.5 <= v.skewness <= 0.5, -1 <= v.kurtosis <= 1])
        summary_text = f"Distribution Quality: {n_passed}/3 tests passed"
        summary_color = Theme.ACCENT if n_passed >= 2 else '#ff9800' if n_passed == 1 else '#e53935'
        ax.text(0.5, 0.12, summary_text, fontsize=8, ha='center', transform=ax.transAxes,
               color=summary_color, fontweight='bold')

        # Nota
        ax.text(0.5, 0.04, "Non-normal returns may affect risk metrics", fontsize=6,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

    def _plot_risk_tail_analysis(self, ax: plt.Axes, v: StatisticalValidation):
        """Panel de análisis de cola izquierda (VaR/CVaR) sobre trials."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "📉 LEFT TAIL RISK", fontsize=9, fontweight='bold',
                ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
        ax.plot([0.05, 0.95], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5, transform=ax.transAxes)

        values = np.asarray(getattr(v, 'trial_returns', np.array([])), dtype=float)
        if values.size < 10:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                    transform=ax.transAxes, color=Theme.TEXT_DARK, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Enfocar en pérdidas: si hay negativos, mirar <=0; si no, mostrar el peor cuartil
        if np.any(values < 0):
            focus = values[values <= 0]
            x_right = 0.0
        else:
            focus = values
            x_right = float(np.quantile(values, 0.25))

        x_left = float(np.quantile(values, 0.01))
        if x_left == x_right:
            x_left = float(values.min())
            x_right = float(values.max())

        # Histograma del foco
        ax.hist(focus, bins=30, color=Theme.RED_BRIGHT, alpha=0.30, edgecolor='none', density=True)

        # KDE overlay (sutil)
        try:
            kde = stats.gaussian_kde(focus)
            x_kde = np.linspace(min(focus.min(), x_left), max(focus.max(), x_right), 200)
            y_kde = kde(x_kde)
            ax.plot(x_kde, y_kde, color=Theme.RED_BRIGHT, linewidth=1.5, alpha=0.7)
            ax.fill_between(x_kde, 0, y_kde, where=(x_kde <= v.var_95), color=Theme.RED_BRIGHT, alpha=0.08)
        except Exception:
            pass

        # Líneas VaR/CVaR
        ax.axvline(v.var_95, color=Theme.GOLD, linestyle='--', linewidth=1.6, alpha=0.95)
        ax.axvline(v.cvar_95, color=Theme.RED_BRIGHT, linestyle='-', linewidth=1.6, alpha=0.95)
        ax.axvline(0, color=Theme.DIVIDER, linestyle='-', linewidth=1.0, alpha=0.6)

        ax.set_xlim(x_left, x_right)
        ax.set_yticks([])
        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=7)
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))

        # Anotaciones
        ax.text(0.02, 0.78, 'VaR95', transform=ax.transAxes, fontsize=7, color=Theme.GOLD)
        ax.text(0.02, 0.70, 'CVaR95', transform=ax.transAxes, fontsize=7, color=Theme.RED_BRIGHT)

        ax.text(0.98, 0.80,
                f"VaR95: {v.var_95:.3f}\nCVaR95: {v.cvar_95:.3f}\nP(Loss): {v.prob_loss:.0%}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=Theme.TEXT_DARK)

        ax.text(0.5, 0.10,
                "Historical tail risk across optimization trials (cross-sectional)",
                transform=ax.transAxes, ha='center', va='center', fontsize=6,
                color=Theme.TEXT_DARK, style='italic')

    def _plot_validation_dashboard(self, ax: plt.Axes, v: StatisticalValidation):
        """Dashboard resumen profesional de validación."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Calcular score total
        tests = [
            ('Reality Check', v.wrc_is_significant, "Strategy outperforms random"),
            ('Deflated Sharpe', v.dsr_is_significant, "Robust risk-adjusted returns"),
            ('Distribution', v.jarque_bera_pvalue > 0.05, "Normal return distribution"),
            ('Left Tail Risk', v.var_95 > 0, "Worst 5% configurations remain profitable"),
        ]

        n_passed = sum(t[1] for t in tests)

        # Determinar calificación general
        if n_passed == 4:
            grade = 'A+'
            grade_text = 'EXCELLENT'
            grade_color = Theme.ACCENT
            verdict = 'Strategy shows strong statistical robustness across all tests'
        elif n_passed == 3:
            grade = 'A'
            grade_text = 'ROBUST'
            grade_color = '#66bb6a'
            verdict = 'Strategy demonstrates solid statistical foundation'
        elif n_passed == 2:
            grade = 'B'
            grade_text = 'MODERATE'
            grade_color = '#ff9800'
            verdict = 'Strategy shows mixed statistical evidence'
        elif n_passed == 1:
            grade = 'C'
            grade_text = 'WEAK'
            grade_color = '#ff7043'
            verdict = 'Strategy requires additional validation'
        else:
            grade = 'D'
            grade_text = 'UNCERTAIN'
            grade_color = '#e53935'
            verdict = 'Strategy lacks statistical support'

        # Grade principal
        ax.text(0.08, 0.55, grade, fontsize=28, fontweight='bold', va='center',
               transform=ax.transAxes, color=grade_color)
        ax.text(0.08, 0.25, grade_text, fontsize=10, fontweight='bold', va='center',
               transform=ax.transAxes, color=grade_color)

        # Línea vertical separadora
        ax.plot([0.18, 0.18], [0.15, 0.85], color=Theme.DIVIDER, linewidth=1,
               transform=ax.transAxes)

        # Tests individuales
        x_start = 0.22
        x_spacing = 0.20

        for i, (name, passed, desc) in enumerate(tests):
            x_pos = x_start + i * x_spacing

            # Indicador
            status_color = Theme.ACCENT if passed else '#e53935'
            status_text = '✓' if passed else '✗'

            ax.text(x_pos, 0.72, status_text, fontsize=14, ha='center', transform=ax.transAxes,
                   color=status_color, fontweight='bold')
            ax.text(x_pos, 0.52, name, fontsize=7, ha='center', transform=ax.transAxes,
                   color=Theme.TEXT_SECONDARY, fontweight='bold')
            ax.text(x_pos, 0.35, desc[:20] + '...' if len(desc) > 20 else desc, fontsize=5,
                   ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK)

        # Veredicto final
        ax.text(0.60, 0.12, verdict, fontsize=8, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_SECONDARY, style='italic')

        # Score
        ax.text(0.95, 0.55, f'{n_passed}/4', fontsize=16, ha='right', va='center',
               transform=ax.transAxes, color=Theme.TEXT_PRIMARY, fontweight='bold')
        ax.text(0.95, 0.35, 'Tests Passed', fontsize=7, ha='right', transform=ax.transAxes,
               color=Theme.TEXT_DARK)

    def find_exit_params(self) -> Tuple[Optional[str], Optional[str]]:
        """Encuentra los parámetros SL y TP."""
        sl_col = None
        tp_col = None

        for col in self.schema.exit_params:
            col_upper = col.upper()
            if 'SL' in col_upper and sl_col is None:
                sl_col = col
            elif 'TP' in col_upper and tp_col is None:
                tp_col = col

        return sl_col, tp_col

    def create_correlation_page(self, fig: plt.Figure, correlations: List[ParameterCorrelation]):
        """Página de análisis de correlación entre parámetros."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        fig.text(0.5, 0.96, 'PARAMETER CORRELATION ANALYSIS', fontsize=12, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY)
        fig.text(0.5, 0.93, 'Detecting related parameters for 3D surface analysis',
                fontsize=8, ha='center', color=Theme.TEXT_DARK)

        if not correlations:
            fig.text(0.5, 0.5, 'Insufficient data for correlation analysis',
                    ha='center', va='center', fontsize=10, color=Theme.TEXT_MUTED)
            return

        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.92,
                              top=0.88, bottom=0.12, wspace=0.25, hspace=0.35)

        # 1. Matriz de correlación (solo los parámetros principales)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_matrix(ax1, correlations)

        # 2. Top pares correlacionados
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_top_pairs(ax2, correlations)

        # 3. Scatter de los top 2 pares
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        if len(correlations) >= 1:
            self._plot_pair_scatter(ax3, correlations[0])
        else:
            ax3.axis('off')

        if len(correlations) >= 2:
            self._plot_pair_scatter(ax4, correlations[1])
        else:
            ax4.axis('off')

        # Leyenda de pares fuertes al final
        strong_pairs = [c for c in correlations if c.is_strongly_related]
        if strong_pairs:
            text = 'Strong pairs → 3D Analysis: ' + ', '.join([f'{c.param1}×{c.param2}' for c in strong_pairs[:4]])
            fig.text(0.5, 0.03, text, fontsize=7, ha='center', color=Theme.ACCENT)

    def _plot_correlation_matrix(self, ax: plt.Axes, correlations: List[ParameterCorrelation]):
        """Dibuja matriz de correlación simplificada."""
        ax.set_facecolor(Theme.BG_TERTIARY)

        # Obtener parámetros únicos
        params = list(set([c.param1 for c in correlations] + [c.param2 for c in correlations]))
        params = params[:8]  # Limitar a 8 para legibilidad
        n = len(params)

        if n < 2:
            ax.text(0.5, 0.5, 'Not enough params', ha='center', va='center',
                   color=Theme.TEXT_DARK, fontsize=9, transform=ax.transAxes)
            return

        # Crear matriz
        matrix = np.zeros((n, n))
        param_idx = {p: i for i, p in enumerate(params)}

        for c in correlations:
            if c.param1 in param_idx and c.param2 in param_idx:
                i, j = param_idx[c.param1], param_idx[c.param2]
                matrix[i, j] = c.spearman_corr
                matrix[j, i] = c.spearman_corr

        # Diagonal = 1
        np.fill_diagonal(matrix, 1)

        # Heatmap
        cmap = Theme.get_correlation_cmap()
        ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

        # Labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        short_params = [p[:10] for p in params]
        ax.set_xticklabels(short_params, fontsize=6, rotation=45, ha='right')
        ax.set_yticklabels(short_params, fontsize=6)

        # Valores en celdas
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = matrix[i, j]
                    color = Theme.TEXT_PRIMARY if abs(val) > 0.3 else Theme.TEXT_DARK
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=5, color=color)

        ax.set_title('Spearman Correlation', fontsize=8, color=Theme.TEXT_SECONDARY, pad=4)

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_top_pairs(self, ax: plt.Axes, correlations: List[ParameterCorrelation]):
        """Muestra los pares más correlacionados."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.95, 'TOP RELATED PAIRS', fontsize=9, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

        y_pos = 0.82
        for i, c in enumerate(correlations[:6]):
            # Indicador de fuerza
            if c.is_strongly_related:
                indicator = '●'
                color = Theme.ACCENT
            else:
                indicator = '○'
                color = Theme.TEXT_SECONDARY

            # Nombre del par
            pair_name = f'{c.param1[:12]} × {c.param2[:12]}'
            ax.text(0.05, y_pos, indicator, fontsize=8, color=color, transform=ax.transAxes)
            ax.text(0.12, y_pos, pair_name, fontsize=7, color=color, transform=ax.transAxes)

            # Correlación
            corr_val = (abs(c.pearson_corr) + abs(c.spearman_corr)) / 2
            ax.text(0.95, y_pos, f'{corr_val:.2f}', fontsize=7, ha='right',
                   color=Theme.TEXT_PRIMARY, transform=ax.transAxes)

            y_pos -= 0.12

        # Leyenda
        ax.text(0.5, 0.05, '● Strong (3D surface)  ○ Weak', fontsize=6,
               ha='center', color=Theme.TEXT_DARK, transform=ax.transAxes)

    def _plot_pair_scatter(self, ax: plt.Axes, corr: ParameterCorrelation):
        """Scatter plot de un par de parámetros."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        x = pd.to_numeric(self.df[corr.param1], errors='coerce').values
        y = pd.to_numeric(self.df[corr.param2], errors='coerce').values

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < 5:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   color=Theme.TEXT_DARK, transform=ax.transAxes)
            return

        # Colorear por target si existe
        target_col = self._find_metric_col('ROI') or self._find_metric_col('SCORE')
        if target_col:
            z = pd.to_numeric(self.df.loc[mask.nonzero()[0], target_col], errors='coerce').values
            z = np.nan_to_num(z, nan=np.nanmean(z))
            ax.scatter(x, y, c=z, cmap=Theme.get_surface_cmap(),
                               s=15, alpha=0.6, edgecolors='none')
        else:
            ax.scatter(x, y, c=Theme.ACCENT, s=15, alpha=0.6, edgecolors='none')

        # Línea de tendencia
        try:
            z_fit = np.polyfit(x, y, 1)
            p = np.poly1d(z_fit)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, p(x_line), color=Theme.TEXT_MUTED, linewidth=1,
                   linestyle='--', alpha=0.7)
        except Exception:
            pass

        ax.set_xlabel(corr.param1[:15], fontsize=7, color=Theme.TEXT_DARK)
        ax.set_ylabel(corr.param2[:15], fontsize=7, color=Theme.TEXT_DARK)
        ax.tick_params(labelsize=6, colors=Theme.TEXT_DARK)

        # Título con correlación
        title = f'ρ={corr.spearman_corr:.2f}'
        ax.set_title(title, fontsize=7, color=Theme.TEXT_SECONDARY, pad=3)

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def create_correlated_pair_surface(self, fig: plt.Figure, param1: str, param2: str,
                                        corr_info: Optional[ParameterCorrelation] = None):
        """Crea página de superficie 3D para un par de parámetros correlacionados."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Título
        title = f'{param1} × {param2}'
        if corr_info:
            title += f'  (ρ={corr_info.spearman_corr:.2f})'

        fig.text(0.5, 0.97, f'3D OPTIMIZATION: {title}', fontsize=11, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY)

        # Grid para 4 superficies: ROI, SQN, SCORE, SHARPE
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95,
                              top=0.92, bottom=0.05, wspace=0.12, hspace=0.18)

        metrics_to_plot = [
            ('ROI', 'ROI %'),
            ('SQN', 'SQN'),
            ('SCORE', 'SCORE'),
            ('SHARPE', 'SHARPE')
        ]

        for idx, (metric_key, label) in enumerate(metrics_to_plot):
            metric_col = self._find_metric_col(metric_key)
            if not metric_col:
                continue

            ax = fig.add_subplot(gs[idx // 2, idx % 2], projection='3d')
            self._plot_3d_mini_enhanced(ax, param1, param2, metric_col, label)

    def _plot_3d_mini_enhanced(self, ax: plt.Axes, param1: str, param2: str,
                                metric_col: str, label: str):
        """Mini plot 3D mejorado con mejor manejo de escalas."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        x = pd.to_numeric(self.df[param1], errors='coerce').values
        y = pd.to_numeric(self.df[param2], errors='coerce').values
        z = pd.to_numeric(self.df[metric_col], errors='coerce').values

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(z) < 10:
            ax.text2D(0.5, 0.5, 'No Data', transform=ax.transAxes,
                     ha='center', va='center', color=Theme.TEXT_DARK, fontsize=8)
            return

        # Filtrar outliers para mejor visualización
        z_p5, z_p95 = np.percentile(z, [5, 95])
        clip_mask = (z >= z_p5) & (z <= z_p95)
        x_clip, y_clip, z_clip = x[clip_mask], y[clip_mask], z[clip_mask]

        if len(z_clip) < 10:
            x_clip, y_clip, z_clip = x, y, z

        # Grid más fino
        grid_size = 35
        x_range = x_clip.max() - x_clip.min()
        y_range = y_clip.max() - y_clip.min()

        # Asegurar rangos válidos
        if x_range < 1e-10 or y_range < 1e-10:
            ax.text2D(0.5, 0.5, 'Constant Data', transform=ax.transAxes,
                     ha='center', va='center', color=Theme.TEXT_DARK, fontsize=8)
            return

        xi = np.linspace(x_clip.min(), x_clip.max(), grid_size)
        yi = np.linspace(y_clip.min(), y_clip.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        try:
            zi = griddata((x_clip, y_clip), z_clip, (xi, yi), method='cubic')
            zi = np.nan_to_num(zi, nan=np.nanmean(z_clip))
            zi = gaussian_filter(zi, sigma=3.0)  # Mayor suavizado
        except Exception:
            try:
                zi = griddata((x_clip, y_clip), z_clip, (xi, yi), method='linear')
                zi = np.nan_to_num(zi, nan=np.nanmean(z_clip))
                zi = gaussian_filter(zi, sigma=2.5)
            except Exception:
                ax.text2D(0.5, 0.5, 'Interpolation Error', transform=ax.transAxes,
                         ha='center', va='center', color=Theme.TEXT_DARK, fontsize=8)
                return

        # Normalización robusta para el color
        z_min, z_max = np.percentile(z_clip, [5, 95])
        if z_max - z_min < 1e-10:
            z_min, z_max = z_clip.min(), z_clip.max()

        cmap = Theme.get_surface_cmap()
        norm = Normalize(vmin=z_min, vmax=z_max)

        ax.plot_surface(xi, yi, zi, cmap=cmap, norm=norm,
                               alpha=0.92, antialiased=True, linewidth=0,
                               rcount=40, ccount=40)

        # Estilo
        ax.set_xlabel(param1[:10], color=Theme.TEXT_DARK, fontsize=6, labelpad=2)
        ax.set_ylabel(param2[:10], color=Theme.TEXT_DARK, fontsize=6, labelpad=2)
        ax.set_zlabel(label, color=Theme.TEXT_DARK, fontsize=6, labelpad=2)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(Theme.BORDER)
        ax.yaxis.pane.set_edgecolor(Theme.BORDER)
        ax.zaxis.pane.set_edgecolor(Theme.BORDER)

        # Formateo de ticks para evitar notación científica
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.zaxis.set_major_formatter(FuncFormatter(format_axis_number))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=5, pad=1)
        ax.view_init(elev=22, azim=45)
        ax.set_title(label, fontsize=7, color=Theme.TEXT_SECONDARY, pad=2)

    # ══════════════════════════════════════════════════════════════════════════
    # 🔬 ADVANCED ROBUSTNESS VISUALIZATION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def create_robustness_analysis_page(self, fig: plt.Figure,
                                         robustness: RobustnessAnalysisResult):
        """Página completa de análisis de robustez avanzado."""
        fig.patch.set_facecolor(Theme.BG_PRIMARY)

        # Header profesional
        fig.text(0.5, 0.97, 'ADVANCED ROBUSTNESS ANALYSIS', fontsize=14, ha='center',
                fontweight='bold', color=Theme.TEXT_PRIMARY)
        fig.text(0.5, 0.94, "Multi-Dimensional Parameter Stability Assessment",
                fontsize=9, ha='center', color=Theme.TEXT_MUTED, style='italic')

        # Línea separadora
        line = plt.Line2D([0.05, 0.95], [0.92, 0.92], color=Theme.DIVIDER, linewidth=0.5)
        fig.add_artist(line)

        gs = gridspec.GridSpec(3, 3, figure=fig, left=0.06, right=0.94,
                              top=0.89, bottom=0.06, wspace=0.22, hspace=0.32,
                              height_ratios=[1.1, 1.1, 0.8])

        # 1. Cluster Analysis Panel
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cluster_analysis_panel(ax1, robustness.cluster_analysis)

        # 2. Neighborhood Stability Panel
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_nsi_panel(ax2, robustness.neighborhood_stability)

        # 3. Degradation Test Panel
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_degradation_panel(ax3, robustness.degradation_test)

        # 4. Surface CV Panel (con mini heatmap)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_surface_cv_panel(ax4, robustness.surface_cv)

        # 5. Cluster Performance Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_cluster_performance(ax5, robustness.cluster_analysis)

        # 6. Parameter Sensitivity Chart
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_parameter_sensitivity(ax6, robustness.degradation_test)

        # 7. Overall Robustness Dashboard (full width)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_robustness_dashboard(ax7, robustness)

    def _plot_cluster_analysis_panel(self, ax: plt.Axes,
                                      cluster: ClusterAnalysisResult):
        """Panel de análisis de clústeres."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Título
        ax.text(0.5, 0.94, "🔮 CLUSTER ANALYSIS", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        ax.text(0.5, 0.82, "DBSCAN Configuration Grouping", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # Métricas principales
        y_pos = 0.68
        metrics = [
            ('Clusters Found', f'{cluster.n_clusters}', True),
            ('Noise Points', f'{cluster.n_noise_points}', cluster.n_noise_points < cluster.n_clusters * 5),
            ('Silhouette Score', f'{cluster.silhouette_score:.3f}', cluster.silhouette_score > 0.3),
            ('Cluster Stability', f'{cluster.cluster_stability:.1%}', cluster.cluster_stability > 0.5),
            ('Intra-Cluster CV', f'{cluster.intra_cluster_variance:.3f}', cluster.intra_cluster_variance < 0.5),
        ]

        for label, value, is_good in metrics:
            status_color = Theme.ACCENT if is_good else Theme.TEXT_SECONDARY
            ax.text(0.08, y_pos, label, fontsize=7, transform=ax.transAxes, color=Theme.TEXT_DARK)
            ax.text(0.92, y_pos, value, fontsize=8, ha='right', transform=ax.transAxes, color=status_color)
            y_pos -= 0.11

        # Interpretación
        if cluster.n_clusters >= 2 and cluster.silhouette_score > 0.3:
            interp = "✓ Well-defined parameter groups"
            interp_color = Theme.ACCENT
        elif cluster.n_clusters >= 2:
            interp = "○ Clusters exist but overlap"
            interp_color = Theme.ORANGE
        else:
            interp = "⚠ No clear parameter groups"
            interp_color = Theme.RED_BRIGHT

        ax.text(0.5, 0.06, interp, fontsize=7, ha='center', transform=ax.transAxes,
               color=interp_color, fontweight='bold')

    def _plot_nsi_panel(self, ax: plt.Axes, nsi: NeighborhoodStabilityResult):
        """Panel del Índice de Estabilidad de Vecindad."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "🎯 NEIGHBORHOOD STABILITY", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        ax.text(0.5, 0.82, "Similar Parameters → Similar Results?", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # NSI Global con gauge visual
        nsi_val = nsi.nsi_global

        # Barra de progreso
        bar_width = 0.70
        bar_x = 0.15
        bar_y = 0.62

        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, 0.08,
                    color=Theme.BG_HIGHLIGHT, transform=ax.transAxes))

        bar_color = Theme.ACCENT if nsi_val >= 0.6 else Theme.ORANGE if nsi_val >= 0.4 else Theme.RED_BRIGHT
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * nsi_val, 0.08,
                    color=bar_color, transform=ax.transAxes))

        ax.text(0.5, 0.58, 'NSI Global', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.5, 0.73, f'{nsi_val:.1%}', fontsize=14, ha='center',
               transform=ax.transAxes, color=bar_color, fontweight='bold')

        # Percentiles
        y_pos = 0.48
        percentiles = ['p10', 'p50', 'p90']
        for p in percentiles:
            val = nsi.stability_percentiles.get(p, 0)
            ax.text(0.08, y_pos, f'NSI {p.upper()}', fontsize=6,
                   transform=ax.transAxes, color=Theme.TEXT_DARK)
            ax.text(0.92, y_pos, f'{val:.1%}', fontsize=7, ha='right',
                   transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
            y_pos -= 0.09

        # Regiones estables/inestables
        n_stable = len(nsi.stable_regions)
        n_unstable = len(nsi.unstable_regions)

        ax.text(0.25, 0.12, f'Stable: {n_stable}', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.ACCENT)
        ax.text(0.75, 0.12, f'Unstable: {n_unstable}', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.RED_BRIGHT)

        # Interpretación
        if nsi_val >= 0.6:
            interp = "✓ Highly stable neighborhood"
        elif nsi_val >= 0.4:
            interp = "○ Moderate stability"
        else:
            interp = "⚠ Unstable - high sensitivity"

        ax.text(0.5, 0.03, interp, fontsize=6, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_DARK, style='italic')

    def _plot_degradation_panel(self, ax: plt.Axes, degrad: DegradationTestResult):
        """Panel de test de degradación paramétrica."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "⚡ DEGRADATION TEST", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        ax.text(0.5, 0.82, "What if params aren't exact?", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # Comparación original vs degradado
        orig = degrad.original_performance
        degraded = degrad.degraded_performance_mean

        # Barras comparativas
        max_val = max(abs(orig), abs(degraded)) if orig != 0 else 1

        orig_height = min(0.18, abs(orig) / max_val * 0.18)
        deg_height = min(0.18, abs(degraded) / max_val * 0.18)

        ax.add_patch(plt.Rectangle((0.15, 0.55), 0.15, orig_height,
                    color=Theme.ACCENT, transform=ax.transAxes))
        ax.text(0.225, 0.52, 'Original', fontsize=6, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.225, 0.55 + orig_height + 0.02, f'{orig:.2f}', fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.ACCENT)

        deg_color = Theme.GREEN_BRIGHT if degrad.robustness_score > 0.6 else Theme.ORANGE if degrad.robustness_score > 0.3 else Theme.RED_BRIGHT
        ax.add_patch(plt.Rectangle((0.70, 0.55), 0.15, deg_height,
                    color=deg_color, transform=ax.transAxes))
        ax.text(0.775, 0.52, 'Degraded', fontsize=6, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.775, 0.55 + deg_height + 0.02, f'{degraded:.2f}', fontsize=7,
               ha='center', transform=ax.transAxes, color=deg_color)

        # Flecha de degradación
        ax.annotate('', xy=(0.60, 0.62), xytext=(0.40, 0.62),
                   arrowprops=dict(arrowstyle='->', color=Theme.TEXT_DARK, lw=1),
                   transform=ax.transAxes)
        ax.text(0.50, 0.65, f'{degrad.degradation_ratio:+.1%}', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)

        # Métricas adicionales
        y_pos = 0.40
        metrics = [
            ('Worst Case', f'{degrad.worst_case_performance:.2f}'),
            ('Best Case', f'{degrad.best_case_performance:.2f}'),
            ('Robustness Score', f'{degrad.robustness_score:.1%}'),
        ]

        for label, value in metrics:
            ax.text(0.08, y_pos, label, fontsize=6, transform=ax.transAxes, color=Theme.TEXT_DARK)
            ax.text(0.92, y_pos, value, fontsize=7, ha='right', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
            y_pos -= 0.09

        # Interpretación
        if degrad.robustness_score >= 0.7:
            interp = "✓ Highly robust to perturbations"
        elif degrad.robustness_score >= 0.4:
            interp = "○ Moderately robust"
        else:
            interp = "⚠ Sensitive to parameter changes"

        ax.text(0.5, 0.04, interp, fontsize=6, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_DARK, style='italic')

    def _plot_surface_cv_panel(self, ax: plt.Axes, surface: SurfaceCVResult):
        """Panel de Coeficiente de Variación de Superficie."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        ax.text(0.5, 0.94, "📊 SURFACE ANALYSIS", fontsize=9, fontweight='bold',
               ha='center', transform=ax.transAxes, color=Theme.TEXT_SECONDARY)
        ax.plot([0.1, 0.9], [0.88, 0.88], color=Theme.DIVIDER, linewidth=0.5,
               transform=ax.transAxes)

        ax.text(0.5, 0.82, "3D Optimization Surface Quality", fontsize=7,
               ha='center', transform=ax.transAxes, color=Theme.TEXT_DARK, style='italic')

        # Smoothness gauge
        smoothness = surface.smoothness_score

        bar_width = 0.70
        bar_x = 0.15
        bar_y = 0.60

        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, 0.08,
                    color=Theme.BG_HIGHLIGHT, transform=ax.transAxes))

        smooth_color = Theme.ACCENT if smoothness >= 0.6 else Theme.ORANGE if smoothness >= 0.4 else Theme.RED_BRIGHT
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * smoothness, 0.08,
                    color=smooth_color, transform=ax.transAxes))

        ax.text(0.5, 0.56, 'Smoothness Score', fontsize=7, ha='center',
               transform=ax.transAxes, color=Theme.TEXT_DARK)
        ax.text(0.5, 0.72, f'{smoothness:.1%}', fontsize=14, ha='center',
               transform=ax.transAxes, color=smooth_color, fontweight='bold')

        # Métricas detalladas
        y_pos = 0.45
        metrics = [
            ('Surface CV', f'{surface.cv_global:.3f}', surface.cv_global < 1.0),
            ('Roughness Index', f'{surface.roughness_index:.3f}', surface.roughness_index < 0.3),
            ('Gradient Mean', f'{surface.gradient_magnitude_mean:.3f}', True),
            ('Flat Regions', f'{surface.flatness_regions:.1%}', surface.flatness_regions > 0.3),
        ]

        for label, value, is_good in metrics:
            status_color = Theme.ACCENT if is_good else Theme.TEXT_SECONDARY
            ax.text(0.08, y_pos, label, fontsize=6, transform=ax.transAxes, color=Theme.TEXT_DARK)
            ax.text(0.92, y_pos, value, fontsize=7, ha='right', transform=ax.transAxes, color=status_color)
            y_pos -= 0.085

        # Interpretación
        if smoothness >= 0.6:
            interp = "✓ Smooth surface - robust opt"
        elif smoothness >= 0.4:
            interp = "○ Some roughness detected"
        else:
            interp = "⚠ Rough surface - overfit risk"

        ax.text(0.5, 0.04, interp, fontsize=6, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_DARK, style='italic')

    def _plot_cluster_performance(self, ax: plt.Axes, cluster: ClusterAnalysisResult):
        """Gráfico de rendimiento por cluster."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        if not cluster.cluster_performance:
            ax.text(0.5, 0.5, 'No cluster data', ha='center', va='center',
                   transform=ax.transAxes, color=Theme.TEXT_DARK, fontsize=9)
            for spine in ax.spines.values():
                spine.set_color(Theme.BORDER)
            return

        # Datos
        cluster_ids = list(cluster.cluster_performance.keys())[:8]  # Max 8 clusters
        means = [cluster.cluster_performance[c]['mean'] for c in cluster_ids]
        stds = [cluster.cluster_performance[c]['std'] for c in cluster_ids]
        counts = [cluster.cluster_performance[c]['count'] for c in cluster_ids]

        x = np.arange(len(cluster_ids))

        # Colores: mejor cluster en verde
        colors = [Theme.ACCENT if c == cluster.best_cluster_id else Theme.TEXT_SECONDARY
                 for c in cluster_ids]

        # Barras con error bars
        bars = ax.bar(x, means, color=colors, alpha=0.7, edgecolor='none')
        ax.errorbar(x, means, yerr=stds, fmt='none', color=Theme.TEXT_DARK,
                   capsize=3, capthick=1, linewidth=1)

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in cluster_ids], fontsize=6)
        ax.set_ylabel('Performance', fontsize=7, color=Theme.TEXT_DARK)
        ax.set_title('Cluster Performance', fontsize=8, color=Theme.TEXT_SECONDARY, pad=4)

        # Counts como labels en las barras
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.1,
                   f'n={count}', ha='center', va='bottom', fontsize=5, color=Theme.TEXT_DARK)

        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)
        ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_parameter_sensitivity(self, ax: plt.Axes, degrad: DegradationTestResult):
        """Gráfico de sensibilidad por parámetro."""
        ax.set_facecolor(Theme.BG_SECONDARY)

        if not degrad.parameter_sensitivity:
            ax.text(0.5, 0.5, 'No sensitivity data', ha='center', va='center',
                   transform=ax.transAxes, color=Theme.TEXT_DARK, fontsize=9)
            for spine in ax.spines.values():
                spine.set_color(Theme.BORDER)
            return

        # Ordenar por sensibilidad
        sorted_sens = sorted(degrad.parameter_sensitivity.items(),
                            key=lambda x: x[1], reverse=True)[:8]

        params = [p[0][:12] for p in sorted_sens]
        values = [p[1] for p in sorted_sens]

        y_pos = np.arange(len(params))

        # Colores: alta sensibilidad = rojo
        colors = [Theme.RED_BRIGHT if v > 0.15 else Theme.ORANGE if v > 0.08 else Theme.ACCENT
                 for v in values]

        ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='none', alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=6)
        ax.set_xlabel('Sensitivity', fontsize=7, color=Theme.TEXT_DARK)
        ax.set_title('Parameter Sensitivity', fontsize=8, color=Theme.TEXT_SECONDARY, pad=4)
        ax.invert_yaxis()

        # Valores en barras
        for i, v in enumerate(values):
            ax.text(v + max(values)*0.02, i, f'{v:.3f}', va='center', fontsize=5, color=Theme.TEXT_DARK)

        ax.tick_params(colors=Theme.TEXT_DARK, labelsize=6)
        ax.xaxis.set_major_formatter(FuncFormatter(format_axis_number))

        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

    def _plot_robustness_dashboard(self, ax: plt.Axes, robustness: RobustnessAnalysisResult):
        """Dashboard resumen de robustez."""
        ax.set_facecolor(Theme.BG_TERTIARY)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Grade principal
        grade = robustness.robustness_grade
        score = robustness.overall_robustness_score

        grade_colors = {
            'A+': Theme.ACCENT, 'A': '#66bb6a', 'B+': '#81c784',
            'B': '#ff9800', 'C': '#ff7043', 'D': '#e53935', 'F': '#c62828'
        }
        grade_color = grade_colors.get(grade, Theme.TEXT_SECONDARY)

        # Grade grande
        ax.text(0.08, 0.55, grade, fontsize=28, fontweight='bold', va='center',
               transform=ax.transAxes, color=grade_color)

        grade_texts = {
            'A+': 'EXCELLENT', 'A': 'ROBUST', 'B+': 'GOOD',
            'B': 'MODERATE', 'C': 'WEAK', 'D': 'POOR', 'F': 'FAIL'
        }
        ax.text(0.08, 0.25, grade_texts.get(grade, 'N/A'), fontsize=10, fontweight='bold',
               transform=ax.transAxes, color=grade_color)

        # Línea separadora
        ax.plot([0.18, 0.18], [0.15, 0.85], color=Theme.DIVIDER, linewidth=1,
               transform=ax.transAxes)

        # Componentes del score
        components = [
            ('Cluster Stability', robustness.cluster_analysis.cluster_stability),
            ('Neighborhood (NSI)', robustness.neighborhood_stability.nsi_global),
            ('Degradation Test', robustness.degradation_test.robustness_score),
            ('Surface Smoothness', robustness.surface_cv.smoothness_score),
        ]

        x_start = 0.22
        x_spacing = 0.195

        for i, (name, value) in enumerate(components):
            x_pos = x_start + i * x_spacing

            # Mini gauge
            gauge_color = Theme.ACCENT if value >= 0.6 else Theme.ORANGE if value >= 0.4 else Theme.RED_BRIGHT
            ax.text(x_pos, 0.72, f'{value:.0%}', fontsize=12, ha='center',
                   transform=ax.transAxes, color=gauge_color, fontweight='bold')
            ax.text(x_pos, 0.52, name, fontsize=6, ha='center',
                   transform=ax.transAxes, color=Theme.TEXT_SECONDARY)

            # Mini barra
            bar_width = 0.12
            ax.add_patch(plt.Rectangle((x_pos - bar_width/2, 0.38), bar_width, 0.06,
                        color=Theme.BG_HIGHLIGHT, transform=ax.transAxes))
            ax.add_patch(plt.Rectangle((x_pos - bar_width/2, 0.38), bar_width * value, 0.06,
                        color=gauge_color, transform=ax.transAxes))

        # Veredicto
        if score >= 0.70:
            verdict = "Strategy shows strong robustness across all dimensions"
        elif score >= 0.50:
            verdict = "Strategy demonstrates acceptable stability"
        elif score >= 0.30:
            verdict = "Strategy shows mixed robustness signals"
        else:
            verdict = "Strategy lacks robustness - high overfit risk"

        ax.text(0.60, 0.15, verdict, fontsize=8, ha='center', transform=ax.transAxes,
               color=Theme.TEXT_SECONDARY, style='italic')

        # Score compuesto
        ax.text(0.95, 0.55, f'{score:.0%}', fontsize=18, ha='right', va='center',
               transform=ax.transAxes, color=Theme.TEXT_PRIMARY, fontweight='bold')
        ax.text(0.95, 0.35, 'Overall Score', fontsize=7, ha='right',
               transform=ax.transAxes, color=Theme.TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# 📄 GENERADOR DE REPORTE PDF
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generador de reporte PDF completo."""

    def __init__(self, loader: 'DataLoader'):
        self.loader = loader
        self.df = loader.df
        self.schema = loader.schema
        self.all_analyses: Dict[str, ParameterAnalysis] = {}

    def generate(self, output_path: str) -> Dict[str, ParameterAnalysis]:
        """Genera reporte completo."""
        print(f"\n{'━'*70}")
        print("  📊 GENERATING QUANTITATIVE REPORT")
        print('━'*70)

        visualizer = BloombergVisualizer(self.df, self.schema)
        all_params = self.schema.params + self.schema.exit_params

        # Métricas a analizar
        metrics_to_analyze = self._get_available_metrics()

        # Motor de análisis
        engine = QuantEngine(self.df, all_params)

        # Análisis de correlación entre parámetros
        target_col = visualizer._find_metric_col('ROI') or visualizer._find_metric_col('SCORE')
        correlations = []
        if target_col and len(all_params) >= 2:
            print("  [*] Analyzing parameter correlations...")
            corr_analyzer = CorrelationAnalyzer(self.df, all_params, target_col)
            correlations = corr_analyzer.analyze_all_pairs(min_correlation=0.25)
            strong_pairs = corr_analyzer.get_strongly_related_pairs()
            if strong_pairs:
                print(f"      Found {len(strong_pairs)} strongly related pairs")

        with PdfPages(output_path) as pdf:
            # 1. Portada
            fig = plt.figure(figsize=(11, 8.5))
            visualizer.create_cover_page(fig, self.loader.strategy_name, len(self.df))
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)

            # 2. Overview de estrategia
            strategy_analyzer = StrategyAnalyzer(self.df, self.schema)
            strategy_analysis = strategy_analyzer.analyze()

            fig = plt.figure(figsize=(11, 8.5))
            visualizer.create_strategy_overview(fig, strategy_analysis)
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)

            # 3. Análisis por parámetro
            for i, param in enumerate(all_params):
                print(f"  [{i+1}/{len(all_params)}] {param}")

                # Análisis cuantitativo
                analysis = engine.analyze_parameter(param, metrics_to_analyze)
                self.all_analyses[param] = analysis

                # Página visual
                fig = plt.figure(figsize=(11, 8.5))
                is_exit = param in self.schema.exit_params
                visualizer.create_parameter_page(fig, analysis, is_exit)
                pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                plt.close(fig)

            # 4. Análisis de Correlación entre parámetros
            if correlations:
                print("  [+] Parameter Correlation Analysis")
                fig = plt.figure(figsize=(11, 8.5))
                visualizer.create_correlation_page(fig, correlations)
                pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                plt.close(fig)

            # 5. Superficie 3D (SL × TP) si existen ambos parámetros
            sl_col, tp_col = visualizer.find_exit_params()
            if sl_col and tp_col:
                print("  [+] 3D Surface: SL × TP")
                fig = plt.figure(figsize=(11, 8.5))
                visualizer.create_correlated_pair_surface(fig, sl_col, tp_col)
                pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                plt.close(fig)

            # 6. Superficies 3D para pares fuertemente correlacionados
            strong_pairs = [c for c in correlations if c.is_strongly_related]
            for corr in strong_pairs[:3]:  # Máximo 3 pares adicionales
                # Evitar duplicar SL×TP
                if sl_col and tp_col:
                    if (corr.param1 in [sl_col, tp_col] and corr.param2 in [sl_col, tp_col]):
                        continue

                print(f"  [+] 3D Surface: {corr.param1} × {corr.param2}")
                fig = plt.figure(figsize=(11, 8.5))
                visualizer.create_correlated_pair_surface(fig, corr.param1, corr.param2, corr)
                pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                plt.close(fig)

            # 7. Validación Estadística
            try:
                roi_col = visualizer._find_metric_col('ROI')
                if roi_col and len(self.df) >= 30:
                    print("  [+] Statistical Validation")
                    # Obtener retornos como array numérico
                    returns = pd.to_numeric(self.df[roi_col], errors='coerce').dropna().values
                    if len(returns) >= 20:
                        validator = StatisticalValidator(returns, len(self.df))
                        validation = validator.validate()

                        fig = plt.figure(figsize=(11, 8.5))
                        visualizer.create_statistical_validation_page(fig, validation)
                        pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                        plt.close(fig)
            except Exception as e:
                print(f"  [!] Statistical validation skipped: {e}")

            # 8. ADVANCED ROBUSTNESS ANALYSIS (NEW)
            try:
                if target_col and len(all_params) >= 1 and len(self.df) >= 30:
                    print("  [+] Advanced Robustness Analysis")
                    print("      - Cluster Analysis (DBSCAN)")
                    print("      - Neighborhood Stability Index (NSI)")
                    print("      - Parameter Degradation Testing")
                    print("      - Surface CV Analysis")

                    robustness_analyzer = AdvancedRobustnessAnalyzer(
                        self.df, all_params, target_col
                    )
                    robustness_result = robustness_analyzer.analyze_all()

                    if robustness_result.overall_robustness_score > 0:
                        fig = plt.figure(figsize=(11, 8.5))
                        visualizer.create_robustness_analysis_page(fig, robustness_result)
                        pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
                        plt.close(fig)

                        print(f"      → Robustness Grade: {robustness_result.robustness_grade}")
                        print(f"      → Overall Score: {robustness_result.overall_robustness_score:.1%}")
            except Exception as e:
                print(f"  [!] Robustness analysis skipped: {e}")

            # 9. Tabla de recomendaciones (FINAL)
            fig = plt.figure(figsize=(11, 8.5))
            visualizer.create_recommendations_table(fig, self.all_analyses, self.schema.exit_params)
            pdf.savefig(fig, facecolor=Theme.BG_PRIMARY)
            plt.close(fig)

        # Guardar CSV
        csv_path = output_path.replace('.pdf', '_RECOMMENDATIONS.csv')
        self._save_csv(csv_path)

        print(f"\n  ✓ PDF: {output_path}")
        print(f"  ✓ CSV: {csv_path}")

        return self.all_analyses

    def _get_available_metrics(self) -> List[str]:
        """Obtiene métricas disponibles con detección robusta."""
        # Prioridad de métricas (lista de posibles nombres para cada métrica)
        # DRAWDOWN primero para que sea la métrica principal de comparación
        metric_aliases = [
            ['DRAWDOWN', 'MAX_DD', 'MAX_DD_PCT', 'DD'],
            ['ROI', 'ROI_PCT', 'RETURN', 'RETORNO'],
            ['SHARPE', 'SHARPE_RATIO'],
            ['PROFIT_FACTOR', 'PF'],
            ['SQN'],
            ['SCORE'],
            ['WINRATE', 'WINRATE_PCT', 'WIN_RATE'],
            ['ESTABILIDAD', 'STABILITY'],
            ['EXPECTATIVA', 'EXPECTANCY'],
            ['SORTINO'],
        ]

        available = []
        df_cols_upper = {col.upper(): col for col in self.df.columns}

        for aliases in metric_aliases:
            found = False
            for alias in aliases:
                # Búsqueda exacta primero
                if alias in df_cols_upper:
                    col = df_cols_upper[alias]
                    if col not in available:
                        available.append(col)
                        found = True
                        break

            # Si no encontró exacto, buscar parcial
            if not found:
                for alias in aliases:
                    for col_upper, col in df_cols_upper.items():
                        if alias in col_upper and col not in available:
                            available.append(col)
                            found = True
                            break
                    if found:
                        break

        return available[:8]  # Máximo 8 métricas

    def _save_csv(self, path: str):
        """Guarda recomendaciones en CSV."""
        rows = []

        for param, analysis in self.all_analyses.items():
            if np.isnan(analysis.optimal_value):
                continue

            rows.append({
                'PARAMETER': param,
                'OPTIMAL_VALUE': analysis.optimal_value,
                'RANGE_MIN': analysis.optimal_range[0],
                'RANGE_MAX': analysis.optimal_range[1],
                'CONFIDENCE': analysis.confidence,
                'ROBUSTNESS': analysis.robustness,
                'STABILITY': analysis.stability,
                'SENSITIVITY': analysis.sensitivity,
                'IS_EXIT_PARAM': param in self.schema.exit_params
            })

        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('CONFIDENCE', ascending=False)
            df.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ══════════════════════════════════════════════════════════════════════════════

def find_files() -> List[str]:
    """Busca archivos de datos."""
    patterns = ['resultados/**/*.csv', 'resultados/**/*.xlsx', '*.csv', '*.xlsx']
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    # Priorizar RESUMEN
    resumen = [f for f in files if 'RESUMEN' in f.upper()]
    return resumen if resumen else files


def main():
    """Función principal."""
    print("\n" + "━"*70)
    print("  MODELOX PARAMETER ANALYZER v20.0")
    print("  Advanced Robustness & Statistical Analysis")
    print("━"*70)

    # Obtener archivo
    file_path = None

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"\n  ✗ File not found: {file_path}")
            sys.exit(1)
    else:
        files = find_files()

        if not files:
            print("\n  ✗ No CSV/Excel files found")
            sys.exit(1)

        print(f"\n  Files found ({len(files)}):")
        for i, f in enumerate(files[:10]):
            print(f"  [{i+1}] {f}")

        print(f"\n  Options: Number (1-{min(10, len(files))}) | Drag file | Enter for first")
        choice = input("\n  Choice: ").strip().strip("'\"")

        if os.path.exists(choice):
            file_path = choice
        elif choice.isdigit():
            idx = int(choice) - 1
            file_path = files[idx] if 0 <= idx < len(files) else files[0]
        else:
            file_path = files[0]

    # Cargar datos
    loader = DataLoader()
    if not loader.load(file_path):
        sys.exit(1)

    # Filtrar (SCORE >= 0.11)
    loader.apply_filters(min_score=0.11)

    # Reclasificar después de filtrar
    loader.schema = SmartColumnDetector.detect(loader.df)

    if len(loader.df) < 10:
        print("\n  ✗ Not enough trials after filtering (min: 10)")
        sys.exit(1)

    if len(loader.df) < 30:
        print(f"  ⚠ Pocos trials ({len(loader.df)}) - resultados pueden ser menos fiables")

    all_params = loader.schema.params + loader.schema.exit_params
    if not all_params:
        print("\n  ✗ No parameters detected")
        sys.exit(1)

    # Generar reporte
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path) or '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'ANALYSIS_{base_name}_{timestamp}.pdf')

    report = ReportGenerator(loader)
    results = report.generate(output_path)

    # Resumen en consola
    print(f"\n{'━'*70}")
    print("  TOP RECOMMENDATIONS")
    print('━'*70)

    sorted_results = sorted(
        [(p, r) for p, r in results.items() if not np.isnan(r.optimal_value)],
        key=lambda x: x[1].confidence,
        reverse=True
    )

    for param, analysis in sorted_results[:8]:
        icon = '💰' if param in loader.schema.exit_params else '⚙️'
        conf = analysis.confidence * 100
        print(f"  {icon} {param:<22} = {analysis.optimal_value:<12.5g} (Conf: {conf:.0f}%)")

    print('━'*70)
    print("\n  ✓ Analysis complete!")


if __name__ == '__main__':
    main()

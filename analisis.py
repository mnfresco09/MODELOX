#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                       MODELOX PARAMETER ANALYZER v9.0 - INSTITUTIONAL                                ║
║                    Análisis Individual por Parámetro con Reducción de Ruido                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  📊 UNA PÁGINA POR PARÁMETRO: Análisis completo y dedicado                                           ║
║  🎯 4 MÉTRICAS CLAVE: ROI, SQN, PROFIT_FACTOR, DRAWDOWN                                              ║
║  🧹 REDUCCIÓN DE RUIDO: ML para aislar efecto real de cada parámetro                                 ║
║  📈 CURVAS SUAVIZADAS: Spline + Gaussian smoothing profesional                                       ║
║  🖥️  USO: python analisis.py <archivo.csv>  o  python analisis.py (busca automático)                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import re
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ML y estadísticas
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, interp1d

warnings.filterwarnings('ignore')


# ==============================================================================
# 🎨 TEMA VISUAL PROFESIONAL (Estilo Bloomberg Terminal)
# ==============================================================================

COLORS = {
    'bg_dark': '#0a0e14',
    'bg_panel': '#0d1117', 
    'bg_card': '#161b22',
    'bg_highlight': '#1f2937',
    'text_white': '#ffffff',
    'text_primary': '#e6edf3',
    'text_gray': '#8b949e',
    'text_muted': '#484f58',
    'green': '#3fb950',
    'green_dark': '#238636',
    'red': '#f85149',
    'red_dark': '#da3633',
    'blue': '#58a6ff',
    'blue_dark': '#1f6feb',
    'purple': '#a371f7',
    'orange': '#f0883e',
    'cyan': '#39d353',
    'yellow': '#f0c14b',
    'gold': '#e3b341',
    'grid': '#21262d',
    'border': '#30363d',
}

# Colores específicos para las 4 métricas clave
METRIC_STYLE = {
    'ROI': {'color': '#3fb950', 'name': 'ROI %', 'higher_better': True},
    'SQN': {'color': '#58a6ff', 'name': 'SQN', 'higher_better': True},
    'PROFIT_FACTOR': {'color': '#a371f7', 'name': 'Profit Factor', 'higher_better': True},
    'DRAWDOWN': {'color': '#f85149', 'name': 'Drawdown %', 'higher_better': False},
}

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': COLORS['bg_dark'],
    'axes.facecolor': COLORS['bg_panel'],
    'axes.edgecolor': COLORS['border'],
    'axes.linewidth': 0.5,
    'axes.labelcolor': COLORS['text_gray'],
    'axes.titlecolor': COLORS['text_white'],
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
    'text.color': COLORS['text_white'],
    'xtick.color': COLORS['text_muted'],
    'ytick.color': COLORS['text_muted'],
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.facecolor': COLORS['bg_card'],
    'legend.edgecolor': COLORS['border'],
    'legend.fontsize': 8,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
})


# ==============================================================================
# 🔬 DETECTOR DE COLUMNAS (Métricas vs Parámetros)
# ==============================================================================

@dataclass
class ColumnInfo:
    """Clasificación de columnas del dataset."""
    metrics: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    system: List[str] = field(default_factory=list)


class ColumnDetector:
    """Detecta y clasifica columnas automáticamente."""
    
    # Métricas conocidas de MODELOX (invariantes)
    KNOWN_METRICS = {
        'ROI', 'ROI_PCT', 'SCORE', 'SHARPE', 'SORTINO', 'CALMAR', 'SQN',
        'PROFIT_FACTOR', 'PF', 'PAYOFF', 'PAYOFF_RATIO',
        'DRAWDOWN', 'MAX_DD', 'DD', 'MDD', 'MAX_DRAWDOWN',
        'WINRATE', 'WIN_RATE', 'PORC_GANADORAS', 'PORC_PERDEDORAS',
        'N_TRADES', 'TOTAL_TRADES', 'NUM_TRADES', 'TRADES', 'TRADES_POR_DIA',
        'EXPECTATIVA', 'EXPECTANCY', 'RETORNO_PROMEDIO',
        'RACHA_GANADORA', 'RACHA_PERDEDORA',
        'ESTABILIDAD', 'STABILITY',
        'SALDO_ACTUAL', 'SALDO_MIN', 'SALDO_MAX', 'SALDO_MEAN',
        'PNL', 'PNL_NETO', 'NET_PNL',
        'AVG_WIN', 'AVG_LOSS', 'MAX_GANANCIA', 'MAX_PERDIDA',
        'N_TRADES_LONG', 'N_TRADES_SHORT', 'NUM_LONGS', 'NUM_SHORTS',
        'COUNT_LONGS', 'COUNT_SHORTS',
        'COMISIONES_TOTAL', 'SALDO_SIN_COMISIONES',
        'DURATION_MEAN_MIN', 'RIESGO_BENEFICIO',
        'PNL_NETO_POR_DIA_OPERADO',
    }
    
    KNOWN_IDENTIFIERS = {
        'TRIAL', 'INDEX', 'ID', 'ESTRATEGIA', 'STRATEGY', 'NOMBRE', 'NAME',
        'COMBO', 'NOMBRE_COMBO', 'PERTURBADO', 'SEED', 'CONFIG',
    }
    
    # Patrones de EXIT que SÍ son parámetros analizables
    EXIT_PATTERNS = [
        r'EXIT_SL', r'EXIT_TP', r'EXIT_TRAIL', r'SL_PCT', r'TP_PCT',
        r'STOP_LOSS', r'TAKE_PROFIT', r'TRAILING', r'^SL$', r'^TP$',
        r'SL_ACTIVATION', r'TP_ACTIVATION', r'TRAIL_ACT', r'TRAIL_DIST',
    ]
    
    @classmethod
    def classify(cls, df: pd.DataFrame) -> ColumnInfo:
        """Clasifica todas las columnas."""
        result = ColumnInfo()
        
        for col in df.columns:
            col_str = str(col).strip()
            col_upper = col_str.upper().replace(' ', '_')
            
            if not col_upper or col_upper.startswith('UNNAMED'):
                continue
            
            # Sistema interno (__ prefix)
            if col_str.startswith('__'):
                result.system.append(col_str)
                continue
            
            # Identificadores
            if col_upper in cls.KNOWN_IDENTIFIERS:
                result.identifiers.append(col_str)
                continue
            
            # Métricas conocidas
            if cls._is_metric(col_upper):
                result.metrics.append(col_str)
                continue
            
            # Parámetros EXIT (SL/TP) - son analizables si tienen variación
            if cls._is_exit_param(col_upper):
                if cls._is_numeric_variable(df[col]):
                    result.parameters.append(col_str)
                else:
                    result.system.append(col_str)
                continue
            
            # Todo lo demás numérico con variación = parámetro
            if cls._is_numeric_variable(df[col]):
                result.parameters.append(col_str)
            else:
                result.system.append(col_str)
        
        return result
    
    @classmethod
    def _is_metric(cls, col: str) -> bool:
        if col in cls.KNOWN_METRICS:
            return True
        for m in cls.KNOWN_METRICS:
            if col.startswith(m + '_') or col.endswith('_' + m):
                return True
        return False
    
    @classmethod
    def _is_exit_param(cls, col: str) -> bool:
        for pattern in cls.EXIT_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def _is_numeric_variable(cls, series: pd.Series) -> bool:
        try:
            numeric = pd.to_numeric(series, errors='coerce')
            valid = numeric.dropna()
            return len(valid) >= 5 and valid.nunique() >= 2
        except:
            return False


# ==============================================================================
# 📊 CARGADOR DE DATOS (CSV + EXCEL)
# ==============================================================================

class DataLoader:
    """Carga CSV y Excel con detección automática."""
    
    HEADER_KEYWORDS = {'ROI', 'SCORE', 'TRIAL', 'DRAWDOWN', 'SQN', 'SHARPE', 'ESTRATEGIA'}
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.columns: Optional[ColumnInfo] = None
        self.file_path: str = ""
        self.strategy_name: str = "unknown"
    
    def load(self, path: str) -> bool:
        """Carga archivo y clasifica columnas."""
        self.file_path = path
        ext = os.path.splitext(path)[1].lower()
        
        print(f"\n{'═'*70}")
        print(f"📂 CARGANDO: {os.path.basename(path)}")
        print('═'*70)
        
        try:
            if ext in ['.xlsx', '.xls']:
                self.df = self._load_excel(path)
            else:
                self.df = self._load_csv(path)
            
            if self.df is None or len(self.df) == 0:
                print("❌ Error: No se pudieron cargar datos")
                return False
            
            self._clean()
            self.columns = ColumnDetector.classify(self.df)
            
            # Extraer nombre de estrategia
            for col in ['ESTRATEGIA', 'STRATEGY', 'NOMBRE']:
                if col in self.df.columns:
                    vals = self.df[col].dropna().unique()
                    if len(vals) > 0:
                        self.strategy_name = str(vals[0])
                        break
            
            self._print_summary()
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_excel(self, path: str) -> pd.DataFrame:
        """Carga Excel con detección inteligente de header."""
        print("   📊 Formato: Excel")
        
        xlsx = pd.ExcelFile(path)
        sheet = xlsx.sheet_names[0]
        
        # Detectar fila de header
        preview = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
        header_row = 0
        best_score = 0
        
        for idx in range(min(10, len(preview))):
            row_vals = [str(v).upper() for v in preview.iloc[idx] if pd.notna(v)]
            score = sum(1 for v in row_vals for kw in self.HEADER_KEYWORDS if kw in v)
            if score > best_score:
                best_score = score
                header_row = idx
        
        print(f"   🔍 Header detectado en fila: {header_row}")
        return pd.read_excel(path, sheet_name=sheet, header=header_row)
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """Carga CSV con detección de delimitador."""
        print("   📋 Formato: CSV")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2048)
        
        delims = {',': sample.count(','), ';': sample.count(';'), '\t': sample.count('\t')}
        best = max(delims, key=delims.get)
        
        return pd.read_csv(path, sep=best)
    
    def _clean(self):
        """Limpia el DataFrame."""
        self.df = self.df.dropna(how='all')
        self.df = self.df.dropna(axis=1, how='all')
        self.df.columns = [str(c).strip() for c in self.df.columns]
        
        cols_drop = [c for c in self.df.columns if str(c).lower().startswith('unnamed')]
        self.df = self.df.drop(columns=cols_drop, errors='ignore')
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                numeric = pd.to_numeric(self.df[col], errors='coerce')
                if numeric.notna().sum() > len(self.df) * 0.5:
                    self.df[col] = numeric
    
    def _print_summary(self):
        """Imprime resumen."""
        print(f"\n   ✅ {len(self.df):,} trials cargados")
        print(f"   📋 Estrategia: {self.strategy_name}")
        print(f"\n   📊 COLUMNAS:")
        print(f"      Métricas:    {len(self.columns.metrics)}")
        print(f"      Parámetros:  {len(self.columns.parameters)}")
        print(f"      Sistema:     {len(self.columns.system)}")
        
        if self.columns.parameters:
            print(f"\n   ⚙️  PARÁMETROS A ANALIZAR:")
            for p in self.columns.parameters:
                try:
                    vals = pd.to_numeric(self.df[p], errors='coerce').dropna()
                    print(f"      • {p:<28} [{vals.min():.4g} → {vals.max():.4g}]")
                except:
                    print(f"      • {p}")


# ==============================================================================
# 🧹 MOTOR DE REDUCCIÓN DE RUIDO (ML-Based)
# ==============================================================================

class NoiseReducer:
    """
    Reduce el ruido causado por otros parámetros usando Machine Learning.
    
    El problema: Cuando analizamos el efecto de un parámetro (ej: RSI_PERIOD)
    sobre una métrica (ej: ROI), los otros parámetros (ATR_MULT, etc.) causan
    variación que oscurece la relación real.
    
    Solución: Usar ML para predecir la métrica basándose en OTROS parámetros,
    luego calcular residuos. Estos residuos son el efecto "limpio" del parámetro target.
    """
    
    @staticmethod
    def isolate_effect(df: pd.DataFrame, 
                       target_param: str, 
                       metric: str,
                       other_params: List[str],
                       method: str = 'residual') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Aísla el efecto de un parámetro sobre una métrica.
        
        Args:
            df: DataFrame con todos los datos
            target_param: Parámetro a analizar
            metric: Métrica objetivo
            other_params: Lista de otros parámetros (causan ruido)
            method: 'residual' (ML), 'binning' (estratificación), 'raw' (sin filtrar)
        
        Returns:
            (x_values, y_values, stats_dict)
        """
        # Extraer datos válidos
        cols_needed = [target_param, metric] + [p for p in other_params if p in df.columns and p != target_param]
        df_work = df[cols_needed].copy()
        
        for col in cols_needed:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
        df_work = df_work.dropna()
        
        if len(df_work) < 30:
            # Datos insuficientes - devolver raw
            x = df_work[target_param].values
            y = df_work[metric].values
            return x, y, {'method': 'raw', 'noise_reduction': 0}
        
        x = df_work[target_param].values
        y = df_work[metric].values
        
        if method == 'raw' or len(other_params) == 0:
            return x, y, {'method': 'raw', 'noise_reduction': 0}
        
        # ═══════════════════════════════════════════════════════════════════
        # MÉTODO RESIDUAL (ML-based noise reduction)
        # ═══════════════════════════════════════════════════════════════════
        if method == 'residual':
            # Obtener otros parámetros como features
            other_cols = [p for p in other_params if p in df_work.columns and p != target_param]
            
            if len(other_cols) == 0:
                return x, y, {'method': 'raw', 'noise_reduction': 0}
            
            X_other = df_work[other_cols].values
            
            # Normalizar features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_other)
            
            # Entrenar modelo para predecir métrica desde OTROS parámetros
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_scaled, y)
            
            # Predecir y calcular residuos
            y_pred = model.predict(X_scaled)
            residuals = y - y_pred
            
            # El efecto "limpio" es: media de y + residuos
            y_clean = np.mean(y) + residuals
            
            # Calcular reducción de ruido
            original_var = np.var(y)
            residual_var = np.var(residuals)
            noise_reduction = 1 - (residual_var / original_var) if original_var > 0 else 0
            
            return x, y_clean, {
                'method': 'residual',
                'noise_reduction': max(0, noise_reduction),
                'r2_other_params': model.score(X_scaled, y),
                'n_other_params': len(other_cols)
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # MÉTODO BINNING (estratificación por grupos)
        # ═══════════════════════════════════════════════════════════════════
        elif method == 'binning':
            other_cols = [p for p in other_params if p in df_work.columns and p != target_param]
            
            if len(other_cols) == 0:
                return x, y, {'method': 'raw', 'noise_reduction': 0}
            
            # Crear bins para otros parámetros
            df_temp = df_work.copy()
            for col in other_cols[:3]:  # Max 3 para evitar explosión combinatoria
                try:
                    df_temp[f'_bin_{col}'] = pd.qcut(df_temp[col], q=3, labels=False, duplicates='drop')
                except:
                    df_temp[f'_bin_{col}'] = 0
            
            bin_cols = [c for c in df_temp.columns if c.startswith('_bin_')]
            
            if len(bin_cols) == 0:
                return x, y, {'method': 'raw', 'noise_reduction': 0}
            
            # Agrupar por target_param y bins, calcular media
            grouped = df_temp.groupby([target_param] + bin_cols)[metric].mean().reset_index()
            final = grouped.groupby(target_param)[metric].mean().reset_index()
            
            return final[target_param].values, final[metric].values, {
                'method': 'binning',
                'noise_reduction': 0.3,  # Estimación conservadora
                'n_bins': len(bin_cols)
            }
        
        return x, y, {'method': 'raw', 'noise_reduction': 0}
    
    @staticmethod
    def smooth_curve(x: np.ndarray, y: np.ndarray, 
                     smoothing: float = 0.3,
                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suaviza la curva usando spline + gaussian.
        
        Args:
            x, y: Datos originales
            smoothing: Factor de suavizado (0=ninguno, 1=máximo)
            n_points: Número de puntos para la curva suavizada
        
        Returns:
            (x_smooth, y_smooth)
        """
        if len(x) < 5:
            return x, y
        
        # Ordenar por x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Agrupar valores repetidos de x (promediar y)
        df_temp = pd.DataFrame({'x': x_sorted, 'y': y_sorted})
        grouped = df_temp.groupby('x')['y'].agg(['mean', 'std', 'count']).reset_index()
        x_unique = grouped['x'].values
        y_mean = grouped['mean'].values
        
        if len(x_unique) < 4:
            return x_unique, y_mean
        
        # Crear puntos para interpolación
        x_smooth = np.linspace(x_unique.min(), x_unique.max(), n_points)
        
        try:
            # Spline suavizado
            # s = smoothing factor (más alto = más suave)
            s_factor = len(x_unique) * smoothing * 0.5
            spline = UnivariateSpline(x_unique, y_mean, s=s_factor)
            y_smooth = spline(x_smooth)
            
            # Aplicar filtro gaussiano adicional
            sigma = max(1, int(n_points * smoothing * 0.1))
            y_smooth = gaussian_filter1d(y_smooth, sigma=sigma)
            
        except Exception:
            # Fallback: interpolación lineal + gaussian
            f = interp1d(x_unique, y_mean, kind='linear', fill_value='extrapolate')
            y_smooth = f(x_smooth)
            sigma = max(1, int(n_points * smoothing * 0.1))
            y_smooth = gaussian_filter1d(y_smooth, sigma=sigma)
        
        return x_smooth, y_smooth


# ==============================================================================
# 📈 ANALIZADOR ESTADÍSTICO
# ==============================================================================

class StatsAnalyzer:
    """Calcula estadísticas avanzadas para el análisis."""
    
    @staticmethod
    def correlation_analysis(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Análisis de correlación completo."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 10:
            return {'pearson': 0, 'spearman': 0, 'strength': 'N/A', 'p_value': 1}
        
        pearson, p_pearson = stats.pearsonr(x, y)
        spearman, p_spearman = stats.spearmanr(x, y)
        
        # Clasificar fuerza
        r = abs(spearman)
        if r >= 0.7:
            strength = 'FUERTE'
        elif r >= 0.4:
            strength = 'MODERADA'
        elif r >= 0.2:
            strength = 'DÉBIL'
        else:
            strength = 'NINGUNA'
        
        return {
            'pearson': pearson,
            'spearman': spearman,
            'p_value': min(p_pearson, p_spearman),
            'strength': strength,
            'significant': min(p_pearson, p_spearman) < 0.05
        }
    
    @staticmethod
    def optimal_zone(x: np.ndarray, y: np.ndarray, 
                     higher_better: bool = True,
                     top_pct: float = 0.2) -> Dict[str, Any]:
        """Encuentra la zona óptima del parámetro."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 10:
            return {'optimal_min': np.nan, 'optimal_max': np.nan, 'optimal_mean': np.nan}
        
        # Encontrar top performers
        n_top = max(5, int(len(y) * top_pct))
        
        if higher_better:
            top_idx = np.argsort(y)[-n_top:]
        else:
            top_idx = np.argsort(y)[:n_top]
        
        x_top = x[top_idx]
        
        return {
            'optimal_min': np.percentile(x_top, 10),
            'optimal_max': np.percentile(x_top, 90),
            'optimal_mean': np.mean(x_top),
            'optimal_median': np.median(x_top),
            'n_samples': len(x_top)
        }
    
    @staticmethod
    def sensitivity_score(x: np.ndarray, y: np.ndarray) -> float:
        """Calcula sensibilidad: cuánto cambia Y por unidad de cambio en X."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 10:
            return 0.0
        
        # Normalizar ambos
        x_norm = (x - x.mean()) / (x.std() + 1e-10)
        y_norm = (y - y.mean()) / (y.std() + 1e-10)
        
        # Pendiente normalizada
        try:
            slope, _, r, _, _ = stats.linregress(x_norm, y_norm)
            return abs(slope) * (r ** 2)  # Ponderado por R²
        except:
            return 0.0


# ==============================================================================
# 📊 GENERADOR DE VISUALIZACIONES
# ==============================================================================

class ParameterVisualizer:
    """
    Genera visualización completa para UN parámetro.
    
    Cada parámetro tiene su propia página con:
    - 4 gráficos (uno por métrica: ROI, SQN, PF, DD)
    - Curvas suavizadas con banda de confianza
    - Zona óptima marcada
    - Estadísticas detalladas
    """
    
    def __init__(self, df: pd.DataFrame, other_params: List[str]):
        self.df = df
        self.other_params = other_params
        self.reducer = NoiseReducer()
        self.stats = StatsAnalyzer()
    
    def find_metric_column(self, metric_key: str) -> Optional[str]:
        """Encuentra la columna correspondiente a una métrica."""
        # Mapeo de nombres alternativos
        aliases = {
            'ROI': ['ROI', 'ROI_PCT', 'RETURN', 'NET_RETURN'],
            'SQN': ['SQN', 'SYSTEM_QUALITY_NUMBER'],
            'PROFIT_FACTOR': ['PROFIT_FACTOR', 'PF', 'PAYOFF_RATIO'],
            'DRAWDOWN': ['DRAWDOWN', 'MAX_DD', 'DD', 'MDD', 'MAX_DRAWDOWN', 'DD_PCT'],
        }
        
        for alias in aliases.get(metric_key, [metric_key]):
            if alias in self.df.columns:
                return alias
            # Buscar case-insensitive
            for col in self.df.columns:
                if col.upper() == alias.upper():
                    return col
        
        return None
    
    def create_parameter_page(self, param: str, ax_array: np.ndarray, 
                               use_noise_reduction: bool = True):
        """
        Crea la visualización completa para un parámetro.
        
        Args:
            param: Nombre del parámetro
            ax_array: Array 2x2 de axes para los 4 gráficos
            use_noise_reduction: Si usar ML para reducir ruido
        """
        metrics_to_plot = ['ROI', 'SQN', 'PROFIT_FACTOR', 'DRAWDOWN']
        
        for idx, metric_key in enumerate(metrics_to_plot):
            row, col = idx // 2, idx % 2
            ax = ax_array[row, col]
            
            # Encontrar columna de métrica
            metric_col = self.find_metric_column(metric_key)
            
            if metric_col is None:
                ax.text(0.5, 0.5, f'{metric_key}\nNo encontrada', 
                       ha='center', va='center', fontsize=12,
                       color=COLORS['text_muted'], transform=ax.transAxes)
                ax.set_facecolor(COLORS['bg_panel'])
                continue
            
            style = METRIC_STYLE[metric_key]
            
            # Obtener datos con reducción de ruido
            if use_noise_reduction:
                x, y, noise_stats = self.reducer.isolate_effect(
                    self.df, param, metric_col, self.other_params, method='residual'
                )
            else:
                x = pd.to_numeric(self.df[param], errors='coerce').values
                y = pd.to_numeric(self.df[metric_col], errors='coerce').values
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                noise_stats = {'method': 'raw', 'noise_reduction': 0}
            
            if len(x) < 5:
                ax.text(0.5, 0.5, f'{metric_key}\nDatos insuficientes', 
                       ha='center', va='center', fontsize=12,
                       color=COLORS['text_muted'], transform=ax.transAxes)
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # DIBUJAR GRÁFICO
            # ═══════════════════════════════════════════════════════════════
            
            # 1. Scatter de puntos originales (semi-transparentes)
            ax.scatter(x, y, c=style['color'], alpha=0.15, s=15, 
                      edgecolors='none', rasterized=True)
            
            # 2. Curva suavizada
            x_smooth, y_smooth = self.reducer.smooth_curve(x, y, smoothing=0.4)
            ax.plot(x_smooth, y_smooth, color=style['color'], linewidth=2.5,
                   label='Tendencia', zorder=10)
            
            # 3. Banda de confianza (bootstrap simplificado)
            # Calcular percentiles por bins
            n_bins = min(15, len(np.unique(x)) // 2)
            if n_bins >= 3:
                try:
                    bins = np.linspace(x.min(), x.max(), n_bins + 1)
                    bin_centers = []
                    y_lower = []
                    y_upper = []
                    
                    for i in range(len(bins) - 1):
                        mask = (x >= bins[i]) & (x < bins[i+1])
                        if mask.sum() >= 3:
                            y_bin = y[mask]
                            bin_centers.append((bins[i] + bins[i+1]) / 2)
                            y_lower.append(np.percentile(y_bin, 20))
                            y_upper.append(np.percentile(y_bin, 80))
                    
                    if len(bin_centers) >= 3:
                        # Suavizar bandas
                        bc = np.array(bin_centers)
                        yl = gaussian_filter1d(np.array(y_lower), sigma=1)
                        yu = gaussian_filter1d(np.array(y_upper), sigma=1)
                        
                        ax.fill_between(bc, yl, yu, color=style['color'], 
                                        alpha=0.15, zorder=5)
                except:
                    pass
            
            # 4. Zona óptima
            opt = self.stats.optimal_zone(x, y, higher_better=style['higher_better'])
            if not np.isnan(opt['optimal_min']):
                ax.axvspan(opt['optimal_min'], opt['optimal_max'], 
                          color=COLORS['green'] if style['higher_better'] else COLORS['red'],
                          alpha=0.1, zorder=1)
                ax.axvline(opt['optimal_mean'], color=COLORS['gold'], 
                          linestyle='--', linewidth=1.5, alpha=0.8,
                          label=f"Óptimo: {opt['optimal_mean']:.3g}")
            
            # 5. Estadísticas
            corr = self.stats.correlation_analysis(x, y)
            sens = self.stats.sensitivity_score(x, y)
            
            # Título con información
            title = f"{style['name']}"
            ax.set_title(title, fontsize=11, fontweight='bold', 
                        color=style['color'], pad=8)
            
            # Texto de estadísticas
            stats_text = (
                f"ρ = {corr['spearman']:+.2f} ({corr['strength']})\n"
                f"Sens: {sens:.2f}"
            )
            if noise_stats['noise_reduction'] > 0:
                stats_text += f"\n🧹 -{noise_stats['noise_reduction']*100:.0f}% ruido"
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_card'],
                            edgecolor=COLORS['border'], alpha=0.9),
                   color=COLORS['text_gray'])
            
            # Zona óptima en texto
            if not np.isnan(opt['optimal_min']):
                opt_text = f"Zona óptima: [{opt['optimal_min']:.3g} - {opt['optimal_max']:.3g}]"
                ax.text(0.02, 0.02, opt_text, transform=ax.transAxes,
                       fontsize=7, va='bottom', ha='left',
                       color=COLORS['gold'], alpha=0.9)
            
            # Ejes
            ax.set_xlabel(param, fontsize=9, color=COLORS['text_gray'])
            ax.set_ylabel(style['name'], fontsize=9, color=COLORS['text_gray'])
            
            # Grid
            ax.grid(True, alpha=0.2, color=COLORS['grid'])
            ax.set_facecolor(COLORS['bg_panel'])
            
            # Leyenda compacta
            ax.legend(loc='upper left', fontsize=7, framealpha=0.8)


# ==============================================================================
# 📄 GENERADOR DE REPORTE PDF
# ==============================================================================

class ReportGenerator:
    """Genera el reporte PDF completo."""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.df = loader.df
        self.params = loader.columns.parameters
        self.strategy = loader.strategy_name
    
    def generate(self, output_path: str):
        """Genera el reporte PDF."""
        print(f"\n{'═'*70}")
        print("📊 GENERANDO REPORTE INSTITUCIONAL")
        print('═'*70)
        
        with PdfPages(output_path) as pdf:
            # 1. Portada
            self._create_cover(pdf)
            
            # 2. Resumen ejecutivo
            self._create_summary(pdf)
            
            # 3. Una página por cada parámetro
            visualizer = ParameterVisualizer(self.df, self.params)
            
            for i, param in enumerate(self.params):
                print(f"   📈 Analizando: {param} ({i+1}/{len(self.params)})")
                self._create_parameter_page(pdf, param, visualizer)
            
            # 4. Matriz de correlaciones
            self._create_correlation_matrix(pdf)
            
            # 5. Ranking de importancia
            self._create_importance_ranking(pdf)
        
        print(f"\n   ✅ Reporte guardado: {output_path}")
        print('═'*70)
    
    def _create_cover(self, pdf: PdfPages):
        """Crea la portada."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(COLORS['bg_dark'])
        
        # Título principal
        fig.text(0.5, 0.65, 'MODELOX', fontsize=48, ha='center', va='center',
                fontweight='bold', color=COLORS['blue'])
        fig.text(0.5, 0.55, 'PARAMETER ANALYSIS REPORT', fontsize=24, ha='center',
                color=COLORS['text_white'])
        
        # Línea decorativa
        ax = fig.add_axes([0.2, 0.50, 0.6, 0.002])
        ax.set_facecolor(COLORS['blue'])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Info
        fig.text(0.5, 0.40, f'Estrategia: {self.strategy}', fontsize=16, ha='center',
                color=COLORS['text_gray'])
        fig.text(0.5, 0.35, f'Trials Analizados: {len(self.df):,}', fontsize=14, ha='center',
                color=COLORS['text_gray'])
        fig.text(0.5, 0.30, f'Parámetros: {len(self.params)}', fontsize=14, ha='center',
                color=COLORS['text_gray'])
        
        # Fecha
        fig.text(0.5, 0.15, datetime.now().strftime('%Y-%m-%d %H:%M'), 
                fontsize=12, ha='center', color=COLORS['text_muted'])
        
        # Métricas analizadas
        metrics_text = "Métricas: ROI • SQN • Profit Factor • Drawdown"
        fig.text(0.5, 0.22, metrics_text, fontsize=11, ha='center',
                color=COLORS['gold'])
        
        pdf.savefig(fig, facecolor=COLORS['bg_dark'])
        plt.close(fig)
    
    def _create_summary(self, pdf: PdfPages):
        """Crea página de resumen ejecutivo."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(COLORS['bg_dark'])
        
        # Título
        fig.text(0.5, 0.95, 'RESUMEN EJECUTIVO', fontsize=18, ha='center',
                fontweight='bold', color=COLORS['text_white'])
        
        # Calcular estadísticas globales
        stats_text = []
        
        for metric_key in ['ROI', 'SQN', 'PROFIT_FACTOR', 'DRAWDOWN']:
            col = None
            for alias in [metric_key, metric_key.lower(), metric_key.replace('_', '')]:
                if alias in self.df.columns:
                    col = alias
                    break
                for c in self.df.columns:
                    if c.upper() == alias.upper():
                        col = c
                        break
            
            if col:
                vals = pd.to_numeric(self.df[col], errors='coerce').dropna()
                style = METRIC_STYLE[metric_key]
                stats_text.append(
                    f"{style['name']:20} | Media: {vals.mean():>10.3f} | "
                    f"Std: {vals.std():>8.3f} | Best: {vals.max() if style['higher_better'] else vals.min():>10.3f}"
                )
        
        # Mostrar estadísticas
        y_pos = 0.85
        fig.text(0.1, y_pos, "📊 ESTADÍSTICAS DE MÉTRICAS", fontsize=12, 
                fontweight='bold', color=COLORS['cyan'])
        y_pos -= 0.03
        
        for line in stats_text:
            y_pos -= 0.025
            fig.text(0.1, y_pos, line, fontsize=10, family='monospace',
                    color=COLORS['text_gray'])
        
        # Parámetros con mayor impacto
        y_pos -= 0.06
        fig.text(0.1, y_pos, "⚙️ PARÁMETROS DETECTADOS", fontsize=12,
                fontweight='bold', color=COLORS['cyan'])
        
        for param in self.params[:10]:
            y_pos -= 0.025
            try:
                vals = pd.to_numeric(self.df[param], errors='coerce').dropna()
                fig.text(0.1, y_pos, f"  • {param:<30} [{vals.min():.4g} → {vals.max():.4g}]",
                        fontsize=9, color=COLORS['text_gray'], family='monospace')
            except:
                fig.text(0.1, y_pos, f"  • {param}", fontsize=9, color=COLORS['text_gray'])
        
        if len(self.params) > 10:
            y_pos -= 0.025
            fig.text(0.1, y_pos, f"  ... y {len(self.params) - 10} más",
                    fontsize=9, color=COLORS['text_muted'])
        
        # Nota metodológica
        y_pos = 0.15
        fig.text(0.1, y_pos, "📋 METODOLOGÍA", fontsize=12,
                fontweight='bold', color=COLORS['cyan'])
        y_pos -= 0.03
        methodology = [
            "• Reducción de ruido: Gradient Boosting para aislar efecto de cada parámetro",
            "• Curvas suavizadas: Spline + Gaussian smoothing (σ adaptativo)",
            "• Zona óptima: Top 20% de trials con mejor rendimiento",
            "• Correlaciones: Spearman (robusta a outliers) + test de significancia",
        ]
        for line in methodology:
            y_pos -= 0.022
            fig.text(0.1, y_pos, line, fontsize=9, color=COLORS['text_gray'])
        
        pdf.savefig(fig, facecolor=COLORS['bg_dark'])
        plt.close(fig)
    
    def _create_parameter_page(self, pdf: PdfPages, param: str, 
                                visualizer: ParameterVisualizer):
        """Crea página completa para un parámetro."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(COLORS['bg_dark'])
        
        # Título del parámetro
        fig.text(0.5, 0.97, f'ANÁLISIS: {param}', fontsize=16, ha='center',
                fontweight='bold', color=COLORS['blue'])
        
        # Rango del parámetro
        try:
            vals = pd.to_numeric(self.df[param], errors='coerce').dropna()
            range_text = f"Rango: [{vals.min():.4g} → {vals.max():.4g}] | N={len(vals):,} | Únicos={vals.nunique()}"
            fig.text(0.5, 0.94, range_text, fontsize=10, ha='center',
                    color=COLORS['text_gray'])
        except:
            pass
        
        # Grid 2x2 para las 4 métricas
        gs = gridspec.GridSpec(2, 2, figure=fig, 
                               left=0.08, right=0.95, top=0.90, bottom=0.08,
                               wspace=0.25, hspace=0.30)
        
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
        
        # Crear visualización
        visualizer.create_parameter_page(param, axes, use_noise_reduction=True)
        
        pdf.savefig(fig, facecolor=COLORS['bg_dark'])
        plt.close(fig)
    
    def _create_correlation_matrix(self, pdf: PdfPages):
        """Crea matriz de correlaciones parámetros vs métricas."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(COLORS['bg_dark'])
        
        fig.text(0.5, 0.97, 'MATRIZ DE CORRELACIONES', fontsize=16, ha='center',
                fontweight='bold', color=COLORS['text_white'])
        fig.text(0.5, 0.94, 'Parámetros vs Métricas (Spearman)', fontsize=11, ha='center',
                color=COLORS['text_gray'])
        
        # Encontrar métricas disponibles
        metrics = []
        metric_cols = []
        for mk in ['ROI', 'SQN', 'PROFIT_FACTOR', 'DRAWDOWN']:
            for alias in [mk, mk.lower()]:
                if alias in self.df.columns:
                    metrics.append(mk)
                    metric_cols.append(alias)
                    break
                for c in self.df.columns:
                    if c.upper() == alias.upper():
                        metrics.append(mk)
                        metric_cols.append(c)
                        break
        
        if len(metrics) == 0 or len(self.params) == 0:
            fig.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center',
                    fontsize=14, color=COLORS['text_muted'])
            pdf.savefig(fig, facecolor=COLORS['bg_dark'])
            plt.close(fig)
            return
        
        # Calcular correlaciones
        n_params = min(15, len(self.params))  # Limitar a 15 para legibilidad
        corr_matrix = np.zeros((n_params, len(metrics)))
        
        for i, param in enumerate(self.params[:n_params]):
            for j, metric_col in enumerate(metric_cols):
                try:
                    x = pd.to_numeric(self.df[param], errors='coerce')
                    y = pd.to_numeric(self.df[metric_col], errors='coerce')
                    mask = ~(x.isna() | y.isna())
                    if mask.sum() > 10:
                        corr, _ = stats.spearmanr(x[mask], y[mask])
                        corr_matrix[i, j] = corr
                except:
                    pass
        
        # Crear heatmap
        ax = fig.add_axes([0.25, 0.15, 0.65, 0.70])
        
        cmap = LinearSegmentedColormap.from_list(
            'corr', [COLORS['red'], COLORS['bg_panel'], COLORS['green']]
        )
        
        im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([METRIC_STYLE[m]['name'] for m in metrics], fontsize=10)
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(self.params[:n_params], fontsize=8)
        
        # Valores en celdas
        for i in range(n_params):
            for j in range(len(metrics)):
                val = corr_matrix[i, j]
                color = COLORS['text_white'] if abs(val) > 0.3 else COLORS['text_muted']
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=8, color=color, fontweight='bold' if abs(val) > 0.5 else 'normal')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlación Spearman', color=COLORS['text_gray'])
        
        ax.set_facecolor(COLORS['bg_panel'])
        
        pdf.savefig(fig, facecolor=COLORS['bg_dark'])
        plt.close(fig)
    
    def _create_importance_ranking(self, pdf: PdfPages):
        """Crea ranking de importancia de parámetros usando Random Forest."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(COLORS['bg_dark'])
        
        fig.text(0.5, 0.97, 'RANKING DE IMPORTANCIA', fontsize=16, ha='center',
                fontweight='bold', color=COLORS['text_white'])
        fig.text(0.5, 0.94, 'Feature Importance (Random Forest) para cada métrica', 
                fontsize=11, ha='center', color=COLORS['text_gray'])
        
        # Preparar datos
        params_to_use = self.params[:20]  # Max 20 parámetros
        
        X = self.df[params_to_use].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median())
        
        if len(X) < 30:
            fig.text(0.5, 0.5, 'Datos insuficientes para análisis ML', 
                    ha='center', va='center', fontsize=14, color=COLORS['text_muted'])
            pdf.savefig(fig, facecolor=COLORS['bg_dark'])
            plt.close(fig)
            return
        
        # Calcular importancia para cada métrica
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               left=0.12, right=0.95, top=0.88, bottom=0.08,
                               wspace=0.30, hspace=0.35)
        
        for idx, metric_key in enumerate(['ROI', 'SQN', 'PROFIT_FACTOR', 'DRAWDOWN']):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Encontrar columna de métrica
            metric_col = None
            for alias in [metric_key, metric_key.lower()]:
                if alias in self.df.columns:
                    metric_col = alias
                    break
                for c in self.df.columns:
                    if c.upper() == alias.upper():
                        metric_col = c
                        break
            
            if metric_col is None:
                ax.text(0.5, 0.5, f'{metric_key}\nNo encontrada', ha='center', va='center',
                       color=COLORS['text_muted'])
                continue
            
            y = pd.to_numeric(self.df[metric_col], errors='coerce')
            
            # Alinear X e y
            mask = ~y.isna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(y_clean) < 30:
                ax.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center',
                       color=COLORS['text_muted'])
                continue
            
            # Random Forest
            try:
                rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                rf.fit(X_clean, y_clean)
                
                importance = rf.feature_importances_
                
                # Ordenar y mostrar top 10
                top_n = min(10, len(importance))
                sorted_idx = np.argsort(importance)[-top_n:]
                
                colors = [METRIC_STYLE[metric_key]['color']] * top_n
                
                ax.barh(range(top_n), importance[sorted_idx], color=colors, alpha=0.8)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([params_to_use[i] for i in sorted_idx], fontsize=7)
                ax.set_xlabel('Importancia', fontsize=9)
                ax.set_title(METRIC_STYLE[metric_key]['name'], fontsize=11,
                            color=METRIC_STYLE[metric_key]['color'], fontweight='bold')
                
                ax.set_facecolor(COLORS['bg_panel'])
                ax.grid(True, alpha=0.2, axis='x')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center',
                       color=COLORS['red'], fontsize=8)
        
        pdf.savefig(fig, facecolor=COLORS['bg_dark'])
        plt.close(fig)


# ==============================================================================
# 🚀 FUNCIÓN PRINCIPAL
# ==============================================================================

def find_data_files() -> List[str]:
    """Busca archivos CSV/Excel en el directorio de resultados."""
    files = []
    
    # Buscar en resultados/
    patterns = [
        'resultados/**/*.csv',
        'resultados/**/*.xlsx',
        'resultados/**/*.xls',
        '*.csv',
        '*.xlsx',
    ]
    
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    # Filtrar solo RESUMEN
    resumen_files = [f for f in files if 'RESUMEN' in f.upper()]
    
    if resumen_files:
        return resumen_files
    return files


def main():
    """Función principal."""
    print("\n" + "═"*70)
    print("   📊 MODELOX PARAMETER ANALYZER v9.0 - INSTITUTIONAL")
    print("   🧹 Análisis Individual con Reducción de Ruido ML")
    print("═"*70)
    
    # Obtener archivo
    file_path = None
    
    # 1. Argumento de línea de comandos
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"\n❌ Error: Archivo no encontrado: {file_path}")
            sys.exit(1)
    
    # 2. Buscar automáticamente
    else:
        print("\n🔍 Buscando archivos de datos...")
        files = find_data_files()
        
        if not files:
            print("\n❌ No se encontraron archivos CSV/Excel")
            print("   Uso: python analisis.py <archivo.csv>")
            print("   O coloca archivos RESUMEN en resultados/")
            sys.exit(1)
        
        print(f"\n📁 Archivos encontrados ({len(files)}):")
        for i, f in enumerate(files[:10]):
            print(f"   [{i+1}] {f}")
        
        if len(files) > 10:
            print(f"   ... y {len(files) - 10} más")
        
        # Seleccionar
        if len(files) == 1:
            file_path = files[0]
            print(f"\n✅ Seleccionado automáticamente: {file_path}")
        else:
            try:
                choice = input(f"\n🔢 Selecciona archivo (1-{min(10, len(files))}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    file_path = files[idx]
                else:
                    file_path = files[0]
            except:
                file_path = files[0]
                print(f"   Usando: {file_path}")
    
    # Cargar datos
    loader = DataLoader()
    if not loader.load(file_path):
        print("\n❌ Error al cargar datos")
        sys.exit(1)
    
    if not loader.columns.parameters:
        print("\n⚠️ No se detectaron parámetros analizables")
        sys.exit(1)
    
    # Generar reporte
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path) or '.'
    output_path = os.path.join(output_dir, f'ANALYSIS_{base_name}_{loader.strategy_name}.pdf')
    
    report = ReportGenerator(loader)
    report.generate(output_path)
    
    print(f"\n🎉 ¡Análisis completado!")
    print(f"   📄 Reporte: {output_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
================================================================================
📄 GENERADOR DE PDF PROFESIONAL V2 - MODELOX
================================================================================

Sistema avanzado de generación de reportes PDF con:
- Gráficas 3D de superficie topográfica SUAVIZADA
- Análisis estadístico y probabilístico global
- Métricas de robustez y estabilidad
- Conclusiones automáticas
- Diseño profesional de nivel institucional

================================================================================
"""

import numpy as np
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Matplotlib con estilo profesional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# Estadísticas
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator, griddata


# =============================================================================
# CONFIGURACIÓN DE ESTILO PROFESIONAL
# =============================================================================

COLORS = {
    'primary': '#1e3a5f',      # Azul oscuro institucional
    'secondary': '#2d5a87',    # Azul medio
    'accent': '#3b82f6',       # Azul brillante
    'success': '#059669',      # Verde esmeralda
    'warning': '#d97706',      # Ámbar
    'danger': '#dc2626',       # Rojo
    'text': '#111827',         # Texto principal
    'text_light': '#6b7280',   # Texto secundario
    'bg_light': '#f8fafc',     # Fondo claro
    'bg_card': '#ffffff',      # Fondo tarjetas
    'border': '#e5e7eb',       # Bordes
    'gold': '#fbbf24',         # Dorado para destacados
}

# Paleta suave para superficies 3D (menos ruidosa)
CMAP_SURFACE = LinearSegmentedColormap.from_list('surface_smooth', [
    '#0d47a1', '#1565c0', '#1976d2', '#1e88e5', '#42a5f5',
    '#64b5f6', '#90caf9', '#bbdefb', '#e3f2fd', '#e8f5e9',
    '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50'
], N=256)

CMAP_HEATMAP = LinearSegmentedColormap.from_list('heatmap_pro', [
    '#1e3a5f', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',
    '#fef3c7', '#fcd34d', '#f59e0b', '#ea580c', '#dc2626'
], N=256)

CMAP_DIVERGENT = LinearSegmentedColormap.from_list('divergent', [
    '#dc2626', '#f87171', '#fca5a5', '#ffffff', '#86efac', '#22c55e', '#059669'
], N=256)


def setup_style():
    """Configura estilo matplotlib limpio y profesional."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'grid.alpha': 0.2,
        'grid.linewidth': 0.3,
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': COLORS['border'],
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 180,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5,
        'lines.antialiased': True,
    })

setup_style()


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class GlobalModelStats:
    mean_r2: float
    std_r2: float
    min_r2: float
    max_r2: float
    overall_quality: str
    stability_score: float
    robustness_score: float
    confidence_level: float
    total_params: int
    total_samples: int
    effective_dimensionality: float


@dataclass 
class ParameterAnalysis:
    name: str
    importance: float
    sensitivity: float
    optimal_value: float
    optimal_ci_lower: float
    optimal_ci_upper: float
    monotonicity: float
    nonlinearity: float
    interaction_strength: float = 0.0


@dataclass
class ConclusionData:
    global_stats: GlobalModelStats
    param_analyses: List[ParameterAnalysis]
    metric_results: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_statements: List[str] = field(default_factory=list)


# =============================================================================
# ANALIZADOR ESTADÍSTICO
# =============================================================================

class GlobalStatisticalAnalyzer:
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str]):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
    
    def compute_global_stats(self) -> GlobalModelStats:
        r2_scores = [r.r2_score for r in self.gpr_results.values()]
        noise_levels = [r.noise_level for r in self.gpr_results.values()]
        
        if not r2_scores:
            return GlobalModelStats(0, 0, 0, 0, "N/A", 0, 0, 0, 0, 0, 0)
        
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores) if len(r2_scores) > 1 else 0
        
        if mean_r2 > 0.75 and std_r2 < 0.1:
            quality = "EXCELENTE"
        elif mean_r2 > 0.6:
            quality = "BUENO"
        elif mean_r2 > 0.4:
            quality = "MODERADO"
        else:
            quality = "BAJO"
        
        stability = max(0, 1.0 - std_r2 * 3)
        avg_noise = np.mean(noise_levels) if noise_levels else 0.1
        robustness = 1.0 / (1.0 + avg_noise * 5)
        confidence = min(mean_r2 * 0.5 + stability * 0.3 + robustness * 0.2, 0.99)
        
        # Dimensionalidad efectiva
        X = self.df[self.param_columns].to_numpy()
        X = X[~np.any(np.isnan(X), axis=1)]
        eff_dim = len(self.param_columns)
        if len(X) > 10:
            try:
                X_c = X - X.mean(axis=0)
                _, s, _ = np.linalg.svd(X_c, full_matrices=False)
                var_exp = (s ** 2) / (s ** 2).sum()
                eff_dim = np.sum(np.cumsum(var_exp) < 0.95) + 1
            except:
                pass
        
        return GlobalModelStats(
            mean_r2=mean_r2, std_r2=std_r2, min_r2=min(r2_scores), max_r2=max(r2_scores),
            overall_quality=quality, stability_score=stability, robustness_score=robustness,
            confidence_level=confidence, total_params=len(self.param_columns),
            total_samples=len(self.df), effective_dimensionality=eff_dim
        )
    
    def analyze_parameters(self) -> List[ParameterAnalysis]:
        analyses = []
        
        for param in self.param_columns:
            importances, optimals, cis_l, cis_u, monos, nonlins = [], [], [], [], [], []
            
            for metric, result in self.gpr_results.items():
                if param in result.predictions:
                    pred = result.predictions[param]
                    y = pred.mean_prediction
                    x = pred.param_values
                    
                    # Importancia: rango normalizado
                    pred_range = np.ptp(y)
                    importances.append(pred_range)
                    
                    # Óptimo
                    idx = np.argmax(y)
                    optimals.append(x[idx])
                    cis_l.append(pred.lower_ci[idx])
                    cis_u.append(pred.upper_ci[idx])
                    
                    # Monotonicidad
                    if len(y) > 1:
                        diffs = np.diff(y)
                        mono = np.mean(np.sign(diffs))
                        monos.append(mono)
                    
                    # No linealidad
                    if len(x) > 2:
                        try:
                            p = np.polyfit(x, y, 1)
                            linear = np.polyval(p, x)
                            nonlin = np.std(y - linear) / (np.std(y) + 1e-8)
                            nonlins.append(min(nonlin, 1.0))
                        except:
                            nonlins.append(0)
            
            if importances:
                analyses.append(ParameterAnalysis(
                    name=param,
                    importance=np.mean(importances),
                    sensitivity=np.std(importances) if len(importances) > 1 else 0,
                    optimal_value=np.mean(optimals),
                    optimal_ci_lower=np.mean(cis_l),
                    optimal_ci_upper=np.mean(cis_u),
                    monotonicity=np.mean(monos) if monos else 0,
                    nonlinearity=np.mean(nonlins) if nonlins else 0
                ))
        
        # Normalizar importancia
        total = sum(a.importance for a in analyses)
        if total > 0:
            for a in analyses:
                a.importance /= total
        
        analyses.sort(key=lambda x: x.importance, reverse=True)
        return analyses
    
    def generate_conclusions(self) -> ConclusionData:
        gs = self.compute_global_stats()
        pa = self.analyze_parameters()
        
        recs, warns, confs = [], [], []
        
        # Conclusiones R²
        if gs.mean_r2 > 0.75:
            confs.append(f"Modelo con ajuste EXCELENTE (R²={gs.mean_r2:.3f}). Predicciones altamente confiables.")
        elif gs.mean_r2 > 0.5:
            confs.append(f"Modelo con ajuste ACEPTABLE (R²={gs.mean_r2:.3f}). Usar predicciones con precaución.")
        else:
            warns.append(f"R² bajo ({gs.mean_r2:.3f}): capacidad predictiva limitada.")
        
        # Estabilidad
        if gs.stability_score > 0.7:
            confs.append("Alta consistencia entre métricas analizadas.")
        elif gs.stability_score < 0.4:
            warns.append("Variabilidad alta entre métricas. Resultados pueden depender de la métrica elegida.")
        
        # Recomendaciones de parámetros
        if pa:
            top = pa[0]
            recs.append(
                f"Parámetro más importante: {top.name.replace('param_', '')} "
                f"({top.importance*100:.0f}%). Óptimo: {top.optimal_value:.2f} "
                f"[{top.optimal_ci_lower:.2f}, {top.optimal_ci_upper:.2f}]"
            )
            
            low_imp = [p for p in pa if p.importance < 0.05]
            if low_imp:
                recs.append(f"Parámetros poco influyentes ({', '.join(p.name.replace('param_','') for p in low_imp[:2])}) pueden simplificarse.")
            
            nonlin = [p for p in pa if p.nonlinearity > 0.6]
            if nonlin:
                warns.append(f"Alta no linealidad en {', '.join(p.name.replace('param_','') for p in nonlin[:2])}. Considerar grid más fino.")
        
        return ConclusionData(gs, pa, self.gpr_results, recs, warns, confs)


# =============================================================================
# GENERADOR DE FIGURAS
# =============================================================================

class ProfessionalFigureGenerator:
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str]):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
    
    def _to_b64(self, fig, dpi=180) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.02)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
    
    def _clean_name(self, name: str) -> str:
        return name.replace('param_', '').replace('_', ' ').title()[:18]
    
    def generate_surface_3d(self, metric: str, p1_idx: int = 0, p2_idx: int = 1) -> str:
        """Genera superficie 3D + Zona Óptima Dorada + Campo de Gradientes."""
        if metric not in self.gpr_models or len(self.param_columns) < 2:
            return ""
        
        gpr = self.gpr_models[metric]
        X = self.df[self.param_columns].to_numpy().astype(np.float64)
        valid = ~np.any(np.isnan(X), axis=1)
        X = X[valid]
        
        if len(X) < 20:
            return ""
        
        # Grid más denso para suavidad
        n_grid = 60
        p1_range = np.linspace(np.percentile(X[:, p1_idx], 2), np.percentile(X[:, p1_idx], 98), n_grid)
        p2_range = np.linspace(np.percentile(X[:, p2_idx], 2), np.percentile(X[:, p2_idx], 98), n_grid)
        P1, P2 = np.meshgrid(p1_range, p2_range)
        
        # Grid de predicción con valores medios para otros parámetros
        X_grid = np.column_stack([
            P1.ravel() if i == p1_idx else P2.ravel() if i == p2_idx else np.full(P1.size, np.median(X[:, i]))
            for i in range(len(self.param_columns))
        ])
        
        try:
            y_pred, y_std = gpr.predict_batch(X_grid)
            Z = y_pred.reshape(P1.shape)
            Z_std = y_std.reshape(P1.shape)
            
            # SUAVIZADO GAUSSIANO para eliminar ruido
            Z = gaussian_filter(Z, sigma=2.0)
            Z_std = gaussian_filter(Z_std, sigma=2.0)
        except Exception as e:
            return ""
        
        # Crear figura con 3 subplots
        fig = plt.figure(figsize=(16, 5.5))
        
        p1_name = self._clean_name(self.param_columns[p1_idx])
        p2_name = self._clean_name(self.param_columns[p2_idx])
        
        # Encontrar óptimo
        opt_idx = np.unravel_index(np.argmax(Z), Z.shape)
        opt_x, opt_y, opt_z = P1[opt_idx], P2[opt_idx], Z[opt_idx]
        
        # ══════════════════════════════════════════════════════════════════════
        # SUBPLOT 1: SUPERFICIE 3D ELEGANTE
        # ══════════════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_facecolor('#fafbfc')
        
        # Paleta elegante azul-dorado
        cmap_elegant = LinearSegmentedColormap.from_list('elegant', [
            '#1a237e', '#283593', '#3949ab', '#5c6bc0', '#7986cb',
            '#9fa8da', '#c5cae9', '#e8eaf6', '#fff8e1', '#ffecb3',
            '#ffe082', '#ffd54f', '#ffca28', '#ffc107', '#ffb300'
        ], N=256)
        
        norm = Normalize(vmin=Z.min(), vmax=Z.max())
        
        # Superficie principal con mejor iluminación
        surf = ax1.plot_surface(P1, P2, Z, cmap=cmap_elegant, norm=norm,
                               edgecolor='none', alpha=0.92, antialiased=True,
                               rcount=60, ccount=60, shade=True)
        
        # Proyección de contornos en el suelo
        z_floor = Z.min() - (Z.max() - Z.min()) * 0.1
        ax1.contour(P1, P2, Z, zdir='z', offset=z_floor, levels=12, 
                   cmap=cmap_elegant, alpha=0.4, linewidths=0.8)
        
        # Marcar punto óptimo en 3D
        ax1.scatter([opt_x], [opt_y], [opt_z], c='#ff6f00', s=300, marker='*',
                   edgecolors='white', linewidths=2, zorder=10, depthshade=False)
        
        # Línea vertical al óptimo
        ax1.plot([opt_x, opt_x], [opt_y, opt_y], [z_floor, opt_z], 
                color='#ff6f00', linestyle='--', alpha=0.6, linewidth=1.5)
        
        ax1.set_xlabel(p1_name, fontsize=10, labelpad=8, fontweight='bold')
        ax1.set_ylabel(p2_name, fontsize=10, labelpad=8, fontweight='bold')
        ax1.set_zlabel(metric.upper(), fontsize=10, labelpad=8, fontweight='bold')
        ax1.set_title('Superficie de Respuesta', fontsize=12, fontweight='bold', 
                     pad=15, color=COLORS['primary'])
        ax1.view_init(elev=28, azim=135)
        
        # Panes transparentes
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('#e0e0e0')
        ax1.yaxis.pane.set_edgecolor('#e0e0e0')
        ax1.zaxis.pane.set_edgecolor('#e0e0e0')
        ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)
        
        # ══════════════════════════════════════════════════════════════════════
        # SUBPLOT 2: ZONA ÓPTIMA DORADA (Mapa de Rendimiento)
        # ══════════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(132)
        
        # Normalizar Z para percentiles de rendimiento
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        
        # Crear zonas de rendimiento con colores distintivos
        cmap_zones = LinearSegmentedColormap.from_list('zones', [
            '#b71c1c', '#c62828', '#d32f2f', '#e53935',  # Zona roja (bajo)
            '#ff5722', '#ff7043', '#ff8a65',              # Naranja
            '#ffb74d', '#ffc107', '#ffca28',              # Amarillo
            '#c8e6c9', '#a5d6a7', '#81c784',              # Verde claro
            '#4caf50', '#43a047', '#388e3c', '#2e7d32'   # Verde intenso (alto)
        ], N=256)
        
        # Heatmap base
        im = ax2.imshow(Z_norm, extent=[p1_range.min(), p1_range.max(), 
                                        p2_range.min(), p2_range.max()],
                       origin='lower', cmap=cmap_zones, aspect='auto', alpha=0.9)
        
        # Contornos de percentiles clave
        percentiles = [50, 75, 90, 95]
        colors_p = ['#ffffff', '#ffd700', '#ff8c00', '#ff4500']
        labels_p = ['50%', '75%', '90%', '95%']
        
        for pct, col, lab in zip(percentiles, colors_p, labels_p):
            threshold = np.percentile(Z, pct)
            cs = ax2.contour(P1, P2, Z, levels=[threshold], colors=[col], 
                           linewidths=2.5 if pct >= 90 else 1.5, alpha=0.9)
        
        # ZONA DORADA: Top 5% con efecto especial
        top_mask = Z >= np.percentile(Z, 95)
        ax2.contourf(P1, P2, top_mask.astype(float), levels=[0.5, 1.5],
                    colors=['#ffd700'], alpha=0.4)
        ax2.contour(P1, P2, top_mask.astype(float), levels=[0.5],
                   colors=['#ff6f00'], linewidths=3, linestyles='-')
        
        # Estrella en el óptimo
        ax2.scatter(opt_x, opt_y, c='#ff6f00', s=400, marker='*',
                   edgecolors='white', linewidths=3, zorder=15)
        ax2.scatter(opt_x, opt_y, c='#ffd700', s=150, marker='*',
                   edgecolors='none', zorder=16, alpha=0.8)
        
        # Anotación del óptimo
        ax2.annotate(f'ÓPTIMO\n{opt_z:.1f}', (opt_x, opt_y),
                    xytext=(25, 25), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#ff6f00',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#ff6f00', 
                             alpha=0.95, linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='#ff6f00', lw=2))
        
        ax2.set_xlabel(p1_name, fontsize=10, fontweight='bold')
        ax2.set_ylabel(p2_name, fontsize=10, fontweight='bold')
        ax2.set_title('Mapa de Rendimiento', fontsize=12, fontweight='bold', 
                     color=COLORS['primary'])
        
        # Colorbar con etiquetas
        cbar = plt.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
        cbar.set_label('Rendimiento Relativo', fontsize=9, fontweight='bold')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Bajo', '25%', '50%', '75%', 'Alto'])
        
        # Leyenda de zonas
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ffd700', edgecolor='#ff6f00', linewidth=2, 
                  label='Zona Óptima (Top 5%)', alpha=0.6),
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.95, edgecolor=COLORS['border'])
        
        # ══════════════════════════════════════════════════════════════════════
        # SUBPLOT 3: CAMPO DE GRADIENTES (Dirección al Óptimo)
        # ══════════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(133)
        
        # Calcular gradientes
        grad_y, grad_x = np.gradient(Z)
        
        # Magnitud del gradiente
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag_smooth = gaussian_filter(grad_mag, sigma=1.5)
        
        # Fondo: Magnitud del gradiente (sensibilidad)
        im3 = ax3.imshow(grad_mag_smooth, extent=[p1_range.min(), p1_range.max(),
                                                   p2_range.min(), p2_range.max()],
                        origin='lower', cmap='YlOrRd', aspect='auto', alpha=0.7)
        
        # Reducir densidad de flechas para claridad
        skip = 4
        P1_s, P2_s = P1[::skip, ::skip], P2[::skip, ::skip]
        grad_x_s, grad_y_s = grad_x[::skip, ::skip], grad_y[::skip, ::skip]
        
        # Normalizar flechas
        mag_s = np.sqrt(grad_x_s**2 + grad_y_s**2) + 1e-8
        grad_x_n = grad_x_s / mag_s
        grad_y_n = grad_y_s / mag_s
        
        # Colores según magnitud
        colors_q = mag_s.flatten()
        colors_q = (colors_q - colors_q.min()) / (colors_q.max() - colors_q.min() + 1e-8)
        
        # Campo de flechas
        quiver = ax3.quiver(P1_s, P2_s, grad_x_n, grad_y_n, colors_q,
                           cmap='coolwarm', scale=25, width=0.004, 
                           headwidth=4, headlength=5, alpha=0.85)
        
        # Contornos de Z como referencia
        ax3.contour(P1, P2, Z, levels=10, colors='white', linewidths=0.5, alpha=0.4)
        
        # Marcar óptimo (punto de equilibrio - gradiente cero)
        ax3.scatter(opt_x, opt_y, c='#4caf50', s=350, marker='o',
                   edgecolors='white', linewidths=3, zorder=15)
        ax3.scatter(opt_x, opt_y, c='white', s=80, marker='o', zorder=16)
        
        # Círculos concéntricos desde el óptimo
        for r in [0.1, 0.2, 0.3]:
            r_scaled = r * (p1_range.max() - p1_range.min())
            circle = plt.Circle((opt_x, opt_y), r_scaled, fill=False,
                               color='#4caf50', linewidth=1.5, alpha=0.5, linestyle='--')
            ax3.add_patch(circle)
        
        ax3.annotate('EQUILIBRIO\n(Gradiente ≈ 0)', (opt_x, opt_y),
                    xytext=(-35, -45), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#2e7d32',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#4caf50', 
                             alpha=0.95, linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='#4caf50', lw=2))
        
        ax3.set_xlabel(p1_name, fontsize=10, fontweight='bold')
        ax3.set_ylabel(p2_name, fontsize=10, fontweight='bold')
        ax3.set_title('Campo de Gradientes', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax3.set_xlim(p1_range.min(), p1_range.max())
        ax3.set_ylim(p2_range.min(), p2_range.max())
        
        # Colorbar para sensibilidad
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.85, pad=0.02)
        cbar3.set_label('Sensibilidad |∇f|', fontsize=9, fontweight='bold')
        
        # Leyenda
        ax3.text(0.02, 0.98, '→ Dirección de mejora\n● Punto de equilibrio',
                transform=ax3.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['border']))
        
        # ══════════════════════════════════════════════════════════════════════
        # TÍTULO GENERAL
        # ══════════════════════════════════════════════════════════════════════
        fig.suptitle(f'{metric.upper()} — Análisis Tridimensional de Optimización', 
                    fontsize=14, fontweight='bold', color=COLORS['primary'], y=1.0)
        
        plt.tight_layout()
        return self._to_b64(fig, dpi=200)
    
    def generate_quality_dashboard(self, global_stats: GlobalModelStats) -> str:
        """Dashboard de calidad con gauges profesionales."""
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)
        
        def draw_gauge(ax, value, title, thresholds=None):
            """Gauge semicircular profesional."""
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Fondo del gauge
            theta = np.linspace(0, np.pi, 100)
            r_outer, r_inner = 0.9, 0.6
            
            # Dibujar arco de fondo
            for i, (t1, t2) in enumerate(zip(theta[:-1], theta[1:])):
                val_at_t = 1 - (t1 / np.pi)
                if thresholds:
                    if val_at_t > 0.7:
                        c = COLORS['success']
                    elif val_at_t > 0.4:
                        c = COLORS['warning']
                    else:
                        c = COLORS['danger']
                else:
                    c = COLORS['border']
                
                wedge = Wedge((0.5, 0.1), r_outer, np.degrees(np.pi - t2), np.degrees(np.pi - t1),
                             width=r_outer-r_inner, facecolor=c, alpha=0.2, edgecolor='none')
                ax.add_patch(wedge)
            
            # Arco de valor
            value_theta = np.pi * (1 - value)
            for i, (t1, t2) in enumerate(zip(theta[:-1], theta[1:])):
                if t1 < value_theta:
                    continue
                val_at_t = 1 - (t1 / np.pi)
                if val_at_t > 0.7:
                    c = COLORS['success']
                elif val_at_t > 0.4:
                    c = COLORS['warning']
                else:
                    c = COLORS['danger']
                
                wedge = Wedge((0.5, 0.1), r_outer, np.degrees(np.pi - t2), np.degrees(np.pi - t1),
                             width=r_outer-r_inner, facecolor=c, alpha=0.9, edgecolor='white', linewidth=0.5)
                ax.add_patch(wedge)
            
            # Aguja
            needle_angle = np.pi * (1 - value)
            needle_len = 0.55
            ax.annotate('', xy=(0.5 + needle_len * np.cos(needle_angle), 0.1 + needle_len * np.sin(needle_angle)),
                       xytext=(0.5, 0.1),
                       arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2.5))
            
            # Centro
            circle = Circle((0.5, 0.1), 0.08, facecolor=COLORS['primary'], edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            
            # Valor
            ax.text(0.5, -0.15, f'{value:.2f}', ha='center', va='top', fontsize=16, 
                   fontweight='bold', color=COLORS['primary'])
            ax.text(0.5, 0.95, title, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.35, 1.1)
        
        # Gauges
        ax1 = fig.add_subplot(gs[0, 0])
        draw_gauge(ax1, min(global_stats.mean_r2, 1.0), 'R² Promedio', thresholds=True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        draw_gauge(ax2, global_stats.stability_score, 'Estabilidad', thresholds=True)
        
        ax3 = fig.add_subplot(gs[0, 2])
        draw_gauge(ax3, global_stats.robustness_score, 'Robustez', thresholds=True)
        
        ax4 = fig.add_subplot(gs[0, 3])
        draw_gauge(ax4, global_stats.confidence_level, 'Confianza', thresholds=True)
        
        # Gráfico de barras por métrica
        ax5 = fig.add_subplot(gs[1, :2])
        metrics = list(self.gpr_results.keys())
        r2s = [self.gpr_results[m].r2_score for m in metrics]
        colors = [COLORS['success'] if r > 0.7 else COLORS['warning'] if r > 0.4 else COLORS['danger'] for r in r2s]
        
        bars = ax5.bar(range(len(metrics)), r2s, color=colors, edgecolor='white', linewidth=1.5, width=0.7)
        ax5.set_xticks(range(len(metrics)))
        ax5.set_xticklabels([m.upper() for m in metrics], fontsize=9)
        ax5.set_ylabel('R² Score', fontsize=9)
        ax5.set_ylim(0, 1.05)
        ax5.set_title('Calidad por Métrica', fontsize=11, fontweight='bold')
        ax5.axhline(0.7, color=COLORS['success'], linestyle='--', alpha=0.5, linewidth=1)
        ax5.axhline(0.4, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, r2s):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.3f}',
                    ha='center', fontsize=9, fontweight='bold', color=COLORS['text'])
        
        # Panel de resumen
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        
        summary = f"""
┌─────────────────────────────────────┐
│     RESUMEN DEL MODELO              │
├─────────────────────────────────────┤
│  Calidad Global:  {global_stats.overall_quality:<17}│
│  R² Promedio:     {global_stats.mean_r2:.4f} ± {global_stats.std_r2:.4f}       │
│  R² Rango:        [{global_stats.min_r2:.3f} - {global_stats.max_r2:.3f}]       │
│                                     │
│  Estabilidad:     {global_stats.stability_score*100:>5.1f}%             │
│  Robustez:        {global_stats.robustness_score*100:>5.1f}%             │
│  Confianza:       {global_stats.confidence_level*100:>5.1f}%             │
│                                     │
│  Parámetros:      {global_stats.total_params:<17}│
│  Samples:         {global_stats.total_samples:<17,}│
│  Dim. Efectiva:   {global_stats.effective_dimensionality:<17.1f}│
└─────────────────────────────────────┘
"""
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
                fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_light'], 
                         edgecolor=COLORS['border'], linewidth=1))
        
        fig.suptitle('Dashboard de Calidad del Modelo', fontsize=14, fontweight='bold',
                    color=COLORS['primary'], y=0.98)
        
        return self._to_b64(fig)
    
    def generate_param_importance(self, param_analyses: List[ParameterAnalysis]) -> str:
        """Gráfico de importancia de parámetros mejorado."""
        if not param_analyses:
            return ""
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        names = [self._clean_name(p.name) for p in param_analyses]
        n = len(names)
        
        # 1. IMPORTANCIA (barras horizontales)
        ax1 = axes[0]
        importance = [p.importance * 100 for p in param_analyses]
        colors = [COLORS['primary'] if i < 3 else COLORS['text_light'] for i in range(n)]
        
        bars = ax1.barh(range(n), importance, color=colors, edgecolor='white', height=0.7)
        ax1.set_yticks(range(n))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('Importancia Relativa (%)', fontsize=9)
        ax1.set_title('Importancia de Parámetros', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_xlim(0, max(importance) * 1.15)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, importance):
            ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=8, fontweight='bold')
        
        # 2. SENSIBILIDAD vs NO LINEALIDAD
        ax2 = axes[1]
        sens = [p.sensitivity for p in param_analyses]
        nonlin = [p.nonlinearity for p in param_analyses]
        sizes = [100 + p.importance * 400 for p in param_analyses]
        
        scatter = ax2.scatter(sens, nonlin, c=importance, cmap=CMAP_HEATMAP,
                             s=sizes, alpha=0.8, edgecolors='white', linewidths=1.5)
        
        for i, name in enumerate(names):
            ax2.annotate(name, (sens[i], nonlin[i]), fontsize=7, alpha=0.9,
                        xytext=(4, 4), textcoords='offset points')
        
        ax2.axhline(0.5, color=COLORS['warning'], linestyle='--', alpha=0.4, linewidth=1)
        ax2.axvline(np.median(sens) if sens else 0, color=COLORS['border'], linestyle=':', alpha=0.5)
        ax2.set_xlabel('Sensibilidad', fontsize=9)
        ax2.set_ylabel('No Linealidad', fontsize=9)
        ax2.set_title('Sensibilidad vs No Linealidad', fontsize=11, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.02)
        cbar.set_label('Importancia (%)', fontsize=8)
        
        # 3. MONOTONICIDAD
        ax3 = axes[2]
        mono = [p.monotonicity for p in param_analyses]
        colors_mono = [COLORS['success'] if m > 0.3 else COLORS['danger'] if m < -0.3 else COLORS['text_light'] for m in mono]
        
        bars = ax3.barh(range(n), mono, color=colors_mono, edgecolor='white', height=0.7)
        ax3.set_yticks(range(n))
        ax3.set_yticklabels(names, fontsize=9)
        ax3.axvline(0, color=COLORS['text'], linewidth=1)
        ax3.axvline(0.3, color=COLORS['success'], linestyle='--', alpha=0.3)
        ax3.axvline(-0.3, color=COLORS['danger'], linestyle='--', alpha=0.3)
        ax3.set_xlabel('Monotonicidad (-1 a 1)', fontsize=9)
        ax3.set_title('Tendencia Monotónica', fontsize=11, fontweight='bold')
        ax3.set_xlim(-1.1, 1.1)
        ax3.invert_yaxis()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Leyenda
        ax3.text(0.95, 0.05, '→ Positiva: a mayor valor, mejor resultado\n← Negativa: a menor valor, mejor resultado',
                transform=ax3.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('Análisis de Parámetros', fontsize=13, fontweight='bold', 
                    color=COLORS['primary'], y=1.02)
        plt.tight_layout()
        
        return self._to_b64(fig)
    
    def generate_pd_grid(self, metric: str) -> str:
        """Grid de dependencia parcial con IC profesional."""
        if metric not in self.gpr_results:
            return ""
        
        result = self.gpr_results[metric]
        preds = result.predictions
        n_params = len(preds)
        
        if n_params == 0:
            return ""
        
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.5*n_rows))
        
        if n_params == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (param, pred) in enumerate(preds.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            x = pred.param_values
            y = pred.mean_prediction
            y_lo = pred.lower_ci
            y_hi = pred.upper_ci
            
            # Banda de confianza con gradiente
            ax.fill_between(x, y_lo, y_hi, alpha=0.15, color=COLORS['primary'], label='IC 95%')
            ax.fill_between(x, y - (y - y_lo)*0.5, y + (y_hi - y)*0.5, alpha=0.25, color=COLORS['primary'])
            
            # Línea principal
            ax.plot(x, y, color=COLORS['primary'], linewidth=2.5, label='Predicción')
            
            # Óptimo
            idx_opt = np.argmax(y)
            ax.scatter(x[idx_opt], y[idx_opt], c=COLORS['gold'], s=150, marker='*',
                      edgecolors='white', linewidths=2, zorder=10)
            ax.annotate(f'{x[idx_opt]:.2f}', (x[idx_opt], y[idx_opt]),
                       xytext=(0, 12), textcoords='offset points', ha='center',
                       fontsize=8, fontweight='bold', color=COLORS['success'])
            
            ax.set_xlabel(self._clean_name(param), fontsize=9)
            ax.set_ylabel(metric.upper(), fontsize=9)
            ax.set_title(f'{self._clean_name(param)}', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        
        # Ocultar ejes vacíos
        for idx in range(n_params, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(f'{metric.upper()} — Dependencia Parcial con IC 95%',
                    fontsize=13, fontweight='bold', color=COLORS['primary'], y=1.02)
        plt.tight_layout()
        
        return self._to_b64(fig)
    
    def generate_distribution_analysis(self, metric: str) -> str:
        """Análisis de distribución mejorado."""
        if metric not in self.gpr_results:
            return ""
        
        result = self.gpr_results[metric]
        
        all_means, all_stds = [], []
        for pred in result.predictions.values():
            all_means.extend(pred.mean_prediction)
            all_stds.extend(pred.std_prediction)
        
        if not all_means:
            return ""
        
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # 1. Histograma de predicciones
        ax1 = axes[0, 0]
        ax1.hist(all_means, bins=35, color=COLORS['primary'], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax1.axvline(np.mean(all_means), color=COLORS['danger'], linestyle='--', lw=2, label=f'Media: {np.mean(all_means):.2f}')
        ax1.axvline(np.median(all_means), color=COLORS['success'], linestyle=':', lw=2, label=f'Mediana: {np.median(all_means):.2f}')
        ax1.set_xlabel(f'{metric.upper()} Predicho', fontsize=9)
        ax1.set_ylabel('Frecuencia', fontsize=9)
        ax1.set_title('Distribución de Predicciones', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. Histograma de incertidumbre
        ax2 = axes[0, 1]
        ax2.hist(all_stds, bins=35, color=COLORS['warning'], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.axvline(np.mean(all_stds), color=COLORS['danger'], linestyle='--', lw=2, label=f'Media σ: {np.mean(all_stds):.3f}')
        ax2.set_xlabel('Desviación Estándar (σ)', fontsize=9)
        ax2.set_ylabel('Frecuencia', fontsize=9)
        ax2.set_title('Distribución de Incertidumbre', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Q-Q Plot
        ax3 = axes[0, 2]
        stats.probplot(all_means, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normalidad)', fontsize=10, fontweight='bold')
        ax3.get_lines()[0].set_color(COLORS['primary'])
        ax3.get_lines()[0].set_markersize(3)
        ax3.get_lines()[1].set_color(COLORS['danger'])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Predicción vs Incertidumbre
        ax4 = axes[1, 0]
        ax4.scatter(all_means, all_stds, alpha=0.4, c=COLORS['primary'], s=8, edgecolors='none')
        z = np.polyfit(all_means, all_stds, 1)
        x_line = np.linspace(all_means.min(), all_means.max(), 100)
        ax4.plot(x_line, np.polyval(z, x_line), color=COLORS['danger'], linestyle='--', lw=2,
                label=f'Tendencia (β={z[0]:.3f})')
        ax4.set_xlabel(f'{metric.upper()} Predicho', fontsize=9)
        ax4.set_ylabel('Incertidumbre (σ)', fontsize=9)
        ax4.set_title('Predicción vs Incertidumbre', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 5. Boxplot por parámetro
        ax5 = axes[1, 1]
        param_names = [self._clean_name(p)[:10] for p in result.predictions.keys()]
        param_data = [pred.mean_prediction for pred in result.predictions.values()]
        
        bp = ax5.boxplot(param_data, labels=param_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color(COLORS['danger'])
            median.set_linewidth(2)
        ax5.set_ylabel(f'{metric.upper()}', fontsize=9)
        ax5.set_title('Distribución por Parámetro', fontsize=10, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 6. CDF de intervalos de confianza
        ax6 = axes[1, 2]
        ci_widths = []
        for pred in result.predictions.values():
            ci_widths.extend(pred.upper_ci - pred.lower_ci)
        ci_widths = np.sort(ci_widths)
        cumulative = np.arange(1, len(ci_widths) + 1) / len(ci_widths)
        
        ax6.plot(ci_widths, cumulative, color=COLORS['primary'], linewidth=2.5)
        ax6.axhline(0.95, color=COLORS['success'], linestyle='--', alpha=0.6, label='95%')
        ax6.axhline(0.5, color=COLORS['warning'], linestyle='--', alpha=0.6, label='50%')
        ax6.fill_between(ci_widths, 0, cumulative, alpha=0.1, color=COLORS['primary'])
        ax6.set_xlabel('Ancho del IC 95%', fontsize=9)
        ax6.set_ylabel('Proporción Acumulada', fontsize=9)
        ax6.set_title('CDF de Intervalos de Confianza', fontsize=10, fontweight='bold')
        ax6.legend(fontsize=8, loc='lower right')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        fig.suptitle(f'{metric.upper()} — Análisis Estadístico Completo',
                    fontsize=13, fontweight='bold', color=COLORS['primary'], y=1.0)
        plt.tight_layout()
        
        return self._to_b64(fig)
    
    def generate_all(self) -> Dict[str, str]:
        """Genera todas las figuras."""
        figures = {}
        
        analyzer = GlobalStatisticalAnalyzer(self.gpr_results, self.gpr_models, self.df, self.param_columns)
        gs = analyzer.compute_global_stats()
        pa = analyzer.analyze_parameters()
        
        # Dashboard
        try:
            figures['quality_dashboard'] = self.generate_quality_dashboard(gs)
        except Exception as e:
            print(f"Error dashboard: {e}")
        
        # Importancia
        try:
            figures['param_importance'] = self.generate_param_importance(pa)
        except Exception as e:
            print(f"Error importance: {e}")
        
        # Por métrica
        for metric in self.gpr_results:
            try:
                if len(self.param_columns) >= 2:
                    figures[f'surface_3d_{metric}'] = self.generate_surface_3d(metric)
            except Exception as e:
                print(f"Error 3D {metric}: {e}")
            
            try:
                figures[f'pd_grid_{metric}'] = self.generate_pd_grid(metric)
            except Exception as e:
                print(f"Error PD {metric}: {e}")
            
            try:
                figures[f'distribution_{metric}'] = self.generate_distribution_analysis(metric)
            except Exception as e:
                print(f"Error dist {metric}: {e}")
        
        return {k: v for k, v in figures.items() if v}


# =============================================================================
# GENERADOR PDF
# =============================================================================

class ProfessionalPDFGenerator:
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str], filepath: str = None):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
        self.filepath = filepath
        
        self.analyzer = GlobalStatisticalAnalyzer(gpr_results, gpr_models, df, param_columns)
        self.conclusions = self.analyzer.generate_conclusions()
        
        self.fig_gen = ProfessionalFigureGenerator(gpr_results, gpr_models, df, param_columns)
        self.figures = self.fig_gen.generate_all()
    
    def _get_template(self) -> str:
        return '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Análisis GPR - MODELOX</title>
    <style>
        @page { 
            size: A4; 
            margin: 1.8cm 2cm;
            @bottom-center { content: "Página " counter(page); font-size: 9px; color: #6b7280; }
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            font-size: 10pt; 
            line-height: 1.55;
            color: #111827;
        }
        
        /* PORTADA */
        .cover {
            page-break-after: always;
            background: linear-gradient(145deg, #1e3a5f 0%, #2d5a87 60%, #3b82f6 100%);
            color: white;
            margin: -1.8cm -2cm;
            padding: 3cm 2.5cm;
            min-height: 100vh;
            text-align: center;
        }
        
        .cover-logo { font-size: 52pt; font-weight: 800; letter-spacing: -3px; margin-bottom: 0.3cm; }
        .cover h1 { font-size: 26pt; font-weight: 300; margin: 1.5cm 0 0.3cm; border: none; }
        .cover .subtitle { font-size: 13pt; color: rgba(255,255,255,0.75); }
        
        .cover-info {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 1.2cm;
            border-radius: 12px;
            margin: 1.5cm auto;
            max-width: 10cm;
        }
        .cover-info p { margin: 0.25cm 0; font-size: 11pt; }
        
        .quality-badge {
            display: inline-block;
            padding: 0.5cm 1.2cm;
            border-radius: 2cm;
            font-size: 15pt;
            font-weight: 700;
            margin-top: 1cm;
        }
        .badge-excellent { background: #059669; }
        .badge-good { background: #3b82f6; }
        .badge-moderate { background: #d97706; }
        .badge-low { background: #dc2626; }
        
        /* SECCIONES */
        h1 { 
            color: #1e3a5f;
            font-size: 16pt;
            font-weight: 700;
            border-bottom: 2.5px solid #3b82f6;
            padding-bottom: 0.3cm;
            margin: 0.8cm 0 0.5cm;
        }
        
        h2 {
            color: #2d5a87;
            font-size: 13pt;
            font-weight: 600;
            border-left: 4px solid #3b82f6;
            padding-left: 0.4cm;
            margin: 0.6cm 0 0.4cm;
        }
        
        h3 { color: #374151; font-size: 11pt; margin: 0.4cm 0 0.3cm; }
        
        .page-break { page-break-before: always; }
        
        /* TABLAS */
        table { width: 100%; border-collapse: collapse; margin: 0.4cm 0; font-size: 9pt; }
        th { background: linear-gradient(135deg, #1e3a5f, #2d5a87); color: white; padding: 0.25cm; font-weight: 600; }
        td { padding: 0.2cm; border: 1px solid #e5e7eb; text-align: center; }
        tr:nth-child(even) { background: #f8fafc; }
        
        .excellent { color: #059669; font-weight: 700; }
        .good { color: #2563eb; font-weight: 600; }
        .moderate { color: #d97706; font-weight: 600; }
        .low { color: #dc2626; font-weight: 600; }
        
        /* FIGURAS */
        .figure { text-align: center; margin: 0.5cm 0; page-break-inside: avoid; }
        .figure img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }
        .figure-caption { font-size: 8pt; color: #6b7280; margin-top: 0.2cm; font-style: italic; }
        
        /* CAJAS */
        .info-box { background: #eff6ff; border: 1px solid #bfdbfe; padding: 0.4cm; border-radius: 6px; margin: 0.4cm 0; }
        .success-box { background: #ecfdf5; border: 1px solid #a7f3d0; padding: 0.4cm; border-radius: 6px; margin: 0.4cm 0; }
        .warning-box { background: #fffbeb; border: 1px solid #fcd34d; padding: 0.4cm; border-radius: 6px; margin: 0.4cm 0; }
        
        .summary-box {
            background: linear-gradient(135deg, #1e3a5f, #2d5a87);
            color: white;
            padding: 0.6cm;
            border-radius: 8px;
            margin: 0.5cm 0;
        }
        .summary-box h3 { color: white; margin-top: 0; }
        
        /* STATS GRID */
        .stats-grid { display: flex; gap: 0.4cm; margin: 0.5cm 0; flex-wrap: wrap; }
        .stat-card { 
            flex: 1; min-width: 3.5cm;
            background: #f8fafc; 
            border: 1px solid #e5e7eb; 
            border-radius: 8px; 
            padding: 0.4cm; 
            text-align: center;
        }
        .stat-value { font-size: 18pt; font-weight: 700; color: #1e3a5f; }
        .stat-label { font-size: 8pt; color: #6b7280; }
        
        /* CONCLUSIONES */
        .conclusion-item {
            padding: 0.35cm 0.4cm;
            margin: 0.25cm 0;
            border-radius: 5px;
            border-left: 4px solid;
            font-size: 9.5pt;
        }
        .conclusion-confidence { background: #eff6ff; border-color: #3b82f6; }
        .conclusion-recommendation { background: #ecfdf5; border-color: #059669; }
        .conclusion-warning { background: #fffbeb; border-color: #d97706; }
        
        .footer { text-align: center; margin-top: 1cm; padding-top: 0.5cm; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 8pt; }
    </style>
</head>
<body>
    <!-- PORTADA -->
    <div class="cover">
        <div class="cover-logo">MODELOX</div>
        <h1>ANÁLISIS DE OPTIMIZACIÓN</h1>
        <div class="subtitle">Regresión de Procesos Gaussianos (GPR)</div>
        
        <div class="cover-info">
            <p>📁 <strong>{{ filename }}</strong></p>
            <p>📊 Trials: <strong>{{ "{:,}".format(n_trials) }}</strong></p>
            <p>🔧 Parámetros: <strong>{{ n_params }}</strong></p>
            <p>📈 Métricas: <strong>{{ n_metrics }}</strong></p>
            <p>📅 {{ date }}</p>
        </div>
        
        <div class="quality-badge badge-{{ quality_class }}">{{ global_stats.overall_quality }}</div>
        <p style="margin-top: 0.4cm; opacity: 0.7;">R² = {{ "%.4f"|format(global_stats.mean_r2) }}</p>
    </div>
    
    <!-- RESUMEN EJECUTIVO -->
    <h1>Resumen Ejecutivo</h1>
    
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">{{ "%.3f"|format(global_stats.mean_r2) }}</div><div class="stat-label">R² Promedio</div></div>
        <div class="stat-card"><div class="stat-value">{{ "%.0f"|format(global_stats.stability_score * 100) }}%</div><div class="stat-label">Estabilidad</div></div>
        <div class="stat-card"><div class="stat-value">{{ "%.0f"|format(global_stats.robustness_score * 100) }}%</div><div class="stat-label">Robustez</div></div>
        <div class="stat-card"><div class="stat-value">{{ "%.0f"|format(global_stats.confidence_level * 100) }}%</div><div class="stat-label">Confianza</div></div>
    </div>
    
    <table>
        <tr><th>Métrica</th><th>R²</th><th>Calidad</th><th>Ruido (σn)</th></tr>
        {% for metric, result in results.items() %}
        <tr>
            <td><strong>{{ metric.upper() }}</strong></td>
            <td>{{ "%.4f"|format(result.r2_score) }}</td>
            <td class="{{ 'excellent' if result.r2_score > 0.75 else 'good' if result.r2_score > 0.6 else 'moderate' if result.r2_score > 0.4 else 'low' }}">
                {{ 'EXCELENTE' if result.r2_score > 0.75 else 'BUENO' if result.r2_score > 0.6 else 'MODERADO' if result.r2_score > 0.4 else 'BAJO' }}
            </td>
            <td>{{ "%.4f"|format(result.noise_level) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <div class="page-break"></div>
    
    <!-- DASHBOARD -->
    <h1>Dashboard de Calidad</h1>
    {% if figures.quality_dashboard %}
    <div class="figure">
        <img src="data:image/png;base64,{{ figures.quality_dashboard }}" alt="Dashboard">
        <div class="figure-caption">Fig. 1: Dashboard de calidad del modelo GPR</div>
    </div>
    {% endif %}
    
    <div class="page-break"></div>
    
    <!-- ANÁLISIS DE PARÁMETROS -->
    <h1>Análisis de Parámetros</h1>
    
    {% if figures.param_importance %}
    <div class="figure">
        <img src="data:image/png;base64,{{ figures.param_importance }}" alt="Importancia">
        <div class="figure-caption">Fig. 2: Análisis de importancia, sensibilidad y monotonicidad</div>
    </div>
    {% endif %}
    
    <h2>Ranking de Importancia</h2>
    <table>
        <tr><th>#</th><th>Parámetro</th><th>Importancia</th><th>Óptimo</th><th>IC 95%</th><th>Mono.</th><th>No Lin.</th></tr>
        {% for p in param_analyses %}
        <tr>
            <td>{{ loop.index }}</td>
            <td><strong>{{ p.name.replace('param_', '').replace('_', ' ').title()[:20] }}</strong></td>
            <td>{{ "%.1f"|format(p.importance * 100) }}%</td>
            <td class="excellent">{{ "%.2f"|format(p.optimal_value) }}</td>
            <td>[{{ "%.2f"|format(p.optimal_ci_lower) }}, {{ "%.2f"|format(p.optimal_ci_upper) }}]</td>
            <td>{{ "%.2f"|format(p.monotonicity) }}</td>
            <td>{{ "%.2f"|format(p.nonlinearity) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <div class="page-break"></div>
    
    <!-- SUPERFICIES 3D -->
    <h1>Superficies de Respuesta 3D</h1>
    {% for metric in results %}
    {% set key = 'surface_3d_' + metric %}
    {% if figures.get(key) %}
    <h2>{{ metric.upper() }}</h2>
    <div class="figure">
        <img src="data:image/png;base64,{{ figures[key] }}" alt="Superficie 3D">
        <div class="figure-caption">Superficie de respuesta 3D, mapa de rendimiento con zona óptima dorada, y campo de gradientes de sensibilidad</div>
    </div>
    {% if not loop.last %}<div class="page-break"></div>{% endif %}
    {% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- DEPENDENCIA PARCIAL -->
    <h1>Dependencia Parcial por Métrica</h1>
    {% for metric, result in results.items() %}
    <h2>{{ metric.upper() }}</h2>
    
    <div class="info-box">
        <strong>R² = {{ "%.4f"|format(result.r2_score) }}</strong> &nbsp;|&nbsp; 
        σn = {{ "%.4f"|format(result.noise_level) }} &nbsp;|&nbsp;
        {{ 'EXCELENTE' if result.r2_score > 0.75 else 'BUENO' if result.r2_score > 0.6 else 'MODERADO' if result.r2_score > 0.4 else 'BAJO' }}
    </div>
    
    {% set key = 'pd_grid_' + metric %}
    {% if figures.get(key) %}
    <div class="figure">
        <img src="data:image/png;base64,{{ figures[key] }}" alt="PD Grid">
        <div class="figure-caption">Dependencia parcial con intervalos de confianza al 95%</div>
    </div>
    {% endif %}
    
    <h3>Valores Óptimos</h3>
    <table>
        <tr><th>Parámetro</th><th>Óptimo</th><th>Predicción</th><th>IC 95%</th></tr>
        {% for param, pred in result.predictions.items() %}
        <tr>
            <td>{{ param.replace('param_', '').replace('_', ' ').title() }}</td>
            <td>{{ "%.3f"|format(optimal[metric][param]['value']) }}</td>
            <td class="good">{{ "%.3f"|format(optimal[metric][param]['pred']) }}</td>
            <td>[{{ "%.3f"|format(optimal[metric][param]['ci_l']) }}, {{ "%.3f"|format(optimal[metric][param]['ci_u']) }}]</td>
        </tr>
        {% endfor %}
    </table>
    
    {% if not loop.last %}<div class="page-break"></div>{% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- ANÁLISIS ESTADÍSTICO -->
    <h1>Análisis Estadístico</h1>
    {% for metric in results %}
    {% set key = 'distribution_' + metric %}
    {% if figures.get(key) %}
    <h2>{{ metric.upper() }}</h2>
    <div class="figure">
        <img src="data:image/png;base64,{{ figures[key] }}" alt="Distribución">
        <div class="figure-caption">Distribuciones, normalidad, y análisis de incertidumbre</div>
    </div>
    {% if not loop.last %}<div class="page-break"></div>{% endif %}
    {% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- CONCLUSIONES -->
    <h1>Conclusiones y Recomendaciones</h1>
    
    <h2>📊 Evaluación del Modelo</h2>
    {% for stmt in conclusions.confidence_statements %}
    <div class="conclusion-item conclusion-confidence">{{ stmt }}</div>
    {% endfor %}
    
    {% if conclusions.recommendations %}
    <h2>✅ Recomendaciones</h2>
    {% for rec in conclusions.recommendations %}
    <div class="conclusion-item conclusion-recommendation">→ {{ rec }}</div>
    {% endfor %}
    {% endif %}
    
    {% if conclusions.warnings %}
    <h2>⚠️ Advertencias</h2>
    {% for warn in conclusions.warnings %}
    <div class="conclusion-item conclusion-warning">⚠ {{ warn }}</div>
    {% endfor %}
    {% endif %}
    
    <div class="summary-box">
        <h3>Resumen Final</h3>
        <p>Se analizaron <strong>{{ "{:,}".format(n_trials) }} trials</strong> con <strong>{{ n_params }} parámetros</strong> 
        para <strong>{{ n_metrics }} métricas</strong>.</p>
        <p>Calidad global: <strong>{{ global_stats.overall_quality }}</strong> (R²={{ "%.3f"|format(global_stats.mean_r2) }}, 
        Confianza={{ "%.0f"|format(global_stats.confidence_level * 100) }}%).</p>
        {% if param_analyses %}
        <p>Parámetro clave: <strong>{{ param_analyses[0].name.replace('param_', '').title() }}</strong> 
        ({{ "%.0f"|format(param_analyses[0].importance * 100) }}% importancia).</p>
        {% endif %}
    </div>
    
    <div class="footer">
        <strong>MODELOX</strong> — Sistema de Análisis de Trading<br>
        Reporte generado el {{ date }}
    </div>
</body>
</html>'''
    
    def generate(self, output_path: str = None) -> str:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        console.print(Panel("[bold blue]📄 Generando PDF Profesional V2...[/bold blue]"))
        
        if output_path is None:
            base = Path(self.filepath).stem if self.filepath else "analisis"
            output_path = f"reporte_profesional_{base}.pdf"
        if not output_path.endswith('.pdf'):
            output_path += '.pdf'
        
        # Calcular óptimos
        optimal = {}
        for metric, result in self.gpr_results.items():
            optimal[metric] = {}
            for param, pred in result.predictions.items():
                idx = np.argmax(pred.mean_prediction)
                optimal[metric][param] = {
                    'value': pred.param_values[idx],
                    'pred': pred.mean_prediction[idx],
                    'ci_l': pred.lower_ci[idx],
                    'ci_u': pred.upper_ci[idx]
                }
        
        quality_class = 'excellent' if self.conclusions.global_stats.mean_r2 > 0.75 else \
                       'good' if self.conclusions.global_stats.mean_r2 > 0.6 else \
                       'moderate' if self.conclusions.global_stats.mean_r2 > 0.4 else 'low'
        
        context = {
            'filename': Path(self.filepath).name if self.filepath else "N/A",
            'n_trials': len(self.df),
            'n_params': len(self.param_columns),
            'n_metrics': len(self.gpr_results),
            'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'results': self.gpr_results,
            'figures': self.figures,
            'optimal': optimal,
            'global_stats': self.conclusions.global_stats,
            'param_analyses': self.conclusions.param_analyses,
            'conclusions': self.conclusions,
            'quality_class': quality_class,
        }
        
        try:
            from jinja2 import Template
            html = Template(self._get_template()).render(**context)
            
            try:
                from weasyprint import HTML
                HTML(string=html).write_pdf(output_path)
                console.print(f"[green]✅ PDF generado: {output_path}[/green]")
                return output_path
            except ImportError:
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                console.print(f"[yellow]⚠ WeasyPrint no disponible. HTML: {html_path}[/yellow]")
                return html_path
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return ""


def generate_professional_report(gpr_results: Dict, gpr_models: Dict, df, 
                                  param_columns: List[str], filepath: str = None,
                                  output_path: str = None) -> str:
    """Genera reporte PDF profesional."""
    gen = ProfessionalPDFGenerator(gpr_results, gpr_models, df, param_columns, filepath)
    return gen.generate(output_path)

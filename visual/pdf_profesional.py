#!/usr/bin/env python3
"""
================================================================================
📄 GENERADOR DE PDF PROFESIONAL - MODELOX
================================================================================

Sistema avanzado de generación de reportes PDF con:
- Gráficas 3D de superficie topográfica
- Análisis estadístico y probabilístico global
- Métricas de robustez y estabilidad
- Conclusiones automáticas generadas por IA
- Diseño profesional de nivel institucional

================================================================================
"""

import numpy as np
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Matplotlib con estilo profesional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

# Estadísticas
from scipy import stats
from scipy.interpolate import griddata


# =============================================================================
# CONFIGURACIÓN DE ESTILO PROFESIONAL
# =============================================================================

# Colores institucionales
COLORS = {
    'primary': '#1a365d',      # Azul oscuro institucional
    'secondary': '#2c5282',    # Azul medio
    'accent': '#3182ce',       # Azul brillante
    'success': '#276749',      # Verde
    'warning': '#c05621',      # Naranja
    'danger': '#c53030',       # Rojo
    'text': '#1a202c',         # Texto principal
    'text_light': '#718096',   # Texto secundario
    'bg_light': '#f7fafc',     # Fondo claro
    'bg_dark': '#edf2f7',      # Fondo gris
    'grid': '#e2e8f0',         # Líneas de grid
}

# Paletas para gráficas
CMAP_SURFACE = LinearSegmentedColormap.from_list('custom_surface', [
    '#1a365d', '#2c5282', '#3182ce', '#4299e1', '#63b3ed', 
    '#90cdf4', '#bee3f8', '#ebf8ff', '#f0fff4', '#9ae6b4', 
    '#48bb78', '#38a169', '#276749'
])

CMAP_HEATMAP = LinearSegmentedColormap.from_list('custom_heat', [
    '#1a365d', '#2c5282', '#3182ce', '#4299e1', '#63b3ed',
    '#9ae6b4', '#48bb78', '#f6e05e', '#ed8936', '#c53030'
])


def setup_professional_style():
    """Configura estilo matplotlib profesional."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


setup_professional_style()


# =============================================================================
# DATACLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class GlobalModelStats:
    """Estadísticas globales del modelo."""
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
    """Análisis detallado de un parámetro."""
    name: str
    importance: float
    sensitivity: float
    optimal_value: float
    optimal_ci_lower: float
    optimal_ci_upper: float
    monotonicity: float  # -1 a 1, qué tan monótona es la relación
    nonlinearity: float  # 0 a 1, qué tan no lineal
    interaction_strength: float  # Fuerza de interacciones


@dataclass
class ConclusionData:
    """Datos para generar conclusiones."""
    global_stats: GlobalModelStats
    param_analyses: List[ParameterAnalysis]
    metric_results: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    confidence_statements: List[str]


# =============================================================================
# ANÁLISIS ESTADÍSTICO GLOBAL
# =============================================================================

class GlobalStatisticalAnalyzer:
    """Analizador estadístico global del modelo."""
    
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str]):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
    
    def compute_global_stats(self) -> GlobalModelStats:
        """Calcula estadísticas globales del modelo."""
        r2_scores = [r.r2_score for r in self.gpr_results.values()]
        noise_levels = [r.noise_level for r in self.gpr_results.values()]
        
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        
        # Calidad global
        if mean_r2 > 0.75 and std_r2 < 0.1:
            quality = "EXCELENTE"
        elif mean_r2 > 0.6 and std_r2 < 0.15:
            quality = "BUENO"
        elif mean_r2 > 0.4:
            quality = "MODERADO"
        else:
            quality = "BAJO"
        
        # Score de estabilidad (basado en varianza de R² entre métricas)
        stability = 1.0 - min(std_r2 * 2, 1.0)
        
        # Score de robustez (inverso del ruido promedio)
        avg_noise = np.mean(noise_levels)
        robustness = 1.0 / (1.0 + avg_noise * 10)
        
        # Nivel de confianza
        confidence = min(mean_r2 * stability * robustness, 0.99)
        
        # Dimensionalidad efectiva (PCA-like)
        X = self.df[self.param_columns].to_numpy()
        X = X[~np.any(np.isnan(X), axis=1)]
        if len(X) > 10:
            X_centered = X - X.mean(axis=0)
            _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
            variance_explained = (s ** 2) / (s ** 2).sum()
            eff_dim = np.sum(variance_explained > 0.01)
        else:
            eff_dim = len(self.param_columns)
        
        return GlobalModelStats(
            mean_r2=mean_r2,
            std_r2=std_r2,
            min_r2=min(r2_scores),
            max_r2=max(r2_scores),
            overall_quality=quality,
            stability_score=stability,
            robustness_score=robustness,
            confidence_level=confidence,
            total_params=len(self.param_columns),
            total_samples=len(self.df),
            effective_dimensionality=eff_dim
        )
    
    def analyze_parameters(self) -> List[ParameterAnalysis]:
        """Analiza cada parámetro en detalle."""
        analyses = []
        
        for param in self.param_columns:
            importances = []
            sensitivities = []
            optimals = []
            optimal_cis = []
            monotonicities = []
            nonlinearities = []
            
            for metric, result in self.gpr_results.items():
                if param in result.predictions:
                    pred = result.predictions[param]
                    
                    # Importancia: rango de predicción / escala total
                    pred_range = np.max(pred.mean_prediction) - np.min(pred.mean_prediction)
                    importances.append(pred_range)
                    
                    # Sensibilidad: varianza de las predicciones
                    sensitivities.append(np.std(pred.mean_prediction))
                    
                    # Óptimo
                    idx_opt = np.argmax(pred.mean_prediction)
                    optimals.append(pred.param_values[idx_opt])
                    optimal_cis.append((pred.lower_ci[idx_opt], pred.upper_ci[idx_opt]))
                    
                    # Monotonicidad
                    diffs = np.diff(pred.mean_prediction)
                    if len(diffs) > 0:
                        mono = np.mean(np.sign(diffs))
                    else:
                        mono = 0
                    monotonicities.append(mono)
                    
                    # No linealidad (diferencia con ajuste lineal)
                    x = pred.param_values
                    y = pred.mean_prediction
                    if len(x) > 2:
                        slope, intercept = np.polyfit(x, y, 1)
                        linear_fit = slope * x + intercept
                        residuals = y - linear_fit
                        nonlin = np.std(residuals) / (np.std(y) + 1e-8)
                        nonlinearities.append(min(nonlin * 2, 1.0))
                    else:
                        nonlinearities.append(0)
            
            if importances:
                analyses.append(ParameterAnalysis(
                    name=param,
                    importance=np.mean(importances),
                    sensitivity=np.mean(sensitivities),
                    optimal_value=np.mean(optimals),
                    optimal_ci_lower=np.mean([ci[0] for ci in optimal_cis]),
                    optimal_ci_upper=np.mean([ci[1] for ci in optimal_cis]),
                    monotonicity=np.mean(monotonicities),
                    nonlinearity=np.mean(nonlinearities),
                    interaction_strength=0.0  # Se calcula después
                ))
        
        # Normalizar importancia
        total_imp = sum(a.importance for a in analyses)
        if total_imp > 0:
            for a in analyses:
                a.importance = a.importance / total_imp
        
        # Ordenar por importancia
        analyses.sort(key=lambda x: x.importance, reverse=True)
        
        return analyses
    
    def generate_conclusions(self) -> ConclusionData:
        """Genera conclusiones automáticas basadas en el análisis."""
        global_stats = self.compute_global_stats()
        param_analyses = self.analyze_parameters()
        
        recommendations = []
        warnings = []
        confidence_statements = []
        
        # Conclusiones basadas en R²
        if global_stats.mean_r2 > 0.75:
            confidence_statements.append(
                f"El modelo presenta un ajuste EXCELENTE (R² promedio = {global_stats.mean_r2:.3f}), "
                "lo que indica que las predicciones son altamente confiables."
            )
        elif global_stats.mean_r2 > 0.5:
            confidence_statements.append(
                f"El modelo presenta un ajuste MODERADO (R² promedio = {global_stats.mean_r2:.3f}). "
                "Las predicciones deben interpretarse con precaución."
            )
            warnings.append("R² moderado sugiere que existen factores no capturados por los parámetros analizados.")
        else:
            warnings.append(
                f"ADVERTENCIA: R² bajo ({global_stats.mean_r2:.3f}). El modelo tiene capacidad predictiva limitada."
            )
        
        # Conclusiones basadas en estabilidad
        if global_stats.stability_score > 0.8:
            confidence_statements.append(
                "Alta estabilidad entre métricas indica consistencia del modelo."
            )
        else:
            warnings.append(
                "Variabilidad entre métricas sugiere que el modelo puede ser sensible a la métrica elegida."
            )
        
        # Recomendaciones basadas en parámetros
        if param_analyses:
            top_param = param_analyses[0]
            recommendations.append(
                f"El parámetro más influyente es '{top_param.name}' "
                f"(importancia: {top_param.importance*100:.1f}%). "
                f"Valor óptimo recomendado: {top_param.optimal_value:.2f} "
                f"[IC 95%: {top_param.optimal_ci_lower:.2f} - {top_param.optimal_ci_upper:.2f}]"
            )
            
            # Parámetros poco importantes
            low_importance = [p for p in param_analyses if p.importance < 0.05]
            if low_importance:
                names = ", ".join([p.name for p in low_importance[:3]])
                recommendations.append(
                    f"Parámetros con baja influencia ({names}) podrían simplificarse o fijarse."
                )
            
            # Parámetros con alta no linealidad
            nonlinear = [p for p in param_analyses if p.nonlinearity > 0.5]
            if nonlinear:
                names = ", ".join([p.name for p in nonlinear[:3]])
                warnings.append(
                    f"Los parámetros {names} muestran comportamiento no lineal. "
                    "Considerar rangos de exploración más finos."
                )
        
        # Recomendaciones de robustez
        if global_stats.robustness_score < 0.5:
            warnings.append(
                "Nivel de ruido elevado detectado. Considerar más trials o filtrar outliers."
            )
        
        return ConclusionData(
            global_stats=global_stats,
            param_analyses=param_analyses,
            metric_results=self.gpr_results,
            recommendations=recommendations,
            warnings=warnings,
            confidence_statements=confidence_statements
        )


# =============================================================================
# GENERADOR DE FIGURAS PROFESIONALES
# =============================================================================

class ProfessionalFigureGenerator:
    """Generador de figuras de calidad publicación."""
    
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str]):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
        self.figures_b64 = {}
    
    def _fig_to_base64(self, fig: plt.Figure, dpi: int = 200) -> str:
        """Convierte figura a base64."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_b64
    
    def generate_3d_surface(self, metric: str, param1_idx: int = 0, param2_idx: int = 1) -> str:
        """Genera superficie 3D topográfica profesional."""
        if metric not in self.gpr_models:
            return ""
        
        gpr = self.gpr_models[metric]
        
        # Obtener datos
        X = self.df[self.param_columns].to_numpy().astype(np.float64)
        valid = ~np.any(np.isnan(X), axis=1)
        X = X[valid]
        
        if len(X) < 10:
            return ""
        
        # Crear grid para superficie
        p1_name = self.param_columns[param1_idx]
        p2_name = self.param_columns[param2_idx]
        
        p1_range = np.linspace(X[:, param1_idx].min(), X[:, param1_idx].max(), 40)
        p2_range = np.linspace(X[:, param2_idx].min(), X[:, param2_idx].max(), 40)
        P1, P2 = np.meshgrid(p1_range, p2_range)
        
        # Crear grid de predicción
        X_grid = np.column_stack([
            P1.ravel() if i == param1_idx else 
            P2.ravel() if i == param2_idx else 
            np.full(P1.size, X[:, i].mean())
            for i in range(len(self.param_columns))
        ])
        
        # Predecir
        try:
            y_pred, y_std = gpr.predict_batch(X_grid)
            Z = y_pred.reshape(P1.shape)
            Z_std = y_std.reshape(P1.shape)
        except Exception:
            return ""
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(14, 5))
        
        # Subplot 1: Superficie 3D
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(P1, P2, Z, cmap=CMAP_SURFACE, 
                                edgecolor='none', alpha=0.9,
                                antialiased=True)
        ax1.set_xlabel(p1_name.replace('param_', '').replace('_', ' ').title(), fontsize=9)
        ax1.set_ylabel(p2_name.replace('param_', '').replace('_', ' ').title(), fontsize=9)
        ax1.set_zlabel(metric.upper(), fontsize=9)
        ax1.set_title('Superficie de Respuesta', fontweight='bold', fontsize=11)
        ax1.view_init(elev=25, azim=45)
        
        # Añadir contornos en la base
        ax1.contour(P1, P2, Z, zdir='z', offset=Z.min(), cmap=CMAP_HEATMAP, alpha=0.5)
        
        # Subplot 2: Contorno con incertidumbre
        ax2 = fig.add_subplot(132)
        contour = ax2.contourf(P1, P2, Z, levels=20, cmap=CMAP_HEATMAP)
        ax2.contour(P1, P2, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
        
        # Marcar óptimo
        opt_idx = np.unravel_index(np.argmax(Z), Z.shape)
        ax2.scatter(P1[opt_idx], P2[opt_idx], c='white', s=100, marker='*', 
                   edgecolors=COLORS['primary'], linewidths=2, zorder=5)
        ax2.annotate(f'Óptimo\n({P1[opt_idx]:.2f}, {P2[opt_idx]:.2f})',
                    (P1[opt_idx], P2[opt_idx]), textcoords='offset points',
                    xytext=(10, 10), fontsize=8, color=COLORS['primary'],
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel(p1_name.replace('param_', '').replace('_', ' ').title())
        ax2.set_ylabel(p2_name.replace('param_', '').replace('_', ' ').title())
        ax2.set_title('Mapa de Contorno', fontweight='bold')
        plt.colorbar(contour, ax=ax2, label=metric.upper())
        
        # Subplot 3: Incertidumbre
        ax3 = fig.add_subplot(133)
        uncert = ax3.contourf(P1, P2, Z_std, levels=20, cmap='YlOrRd')
        ax3.contour(P1, P2, Z_std, levels=10, colors='white', linewidths=0.3, alpha=0.5)
        ax3.set_xlabel(p1_name.replace('param_', '').replace('_', ' ').title())
        ax3.set_ylabel(p2_name.replace('param_', '').replace('_', ' ').title())
        ax3.set_title('Mapa de Incertidumbre (σ)', fontweight='bold')
        plt.colorbar(uncert, ax=ax3, label='Desviación Estándar')
        
        fig.suptitle(f'{metric.upper()} - Análisis de Superficie', 
                    fontsize=14, fontweight='bold', color=COLORS['primary'])
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_parameter_importance_chart(self, param_analyses: List[ParameterAnalysis]) -> str:
        """Genera gráfico de importancia de parámetros."""
        if not param_analyses:
            return ""
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        names = [p.name.replace('param_', '').replace('_', ' ')[:15] for p in param_analyses]
        
        # 1. Importancia
        ax1 = axes[0]
        importance = [p.importance * 100 for p in param_analyses]
        colors = [COLORS['primary'] if i < 3 else COLORS['text_light'] for i in range(len(names))]
        bars = ax1.barh(names, importance, color=colors, edgecolor='white', linewidth=0.5)
        ax1.set_xlabel('Importancia Relativa (%)')
        ax1.set_title('Importancia de Parámetros', fontweight='bold')
        ax1.invert_yaxis()
        
        # Añadir valores
        for bar, val in zip(bars, importance):
            ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=8)
        
        # 2. Sensibilidad vs No linealidad
        ax2 = axes[1]
        sensitivity = [p.sensitivity for p in param_analyses]
        nonlinearity = [p.nonlinearity for p in param_analyses]
        
        scatter = ax2.scatter(sensitivity, nonlinearity, 
                             c=importance, cmap=CMAP_HEATMAP,
                             s=[100 + imp*300 for imp in [p.importance for p in param_analyses]],
                             alpha=0.7, edgecolors='white', linewidths=1)
        
        for i, name in enumerate(names):
            ax2.annotate(name, (sensitivity[i], nonlinearity[i]), 
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords='offset points')
        
        ax2.set_xlabel('Sensibilidad')
        ax2.set_ylabel('No Linealidad')
        ax2.set_title('Sensibilidad vs No Linealidad', fontweight='bold')
        ax2.axhline(0.5, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Alta no linealidad')
        ax2.legend(fontsize=8)
        
        # 3. Monotonicidad
        ax3 = axes[2]
        monotonicity = [p.monotonicity for p in param_analyses]
        colors_mono = [COLORS['success'] if m > 0.3 else COLORS['danger'] if m < -0.3 else COLORS['text_light'] 
                      for m in monotonicity]
        bars = ax3.barh(names, monotonicity, color=colors_mono, edgecolor='white', linewidth=0.5)
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.axvline(0.3, color=COLORS['success'], linestyle='--', alpha=0.3)
        ax3.axvline(-0.3, color=COLORS['danger'], linestyle='--', alpha=0.3)
        ax3.set_xlabel('Monotonicidad (-1 a 1)')
        ax3.set_title('Tendencia Monotónica', fontweight='bold')
        ax3.set_xlim(-1.1, 1.1)
        ax3.invert_yaxis()
        
        fig.suptitle('Análisis de Parámetros', fontsize=14, fontweight='bold', color=COLORS['primary'])
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_model_quality_dashboard(self, global_stats: GlobalModelStats) -> str:
        """Genera dashboard de calidad del modelo."""
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Gauge de R² promedio
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_gauge(ax1, global_stats.mean_r2, 'R² Promedio', 
                        [(0, 0.4, COLORS['danger']), (0.4, 0.7, COLORS['warning']), (0.7, 1.0, COLORS['success'])])
        
        # 2. Gauge de Estabilidad
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_gauge(ax2, global_stats.stability_score, 'Estabilidad',
                        [(0, 0.5, COLORS['danger']), (0.5, 0.8, COLORS['warning']), (0.8, 1.0, COLORS['success'])])
        
        # 3. Gauge de Robustez
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_gauge(ax3, global_stats.robustness_score, 'Robustez',
                        [(0, 0.4, COLORS['danger']), (0.4, 0.7, COLORS['warning']), (0.7, 1.0, COLORS['success'])])
        
        # 4. Gauge de Confianza
        ax4 = fig.add_subplot(gs[0, 3])
        self._draw_gauge(ax4, global_stats.confidence_level, 'Confianza',
                        [(0, 0.4, COLORS['danger']), (0.4, 0.7, COLORS['warning']), (0.7, 1.0, COLORS['success'])])
        
        # 5. Distribución de R² por métrica
        ax5 = fig.add_subplot(gs[1, :2])
        r2_scores = [r.r2_score for r in self.gpr_results.values()]
        metric_names = [m.upper() for m in self.gpr_results.keys()]
        
        colors = [COLORS['success'] if r > 0.7 else COLORS['warning'] if r > 0.4 else COLORS['danger'] 
                 for r in r2_scores]
        bars = ax5.bar(metric_names, r2_scores, color=colors, edgecolor='white', linewidth=1)
        ax5.axhline(0.7, color=COLORS['success'], linestyle='--', alpha=0.5, label='Excelente (0.7)')
        ax5.axhline(0.4, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Moderado (0.4)')
        ax5.set_ylabel('R² Score')
        ax5.set_title('Calidad por Métrica', fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.legend(fontsize=8, loc='lower right')
        
        for bar, val in zip(bars, r2_scores):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
        
        # 6. Resumen numérico
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        
        summary_text = f"""
RESUMEN DEL MODELO

Calidad Global:     {global_stats.overall_quality}
R² Promedio:        {global_stats.mean_r2:.4f} ± {global_stats.std_r2:.4f}
R² Rango:           [{global_stats.min_r2:.4f} - {global_stats.max_r2:.4f}]

Estabilidad:        {global_stats.stability_score*100:.1f}%
Robustez:           {global_stats.robustness_score*100:.1f}%
Confianza:          {global_stats.confidence_level*100:.1f}%

Parámetros:         {global_stats.total_params}
Samples:            {global_stats.total_samples:,}
Dim. Efectiva:      {global_stats.effective_dimensionality:.1f}
"""
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_light'], 
                         edgecolor=COLORS['grid']))
        
        fig.suptitle('Dashboard de Calidad del Modelo', fontsize=14, fontweight='bold', 
                    color=COLORS['primary'])
        
        return self._fig_to_base64(fig)
    
    def _draw_gauge(self, ax, value: float, title: str, ranges: List[Tuple[float, float, str]]):
        """Dibuja un gauge semicircular."""
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Dibujar arcos de fondo
        for start, end, color in ranges:
            theta_start = 180 - start * 180
            theta_end = 180 - end * 180
            wedge = plt.matplotlib.patches.Wedge(
                (0.5, 0), 0.4, theta_end, theta_start,
                width=0.15, facecolor=color, alpha=0.3
            )
            ax.add_patch(wedge)
        
        # Dibujar valor actual
        theta = 180 - value * 180
        ax.annotate('', xy=(0.5 + 0.35 * np.cos(np.radians(theta)), 
                           0.35 * np.sin(np.radians(theta))),
                   xytext=(0.5, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
        
        # Valor numérico
        ax.text(0.5, -0.15, f'{value:.2f}', ha='center', fontsize=14, fontweight='bold',
               color=COLORS['primary'])
        ax.text(0.5, 0.5, title, ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 0.6)
    
    def generate_distribution_analysis(self, metric: str) -> str:
        """Genera análisis de distribución para una métrica."""
        if metric not in self.gpr_results:
            return ""
        
        result = self.gpr_results[metric]
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # Obtener predicciones de todos los parámetros
        all_means = []
        all_stds = []
        param_names = []
        
        for param, pred in result.predictions.items():
            all_means.extend(pred.mean_prediction)
            all_stds.extend(pred.std_prediction)
            param_names.append(param.replace('param_', '').replace('_', ' ')[:12])
        
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        
        # 1. Histograma de predicciones
        ax1 = axes[0, 0]
        ax1.hist(all_means, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
        ax1.axvline(np.mean(all_means), color=COLORS['danger'], linestyle='--', 
                   label=f'Media: {np.mean(all_means):.2f}')
        ax1.axvline(np.median(all_means), color=COLORS['success'], linestyle=':',
                   label=f'Mediana: {np.median(all_means):.2f}')
        ax1.set_xlabel(f'{metric.upper()} Predicho')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Predicciones', fontweight='bold')
        ax1.legend(fontsize=8)
        
        # 2. Histograma de incertidumbre
        ax2 = axes[0, 1]
        ax2.hist(all_stds, bins=30, color=COLORS['warning'], alpha=0.7, edgecolor='white')
        ax2.axvline(np.mean(all_stds), color=COLORS['danger'], linestyle='--',
                   label=f'Media σ: {np.mean(all_stds):.3f}')
        ax2.set_xlabel('Desviación Estándar (σ)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Incertidumbre', fontweight='bold')
        ax2.legend(fontsize=8)
        
        # 3. Q-Q Plot
        ax3 = axes[0, 2]
        stats.probplot(all_means, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normalidad)', fontweight='bold')
        ax3.get_lines()[0].set_color(COLORS['primary'])
        ax3.get_lines()[1].set_color(COLORS['danger'])
        
        # 4. Predicción vs Incertidumbre
        ax4 = axes[1, 0]
        ax4.scatter(all_means, all_stds, alpha=0.3, c=COLORS['primary'], s=10)
        z = np.polyfit(all_means, all_stds, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_means.min(), all_means.max(), 100)
        ax4.plot(x_line, p(x_line), color=COLORS['danger'], linestyle='--', 
                label=f'Tendencia (pendiente: {z[0]:.3f})')
        ax4.set_xlabel(f'{metric.upper()} Predicho')
        ax4.set_ylabel('Incertidumbre (σ)')
        ax4.set_title('Predicción vs Incertidumbre', fontweight='bold')
        ax4.legend(fontsize=8)
        
        # 5. Boxplot por parámetro
        ax5 = axes[1, 1]
        param_data = []
        for param, pred in result.predictions.items():
            param_data.append(pred.mean_prediction)
        
        bp = ax5.boxplot(param_data, labels=param_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.6)
        ax5.set_ylabel(f'{metric.upper()}')
        ax5.set_title('Distribución por Parámetro', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Intervalo de confianza acumulativo
        ax6 = axes[1, 2]
        ci_widths = []
        for param, pred in result.predictions.items():
            widths = pred.upper_ci - pred.lower_ci
            ci_widths.extend(widths)
        ci_widths = np.sort(ci_widths)
        cumulative = np.arange(1, len(ci_widths) + 1) / len(ci_widths)
        ax6.plot(ci_widths, cumulative, color=COLORS['primary'], linewidth=2)
        ax6.axhline(0.95, color=COLORS['success'], linestyle='--', alpha=0.5, label='95%')
        ax6.axhline(0.5, color=COLORS['warning'], linestyle='--', alpha=0.5, label='50%')
        ax6.set_xlabel('Ancho del IC 95%')
        ax6.set_ylabel('Proporción Acumulada')
        ax6.set_title('CDF de Intervalos de Confianza', fontweight='bold')
        ax6.legend(fontsize=8)
        
        fig.suptitle(f'{metric.upper()} - Análisis Estadístico Completo', 
                    fontsize=14, fontweight='bold', color=COLORS['primary'])
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_partial_dependence_grid(self, metric: str) -> str:
        """Genera grid de dependencia parcial con IC."""
        if metric not in self.gpr_results:
            return ""
        
        result = self.gpr_results[metric]
        n_params = len(result.predictions)
        
        if n_params == 0:
            return ""
        
        # Calcular layout óptimo
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (param, pred) in enumerate(result.predictions.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            x = pred.param_values
            y = pred.mean_prediction
            y_lower = pred.lower_ci
            y_upper = pred.upper_ci
            
            # Banda de confianza
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=COLORS['primary'],
                           label='IC 95%')
            
            # Línea principal
            ax.plot(x, y, color=COLORS['primary'], linewidth=2, label='Predicción')
            
            # Marcar óptimo
            idx_opt = np.argmax(y)
            ax.scatter(x[idx_opt], y[idx_opt], c=COLORS['success'], s=100, 
                      marker='*', zorder=5, edgecolors='white', linewidths=1)
            ax.annotate(f'Óptimo: {x[idx_opt]:.2f}', (x[idx_opt], y[idx_opt]),
                       textcoords='offset points', xytext=(5, 5), fontsize=8,
                       color=COLORS['success'], fontweight='bold')
            
            # Etiquetas
            param_name = param.replace('param_', '').replace('_', ' ').title()
            ax.set_xlabel(param_name, fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.set_title(f'PD: {param_name}', fontweight='bold', fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for idx in range(n_params, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(f'{metric.upper()} - Dependencia Parcial con IC 95%',
                    fontsize=14, fontweight='bold', color=COLORS['primary'])
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_all_figures(self) -> Dict[str, str]:
        """Genera todas las figuras."""
        figures = {}
        
        # Análisis global
        analyzer = GlobalStatisticalAnalyzer(
            self.gpr_results, self.gpr_models, self.df, self.param_columns
        )
        global_stats = analyzer.compute_global_stats()
        param_analyses = analyzer.analyze_parameters()
        
        # Dashboard de calidad
        figures['quality_dashboard'] = self.generate_model_quality_dashboard(global_stats)
        
        # Importancia de parámetros
        figures['param_importance'] = self.generate_parameter_importance_chart(param_analyses)
        
        # Para cada métrica
        for metric in self.gpr_results:
            # Superficie 3D (si hay al menos 2 parámetros)
            if len(self.param_columns) >= 2:
                figures[f'surface_3d_{metric}'] = self.generate_3d_surface(metric)
            
            # Análisis de distribución
            figures[f'distribution_{metric}'] = self.generate_distribution_analysis(metric)
            
            # Grid de dependencia parcial
            figures[f'pd_grid_{metric}'] = self.generate_partial_dependence_grid(metric)
        
        # Filtrar figuras vacías
        figures = {k: v for k, v in figures.items() if v}
        
        return figures


# =============================================================================
# GENERADOR DE HTML/PDF PROFESIONAL
# =============================================================================

class ProfessionalPDFGenerator:
    """Generador de PDF profesional con análisis completo."""
    
    def __init__(self, gpr_results: Dict, gpr_models: Dict, df, param_columns: List[str],
                 filepath: str = None):
        self.gpr_results = gpr_results
        self.gpr_models = gpr_models
        self.df = df
        self.param_columns = param_columns
        self.filepath = filepath
        
        # Generar análisis
        self.analyzer = GlobalStatisticalAnalyzer(gpr_results, gpr_models, df, param_columns)
        self.conclusions = self.analyzer.generate_conclusions()
        
        # Generar figuras
        self.fig_generator = ProfessionalFigureGenerator(gpr_results, gpr_models, df, param_columns)
        self.figures = self.fig_generator.generate_all_figures()
    
    def _get_professional_template(self) -> str:
        """Template HTML profesional."""
        return '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Análisis GPR Profesional - MODELOX</title>
    <style>
        @page { 
            size: A4; 
            margin: 1.5cm 2cm;
            @top-center { content: "MODELOX - Análisis GPR"; font-size: 9px; color: #718096; }
            @bottom-center { content: "Página " counter(page) " de " counter(pages); font-size: 9px; color: #718096; }
        }
        
        * { box-sizing: border-box; }
        
        body { 
            font-family: 'Helvetica Neue', Arial, sans-serif; 
            font-size: 10pt; 
            line-height: 1.6; 
            color: #1a202c;
            background: white;
        }
        
        /* PORTADA */
        .cover {
            text-align: center;
            padding: 60px 40px;
            page-break-after: always;
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #3182ce 100%);
            color: white;
            margin: -1.5cm -2cm;
            padding: 80px 40px;
            min-height: 100vh;
        }
        
        .cover-logo {
            font-size: 48pt;
            font-weight: 900;
            letter-spacing: -2px;
            margin-bottom: 20px;
        }
        
        .cover h1 {
            font-size: 28pt;
            font-weight: 300;
            margin: 30px 0 10px;
            border: none;
            color: white;
        }
        
        .cover .subtitle {
            font-size: 14pt;
            color: rgba(255,255,255,0.8);
            margin-bottom: 40px;
        }
        
        .cover-info {
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 10px;
            margin: 40px auto;
            max-width: 400px;
        }
        
        .cover-info p {
            margin: 8px 0;
            font-size: 11pt;
        }
        
        .cover-quality {
            margin-top: 50px;
            padding: 20px;
        }
        
        .cover-quality .badge {
            display: inline-block;
            padding: 12px 30px;
            border-radius: 30px;
            font-size: 16pt;
            font-weight: bold;
        }
        
        .badge-excellent { background: #48bb78; }
        .badge-good { background: #4299e1; }
        .badge-moderate { background: #ed8936; }
        .badge-low { background: #fc8181; }
        
        /* ENCABEZADOS */
        h1 { 
            color: #1a365d; 
            font-size: 18pt;
            font-weight: 700;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 10px;
            margin: 30px 0 20px;
        }
        
        h2 { 
            color: #2c5282; 
            font-size: 14pt;
            font-weight: 600;
            border-left: 4px solid #3182ce;
            padding-left: 12px;
            margin: 25px 0 15px;
        }
        
        h3 { 
            color: #4a5568; 
            font-size: 12pt;
            font-weight: 600;
            margin: 20px 0 10px;
        }
        
        /* SECCIONES */
        .section {
            margin-bottom: 25px;
        }
        
        .methodology {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #3182ce;
        }
        
        .page-break { page-break-before: always; }
        
        /* TABLAS */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 9pt;
        }
        
        th {
            background: linear-gradient(135deg, #1a365d, #2c5282);
            color: white;
            padding: 10px 8px;
            text-align: center;
            font-weight: 600;
        }
        
        td {
            padding: 8px;
            border: 1px solid #e2e8f0;
            text-align: center;
        }
        
        tr:nth-child(even) { background: #f7fafc; }
        tr:hover { background: #edf2f7; }
        
        .metric-excellent { color: #276749; font-weight: bold; }
        .metric-good { color: #2c7a7b; font-weight: bold; }
        .metric-moderate { color: #c05621; font-weight: bold; }
        .metric-low { color: #c53030; font-weight: bold; }
        
        /* FIGURAS */
        .figure-container {
            text-align: center;
            margin: 20px 0;
            page-break-inside: avoid;
        }
        
        .figure-container img {
            max-width: 100%;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .figure-caption {
            font-size: 9pt;
            color: #718096;
            margin-top: 8px;
            font-style: italic;
        }
        
        .figure-full {
            page-break-inside: avoid;
            margin: 15px 0;
        }
        
        /* CAJAS DE RESUMEN */
        .summary-box {
            background: linear-gradient(135deg, #1a365d, #2c5282);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .summary-box h3 { color: white; margin-top: 0; }
        
        .info-box {
            background: #ebf8ff;
            border: 1px solid #90cdf4;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .warning-box {
            background: #fffaf0;
            border: 1px solid #fbd38d;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .success-box {
            background: #f0fff4;
            border: 1px solid #9ae6b4;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        /* CONCLUSIONES */
        .conclusions {
            page-break-before: always;
        }
        
        .conclusion-item {
            padding: 12px 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid;
        }
        
        .conclusion-confidence {
            background: #ebf8ff;
            border-color: #3182ce;
        }
        
        .conclusion-recommendation {
            background: #f0fff4;
            border-color: #48bb78;
        }
        
        .conclusion-warning {
            background: #fffaf0;
            border-color: #ed8936;
        }
        
        /* ESTADÍSTICAS RÁPIDAS */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 20pt;
            font-weight: 700;
            color: #1a365d;
        }
        
        .stat-label {
            font-size: 9pt;
            color: #718096;
            margin-top: 5px;
        }
        
        /* LISTA DE PARÁMETROS */
        .param-list {
            columns: 2;
            column-gap: 30px;
        }
        
        .param-item {
            break-inside: avoid;
            padding: 10px;
            margin: 5px 0;
            background: #f7fafc;
            border-radius: 6px;
        }
        
        .param-name {
            font-weight: 600;
            color: #2c5282;
        }
        
        .param-value {
            float: right;
            font-weight: 700;
            color: #276749;
        }
    </style>
</head>
<body>
    <!-- PORTADA -->
    <div class="cover">
        <div class="cover-logo">MODELOX</div>
        <h1>ANÁLISIS DE OPTIMIZACIÓN</h1>
        <div class="subtitle">Regresión de Procesos Gaussianos (GPR)</div>
        
        <div class="cover-info">
            <p><strong>📁 Archivo:</strong> {{ filename }}</p>
            <p><strong>📊 Trials:</strong> {{ n_trials | format_number }}</p>
            <p><strong>🔧 Parámetros:</strong> {{ n_params }}</p>
            <p><strong>📈 Métricas:</strong> {{ n_metrics }}</p>
            <p><strong>📅 Fecha:</strong> {{ date }}</p>
        </div>
        
        <div class="cover-quality">
            <div class="badge badge-{{ quality_class }}">
                {{ global_stats.overall_quality }}
            </div>
            <p style="margin-top: 15px; opacity: 0.8;">R² Promedio: {{ "%.4f"|format(global_stats.mean_r2) }}</p>
        </div>
    </div>
    
    <!-- ÍNDICE -->
    <h1>📋 Contenido</h1>
    <div class="section">
        <ol>
            <li>Resumen Ejecutivo</li>
            <li>Metodología</li>
            <li>Dashboard de Calidad del Modelo</li>
            <li>Análisis de Parámetros</li>
            <li>Análisis por Métrica</li>
            <li>Visualizaciones 3D</li>
            <li>Análisis Estadístico</li>
            <li>Conclusiones y Recomendaciones</li>
        </ol>
    </div>
    
    <div class="page-break"></div>
    
    <!-- RESUMEN EJECUTIVO -->
    <h1>1. Resumen Ejecutivo</h1>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{{ "%.3f"|format(global_stats.mean_r2) }}</div>
            <div class="stat-label">R² Promedio</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ "%.1f"|format(global_stats.stability_score * 100) }}%</div>
            <div class="stat-label">Estabilidad</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ "%.1f"|format(global_stats.robustness_score * 100) }}%</div>
            <div class="stat-label">Robustez</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ "%.1f"|format(global_stats.confidence_level * 100) }}%</div>
            <div class="stat-label">Confianza</div>
        </div>
    </div>
    
    <table>
        <tr>
            <th>Métrica</th>
            <th>R²</th>
            <th>Calidad</th>
            <th>Ruido (σn)</th>
        </tr>
        {% for metric, result in results.items() %}
        <tr>
            <td><strong>{{ metric.upper() }}</strong></td>
            <td>{{ "%.4f"|format(result.r2_score) }}</td>
            <td class="{{ 'metric-excellent' if result.r2_score > 0.75 else 'metric-good' if result.r2_score > 0.6 else 'metric-moderate' if result.r2_score > 0.4 else 'metric-low' }}">
                {{ 'EXCELENTE' if result.r2_score > 0.75 else 'BUENO' if result.r2_score > 0.6 else 'MODERADO' if result.r2_score > 0.4 else 'BAJO' }}
            </td>
            <td>{{ "%.4f"|format(result.noise_level) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <!-- METODOLOGÍA -->
    <h1>2. Metodología</h1>
    <div class="methodology">
        <h3>🔬 Regresión de Procesos Gaussianos (GPR)</h3>
        <p>El análisis utiliza <strong>Gaussian Process Regression</strong> para modelar la relación entre 
        los hiperparámetros de la estrategia y las métricas de rendimiento. GPR proporciona no solo 
        predicciones puntuales, sino también <strong>intervalos de confianza</strong> que cuantifican 
        la incertidumbre del modelo.</p>
        
        <h3>📐 Kernel Utilizado</h3>
        <p style="text-align: center; font-size: 12pt; background: white; padding: 10px; border-radius: 5px;">
            <code>K(x, x') = σ² · Matérn(ν=2.5, l) + σn² · I</code>
        </p>
        <ul>
            <li><strong>Matérn (ν=2.5):</strong> Modela la rugosidad de la superficie de respuesta</li>
            <li><strong>Lengthscales (l):</strong> Aprende la escala de variación por parámetro</li>
            <li><strong>WhiteKernel (σn²):</strong> Captura el ruido intrínseco ("desionización")</li>
        </ul>
        
        <h3>📊 Dependencia Parcial</h3>
        <p>La dependencia parcial marginaliza sobre las variables no analizadas:</p>
        <p style="text-align: center; font-size: 12pt; background: white; padding: 10px; border-radius: 5px;">
            <code>f̂(xⱼ) = (1/N) Σᵢ f̂(xⱼ, xᵢ,\j)</code>
        </p>
    </div>
    
    <div class="page-break"></div>
    
    <!-- DASHBOARD DE CALIDAD -->
    <h1>3. Dashboard de Calidad del Modelo</h1>
    {% if figures.quality_dashboard %}
    <div class="figure-full">
        <img src="data:image/png;base64,{{ figures.quality_dashboard }}" alt="Dashboard de Calidad">
        <div class="figure-caption">Figura 1: Dashboard de calidad mostrando métricas globales del modelo GPR.</div>
    </div>
    {% endif %}
    
    <div class="page-break"></div>
    
    <!-- ANÁLISIS DE PARÁMETROS -->
    <h1>4. Análisis de Parámetros</h1>
    
    {% if figures.param_importance %}
    <div class="figure-full">
        <img src="data:image/png;base64,{{ figures.param_importance }}" alt="Importancia de Parámetros">
        <div class="figure-caption">Figura 2: Análisis de importancia, sensibilidad y comportamiento de parámetros.</div>
    </div>
    {% endif %}
    
    <h2>Ranking de Importancia</h2>
    <table>
        <tr>
            <th>#</th>
            <th>Parámetro</th>
            <th>Importancia</th>
            <th>Óptimo</th>
            <th>IC 95%</th>
            <th>Monotonicidad</th>
            <th>No Linealidad</th>
        </tr>
        {% for param in param_analyses %}
        <tr>
            <td>{{ loop.index }}</td>
            <td><strong>{{ param.name.replace('param_', '').replace('_', ' ').title()[:20] }}</strong></td>
            <td>{{ "%.1f"|format(param.importance * 100) }}%</td>
            <td class="metric-excellent">{{ "%.2f"|format(param.optimal_value) }}</td>
            <td>[{{ "%.2f"|format(param.optimal_ci_lower) }} - {{ "%.2f"|format(param.optimal_ci_upper) }}]</td>
            <td>{{ "%.2f"|format(param.monotonicity) }}</td>
            <td>{{ "%.2f"|format(param.nonlinearity) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <div class="page-break"></div>
    
    <!-- ANÁLISIS POR MÉTRICA -->
    <h1>5. Análisis por Métrica</h1>
    
    {% for metric, result in results.items() %}
    <h2>{{ metric.upper() }}</h2>
    
    <div class="info-box">
        <strong>R² = {{ "%.4f"|format(result.r2_score) }}</strong> | 
        Ruido σn = {{ "%.4f"|format(result.noise_level) }} |
        Calidad: {{ 'EXCELENTE' if result.r2_score > 0.75 else 'BUENO' if result.r2_score > 0.6 else 'MODERADO' if result.r2_score > 0.4 else 'BAJO' }}
    </div>
    
    {% set pd_key = 'pd_grid_' + metric %}
    {% if figures.get(pd_key) %}
    <div class="figure-container">
        <img src="data:image/png;base64,{{ figures[pd_key] }}" alt="Dependencia Parcial {{ metric }}">
        <div class="figure-caption">Dependencia parcial con intervalos de confianza al 95% para {{ metric.upper() }}.</div>
    </div>
    {% endif %}
    
    <h3>Valores Óptimos por Parámetro</h3>
    <table>
        <tr>
            <th>Parámetro</th>
            <th>Valor Óptimo</th>
            <th>Predicción</th>
            <th>IC 95%</th>
        </tr>
        {% for param, pred in result.predictions.items() %}
        <tr>
            <td>{{ param.replace('param_', '').replace('_', ' ').title() }}</td>
            <td>{{ "%.3f"|format(optimal[metric][param]['value']) }}</td>
            <td class="metric-good">{{ "%.3f"|format(optimal[metric][param]['pred']) }}</td>
            <td>[{{ "%.3f"|format(optimal[metric][param]['ci_l']) }} - {{ "%.3f"|format(optimal[metric][param]['ci_u']) }}]</td>
        </tr>
        {% endfor %}
    </table>
    
    {% if not loop.last %}
    <div class="page-break"></div>
    {% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- VISUALIZACIONES 3D -->
    <h1>6. Visualizaciones 3D</h1>
    <p>Superficies de respuesta 3D mostrando la interacción entre los dos parámetros más importantes.</p>
    
    {% for metric in results %}
    {% set surface_key = 'surface_3d_' + metric %}
    {% if figures.get(surface_key) %}
    <h2>{{ metric.upper() }}</h2>
    <div class="figure-full">
        <img src="data:image/png;base64,{{ figures[surface_key] }}" alt="Superficie 3D {{ metric }}">
        <div class="figure-caption">Superficie de respuesta 3D, mapa de contorno y mapa de incertidumbre para {{ metric.upper() }}.</div>
    </div>
    {% if not loop.last %}<div class="page-break"></div>{% endif %}
    {% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- ANÁLISIS ESTADÍSTICO -->
    <h1>7. Análisis Estadístico</h1>
    
    {% for metric in results %}
    {% set dist_key = 'distribution_' + metric %}
    {% if figures.get(dist_key) %}
    <h2>{{ metric.upper() }}</h2>
    <div class="figure-full">
        <img src="data:image/png;base64,{{ figures[dist_key] }}" alt="Distribución {{ metric }}">
        <div class="figure-caption">Análisis estadístico completo: distribuciones, Q-Q plot, y análisis de incertidumbre.</div>
    </div>
    {% if not loop.last %}<div class="page-break"></div>{% endif %}
    {% endif %}
    {% endfor %}
    
    <div class="page-break"></div>
    
    <!-- CONCLUSIONES -->
    <div class="conclusions">
        <h1>8. Conclusiones y Recomendaciones</h1>
        
        <h2>📊 Evaluación del Modelo</h2>
        {% for statement in conclusions.confidence_statements %}
        <div class="conclusion-item conclusion-confidence">
            {{ statement }}
        </div>
        {% endfor %}
        
        {% if conclusions.recommendations %}
        <h2>✅ Recomendaciones</h2>
        {% for rec in conclusions.recommendations %}
        <div class="conclusion-item conclusion-recommendation">
            <strong>→</strong> {{ rec }}
        </div>
        {% endfor %}
        {% endif %}
        
        {% if conclusions.warnings %}
        <h2>⚠️ Advertencias</h2>
        {% for warning in conclusions.warnings %}
        <div class="conclusion-item conclusion-warning">
            <strong>⚠</strong> {{ warning }}
        </div>
        {% endfor %}
        {% endif %}
        
        <div class="summary-box">
            <h3>📋 Resumen Final</h3>
            <p>El análisis GPR ha evaluado <strong>{{ n_trials | format_number }} trials</strong> con 
            <strong>{{ n_params }} parámetros</strong> para <strong>{{ n_metrics }} métricas</strong> 
            de rendimiento.</p>
            
            <p>El modelo global presenta una calidad <strong>{{ global_stats.overall_quality }}</strong> 
            con un R² promedio de <strong>{{ "%.3f"|format(global_stats.mean_r2) }}</strong> y un nivel 
            de confianza del <strong>{{ "%.1f"|format(global_stats.confidence_level * 100) }}%</strong>.</p>
            
            {% if param_analyses %}
            <p>El parámetro más influyente es <strong>{{ param_analyses[0].name.replace('param_', '').replace('_', ' ').title() }}</strong> 
            con una importancia del <strong>{{ "%.1f"|format(param_analyses[0].importance * 100) }}%</strong>.</p>
            {% endif %}
        </div>
    </div>
    
    <!-- PIE DE PÁGINA -->
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 2px solid #e2e8f0; color: #718096; font-size: 9pt;">
        <p><strong>MODELOX</strong> - Sistema de Análisis de Trading</p>
        <p>Reporte generado automáticamente el {{ date }}</p>
    </div>
</body>
</html>'''
    
    def generate(self, output_path: str = None) -> str:
        """Genera el PDF profesional."""
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        console.print(Panel("[bold blue]📄 Generando PDF Profesional...[/bold blue]"))
        
        if output_path is None:
            base = Path(self.filepath).stem if self.filepath else "analisis"
            output_path = f"reporte_gpr_profesional_{base}.pdf"
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
        
        # Determinar clase de calidad
        quality_class = 'excellent' if self.conclusions.global_stats.mean_r2 > 0.75 else \
                       'good' if self.conclusions.global_stats.mean_r2 > 0.6 else \
                       'moderate' if self.conclusions.global_stats.mean_r2 > 0.4 else 'low'
        
        # Contexto para template
        context = {
            'filename': Path(self.filepath).name if self.filepath else "N/A",
            'n_trials': len(self.df),
            'n_params': len(self.param_columns),
            'n_metrics': len(self.gpr_results),
            'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'results': self.gpr_results,
            'figures': self.figures,
            'params': self.param_columns,
            'optimal': optimal,
            'global_stats': self.conclusions.global_stats,
            'param_analyses': self.conclusions.param_analyses,
            'conclusions': self.conclusions,
            'quality_class': quality_class,
        }
        
        try:
            from jinja2 import Template, Environment
            
            # Crear environment con filtros personalizados
            env = Environment()
            env.filters['format_number'] = lambda x: f"{x:,}"
            
            template = env.from_string(self._get_professional_template())
            html = template.render(**context)
            
            # Intentar generar PDF
            try:
                from weasyprint import HTML
                HTML(string=html).write_pdf(output_path)
                console.print(f"[green]✅ PDF generado: {output_path}[/green]")
                return output_path
            except ImportError:
                # Guardar como HTML
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                console.print(f"[yellow]⚠ WeasyPrint no disponible. HTML guardado: {html_path}[/yellow]")
                return html_path
                
        except Exception as e:
            console.print(f"[red]❌ Error generando PDF: {e}[/red]")
            import traceback
            traceback.print_exc()
            return ""


# =============================================================================
# FUNCIÓN PRINCIPAL DE INTEGRACIÓN
# =============================================================================

def generate_professional_report(gpr_results: Dict, gpr_models: Dict, df, 
                                  param_columns: List[str], filepath: str = None,
                                  output_path: str = None) -> str:
    """
    Genera un reporte PDF profesional completo.
    
    Args:
        gpr_results: Diccionario con resultados GPR por métrica
        gpr_models: Diccionario con modelos GPR entrenados
        df: DataFrame con los datos de trials
        param_columns: Lista de nombres de columnas de parámetros
        filepath: Ruta del archivo de datos original
        output_path: Ruta de salida para el PDF
    
    Returns:
        Ruta del archivo generado
    """
    generator = ProfessionalPDFGenerator(
        gpr_results=gpr_results,
        gpr_models=gpr_models,
        df=df,
        param_columns=param_columns,
        filepath=filepath
    )
    
    return generator.generate(output_path)

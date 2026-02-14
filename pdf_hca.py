"""
================================================================================
PDF_HCA.PY - Reporte PDF Profesional de Clusters HCA
================================================================================
Genera un PDF de análisis completo a partir de los resultados del HCA:

  • Portada institucional con resumen ejecutivo
  • Tablas de trials agrupados por cluster (métricas + parámetros)
  • Gráficas: cada parámetro vs ROI (scatter + línea de tendencia)
  • Gráficas: cada parámetro vs Profit Factor (scatter + tendencia)
  • Panel de tendencias: todas las métricas clave vs cada parámetro
  • Resumen estadístico por cluster

Diseño profesional nivel institucional, coherente con MODELOX.

Uso:
    from pdf_hca import generar_pdf_hca
    generar_pdf_hca(df, labels, id_cols, metric_cols, param_cols,
                    df_resumen, param_info, output_path, source_file)

Autor: Sistema MODELOX
================================================================================
"""

import os
import io
import base64
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats

from jinja2 import Template

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# CONFIGURACIÓN VISUAL
# ==============================================================================

COLORS = {
    "primary":     "#1e3a5f",
    "secondary":   "#2d5a87",
    "accent":      "#3b82f6",
    "success":     "#059669",
    "warning":     "#d97706",
    "danger":      "#dc2626",
    "text":        "#111827",
    "text_light":  "#6b7280",
    "bg_light":    "#f8fafc",
    "bg_card":     "#ffffff",
    "border":      "#e5e7eb",
    "gold":        "#fbbf24",
}

CLUSTER_PALETTE = [
    "#2E86C1", "#28B463", "#D4AC0D", "#CB4335", "#8E44AD",
    "#E67E22", "#1ABC9C", "#EC7063", "#5DADE2", "#45B39D",
    "#F4D03F", "#AF7AC5", "#EB984E", "#85C1E9", "#82E0AA",
    "#F1948A", "#BB8FCE", "#F0B27A", "#76D7C4", "#AEB6BF",
    "#1F618D", "#196F3D", "#B7950B", "#922B21", "#6C3483",
    "#CA6F1E", "#148F77", "#C0392B", "#2980B9", "#27AE60",
]

def _setup_mpl_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.3,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": COLORS["border"],
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "lines.antialiased": True,
    })

_setup_mpl_style()


# ==============================================================================
# UTILIDADES
# ==============================================================================

def _fig_to_b64(fig, dpi=180) -> str:
    """Convierte una figura matplotlib a base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none", pad_inches=0.03)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _clean_name(name: str) -> str:
    """Limpia nombre de parámetro para display."""
    return name.replace("_", " ").title()[:25]


def _detect_roi_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["ROI_PCT", "ROI%", "ROI"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "ROI" in c.upper():
            return c
    return None


def _detect_pf_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["PROFIT_FACTOR", "PF", "PROFIT FACTOR"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "PROFIT" in c.upper() and "FACTOR" in c.upper():
            return c
    return None


def _detect_score_col(df: pd.DataFrame) -> Optional[str]:
    if "SCORE" in df.columns:
        return "SCORE"
    return None


# ==============================================================================
# GENERADORES DE FIGURAS
# ==============================================================================

def _gen_param_vs_metric_scatter(
    df: pd.DataFrame,
    labels: np.ndarray,
    param_cols: List[str],
    metric_col: str,
    metric_label: str,
) -> str:
    """
    Genera un grid de scatterplots: cada parámetro vs una métrica.
    Coloreado por cluster, con línea de tendencia + R² + intervalo de confianza.
    """
    n_params = len(param_cols)
    if n_params == 0:
        return ""

    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows))

    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    cluster_ids = sorted(set(labels))
    color_map = {cid: CLUSTER_PALETTE[(i) % len(CLUSTER_PALETTE)]
                 for i, cid in enumerate(cluster_ids)}

    for idx, param in enumerate(param_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        x = df[param].values
        y = df[metric_col].values
        c_colors = [color_map[int(l)] for l in labels]

        # Scatter por cluster
        for cid in cluster_ids:
            mask = labels == cid
            ax.scatter(
                x[mask], y[mask],
                c=color_map[cid], s=35, alpha=0.75,
                edgecolors="white", linewidths=0.4,
                label=f"C{cid}", zorder=3,
            )

        # Línea de tendencia global (regresión lineal)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 3:
            xv, yv = x[valid], y[valid]
            slope, intercept, r_val, p_val, _ = sp_stats.linregress(xv, yv)
            x_line = np.linspace(xv.min(), xv.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=COLORS["danger"], linewidth=2.2,
                    linestyle="-", zorder=5, label=f"Tendencia (R²={r_val**2:.3f})")

            # Intervalo de confianza 95%
            n_v = len(xv)
            x_mean = xv.mean()
            se = np.sqrt(np.sum((yv - (slope * xv + intercept))**2) / (n_v - 2))
            t_val = sp_stats.t.ppf(0.975, n_v - 2)
            conf = t_val * se * np.sqrt(1/n_v + (x_line - x_mean)**2 / np.sum((xv - x_mean)**2))
            ax.fill_between(x_line, y_line - conf, y_line + conf,
                            color=COLORS["danger"], alpha=0.10, zorder=2)

            # Anotación R² y pendiente
            r2 = r_val**2
            color_r2 = COLORS["success"] if r2 > 0.3 else COLORS["warning"] if r2 > 0.1 else COLORS["text_light"]
            ax.text(0.97, 0.97, f"R² = {r2:.3f}\nβ = {slope:.4f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    fontweight="bold", color=color_r2,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color_r2, alpha=0.92, linewidth=1.2))

        ax.set_xlabel(_clean_name(param), fontsize=9, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=9, fontweight="bold")
        ax.set_title(f"{_clean_name(param)} vs {metric_label}", fontsize=10, fontweight="bold",
                     color=COLORS["primary"])
        ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    # Leyenda global
    if len(cluster_ids) <= 15:
        handles, lbl = axes[0, 0].get_legend_handles_labels()
        # Solo los clusters, no la tendencia
        cluster_handles = [(h, l) for h, l in zip(handles, lbl) if l.startswith("C")]
        if cluster_handles:
            fig.legend(
                [h for h, _ in cluster_handles],
                [l for _, l in cluster_handles],
                loc="upper center", ncol=min(10, len(cluster_handles)),
                fontsize=7, frameon=True, framealpha=0.9,
                bbox_to_anchor=(0.5, 1.0),
            )

    fig.suptitle(
        f"Parámetros vs {metric_label} — Tendencia por Cluster",
        fontsize=13, fontweight="bold", color=COLORS["primary"], y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _fig_to_b64(fig, dpi=170)


def _gen_trend_panel(
    df: pd.DataFrame,
    labels: np.ndarray,
    param_cols: List[str],
    metric_cols: List[str],
) -> str:
    """
    Panel de líneas de tendencia: para cada parámetro, superpone la tendencia
    lineal de CADA métrica (normalizada) → se ven juntas las direcciones.
    """
    n_params = len(param_cols)
    if n_params == 0:
        return ""

    # Filtrar métricas numéricas con varianza
    valid_metrics = []
    for mc in metric_cols:
        if mc in df.columns and pd.api.types.is_numeric_dtype(df[mc]):
            if df[mc].std() > 1e-9:
                valid_metrics.append(mc)

    if not valid_metrics:
        return ""

    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows))

    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Paleta para métricas
    metric_colors = plt.cm.Set1(np.linspace(0, 1, max(len(valid_metrics), 9)))

    for idx, param in enumerate(param_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        x = df[param].values
        valid_x = np.isfinite(x)

        for mi, mc in enumerate(valid_metrics):
            y = df[mc].values
            valid = valid_x & np.isfinite(y)
            if valid.sum() < 4:
                continue

            xv, yv = x[valid], y[valid]

            # Normalizar métrica a [0, 1] para comparar en mismo eje
            y_min, y_max = yv.min(), yv.max()
            if y_max - y_min < 1e-9:
                continue
            yn = (yv - y_min) / (y_max - y_min)

            slope, intercept, r_val, _, _ = sp_stats.linregress(xv, yn)
            x_line = np.linspace(xv.min(), xv.max(), 200)
            y_line = slope * x_line + intercept

            lw = 2.5 if abs(r_val) > 0.3 else 1.5
            alpha = 0.9 if abs(r_val) > 0.3 else 0.5
            ax.plot(x_line, y_line, color=metric_colors[mi % len(metric_colors)],
                    linewidth=lw, alpha=alpha,
                    label=f"{mc} (R²={r_val**2:.2f})")

        ax.set_xlabel(_clean_name(param), fontsize=9, fontweight="bold")
        ax.set_ylabel("Métrica (normalizada)", fontsize=8)
        ax.set_title(f"Tendencias — {_clean_name(param)}", fontsize=10,
                     fontweight="bold", color=COLORS["primary"])
        ax.set_ylim(-0.15, 1.15)
        ax.axhline(0.5, color=COLORS["border"], linestyle="--", linewidth=0.6, alpha=0.5)
        ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.3)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.9, ncol=1)

    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(
        "Panel de Tendencias — Todas las Métricas por Parámetro",
        fontsize=13, fontweight="bold", color=COLORS["primary"], y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _fig_to_b64(fig, dpi=170)


def _gen_cluster_comparison(
    df: pd.DataFrame,
    labels: np.ndarray,
    metric_cols: List[str],
    df_resumen: pd.DataFrame,
) -> str:
    """
    Gráfica de barras comparativa: métricas clave promedio por cluster.
    """
    # Elegir las métricas más importantes (máx 6)
    priority = ["ROI_PCT", "ROI%", "PROFIT_FACTOR", "SHARPE", "WINRATE_PCT",
                "MAX_DD_PCT", "SCORE", "TOTAL_TRADES", "SQN", "ESTABILIDAD"]
    selected = []
    for p in priority:
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            selected.append(p)
        if len(selected) >= 6:
            break
    if not selected:
        for mc in metric_cols[:6]:
            if mc in df.columns and pd.api.types.is_numeric_dtype(df[mc]):
                selected.append(mc)

    if not selected:
        return ""

    df_work = df.copy()
    df_work["__CL__"] = labels
    cluster_ids = sorted(df_work["__CL__"].unique())

    n_metrics = len(selected)
    n_clusters = len(cluster_ids)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    for mi, mc in enumerate(selected):
        ax = axes[mi]
        means = []
        errs = []
        colors = []
        for cid in cluster_ids:
            sub = df_work[df_work["__CL__"] == cid][mc].dropna()
            means.append(sub.mean())
            errs.append(sub.std() if len(sub) > 1 else 0)
            colors.append(CLUSTER_PALETTE[(cid - 1) % len(CLUSTER_PALETTE)])

        x_pos = np.arange(n_clusters)
        bars = ax.bar(x_pos, means, yerr=errs, color=colors,
                      edgecolor="white", linewidth=0.8, capsize=3, width=0.7,
                      error_kw={"linewidth": 1, "capthick": 1, "ecolor": "#555"})

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"C{cid}" for cid in cluster_ids], fontsize=7, rotation=45)
        ax.set_title(mc.replace("_", " "), fontsize=9, fontweight="bold",
                     color=COLORS["primary"])
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.3)

        # Valor encima de cada barra
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold", color=COLORS["text"])

    fig.suptitle("Comparativa de Métricas por Cluster",
                 fontsize=13, fontweight="bold", color=COLORS["primary"], y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _fig_to_b64(fig, dpi=170)


def _gen_correlation_heatmap(
    df: pd.DataFrame,
    param_cols: List[str],
    metric_cols: List[str],
) -> str:
    """
    Mapa de calor: correlación de Spearman entre parámetros y métricas clave.
    """
    # Seleccionar métricas numéricas
    valid_metrics = []
    for mc in metric_cols:
        if mc in df.columns and pd.api.types.is_numeric_dtype(df[mc]):
            if df[mc].std() > 1e-9:
                valid_metrics.append(mc)

    if not valid_metrics or not param_cols:
        return ""

    # Calcular correlación Spearman
    corr_data = []
    for pc in param_cols:
        row = []
        for mc in valid_metrics:
            try:
                r, _ = sp_stats.spearmanr(df[pc].values, df[mc].values, nan_policy="omit")
                row.append(r if np.isfinite(r) else 0)
            except Exception:
                row.append(0)
        corr_data.append(row)

    corr_matrix = np.array(corr_data)

    fig, ax = plt.subplots(figsize=(max(6, len(valid_metrics) * 1.1), max(4, len(param_cols) * 0.6)))

    cmap = LinearSegmentedColormap.from_list("divergent", [
        "#dc2626", "#f87171", "#fca5a5", "#ffffff", "#86efac", "#22c55e", "#059669"
    ], N=256)

    vmax = max(abs(corr_matrix.max()), abs(corr_matrix.min()), 0.5)
    im = ax.imshow(corr_matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(valid_metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in valid_metrics],
                       fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(param_cols)))
    ax.set_yticklabels([_clean_name(p) for p in param_cols], fontsize=8)

    # Anotaciones
    for i in range(len(param_cols)):
        for j in range(len(valid_metrics)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.45 else COLORS["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Correlación Spearman (ρ)", fontsize=8, fontweight="bold")

    ax.set_title("Correlación Parámetros × Métricas",
                 fontsize=12, fontweight="bold", color=COLORS["primary"], pad=12)
    plt.tight_layout()
    return _fig_to_b64(fig, dpi=170)


# ==============================================================================
# TEMPLATE HTML
# ==============================================================================

_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>HCA Cluster Report — MODELOX</title>
<style>
    @page {
        size: A4 landscape;
        margin: 1.5cm 1.8cm;
        @bottom-center { content: "Página " counter(page) " / " counter(pages); font-size: 8px; color: #6b7280; }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        font-size: 9pt; line-height: 1.5; color: #111827;
    }

    /* ── PORTADA ── */
    .cover {
        page-break-after: always;
        background: linear-gradient(145deg, #1e3a5f 0%, #2d5a87 55%, #3b82f6 100%);
        color: white;
        margin: -1.5cm -1.8cm;
        padding: 2.5cm 3cm;
        min-height: 100vh;
        text-align: center;
    }
    .cover-logo { font-size: 48pt; font-weight: 800; letter-spacing: -3px; margin-bottom: 0.2cm; }
    .cover h1 { font-size: 24pt; font-weight: 300; border: none; margin: 1cm 0 0.3cm; }
    .cover .subtitle { font-size: 12pt; color: rgba(255,255,255,0.7); }
    .cover-info {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 1cm 1.5cm;
        border-radius: 12px;
        margin: 1.2cm auto;
        max-width: 14cm;
        display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5cm;
    }
    .cover-stat {
        flex: 1; min-width: 3.5cm; padding: 0.4cm;
    }
    .cover-stat .val { font-size: 22pt; font-weight: 700; }
    .cover-stat .lbl { font-size: 9pt; opacity: 0.75; }
    .badge {
        display: inline-block; padding: 0.3cm 1cm; border-radius: 2cm;
        font-size: 13pt; font-weight: 700; margin-top: 0.8cm;
        background: rgba(255,255,255,0.15); border: 2px solid rgba(255,255,255,0.3);
    }

    /* ── SECCIONES ── */
    h1 {
        color: #1e3a5f; font-size: 15pt; font-weight: 700;
        border-bottom: 2.5px solid #3b82f6; padding-bottom: 0.25cm;
        margin: 0.7cm 0 0.4cm;
    }
    h2 {
        color: #2d5a87; font-size: 12pt; font-weight: 600;
        border-left: 4px solid #3b82f6; padding-left: 0.4cm;
        margin: 0.5cm 0 0.3cm;
    }
    h3 { color: #374151; font-size: 10pt; margin: 0.3cm 0 0.2cm; }
    .page-break { page-break-before: always; }

    /* ── TABLAS ── */
    table {
        width: 100%; border-collapse: collapse;
        margin: 0.3cm 0; font-size: 8pt;
    }
    th {
        background: linear-gradient(135deg, #1e3a5f, #2d5a87);
        color: white; padding: 0.18cm 0.25cm; font-weight: 600;
        font-size: 7.5pt; white-space: nowrap;
    }
    td {
        padding: 0.14cm 0.2cm; border: 1px solid #e5e7eb;
        text-align: center; font-size: 7.5pt;
    }
    tr:nth-child(even) { background: #f8fafc; }
    .cluster-header td {
        font-weight: 700; font-size: 8.5pt; color: white;
        padding: 0.2cm 0.3cm; text-align: left; border: none;
    }

    .positive { color: #059669; font-weight: 600; }
    .negative { color: #dc2626; font-weight: 600; }

    /* ── FIGURAS ── */
    .figure { text-align: center; margin: 0.4cm 0; page-break-inside: avoid; }
    .figure img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }
    .figure-caption { font-size: 7.5pt; color: #6b7280; margin-top: 0.15cm; font-style: italic; }

    /* ── CAJAS ── */
    .info-box { background: #eff6ff; border: 1px solid #bfdbfe; padding: 0.35cm; border-radius: 6px; margin: 0.3cm 0; font-size: 8.5pt; }
    .summary-box {
        background: linear-gradient(135deg, #1e3a5f, #2d5a87);
        color: white; padding: 0.5cm; border-radius: 8px; margin: 0.4cm 0;
    }
    .summary-box h3 { color: white; margin-top: 0; }

    /* ── STATS GRID ── */
    .stats-grid { display: flex; gap: 0.35cm; margin: 0.4cm 0; flex-wrap: wrap; }
    .stat-card {
        flex: 1; min-width: 3cm; background: #f8fafc;
        border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 0.3cm; text-align: center;
    }
    .stat-value { font-size: 16pt; font-weight: 700; color: #1e3a5f; }
    .stat-label { font-size: 7.5pt; color: #6b7280; }

    /* ── TOLERANCIAS ── */
    .tol-table td { font-size: 7pt; padding: 0.1cm 0.15cm; }
    .tol-table th { font-size: 7pt; padding: 0.12cm 0.15cm; }

    .footer {
        text-align: center; margin-top: 0.8cm; padding-top: 0.4cm;
        border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 7.5pt;
    }
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- PORTADA -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="cover">
    <div class="cover-logo">MODELOX</div>
    <h1>ANÁLISIS HCA — CLUSTERS DE PARÁMETROS</h1>
    <div class="subtitle">Agrupamiento Jerárquico Complete + Seriación Óptima (OLO)</div>

    <div class="cover-info">
        <div class="cover-stat"><div class="val">{{ n_trials }}</div><div class="lbl">Trials Válidos</div></div>
        <div class="cover-stat"><div class="val">{{ n_clusters }}</div><div class="lbl">Clusters</div></div>
        <div class="cover-stat"><div class="val">{{ n_params }}</div><div class="lbl">Parámetros</div></div>
        <div class="cover-stat"><div class="val">{{ n_metrics }}</div><div class="lbl">Métricas</div></div>
        {% if avg_roi is not none %}
        <div class="cover-stat"><div class="val">{{ "%.1f"|format(avg_roi) }}%</div><div class="lbl">ROI Medio</div></div>
        {% endif %}
    </div>

    <div class="badge">{{ strategy_name }}</div>
    <p style="margin-top: 0.5cm; opacity: 0.6; font-size: 9pt;">📁 {{ filename }}<br>📅 {{ date }}</p>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- RESUMEN EJECUTIVO -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<h1>1. Resumen Ejecutivo</h1>

<div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{{ n_trials }}</div><div class="stat-label">Trials (100% ROI ≥ 0)</div></div>
    <div class="stat-card"><div class="stat-value">{{ n_clusters }}</div><div class="stat-label">Clusters Válidos</div></div>
    {% if avg_roi is not none %}
    <div class="stat-card"><div class="stat-value {{ 'positive' if avg_roi > 0 else 'negative' }}">{{ "%.1f"|format(avg_roi) }}%</div><div class="stat-label">ROI Medio</div></div>
    {% endif %}
    {% if avg_pf is not none %}
    <div class="stat-card"><div class="stat-value">{{ "%.2f"|format(avg_pf) }}</div><div class="stat-label">Profit Factor Medio</div></div>
    {% endif %}
    {% if best_cluster_score is not none %}
    <div class="stat-card"><div class="stat-value">{{ "%.1f"|format(best_cluster_score) }}</div><div class="stat-label">Mejor Score (C{{ best_cluster_id }})</div></div>
    {% endif %}
</div>

<div class="info-box">
    <strong>Método:</strong> HCA Complete Linkage + Seriación OLO + Enforcement de Tolerancias<br>
    <strong>Filtros:</strong> Cluster entero eliminado si algún trial tiene ROI &lt; 0 | Mínimo 2 trials por cluster<br>
    <strong>Pesos:</strong> Uniformes (1.0) | <strong>Tolerancia:</strong> Valores base sin reducción
</div>

<!-- Tolerancias -->
{% if tolerancias %}
<h2>Tolerancias de Parámetros</h2>
<table class="tol-table">
    <tr><th>Parámetro</th><th>Tipo</th><th>Rango</th><th>Radio (±tol)</th><th>Diámetro Max</th></tr>
    {% for t in tolerancias %}
    <tr>
        <td><strong>{{ t.name }}</strong></td>
        <td>{{ t.type }}</td>
        <td>{{ "%.2f"|format(t.min) }} — {{ "%.2f"|format(t.max) }}</td>
        <td>{{ "%.2f"|format(t.tol) }}</td>
        <td>{{ "%.2f"|format(t.diam) }}</td>
    </tr>
    {% endfor %}
</table>
{% endif %}

<div class="page-break"></div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- TABLAS POR CLUSTER -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<h1>2. Trials por Cluster</h1>

{% for cluster in clusters %}
<h2 style="border-left-color: {{ cluster.color }};">Cluster {{ cluster.id }} — {{ cluster.n_trials }} trials
{% if cluster.score_medio is not none %} (Score medio: {{ "%.2f"|format(cluster.score_medio) }}){% endif %}</h2>

{% if cluster.param_ranges %}
<div class="info-box" style="font-size: 7.5pt;">
    <strong>Rangos:</strong>
    {% for pr in cluster.param_ranges %}
    {{ pr.name }}: [{{ "%.2f"|format(pr.min) }} – {{ "%.2f"|format(pr.max) }}]{{ "  |  " if not loop.last else "" }}
    {% endfor %}
</div>
{% endif %}

<table>
    <tr>
        {% for col in display_cols %}
        <th>{{ col }}</th>
        {% endfor %}
    </tr>
    {% for trial in cluster.trials %}
    <tr>
        {% for col in display_cols %}
        <td{% if col == roi_col and trial[col] is defined %}
            class="{{ 'positive' if trial[col] >= 0 else 'negative' }}"
        {% endif %}>
            {% if trial[col] is defined %}
                {% if trial[col] is number %}{{ "%.2f"|format(trial[col]) }}{% else %}{{ trial[col] }}{% endif %}
            {% else %}—{% endif %}
        </td>
        {% endfor %}
    </tr>
    {% endfor %}
</table>

{% if not loop.last %}<div style="margin-bottom: 0.3cm;"></div>{% endif %}
{% if loop.index is divisibleby(3) and not loop.last %}<div class="page-break"></div>{% endif %}
{% endfor %}

<div class="page-break"></div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- RESUMEN CLUSTERS -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<h1>3. Resumen Estadístico de Clusters</h1>

<table>
    <tr>
        {% for col in resumen_cols %}
        <th>{{ col }}</th>
        {% endfor %}
    </tr>
    {% for _, row in df_resumen.iterrows() %}
    <tr>
        {% for col in resumen_cols %}
        <td>{% if row[col] is number %}{{ "%.2f"|format(row[col]) }}{% else %}{{ row[col] }}{% endif %}</td>
        {% endfor %}
    </tr>
    {% endfor %}
</table>

<div class="page-break"></div>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- GRÁFICAS: PARÁMETROS vs ROI -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
{% if fig_param_roi %}
<h1>4. Parámetros vs ROI</h1>
<p style="font-size: 8.5pt; color: #6b7280; margin-bottom: 0.3cm;">
    Cada gráfica muestra el valor del parámetro (eje X) contra el ROI (eje Y).
    La línea roja es la tendencia lineal con su intervalo de confianza al 95%.
    R² indica qué proporción de la variabilidad del ROI explica ese parámetro.
</p>
<div class="figure">
    <img src="data:image/png;base64,{{ fig_param_roi }}" alt="Params vs ROI">
    <div class="figure-caption">Fig. 1: Scatter de cada parámetro vs ROI con regresión lineal y IC 95%</div>
</div>
<div class="page-break"></div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- GRÁFICAS: PARÁMETROS vs PROFIT FACTOR -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
{% if fig_param_pf %}
<h1>5. Parámetros vs Profit Factor</h1>
<p style="font-size: 8.5pt; color: #6b7280; margin-bottom: 0.3cm;">
    Misma lógica: eje X = parámetro, eje Y = Profit Factor.
    Un Profit Factor &gt; 1 indica que las ganancias superan a las pérdidas.
</p>
<div class="figure">
    <img src="data:image/png;base64,{{ fig_param_pf }}" alt="Params vs PF">
    <div class="figure-caption">Fig. 2: Scatter de cada parámetro vs Profit Factor con regresión lineal y IC 95%</div>
</div>
<div class="page-break"></div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- PANEL DE TENDENCIAS -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
{% if fig_trends %}
<h1>6. Panel de Tendencias</h1>
<p style="font-size: 8.5pt; color: #6b7280; margin-bottom: 0.3cm;">
    Para cada parámetro, se superponen las líneas de tendencia de todas las métricas
    (normalizadas a [0,1]). Permite ver de un vistazo la dirección de influencia
    de cada parámetro sobre el rendimiento global.
</p>
<div class="figure">
    <img src="data:image/png;base64,{{ fig_trends }}" alt="Trend Panel">
    <div class="figure-caption">Fig. 3: Líneas de tendencia de todas las métricas por parámetro (normalizadas)</div>
</div>
<div class="page-break"></div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- COMPARATIVA DE CLUSTERS -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
{% if fig_cluster_comp %}
<h1>7. Comparativa entre Clusters</h1>
<p style="font-size: 8.5pt; color: #6b7280; margin-bottom: 0.3cm;">
    Barras con media ± desviación estándar de las métricas principales por cluster.
</p>
<div class="figure">
    <img src="data:image/png;base64,{{ fig_cluster_comp }}" alt="Cluster Comparison">
    <div class="figure-caption">Fig. 4: Métricas promedio ± σ por cluster</div>
</div>
<div class="page-break"></div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- CORRELACIÓN -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
{% if fig_corr %}
<h1>8. Correlaciones Parámetro × Métrica</h1>
<p style="font-size: 8.5pt; color: #6b7280; margin-bottom: 0.3cm;">
    Mapa de calor con correlación de Spearman (ρ) entre cada parámetro y cada métrica.
    Verde = correlación positiva, Rojo = negativa. Valores cercanos a ±1 indican relación fuerte.
</p>
<div class="figure">
    <img src="data:image/png;base64,{{ fig_corr }}" alt="Correlación">
    <div class="figure-caption">Fig. 5: Correlación de Spearman entre parámetros y métricas</div>
</div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- PIE -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="footer">
    <strong>MODELOX</strong> — Sistema de Análisis y Optimización de Trading<br>
    Reporte HCA generado el {{ date }}
</div>

</body>
</html>'''


# ==============================================================================
# GENERADOR PRINCIPAL
# ==============================================================================

def generar_pdf_hca(
    df: pd.DataFrame,
    labels: np.ndarray,
    id_cols: List[str],
    metric_cols: List[str],
    param_cols: List[str],
    df_resumen: pd.DataFrame,
    param_info: Dict[str, Dict],
    output_path: str,
    source_file: str = "",
):
    """
    Genera el PDF completo de análisis HCA.

    Parámetros:
        df:             DataFrame filtrado (solo trials válidos)
        labels:         Array de cluster IDs (ya filtrados y renumerados)
        id_cols:        Columnas de identificación (TRIAL, SCORE, etc.)
        metric_cols:    Columnas de métricas (ROI, PF, etc.)
        param_cols:     Columnas de parámetros de la estrategia
        df_resumen:     DataFrame de resumen por cluster
        param_info:     Diccionario con info de cada parámetro
        output_path:    Ruta de salida del PDF
        source_file:    Nombre del archivo fuente
    """
    print(f"\n  📄 Generando PDF profesional...")

    # ── Detectar columnas clave ──
    roi_col = _detect_roi_col(df)
    pf_col = _detect_pf_col(df)
    score_col = _detect_score_col(df)

    # ── Estadísticas globales ──
    n_trials = len(df)
    n_clusters = len(set(labels))
    n_params = len(param_cols)
    n_metrics = len(metric_cols)

    avg_roi = float(df[roi_col].mean()) if roi_col and roi_col in df.columns else None
    avg_pf = float(df[pf_col].mean()) if pf_col and pf_col in df.columns else None

    best_cluster_id = None
    best_cluster_score = None
    if "SCORE_MEDIO" in df_resumen.columns and len(df_resumen) > 0:
        best_row = df_resumen.iloc[0]
        best_cluster_id = int(best_row["CLUSTER"])
        best_cluster_score = float(best_row["SCORE_MEDIO"])

    # Detectar estrategia
    strategy_name = "ESTRATEGIA"
    for c in ["ESTRATEGIA", "STRATEGY"]:
        if c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) > 0:
                strategy_name = str(vals[0])
            break

    # ── Tolerancias ──
    tolerancias = []
    for col in param_cols:
        info = param_info.get(col, {})
        tol = info.get("neighbor_tolerance", 0)
        if tol > 0:
            tolerancias.append({
                "name": col,
                "type": info.get("type", "?"),
                "min": info.get("min", 0),
                "max": info.get("max", 0),
                "tol": tol,
                "diam": tol * 2,
            })

    # ── Construir datos de clusters ──
    df_work = df.copy()
    df_work["__CL__"] = labels

    # Columnas a mostrar en tablas
    display_cols = []
    for c in id_cols:
        if c in df.columns:
            display_cols.append(c)
    for c in metric_cols:
        if c in df.columns:
            display_cols.append(c)
    for c in param_cols:
        if c in df.columns:
            display_cols.append(c)

    # Orden de clusters: por score medio descendente (como en df_resumen)
    clusters_data = []
    for _, res_row in df_resumen.iterrows():
        cid = int(res_row["CLUSTER"])
        sub = df_work[df_work["__CL__"] == cid]
        color_idx = len(clusters_data) % len(CLUSTER_PALETTE)

        score_medio = None
        if "SCORE_MEDIO" in res_row:
            score_medio = float(res_row["SCORE_MEDIO"])

        # Rangos de parámetros
        param_ranges = []
        for pc in param_cols:
            if pc in sub.columns:
                param_ranges.append({
                    "name": pc,
                    "min": float(sub[pc].min()),
                    "max": float(sub[pc].max()),
                })

        # Trials como dicts
        trials = []
        for _, trial_row in sub.iterrows():
            trial_dict = {}
            for c in display_cols:
                if c in trial_row:
                    val = trial_row[c]
                    if pd.notna(val):
                        trial_dict[c] = val
            trials.append(trial_dict)

        clusters_data.append({
            "id": cid,
            "n_trials": len(sub),
            "color": CLUSTER_PALETTE[color_idx],
            "score_medio": score_medio,
            "param_ranges": param_ranges,
            "trials": trials,
        })

    # ── Resumen cols ──
    resumen_cols = [c for c in df_resumen.columns]

    # ── Generar figuras ──
    print(f"     • Generando gráficas...")

    fig_param_roi = ""
    if roi_col and roi_col in df.columns:
        print(f"       - Parámetros vs {roi_col}...")
        fig_param_roi = _gen_param_vs_metric_scatter(
            df, labels, param_cols, roi_col, roi_col.replace("_", " "))

    fig_param_pf = ""
    if pf_col and pf_col in df.columns:
        print(f"       - Parámetros vs {pf_col}...")
        fig_param_pf = _gen_param_vs_metric_scatter(
            df, labels, param_cols, pf_col, pf_col.replace("_", " "))

    print(f"       - Panel de tendencias...")
    fig_trends = _gen_trend_panel(df, labels, param_cols, metric_cols)

    print(f"       - Comparativa de clusters...")
    fig_cluster_comp = _gen_cluster_comparison(df, labels, metric_cols, df_resumen)

    print(f"       - Mapa de correlación...")
    fig_corr = _gen_correlation_heatmap(df, param_cols, metric_cols)

    # ── Renderizar HTML ──
    print(f"     • Renderizando HTML...")
    context = {
        "n_trials": n_trials,
        "n_clusters": n_clusters,
        "n_params": n_params,
        "n_metrics": n_metrics,
        "avg_roi": avg_roi,
        "avg_pf": avg_pf,
        "best_cluster_id": best_cluster_id,
        "best_cluster_score": best_cluster_score,
        "strategy_name": strategy_name,
        "filename": os.path.basename(source_file) if source_file else "N/A",
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "tolerancias": tolerancias,
        "clusters": clusters_data,
        "display_cols": display_cols,
        "roi_col": roi_col or "",
        "resumen_cols": resumen_cols,
        "df_resumen": df_resumen,
        "fig_param_roi": fig_param_roi,
        "fig_param_pf": fig_param_pf,
        "fig_trends": fig_trends,
        "fig_cluster_comp": fig_cluster_comp,
        "fig_corr": fig_corr,
    }

    html = Template(_HTML_TEMPLATE).render(**context)

    # ── Generar PDF ──
    print(f"     • Generando PDF...")
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html).write_pdf(output_path)
        print(f"\n  ✅ PDF generado: {os.path.basename(output_path)}")
        return output_path
    except ImportError:
        # Fallback: guardar como HTML
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  ⚠️  WeasyPrint no disponible. HTML guardado: {html_path}")
        return html_path
    except Exception as e:
        # Fallback: guardar HTML + reportar error
        html_path = output_path.replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  ⚠️  Error generando PDF: {e}")
        print(f"     HTML guardado como fallback: {html_path}")
        return html_path

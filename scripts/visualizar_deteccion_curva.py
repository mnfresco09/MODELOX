#!/usr/bin/env python3
"""
Visualización de Detección de Parámetro Óptimo - Algoritmo Percentil 80.

ALGORITMO ANTI-OVERFITTING:
1. Filtrar trials con ROI > 0 (solo rentables)
2. Filtrar trials con trades/día >= 0.25 (suficiente actividad)
3. Ordenar por SQN y tomar percentil 80 (top 20%)
4. Calcular MEDIANA del parámetro del top

Ejecutar:
    python scripts/visualizar_deteccion_curva.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración visual
plt.style.use('dark_background')
COLORS = {
    'all_points': '#555555',     # Gris - todos los puntos
    'filtered_roi': '#8B0000',   # Rojo oscuro - filtrados ROI
    'filtered_tpd': '#FF4500',   # Naranja - filtrados TPD
    'valid': '#4ECDC4',          # Teal - válidos
    'top20': '#FFE66D',          # Amarillo - top 20%
    'median': '#FF6B6B',         # Rojo - mediana
    'grid': '#333333',
}

# Archivo Excel con datos reales
EXCEL_FILE = Path(__file__).parent.parent / "RESUMEN_UNKNOWN_LASTTRY_unknown.xlsx 19-49-15-950.xlsx"


def cargar_datos_excel(param_name: str = "ZLEMA_FAST_LEN") -> dict:
    """
    Carga datos reales del archivo Excel de resultados.
    
    Args:
        param_name: Nombre del parámetro a analizar (columna del Excel)
    
    Returns:
        dict con params, sqn, roi, tpd
    """
    df = pd.read_excel(EXCEL_FILE, header=1)
    
    # Filtrar filas válidas (sin NaN en las columnas clave)
    df = df.dropna(subset=[param_name, 'SQN', 'ROI_PCT', 'TRADES_DIA'])
    
    return {
        'params': df[param_name].values,
        'sqn': df['SQN'].values,
        'roi': df['ROI_PCT'].values,
        'tpd': df['TRADES_DIA'].values,
        'param_name': param_name,
        'n_total': len(df),
    }


def aplicar_filtros(data: dict, min_tpd: float = 0.22) -> dict:
    """
    Aplica los filtros de calidad y devuelve máscaras.
    """
    params = data['params']
    sqn = data['sqn']
    roi = data['roi']
    tpd = data['tpd']
    
    # Filtros
    mask_roi = roi > 0
    mask_tpd = tpd >= min_tpd
    
    # Combinaciones
    mask_valid = mask_roi & mask_tpd
    
    # Filtrados por cada razón
    filtered_roi = ~mask_roi
    filtered_tpd = mask_roi & ~mask_tpd
    
    return {
        'mask_valid': mask_valid,
        'filtered_roi': filtered_roi,
        'filtered_tpd': filtered_tpd,
        'n_filtered_roi': np.sum(filtered_roi),
        'n_filtered_tpd': np.sum(filtered_tpd),
        'n_valid': np.sum(mask_valid),
    }


def calcular_percentil80(data: dict, masks: dict) -> dict:
    """
    Calcula el percentil 80 (top 20%) de los válidos y la mediana.
    """
    params = data['params'][masks['mask_valid']]
    sqn = data['sqn'][masks['mask_valid']]
    
    if len(params) < 5:
        return None
    
    # Ordenar por SQN descendente
    sorted_indices = np.argsort(sqn)[::-1]
    sorted_params = params[sorted_indices]
    sorted_sqn = sqn[sorted_indices]
    
    # Top 20%
    n_top = max(1, int(len(sorted_params) * 0.20))
    top_params = sorted_params[:n_top]
    top_sqn = sorted_sqn[:n_top]
    
    # Mediana
    median = np.median(top_params)
    
    return {
        'top_params': top_params,
        'top_sqn': top_sqn,
        'n_top': n_top,
        'median': median,
        'median_sqn': np.interp(median, data['params'], data['sqn']),
    }


def crear_visualizacion(param_name: str = "ZLEMA_FAST_LEN", output_path: str = "deteccion_curva.png"):
    """Genera el PNG con la visualización completa usando datos reales."""
    
    # Cargar datos reales del Excel
    print(f"📂 Cargando datos de: {EXCEL_FILE.name}")
    data = cargar_datos_excel(param_name=param_name)
    print(f"   • Total trials en Excel: {data['n_total']}")
    
    # Aplicar filtros
    masks = aplicar_filtros(data, min_tpd=0.22)
    
    # Calcular percentil 80
    result = calcular_percentil80(data, masks)
    
    if result is None:
        print("❌ No hay suficientes datos válidos")
        return
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    
    param_label = data.get('param_name', 'Parámetro')
    
    # ═══════════════════════════════════════════════════════════════════
    # SUBPLOT 1: Parámetro vs SQN con filtros
    # ═══════════════════════════════════════════════════════════════════
    
    # Puntos filtrados por ROI
    ax1.scatter(data['params'][masks['filtered_roi']], 
                data['sqn'][masks['filtered_roi']],
                color=COLORS['filtered_roi'], s=20, alpha=0.4, 
                label=f'ROI ≤ 0 ({masks["n_filtered_roi"]})')
    
    # Puntos filtrados por trades/día
    ax1.scatter(data['params'][masks['filtered_tpd']], 
                data['sqn'][masks['filtered_tpd']],
                color=COLORS['filtered_tpd'], s=20, alpha=0.4, 
                label=f'TPD < 0.22 ({masks["n_filtered_tpd"]})')
    
    # Puntos válidos
    valid_params = data['params'][masks['mask_valid']]
    valid_sqn = data['sqn'][masks['mask_valid']]
    ax1.scatter(valid_params, valid_sqn,
                color=COLORS['valid'], s=40, alpha=0.7, 
                label=f'Válidos ({masks["n_valid"]})')
    
    ax1.set_xlabel(f'{param_label}', fontsize=12, color='white')
    ax1.set_ylabel('SQN', fontsize=12, color='white')
    ax1.set_title('PASO 1-2: Aplicar Filtros de Calidad\n'
                  'ROI > 0, TPD ≥ 0.22',
                  fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # ═══════════════════════════════════════════════════════════════════
    # SUBPLOT 2: Percentil 80 y Mediana
    # ═══════════════════════════════════════════════════════════════════
    
    # Todos los válidos (más tenues)
    ax2.scatter(valid_params, valid_sqn,
                color=COLORS['valid'], s=30, alpha=0.3, 
                label=f'Válidos ({masks["n_valid"]})')
    
    # Top 20% (percentil 80)
    ax2.scatter(result['top_params'], result['top_sqn'],
                color=COLORS['top20'], s=80, alpha=0.9, 
                edgecolors='white', linewidths=1,
                label=f'Top 20% ({result["n_top"]} trials)')
    
    # Mediana (estrella grande)
    ax2.scatter([result['median']], [result['median_sqn']], 
                color=COLORS['median'], s=400, zorder=20,
                marker='*', edgecolors='white', linewidths=2,
                label=f'MEDIANA = {result["median"]:.1f}')
    
    # Línea vertical en la mediana
    ax2.axvline(x=result['median'], color=COLORS['median'], 
                linestyle=':', linewidth=2, alpha=0.8)
    
    # Anotación
    offset_x = (np.max(data['params']) - np.min(data['params'])) * 0.08
    offset_y = (np.max(valid_sqn) - np.min(valid_sqn)) * 0.1 if len(valid_sqn) > 0 else 1
    ax2.annotate(f'MEDIANA\n{result["median"]:.1f}',
                xy=(result['median'], result['median_sqn']),
                xytext=(result['median'] + offset_x, result['median_sqn'] + offset_y),
                fontsize=14, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Rango del top 20%
    min_top = np.min(result['top_params'])
    max_top = np.max(result['top_params'])
    ax2.axvspan(min_top, max_top, alpha=0.15, color=COLORS['top20'],
                label=f'Rango top: [{min_top:.0f} - {max_top:.0f}]')
    
    ax2.set_xlabel(f'{param_label}', fontsize=12, color='white')
    ax2.set_ylabel('SQN', fontsize=12, color='white')
    ax2.set_title('PASO 3-4: Percentil 80 por SQN → Mediana\n'
                  'Ordenar por SQN, tomar top 20%, calcular mediana',
                  fontsize=14, color='white', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # ═══════════════════════════════════════════════════════════════════
    # BOX DE ESTADÍSTICAS
    # ═══════════════════════════════════════════════════════════════════
    stats_text = (
        f"━━━ {param_label} ━━━\n"
        f"Total trials: {len(data['params'])}\n"
        f"Filtrados ROI ≤ 0: {masks['n_filtered_roi']}\n"
        f"Filtrados TPD < 0.22: {masks['n_filtered_tpd']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Válidos: {masks['n_valid']}\n"
        f"Top 20%: {result['n_top']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"MEDIANA: {result['median']:.1f}\n"
        f"Rango top: [{min_top:.0f} - {max_top:.0f}]\n"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', 
                 edgecolor=COLORS['median'], alpha=0.95)
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=props, color='white')
    
    # Título general
    fig.suptitle(f'Algoritmo Percentil 80 - Parámetro: {param_label}',
                 fontsize=18, color='white', fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(output_path)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='#0f0f0f', edgecolor='none')
    plt.close()
    
    print(f"✅ Gráfico guardado en: {output_file.absolute()}")
    print(f"\n📊 Resumen para {param_label}:")
    print(f"   • Total trials: {len(data['params'])}")
    print(f"   • Filtrados: {masks['n_filtered_roi'] + masks['n_filtered_tpd']}")
    print(f"   • Válidos: {masks['n_valid']}")
    print(f"   • Top 20%: {result['n_top']} trials")
    print(f"   • MEDIANA: {result['median']:.1f}")
    print(f"   • Rango top: [{min_top:.0f} - {max_top:.0f}]")


if __name__ == "__main__":
    # Parámetros disponibles: EXIT_SL%, EXIT_TP%, EXIT_TRAIL_ACT%, EXIT_TRAIL_DIST%, 
    # LOOKBAR, REQ_DIST%, ZLEMA_FAST_LEN, ZLEMA_SLOW_LEN
    crear_visualizacion(param_name="ZLEMA_FAST_LEN", output_path="deteccion_curva.png")

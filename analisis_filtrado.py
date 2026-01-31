#!/usr/bin/env python3
"""
# =============================================================================
#
#     ███████╗██╗██╗  ████████╗██████╗  █████╗ ██████╗  ██████╗ 
#     ██╔════╝██║██║  ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗
#     █████╗  ██║██║     ██║   ██████╔╝███████║██║  ██║██║   ██║
#     ██╔══╝  ██║██║     ██║   ██╔══██╗██╔══██║██║  ██║██║   ██║
#     ██║     ██║███████╗██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝
#     ╚═╝     ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ 
#
#     ANALISIS_FILTRADO.PY - ANÁLISIS GPR CON FILTRADO INTELIGENTE
#
# =============================================================================
#
#     FLUJO:
#     1. Cargar Excel/CSV con resultados de optimización
#     2. FILTRAR: Elimina ROI < 0 y Score < Media
#     3. Análisis GPR (Gaussian Process Regression)
#     4. Generación de reportes (Excel, PDF)
#
# =============================================================================
"""

from __future__ import annotations

import os
import sys
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich import box
from rich.traceback import install as install_rich_traceback

install_rich_traceback(show_locals=False)
warnings.filterwarnings('ignore')

# Inicializar consola
console = Console()


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class FilterConfig:
    """Configuración de filtrado."""
    min_roi: float = 0.0              # ROI mínimo (eliminar negativos)
    score_percentile: float = 50.0    # Percentil de score para filtrar (50 = media)
    min_trials_after_filter: int = 50 # Mínimo de trials después del filtrado


# =============================================================================
# INTERFAZ PROFESIONAL
# =============================================================================

class ProfessionalUI:
    """Sistema de interfaz profesional."""
    
    @staticmethod
    def banner() -> Panel:
        """Banner principal."""
        banner_text = """
 ███████╗██╗██╗  ████████╗██████╗  █████╗ ██████╗  ██████╗ 
 ██╔════╝██║██║  ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗
 █████╗  ██║██║     ██║   ██████╔╝███████║██║  ██║██║   ██║
 ██╔══╝  ██║██║     ██║   ██╔══██╗██╔══██║██║  ██║██║   ██║
 ██║     ██║███████╗██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝
 ╚═╝     ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ 
        """
        return Panel(
            Text(banner_text, style="bold cyan", justify="center"),
            title="[bold white]ANÁLISIS GPR CON FILTRADO INTELIGENTE[/bold white]",
            subtitle="[grey50]v1.0 - MODELOX[/grey50]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(0, 2)
        )
    
    @staticmethod
    def section(title: str) -> Rule:
        """Separador de sección."""
        return Rule(f"[bold white]{title.upper()}[/bold white]", style="grey50")
    
    @staticmethod
    def stats_table(stats: Dict[str, Any], title: str = "ESTADÍSTICAS") -> Table:
        """Tabla de estadísticas."""
        table = Table(
            title=f"[bold white]{title}[/bold white]",
            box=box.ROUNDED,
            border_style="grey50",
            show_header=True,
            header_style="bold white"
        )
        table.add_column("MÉTRICA", style="grey70", justify="right")
        table.add_column("VALOR", style="white", justify="center")
        
        for key, value in stats.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:,.4f}")
            elif isinstance(value, int):
                table.add_row(key, f"{value:,}")
            else:
                table.add_row(key, str(value))
        
        return table
    
    @staticmethod
    def filter_summary(before: int, after: int, removed_roi: int, removed_score: int) -> Panel:
        """Resumen del filtrado."""
        pct_kept = (after / before * 100) if before > 0 else 0
        
        content = Text()
        content.append("FILTRADO COMPLETADO\n\n", style="bold green")
        content.append(f"  TRIALS ORIGINALES:    ", style="grey70")
        content.append(f"{before:,}\n", style="white")
        content.append(f"  ELIMINADOS (ROI<0):   ", style="grey70")
        content.append(f"{removed_roi:,}\n", style="red")
        content.append(f"  ELIMINADOS (SCORE):   ", style="grey70")
        content.append(f"{removed_score:,}\n", style="yellow")
        content.append(f"  TRIALS FINALES:       ", style="grey70")
        content.append(f"{after:,} ", style="bold green")
        content.append(f"({pct_kept:.1f}%)", style="grey50")
        
        return Panel(
            content,
            title="[bold white]◆ RESUMEN FILTRADO[/bold white]",
            border_style="green",
            box=box.ROUNDED
        )


UI = ProfessionalUI()


# =============================================================================
# CARGADOR DE DATOS
# =============================================================================

class DataLoader:
    """Carga y valida archivos Excel/CSV."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filepath: str = ""
        self.param_columns: List[str] = []
        self.metric_columns: List[str] = []
        self.score_column: Optional[str] = None
        self.roi_column: Optional[str] = None
    
    def load(self, filepath: str) -> bool:
        """Carga archivo Excel o CSV."""
        filepath = filepath.strip().strip('"').strip("'")
        
        if not os.path.exists(filepath):
            console.print(f"[red]  ❌ ARCHIVO NO ENCONTRADO: {filepath}[/red]")
            return False
        
        try:
            ext = Path(filepath).suffix.lower()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Cargando archivo...", total=None)
                
                if ext in ['.xlsx', '.xls']:
                    self.df = pd.read_excel(filepath)
                elif ext == '.csv':
                    self.df = pd.read_csv(filepath)
                elif ext == '.feather':
                    self.df = pd.read_feather(filepath)
                elif ext == '.parquet':
                    self.df = pd.read_parquet(filepath)
                else:
                    console.print(f"[red]  ❌ FORMATO NO SOPORTADO: {ext}[/red]")
                    return False
                
                progress.update(task, completed=True)
            
            self.filepath = filepath
            console.print(f"[green]  ✓ CARGADO: {Path(filepath).name}[/green]")
            console.print(f"[grey70]    FILAS: {len(self.df):,} | COLUMNAS: {len(self.df.columns)}[/grey70]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]  ❌ ERROR: {e}[/red]")
            return False
    
    def detect_columns(self) -> bool:
        """Detecta columnas de parámetros, métricas, score y ROI."""
        if self.df is None:
            return False
        
        cols = self.df.columns.tolist()
        cols_lower = [c.lower() for c in cols]
        
        # Detectar columnas de parámetros (empiezan con param_)
        self.param_columns = [c for c in cols if c.lower().startswith('param_')]
        
        # Detectar columna de score
        score_candidates = ['score', 'valor', 'value', 'fitness', 'objective']
        for candidate in score_candidates:
            matches = [c for c in cols if candidate in c.lower()]
            if matches:
                self.score_column = matches[0]
                break
        
        # Detectar columna de ROI
        roi_candidates = ['roi', 'return', 'profit', 'pnl', 'gain']
        for candidate in roi_candidates:
            matches = [c for c in cols if candidate in c.lower() and 'max' not in c.lower()]
            if matches:
                self.roi_column = matches[0]
                break
        
        # Detectar métricas numéricas (excluyendo params)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.metric_columns = [
            c for c in numeric_cols 
            if c not in self.param_columns 
            and c not in ['number', 'trial', 'iteration', 'index', 'Unnamed: 0']
        ]
        
        # Mostrar detección
        console.print(f"\n[bold white]  COLUMNAS DETECTADAS:[/bold white]")
        console.print(f"[grey70]    PARÁMETROS: {len(self.param_columns)}[/grey70]")
        for p in self.param_columns[:5]:
            console.print(f"[grey50]      - {p}[/grey50]")
        if len(self.param_columns) > 5:
            console.print(f"[grey50]      ... y {len(self.param_columns)-5} más[/grey50]")
        
        console.print(f"[grey70]    SCORE: {self.score_column or 'NO DETECTADO'}[/grey70]")
        console.print(f"[grey70]    ROI: {self.roi_column or 'NO DETECTADO'}[/grey70]")
        console.print(f"[grey70]    MÉTRICAS: {len(self.metric_columns)}[/grey70]")
        
        if not self.param_columns:
            console.print("[red]  ❌ NO SE DETECTARON COLUMNAS DE PARÁMETROS (param_*)[/red]")
            return False
        
        return True
    
    def show_columns(self):
        """Muestra todas las columnas para selección manual."""
        table = Table(title="[bold white]COLUMNAS DISPONIBLES[/bold white]", box=box.ROUNDED)
        table.add_column("#", style="grey50", justify="right")
        table.add_column("COLUMNA", style="white")
        table.add_column("TIPO", style="grey70")
        table.add_column("EJEMPLO", style="grey50")
        
        for i, col in enumerate(self.df.columns):
            dtype = str(self.df[col].dtype)
            example = str(self.df[col].iloc[0])[:30] if len(self.df) > 0 else "N/A"
            table.add_row(str(i), col, dtype, example)
        
        console.print(table)


# =============================================================================
# FILTRO INTELIGENTE
# =============================================================================

class IntelligentFilter:
    """Filtra datos por ROI y Score."""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.stats_before: Dict[str, Any] = {}
        self.stats_after: Dict[str, Any] = {}
        self.removed_roi: int = 0
        self.removed_score: int = 0
    
    def apply(self, df: pd.DataFrame, roi_col: str = None, score_col: str = None) -> pd.DataFrame:
        """Aplica filtros al DataFrame."""
        original_len = len(df)
        self.stats_before = self._compute_stats(df, roi_col, score_col)
        
        console.print(f"\n[bold white]  APLICANDO FILTROS:[/bold white]")
        
        # Paso 1: Filtrar ROI < 0
        if roi_col and roi_col in df.columns:
            before_roi = len(df)
            df = df[df[roi_col] >= self.config.min_roi].copy()
            self.removed_roi = before_roi - len(df)
            console.print(f"[grey70]    1. ROI >= {self.config.min_roi}: [/grey70]", end="")
            console.print(f"[red]-{self.removed_roi:,}[/red] [grey50]eliminados[/grey50]")
        else:
            console.print(f"[yellow]    ⚠ COLUMNA ROI NO ENCONTRADA - SALTANDO FILTRO ROI[/yellow]")
        
        # Paso 2: Filtrar Score < Media
        if score_col and score_col in df.columns:
            before_score = len(df)
            score_threshold = df[score_col].quantile(self.config.score_percentile / 100)
            df = df[df[score_col] >= score_threshold].copy()
            self.removed_score = before_score - len(df)
            console.print(f"[grey70]    2. SCORE >= {score_threshold:.4f} (P{self.config.score_percentile:.0f}): [/grey70]", end="")
            console.print(f"[yellow]-{self.removed_score:,}[/yellow] [grey50]eliminados[/grey50]")
        else:
            console.print(f"[yellow]    ⚠ COLUMNA SCORE NO ENCONTRADA - SALTANDO FILTRO SCORE[/yellow]")
        
        self.stats_after = self._compute_stats(df, roi_col, score_col)
        
        # Verificar mínimo de trials
        if len(df) < self.config.min_trials_after_filter:
            console.print(f"[yellow]  ⚠ POCOS TRIALS ({len(df)}). Mínimo recomendado: {self.config.min_trials_after_filter}[/yellow]")
        
        return df
    
    def _compute_stats(self, df: pd.DataFrame, roi_col: str, score_col: str) -> Dict[str, Any]:
        """Calcula estadísticas del DataFrame."""
        stats = {'TRIALS': len(df)}
        
        if roi_col and roi_col in df.columns:
            stats['ROI_MEAN'] = df[roi_col].mean()
            stats['ROI_STD'] = df[roi_col].std()
            stats['ROI_MIN'] = df[roi_col].min()
            stats['ROI_MAX'] = df[roi_col].max()
        
        if score_col and score_col in df.columns:
            stats['SCORE_MEAN'] = df[score_col].mean()
            stats['SCORE_STD'] = df[score_col].std()
            stats['SCORE_MIN'] = df[score_col].min()
            stats['SCORE_MAX'] = df[score_col].max()
        
        return stats
    
    def show_comparison(self):
        """Muestra comparación antes/después."""
        table = Table(
            title="[bold white]COMPARACIÓN ANTES/DESPUÉS[/bold white]",
            box=box.ROUNDED,
            border_style="grey50"
        )
        table.add_column("MÉTRICA", style="grey70", justify="right")
        table.add_column("ANTES", style="white", justify="center")
        table.add_column("DESPUÉS", style="green", justify="center")
        table.add_column("CAMBIO", style="yellow", justify="center")
        
        for key in self.stats_before:
            before = self.stats_before.get(key, 0)
            after = self.stats_after.get(key, 0)
            
            if isinstance(before, float):
                before_str = f"{before:.4f}"
                after_str = f"{after:.4f}"
                if before != 0:
                    change = ((after - before) / abs(before)) * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
            else:
                before_str = f"{before:,}"
                after_str = f"{after:,}"
                change_str = f"{after - before:+,}"
            
            table.add_row(key, before_str, after_str, change_str)
        
        console.print(table)


# =============================================================================
# ANÁLISIS GPR (WRAPPER)
# =============================================================================

def run_gpr_analysis(df: pd.DataFrame, param_columns: List[str], 
                     metric_columns: List[str], filepath: str) -> bool:
    """Ejecuta el análisis GPR usando el módulo analisis.py."""
    try:
        # Importar el analizador principal
        from analisis import BayesianDenoisingAnalyzer, console as analisis_console
        
        analyzer = BayesianDenoisingAnalyzer()
        
        # Cargar datos directamente
        analyzer.df = df
        analyzer.filepath = filepath
        analyzer.param_columns = param_columns
        
        # Seleccionar métricas
        console.print(f"\n[bold white]  MÉTRICAS DISPONIBLES:[/bold white]")
        for i, col in enumerate(metric_columns):
            console.print(f"[grey70]    {i+1}. {col}[/grey70]")
        
        # Preguntar qué métricas analizar
        default_metrics = ['roi', 'sharpe', 'calmar', 'profit_factor']
        selected = []
        for metric in metric_columns:
            metric_lower = metric.lower()
            if any(dm in metric_lower for dm in default_metrics):
                selected.append(metric)
        
        if not selected:
            selected = metric_columns[:3]  # Primeras 3 si no hay coincidencias
        
        console.print(f"\n[grey70]  MÉTRICAS SELECCIONADAS: {', '.join(selected)}[/grey70]")
        
        if Confirm.ask("[white]  ¿CAMBIAR SELECCIÓN?[/white]", default=False):
            indices = Prompt.ask("[white]  NÚMEROS SEPARADOS POR COMA[/white]")
            indices = [int(i.strip()) - 1 for i in indices.split(',')]
            selected = [metric_columns[i] for i in indices if 0 <= i < len(metric_columns)]
        
        analyzer.selected_metrics = selected
        
        # Ejecutar análisis
        console.print(UI.section("ANÁLISIS GPR"))
        analyzer.run_gpr_analysis(n_grid_points=30)
        
        if not analyzer.gpr_results:
            console.print("[red]  ❌ ERROR: SIN RESULTADOS GPR[/red]")
            return False
        
        # Generar visualizaciones
        console.print(UI.section("GENERANDO VISUALIZACIONES"))
        analyzer.generate_all_figures()
        
        # Generar PDF
        console.print(UI.section("GENERANDO REPORTES"))
        base = Path(filepath).stem + "_FILTRADO"
        
        pdf_type = Prompt.ask(
            "  [white]TIPO DE REPORTE[/white]",
            choices=["profesional", "simple"],
            default="profesional"
        )
        
        pdf_path = f"reporte_gpr_{pdf_type}_{base}.pdf"
        
        if pdf_type == "profesional":
            try:
                from visual.pdf_profesional import generate_professional_report
                pdf_result = generate_professional_report(
                    gpr_results=analyzer.gpr_results,
                    gpr_models=analyzer.gpr_models,
                    df=analyzer.df,
                    param_columns=analyzer.param_columns,
                    filepath=filepath,
                    output_path=pdf_path
                )
            except Exception as e:
                console.print(f"[yellow]  ⚠ Error PDF profesional: {e}[/yellow]")
                pdf_result = analyzer.generate_pdf_report(pdf_path)
        else:
            pdf_result = analyzer.generate_pdf_report(pdf_path)
        
        # Excel
        if Confirm.ask("\n[white]  ¿EXPORTAR EXCEL?[/white]", default=True):
            excel_path = pdf_path.replace('.pdf', '.xlsx')
            analyzer.export_to_excel(excel_path)
        
        # Abrir PDF
        if pdf_result and Confirm.ask("\n[white]  ¿ABRIR REPORTE?[/white]", default=True):
            import subprocess
            if sys.platform == 'darwin':
                subprocess.run(['open', pdf_result])
            elif sys.platform == 'win32':
                os.startfile(pdf_result)
            else:
                subprocess.run(['xdg-open', pdf_result])
        
        return True
        
    except Exception as e:
        console.print(f"[red]  ❌ ERROR EN ANÁLISIS GPR: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Punto de entrada principal."""
    
    # Banner
    console.print(UI.banner())
    console.print()
    
    start_time = time.perf_counter()
    
    # =========================================================================
    # PASO 1: CARGAR ARCHIVO
    # =========================================================================
    console.print(UI.section("PASO 1: CARGAR ARCHIVO"))
    
    loader = DataLoader()
    
    # Verificar si se pasó archivo por argumento
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        console.print(f"[grey70]  ARCHIVO: {filepath}[/grey70]")
        if not loader.load(filepath):
            return
    else:
        console.print("[white]  ARRASTRA EL ARCHIVO EXCEL/CSV:[/white]\n")
        while True:
            filepath = Prompt.ask("  [bold white]ARCHIVO[/bold white]")
            if loader.load(filepath):
                break
            console.print("[red]  INTENTA DE NUEVO.[/red]\n")
    
    # =========================================================================
    # PASO 2: DETECTAR COLUMNAS
    # =========================================================================
    console.print(UI.section("PASO 2: DETECTAR COLUMNAS"))
    
    if not loader.detect_columns():
        console.print("[red]  ❌ ERROR DETECTANDO COLUMNAS[/red]")
        return
    
    # Permitir selección manual de columnas si es necesario
    if not loader.score_column:
        if Confirm.ask("[yellow]  ¿SELECCIONAR COLUMNA SCORE MANUALMENTE?[/yellow]", default=True):
            loader.show_columns()
            idx = int(Prompt.ask("  [white]NÚMERO DE COLUMNA SCORE[/white]"))
            loader.score_column = loader.df.columns[idx]
            console.print(f"[green]  ✓ SCORE: {loader.score_column}[/green]")
    
    if not loader.roi_column:
        if Confirm.ask("[yellow]  ¿SELECCIONAR COLUMNA ROI MANUALMENTE?[/yellow]", default=True):
            loader.show_columns()
            idx = int(Prompt.ask("  [white]NÚMERO DE COLUMNA ROI[/white]"))
            loader.roi_column = loader.df.columns[idx]
            console.print(f"[green]  ✓ ROI: {loader.roi_column}[/green]")
    
    # =========================================================================
    # PASO 3: ESTADÍSTICAS ANTES DEL FILTRADO
    # =========================================================================
    console.print(UI.section("PASO 3: ESTADÍSTICAS ORIGINALES"))
    
    if loader.score_column:
        console.print(UI.stats_table({
            'TRIALS': len(loader.df),
            'SCORE_MEAN': loader.df[loader.score_column].mean(),
            'SCORE_STD': loader.df[loader.score_column].std(),
            'SCORE_MIN': loader.df[loader.score_column].min(),
            'SCORE_MAX': loader.df[loader.score_column].max(),
        }, "ESTADÍSTICAS SCORE"))
    
    if loader.roi_column:
        roi_positive = (loader.df[loader.roi_column] >= 0).sum()
        roi_negative = (loader.df[loader.roi_column] < 0).sum()
        console.print(UI.stats_table({
            'ROI_MEAN': loader.df[loader.roi_column].mean(),
            'ROI_STD': loader.df[loader.roi_column].std(),
            'ROI_MIN': loader.df[loader.roi_column].min(),
            'ROI_MAX': loader.df[loader.roi_column].max(),
            'ROI >= 0': roi_positive,
            'ROI < 0': roi_negative,
        }, "ESTADÍSTICAS ROI"))
    
    # =========================================================================
    # PASO 4: CONFIGURAR FILTROS
    # =========================================================================
    console.print(UI.section("PASO 4: CONFIGURAR FILTROS"))
    
    config = FilterConfig()
    
    # ROI mínimo
    config.min_roi = float(Prompt.ask(
        "  [white]ROI MÍNIMO[/white]",
        default="0.0"
    ))
    
    # Percentil de score
    config.score_percentile = float(Prompt.ask(
        "  [white]PERCENTIL SCORE (50=MEDIA)[/white]",
        default="50"
    ))
    
    console.print(f"[grey70]  CONFIGURACIÓN:[/grey70]")
    console.print(f"[grey50]    - ROI >= {config.min_roi}[/grey50]")
    console.print(f"[grey50]    - SCORE >= Percentil {config.score_percentile}[/grey50]")
    
    # =========================================================================
    # PASO 5: APLICAR FILTROS
    # =========================================================================
    console.print(UI.section("PASO 5: APLICAR FILTROS"))
    
    filter_engine = IntelligentFilter(config)
    df_filtered = filter_engine.apply(
        loader.df,
        roi_col=loader.roi_column,
        score_col=loader.score_column
    )
    
    # Mostrar resumen
    console.print()
    console.print(UI.filter_summary(
        before=len(loader.df),
        after=len(df_filtered),
        removed_roi=filter_engine.removed_roi,
        removed_score=filter_engine.removed_score
    ))
    
    # Comparación detallada
    console.print()
    filter_engine.show_comparison()
    
    # Verificar si hay suficientes datos
    if len(df_filtered) < 30:
        console.print(f"[red]  ❌ MUY POCOS DATOS DESPUÉS DEL FILTRADO ({len(df_filtered)})[/red]")
        console.print("[yellow]  CONSIDERA AJUSTAR LOS FILTROS[/yellow]")
        if not Confirm.ask("[yellow]  ¿CONTINUAR DE TODOS MODOS?[/yellow]", default=False):
            return
    
    # Guardar datos filtrados
    if Confirm.ask("\n[white]  ¿GUARDAR DATOS FILTRADOS?[/white]", default=True):
        filtered_path = Path(loader.filepath).stem + "_FILTRADO.xlsx"
        filtered_path = Prompt.ask("  [white]NOMBRE ARCHIVO[/white]", default=filtered_path)
        df_filtered.to_excel(filtered_path, index=False)
        console.print(f"[green]  ✓ GUARDADO: {filtered_path}[/green]")
    
    # =========================================================================
    # PASO 6: ANÁLISIS GPR
    # =========================================================================
    console.print(UI.section("PASO 6: ANÁLISIS GPR"))
    
    if not Confirm.ask("[white]  ¿EJECUTAR ANÁLISIS GPR?[/white]", default=True):
        console.print("[yellow]  ANÁLISIS CANCELADO[/yellow]")
        return
    
    # Filtrar métricas válidas (que existan en el df filtrado)
    valid_metrics = [m for m in loader.metric_columns if m in df_filtered.columns]
    
    success = run_gpr_analysis(
        df=df_filtered,
        param_columns=loader.param_columns,
        metric_columns=valid_metrics,
        filepath=loader.filepath
    )
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    total_time = time.perf_counter() - start_time
    
    console.print()
    console.print(Panel(
        f"[bold white]TIEMPO TOTAL: {total_time:.1f}s[/bold white]\n\n"
        f"[grey70]TRIALS PROCESADOS: {len(df_filtered):,} de {len(loader.df):,}[/grey70]",
        title="[bold green]◆ COMPLETADO[/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    main()

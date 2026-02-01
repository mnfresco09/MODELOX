#!/usr/bin/env python3
"""
# =============================================================================
#
#      █████╗ ███╗   ██╗ █████╗ ██╗     ██╗███████╗██╗███████╗
#     ██╔══██╗████╗  ██║██╔══██╗██║     ██║██╔════╝██║██╔════╝
#     ███████║██╔██╗ ██║███████║██║     ██║███████╗██║███████╗
#     ██╔══██║██║╚██╗██║██╔══██║██║     ██║╚════██║██║╚════██║
#     ██║  ██║██║ ╚████║██║  ██║███████╗██║███████║██║███████║
#     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚═╝╚══════╝
#
#     ANALISIS.PY - ANÁLISIS BAYESIANO DE RESULTADOS
#
# =============================================================================
#
#     TECNOLOGÍAS:
#     - GPyTorch para Gaussian Process Regression
#     - PyArrow + Parquet para I/O columnar
#     - Numba JIT para loops críticos
#     - Joblib para paralelización
#
# =============================================================================
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import os
import sys
import io
import base64
import warnings
import tempfile
import atexit
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pa_csv
import pandas as pd
import torch
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel, RQKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.optim import NGD
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from joblib import Parallel, delayed, cpu_count
from numba import jit, prange

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    TaskProgressColumn
)
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich import box
from rich.traceback import install as install_rich_traceback
import time

try:
    from modelox.core.types import full_system_cleanup
except ImportError:
    # Fallback si el types.py no tiene full_system_cleanup
    def full_system_cleanup():
        import gc
        gc.collect()
        gc.collect()
        gc.collect()

install_rich_traceback(show_locals=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE VISUALIZACIÓN PROFESIONAL
# ═══════════════════════════════════════════════════════════════════════════════

class ProfessionalUI:
    """Sistema de interfaz profesional en tonos neutros (blanco/gris)."""
    
    # Estilos profesionales neutros
    STYLES = {
        'header': 'bold white',
        'subheader': 'white',
        'text': 'grey70',
        'dim': 'grey50',
        'success': 'green',
        'warning': 'yellow',
        'error': 'red',
        'highlight': 'bold white',
        'value': 'white',
        'label': 'grey70',
        'box_border': 'grey50',
        'progress': 'grey70',
    }
    
    @staticmethod
    def header(title: str, subtitle: str = "") -> Panel:
        """Crea header principal profesional."""
        content = Text()
        content.append(title.upper(), style="bold white")
        if subtitle:
            content.append(f"\n{subtitle}", style="grey70")
        return Panel(
            content,
            border_style="grey50",
            box=box.DOUBLE,
            padding=(1, 2)
        )
    
    @staticmethod
    def section(title: str) -> Rule:
        """Crea separador de sección."""
        return Rule(f"[bold white]{title.upper()}[/bold white]", style="grey50")
    
    @staticmethod
    def info_box(title: str, items: dict, border_style: str = "grey50") -> Panel:
        """Crea caja de información profesional."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("LABEL", style="grey70", justify="right")
        table.add_column("VALUE", style="white")
        
        for key, value in items.items():
            table.add_row(f"{key.upper()}:", str(value))
        
        return Panel(
            table,
            title=f"[bold white]{title.upper()}[/bold white]",
            border_style=border_style,
            box=box.ROUNDED
        )
    
    @staticmethod
    def status_box(title: str, status: str, details: list = None) -> Panel:
        """Crea caja de estado con detalles."""
        content = Text()
        content.append(f"{status}\n", style="bold white")
        if details:
            for detail in details:
                content.append(f"  • {detail}\n", style="grey70")
        
        return Panel(
            content,
            title=f"[grey70]{title.upper()}[/grey70]",
            border_style="grey50",
            box=box.ROUNDED
        )
    
    @staticmethod
    def progress_table(title: str, headers: list, rows: list) -> Table:
        """Crea tabla profesional."""
        table = Table(
            title=f"[bold white]{title.upper()}[/bold white]",
            box=box.ROUNDED,
            border_style="grey50",
            header_style="bold grey70",
            show_lines=False
        )
        
        for h in headers:
            justify = "right" if any(x in h.lower() for x in ['min', 'max', 'step', 'valor']) else "left"
            table.add_column(h.upper(), style="white", justify=justify)
        
        for row in rows:
            table.add_row(*[str(v) for v in row])
        
        return table
    
    @staticmethod
    def metric_result(metric: str, r2: float, r2_cv: float, noise: float, model_type: str, 
                      n_samples: int, n_inducing: int = 0, time_elapsed: float = 0,
                      is_overfit: bool = False) -> Panel:
        """Muestra resultado de métrica con formato profesional."""
        # Calidad visual basada en R² CV (más realista)
        if is_overfit:
            quality = "[red]⚠⚠⚠⚠⚠[/red] SOBREAJUSTE"
        elif r2_cv > 0.7:
            quality = "[green]■■■■■[/green] EXCELENTE"
        elif r2_cv > 0.4:
            quality = "[yellow]■■■░░[/yellow] MODERADO"
        else:
            quality = "[red]■░░░░[/red] BAJO"
        
        content = Text()
        content.append(f"R² TRAIN:     ", style="grey70")
        content.append(f"{r2:.4f}\n", style="white")
        content.append(f"R² CV:       ", style="grey70")
        content.append(f"{r2_cv:.4f} ", style="bold white")
        content.append(f"({quality})\n")
        
        # Mostrar gap y advertencia de sobreajuste
        gap = r2 - r2_cv
        content.append(f"Δ (Train-CV): ", style="grey70")
        if is_overfit:
            content.append(f"{gap:.4f} ", style="bold red")
            content.append(f"[⚠ DESIONIZACIÓN NO CONFIABLE]\n", style="red")
        else:
            content.append(f"{gap:.4f} [✓ OK]\n", style="white")
        
        content.append(f"NOISE (σn):   ", style="grey70")
        content.append(f"{noise:.4f}\n", style="white")
        content.append(f"MODELO:       ", style="grey70")
        
        if model_type == "SPARSE":
            content.append(f"SPARSE GP ({n_inducing:,} INDUCING POINTS)\n", style="white")
            content.append(f"SAMPLES:      ", style="grey70")
            content.append(f"{n_samples:,} → {n_inducing:,} (SPARSE)\n", style="white")
        else:
            content.append(f"EXACT GP (FULL)\n", style="white")
            content.append(f"SAMPLES:      ", style="grey70")
            content.append(f"{n_samples:,} (TODOS)\n", style="white")
        
        if time_elapsed > 0:
            content.append(f"TIEMPO:       ", style="grey70")
            content.append(f"{time_elapsed:.2f}s\n", style="white")
        
        return Panel(
            content,
            title=f"[bold white]◆ {metric.upper()}[/bold white]",
            border_style="grey50",
            box=box.ROUNDED
        )

UI = ProfessionalUI()


# =============================================================================
# PROMPT CON TIMEOUT Y CUENTA ATRÁS
# =============================================================================

def prompt_with_timeout(prompt_text: str, default: str, timeout: int = 18, 
                        choices: List[str] = None) -> str:
    """
    Muestra un prompt con cuenta atrás. Si no hay input en 'timeout' segundos,
    devuelve el valor por defecto automáticamente.
    
    Args:
        prompt_text: Texto del prompt
        default: Valor por defecto
        timeout: Segundos de espera (default 18)
        choices: Lista de opciones válidas (opcional)
    
    Returns:
        Valor introducido o default si hay timeout
    """
    import sys
    import select
    import termios
    import tty
    import shutil
    
    # Obtener ancho del terminal
    term_width = shutil.get_terminal_size().columns
    
    # Truncar default si es muy largo para evitar wrap
    default_display = default
    if len(default) > 50:
        default_display = default[:25] + "..." + default[-20:]
    
    # Construir texto base del prompt
    if choices:
        choices_str = "/".join(choices)
        base_text = f"{prompt_text} [{choices_str}] ({default_display}): "
    else:
        base_text = f"{prompt_text} ({default_display}): "
    
    # Guardar configuración del terminal
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        # Configurar terminal para lectura no bloqueante
        tty.setcbreak(sys.stdin.fileno())
        
        input_chars = []
        start_time = time.time()
        last_countdown = timeout + 1
        
        # Imprimir línea inicial
        sys.stdout.write(f"\r\033[K{base_text}")  # \033[K = borrar hasta fin de línea
        sys.stdout.flush()
        
        while True:
            remaining = timeout - int(time.time() - start_time)
            
            # Actualizar solo el contador (misma línea)
            if remaining != last_countdown and remaining >= 0:
                user_input = ''.join(input_chars)
                # \033[K = borrar desde cursor hasta fin de línea
                line = f"\r\033[K{base_text}{user_input} \033[93m⏱ {remaining:2d}s\033[0m"
                sys.stdout.write(line)
                sys.stdout.flush()
                last_countdown = remaining
            
            # Timeout alcanzado
            if remaining <= 0:
                sys.stdout.write(f"\n  \033[92m✓ Auto-seleccionado:\033[0m {default_display}\n")
                sys.stdout.flush()
                return default
            
            # Verificar si hay input disponible (espera 0.1s)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                
                # Enter - confirmar input
                if char == '\n' or char == '\r':
                    result = ''.join(input_chars).strip()
                    if not result:
                        result = default
                    # Validar choices si existen
                    if choices and result not in choices:
                        result = default
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return result
                
                # Backspace
                elif char == '\x7f' or char == '\x08':
                    if input_chars:
                        input_chars.pop()
                        user_input = ''.join(input_chars)
                        line = f"\r\033[K{base_text}{user_input} \033[93m⏱ {remaining:2d}s\033[0m"
                        sys.stdout.write(line)
                        sys.stdout.flush()
                
                # Ctrl+C
                elif char == '\x03':
                    sys.stdout.write("\n\033[91mCancelado\033[0m\n")
                    sys.stdout.flush()
                    raise KeyboardInterrupt
                
                # Carácter normal
                elif char.isprintable():
                    input_chars.append(char)
                    sys.stdout.write(char)
                    sys.stdout.flush()
    
    except Exception as e:
        # Fallback a prompt normal si hay error
        sys.stdout.write("\n")
        sys.stdout.flush()
        if choices:
            return Prompt.ask(prompt_text, choices=choices, default=default)
        else:
            return Prompt.ask(prompt_text, default=default)
    
    finally:
        # Restaurar configuración del terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# Suprimir warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# Número de cores disponibles
N_JOBS = max(1, cpu_count() - 1)

# ============================================================================
# CONFIGURACIÓN PRINCIPAL - MODIFICAR AQUÍ SEGÚN TUS NECESIDADES
# ============================================================================

# ============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO GPR - EARLY STOPPING
# ============================================================================
# GPR_MAX_ITERATIONS: Número máximo de iteraciones de entrenamiento
#   - Más iteraciones = mejor ajuste pero más lento
#   - Recomendado: 500-2000
GPR_MAX_ITERATIONS = 1000

# GPR_PATIENCE: Iteraciones SIN MEJORA antes de parar (early stopping)
#   - Menor = para antes si no mejora (más rápido)
#   - Mayor = más oportunidades de mejorar
#   - Recomendado: 1-50 (1 = para inmediatamente si no mejora)
GPR_PATIENCE = 25

# GPR_MIN_DELTA: Mejora mínima para considerar progreso
#   - Menor = más sensible a mejoras pequeñas
#   - Recomendado: 0.0001-0.01
GPR_MIN_DELTA = 0.001

# ============================================================================
# CONFIGURACIÓN DE SPARSE GP (para datasets muy grandes)
# ============================================================================
# SPARSE_GP_ENABLED: Activa/desactiva Sparse GP (SVGP)
#   - True:  Usa Sparse Variational GP con inducing points (más rápido para N > 1000)
#   - False: Usa Exact GP (más preciso pero O(N³) en memoria/tiempo)
SPARSE_GP_ENABLED = True
SPARSE_INDUCING_POINTS = 150

# ============================================================================
# CONFIGURACIÓN DE RENDIMIENTO AVANZADO
# ============================================================================
# MINIBATCH_SIZE: Tamaño de mini-batch para entrenamiento Sparse GP
#   - Si MINIBATCH_SIZE >= N_SAMPLES: Full Batch (más preciso, menos epochs)
#   - Recomendado: 10000+ para forzar Full Batch con datasets pequeños
MINIBATCH_SIZE = 10000  # Forzar Full Batch (gradientes más precisos)

# PREDICTION_BATCH_SIZE: Tamaño de batch para predicciones
#   - Mayor = más rápido pero más memoria
#   - Recomendado: 2000-10000
PREDICTION_BATCH_SIZE = 5000

# NUM_THREADS: Threads para operaciones PyTorch (0 = auto)
NUM_TORCH_THREADS = 0

# Configurar PyTorch para máximo rendimiento CPU
torch.set_num_threads(NUM_TORCH_THREADS if NUM_TORCH_THREADS > 0 else max(1, mp.cpu_count() - 1))
torch.set_num_interop_threads(max(1, mp.cpu_count() // 2))

# Deshabilitar gradientes por defecto (solo activar cuando se necesiten)
torch.set_grad_enabled(False)

# torch.compile disponible en PyTorch 2.0+
# NOTA: torch.compile puede fallar en entornos sin compilador C++ (Nix, containers, etc.)
# Desactivar con: export MODELOX_DISABLE_TORCH_COMPILE=1
_TORCH_COMPILE_DISABLED = os.environ.get("MODELOX_DISABLE_TORCH_COMPILE", "0") in ("1", "true", "True")

# Auto-detectar entornos problemáticos (Nix, Replit, etc.)
def _detect_problematic_env():
    """Detecta entornos donde torch.compile suele fallar."""
    # Nix environment
    if "/nix/store" in os.environ.get("PATH", ""):
        return True
    # Replit
    if os.environ.get("REPL_ID") or os.environ.get("REPLIT_DB_URL"):
        return True
    # Verificar si cc/g++ están disponibles
    import shutil
    if not shutil.which("cc") and not shutil.which("gcc") and not shutil.which("clang"):
        return True
    return False

_IN_PROBLEMATIC_ENV = _detect_problematic_env()

# IMPORTANTE: torch.compile NO es compatible con GPyTorch
# GPyTorch usa LazyEvaluatedKernelTensor que torch._dynamo no puede manejar
# Error: "Unknown attribute __defaults__ for LazyEvaluatedKernelTensor"
# Por lo tanto, SIEMPRE desactivamos torch.compile cuando usamos GPyTorch
USE_TORCH_COMPILE = False  # Desactivado permanentemente para GPyTorch

# Configurar GPyTorch para máximo rendimiento (Optimización 3)
gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
gpytorch.settings.max_cholesky_size(2000)  # Usar Cholesky hasta 2000x2000
gpytorch.settings.cholesky_jitter(1e-4)  # Estabilidad numérica

# ============================================================================
# EVITAR CONFLICTOS DE OPENMP (causa segmentation fault)
# ============================================================================
# Numba y PyTorch usan diferentes versiones de OpenMP que pueden colisionar
# Desactivamos threading anidado para evitar el problema
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'


# =============================================================================
# FUNCIONES NUMBA JIT - OPERACIONES CRÍTICAS (SIN PARALELIZACIÓN)
# =============================================================================
# NOTA: parallel=False para evitar conflictos con PyTorch OpenMP

@jit(nopython=True, cache=True, fastmath=True)
def _compute_bin_stats_numba(x_vals: np.ndarray, y_vals: np.ndarray, 
                              bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula estadísticas por bin de forma ultra-rápida con Numba.
    """
    n_bins = len(bin_edges) - 1
    bin_sums = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_centers = np.zeros(n_bins, dtype=np.float32)
    
    # Calcular centros de bins
    for i in range(n_bins):
        bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0
    
    # Acumular valores en bins
    for i in range(len(x_vals)):
        x = x_vals[i]
        y = y_vals[i]
        # Encontrar bin
        for b in range(n_bins):
            if bin_edges[b] <= x < bin_edges[b + 1]:
                bin_sums[b] += y
                bin_counts[b] += 1
                break
        # Último bin incluye el máximo
        if x == bin_edges[n_bins]:
            bin_sums[n_bins - 1] += y
            bin_counts[n_bins - 1] += 1
    
    return bin_centers, bin_sums, bin_counts


@jit(nopython=True, cache=True, fastmath=True)
def _expand_matrix_for_pd_numba(X: np.ndarray, param_grid: np.ndarray, 
                                 param_idx: int) -> np.ndarray:
    """
    Expande matriz para cálculo de Partial Dependence - Numba optimizado.
    """
    n_points = len(param_grid)
    n_samples, n_features = X.shape
    result = np.empty((n_points * n_samples, n_features), dtype=np.float32)
    
    for i in range(n_points):
        start = i * n_samples
        for j in range(n_samples):
            for k in range(n_features):
                if k == param_idx:
                    result[start + j, k] = param_grid[i]
                else:
                    result[start + j, k] = X[j, k]
    
    return result


@jit(nopython=True, cache=True, fastmath=True)
def _expand_all_params_numba(X: np.ndarray, param_grids: np.ndarray, 
                              n_points: int, n_params: int) -> np.ndarray:
    """
    Expande matriz para TODOS los parámetros en una sola pasada.
    """
    n_samples, n_features = X.shape
    total_points = n_params * n_points * n_samples
    result = np.empty((total_points, n_features), dtype=np.float32)
    
    for param_idx in range(n_params):
        param_offset = param_idx * n_points * n_samples
        
        for point_idx in range(n_points):
            point_offset = param_offset + point_idx * n_samples
            grid_value = param_grids[param_idx, point_idx]
            
            for sample_idx in range(n_samples):
                row_idx = point_offset + sample_idx
                
                # Copiar toda la fila de X
                for feat_idx in range(n_features):
                    result[row_idx, feat_idx] = X[sample_idx, feat_idx]
                
                # Sobrescribir solo el parámetro actual
                result[row_idx, param_idx] = grid_value
    
    return result


@jit(nopython=True, cache=True, fastmath=True)
def _compute_r2_numba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula R² de forma optimizada con Numba."""
    n = len(y_true)
    mean_y = 0.0
    for i in range(n):
        mean_y += y_true[i]
    mean_y /= n
    
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - mean_y) ** 2
    
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


@jit(nopython=True, cache=True, fastmath=True)
def _reshape_and_marginalize_numba(y_pred: np.ndarray, n_points: int, 
                                    n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape y marginalize predicciones en una pasada.
    Retorna (mean, std) para cada punto del grid.
    """
    result_mean = np.empty(n_points, dtype=np.float32)
    result_std = np.empty(n_points, dtype=np.float32)
    
    for i in range(n_points):
        start = i * n_samples
        end = start + n_samples
        
        # Calcular media
        total = 0.0
        for j in range(start, end):
            total += y_pred[j]
        mean_val = total / n_samples
        result_mean[i] = mean_val
        
        # Calcular std
        var_sum = 0.0
        for j in range(start, end):
            var_sum += (y_pred[j] - mean_val) ** 2
        result_std[i] = np.sqrt(var_sum / n_samples)
    
    return result_mean, result_std


# =============================================================================
# CLASE PARA TRACKING DE PROCESOS (PROFESIONAL)
# =============================================================================

class ProcessTracker:
    """Sistema de tracking de procesos con formato profesional."""
    
    __slots__ = ['start_time', 'process_times', 'current_process', 'session_info']
    
    def __init__(self):
        self.start_time: float = 0
        self.process_times: Dict[str, Dict[str, Any]] = {}
        self.current_process: str = ""
        self.session_info: Dict[str, Any] = {}
        
    def start_session(self):
        self.start_time = time.perf_counter()
        self.process_times = {}
        self.session_info = {
            'sparse_enabled': SPARSE_GP_ENABLED,
            'inducing_points': SPARSE_INDUCING_POINTS,
            'n_jobs': N_JOBS,
            'torch_threads': torch.get_num_threads(),
            'torch_compile': USE_TORCH_COMPILE,
        }
        
    def start_process(self, name: str, description: str = ""):
        self.current_process = name
        self.process_times[name] = {
            'start': time.perf_counter(),
            'end': None,
            'duration': 0,
            'description': description,
            'status': 'running',
            'sub_tasks': []
        }
        
    def end_process(self, name: str = None, status: str = 'completed'):
        name = name or self.current_process
        if name in self.process_times:
            self.process_times[name]['end'] = time.perf_counter()
            self.process_times[name]['duration'] = (
                self.process_times[name]['end'] - self.process_times[name]['start']
            )
            self.process_times[name]['status'] = status
            
    def add_sub_task(self, process_name: str, task_name: str, duration: float):
        if process_name in self.process_times:
            self.process_times[process_name]['sub_tasks'].append({
                'name': task_name, 'duration': duration
            })
    
    def get_total_elapsed(self) -> float:
        return time.perf_counter() - self.start_time if self.start_time else 0
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 0.001:
            return f"{seconds*1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    
    def get_summary_table(self) -> Table:
        """Genera tabla de resumen profesional."""
        table = Table(
            title="[bold white]◆ RESUMEN DE TIEMPOS[/bold white]",
            box=box.ROUNDED,
            border_style="grey50",
            header_style="bold grey70"
        )
        table.add_column("PROCESO", style="white")
        table.add_column("ESTADO", justify="center", style="grey70")
        table.add_column("DURACIÓN", justify="right", style="white")
        table.add_column("% TOTAL", justify="right", style="grey70")
        
        total = self.get_total_elapsed()
        status_icons = {
            'completed': '[green]✓[/green]', 
            'running': '[yellow]⟳[/yellow]',
            'failed': '[red]✗[/red]', 
            'skipped': '[dim]○[/dim]'
        }
        
        for name, data in self.process_times.items():
            duration = data['duration'] if data['duration'] > 0 else (time.perf_counter() - data['start'])
            pct = (duration / total * 100) if total > 0 else 0
            table.add_row(
                (data['description'] or name).upper(),
                status_icons.get(data['status'], '?'),
                self.format_duration(duration),
                f"{pct:.1f}%"
            )
            for sub in data.get('sub_tasks', []):
                table.add_row(
                    f"  └─ {sub['name'].upper()}", 
                    "[grey50]·[/grey50]", 
                    f"[grey50]{self.format_duration(sub['duration'])}[/grey50]", 
                    ""
                )
        return table
    
    def get_config_panel(self) -> Panel:
        """Genera panel de configuración del sistema."""
        content = Text()
        content.append("CONFIGURACIÓN DEL MOTOR GPR\n\n", style="bold white")
        
        # Sparse GP
        if self.session_info.get('sparse_enabled'):
            content.append("SPARSE GP:        ", style="grey70")
            content.append(f"ACTIVADO ({self.session_info.get('inducing_points', 0):,} INDUCING POINTS)\n", style="green")
        else:
            content.append("SPARSE GP:        ", style="grey70")
            content.append("DESACTIVADO (EXACT GP)\n", style="yellow")
        
        # Threads
        content.append("TORCH THREADS:    ", style="grey70")
        content.append(f"{self.session_info.get('torch_threads', 0)}\n", style="white")
        
        # Jobs
        content.append("PARALLEL JOBS:    ", style="grey70")
        content.append(f"{self.session_info.get('n_jobs', 1)}\n", style="white")
        
        # Torch Compile
        content.append("TORCH.COMPILE:    ", style="grey70")
        compile_status = "DISPONIBLE" if self.session_info.get('torch_compile') else "NO DISPONIBLE"
        content.append(f"{compile_status}\n", style="white")
        
        return Panel(
            content,
            title="[grey70]◆ MOTOR GPR[/grey70]",
            border_style="grey50",
            box=box.ROUNDED
        )


tracker = ProcessTracker()


def create_progress() -> Progress:
    """Crea barra de progreso profesional en tonos neutros."""
    return Progress(
        SpinnerColumn("dots12", style="white"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=40, complete_style="white", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("[grey50]•[/grey50]"),
        MofNCompleteColumn(),
        TextColumn("[grey50]•[/grey50]"),
        TimeElapsedColumn(),
        TextColumn("[grey50]→[/grey50]"),
        TimeRemainingColumn(),
        console=console,
        transient=False
    )


def print_data_source_info(source_type: str, original_path: str, parquet_path: str = None,
                            original_size: float = 0, parquet_size: float = 0,
                            n_rows: int = 0, n_cols: int = 0):
    """Muestra información detallada sobre la fuente de datos."""
    content = Text()
    content.append("FUENTE DE DATOS\n\n", style="bold white")
    
    content.append("ARCHIVO ORIGINAL:  ", style="grey70")
    content.append(f"{original_path}\n", style="white")
    
    content.append("FORMATO ORIGINAL:  ", style="grey70")
    content.append(f"{source_type.upper()}\n", style="white")
    
    content.append("TAMAÑO ORIGINAL:   ", style="grey70")
    content.append(f"{original_size:.2f} MB\n", style="white")
    
    if parquet_path:
        content.append("\n", style="white")
        content.append("PARQUET GENERADO:  ", style="grey70")
        content.append(f"[green]SÍ[/green]\n", style="white")
        
        content.append("PARQUET PATH:      ", style="grey70")
        content.append(f"{parquet_path}\n", style="grey50")
        
        content.append("TAMAÑO PARQUET:    ", style="grey70")
        content.append(f"{parquet_size:.2f} MB\n", style="white")
        
        compression = (1 - parquet_size / original_size) * 100 if original_size > 0 else 0
        content.append("COMPRESIÓN:        ", style="grey70")
        content.append(f"{compression:.1f}% (SNAPPY)\n", style="green")
        
        content.append("\n", style="white")
        content.append("DATOS USADOS:      ", style="grey70")
        content.append(f"[green]PARQUET (MEMORY-MAPPED)[/green]\n", style="white")
    else:
        content.append("\n", style="white")
        content.append("PARQUET GENERADO:  ", style="grey70")
        content.append(f"[grey50]NO (YA ERA PARQUET)[/grey50]\n", style="white")
        
        content.append("DATOS USADOS:      ", style="grey70")
        content.append(f"[green]PARQUET DIRECTO[/green]\n", style="white")
    
    content.append("\n", style="white")
    content.append("FILAS CARGADAS:    ", style="grey70")
    content.append(f"{n_rows:,}\n", style="white")
    
    content.append("COLUMNAS:          ", style="grey70")
    content.append(f"{n_cols}\n", style="white")
    
    console.print(Panel(
        content,
        title="[grey70]◆ ORIGEN DE DATOS[/grey70]",
        border_style="grey50",
        box=box.ROUNDED
    ))


# =============================================================================
# CONFIGURACIÓN MATPLOTLIB
# =============================================================================

def configure_style():
    """Configura estilo matplotlib."""
    try:
        import scienceplots
        plt.style.use(['science', 'no-latex', 'grid'])
    except ImportError:
        plt.rcParams.update({
            'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
            'axes.titlesize': 12, 'axes.grid': True, 'grid.alpha': 0.3,
            'figure.figsize': (8, 6), 'figure.dpi': 100, 'savefig.dpi': 200,
        })


COLORS = {
    'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
    'danger': '#d62728', 'confidence': '#aec7e8', 'scatter': '#7f7f7f',
}


# =============================================================================
# DATACLASSES OPTIMIZADAS
# =============================================================================

@dataclass
class GPRResult:
    """Resultado del análisis GPR."""
    metric: str
    r2_score: float  # R² de entrenamiento (sobre todos los datos)
    r2_cv_score: float  # R² de validación cruzada (K-Fold)
    noise_level: float
    length_scales: Dict[str, float]
    predictions: Dict[str, 'ParameterPrediction']
    is_overfit: bool = False  # True si R²_train - R²_cv > umbral


@dataclass
class ParameterPrediction:
    """Predicción desionizada para un parámetro."""
    param_name: str
    param_values: np.ndarray
    mean_prediction: np.ndarray
    std_prediction: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    raw_data_x: np.ndarray
    raw_data_y: np.ndarray


@dataclass
class ParameterMetricPoint:
    """Un punto en la curva parámetro vs métrica (media agrupada)."""
    param_value: float
    metric_mean: float
    metric_std: float
    sample_count: int


@dataclass
class ParameterMetricCurve:
    """
    Análisis de un parámetro individual vs una métrica.
    Agrupa por valor de parámetro y calcula media de la métrica.
    """
    param_name: str
    metric_name: str
    # Datos agrupados
    unique_params: np.ndarray       # Valores únicos del parámetro
    mean_metrics: np.ndarray        # Media de métrica por valor
    std_metrics: np.ndarray         # Std de métrica por valor
    sample_counts: np.ndarray       # Número de muestras por valor
    # Curva de tendencia ajustada
    trend_x: np.ndarray             # X suavizado para la tendencia
    trend_y: np.ndarray             # Y de la tendencia polinomial
    trend_ci_lower: np.ndarray      # Banda inferior de confianza
    trend_ci_upper: np.ndarray      # Banda superior de confianza
    # Óptimo encontrado
    optimal_param: float            # Valor óptimo del parámetro
    optimal_metric: float           # Métrica en el óptimo
    # Estadísticas
    total_samples: int
    trend_degree: int               # Grado del polinomio ajustado
    r2_trend: float                 # R² del ajuste de tendencia


# =============================================================================
# GPyTorch Model Definitions
# =============================================================================

class ExactGPModel(ExactGP):
    """
    Modelo Exact GP con kernel aditivo multi-escala.
    
    Kernel = Scale(Matérn_2.5) + Scale(RationalQuadratic)
    
    - Matérn(nu=2.5): Captura estructura LOCAL (picos específicos, variaciones rápidas)
    - RationalQuadratic: Captura estructura GLOBAL (tendencias generales como 
      "a mayor Stop Loss, mejor ROI")
    
    Ambos con ARD (Automatic Relevance Determination) para aprender importancia
    de cada parámetro independientemente.
    """
    def __init__(self, train_x, train_y, likelihood, n_features):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        # Kernel aditivo: estructura local + estructura global
        # Matérn 2.5: suavidad local, derivadas continuas
        self.local_kernel = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=n_features)
        )
        # RationalQuadratic: mezcla de RBFs con diferentes lengthscales
        # Captura tendencias a múltiples escalas simultáneamente
        self.global_kernel = ScaleKernel(
            RQKernel(ard_num_dims=n_features)
        )
        # Kernel aditivo: K = K_local + K_global
        self.covar_module = self.local_kernel + self.global_kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGPModel(ApproximateGP):
    """
    Sparse Variational GP (SVGP) con kernel aditivo multi-escala.
    Usa inducing points para aproximar el GP completo.
    Complejidad: O(NM²) vs O(N³) del Exact GP.
    
    Kernel = Scale(Matérn_2.5) + Scale(RationalQuadratic)
    
    - Matérn(nu=2.5): Captura estructura LOCAL (picos específicos)
    - RationalQuadratic: Captura estructura GLOBAL (tendencias generales)
    """
    def __init__(self, inducing_points, n_features):
        # Variational distribution q(u) sobre los inducing points
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        # Estrategia variacional que conecta inducing points con predicciones
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True  # Optimizar ubicación de inducing points
        )
        super().__init__(variational_strategy)
        
        self.mean_module = ConstantMean()
        
        # Kernel aditivo: estructura local + estructura global
        self.local_kernel = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=n_features)
        )
        self.global_kernel = ScaleKernel(
            RQKernel(ard_num_dims=n_features)
        )
        self.covar_module = self.local_kernel + self.global_kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# =============================================================================
# GPROptimizer - Gaussian Process Regression con GPyTorch
# =============================================================================

class GPROptimizer:
    """
    GPR optimizado con GPyTorch (CPU) - Versión Ultra-Rápida.
    
    Optimizaciones:
    - Mini-batch training para Sparse GP (mejor convergencia)
    - Predicciones en chunks para evitar OOM
    - Cache LRU para predicciones repetidas (Optimización 1)
    - Pre-cómputo de Cholesky después del entrenamiento (Optimización 4)
    - GPyTorch fast settings habilitados
    - torch.inference_mode() para predicciones (Optimización 6)
    
    Kernel Aditivo Multi-escala:
        K = Scale(Matérn_2.5) + Scale(RationalQuadratic)
        
        - Matérn(nu=2.5): Estructura LOCAL - captura picos específicos y 
          variaciones rápidas en los parámetros
        - RationalQuadratic: Estructura GLOBAL - captura tendencias generales
          (ej. "a mayor Stop Loss, mejor ROI")
        
        Ambos con ARD (Automatic Relevance Determination) para aprender
        la importancia de cada parámetro independientemente.
    """
    
    # Cache LRU a nivel de clase para predicciones (Optimización 1)
    _global_cache = {}
    _cache_max_size = 500
    
    def __init__(self, n_restarts=2):
        self.n_restarts = n_restarts
        # Usar configuración global de entrenamiento
        self.training_iterations = GPR_MAX_ITERATIONS
        self.patience = GPR_PATIENCE
        self.min_delta = GPR_MIN_DELTA
        self.model = None
        self.likelihood = None
        self.scaler_X = None
        self.scaler_y = None
        self.learned_noise = 0.0
        self.learned_length_scales = np.array([])
        self.feature_names = []
        self._train_x = None
        self._train_y = None
        self.is_sparse = False
        self.n_inducing = 0
        self._prediction_cache = {}  # Cache local para este modelo
        self._cache_id = id(self)  # ID único para cache
        self._precomputed_cache = None  # Cache de Cholesky pre-computado
    
    def _initialize_inducing_points(self, X_scaled: np.ndarray, n_inducing: int) -> torch.Tensor:
        """
        Inicializa inducing points usando k-means con progreso visual.
        """
        n_inducing = min(n_inducing, len(X_scaled))
        n_samples = len(X_scaled)
        
        # K-Means con progreso: usar max_iter bajo y iterar manualmente
        max_iterations = 500  # Máximo de iteraciones
        kmeans_patience = 65  # Iteraciones sin mejora para converger
        min_improvement = 0.01  # 0.05% mínimo de mejora para considerar progreso
        batch_size = min(1000, n_samples)
        
        # Inicialización k-means++
        kmeans = MiniBatchKMeans(
            n_clusters=n_inducing,
            random_state=42,
            batch_size=batch_size,
            n_init=1,
            max_iter=1,  # Solo 1 iteración por llamada
            init='k-means++',
            reassignment_ratio=0.01,
        )
        
        # Primera iteración (incluye inicialización k-means++)
        kmeans.fit(X_scaled)
        prev_inertia = kmeans.inertia_
        best_inertia = prev_inertia
        no_improve_count = 0
        
        # Iteraciones con progreso visual
        for iteration in range(2, max_iterations + 1):
            # Continuar entrenamiento
            kmeans = MiniBatchKMeans(
                n_clusters=n_inducing,
                random_state=42,
                batch_size=batch_size,
                n_init=1,
                max_iter=1,
                init=kmeans.cluster_centers_,  # Usar centroides previos
                reassignment_ratio=0.01,
            )
            kmeans.fit(X_scaled)
            
            # Calcular mejora
            improvement = (prev_inertia - kmeans.inertia_) / prev_inertia * 100 if prev_inertia > 0 else 0
            prev_inertia = kmeans.inertia_
            
            # Barra de progreso
            pct = iteration / max_iterations
            bar_width = 15
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            console.print(f"[grey50]      │   [{bar}] ITER {iteration:2d}/{max_iterations} │ INERTIA={kmeans.inertia_:.0f} │ Δ={improvement:+.2f}%[/grey50]")
            
            # Early stopping con patience real
            if kmeans.inertia_ < best_inertia - (best_inertia * min_improvement / 100):
                best_inertia = kmeans.inertia_
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= kmeans_patience:
                console.print(f"[yellow]      │   ✓ CONVERGENCIA: {kmeans_patience} ITERS SIN MEJORA >{min_improvement}%[/yellow]")
                break
        
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    def _minibatch_iterator(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        """Generador de mini-batches para entrenamiento eficiente."""
        n_samples = len(X)
        indices = torch.randperm(n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> 'GPROptimizer':
        """Ajusta GPR con GPyTorch - Optimizado con mini-batches."""
        fit_total_start = time.perf_counter()
        
        self.feature_names = feature_names or [f"p{i}" for i in range(X.shape[1])]
        self.original_size = len(X)
        self._prediction_cache.clear()  # Limpiar cache
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 1: NORMALIZACIÓN
        # ═══════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        # RobustScaler: usa mediana e IQR, resistente a outliers extremos en trading
        # (trades con ROI 500% no distorsionan el escalado del resto)
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        # Forzar float32 ANTES del escalado para consistencia total
        X_f32 = X.astype(np.float32) if X.dtype != np.float32 else X
        y_f32 = y.astype(np.float32) if y.dtype != np.float32 else y
        X_scaled = self.scaler_X.fit_transform(X_f32).astype(np.float32)
        y_scaled = self.scaler_y.fit_transform(y_f32.reshape(-1, 1)).ravel().astype(np.float32)
        
        # Validar NaN/Inf después de normalización y limpiar si es necesario
        x_nan_mask = np.isnan(X_scaled).any(axis=1) | np.isinf(X_scaled).any(axis=1)
        y_nan_mask = np.isnan(y_scaled) | np.isinf(y_scaled)
        combined_mask = x_nan_mask | y_nan_mask
        
        if combined_mask.any():
            n_invalid = combined_mask.sum()
            console.print(f"[yellow]      │ ⚠ LIMPIANDO {n_invalid} MUESTRAS CON NaN/Inf[/yellow]")
            X_scaled = X_scaled[~combined_mask]
            y_scaled = y_scaled[~combined_mask]
            
            # Si quedan muy pocas muestras, error
            if len(X_scaled) < 10:
                raise ValueError(f"Solo quedan {len(X_scaled)} muestras válidas después de limpiar NaN/Inf")
        
        # Clipping extremo para evitar desbordamiento numérico
        X_scaled = np.clip(X_scaled, -10.0, 10.0)
        y_scaled = np.clip(y_scaled, -10.0, 10.0)
        
        t_norm = time.perf_counter() - t0
        console.print(f"[grey50]      │ NORMALIZACIÓN:    {t_norm*1000:.1f}ms ({len(X_scaled):,} SAMPLES × {X_scaled.shape[1]} FEATURES) [FLOAT32][/grey50]")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 2: CREAR TENSORES
        # ═══════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        train_x = torch.tensor(X_scaled, dtype=torch.float32).contiguous()
        train_y = torch.tensor(y_scaled, dtype=torch.float32).contiguous()
        self._train_x = train_x
        self._train_y = train_y
        n_features = X_scaled.shape[1]
        t_tensors = time.perf_counter() - t0
        console.print(f"[grey50]      │ TENSORES:         {t_tensors*1000:.1f}ms (SHAPE: {list(train_x.shape)})[/grey50]")
        
        # Habilitar gradientes solo para entrenamiento
        with torch.enable_grad():
            # Decidir entre Exact GP y Sparse GP
            if SPARSE_GP_ENABLED and len(X) > SPARSE_INDUCING_POINTS:
                self.is_sparse = True
                self.n_inducing = min(SPARSE_INDUCING_POINTS, len(X))
                
                # ═══════════════════════════════════════════════════════════════
                # FASE 3A: SPARSE GP - INDUCING POINTS
                # ═══════════════════════════════════════════════════════════════
                console.print(f"[white]      │ [bold cyan]◈ K-MEANS CLUSTERING[/bold cyan][/white]")
                console.print(f"[grey70]      │   OBJETIVO:    {self.n_inducing:,} INDUCING POINTS[/grey70]")
                console.print(f"[grey70]      │   DATOS:       {len(X):,} SAMPLES[/grey70]")
                t0 = time.perf_counter()
                inducing_points = self._initialize_inducing_points(X_scaled, self.n_inducing)
                t_inducing = time.perf_counter() - t0
                console.print(f"[green]      │ K-MEANS COMPLETADO: {t_inducing*1000:.1f}ms ✓[/green]")
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                console.print(f"[white]      │ [bold green]◆ MODELO: SPARSE GP (SVGP)[/bold green][/white]")
                console.print(f"[grey70]      │   DATOS ORIGINALES:    {len(X):,} SAMPLES[/grey70]")
                console.print(f"[grey70]      │   INDUCING POINTS:     {self.n_inducing:,} PUNTOS[/grey70]")
                console.print(f"[grey70]      │   REDUCCIÓN:           {100*(1-self.n_inducing/len(X)):.0f}% MENOS COMPLEJIDAD[/grey70]")
                console.print(f"[grey70]      │   MATRIZ COVARIANZA:   {self.n_inducing}×{self.n_inducing} (vs {len(X)}×{len(X)})[/grey70]")
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                
                # Crear modelo Sparse GP
                console.print(f"[grey50]      │ CREANDO MODELO...[/grey50]")
                t0 = time.perf_counter()
                self.likelihood = GaussianLikelihood()
                self.model = SparseGPModel(inducing_points, n_features)
                t_model = time.perf_counter() - t0
                console.print(f"[green]      │ MODELO CREADO:    {t_model*1000:.1f}ms ✓[/green]")
                
                # Entrenar con ELBO usando NGD + Adam (converge 3-4x más rápido)
                # NGD: Natural Gradient Descent para parámetros variacionales
                # Adam: Para hiperparámetros del kernel y likelihood
                self.model.train()
                self.likelihood.train()
                
                # Separar parámetros: variacionales vs hiperparámetros
                variational_params = set(self.model.variational_parameters())
                hyperparams = [
                    {'params': [p for p in self.model.parameters() if p not in variational_params]},
                    {'params': self.likelihood.parameters()},
                ]
                
                # Dual optimizer: NGD para distribución variacional (geometría de información)
                # Adam para kernel/likelihood (espacio euclidiano estándar)
                optimizer_ngd = NGD(
                    self.model.variational_parameters(), 
                    num_data=len(train_y), 
                    lr=0.05  # LR reducido para mayor estabilidad numérica
                )
                optimizer_hyper = torch.optim.Adam(hyperparams, lr=0.01)
                
                # Learning rate scheduler para hiperparámetros (NGD no necesita scheduler)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_hyper, T_max=self.training_iterations, eta_min=0.001
                )
                
                mll = VariationalELBO(self.likelihood, self.model, num_data=len(train_y))
                
                # Mini-batch training para datasets grandes
                use_minibatch = len(X) > MINIBATCH_SIZE * 2
                batch_info = f"MINIBATCH={MINIBATCH_SIZE}" if use_minibatch else "FULL-BATCH"
                n_batches_per_epoch = (len(X) + MINIBATCH_SIZE - 1) // MINIBATCH_SIZE if use_minibatch else 1
                
                # ═══════════════════════════════════════════════════════════════
                # FASE 4: ENTRENAMIENTO SPARSE GP
                # ═══════════════════════════════════════════════════════════════
                t0 = time.perf_counter()
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                console.print(f"[bold white]      │ ENTRENAMIENTO SVGP[/bold white]")
                console.print(f"[grey70]      │   EPOCHS:              {self.training_iterations}[/grey70]")
                console.print(f"[grey70]      │   MODO:                {batch_info}[/grey70]")
                if use_minibatch:
                    console.print(f"[grey70]      │   BATCHES/EPOCH:       {n_batches_per_epoch}[/grey70]")
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                
                losses = []
                best_loss = float('inf')
                best_model_state = None  # Guardar mejor estado
                best_likelihood_state = None
                best_epoch = 0
                no_improve_count = 0
                stopped_early = False
                prev_loss = None
                
                for epoch in range(self.training_iterations):
                    epoch_loss = 0.0
                    nan_detected = False
                    if use_minibatch:
                        n_batches = 0
                        for batch_x, batch_y in self._minibatch_iterator(train_x, train_y, MINIBATCH_SIZE):
                            # Zero gradients para ambos optimizadores
                            optimizer_ngd.zero_grad()
                            optimizer_hyper.zero_grad()
                            
                            output = self.model(batch_x)
                            loss = -mll(output, batch_y)
                            
                            # Validar NaN/Inf antes de backward
                            if torch.isnan(loss) or torch.isinf(loss):
                                nan_detected = True
                                break
                            
                            loss.backward()
                            
                            # Gradient clipping para evitar explosión
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(self.likelihood.parameters(), max_norm=1.0)
                            
                            # Step para ambos optimizadores
                            optimizer_ngd.step()
                            optimizer_hyper.step()
                            
                            epoch_loss += loss.item()
                            n_batches += 1
                        
                        if nan_detected:
                            console.print(f"[red]      │   ⚠ NaN/Inf DETECTADO EN EPOCH {epoch+1}, DETENIENDO...[/red]")
                            break
                        epoch_loss /= n_batches
                    else:
                        optimizer_ngd.zero_grad()
                        optimizer_hyper.zero_grad()
                        
                        output = self.model(train_x)
                        loss = -mll(output, train_y)
                        
                        # Validar NaN/Inf antes de backward
                        if torch.isnan(loss) or torch.isinf(loss):
                            console.print(f"[red]      │   ⚠ NaN/Inf DETECTADO EN EPOCH {epoch+1}, DETENIENDO...[/red]")
                            break
                        
                        loss.backward()
                        
                        # Gradient clipping para evitar explosión
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.likelihood.parameters(), max_norm=1.0)
                        
                        optimizer_ngd.step()
                        optimizer_hyper.step()
                        
                        epoch_loss = loss.item()
                    scheduler.step()
                    losses.append(epoch_loss)
                    
                    # Calcular cambio porcentual
                    if prev_loss is not None:
                        delta = (prev_loss - epoch_loss) / abs(prev_loss) * 100 if prev_loss != 0 else 0
                    else:
                        delta = 0.0
                    prev_loss = epoch_loss
                    
                    # Barra de progreso estilo K-Means (CADA EPOCH)
                    pct = (epoch + 1) / self.training_iterations
                    bar_width = 15
                    filled = int(bar_width * pct)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    console.print(f"[grey50]      │   [{bar}] EPOCH {epoch+1:3d}/{self.training_iterations} │ LOSS={epoch_loss:.4f} │ Δ={delta:+.2f}%[/grey50]")
                    
                    # Early stopping check CON CHECKPOINT del mejor modelo
                    if epoch_loss < best_loss - self.min_delta:
                        best_loss = epoch_loss
                        best_epoch = epoch + 1
                        # Guardar estado del mejor modelo
                        best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        best_likelihood_state = {k: v.clone() for k, v in self.likelihood.state_dict().items()}
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= self.patience:
                        console.print(f"[yellow]      │   ✋ EARLY STOP: {self.patience} EPOCHS SIN MEJORA[/yellow]")
                        stopped_early = True
                        break
                
                # Restaurar mejor modelo si hubo degradación
                if best_model_state is not None and losses[-1] > best_loss + 0.01:
                    self.model.load_state_dict(best_model_state)
                    self.likelihood.load_state_dict(best_likelihood_state)
                    console.print(f"[cyan]      │   🔄 RESTAURADO MEJOR MODELO (EPOCH {best_epoch}, LOSS={best_loss:.4f})[/cyan]")
                
                t_train = time.perf_counter() - t0
                epochs_used = len(losses)
                loss_reduction = (losses[0] - best_loss) / abs(losses[0]) * 100 if losses[0] != 0 else 0
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                console.print(f"[bold green]      │ ENTRENAMIENTO OK: {t_train:.2f}s ({epochs_used}/{self.training_iterations} epochs)[/bold green]")
                console.print(f"[grey70]      │   LOSS INICIAL:        {losses[0]:.4f}[/grey70]")
                console.print(f"[grey70]      │   LOSS MEJOR:          {best_loss:.4f} (epoch {best_epoch})[/grey70]")
                console.print(f"[grey70]      │   REDUCCIÓN:           {abs(loss_reduction):.1f}%[/grey70]")
                
            else:
                self.is_sparse = False
                self.n_inducing = 0
                
                # ═══════════════════════════════════════════════════════════════
                # FASE 3B: EXACT GP
                # ═══════════════════════════════════════════════════════════════
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                console.print(f"[white]      │ [bold yellow]◆ MODELO: EXACT GP (FULL)[/bold yellow][/white]")
                console.print(f"[grey70]      │   DATOS:               {len(X):,} SAMPLES (TODOS)[/grey70]")
                console.print(f"[grey70]      │   MATRIZ COVARIANZA:   {len(X)}×{len(X)}[/grey70]")
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                
                console.print(f"[grey50]      │ CREANDO MODELO...[/grey50]")
                t0 = time.perf_counter()
                self.likelihood = GaussianLikelihood()
                self.model = ExactGPModel(train_x, train_y, self.likelihood, n_features)
                t_model = time.perf_counter() - t0
                console.print(f"[green]      │ MODELO CREADO:    {t_model*1000:.1f}ms ✓[/green]")
                
                # Entrenar con MLL
                self.model.train()
                self.likelihood.train()
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.training_iterations, eta_min=0.001
                )
                mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
                
                # ═══════════════════════════════════════════════════════════════
                # FASE 4: ENTRENAMIENTO EXACT GP
                # ═══════════════════════════════════════════════════════════════
                t0 = time.perf_counter()
                console.print(f"[white]      │ [bold cyan]⚡ ENTRENAMIENTO EXACT GP[/bold cyan][/white]")
                console.print(f"[grey70]      │   MODO:    FULL-BATCH (TODAS LAS MUESTRAS)[/grey70]")
                console.print(f"[grey70]      │   EPOCHS:  {self.training_iterations} (max)[/grey70]")
                
                losses = []
                log_interval = 10  # Más frecuente para mejor visibilidad
                best_loss = float('inf')
                best_model_state = None
                best_likelihood_state = None
                best_epoch = 0
                no_improve_count = 0
                stopped_early = False
                prev_loss = None  # Para calcular delta
                
                for epoch in range(self.training_iterations):
                    optimizer.zero_grad()
                    output = self.model(train_x)
                    loss = -mll(output, train_y)
                    
                    # Validar NaN/Inf antes de backward
                    if torch.isnan(loss) or torch.isinf(loss):
                        console.print(f"[red]      │   ⚠ NaN/Inf DETECTADO EN EPOCH {epoch+1}, DETENIENDO...[/red]")
                        break
                    
                    loss.backward()
                    
                    # Gradient clipping para evitar explosión
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.likelihood.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    current_loss = loss.item()
                    losses.append(current_loss)
                    
                    # Calcular cambio porcentual
                    if prev_loss is not None:
                        delta = (prev_loss - current_loss) / abs(prev_loss) * 100 if prev_loss != 0 else 0
                    else:
                        delta = 0.0
                    prev_loss = current_loss
                    
                    # Early stopping check CON CHECKPOINT
                    if current_loss < best_loss - self.min_delta:
                        best_loss = current_loss
                        best_epoch = epoch + 1
                        best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        best_likelihood_state = {k: v.clone() for k, v in self.likelihood.state_dict().items()}
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= self.patience:
                        console.print(f"[yellow]      │   ✋ EARLY STOP: Convergencia en epoch {epoch+1} (sin mejora por {self.patience} epochs)[/yellow]")
                        stopped_early = True
                        break
                    
                    # Log CADA EPOCH (igual que Sparse GP) con delta
                    pct = (epoch + 1) / self.training_iterations
                    bar_width = 15
                    filled = int(bar_width * pct)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    console.print(f"[grey50]      │   [{bar}] EPOCH {epoch+1:3d}/{self.training_iterations} │ LOSS={current_loss:.4f} │ Δ={delta:+.2f}%[/grey50]")
                
                # Restaurar mejor modelo si hubo degradación
                if best_model_state is not None and losses[-1] > best_loss + 0.01:
                    self.model.load_state_dict(best_model_state)
                    self.likelihood.load_state_dict(best_likelihood_state)
                    console.print(f"[cyan]      │   🔄 RESTAURADO MEJOR MODELO (EPOCH {best_epoch}, LOSS={best_loss:.4f})[/cyan]")
                
                t_train = time.perf_counter() - t0
                epochs_used = len(losses)
                loss_reduction = (losses[0] - best_loss) / losses[0] * 100 if losses[0] != 0 else 0
                
                # Resumen final del entrenamiento
                console.print(f"[white]      ├──────────────────────────────────────────────────────────────[/white]")
                early_str = " (EARLY STOP)" if stopped_early else ""
                console.print(f"[green]      │ [bold]✓ ENTRENAMIENTO COMPLETADO{early_str}[/bold][/green]")
                console.print(f"[grey70]      │   EPOCHS:              {epochs_used}/{self.training_iterations}[/grey70]")
                console.print(f"[grey70]      │   TIEMPO TOTAL:        {t_train:.2f}s[/grey70]")
                console.print(f"[grey70]      │   LOSS INICIAL:        {losses[0]:.4f}[/grey70]")
                console.print(f"[grey70]      │   LOSS MEJOR:          {best_loss:.4f} (epoch {best_epoch})[/grey70]")
                console.print(f"[grey70]      │   REDUCCIÓN:           {abs(loss_reduction):.1f}%[/grey70]")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 5: MODO EVALUACIÓN Y WARMUP
        # ═══════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        self.model.eval()
        self.likelihood.eval()
        
        # Optimización 4: Pre-computar cache de predicción (warm-up)
        # Esto pre-computa la descomposición de Cholesky
        with torch.inference_mode(), gpytorch.settings.fast_pred_var():
            try:
                _ = self.model(self._train_x[:10])  # Warm-up con subset pequeño
            except Exception:
                pass  # Ignorar errores de warmup, el modelo funcionará de todas formas
        t_eval = time.perf_counter() - t0
        console.print(f"[grey50]      │ WARM-UP CHOLESKY: {t_eval*1000:.1f}ms[/grey50]")
        
        # Extraer hiperparámetros aprendidos
        # Con kernel aditivo: local_kernel (Matérn) + global_kernel (RQ)
        try:
            self.learned_noise = self.likelihood.noise.item()
            # Extraer lengthscales del kernel local (Matérn) - el más interpretable
            # En kernel aditivo: model.local_kernel.base_kernel.lengthscale
            if hasattr(self.model, 'local_kernel'):
                ls = self.model.local_kernel.base_kernel.lengthscale.detach().numpy().flatten()
            else:
                # Fallback para kernel simple (no aditivo)
                ls = self.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
            self.learned_length_scales = ls
        except:
            self.learned_noise = 0.0
            self.learned_length_scales = np.array([])
        
        # ═══════════════════════════════════════════════════════════════════════
        # RESUMEN FIT
        # ═══════════════════════════════════════════════════════════════════════
        fit_total_time = time.perf_counter() - fit_total_start
        model_type = "SPARSE GP" if self.is_sparse else "EXACT GP"
        console.print(f"[white]      └─ {model_type} FIT TOTAL: {fit_total_time:.2f}s[/white]")
        
        return self
    
    def _get_cache_key(self, X_batch: np.ndarray) -> int:
        """Genera clave de cache basada en hash de los datos."""
        # Usar hash de shape + primeros/últimos valores para key rápida
        return hash((X_batch.shape, X_batch[0, 0], X_batch[-1, -1], X_batch.sum()))
    
    def predict_batch(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicción vectorizada en batch con chunking para datasets grandes.
        Optimizaciones:
        - Cache LRU para predicciones repetidas (Optimización 1)
        - torch.inference_mode() más rápido que no_grad (Optimización 6)
        - GPyTorch fast settings habilitados
        """
        # Optimización 1: Verificar cache
        cache_key = self._get_cache_key(X_batch)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        # Forzar float32 para consistencia
        X_batch_f32 = X_batch.astype(np.float32) if X_batch.dtype != np.float32 else X_batch
        X_scaled = self.scaler_X.transform(X_batch_f32).astype(np.float32)
        n_samples = len(X_scaled)
        
        # Optimización 6: Usar inference_mode (más rápido que no_grad)
        # Para batches pequeños, predecir directamente
        if n_samples <= PREDICTION_BATCH_SIZE:
            test_x = torch.tensor(X_scaled, dtype=torch.float32).contiguous()
            
            with torch.inference_mode(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True):
                pred = self.likelihood(self.model(test_x))
                y_pred = pred.mean.numpy()
                y_std = pred.stddev.numpy()
        else:
            # Chunking para batches grandes (evita OOM y es más rápido)
            y_pred_chunks = []
            y_std_chunks = []
            
            with torch.inference_mode(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True):
                
                for start_idx in range(0, n_samples, PREDICTION_BATCH_SIZE):
                    end_idx = min(start_idx + PREDICTION_BATCH_SIZE, n_samples)
                    chunk = torch.tensor(X_scaled[start_idx:end_idx], dtype=torch.float32).contiguous()
                    
                    pred = self.likelihood(self.model(chunk))
                    y_pred_chunks.append(pred.mean.numpy())
                    y_std_chunks.append(pred.stddev.numpy())
            
            y_pred = np.concatenate(y_pred_chunks)
            y_std = np.concatenate(y_std_chunks)
        
        # Desescalar
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_std = y_std * self.scaler_y.scale_[0]
        
        # Optimización 1: Guardar en cache (limitar tamaño)
        if len(self._prediction_cache) < 20:  # Max 20 entradas por modelo
            self._prediction_cache[cache_key] = (y_pred, y_std)
        
        return y_pred, y_std
    
    def compute_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula R² de forma eficiente."""
        y_pred, _ = self.predict_batch(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# =============================================================================
# FUNCIONES DE CÁLCULO VECTORIZADO
# =============================================================================

def compute_all_partial_dependences(
    gpr: GPROptimizer,
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str],
    n_points: int = 50
) -> Dict[str, ParameterPrediction]:
    """
    Optimización 5: Calcula Dependencia Parcial de TODOS los parámetros en UNA llamada.
    
    En lugar de hacer N llamadas a predict_batch (una por parámetro),
    crea UNA matriz gigante con todos los parámetros y hace UNA sola predicción.
    
    Esto reduce drásticamente el overhead de:
    - Conversiones numpy ↔ torch
    - Overhead de llamadas a función
    - Mejor uso de caché de CPU
    """
    pd_start = time.perf_counter()
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_params = len(param_names)
    
    console.print(f"[grey50]      │ PARTIAL DEPENDENCE: {n_params} PARAMS × {n_points} PUNTOS[/grey50]")
    
    # Submuestreo solo si SPARSE_GP está habilitado (modo rápido)
    # En modo FULL (SPARSE_GP_ENABLED=False), usar todas las muestras
    if SPARSE_GP_ENABLED:
        max_samples_pd = SPARSE_INDUCING_POINTS  # Usar mismo tamaño que inducing points
        if n_samples > max_samples_pd:
            np.random.seed(42)
            sample_indices = np.random.choice(n_samples, max_samples_pd, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            n_samples_eff = max_samples_pd
            console.print(f"[grey50]      │   SUBMUESTREO:    {n_samples:,} → {n_samples_eff} SAMPLES (SPARSE MODE)[/grey50]")
        else:
            X_sample = X
            y_sample = y
            n_samples_eff = n_samples
    else:
        # MODO FULL: usar todas las muestras
        X_sample = X
        y_sample = y
        n_samples_eff = n_samples
        console.print(f"[grey50]      │   FULL MODE:      {n_samples_eff:,} SAMPLES (TODAS)[/grey50]")
    
    # Pre-calcular grids para todos los parámetros como matriz 2D para Numba
    t0 = time.perf_counter()
    param_grids_2d = np.empty((n_params, n_points), dtype=np.float32)
    for param_idx in range(n_params):
        param_min, param_max = X[:, param_idx].min(), X[:, param_idx].max()
        param_grids_2d[param_idx] = np.linspace(param_min, param_max, n_points, dtype=np.float32)
    
    # Usar función Numba fusionada para expandir TODOS los parámetros en una pasada
    # Esto elimina el overhead de Python y paraleliza automáticamente
    X_sample_f32 = X_sample.astype(np.float32) if X_sample.dtype != np.float32 else X_sample
    total_points = n_params * n_points * n_samples_eff
    
    X_all_expanded = _expand_all_params_numba(
        X_sample_f32, param_grids_2d, n_points, n_params
    )
    
    t_expand = time.perf_counter() - t0
    mem_mb = X_all_expanded.nbytes / (1024 * 1024)
    console.print(f"[grey50]      │   MATRIZ NUMBA:   {total_points:,} PUNTOS ({mem_mb:.1f} MB) EN {t_expand*1000:.1f}ms[/grey50]")
    
    # UNA SOLA predicción masiva (Optimización 5)
    t0 = time.perf_counter()
    y_pred_all, y_std_all = gpr.predict_batch(X_all_expanded)
    t_predict = time.perf_counter() - t0
    throughput = total_points / t_predict
    console.print(f"[grey50]      │   PREDICCIÓN:     {t_predict:.2f}s ({throughput:,.0f} PTS/S)[/grey50]")
    
    # Procesar resultados para cada parámetro
    t0 = time.perf_counter()
    results = {}
    offset = 0
    
    for param_idx, param_name in enumerate(param_names):
        # Extraer predicciones para este parámetro
        start = offset
        end = offset + n_points * n_samples_eff
        y_pred_param = y_pred_all[start:end]
        y_std_param = y_std_all[start:end]
        offset = end
        
        # Reshape y marginalizar
        y_pred_matrix = y_pred_param.reshape(n_points, n_samples_eff)
        y_std_matrix = y_std_param.reshape(n_points, n_samples_eff)
        
        pd_values = np.mean(y_pred_matrix, axis=1)
        pd_stds = np.mean(y_std_matrix, axis=1) / np.sqrt(n_samples_eff)
        
        # ICs
        z = 1.96
        lower_ci = pd_values - z * pd_stds
        upper_ci = pd_values + z * pd_stds
        
        # Datos agregados en BINS
        n_bins = min(25, len(np.unique(X[:, param_idx])))
        x_vals = X[:, param_idx]
        bin_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(x_vals, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_sums = np.bincount(bin_indices, weights=y, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        valid_bins = bin_counts > 0
        raw_x = bin_centers[valid_bins]
        raw_y = bin_sums[valid_bins] / bin_counts[valid_bins]
        
        results[param_name] = ParameterPrediction(
            param_name=param_name,
            param_values=param_grids_2d[param_idx],  # Usar matriz 2D
            mean_prediction=pd_values,
            std_prediction=pd_stds,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            raw_data_x=raw_x,
            raw_data_y=raw_y
        )
    
    t_process = time.perf_counter() - t0
    pd_total = time.perf_counter() - pd_start
    console.print(f"[grey50]      │   PROCESAMIENTO:  {t_process*1000:.1f}ms[/grey50]")
    console.print(f"[green]      │ PD TOTAL:         {pd_total:.2f}s PARA {n_params} PARÁMETROS[/green]")
    
    return results


def compute_partial_dependence_vectorized(
    gpr: GPROptimizer,
    X: np.ndarray,
    y: np.ndarray,
    param_idx: int,
    param_name: str,
    n_points: int = 50
) -> ParameterPrediction:
    """
    Calcula Dependencia Parcial - Versión Ultra-Optimizada.
    (Mantener para compatibilidad, pero preferir compute_all_partial_dependences)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Submuestreo solo si SPARSE_GP está habilitado
    # En modo FULL (SPARSE_GP_ENABLED=False), usar todas las muestras
    if SPARSE_GP_ENABLED:
        max_samples_pd = SPARSE_INDUCING_POINTS  # Usar mismo tamaño que inducing points
        if n_samples > max_samples_pd:
            np.random.seed(42)
            sample_indices = np.random.choice(n_samples, max_samples_pd, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            n_samples_eff = max_samples_pd
        else:
            X_sample = X
            y_sample = y
            n_samples_eff = n_samples
    else:
        # MODO FULL: usar todas las muestras
        X_sample = X
        y_sample = y
        n_samples_eff = n_samples
    
    param_min, param_max = X[:, param_idx].min(), X[:, param_idx].max()
    param_grid = np.linspace(param_min, param_max, n_points)
    
    # Usar función Numba para expansión de matriz
    X_expanded = _expand_matrix_for_pd_numba(X_sample, param_grid, param_idx)
    
    # Predicción en batch
    y_pred_all, y_std_all = gpr.predict_batch(X_expanded)
    
    # Reshape y calcular medias
    y_pred_matrix = y_pred_all.reshape(n_points, n_samples_eff)
    y_std_matrix = y_std_all.reshape(n_points, n_samples_eff)
    
    pd_values = np.mean(y_pred_matrix, axis=1)
    pd_stds = np.mean(y_std_matrix, axis=1) / np.sqrt(n_samples_eff)
    
    z = 1.96
    lower_ci = pd_values - z * pd_stds
    upper_ci = pd_values + z * pd_stds
    
    # Bins
    n_bins = min(25, len(np.unique(X[:, param_idx])))
    x_vals = X[:, param_idx]
    bin_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(x_vals, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_sums = np.bincount(bin_indices, weights=y, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    valid_bins = bin_counts > 0
    raw_x = bin_centers[valid_bins]
    raw_y = bin_sums[valid_bins] / bin_counts[valid_bins]
    
    return ParameterPrediction(
        param_name=param_name,
        param_values=param_grid,
        mean_prediction=pd_values,
        std_prediction=pd_stds,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        raw_data_x=raw_x,
        raw_data_y=raw_y
    )


# =============================================================================
# GENERACIÓN DE FIGURAS EN PARALELO
# =============================================================================

def _generate_heatmap(gpr: GPROptimizer, X: np.ndarray, metric: str,
                      param1_idx: int, param2_idx: int, param_names: List[str]) -> Tuple[str, str]:
    """Genera heatmap - función para paralelización."""
    n_grid = 20  # Reducido de 25 para velocidad (400 vs 625 puntos)
    x1_range = np.linspace(X[:, param1_idx].min(), X[:, param1_idx].max(), n_grid)
    x2_range = np.linspace(X[:, param2_idx].min(), X[:, param2_idx].max(), n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Crear grid completo para predicción batch
    X_mean = np.mean(X, axis=0)
    grid_points = []
    for i in range(n_grid):
        for j in range(n_grid):
            point = X_mean.copy()
            point[param1_idx] = X1[i, j]
            point[param2_idx] = X2[i, j]
            grid_points.append(point)
    
    X_grid = np.array(grid_points)
    Z_flat, _ = gpr.predict_batch(X_grid)
    Z = Z_flat.reshape(n_grid, n_grid)
    
    # Figura thread-safe: API de objetos sin estado global plt
    fig = Figure(figsize=(7, 5.5), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.contourf(X1, X2, Z, levels=12, cmap='RdYlGn')  # Reducido de 15
    ax.contour(X1, X2, Z, levels=6, colors='white', alpha=0.3, linewidths=0.4)  # Reducido de 8
    fig.colorbar(im, ax=ax, label=f'{metric.upper()} (Desionizado)')
    
    # Submuestrear puntos si hay muchos para scatter más rápido
    max_scatter_points = 150
    if len(X) > max_scatter_points:
        indices = np.random.choice(len(X), max_scatter_points, replace=False)
        ax.scatter(X[indices, param1_idx], X[indices, param2_idx], c='black', s=12, alpha=0.35, marker='x')
    else:
        ax.scatter(X[:, param1_idx], X[:, param2_idx], c='black', s=12, alpha=0.35, marker='x')
    
    p1_name = param_names[param1_idx].replace('param_', '').replace('_', ' ').title()
    p2_name = param_names[param2_idx].replace('param_', '').replace('_', ' ').title()
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_title(f'Superficie Desionizada: {metric.upper()}', fontweight='bold')
    
    # Renderizado thread-safe sin plt global
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    buf.seek(0)
    return f"heatmap_{metric}", base64.b64encode(buf.read()).decode('utf-8')


def _generate_uncertainty_plot(pred: ParameterPrediction, metric: str, 
                               r2: float, noise: float) -> Tuple[str, str]:
    """Genera plot de incertidumbre - función 100% thread-safe para paralelización."""
    param_clean = pred.param_name.replace('param_', '').replace('_', ' ').title()
    
    # API de objetos pura: sin estado global plt (thread-safe)
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    ax.fill_between(pred.param_values, pred.lower_ci, pred.upper_ci,
                   alpha=0.3, color=COLORS['confidence'], label='IC 95%')
    ax.plot(pred.param_values, pred.mean_prediction, color=COLORS['primary'],
           linewidth=2.5, label='Predicción Desionizada')
    
    # Puntos: datos reales agregados por bins (más limpio)
    ax.scatter(pred.raw_data_x, pred.raw_data_y, color=COLORS['secondary'],
              s=60, alpha=0.8, edgecolors='white', linewidths=1, 
              label=f'Datos (media por zona, n={len(pred.raw_data_x)})', zorder=5)
    
    ax.set_xlabel(param_clean)
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} vs {param_clean} | R²={r2:.3f} | σn={noise:.3f}', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Renderizado thread-safe sin plt global
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    buf.seek(0)
    return f"uncertainty_{metric}_{pred.param_name}", base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# CLASE PRINCIPAL OPTIMIZADA
# =============================================================================

class BayesianDenoisingAnalyzer:
    """Sistema de Análisis de Trading - Versión Ultra Optimizada con PyArrow/Parquet."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None  # DataFrame pandas (desde PyArrow)
        self.table: Optional[pa.Table] = None   # Tabla PyArrow original
        self.filepath: Optional[str] = None
        self.temp_parquet_path: Optional[str] = None  # Archivo temporal Parquet
        self.exit_type: str = "fixed"  # 'fixed' o 'trailing'
        self.param_columns: List[str] = []
        self.exit_columns: List[str] = []  # Parámetros de salida
        self.strategy_columns: List[str] = []  # Parámetros de estrategia
        self.metric_columns: List[str] = []
        self.selected_metrics: List[str] = []
        self.gpr_models: Dict[str, GPROptimizer] = {}
        self.gpr_results: Dict[str, GPRResult] = {}
        self.param_metric_curves: Dict[str, Dict[str, ParameterMetricCurve]] = {}  # {metric: {param: curve}}
        self.figures_base64: Dict[str, str] = {}
        configure_style()
        # Registrar limpieza automática al salir
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Elimina archivos temporales generados."""
        if self.temp_parquet_path and os.path.exists(self.temp_parquet_path):
            try:
                os.remove(self.temp_parquet_path)
            except:
                pass
            self.temp_parquet_path = None
    
    def __del__(self):
        """Destructor - limpia archivos temporales."""
        self.cleanup()
    
    def _detect_param_step(self, param: str) -> float:
        """Detecta automáticamente el STEP de un parámetro."""
        try:
            unique_vals = np.sort(self.df[param].dropna().to_numpy())
            if len(unique_vals) < 2:
                return 1.0
            diffs = np.diff(unique_vals)
            diffs = diffs[diffs > 1e-10]
            if len(diffs) == 0:
                return 1.0
            min_diff = np.min(diffs)
            if min_diff >= 1:
                return round(min_diff)
            elif min_diff >= 0.1:
                return round(min_diff, 1)
            elif min_diff >= 0.01:
                return round(min_diff, 2)
            else:
                return round(min_diff, 4)
        except:
            return 1.0
    
    def _detect_param_type(self, param: str) -> str:
        """Detecta el tipo de parámetro basándose en su nombre."""
        param_lower = param.lower()
        
        # Primero detectar parámetros de salida (más específicos)
        # Trailing Activation (TP Activation)
        if any(x in param_lower for x in ['trail_act', 'trailing_act', 'tp_act', 'activation_pct']):
            return 'exit_trailing_act'
        # Trailing Distance (TP Distance)
        if any(x in param_lower for x in ['trail_dist', 'trailing_dist', 'tp_dist', 'distance_pct']):
            return 'exit_trailing_dist'
        # Trailing general
        if any(x in param_lower for x in ['trailing', 'trail']) and 'pct' in param_lower:
            return 'exit_trailing'
        # Stop Loss
        if any(x in param_lower for x in ['exit_sl', 'sl_pct', 'stop_loss', '_sl_']):
            return 'exit_fixed_sl'
        # Take Profit fijo
        if any(x in param_lower for x in ['exit_tp', 'tp_pct', 'take_profit', '_tp_']) and 'trail' not in param_lower:
            return 'exit_fixed_tp'
        
        # Parámetros de estrategia
        if any(x in param_lower for x in ['dist', 'distance', 'req_dist']):
            return 'distance'
        if any(x in param_lower for x in ['lookbar', 'period', 'length', 'len', 'window', 'bars', 'fast', 'slow']):
            return 'period'
        if any(x in param_lower for x in ['_pct', 'percent', 'ratio']):
            return 'percentage'
        return 'other'
    
    def load_data(self, filepath: str) -> bool:
        """Carga datos y convierte a Parquet temporal para máxima velocidad."""
        filepath = filepath.strip().strip("'\"")
        tracker.start_process('load_data', 'CARGA DE DATOS')
        
        console.print(UI.section("CARGA DE DATOS"))
        
        try:
            if not os.path.exists(filepath):
                console.print(f"[red]ERROR: ARCHIVO NO EXISTE: {filepath}[/red]")
                tracker.end_process('load_data', 'failed')
                return False
            
            ext = Path(filepath).suffix.lower()
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            # Mostrar info inicial
            console.print(f"[grey70]  ARCHIVO:          [white]{Path(filepath).name}[/white][/grey70]")
            console.print(f"[grey70]  FORMATO:          [white]{ext.upper().replace('.', '')}[/white][/grey70]")
            console.print(f"[grey70]  TAMAÑO:           [white]{file_size:.2f} MB[/white][/grey70]")
            console.print("")
            
            load_start = time.perf_counter()
            converted_to_parquet = False
            parquet_size = 0
            compression_ratio = 0
            
            # Si ya es Parquet, cargar directamente con PyArrow
            if ext == '.parquet':
                t0 = time.perf_counter()
                console.print(f"[grey50]  LEYENDO PARQUET...[/grey50]")
                self.table = pq.read_table(filepath)
                self.df = self.table.to_pandas()
                t_read = time.perf_counter() - t0
                parquet_size = file_size
                console.print(f"[green]  ✓ PARQUET CARGADO: {t_read*1000:.1f}ms ({len(self.df):,} FILAS)[/green]")
                console.print(f"[grey50]  NOTA: YA OPTIMIZADO - NO NECESITA CONVERSIÓN[/grey50]")
                tracker.add_sub_task('load_data', 'LECTURA PARQUET', t_read)
            else:
                # Cargar desde CSV o Excel
                if ext == '.csv':
                    t0 = time.perf_counter()
                    console.print(f"[grey50]  LEYENDO CSV CON PYARROW...[/grey50]")
                    self.table = pa_csv.read_csv(filepath)
                    self.df = self.table.to_pandas()
                    t_read = time.perf_counter() - t0
                    console.print(f"[green]  ✓ CSV LEÍDO: {t_read:.2f}s ({len(self.df):,} FILAS)[/green]")
                    
                elif ext in ['.xlsx', '.xls']:
                    t0 = time.perf_counter()
                    console.print(f"[grey50]  LEYENDO EXCEL (PUEDE SER LENTO)...[/grey50]")
                    # Excel requiere pandas
                    df_temp = pd.read_excel(filepath, header=None, nrows=5)
                    keywords = ['TRIAL', 'ROI', 'SCORE', 'SHARPE', 'DRAWDOWN', 'PARAM', 
                               'EXIT_SL', 'EXIT_TP', 'LOOKBAR', 'WINRATE', 'SQN']
                    header_row = 0
                    
                    for idx in range(min(5, len(df_temp))):
                        row_str = ' '.join(str(v).upper() for v in df_temp.iloc[idx].values if pd.notna(v))
                        matches = sum(1 for kw in keywords if kw in row_str)
                        if matches >= 3:
                            header_row = idx
                            break
                    
                    console.print(f"[grey50]  HEADER DETECTADO EN FILA {header_row}[/grey50]")
                    self.df = pd.read_excel(filepath, header=header_row)
                    self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]
                    self.df.columns = self.df.columns.astype(str).str.strip()
                    
                    # Convertir tipos
                    n_converted = 0
                    for col in self.df.columns:
                        if self.df[col].dtype == 'object':
                            try:
                                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                                n_converted += 1
                            except:
                                pass
                    
                    t_read = time.perf_counter() - t0
                    console.print(f"[green]  ✓ EXCEL LEÍDO: {t_read:.2f}s ({len(self.df):,} FILAS × {len(self.df.columns)} COLS)[/green]")
                    if n_converted > 0:
                        console.print(f"[grey50]  {n_converted} COLUMNAS CONVERTIDAS A NUMÉRICO[/grey50]")
                else:
                    console.print(f"[red]ERROR: FORMATO NO SOPORTADO: {ext}[/red]")
                    tracker.end_process('load_data', 'failed')
                    return False
                
                read_time = time.perf_counter() - load_start
                tracker.add_sub_task('load_data', f'LECTURA {ext.upper()}', read_time)
                
                # Normalizar columnas ANTES de convertir a Parquet
                t0 = time.perf_counter()
                console.print(f"[grey50]  NORMALIZANDO COLUMNAS...[/grey50]")
                self.df = self._normalize_columns(self.df)
                t_norm = time.perf_counter() - t0
                console.print(f"[green]  ✓ COLUMNAS NORMALIZADAS: {t_norm*1000:.1f}ms[/green]")
                tracker.add_sub_task('load_data', 'NORMALIZACIÓN', t_norm)
                
                # Forzar FLOAT32 en columnas numéricas para mejor rendimiento
                t0 = time.perf_counter()
                float_cols = self.df.select_dtypes(include=['float64']).columns
                if len(float_cols) > 0:
                    self.df[float_cols] = self.df[float_cols].astype(np.float32)
                int_cols_64 = self.df.select_dtypes(include=['int64']).columns
                if len(int_cols_64) > 0:
                    # Solo convertir int64 pequeños a int32
                    for col in int_cols_64:
                        if self.df[col].max() < 2**31 and self.df[col].min() > -2**31:
                            self.df[col] = self.df[col].astype(np.int32)
                t_dtype = time.perf_counter() - t0
                console.print(f"[green]  ✓ DTYPE OPTIMIZADO: {t_dtype*1000:.1f}ms ({len(float_cols)} COLS → FLOAT32)[/green]")
                tracker.add_sub_task('load_data', 'DTYPE FLOAT32', t_dtype)
                
                # Convertir a Parquet temporal para acceso ultra-rápido
                t0 = time.perf_counter()
                console.print(f"[grey50]  CONVIRTIENDO A PARQUET TEMPORAL...[/grey50]")
                base_name = Path(filepath).stem
                self.temp_parquet_path = tempfile.mktemp(prefix=f"{base_name}_", suffix='.parquet')
                
                # Convertir a PyArrow Table y guardar como Parquet
                self.table = pa.Table.from_pandas(self.df)
                pq.write_table(self.table, self.temp_parquet_path, compression='snappy')
                
                parquet_size = os.path.getsize(self.temp_parquet_path) / (1024 * 1024)
                compression_ratio = (1 - parquet_size / file_size) * 100 if file_size > 0 else 0
                
                # Recargar desde Parquet (memory-mapped, ultra rápido)
                self.table = pq.read_table(self.temp_parquet_path)
                self.df = self.table.to_pandas()
                
                t_parquet = time.perf_counter() - t0
                console.print(f"[green]  ✓ PARQUET CREADO: {t_parquet:.2f}s[/green]")
                tracker.add_sub_task('load_data', 'CONVERSIÓN PARQUET', t_parquet)
                converted_to_parquet = True
            
            load_time = time.perf_counter() - load_start
            
            # Normalizar si no se hizo antes (caso Parquet directo)
            if ext == '.parquet':
                t0 = time.perf_counter()
                self.df = self._normalize_columns(self.df)
                self.table = pa.Table.from_pandas(self.df)
                t_norm = time.perf_counter() - t0
                console.print(f"[green]  ✓ COLUMNAS NORMALIZADAS: {t_norm*1000:.1f}ms[/green]")
                tracker.add_sub_task('load_data', 'NORMALIZACIÓN', t_norm)
            
            self.filepath = filepath
            tracker.end_process('load_data')
            
            # Panel de información de datos
            console.print("")
            print_data_source_info(
                source_type=ext.replace('.', ''),
                original_path=Path(filepath).name,
                parquet_path=self.temp_parquet_path if converted_to_parquet else None,
                original_size=file_size,
                parquet_size=parquet_size,
                n_rows=len(self.df),
                n_cols=len(self.df.columns)
            )
            
            # Tiempo total
            console.print(f"\n[bold white]  ◆ CARGA COMPLETADA: {tracker.format_duration(load_time)}[/bold white]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            import traceback
            traceback.print_exc()
            tracker.end_process('load_data', 'failed')
            return False
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas con Pandas."""
        rename_map = {}
        
        for col in df.columns:
            col_upper = col.upper().strip()
            
            # Métricas - detectar primero
            if col_upper in ['ROI_PCT', 'ROI%', 'ROI']:
                rename_map[col] = 'roi'
            elif col_upper == 'SHARPE':
                rename_map[col] = 'sharpe'
            elif col_upper == 'SQN':
                rename_map[col] = 'sqn'
            elif col_upper in ['MAX_DD_PCT', 'MAX_DD%', 'DRAWDOWN', 'MAX_DRAWDOWN']:
                rename_map[col] = 'drawdown'
            elif col_upper in ['WINRATE_PCT', 'WINRATE%', 'WINRATE', 'WIN_RATE']:
                rename_map[col] = 'winrate'
            elif col_upper in ['PROFIT_FACTOR', 'PF']:
                rename_map[col] = 'profit_factor'
            elif col_upper == 'SCORE':
                rename_map[col] = 'score'
            # Parámetros de salida FIXED (SL/TP)
            elif 'EXIT_SL' in col_upper or col_upper in ['SL%', 'SL_PCT']:
                rename_map[col] = 'param_exit_sl_pct'
            elif 'EXIT_TP' in col_upper or col_upper in ['TP%', 'TP_PCT']:
                rename_map[col] = 'param_exit_tp_pct'
            # Parámetros de salida TRAILING
            elif 'TRAIL_ACT' in col_upper or 'TRAILING_ACT' in col_upper:
                rename_map[col] = 'param_exit_trail_act_pct'
            elif 'TRAIL_DIST' in col_upper or 'TRAILING_DIST' in col_upper:
                rename_map[col] = 'param_exit_trail_dist_pct'
            # Parámetros de estrategia
            elif col_upper == 'LOOKBAR':
                rename_map[col] = 'param_lookbar'
            elif 'REQ_DIST' in col_upper:
                rename_map[col] = 'param_req_dist_pct'
            elif 'ZLEMA_FAST' in col_upper:
                rename_map[col] = 'param_zlema_fast_len'
            elif 'ZLEMA_SLOW' in col_upper:
                rename_map[col] = 'param_zlema_slow_len'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def select_exit_type(self) -> str:
        """Permite seleccionar el tipo de salida."""
        console.print(UI.section("CONFIGURACIÓN DE SALIDA"))
        
        # Detectar automáticamente qué tipo hay en los datos
        all_param_cols = [c for c in self.df.columns if c.startswith('param_')]
        
        # TRAILING: tiene trail_act o trail_dist
        has_trailing = any('trail_act' in c.lower() or 'trail_dist' in c.lower() for c in all_param_cols)
        # FIXED: solo tiene sl_pct y tp_pct (sin trailing)
        has_fixed = any('sl_pct' in c.lower() or 'tp_pct' in c.lower() for c in all_param_cols)
        
        console.print("\n[bold white]  TIPOS DE SALIDA DISPONIBLES:[/bold white]")
        console.print("  [white]1. FIXED[/white]    [grey70]- STOP LOSS (SL) + TAKE PROFIT (TP) FIJOS[/grey70]")
        console.print("  [white]2. TRAILING[/white] [grey70]- SL FIJO + TP CON TRAILING (ACTIVATION + DISTANCE)[/grey70]")
        
        if has_trailing:
            default = "2"
            console.print("\n[green]  ✓ DETECTADO: PARÁMETROS DE TRAILING[/green]")
        elif has_fixed:
            default = "1"
            console.print("\n[green]  ✓ DETECTADO: PARÁMETROS FIXED[/green]")
        else:
            default = "1"
        
        choice = Prompt.ask("\n  [bold white]SELECCIONA TIPO[/bold white]", choices=["1", "2"], default=default)
        self.exit_type = "fixed" if choice == "1" else "trailing"
        
        exit_desc = "SL + TP FIJOS" if self.exit_type == "fixed" else "SL FIJO + TP TRAILING"
        console.print(Panel(
            f"[bold white]{self.exit_type.upper()}[/bold white]\n[grey70]{exit_desc}[/grey70]",
            title="[grey70]◆ TIPO DE SALIDA SELECCIONADO[/grey70]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        
        return self.exit_type
    
    def detect_columns(self) -> bool:
        """Detecta y clasifica las columnas según el tipo de salida."""
        console.print(UI.section("ANÁLISIS DE ESTRUCTURA"))
        
        with create_progress() as progress:
            task = progress.add_task("[white]DETECTANDO COLUMNAS...", total=100)
            
            # 1. Obtener todas las columnas param_
            all_param_cols = [c for c in self.df.columns if c.startswith('param_')]
            progress.update(task, completed=20)
            
            # 2. Separar parámetros de salida - SIEMPRE detectar todos los tipos de salida
            # Incluir: SL, TP fijos, y también trailing (activation + distance)
            exit_keywords = [
                'exit_sl', 'sl_pct', 'stop_loss',           # Stop Loss
                'exit_tp', 'tp_pct', 'take_profit',          # Take Profit fijo
                'trail_act', 'trailing_act', 'activation',   # Trailing Activation
                'trail_dist', 'trailing_dist',               # Trailing Distance
                'trail_pct',                                  # Trailing general
            ]
            
            self.exit_columns = [c for c in all_param_cols 
                                if any(x in c.lower() for x in exit_keywords)]
            
            progress.update(task, completed=40)
            
            # 3. Parámetros de estrategia (todo lo que no es salida)
            self.strategy_columns = [c for c in all_param_cols if c not in self.exit_columns]
            progress.update(task, completed=60)
            
            # 4. Filtrar parámetros sin variación (min == max)
            valid_params = []
            excluded_params = []
            
            for col in self.strategy_columns + self.exit_columns:
                try:
                    # Usar dropna para evitar problemas con valores nulos
                    col_data = self.df[col].dropna().to_numpy()
                    if len(col_data) > 0:
                        col_min, col_max = float(col_data.min()), float(col_data.max())
                        if col_max > col_min:
                            valid_params.append(col)
                        else:
                            excluded_params.append((col, col_min))
                    else:
                        excluded_params.append((col, None))
                except Exception as e:
                    excluded_params.append((col, None))
            
            progress.update(task, completed=80)
            
            # 5. Actualizar listas
            self.strategy_columns = [c for c in self.strategy_columns if c in valid_params]
            self.exit_columns = [c for c in self.exit_columns if c in valid_params]
            self.param_columns = self.strategy_columns + self.exit_columns
            
            # 6. Detectar métricas (usar set para evitar duplicados)
            metric_candidates = ['roi', 'sharpe', 'sqn', 'drawdown', 'winrate', 'profit_factor', 'score']
            self.metric_columns = list(set(c for c in self.df.columns if c.lower() in metric_candidates))
            
            progress.update(task, completed=100)
        
        # Mostrar parámetros excluidos
        if excluded_params:
            console.print(f"\n[yellow]  ⚠ PARÁMETROS EXCLUIDOS (SIN VARIACIÓN):[/yellow]")
            for col, val in excluded_params:
                col_clean = col.replace('param_', '').upper()
                val_str = f"{val:.4f}" if val is not None else "N/A"
                console.print(f"[grey50]    • {col_clean} = {val_str} (VALOR FIJO)[/grey50]")
        
        # Mostrar configuración
        exit_desc = "SL + TP FIJOS" if self.exit_type == "fixed" else "SL + TP TRAILING (ACT + DIST)"
        console.print(Panel(
            f"[bold white]TIPO SALIDA:[/bold white] {self.exit_type.upper()} ({exit_desc})",
            title="[grey70]◆ CONFIGURACIÓN[/grey70]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        
        # Tabla de parámetros de estrategia
        if self.strategy_columns:
            table_strat = Table(
                title="[bold white]◆ PARÁMETROS DE ESTRATEGIA[/bold white]",
                box=box.ROUNDED,
                border_style="grey50",
                header_style="bold grey70"
            )
            table_strat.add_column("#", style="grey50", justify="right")
            table_strat.add_column("PARÁMETRO", style="white")
            table_strat.add_column("MÍN", style="white", justify="right")
            table_strat.add_column("MÁX", style="white", justify="right")
            table_strat.add_column("STEP", style="grey70", justify="right")
            table_strat.add_column("TIPO", style="grey50", justify="center")
            
            type_icons = {'period': 'PERIODO', 'distance': 'DISTANCIA', 'percentage': '%', 'other': 'OTRO'}
            
            for i, col in enumerate(self.strategy_columns, 1):
                col_clean = col.replace('param_', '').upper()
                col_data = self.df[col].dropna().to_numpy()
                col_min, col_max = float(col_data.min()), float(col_data.max())
                step = self._detect_param_step(col)
                param_type = self._detect_param_type(col)
                step_str = f"{step:.0f}" if step >= 1 else f"{step:.2f}"
                
                table_strat.add_row(str(i), col_clean, f"{col_min:.2f}", f"{col_max:.2f}",
                                   step_str, type_icons.get(param_type, 'OTRO'))
            console.print(table_strat)
        
        # Tabla de parámetros de salida
        if self.exit_columns:
            table_exit = Table(
                title="[bold white]◆ PARÁMETROS DE SALIDA[/bold white]",
                box=box.ROUNDED,
                border_style="grey50",
                header_style="bold grey70"
            )
            table_exit.add_column("PARÁMETRO", style="white")
            table_exit.add_column("MÍN", style="white", justify="right")
            table_exit.add_column("MÁX", style="white", justify="right")
            table_exit.add_column("STEP", style="grey70", justify="right")
            table_exit.add_column("TIPO", style="grey50", justify="center")
            
            type_map = {
                'exit_trailing_act': "TP ACTIVATION",
                'exit_trailing_dist': "TP DISTANCE",
                'exit_trailing': "TRAILING",
                'exit_fixed_sl': "STOP LOSS",
                'exit_fixed_tp': "TAKE PROFIT",
            }
            
            for col in self.exit_columns:
                col_clean = col.replace('param_', '').upper()
                col_data = self.df[col].dropna().to_numpy()
                col_min, col_max = float(col_data.min()), float(col_data.max())
                step = self._detect_param_step(col)
                param_type = self._detect_param_type(col)
                step_str = f"{step:.0f}" if step >= 1 else f"{step:.2f}"
                type_str = type_map.get(param_type, "EXIT")
                
                table_exit.add_row(col_clean, f"{col_min:.2f}", f"{col_max:.2f}", step_str, type_str)
            console.print(table_exit)
        
        # Tabla de métricas
        if self.metric_columns:
            table_metrics = Table(
                title="[bold white]◆ MÉTRICAS DETECTADAS[/bold white]",
                box=box.ROUNDED,
                border_style="grey50",
                header_style="bold grey70"
            )
            table_metrics.add_column("#", style="grey50", justify="right")
            table_metrics.add_column("MÉTRICA", style="white")
            table_metrics.add_column("MÍN", style="red", justify="right")
            table_metrics.add_column("MÁX", style="green", justify="right")
            table_metrics.add_column("MEDIA", style="white", justify="right")
            
            for i, col in enumerate(self.metric_columns, 1):
                col_data = self.df[col].dropna().to_numpy()
                if len(col_data) > 0:
                    col_min, col_max, col_mean = float(col_data.min()), float(col_data.max()), float(col_data.mean())
                    table_metrics.add_row(str(i), col.upper(), f"{col_min:.2f}", 
                                         f"{col_max:.2f}", f"{col_mean:.2f}")
            console.print(table_metrics)
        
        # Validar
        if not self.param_columns:
            console.print("[red]❌ No se detectaron parámetros válidos[/red]")
            return False
        
        if not self.metric_columns:
            console.print("[red]❌ No se detectaron métricas[/red]")
            return False
        
        return True
    
    def select_metrics(self) -> List[str]:
        """Selecciona métricas a analizar."""
        console.print("\n[bold white]  MÉTRICAS DISPONIBLES:[/bold white]")
        for i, col in enumerate(self.metric_columns, 1):
            col_data = self.df[col].dropna().to_numpy()
            if len(col_data) > 0:
                col_min, col_max = float(col_data.min()), float(col_data.max())
                console.print(f"  [white]{i}. {col.upper()}[/white] [grey50][{col_min:.2f} → {col_max:.2f}][/grey50]")
        
        console.print("\n[grey70]  NÚMEROS SEPARADOS POR COMA, O ENTER PARA TODAS[/grey70]")
        selection = Prompt.ask("  [bold white]SELECCIÓN[/bold white]", default="")
        
        if not selection:
            self.selected_metrics = list(self.metric_columns)
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(',')]
                self.selected_metrics = [self.metric_columns[i-1] for i in indices 
                                        if 1 <= i <= len(self.metric_columns)]
            except:
                self.selected_metrics = list(self.metric_columns)
        
        metrics_str = " | ".join([f"[white]{m.upper()}[/white]" for m in self.selected_metrics])
        console.print(Panel(
            metrics_str,
            title="[grey70]◆ MÉTRICAS SELECCIONADAS[/grey70]",
            border_style="grey50",
            box=box.ROUNDED
        ))
        return self.selected_metrics
    
    def _train_single_metric(self, metric: str, X: np.ndarray, valid_mask: np.ndarray, 
                               n_grid_points: int) -> Optional[Tuple[str, GPROptimizer, GPRResult, float]]:
        """
        Entrena GPR para una métrica individual.
        Usado por ThreadPoolExecutor para paralelización (Optimización 7).
        """
        y = self.df[metric].to_numpy()[valid_mask].astype(np.float64)
        metric_valid = ~(np.isnan(y) | np.isinf(y))
        X_clean, y_clean = X[metric_valid], y[metric_valid]
        
        if len(X_clean) < 5:
            return None
        
        metric_start = time.perf_counter()
        
        # Entrenar GPR
        gpr = GPROptimizer(n_restarts=5)
        gpr.fit(X_clean, y_clean, self.param_columns)
        
        # R² de entrenamiento
        r2_train = gpr.compute_r2(X_clean, y_clean)
        
        # Optimización 5: Calcular TODAS las dependencias parciales en UNA llamada
        predictions = compute_all_partial_dependences(
            gpr, X_clean, y_clean, self.param_columns, n_grid_points
        )
        
        # Guardar resultados
        length_scales = {p: gpr.learned_length_scales[i] 
                       for i, p in enumerate(self.param_columns)
                       } if len(gpr.learned_length_scales) == len(self.param_columns) else {}
        
        result = GPRResult(
            metric=metric, r2_score=r2_train, r2_cv_score=r2_train, 
            noise_level=gpr.learned_noise, length_scales=length_scales, 
            predictions=predictions, is_overfit=False
        )
        
        total_time = time.perf_counter() - metric_start
        return (metric, gpr, result, total_time)
    
    def run_gpr_analysis(self, n_grid_points: int = 50) -> Dict[str, GPRResult]:
        """
        Ejecuta análisis GPR Ultra-Optimizado.
        
        Optimizaciones aplicadas:
        - Optimización 5: Una sola predicción para todas las PDs por métrica
        - Optimización 7: Entrenamiento paralelo de múltiples métricas (ThreadPool)
        """
        console.print(UI.section("ANÁLISIS GPR - DESIONIZACIÓN BAYESIANA"))
        tracker.start_process('gpr_analysis', 'ANÁLISIS GPR')
        
        # Mostrar configuración del motor
        console.print(tracker.get_config_panel())
        
        # Convertir a NumPy una vez (más eficiente que acceso repetido)
        X = self.df[self.param_columns].to_numpy().astype(np.float64)
        
        # Eliminar NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        
        n_metrics = len(self.selected_metrics)
        n_samples_total = len(X)
        
        # Info de datos
        console.print(f"\n[grey70]  DATOS DE ENTRADA:[/grey70]")
        console.print(f"[white]    • SAMPLES TOTALES:      {n_samples_total:,}[/white]")
        console.print(f"[white]    • PARÁMETROS:           {len(self.param_columns)}[/white]")
        console.print(f"[white]    • MÉTRICAS A ANALIZAR:  {n_metrics}[/white]")
        
        if SPARSE_GP_ENABLED and n_samples_total > SPARSE_INDUCING_POINTS:
            console.print(f"[green]    • MODO:                 SPARSE GP ({SPARSE_INDUCING_POINTS:,} INDUCING POINTS)[/green]")
            console.print(f"[grey50]    • REDUCCIÓN:            {100*(1-SPARSE_INDUCING_POINTS/n_samples_total):.0f}% MENOS COMPLEJIDAD[/grey50]")
        else:
            console.print(f"[yellow]    • MODO:                 EXACT GP (TODOS LOS DATOS)[/yellow]")
        
        console.print("")
        
        # Optimización 7: Paralelizar entrenamiento de métricas si hay múltiples
        if n_metrics > 1 and N_JOBS > 1:
            console.print(f"[white]  ENTRENAMIENTO PARALELO: {n_metrics} MÉTRICAS EN {min(N_JOBS, n_metrics)} THREADS[/white]")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with create_progress() as progress:
                main_task = progress.add_task("[white]ANÁLISIS GPR (PARALELO)", total=n_metrics)
                
                # Usar ThreadPoolExecutor (GPyTorch libera GIL en operaciones pesadas)
                with ThreadPoolExecutor(max_workers=min(N_JOBS, n_metrics)) as executor:
                    futures = {
                        executor.submit(
                            self._train_single_metric, metric, X, valid_mask, n_grid_points
                        ): metric 
                        for metric in self.selected_metrics
                    }
                    
                    for future in as_completed(futures):
                        metric = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                metric_name, gpr, gpr_result, total_time = result
                                self.gpr_models[metric_name] = gpr
                                self.gpr_results[metric_name] = gpr_result
                                
                            # Mostrar resultado con formato profesional
                                console.print(UI.metric_result(
                                    metric=metric_name,
                                    r2=gpr_result.r2_score,
                                    r2_cv=gpr_result.r2_cv_score,
                                    noise=gpr_result.noise_level,
                                    model_type="SPARSE" if gpr.is_sparse else "EXACT",
                                    n_samples=gpr.original_size,
                                    n_inducing=gpr.n_inducing if gpr.is_sparse else 0,
                                    time_elapsed=total_time,
                                    is_overfit=gpr_result.is_overfit
                                ))
                            else:
                                console.print(f"[yellow]  ⚠ {metric.upper()}: DATOS INSUFICIENTES[/yellow]")
                        except Exception as e:
                            console.print(f"[red]  ✗ {metric.upper()}: ERROR - {e}[/red]")
                        
                        progress.update(main_task, advance=1)
        else:
            # Ejecución secuencial (una métrica o N_JOBS=1)
            with create_progress() as progress:
                main_task = progress.add_task("[white]ANÁLISIS GPR", total=n_metrics)
                
                for metric in self.selected_metrics:
                    metric_start = time.perf_counter()
                    console.print(f"\n[bold white]  ┌─ {metric.upper()}[/bold white]")
                    
                    y = self.df[metric].to_numpy()[valid_mask].astype(np.float64)
                    metric_valid = ~(np.isnan(y) | np.isinf(y))
                    X_clean, y_clean = X[metric_valid], y[metric_valid]
                    
                    if len(X_clean) < 5:
                        console.print(f"[yellow]  │  ⚠ DATOS INSUFICIENTES ({len(X_clean)})[/yellow]")
                        progress.update(main_task, advance=1)
                        continue
                    
                    # Pausar progress para mostrar entrenamiento detallado
                    progress.stop()
                    
                    # Entrenar GPR
                    fit_start = time.perf_counter()
                    gpr = GPROptimizer(n_restarts=5)
                    gpr.fit(X_clean, y_clean, self.param_columns)
                    
                    # R² de entrenamiento
                    r2_train = gpr.compute_r2(X_clean, y_clean)
                    fit_time = time.perf_counter() - fit_start
                    
                    # Reanudar progress
                    progress.start()
                    
                    # Calidad basada en R²
                    if r2_train > 0.7:
                        quality = "[green]EXCELENTE[/green]"
                    elif r2_train > 0.4:
                        quality = "[yellow]MODERADO[/yellow]"
                    else:
                        quality = "[red]BAJO[/red]"
                    
                    console.print(f"[grey70]  │  R² = {r2_train:.4f} ({quality}) - {tracker.format_duration(fit_time)}[/grey70]")
                    console.print(f"[grey50]  │  σn = {gpr.learned_noise:.4f}[/grey50]")
                    
                    if gpr.is_sparse:
                        console.print(f"[grey50]  │  MODELO: SPARSE GP ({gpr.n_inducing:,} INDUCING DE {gpr.original_size:,})[/grey50]")
                    else:
                        console.print(f"[grey50]  │  MODELO: EXACT GP ({gpr.original_size:,} DATOS)[/grey50]")
                    
                    # Optimización 5: Calcular TODAS las PDs en UNA llamada
                    pd_start = time.perf_counter()
                    predictions = compute_all_partial_dependences(
                        gpr, X_clean, y_clean, self.param_columns, n_grid_points
                    )
                    pd_time = time.perf_counter() - pd_start
                    console.print(f"[dim]│  PD calc: {tracker.format_duration(pd_time)}[/dim]")
                    
                    # Guardar resultados
                    length_scales = {p: gpr.learned_length_scales[i] 
                                   for i, p in enumerate(self.param_columns)
                                   } if len(gpr.learned_length_scales) == len(self.param_columns) else {}
                    
                    self.gpr_models[metric] = gpr
                    self.gpr_results[metric] = GPRResult(
                        metric=metric, r2_score=r2_train, r2_cv_score=r2_train,
                        noise_level=gpr.learned_noise, length_scales=length_scales, 
                        predictions=predictions, is_overfit=False
                    )
                    
                    total_time = time.perf_counter() - metric_start
                    console.print(f"[dim]└─ ✓ {tracker.format_duration(total_time)}[/dim]")
                    progress.update(main_task, advance=1)
        
        tracker.end_process('gpr_analysis')
        return self.gpr_results
    
    def compute_parameter_metric_curves(self) -> Dict[str, Dict[str, ParameterMetricCurve]]:
        """
        Calcula las curvas de parámetro vs métrica con medias agrupadas.
        Para cada métrica seleccionada y cada parámetro:
        - Agrupa por valor único del parámetro
        - Calcula media y std de la métrica
        - Ajusta curva de tendencia polinomial
        - Encuentra el óptimo
        
        Returns:
            Dict[metric_name, Dict[param_name, ParameterMetricCurve]]
        """
        console.print(Rule("[bold blue]📊 ANÁLISIS DE PARÁMETROS INDIVIDUALES"))
        tracker.start_process('param_curves', 'Curvas Parámetro-Métrica')
        
        results: Dict[str, Dict[str, ParameterMetricCurve]] = {}
        
        for metric in self.selected_metrics:
            if metric not in self.df.columns:
                continue
            
            results[metric] = {}
            metric_data = self.df[metric].to_numpy()
            
            for param in self.param_columns:
                param_data = self.df[param].to_numpy()
                
                # Filtrar NaN
                valid_mask = ~(np.isnan(param_data) | np.isnan(metric_data))
                param_valid = param_data[valid_mask]
                metric_valid = metric_data[valid_mask]
                
                if len(param_valid) < 3:
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # MÉTODO: BINNING + MEDIA POR BIN
                # Divide el rango del parámetro en N_BINS segmentos y calcula
                # la media de la métrica en cada bin. Esto elimina TODO el ruido.
                # ═══════════════════════════════════════════════════════════════
                
                N_BINS = 25  # Número de segmentos (ajustar para más/menos suavidad)
                
                param_min = param_valid.min()
                param_max = param_valid.max()
                
                # Crear bins
                bin_edges = np.linspace(param_min, param_max, N_BINS + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calcular media y std por bin
                bin_means = []
                bin_stds = []
                bin_counts = []
                valid_bins = []
                valid_centers = []
                
                for i in range(N_BINS):
                    # Datos en este bin
                    in_bin = (param_valid >= bin_edges[i]) & (param_valid < bin_edges[i+1])
                    # Incluir el último valor en el último bin
                    if i == N_BINS - 1:
                        in_bin = (param_valid >= bin_edges[i]) & (param_valid <= bin_edges[i+1])
                    
                    values_in_bin = metric_valid[in_bin]
                    
                    if len(values_in_bin) > 0:
                        bin_means.append(np.mean(values_in_bin))
                        bin_stds.append(np.std(values_in_bin) if len(values_in_bin) > 1 else 0)
                        bin_counts.append(len(values_in_bin))
                        valid_bins.append(i)
                        valid_centers.append(bin_centers[i])
                
                if len(valid_centers) < 2:
                    continue
                
                bin_means = np.array(bin_means)
                bin_stds = np.array(bin_stds)
                bin_counts = np.array(bin_counts)
                valid_centers = np.array(valid_centers)
                
                # Guardar como unique_params para compatibilidad
                unique_params = valid_centers
                mean_metrics = bin_means
                std_metrics = bin_stds
                counts = bin_counts
                
                # Crear curva suave interpolando los bins
                from scipy.ndimage import gaussian_filter1d
                
                # Grid final para la curva
                trend_x = np.linspace(valid_centers.min(), valid_centers.max(), 200)
                
                # Interpolar y suavizar ligeramente
                trend_y = np.interp(trend_x, valid_centers, bin_means)
                
                # Suavizado gaussiano ligero para conectar los bins suavemente
                trend_y = gaussian_filter1d(trend_y, sigma=5)
                
                # Calcular R² (correlación entre bins)
                if len(bin_means) > 2:
                    ss_tot = np.sum((bin_means - np.mean(bin_means))**2)
                    y_pred = np.interp(valid_centers, trend_x, trend_y)
                    ss_res = np.sum((bin_means - y_pred)**2)
                    r2_trend = max(0, 1 - (ss_res / (ss_tot + 1e-10)))
                else:
                    r2_trend = 0.0
                
                # Banda de confianza
                std_interp = np.interp(trend_x, valid_centers, bin_stds)
                std_interp = gaussian_filter1d(std_interp, sigma=5)
                trend_ci_lower = trend_y - std_interp
                trend_ci_upper = trend_y + std_interp
                
                # Óptimo: máximo de la curva
                opt_idx = np.argmax(trend_y)
                optimal_param = trend_x[opt_idx]
                optimal_metric = trend_y[opt_idx]
                degree = N_BINS  # Indicar que usamos binning
                
                # Crear objeto resultado
                curve = ParameterMetricCurve(
                    param_name=param,
                    metric_name=metric,
                    unique_params=unique_params,
                    mean_metrics=mean_metrics,
                    std_metrics=std_metrics,
                    sample_counts=counts,
                    trend_x=trend_x,
                    trend_y=trend_y,
                    trend_ci_lower=trend_ci_lower,
                    trend_ci_upper=trend_ci_upper,
                    optimal_param=optimal_param,
                    optimal_metric=optimal_metric,
                    total_samples=len(param_valid),
                    trend_degree=degree,
                    r2_trend=r2_trend
                )
                
                results[metric][param] = curve
        
        self.param_metric_curves = results
        
        # Mostrar resumen
        total_curves = sum(len(params) for params in results.values())
        console.print(f"[dim]   📈 {total_curves} curvas calculadas ({len(results)} métricas × {len(self.param_columns)} parámetros)[/dim]")
        
        tracker.end_process('param_curves')
        return results
    
    def generate_all_figures(self):
        """Genera figuras en paralelo."""
        console.print(Rule("[bold blue]🎨 GENERACIÓN DE VISUALIZACIONES"))
        tracker.start_process('visualization', 'Visualizaciones')
        fig_start = time.perf_counter()
        
        t0 = time.perf_counter()
        X = self.df[self.param_columns].to_numpy().astype(np.float64)
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        console.print(f"[dim]   📊 Datos para figuras: {X.shape[0]:,} × {X.shape[1]} ({(time.perf_counter()-t0)*1000:.1f}ms)[/dim]")
        
        tasks = []
        
        # Preparar tareas de heatmap
        if len(self.param_columns) >= 2:
            for metric, gpr in self.gpr_models.items():
                tasks.append(('heatmap', metric, gpr, X))
        
        # Preparar tareas de incertidumbre
        for metric, result in self.gpr_results.items():
            for param, pred in result.predictions.items():
                tasks.append(('uncertainty', metric, pred, result.r2_score, result.noise_level))
        
        n_heatmaps = sum(1 for t in tasks if t[0] == 'heatmap')
        n_uncertainty = len(tasks) - n_heatmaps
        n_workers = min(N_JOBS * 2, len(tasks), 8)
        
        console.print(f"[dim]   📈 Tareas: {n_heatmaps} heatmaps + {n_uncertainty} uncertainty plots[/dim]")
        console.print(f"[dim]   ⚡ Workers paralelos: {n_workers}[/dim]")
        
        with create_progress() as progress:
            main_task = progress.add_task("[cyan]Generando figuras", total=len(tasks))
            
            def process_task(task):
                task_start = time.perf_counter()
                if task[0] == 'heatmap':
                    _, metric, gpr, X_data = task
                    result = _generate_heatmap(gpr, X_data, metric, 0, 1, self.param_columns)
                else:
                    _, metric, pred, r2, noise = task
                    result = _generate_uncertainty_plot(pred, metric, r2, noise)
                return result
            
            # Generar en paralelo con threads (matplotlib no es process-safe)
            t0 = time.perf_counter()
            results = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(process_task)(task) for task in tasks
            )
            t_parallel = time.perf_counter() - t0
            
            for key, img_b64 in results:
                self.figures_base64[key] = img_b64
                progress.update(main_task, advance=1)
        
        fig_total = time.perf_counter() - fig_start
        console.print(f"[dim]   ⏱️  Renderizado paralelo: {t_parallel:.2f}s[/dim]")
        console.print(f"[dim]   📦 Tamaño total figuras: {sum(len(v) for v in self.figures_base64.values()) / 1024:.1f} KB (base64)[/dim]")
        
        tracker.end_process('visualization')
        console.print(f"[bold green]   🏁 {len(self.figures_base64)} figuras generadas en {fig_total:.2f}s[/bold green]")
    
    def _get_html_template(self) -> str:
        """Template HTML para PDF."""
        return '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Análisis GPR</title>
    <style>
        @page { size: A4; margin: 2cm; @bottom-center { content: "Página " counter(page); font-size: 10px; } }
        body { font-family: Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }
        h1 { color: #1f77b4; text-align: center; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }
        h2 { color: #2c3e50; border-bottom: 1px solid #ddd; margin-top: 25px; }
        h3 { color: #34495e; }
        .cover { text-align: center; padding: 80px 0; page-break-after: always; }
        .cover h1 { font-size: 28pt; border: none; }
        .cover .subtitle { font-size: 16pt; color: #666; margin-top: 15px; }
        .cover .info { margin-top: 40px; color: #888; }
        .methodology { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; page-break-after: always; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 10pt; }
        th { background: #1f77b4; color: white; padding: 8px; }
        td { padding: 6px; border: 1px solid #ddd; text-align: center; }
        tr:nth-child(even) { background: #f8f9fa; }
        .metric-good { color: #27ae60; font-weight: bold; }
        .metric-moderate { color: #f39c12; font-weight: bold; }
        .metric-low { color: #e74c3c; font-weight: bold; }
        .figure-container { text-align: center; margin: 15px 0; page-break-inside: avoid; }
        .figure-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .figure-caption { font-size: 9pt; color: #666; margin-top: 5px; font-style: italic; }
        .summary-box { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .summary-box h3 { color: white; margin-top: 0; }
        .page-break { page-break-before: always; }
    </style>
</head>
<body>
    <div class="cover">
        <h1>ANÁLISIS DE TRADING</h1>
        <div class="subtitle">Desionización Bayesiana (GPR)</div>
        <div class="info">
            <p><strong>Archivo:</strong> {{ filename }}</p>
            <p><strong>Trials:</strong> {{ n_trials }} | <strong>Parámetros:</strong> {{ n_params }}</p>
            <p><strong>Fecha:</strong> {{ date }}</p>
        </div>
    </div>
    
    <div class="methodology">
        <h2>Metodología</h2>
        <h3>Kernel GPR</h3>
        <p style="text-align: center; font-size: 13pt;"><code>ConstantKernel × Matérn(ν=2.5) + WhiteKernel</code></p>
        <ul>
            <li><strong>Matérn (ν=2.5):</strong> Modela rugosidad de superficie de parámetros</li>
            <li><strong>WhiteKernel:</strong> Aísla ruido (desionización)</li>
        </ul>
        <h3>Dependencia Parcial</h3>
        <p style="text-align: center;">ROI<sub>limpio</sub>(x<sub>j</sub>) ≈ (1/N) Σ f̂(x<sub>j</sub>, x<sub>i,\\j</sub>)</p>
    </div>
    
    <h2>Resumen GPR</h2>
    <table>
        <tr><th>Métrica</th><th>R² Train</th><th>R² CV</th><th>Calidad</th><th>σn</th><th>Estado</th></tr>
        {% for metric, result in results.items() %}
        <tr>
            <td><strong>{{ metric.upper() }}</strong></td>
            <td>{{ "%.4f"|format(result.r2_score) }}</td>
            <td>{{ "%.4f"|format(result.r2_cv_score) }}</td>
            <td class="{{ 'metric-good' if result.r2_cv_score > 0.7 else 'metric-moderate' if result.r2_cv_score > 0.4 else 'metric-low' }}">
                {{ 'EXCELENTE' if result.r2_cv_score > 0.7 else 'MODERADO' if result.r2_cv_score > 0.4 else 'BAJO' }}
            </td>
            <td>{{ "%.4f"|format(result.noise_level) }}</td>
            <td class="{{ 'metric-low' if result.is_overfit else 'metric-good' }}">
                {{ '⚠ SOBREAJUSTE' if result.is_overfit else '✓ OK' }}
            </td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Parámetros Óptimos</h2>
    {% for metric, result in results.items() %}
    <h3>{{ metric.upper() }}</h3>
    <table>
        <tr><th>Parámetro</th><th>Valor Óptimo</th><th>Predicción</th><th>IC 95%</th></tr>
        {% for param, pred in result.predictions.items() %}
        <tr>
            <td>{{ param.replace('param_', '').replace('_', ' ').title() }}</td>
            <td>{{ "%.2f"|format(optimal[metric][param]['value']) }}</td>
            <td class="metric-good">{{ "%.2f"|format(optimal[metric][param]['pred']) }}</td>
            <td>[{{ "%.2f"|format(optimal[metric][param]['ci_l']) }} - {{ "%.2f"|format(optimal[metric][param]['ci_u']) }}]</td>
        </tr>
        {% endfor %}
    </table>
    {% endfor %}
    
    <div class="page-break"></div>
    <h2>Visualizaciones</h2>
    {% for metric in results %}
    <h3>{{ metric.upper() }}</h3>
    {% if figures.get('heatmap_' + metric) %}
    <div class="figure-container">
        <img src="data:image/png;base64,{{ figures['heatmap_' + metric] }}" alt="Heatmap">
        <div class="figure-caption">Superficie Desionizada: {{ metric.upper() }}</div>
    </div>
    {% endif %}
    {% for param in params %}
    {% set key = 'uncertainty_' + metric + '_' + param %}
    {% if figures.get(key) %}
    <div class="figure-container">
        <img src="data:image/png;base64,{{ figures[key] }}" alt="Incertidumbre">
        <div class="figure-caption">{{ metric.upper() }} vs {{ param.replace('param_', '').title() }}</div>
    </div>
    {% endif %}
    {% endfor %}
    <div class="page-break"></div>
    {% endfor %}
    
    <div class="summary-box">
        <h3>Interpretación</h3>
        <ul>
            <li><strong>R² > 0.7:</strong> Modelo excelente, predicciones confiables</li>
            <li><strong>R² 0.4-0.7:</strong> Usar con precaución</li>
            <li><strong>R² < 0.4:</strong> Alta incertidumbre</li>
        </ul>
    </div>
</body>
</html>'''
    
    def generate_pdf_report(self, output_path: str = None) -> str:
        """Genera PDF con Jinja2 + WeasyPrint."""
        console.print(Rule("[bold blue]Generando PDF"))
        tracker.start_process('pdf', 'Generación PDF')
        
        if output_path is None:
            base = Path(self.filepath).stem if self.filepath else "analisis"
            output_path = f"reporte_gpr_{base}.pdf"
        if not output_path.endswith('.pdf'):
            output_path += '.pdf'
        
        if not self.figures_base64:
            self.generate_all_figures()
        
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
        
        context = {
            'filename': Path(self.filepath).name if self.filepath else "N/A",
            'n_trials': len(self.df),
            'n_params': len(self.param_columns),
            'date': datetime.now().strftime("%d/%m/%Y %H:%M"),
            'results': self.gpr_results,
            'figures': self.figures_base64,
            'params': self.param_columns,
            'optimal': optimal,
        }
        
        try:
            from jinja2 import Template
            html = Template(self._get_html_template()).render(**context)
            
            from weasyprint import HTML
            HTML(string=html).write_pdf(output_path)
            
            tracker.end_process('pdf')
            console.print(Panel(f"[green]✅ PDF: {output_path}[/green]", border_style="green"))
            return output_path
            
        except ImportError as e:
            tracker.end_process('pdf', 'failed')
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w') as f:
                f.write(html)
            console.print(f"[yellow]⚠ WeasyPrint no disponible. HTML: {html_path}[/yellow]")
            return html_path
        except Exception as e:
            tracker.end_process('pdf', 'failed')
            console.print(f"[red]❌ Error PDF: {e}[/red]")
            return ""
    
    def export_to_excel(self, output_path: str = None) -> str:
        """Exporta a Excel con Pandas (desde Parquet)."""
        console.print(Rule("[bold blue]Exportando Excel"))
        tracker.start_process('excel', 'Exportación Excel')
        
        if output_path is None:
            base = Path(self.filepath).stem if self.filepath else "analisis"
            output_path = f"resultados_gpr_{base}.xlsx"
        
        try:
            import pandas as pd
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Resumen
                resumen = [{
                    'Métrica': m.upper(),
                    'R²_Train': r.r2_score,
                    'R²_CV': r.r2_cv_score,
                    'Calidad': 'EXCELENTE' if r.r2_cv_score > 0.7 else 'MODERADO' if r.r2_cv_score > 0.4 else 'BAJO',
                    'σn': r.noise_level,
                    'Sobreajuste': '⚠ SÍ' if r.is_overfit else 'NO',
                    'Δ_R²': r.r2_score - r.r2_cv_score
                } for m, r in self.gpr_results.items()]
                pd.DataFrame(resumen).to_excel(writer, sheet_name='Resumen', index=False)
                
                # Predicciones
                for metric, result in self.gpr_results.items():
                    data = []
                    for param, pred in result.predictions.items():
                        for i in range(len(pred.param_values)):
                            data.append({
                                'Parámetro': param.replace('param_', ''),
                                'Valor': pred.param_values[i],
                                'Predicción': pred.mean_prediction[i],
                                'IC95_Inf': pred.lower_ci[i],
                                'IC95_Sup': pred.upper_ci[i]
                            })
                    pd.DataFrame(data).to_excel(writer, sheet_name=f'Pred_{metric[:8]}', index=False)
                
                # Óptimos
                optimos = []
                for metric, result in self.gpr_results.items():
                    for param, pred in result.predictions.items():
                        idx = np.argmax(pred.mean_prediction)
                        optimos.append({
                            'Métrica': metric.upper(),
                            'Parámetro': param.replace('param_', ''),
                            'Valor_Óptimo': pred.param_values[idx],
                            'Predicción': pred.mean_prediction[idx]
                        })
                pd.DataFrame(optimos).to_excel(writer, sheet_name='Óptimos', index=False)
            
            tracker.end_process('excel')
            console.print(Panel(f"[green]✅ Excel: {output_path}[/green]", border_style="green"))
            return output_path
            
        except Exception as e:
            tracker.end_process('excel', 'failed')
            console.print(f"[red]❌ Error Excel: {e}[/red]")
            return ""


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis GPR de Trading')
    parser.add_argument('archivo', nargs='?', help='Archivo CSV/Excel a analizar')
    parser.add_argument('--exit-type', choices=['fixed', 'trailing'], help='Tipo de salida')
    args = parser.parse_args()
    
    console.clear()
    
    # Header profesional
    sparse_info = "SPARSE GP ACTIVADO" if SPARSE_GP_ENABLED else "EXACT GP"
    console.print(Panel(
        f"[bold white]SISTEMA DE ANÁLISIS DE TRADING[/bold white]\n"
        f"[grey70]DESIONIZACIÓN BAYESIANA (GPR) - ULTRA OPTIMIZADO[/grey70]\n\n"
        f"[white]BACKEND:[/white] [grey70]PYARROW/PARQUET + NUMBA + JOBLIB ({N_JOBS} CORES)[/grey70]\n"
        f"[white]MOTOR:[/white] [grey70]GPYTORCH (CPU) - {sparse_info}[/grey70]\n"
        f"[white]INDUCING:[/white] [grey70]{SPARSE_INDUCING_POINTS:,} POINTS[/grey70]",
        border_style="grey50",
        box=box.DOUBLE,
        padding=(1, 4)
    ))
    
    tracker.start_session()
    session_start = time.perf_counter()
    
    analyzer = BayesianDenoisingAnalyzer()
    
    # PASO 1: Cargar
    console.print(UI.section("PASO 1: CARGAR ARCHIVO"))
    
    if args.archivo:
        filepath = args.archivo
        console.print(f"[grey70]  ARCHIVO: {filepath}[/grey70]")
        if not analyzer.load_data(filepath):
            console.print("[red]  ERROR AL CARGAR ARCHIVO[/red]")
            return
    else:
        console.print("[white]  ARRASTRA EL ARCHIVO CSV/EXCEL:[/white]\n")
        while True:
            filepath = Prompt.ask("  [bold white]ARCHIVO[/bold white]")
            if analyzer.load_data(filepath):
                break
            console.print("[red]  INTENTA DE NUEVO.[/red]\n")
    
    # PASO 2: Seleccionar tipo de salida
    console.print(UI.section("PASO 2: CONFIGURACIÓN"))
    if args.exit_type:
        analyzer.exit_type = args.exit_type
        console.print(f"[green]  ✓ TIPO DE SALIDA: {args.exit_type.upper()}[/green]")
    else:
        analyzer.select_exit_type()
    
    # PASO 3: Detectar columnas
    console.print(UI.section("PASO 3: DETECCIÓN DE COLUMNAS"))
    if not analyzer.detect_columns():
        return
    
    # PASO 4: Seleccionar métricas
    analyzer.select_metrics()
    
    # PASO 5: GPR
    console.print(UI.section("PASO 5: ANÁLISIS GPR"))
    if not Confirm.ask("[white]  ¿INICIAR ANÁLISIS?[/white]", default=True):
        return
    
    analyzer.run_gpr_analysis(n_grid_points=30)
    
    if not analyzer.gpr_results:
        console.print("[red]  ERROR: SIN RESULTADOS[/red]")
        return
    
    # PASO 6: Visualizaciones
    console.print(UI.section("PASO 6: VISUALIZACIONES"))
    analyzer.generate_all_figures()
    
    # PASO 6.5: Análisis de parámetros individuales
    console.print(UI.section("PASO 6.5: ANÁLISIS PARÁMETROS INDIVIDUALES"))
    analyzer.compute_parameter_metric_curves()
    
    # PASO 7: PDF
    console.print(UI.section("PASO 7: GENERACIÓN PDF"))
    base = Path(analyzer.filepath).stem
    
    # Preguntar tipo de PDF (con timeout de 18s)
    pdf_type = prompt_with_timeout(
        "  [white]TIPO DE REPORTE[/white]",
        default="profesional",
        timeout=18,
        choices=["profesional", "simple"]
    )
    
    if pdf_type == "profesional":
        pdf_path = prompt_with_timeout(
            "  [white]ARCHIVO PDF[/white]", 
            default=f"reporte_gpr_profesional_{base}.pdf",
            timeout=18
        )
        try:
            from visual.pdf_profesional import generate_professional_report
            pdf_result = generate_professional_report(
                gpr_results=analyzer.gpr_results,
                gpr_models=analyzer.gpr_models,
                df=analyzer.df,
                param_columns=analyzer.param_columns,
                filepath=analyzer.filepath,
                output_path=pdf_path,
                param_metric_curves=analyzer.param_metric_curves  # NUEVO
            )
        except ImportError as e:
            console.print(f"[yellow]⚠ Error importando PDF profesional: {e}[/yellow]")
            console.print("[yellow]  Usando generador simple...[/yellow]")
            pdf_result = analyzer.generate_pdf_report(pdf_path)
        except Exception as e:
            console.print(f"[red]❌ Error generando PDF profesional: {e}[/red]")
            console.print("[yellow]  Usando generador simple...[/yellow]")
            pdf_result = analyzer.generate_pdf_report(pdf_path)
    else:
        pdf_path = prompt_with_timeout(
            "  [white]ARCHIVO PDF[/white]", 
            default=f"reporte_gpr_{base}.pdf",
            timeout=18
        )
        pdf_result = analyzer.generate_pdf_report(pdf_path)
    
    # PASO 8: Excel
    if Confirm.ask("\n[white]  ¿EXPORTAR EXCEL?[/white]", default=True):
        excel_path = Prompt.ask("  [white]ARCHIVO EXCEL[/white]", default=pdf_path.replace('.pdf', '.xlsx'))
        analyzer.export_to_excel(excel_path)
    
    # Resumen
    console.print("\n")
    console.print(tracker.get_summary_table())
    
    total = time.perf_counter() - session_start
    console.print(Panel(
        f"[bold white]TIEMPO TOTAL: {tracker.format_duration(total)}[/bold white]",
        title="[grey70]◆ COMPLETADO[/grey70]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    # Abrir PDF
    if pdf_result and Confirm.ask("\n[white]  ¿ABRIR REPORTE?[/white]", default=True):
        import subprocess
        if sys.platform == 'darwin':
            subprocess.run(['open', pdf_result])
        elif sys.platform == 'win32':
            os.startfile(pdf_result)
        else:
            subprocess.run(['xdg-open', pdf_result])
    
    # Limpiar archivo temporal
    if analyzer.temp_parquet_path:
        analyzer.cleanup()
        console.print("[grey50]  ARCHIVO TEMPORAL ELIMINADO[/grey50]")
    
    console.print(UI.section("COMPLETADO"))
    
    # LIMPIEZA TOTAL DE MEMORIA Y RECURSOS
    full_system_cleanup()
    console.print("[green]✅ MEMORIA Y RECURSOS LIBERADOS[/green]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🔬 PROFILER DETALLADO PARA analisis2.py
========================================
Mide tiempos de ejecución con máximo detalle para identificar:
- Cuellos de botella (embudos)
- Bucles lentos
- Funciones que consumen más tiempo
- Puntos a optimizar

Uso: python profile_analisis.py [archivo_csv]
"""

import cProfile
import pstats
import io
import time
import sys
import functools
import tracemalloc
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import importlib.util

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
ARCHIVO_PRUEBA = "/Users/manuel/Desktop/MODELOX/RESUMEN_UNKNOWN_MOMEN_unknown.xlsx"  # Archivo Excel
MUESTRA_SIZE = 500  # Tamaño de muestra
TOP_FUNCIONES = 200  # Mostrar top N funciones más lentas
MEMORIA_TRACKING = True  # Trackear uso de memoria

# ============================================================================
# COLORES ANSI PARA TERMINAL
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def cprint(text: str, color: str = Colors.END):
    """Print con color."""
    print(f"{color}{text}{Colors.END}")

# ============================================================================
# TIMING DETALLADO
# ============================================================================
class DetailedTimer:
    """Tracker de tiempos con máximo detalle."""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.current_stack: List[Tuple[str, float]] = []
        self.hierarchy: Dict[str, Dict] = {}
        self.start_time = None
        self.memory_snapshots: Dict[str, int] = {}
        
    def start(self, name: str):
        """Inicia timer para una operación."""
        self.current_stack.append((name, time.perf_counter()))
        self.call_counts[name] += 1
        if MEMORIA_TRACKING:
            current, peak = tracemalloc.get_traced_memory()
            self.memory_snapshots[f"{name}_start"] = current
        
    def stop(self, name: str) -> float:
        """Para timer y retorna duración."""
        if not self.current_stack:
            return 0.0
        
        stack_name, start_time = self.current_stack.pop()
        if stack_name != name:
            cprint(f"⚠️  Timer mismatch: esperado '{name}', encontrado '{stack_name}'", Colors.YELLOW)
        
        elapsed = time.perf_counter() - start_time
        self.times[name].append(elapsed)
        
        if MEMORIA_TRACKING:
            current, peak = tracemalloc.get_traced_memory()
            self.memory_snapshots[f"{name}_end"] = current
        
        return elapsed
    
    @contextmanager
    def measure(self, name: str):
        """Context manager para medir tiempo."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_stats(self) -> Dict[str, Dict]:
        """Obtiene estadísticas de todos los timers."""
        stats = {}
        for name, times in self.times.items():
            if times:
                stats[name] = {
                    'total': sum(times),
                    'count': len(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'calls': self.call_counts[name]
                }
        return stats
    
    def print_report(self):
        """Imprime reporte detallado."""
        stats = self.get_stats()
        if not stats:
            cprint("No hay datos de timing.", Colors.YELLOW)
            return
        
        # Ordenar por tiempo total
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        total_time = sum(s['total'] for s in stats.values())
        
        cprint("\n" + "="*80, Colors.BLUE)
        cprint("📊 REPORTE DE TIEMPOS DETALLADO", Colors.BOLD + Colors.CYAN)
        cprint("="*80, Colors.BLUE)
        
        cprint(f"\n⏱️  Tiempo total rastreado: {total_time:.4f}s", Colors.GREEN)
        
        # Tabla de tiempos
        cprint("\n┌" + "─"*78 + "┐", Colors.DIM)
        header = f"│ {'Operación':<40} │ {'Total':>10} │ {'Calls':>6} │ {'Mean':>10} │ {'%':>6} │"
        cprint(header, Colors.BOLD)
        cprint("├" + "─"*78 + "┤", Colors.DIM)
        
        for name, data in sorted_stats[:TOP_FUNCIONES]:
            pct = (data['total'] / total_time * 100) if total_time > 0 else 0
            
            # Color según porcentaje
            if pct > 30:
                color = Colors.RED
            elif pct > 15:
                color = Colors.YELLOW
            elif pct > 5:
                color = Colors.CYAN
            else:
                color = Colors.DIM
            
            row = f"│ {name:<40} │ {data['total']:>9.3f}s │ {data['calls']:>6} │ {data['mean']:>9.4f}s │ {pct:>5.1f}% │"
            cprint(row, color)
        
        cprint("└" + "─"*78 + "┘", Colors.DIM)
        
        # Top 5 cuellos de botella
        cprint("\n🚨 TOP 5 CUELLOS DE BOTELLA:", Colors.RED + Colors.BOLD)
        for i, (name, data) in enumerate(sorted_stats[:5], 1):
            pct = (data['total'] / total_time * 100) if total_time > 0 else 0
            cprint(f"   {i}. {name}: {data['total']:.3f}s ({pct:.1f}%) - {data['calls']} llamadas", Colors.RED)

# ============================================================================
# WRAPPER PARA INSTRUMENTAR FUNCIONES
# ============================================================================
timer = DetailedTimer()

def instrument_function(func, name: str = None):
    """Wrapper que mide tiempo de ejecución de una función."""
    fname = name or f"{func.__module__}.{func.__name__}"
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer.start(fname)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            timer.stop(fname)
    
    return wrapper

def instrument_class_methods(cls, prefix: str = ""):
    """Instrumenta todos los métodos de una clase."""
    for name in dir(cls):
        if name.startswith('_') and not name.startswith('__'):
            continue
        if name.startswith('__') and name not in ('__init__', '__call__'):
            continue
        
        attr = getattr(cls, name)
        if callable(attr) and not isinstance(attr, type):
            try:
                method_name = f"{prefix}{cls.__name__}.{name}"
                setattr(cls, name, instrument_function(attr, method_name))
            except (AttributeError, TypeError):
                pass
    return cls

# ============================================================================
# PROFILER CON cProfile
# ============================================================================
def run_cprofile(func, *args, **kwargs):
    """Ejecuta función con cProfile y retorna stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Crear stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(TOP_FUNCIONES)
    
    return result, stream.getvalue(), stats

# ============================================================================
# MAIN PROFILER
# ============================================================================
def main():
    cprint("\n" + "="*80, Colors.BLUE)
    cprint("🔬 PROFILER DETALLADO PARA SISTEMA DE ANÁLISIS", Colors.BOLD + Colors.CYAN)
    cprint("="*80, Colors.BLUE)
    
    # Archivo a analizar
    archivo = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_PRUEBA
    
    if not Path(archivo).exists():
        cprint(f"❌ Archivo no encontrado: {archivo}", Colors.RED)
        sys.exit(1)
    
    cprint(f"\n📁 Archivo de prueba: {archivo}", Colors.GREEN)
    cprint(f"📊 Top funciones a mostrar: {TOP_FUNCIONES}", Colors.DIM)
    
    # Iniciar tracking de memoria
    if MEMORIA_TRACKING:
        tracemalloc.start()
        cprint("🧠 Tracking de memoria: ACTIVADO", Colors.GREEN)
    
    # =========================================================================
    # IMPORTAR Y PREPARAR analisis2.py
    # =========================================================================
    cprint("\n" + "-"*80, Colors.DIM)
    cprint("📦 Importando analisis2.py...", Colors.CYAN)
    
    import_start = time.perf_counter()
    
    # Importar el módulo
    spec = importlib.util.spec_from_file_location("analisis2", "analisis2.py")
    analisis2 = importlib.util.module_from_spec(spec)
    sys.modules["analisis2"] = analisis2
    spec.loader.exec_module(analisis2)
    
    import_time = time.perf_counter() - import_start
    cprint(f"✅ Importación completada en {import_time:.3f}s", Colors.GREEN)
    
    # =========================================================================
    # INSTRUMENTAR CLASES Y FUNCIONES CON DETALLE MÁXIMO
    # =========================================================================
    cprint("\n📐 Instrumentando para profiling detallado del GPR...", Colors.CYAN)
    
    # Guardar referencias originales
    OriginalAnalyzer = analisis2.BayesianDenoisingAnalyzer
    OriginalGPROptimizer = analisis2.GPROptimizer
    
    # =========================================================================
    # INSTRUMENTAR GPROptimizer CON DETALLE INTERNO (THREAD-SAFE)
    # =========================================================================
    import threading
    _timer_lock = threading.Lock()
    
    class InstrumentedGPR(OriginalGPROptimizer):
        """GPROptimizer con timing detallado de cada subfase (thread-safe)."""
        
        def fit(self, X, y, feature_names=None):
            """Fit con timing de cada fase interna - SIN usar timer global para evitar race conditions."""
            import torch
            import gpytorch
            from gpytorch.likelihoods import GaussianLikelihood
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from sklearn.preprocessing import StandardScaler
            
            # Tiempos locales (thread-safe)
            times_local = {}
            total_start = time.perf_counter()
            
            self.feature_names = feature_names or [f"p{i}" for i in range(X.shape[1])]
            self.original_size = len(X)
            self._prediction_cache.clear()
            
            # FASE 1: Normalización
            t0 = time.perf_counter()
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            times_local['1.Normalización'] = time.perf_counter() - t0
            
            # FASE 2: Crear tensores
            t0 = time.perf_counter()
            train_x = torch.tensor(X_scaled, dtype=torch.float32).contiguous()
            train_y = torch.tensor(y_scaled, dtype=torch.float32).contiguous()
            self._train_x = train_x
            self._train_y = train_y
            n_features = X_scaled.shape[1]
            times_local['2.Crear tensores'] = time.perf_counter() - t0
            
            # FASE 3: Crear modelo
            t0 = time.perf_counter()
            SPARSE_GP_ENABLED = analisis2.SPARSE_GP_ENABLED
            SPARSE_INDUCING_POINTS = analisis2.SPARSE_INDUCING_POINTS
            
            if SPARSE_GP_ENABLED and len(X) > SPARSE_INDUCING_POINTS:
                self.is_sparse = True
                self.n_inducing = min(SPARSE_INDUCING_POINTS, len(X))
                inducing_points = self._initialize_inducing_points(X_scaled, self.n_inducing)
                self.likelihood = GaussianLikelihood()
                self.model = analisis2.SparseGPModel(inducing_points, n_features)
            else:
                self.is_sparse = False
                self.n_inducing = 0
                self.likelihood = GaussianLikelihood()
                self.model = analisis2.ExactGPModel(train_x, train_y, self.likelihood, n_features)
            times_local['3.Crear modelo'] = time.perf_counter() - t0
            
            # FASE 4: Entrenar
            t0 = time.perf_counter()
            self.model.train()
            self.likelihood.train()
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training_iterations, eta_min=0.001
            )
            mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
            
            n_iters = self.training_iterations
            with torch.enable_grad():
                for epoch in range(n_iters):
                    optimizer.zero_grad()
                    output = self.model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            times_local['4.Entrenamiento'] = time.perf_counter() - t0
            
            # FASE 5: Modo evaluación
            t0 = time.perf_counter()
            self.model.eval()
            self.likelihood.eval()
            
            try:
                self.learned_noise = self.likelihood.noise.item()
                ls = self.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
                self.learned_length_scales = ls
            except:
                self.learned_noise = 0.0
                self.learned_length_scales = np.array([])
            times_local['5.Modo eval'] = time.perf_counter() - t0
            
            total_time = time.perf_counter() - total_start
            
            # Registrar tiempos de forma thread-safe
            with _timer_lock:
                for phase, duration in times_local.items():
                    key = f"  GPR.fit > {phase}"
                    timer.times[key].append(duration)
                    timer.call_counts[key] += 1
                timer.times["GPR.fit [TOTAL]"].append(total_time)
                timer.call_counts["GPR.fit [TOTAL]"] += 1
            
            return self
        
        def predict_batch(self, X_batch):
            t0 = time.perf_counter()
            result = super().predict_batch(X_batch)
            duration = time.perf_counter() - t0
            with _timer_lock:
                timer.times["  GPR.predict_batch"].append(duration)
                timer.call_counts["  GPR.predict_batch"] += 1
            return result
        
        def compute_r2(self, X, y):
            t0 = time.perf_counter()
            result = super().compute_r2(X, y)
            duration = time.perf_counter() - t0
            with _timer_lock:
                timer.times["  GPR.compute_r2"].append(duration)
                timer.call_counts["  GPR.compute_r2"] += 1
            return result
    
    # Reemplazar GPROptimizer
    analisis2.GPROptimizer = InstrumentedGPR
    
    # =========================================================================
    # INSTRUMENTAR compute_partial_dependence (THREAD-SAFE)
    # =========================================================================
    original_pd = analisis2.compute_partial_dependence_vectorized
    original_pd_all = analisis2.compute_all_partial_dependences
    
    def instrumented_pd(gpr, X, y, param_idx, param_name, n_points=50):
        t0 = time.perf_counter()
        result = original_pd(gpr, X, y, param_idx, param_name, n_points)
        duration = time.perf_counter() - t0
        with _timer_lock:
            timer.times["  compute_partial_dependence"].append(duration)
            timer.call_counts["  compute_partial_dependence"] += 1
        return result
    
    def instrumented_pd_all(gpr, X, y, param_names, n_points=50):
        t0 = time.perf_counter()
        result = original_pd_all(gpr, X, y, param_names, n_points)
        duration = time.perf_counter() - t0
        with _timer_lock:
            timer.times["  compute_all_partial_dependences"].append(duration)
            timer.call_counts["  compute_all_partial_dependences"] += 1
        return result
    
    analisis2.compute_partial_dependence_vectorized = instrumented_pd
    analisis2.compute_all_partial_dependences = instrumented_pd_all
    
    # =========================================================================
    # INSTRUMENTAR funciones de figuras (THREAD-SAFE)
    # =========================================================================
    original_heatmap = analisis2._generate_heatmap
    def instrumented_heatmap(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_heatmap(*args, **kwargs)
        duration = time.perf_counter() - t0
        with _timer_lock:
            timer.times["_generate_heatmap"].append(duration)
            timer.call_counts["_generate_heatmap"] += 1
        return result
    analisis2._generate_heatmap = instrumented_heatmap
    
    original_uncertainty = analisis2._generate_uncertainty_plot
    def instrumented_uncertainty(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_uncertainty(*args, **kwargs)
        duration = time.perf_counter() - t0
        with _timer_lock:
            timer.times["_generate_uncertainty_plot"].append(duration)
            timer.call_counts["_generate_uncertainty_plot"] += 1
        return result
    analisis2._generate_uncertainty_plot = instrumented_uncertainty
    
    # =========================================================================
    # CREAR ANALYZER INSTRUMENTADO
    # =========================================================================
    class InstrumentedAnalyzer(OriginalAnalyzer):
        """Versión instrumentada con timing detallado."""
        
        def __init__(self):
            with timer.measure("Analyzer.__init__"):
                super().__init__()
        
        def load_data(self, filepath: str):
            with timer.measure("Analyzer.load_data"):
                # Soportar Excel y CSV
                import pandas as pd
                import numpy as np
                import pyarrow as pa
                import pyarrow.parquet as pq
                import tempfile
                import os
                
                cprint(f"      📄 Cargando: {filepath}", Colors.DIM)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                cprint(f"      📏 Tamaño original: {file_size:.2f} MB", Colors.DIM)
                t0 = time.perf_counter()
                
                if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                    # Detectar header automáticamente
                    df_temp = pd.read_excel(filepath, header=None, nrows=10)
                    keywords = ['TRIAL', 'ROI', 'SCORE', 'SHARPE', 'DRAWDOWN', 'PARAM', 
                               'EXIT_SL', 'EXIT_TP', 'LOOKBAR', 'WINRATE', 'SQN']
                    header_row = 0
                    
                    for idx in range(min(10, len(df_temp))):
                        row_str = ' '.join(str(v).upper() for v in df_temp.iloc[idx].values if pd.notna(v))
                        matches = sum(1 for kw in keywords if kw in row_str)
                        if matches >= 3:
                            header_row = idx
                            break
                    
                    cprint(f"      📍 Header detectado en fila {header_row}", Colors.DIM)
                    df = pd.read_excel(filepath, header=header_row)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
                    df.columns = df.columns.astype(str).str.strip()
                    
                    # Convertir columnas object a numérico
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            except:
                                pass
                    
                    t_read = time.perf_counter() - t0
                    cprint(f"      📊 Excel leído: {t_read:.2f}s ({len(df):,} filas)", Colors.GREEN)
                else:
                    df = pd.read_csv(filepath)
                    t_read = time.perf_counter() - t0
                    cprint(f"      📊 CSV leído: {t_read:.2f}s ({len(df):,} filas)", Colors.GREEN)
                
                # Aplicar muestra si está configurada
                if MUESTRA_SIZE and len(df) > MUESTRA_SIZE:
                    df = df.sample(n=MUESTRA_SIZE, random_state=42)
                    cprint(f"      📉 Muestra aplicada: {MUESTRA_SIZE} filas", Colors.YELLOW)
                
                # Normalizar columnas
                t0 = time.perf_counter()
                df = self._normalize_columns(df)
                t_norm = time.perf_counter() - t0
                cprint(f"      🔄 Columnas normalizadas: {t_norm*1000:.1f}ms", Colors.DIM)
                
                # ═══════════════════════════════════════════════════════════════
                # CREAR PARQUET TEMPORAL (igual que analisis2.py)
                # ═══════════════════════════════════════════════════════════════
                t0 = time.perf_counter()
                cprint(f"      ⏳ Creando Parquet temporal...", Colors.DIM)
                
                from pathlib import Path
                base_name = Path(filepath).stem
                self.temp_parquet_path = tempfile.mktemp(prefix=f"{base_name}_", suffix='.parquet')
                
                # Convertir a PyArrow Table y guardar como Parquet
                self.table = pa.Table.from_pandas(df)
                pq.write_table(self.table, self.temp_parquet_path, compression='snappy')
                
                parquet_size = os.path.getsize(self.temp_parquet_path) / (1024 * 1024)
                compression_ratio = (1 - parquet_size / file_size) * 100 if file_size > 0 else 0
                
                # Recargar desde Parquet (memory-mapped)
                self.table = pq.read_table(self.temp_parquet_path)
                self.df = self.table.to_pandas()
                
                t_parquet = time.perf_counter() - t0
                cprint(f"      ✅ Parquet creado: {t_parquet:.2f}s", Colors.GREEN)
                cprint(f"      📦 Path: {self.temp_parquet_path}", Colors.DIM)
                cprint(f"      📊 Tamaño Parquet: {parquet_size:.2f} MB (compresión: {compression_ratio:.1f}%)", Colors.DIM)
                
                cprint(f"      📊 Shape final: {self.df.shape}", Colors.DIM)
                cprint(f"      📋 Columnas: {list(self.df.columns[:5])}...", Colors.DIM)
                return self.df
        
        def _normalize_columns(self, df):
            with timer.measure("Analyzer._normalize_columns"):
                return super()._normalize_columns(df)
        
        def _detect_param_step(self, col):
            with timer.measure("Analyzer._detect_param_step"):
                return super()._detect_param_step(col)
        
        def run_gpr_analysis(self):
            with timer.measure("Analyzer.run_gpr_analysis [TOTAL]"):
                return super().run_gpr_analysis()
        
        def generate_all_figures(self):
            with timer.measure("Analyzer.generate_all_figures [TOTAL]"):
                return super().generate_all_figures()
        
        def cleanup(self):
            with timer.measure("Analyzer.cleanup"):
                return super().cleanup()
    
    # =========================================================================
    # EJECUTAR CON PROFILING DETALLADO
    # =========================================================================
    cprint("\n" + "="*80, Colors.BLUE)
    cprint("🚀 INICIANDO ANÁLISIS CON PROFILING", Colors.BOLD + Colors.GREEN)
    cprint("="*80, Colors.BLUE)
    
    total_start = time.perf_counter()
    
    # Crear analyzer instrumentado
    cprint("\n[1/7] Creando Analyzer...", Colors.CYAN)
    analyzer = InstrumentedAnalyzer()
    
    # Cargar datos
    cprint("\n[2/7] Cargando datos...", Colors.CYAN)
    step_start = time.perf_counter()
    analyzer.load_data(archivo)
    cprint(f"      ✅ Datos cargados en {time.perf_counter() - step_start:.3f}s", Colors.GREEN)
    cprint(f"      📊 {len(analyzer.df):,} filas × {len(analyzer.df.columns)} columnas", Colors.DIM)
    
    # Configurar tipo de salida (automático para profiling - usar FIXED)
    cprint("\n[3/7] Configurando tipo de salida...", Colors.CYAN)
    step_start = time.perf_counter()
    analyzer.exit_type = "fixed"  # Configurar directamente sin prompt
    cprint(f"      📤 Tipo de salida: {analyzer.exit_type.upper()}", Colors.DIM)
    cprint(f"      ✅ Completado en {time.perf_counter() - step_start:.3f}s", Colors.GREEN)
    
    # Detectar columnas
    cprint("\n[4/7] Detectando columnas...", Colors.CYAN)
    step_start = time.perf_counter()
    with timer.measure("Analyzer.detect_columns"):
        analyzer.detect_columns()
    cprint(f"      📊 Parámetros: {len(analyzer.param_columns)}", Colors.DIM)
    cprint(f"      📈 Métricas: {len(analyzer.metric_columns)}", Colors.DIM)
    cprint(f"      ✅ Completado en {time.perf_counter() - step_start:.3f}s", Colors.GREEN)
    
    # Seleccionar métricas (automático para profiling)
    cprint("\n[5/7] Seleccionando métricas (automático)...", Colors.CYAN)
    step_start = time.perf_counter()
    
    # Seleccionar primeras 3 métricas para prueba
    if analyzer.metric_columns:
        analyzer.selected_metrics = analyzer.metric_columns[:3]
        cprint(f"      📈 Métricas seleccionadas: {analyzer.selected_metrics}", Colors.DIM)
    else:
        cprint(f"      ⚠️ No se detectaron métricas!", Colors.YELLOW)
    
    cprint(f"      ✅ Completado en {time.perf_counter() - step_start:.3f}s", Colors.GREEN)
    
    # Verificar que hay métricas antes de continuar
    if not analyzer.selected_metrics:
        cprint("\n❌ No hay métricas seleccionadas. Abortando.", Colors.RED)
        cprint("   Columnas disponibles:", Colors.YELLOW)
        for col in analyzer.df.columns:
            cprint(f"      - {col}", Colors.DIM)
        sys.exit(1)
    
    # Análisis GPR (la parte más pesada)
    cprint("\n[6/7] Ejecutando análisis GPR...", Colors.CYAN)
    cprint("      ⏳ Esta es la parte más pesada, espere...", Colors.YELLOW)
    step_start = time.perf_counter()
    
    # Ejecutar con cProfile para obtener detalles internos
    try:
        _, cprofile_output, cprofile_stats = run_cprofile(analyzer.run_gpr_analysis)
        gpr_time = time.perf_counter() - step_start
        cprint(f"      ✅ GPR completado en {gpr_time:.3f}s", Colors.GREEN)
    except Exception as e:
        cprint(f"      ❌ Error en GPR: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        gpr_time = time.perf_counter() - step_start
        cprofile_output = ""
    
    # Generar figuras
    cprint("\n[7/7] Generando visualizaciones...", Colors.CYAN)
    step_start = time.perf_counter()
    try:
        analyzer.generate_all_figures()
        cprint(f"      ✅ Completado en {time.perf_counter() - step_start:.3f}s", Colors.GREEN)
    except Exception as e:
        cprint(f"      ⚠️ Error: {e}", Colors.YELLOW)
    
    # Cleanup
    analyzer.cleanup()
    
    total_time = time.perf_counter() - total_start
    
    # =========================================================================
    # REPORTE FINAL
    # =========================================================================
    cprint("\n" + "="*80, Colors.BLUE)
    cprint("📊 RESULTADOS DEL PROFILING", Colors.BOLD + Colors.CYAN)
    cprint("="*80, Colors.BLUE)
    
    # Reporte de nuestros timers
    timer.print_report()
    
    # =========================================================================
    # DESGLOSE ESPECÍFICO DEL GPR
    # =========================================================================
    stats = timer.get_stats()
    gpr_stats = {k: v for k, v in stats.items() if 'GPR' in k or 'gpr' in k.lower() or 'partial_dependence' in k}
    
    if gpr_stats:
        cprint("\n" + "="*80, Colors.YELLOW)
        cprint("🔬 DESGLOSE DETALLADO DEL ANÁLISIS GPR", Colors.BOLD + Colors.YELLOW)
        cprint("="*80, Colors.YELLOW)
        
        # Calcular tiempo total de GPR
        gpr_total = sum(v['total'] for k, v in gpr_stats.items() if 'TOTAL' not in k)
        
        cprint(f"\n📊 Tiempo total en GPR: {gpr_total:.2f}s", Colors.BOLD)
        
        # Ordenar por tiempo
        sorted_gpr = sorted(gpr_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        cprint("\n┌" + "─"*78 + "┐", Colors.DIM)
        header = f"│ {'Subproceso GPR':<45} │ {'Total':>8} │ {'Calls':>5} │ {'Media':>8} │ {'%GPR':>5} │"
        cprint(header, Colors.BOLD)
        cprint("├" + "─"*78 + "┤", Colors.DIM)
        
        for name, data in sorted_gpr:
            if 'TOTAL' in name:
                continue  # Saltar totales
            pct = (data['total'] / gpr_total * 100) if gpr_total > 0 else 0
            
            # Color según tiempo
            if data['total'] > 10:
                color = Colors.RED
            elif data['total'] > 3:
                color = Colors.YELLOW
            else:
                color = Colors.GREEN
            
            row = f"│ {name:<45} │ {data['total']:>7.2f}s │ {data['calls']:>5} │ {data['mean']:>7.3f}s │ {pct:>4.1f}% │"
            cprint(row, color)
        
        cprint("└" + "─"*78 + "┘", Colors.DIM)
        
        # Identificar cuellos de botella específicos
        cprint("\n🎯 CUELLOS DE BOTELLA EN GPR:", Colors.BOLD + Colors.RED)
        for name, data in sorted_gpr[:3]:
            if 'TOTAL' in name:
                continue
            pct = (data['total'] / gpr_total * 100) if gpr_total > 0 else 0
            cprint(f"   • {name}: {data['total']:.2f}s ({pct:.1f}%) - {data['calls']} llamadas", Colors.RED)
            
            # Recomendaciones específicas
            if 'Entrenamiento' in name or 'loop' in name:
                cprint(f"     → Reducir training_iterations (actualmente 100)", Colors.DIM)
                cprint(f"     → Añadir early stopping", Colors.DIM)
            elif 'partial_dependence' in name:
                cprint(f"     → Reducir n_points (actualmente 50)", Colors.DIM)
                cprint(f"     → Reducir max_samples_pd (actualmente 500)", Colors.DIM)
            elif 'predict' in name.lower():
                cprint(f"     → Aumentar PREDICTION_BATCH_SIZE", Colors.DIM)
                cprint(f"     → Cachear predicciones repetidas", Colors.DIM)
    
    # Reporte de cProfile (GPR)
    if cprofile_output:
        cprint("\n" + "-"*80, Colors.DIM)
        cprint("🔍 FUNCIONES PYTORCH/GPYTORCH MÁS COSTOSAS:", Colors.BOLD + Colors.YELLOW)
        cprint("-"*80, Colors.DIM)
        
        # Filtrar y mostrar las líneas más relevantes
        lines = cprofile_output.split('\n')
        relevant_lines = []
        for line in lines:
            if any(x in line for x in ['gpytorch', 'torch', 'cholesky', 'matmul', 
                                        'backward', 'forward', 'kernel', 'linear_operator']):
                relevant_lines.append(line)
            elif 'cumtime' in line or 'ncalls' in line:
                relevant_lines.append(line)
        
        for line in relevant_lines[:25]:
            print(line)
    
    # Memoria
    if MEMORIA_TRACKING:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        cprint("\n" + "-"*80, Colors.DIM)
        cprint("🧠 USO DE MEMORIA:", Colors.BOLD + Colors.CYAN)
        cprint(f"   Memoria actual: {current / 1024 / 1024:.1f} MB", Colors.GREEN)
        cprint(f"   Pico máximo: {peak / 1024 / 1024:.1f} MB", Colors.YELLOW)
    
    # Resumen final
    cprint("\n" + "="*80, Colors.BLUE)
    cprint("📋 RESUMEN EJECUTIVO", Colors.BOLD + Colors.GREEN)
    cprint("="*80, Colors.BLUE)
    
    cprint(f"\n⏱️  TIEMPO TOTAL: {total_time:.2f}s ({total_time/60:.1f} min)", Colors.BOLD + Colors.GREEN)
    
    # Distribución principal
    cprint("\n🎯 DISTRIBUCIÓN GENERAL:", Colors.CYAN)
    if stats:
        main_categories = ['Analyzer.run_gpr_analysis [TOTAL]', 'Analyzer.generate_all_figures [TOTAL]', 
                          'Analyzer.load_data', 'Analyzer.detect_columns']
        for cat in main_categories:
            if cat in stats:
                data = stats[cat]
                pct = (data['total'] / total_time * 100)
                bar_len = int(pct / 2)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                cprint(f"   {cat:<40} {bar} {pct:>5.1f}%", Colors.CYAN)
    
    # Recomendaciones finales
    cprint("\n💡 RECOMENDACIONES DE OPTIMIZACIÓN:", Colors.BOLD + Colors.YELLOW)
    
    if gpr_stats:
        entrenamiento = [v for k, v in gpr_stats.items() if 'Entrenamiento' in k or 'loop' in k]
        if entrenamiento and entrenamiento[0]['total'] > 5:
            cprint("   🔴 ENTRENAMIENTO GPR muy lento:", Colors.RED)
            cprint("      → Reducir training_iterations de 100 a 50", Colors.DIM)
            cprint("      → Implementar early stopping (parar si loss converge)", Colors.DIM)
            cprint("      → Para datasets >1000 filas, activar Sparse GP", Colors.DIM)
        
        pd_stats = [v for k, v in gpr_stats.items() if 'partial_dependence' in k]
        if pd_stats and pd_stats[0]['total'] > 10:
            cprint("   🟠 DEPENDENCIAS PARCIALES lentas:", Colors.YELLOW)
            cprint("      → Reducir n_points de 50 a 30", Colors.DIM)
            cprint("      → Reducir max_samples_pd de 500 a 300", Colors.DIM)
            cprint("      → Paralelizar por métrica (ya está, verificar N_JOBS)", Colors.DIM)
    
    cprint("\n" + "="*80, Colors.BLUE)
    cprint("✅ PROFILING COMPLETADO", Colors.BOLD + Colors.GREEN)
    cprint("="*80 + "\n", Colors.BLUE)

if __name__ == "__main__":
    main()

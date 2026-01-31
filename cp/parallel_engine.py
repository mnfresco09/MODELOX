"""
# =============================================================================
#
#     ██████╗  █████╗ ██████╗  █████╗ ██╗     ██╗     ███████╗██╗
#     ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║     ██║     ██╔════╝██║
#     ██████╔╝███████║██████╔╝███████║██║     ██║     █████╗  ██║
#     ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║     ██║     ██╔══╝  ██║
#     ██║     ██║  ██║██║  ██║██║  ██║███████╗███████╗███████╗███████╗
#     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝
#
#     PARALLEL_ENGINE.PY - MULTIPROCESAMIENTO
#
# =============================================================================
#
#     USO:
#     - Análisis de vecindario (K backtests por trial)
#     - Perturbación de datos (múltiples seeds)
#     - Trials paralelos de Optuna
#
#     ARQUITECTURA:
#     - Pool de workers persistente
#     - Memoria compartida para OHLCV (zero-copy)
#     - Batch processing
#
# =============================================================================
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

_PARALLEL_ENABLED: bool = os.environ.get("MODELOX_PARALLEL", "1") in {"1", "true", "True", "YES", "yes"}
_MAX_WORKERS: int = int(os.environ.get("MODELOX_MAX_WORKERS", str(min(mp.cpu_count(), 8))))
_USE_THREADS: bool = os.environ.get("MODELOX_USE_THREADS", "0") in {"1", "true", "True"}

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_THREAD_POOL: Optional[ThreadPoolExecutor] = None


def get_max_workers() -> int:
    """RETORNA EL NÚMERO MÁXIMO DE WORKERS."""
    return _MAX_WORKERS


def is_parallel_enabled() -> bool:
    """VERIFICA SI LA PARALELIZACIÓN ESTÁ HABILITADA."""
    return _PARALLEL_ENABLED and _MAX_WORKERS > 1


def _get_pool(use_threads: bool = False) -> ProcessPoolExecutor | ThreadPoolExecutor:
    """OBTIENE EL POOL DE WORKERS (LAZY INITIALIZATION)."""
    global _PROCESS_POOL, _THREAD_POOL
    
    if use_threads or _USE_THREADS:
        if _THREAD_POOL is None:
            _THREAD_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        return _THREAD_POOL
    else:
        if _PROCESS_POOL is None:
            # start_method='fork' es más rápido pero puede tener problemas en macOS
            # 'spawn' es más seguro pero más lento
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Ya está configurado
            _PROCESS_POOL = ProcessPoolExecutor(max_workers=_MAX_WORKERS)
        return _PROCESS_POOL


def shutdown_pools():
    """Cierra los pools de workers."""
    global _PROCESS_POOL, _THREAD_POOL
    
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False)
        _PROCESS_POOL = None
    
    if _THREAD_POOL is not None:
        _THREAD_POOL.shutdown(wait=False)
        _THREAD_POOL = None


# =============================================================================
# EJECUCIÓN PARALELA DE VECINOS
# =============================================================================

def _run_single_neighbor_backtest(args: Tuple) -> Dict[str, Any]:
    """
    Función worker para ejecutar un backtest de vecino.
    
    Args:
        args: Tupla con (neighbor_params, shared_data_key, backtest_config)
    
    Esta función se ejecuta en un proceso/thread separado.
    """
    neighbor_params, ohlcv_data, config_dict, strategy_name = args
    
    try:
        # Importar aquí para evitar problemas de serialización
        from modelox.core.engine import calculate_performance_vectorized_numba, BacktestParams
        from modelox.core.metrics import resumen_metricas
        from modelox.core.scoring import score_quality_only
        import polars as pl
        
        # Reconstruir DataFrame desde arrays
        df = pl.DataFrame({
            "open": ohlcv_data["open"],
            "high": ohlcv_data["high"],
            "low": ohlcv_data["low"],
            "close": ohlcv_data["close"],
            "volume": ohlcv_data.get("volume", np.zeros(len(ohlcv_data["close"]))),
        })
        
        # Reconstruir BacktestParams
        bp = BacktestParams(
            saldo_inicial=config_dict["saldo_inicial"],
            comision_pct=config_dict["comision_pct"],
            comision_sides=config_dict["comision_sides"],
            saldo_minimo_operativo=config_dict["saldo_minimo_operativo"],
            qty_max_activo=config_dict["qty_max_activo"],
            saldo_usado=config_dict["saldo_usado"],
            apalancamiento_max=config_dict["apalancamiento_max"],
            exit_type=config_dict["exit_type"],
            exit_sl_pct=config_dict["exit_sl_pct"],
            exit_tp_pct=config_dict["exit_tp_pct"],
            exit_trail_act_pct=config_dict["exit_trail_act_pct"],
            exit_trail_dist_pct=config_dict["exit_trail_dist_pct"],
            block_velas_after_exit=config_dict.get("block_velas_after_exit", 0),
            time_stop_bars=config_dict.get("time_stop_bars", 0),
        )
        
        # Importar estrategia dinámicamente
        from modelox.strategies.registry import get_strategy_class
        strategy_class = get_strategy_class(strategy_name)
        strategy = strategy_class()
        
        # Generar señales con parámetros del vecino
        signals_df = strategy.generate_signals(df, neighbor_params)
        
        if signals_df is None or signals_df.is_empty():
            return {"success": False, "error": "No signals generated"}
        
        # Ejecutar backtest
        trades_df, equity_curve, _ = calculate_performance_vectorized_numba(
            df=signals_df,
            bp=bp,
        )
        
        if trades_df.is_empty():
            return {"success": False, "error": "No trades"}
        
        # Calcular métricas
        metrics = resumen_metricas(
            trades_df,
            saldo_inicial=bp.saldo_inicial,
            equity_curve=equity_curve,
        )
        
        score = score_quality_only(metrics)
        
        return {
            "success": True,
            "metrics": metrics,
            "score": score,
            "equity_curve": equity_curve,
            "n_trades": len(trades_df),
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_neighbors_parallel(
    neighbors: List[Dict[str, Any]],
    ohlcv_data: Dict[str, np.ndarray],
    config_dict: Dict[str, Any],
    strategy_name: str,
    use_threads: bool = True,  # Threads son más rápidos para I/O bound
) -> List[Dict[str, Any]]:
    """
    Ejecuta backtests para múltiples vecinos en paralelo.
    
    Args:
        neighbors: Lista de diccionarios de parámetros para cada vecino
        ohlcv_data: Diccionario con arrays de OHLCV
        config_dict: Configuración del backtest serializable
        strategy_name: Nombre de la estrategia
        use_threads: Usar threads en lugar de procesos
    
    Returns:
        Lista de resultados de cada vecino
    """
    if not is_parallel_enabled() or len(neighbors) <= 1:
        # Ejecutar secuencialmente si no hay paralelización o solo 1 vecino
        return [
            _run_single_neighbor_backtest((n, ohlcv_data, config_dict, strategy_name))
            for n in neighbors
        ]
    
    # Preparar argumentos para cada worker
    args_list = [
        (neighbor, ohlcv_data, config_dict, strategy_name)
        for neighbor in neighbors
    ]
    
    results = []
    pool = _get_pool(use_threads=use_threads)
    
    # Ejecutar en paralelo
    futures = [pool.submit(_run_single_neighbor_backtest, args) for args in args_list]
    
    for future in as_completed(futures):
        try:
            result = future.result(timeout=30)  # 30 segundos timeout por vecino
            results.append(result)
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    
    return results


# =============================================================================
# FUNCIONES HELPER PARA SERIALIZACIÓN
# =============================================================================

def config_to_dict(bp) -> Dict[str, Any]:
    """Convierte BacktestParams a diccionario serializable."""
    return {
        "saldo_inicial": bp.saldo_inicial,
        "comision_pct": bp.comision_pct,
        "comision_sides": bp.comision_sides,
        "saldo_minimo_operativo": bp.saldo_minimo_operativo,
        "qty_max_activo": bp.qty_max_activo,
        "saldo_usado": bp.saldo_usado,
        "apalancamiento_max": bp.apalancamiento_max,
        "exit_type": bp.exit_type,
        "exit_sl_pct": bp.exit_sl_pct,
        "exit_tp_pct": bp.exit_tp_pct,
        "exit_trail_act_pct": bp.exit_trail_act_pct,
        "exit_trail_dist_pct": bp.exit_trail_dist_pct,
        "block_velas_after_exit": bp.block_velas_after_exit,
        "time_stop_bars": bp.time_stop_bars,
    }


def df_to_arrays(df) -> Dict[str, np.ndarray]:
    """Convierte DataFrame Polars a diccionario de arrays."""
    return {
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "volume": df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df)),
    }


# =============================================================================
# BATCH PROCESSING PARA TRIALS
# =============================================================================

def batch_run_trials(
    trial_configs: List[Dict[str, Any]],
    ohlcv_data: Dict[str, np.ndarray],
    strategy_name: str,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ejecuta múltiples trials en paralelo (para n_jobs > 1 en Optuna).
    
    Args:
        trial_configs: Lista de configuraciones de trial
        ohlcv_data: Datos OHLCV compartidos
        strategy_name: Nombre de la estrategia
        max_workers: Número de workers (default: _MAX_WORKERS)
    
    Returns:
        Lista de resultados de cada trial
    """
    n_workers = max_workers or _MAX_WORKERS
    
    if n_workers <= 1 or len(trial_configs) <= 1:
        return [
            _run_single_neighbor_backtest((tc["params"], ohlcv_data, tc["config"], strategy_name))
            for tc in trial_configs
        ]
    
    args_list = [
        (tc["params"], ohlcv_data, tc["config"], strategy_name)
        for tc in trial_configs
    ]
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single_neighbor_backtest, args): i 
                   for i, args in enumerate(args_list)}
        
        # Mantener orden original
        results = [None] * len(args_list)
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=60)
            except Exception as e:
                results[idx] = {"success": False, "error": str(e)}
    
    return results


# =============================================================================
# CONTEXT MANAGER PARA POOL LIFECYCLE
# =============================================================================

class ParallelContext:
    """
    Context manager para gestionar el ciclo de vida del pool.
    
    Uso:
        with ParallelContext(max_workers=4) as ctx:
            results = ctx.run_neighbors(neighbors, ...)
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_threads: bool = True):
        self.max_workers = max_workers or _MAX_WORKERS
        self.use_threads = use_threads
        self._pool = None
    
    def __enter__(self):
        if self.use_threads:
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
    
    def run_neighbors(
        self,
        neighbors: List[Dict[str, Any]],
        ohlcv_data: Dict[str, np.ndarray],
        config_dict: Dict[str, Any],
        strategy_name: str,
    ) -> List[Dict[str, Any]]:
        """Ejecuta vecinos usando el pool de este contexto."""
        if len(neighbors) <= 1 or self._pool is None:
            return [
                _run_single_neighbor_backtest((n, ohlcv_data, config_dict, strategy_name))
                for n in neighbors
            ]
        
        args_list = [
            (neighbor, ohlcv_data, config_dict, strategy_name)
            for neighbor in neighbors
        ]
        
        results = list(self._pool.map(
            _run_single_neighbor_backtest,
            args_list,
            timeout=30 * len(args_list)
        ))
        
        return results


# Info de configuración
def get_parallel_info() -> Dict[str, Any]:
    """Retorna información sobre la configuración de paralelización."""
    return {
        "enabled": is_parallel_enabled(),
        "max_workers": _MAX_WORKERS,
        "cpu_count": mp.cpu_count(),
        "use_threads": _USE_THREADS,
        "pool_active": _PROCESS_POOL is not None or _THREAD_POOL is not None,
    }

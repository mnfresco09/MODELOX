#!/usr/bin/env python3
"""
MODELOX Nuclear Engine - Benchmark de Rendimiento v3
=====================================================

Compara rendimiento entre:
- Extensiones C (Cython)
- Numba JIT
- Python puro (baseline)

Funciones probadas:
- simulate_trades: Kernel principal de simulación
- compute_metrics: Cálculo de métricas
- perturb_returns: Perturbación de datos
- compute_cvar_95: CVaR para análisis de riesgo
- compute_equity_r2: R² de equity
- aggregate_neighbor_metrics: Agregación vecinal

USO:
    cd cp && python benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Añadir raíz del proyecto al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def generate_test_data(n_entries: int = 2000, n_bars: int = 100_000):
    """Genera datos de prueba realistas."""
    np.random.seed(42)
    
    # Simular precios tipo random walk
    returns = np.random.normal(0.0001, 0.01, n_bars)
    close = 50000.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    
    # Entradas distribuidas uniformemente
    entry_indices = np.sort(np.random.choice(n_bars - 1000, n_entries, replace=False)).astype(np.int64)
    entry_prices = close[entry_indices]
    entry_types = np.random.choice([1, -1], n_entries).astype(np.int64)
    
    return {
        "entry_indices": entry_indices,
        "entry_prices": entry_prices,
        "entry_types": entry_types,
        "close": close.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
    }


def benchmark_simulate_trades():
    """Benchmark del kernel de simulación."""
    print("\n" + "=" * 70)
    print("BENCHMARK: simulate_trades")
    print("=" * 70)
    
    data = generate_test_data()
    
    params = {
        "saldo_inicial": 10000.0,
        "fee_rate": 0.001,
        "min_op": 10.0,
        "apalancamiento_max": 60.0,
        "qty_max": 1.0,
        "saldo_usado_cfg": 75.0,
        "is_trailing": False,
        "sl_pct": 3.0,
        "tp_pct": 6.0,
        "trail_act_pct": 4.0,
        "trail_dist_pct": 1.5,
        "time_stop_bars": 100,
        "comision_sides": 2,
    }
    
    # Test C
    t_c = None
    try:
        from cp import simulate_trades_c, C_AVAILABLE
        if not C_AVAILABLE:
            raise ImportError("C no disponible")
        
        # Warmup
        simulate_trades_c(
            data["entry_indices"], data["entry_prices"], data["entry_types"],
            data["close"], data["high"], data["low"],
            **params
        )
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 100
        for _ in range(n_runs):
            result_c = simulate_trades_c(
                data["entry_indices"], data["entry_prices"], data["entry_types"],
                data["close"], data["high"], data["low"],
                **params
            )
        t_c = (time.perf_counter() - t0) / n_runs * 1000
        
        trade_count_c = result_c[-1]
        print(f"\n✅ Extensión C:")
        print(f"   Tiempo promedio: {t_c:.3f} ms")
        print(f"   Trades ejecutados: {trade_count_c}")
        
    except ImportError as e:
        print(f"\n❌ Extensión C no disponible: {e}")
        print("   Ejecuta: cd cp && python setup.py build_ext --inplace")
    
    # Test Numba
    t_numba = None
    try:
        from modelox.core.engine import calculate_performance_vectorized_numba, BacktestParams
        import polars as pl
        
        n_bars = len(data["close"])
        df = pl.DataFrame({
            "close": data["close"],
            "high": data["high"],
            "low": data["low"],
            "open": data["close"],
            "volume": np.ones(n_bars),
            "signal": np.zeros(n_bars),
        })
        
        signal = np.zeros(n_bars)
        for idx, etype in zip(data["entry_indices"], data["entry_types"]):
            signal[idx] = etype
        df = df.with_columns(pl.Series("signal", signal))
        
        bp = BacktestParams(
            saldo_inicial=10000.0,
            comision_pct=0.1,
            comision_sides=2,
            saldo_minimo_operativo=10.0,
            qty_max_activo=1.0,
            saldo_usado=75.0,
            apalancamiento_max=60.0,
            exit_type="pnl_fixed",
            exit_sl_pct=3.0,
            exit_tp_pct=6.0,
            exit_trail_act_pct=4.0,
            exit_trail_dist_pct=1.5,
            block_velas_after_exit=0,
            time_stop_bars=100,
        )
        
        # Warmup
        calculate_performance_vectorized_numba(df, bp)
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 100
        for _ in range(n_runs):
            trades_df, equity, _ = calculate_performance_vectorized_numba(df, bp)
        t_numba = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"\n🔵 Numba JIT:")
        print(f"   Tiempo promedio: {t_numba:.3f} ms")
        print(f"   Trades ejecutados: {len(trades_df)}")
        
    except Exception as e:
        print(f"\n❌ Numba no disponible: {e}")
    
    if t_c and t_numba:
        speedup = t_numba / t_c
        print(f"\n⚡ SPEEDUP: {speedup:.2f}x más rápido con C")


def benchmark_metrics():
    """Benchmark de cálculo de métricas."""
    print("\n" + "=" * 70)
    print("BENCHMARK: compute_metrics")
    print("=" * 70)
    
    n_trades = 500
    np.random.seed(42)
    
    pnl_neto = np.random.randn(n_trades) * 50 + 5
    pnl_pct = pnl_neto / 75 * 100
    saldo_inicial = 10000.0
    saldo_despues = saldo_inicial + np.cumsum(pnl_neto)
    
    pnl_neto = pnl_neto.astype(np.float64)
    pnl_pct = pnl_pct.astype(np.float64)
    saldo_despues = saldo_despues.astype(np.float64)
    
    # Test C
    t_c = None
    try:
        from cp import compute_metrics_c, C_AVAILABLE
        if not C_AVAILABLE:
            raise ImportError("C no disponible")
        
        # Warmup
        compute_metrics_c(pnl_neto, pnl_pct, saldo_despues, saldo_inicial)
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 10000
        for _ in range(n_runs):
            result_c = compute_metrics_c(pnl_neto, pnl_pct, saldo_despues, saldo_inicial)
        t_c = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"\n✅ Extensión C:")
        print(f"   Tiempo promedio: {t_c:.4f} ms")
        print(f"   ROI: {result_c[0]:.2f}%, Winrate: {result_c[1]:.2f}%, DD: {result_c[2]:.2f}%")
        
    except ImportError as e:
        print(f"\n❌ Extensión C no disponible: {e}")
    
    # Test Python/NumPy
    def compute_metrics_numpy(pnl, pnl_pct, saldo, saldo_ini):
        n = len(pnl)
        roi = 100 * (saldo[-1] - saldo_ini) / saldo_ini
        winrate = 100 * np.sum(pnl > 0) / n
        peak = np.maximum.accumulate(saldo)
        dd = 100 * (peak - saldo) / peak
        max_dd = np.max(dd)
        return roi, winrate, max_dd
    
    # Warmup
    compute_metrics_numpy(pnl_neto, pnl_pct, saldo_despues, saldo_inicial)
    
    # Benchmark
    t0 = time.perf_counter()
    n_runs = 10000
    for _ in range(n_runs):
        result_py = compute_metrics_numpy(pnl_neto, pnl_pct, saldo_despues, saldo_inicial)
    t_py = (time.perf_counter() - t0) / n_runs * 1000
    
    print(f"\n🟡 Python/NumPy:")
    print(f"   Tiempo promedio: {t_py:.4f} ms")
    print(f"   ROI: {result_py[0]:.2f}%, Winrate: {result_py[1]:.2f}%, DD: {result_py[2]:.2f}%")
    
    if t_c:
        speedup = t_py / t_c
        print(f"\n⚡ SPEEDUP: {speedup:.2f}x más rápido con C")


def benchmark_perturb_returns():
    """Benchmark de perturbación de retornos."""
    print("\n" + "=" * 70)
    print("BENCHMARK: perturb_returns")
    print("=" * 70)
    
    n_bars = 100_000
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    close = np.maximum(close, 1000).astype(np.float64)
    
    noise_factor = 0.3
    seed = 12345
    
    # Test C
    t_c = None
    try:
        from cp import perturb_returns_c, C_AVAILABLE
        if not C_AVAILABLE:
            raise ImportError("C no disponible")
        
        # Warmup
        perturb_returns_c(close, noise_factor, seed)
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 100
        for _ in range(n_runs):
            result_c = perturb_returns_c(close, noise_factor, seed)
        t_c = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"\n✅ Extensión C:")
        print(f"   Tiempo promedio: {t_c:.3f} ms")
        print(f"   Diff media: {np.mean(np.abs(result_c - close)):.2f}")
        
    except ImportError as e:
        print(f"\n❌ Extensión C no disponible: {e}")
    
    # Test Python/NumPy
    def perturb_returns_numpy(close_prices, noise_factor, seed):
        rng = np.random.default_rng(seed)
        returns = np.diff(close_prices) / np.maximum(close_prices[:-1], 1e-10)
        volatility = np.std(returns)
        noise = rng.normal(0, volatility * noise_factor, len(returns))
        perturbed_returns = returns + noise
        new_close = np.zeros(len(close_prices))
        new_close[0] = close_prices[0]
        new_close[1:] = close_prices[0] * (1 + np.cumsum(perturbed_returns))
        return new_close
    
    # Warmup
    perturb_returns_numpy(close, noise_factor, seed)
    
    # Benchmark
    t0 = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        result_py = perturb_returns_numpy(close, noise_factor, seed)
    t_py = (time.perf_counter() - t0) / n_runs * 1000
    
    print(f"\n🟡 Python/NumPy:")
    print(f"   Tiempo promedio: {t_py:.3f} ms")
    print(f"   Diff media: {np.mean(np.abs(result_py - close)):.2f}")
    
    if t_c:
        speedup = t_py / t_c
        print(f"\n⚡ SPEEDUP: {speedup:.2f}x más rápido con C")


def benchmark_cvar_r2():
    """Benchmark de CVaR y R²."""
    print("\n" + "=" * 70)
    print("BENCHMARK: compute_cvar_95 + compute_equity_r2")
    print("=" * 70)
    
    n_points = 10_000
    np.random.seed(42)
    equity = 10000 + np.cumsum(np.random.randn(n_points) * 10 + 1)
    equity = np.maximum(equity, 100).astype(np.float64)
    
    # Test C
    t_c = None
    try:
        from cp import compute_cvar_95_c, compute_equity_r2_c, C_AVAILABLE
        if not C_AVAILABLE:
            raise ImportError("C no disponible")
        
        # Warmup
        compute_cvar_95_c(equity)
        compute_equity_r2_c(equity)
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 1000
        for _ in range(n_runs):
            cvar_c = compute_cvar_95_c(equity)
            r2_c = compute_equity_r2_c(equity)
        t_c = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"\n✅ Extensión C:")
        print(f"   Tiempo promedio: {t_c:.4f} ms")
        print(f"   CVaR 95%: {cvar_c:.2f}%, R²: {r2_c:.4f}")
        
    except ImportError as e:
        print(f"\n❌ Extensión C no disponible: {e}")
    
    # Test Python/NumPy
    def compute_cvar_numpy(eq):
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
        sorted_ret = np.sort(returns)
        n_tail = max(1, int(len(sorted_ret) * 0.05))
        return -100 * np.mean(sorted_ret[:n_tail])
    
    def compute_r2_numpy(eq):
        log_eq = np.log(np.maximum(eq, 1e-10))
        x = np.arange(len(eq))
        corr = np.corrcoef(x, log_eq)[0, 1]
        return corr ** 2
    
    # Warmup
    compute_cvar_numpy(equity)
    compute_r2_numpy(equity)
    
    # Benchmark
    t0 = time.perf_counter()
    n_runs = 1000
    for _ in range(n_runs):
        cvar_py = compute_cvar_numpy(equity)
        r2_py = compute_r2_numpy(equity)
    t_py = (time.perf_counter() - t0) / n_runs * 1000
    
    print(f"\n🟡 Python/NumPy:")
    print(f"   Tiempo promedio: {t_py:.4f} ms")
    print(f"   CVaR 95%: {cvar_py:.2f}%, R²: {r2_py:.4f}")
    
    if t_c:
        speedup = t_py / t_c
        print(f"\n⚡ SPEEDUP: {speedup:.2f}x más rápido con C")


def benchmark_aggregate_metrics():
    """Benchmark de agregación de métricas vecinales."""
    print("\n" + "=" * 70)
    print("BENCHMARK: aggregate_neighbor_metrics")
    print("=" * 70)
    
    n_neighbors = 10
    np.random.seed(42)
    
    scores = (np.random.rand(n_neighbors) * 50 + 10).astype(np.float64)
    sharpes = (np.random.rand(n_neighbors) * 2 - 0.5).astype(np.float64)
    cvars = (np.random.rand(n_neighbors) * 30 + 5).astype(np.float64)
    r2s = (np.random.rand(n_neighbors) * 0.8 + 0.1).astype(np.float64)
    
    lambda_penalty = 1.5
    
    # Test C
    t_c = None
    try:
        from cp import aggregate_neighbor_metrics_c, C_AVAILABLE
        if not C_AVAILABLE:
            raise ImportError("C no disponible")
        
        # Warmup
        aggregate_neighbor_metrics_c(scores, sharpes, cvars, r2s, lambda_penalty)
        
        # Benchmark
        t0 = time.perf_counter()
        n_runs = 10000
        for _ in range(n_runs):
            result_c = aggregate_neighbor_metrics_c(scores, sharpes, cvars, r2s, lambda_penalty)
        t_c = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"\n✅ Extensión C:")
        print(f"   Tiempo promedio: {t_c:.5f} ms")
        print(f"   Robust: {result_c[0]:.2f}, Mean: {result_c[1]:.2f}, Std: {result_c[2]:.2f}")
        
    except ImportError as e:
        print(f"\n❌ Extensión C no disponible: {e}")
    
    # Test Python/NumPy
    def aggregate_numpy(scores, sharpes, cvars, r2s, lp):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        worst_cvar = np.max(cvars)
        avg_r2 = np.mean(r2s)
        robust = mean_score - lp * std_score
        return (max(0, robust), mean_score, std_score, worst_cvar, avg_r2)
    
    # Warmup
    aggregate_numpy(scores, sharpes, cvars, r2s, lambda_penalty)
    
    # Benchmark
    t0 = time.perf_counter()
    n_runs = 10000
    for _ in range(n_runs):
        result_py = aggregate_numpy(scores, sharpes, cvars, r2s, lambda_penalty)
    t_py = (time.perf_counter() - t0) / n_runs * 1000
    
    print(f"\n🟡 Python/NumPy:")
    print(f"   Tiempo promedio: {t_py:.5f} ms")
    print(f"   Robust: {result_py[0]:.2f}, Mean: {result_py[1]:.2f}, Std: {result_py[2]:.2f}")
    
    if t_c:
        speedup = t_py / t_c
        print(f"\n⚡ SPEEDUP: {speedup:.2f}x más rápido con C")


def main():
    print("=" * 70)
    print("🚀 MODELOX NUCLEAR ENGINE - BENCHMARK DE RENDIMIENTO v3")
    print("=" * 70)
    
    # Mostrar info de C
    try:
        from cp import C_AVAILABLE, get_version
        print(f"\n📦 Extensiones C: {'✅ Disponibles' if C_AVAILABLE else '❌ No compiladas'}")
        if C_AVAILABLE:
            print(f"   Versión: {get_version()}")
    except ImportError:
        print("\n📦 Extensiones C: ❌ Módulo no encontrado")
    
    benchmark_simulate_trades()
    benchmark_metrics()
    benchmark_perturb_returns()
    benchmark_cvar_r2()
    benchmark_aggregate_metrics()
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()

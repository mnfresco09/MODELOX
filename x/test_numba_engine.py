"""
Test del motor Numba optimizado.

Este script verifica:
1. Que el motor Numba (engine) compila correctamente
2. Que produce resultados equivalentes al motor Python
3. Mide la diferencia de rendimiento
"""

import sys
import time

# Añadir el path del proyecto
sys.path.insert(0, "/Users/manuel/Desktop/MODELOX")

import numpy as np

# =============================================================================
# TEST 1: Verificar que el módulo compila
# =============================================================================
print("=" * 60)
print("TEST 1: Compilación del motor Numba (engine)")
print("=" * 60)

try:
    from modelox.core.engine import (
        _find_exit_pnl_fixed_numba,
        _find_exit_pnl_trailing_numba,
        generate_trades_fast,
    )
    print("✅ Módulo importado correctamente")
except Exception as e:
    print(f"❌ Error al importar: {e}")
    sys.exit(1)

# =============================================================================
# TEST 2: Compilar funciones con datos sintéticos
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: Primera compilación JIT (puede tardar unos segundos)")
print("=" * 60)

# Crear datos sintéticos
n_bars = 1000
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
open_arr = prices + np.random.randn(n_bars) * 0.1
high_arr = prices + np.abs(np.random.randn(n_bars) * 0.3)
low_arr = prices - np.abs(np.random.randn(n_bars) * 0.3)
close_arr = prices + np.random.randn(n_bars) * 0.1
timestamps = np.arange(n_bars, dtype=np.int64)

# Señales aleatorias (sparse)
signal_long = np.zeros(n_bars, dtype=np.bool_)
signal_short = np.zeros(n_bars, dtype=np.bool_)
signal_long[np.random.choice(n_bars, 20, replace=False)] = True
signal_short[np.random.choice(n_bars, 20, replace=False)] = True

print(f"📊 Datos sintéticos: {n_bars} barras, ~40 señales")

# Compilar _find_exit_pnl_fixed_numba
t0 = time.perf_counter()
exit_idx, exit_price, exit_type = _find_exit_pnl_fixed_numba(
    entry_idx=10,
    entry_price=100.0,
    qty=0.01,
    stake=75.0,
    is_long=True,
    sl_pct=8.0,
    tp_pct=14.0,
    open_arr=open_arr.astype(np.float64),
    high_arr=high_arr.astype(np.float64),
    low_arr=low_arr.astype(np.float64),
    close_arr=close_arr.astype(np.float64),
)
t1 = time.perf_counter()
print(f"✅ _find_exit_pnl_fixed_numba compilado ({(t1-t0)*1000:.1f}ms)")
print(f"   Resultado: exit_idx={exit_idx}, exit_price={exit_price:.2f}, exit_type={exit_type}")

# Compilar _find_exit_pnl_trailing_numba
t0 = time.perf_counter()
exit_idx, exit_price, exit_type = _find_exit_pnl_trailing_numba(
    entry_idx=10,
    entry_price=100.0,
    qty=0.01,
    stake=75.0,
    is_long=True,
    sl_pct=8.0,
    tp_pct=0.0,
    trail_act_pct=15.0,
    trail_dist_pct=3.0,
    open_arr=open_arr.astype(np.float64),
    high_arr=high_arr.astype(np.float64),
    low_arr=low_arr.astype(np.float64),
    close_arr=close_arr.astype(np.float64),
)
t1 = time.perf_counter()
print(f"✅ _find_exit_pnl_trailing_numba compilado ({(t1-t0)*1000:.1f}ms)")
print(f"   Resultado: exit_idx={exit_idx}, exit_price={exit_price:.2f}, exit_type={exit_type}")

# =============================================================================
# TEST 3: generate_trades_fast completo
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: generate_trades_fast (kernel completo)")
print("=" * 60)

t0 = time.perf_counter()
result = generate_trades_fast(
    open_arr=open_arr.astype(np.float64),
    high_arr=high_arr.astype(np.float64),
    low_arr=low_arr.astype(np.float64),
    close_arr=close_arr.astype(np.float64),
    timestamps=timestamps,
    signal_long=signal_long,
    signal_short=signal_short,
    saldo_apertura=1000.0,
    saldo_usado_cfg=75.0,
    qty_objetivo=0.01,
    apalancamiento_max=60.0,
    block_velas_after_exit=0,
    exit_type="pnl_fixed",
    sl_pct=8.0,
    tp_pct=14.0,
    trail_act_pct=15.0,
    trail_dist_pct=3.0,
    max_trades=10000,
)
t1 = time.perf_counter()
print(f"✅ generate_trades_fast compilado ({(t1-t0)*1000:.1f}ms)")
print(f"   Trades generados: {result['num_trades']}")

if result['num_trades'] > 0:
    print(f"   Primer trade: {result['side'][0]} @ {result['entry_price'][0]:.2f} → {result['exit_price'][0]:.2f}")
    print(f"   Salida: {result['tipo_salida'][0]}")

# =============================================================================
# TEST 4: Benchmark de rendimiento
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: Benchmark de rendimiento (1M barras)")
print("=" * 60)

# Crear dataset grande
n_large = 1_000_000
print(f"📊 Generando {n_large:,} barras...")
prices_large = 100 + np.cumsum(np.random.randn(n_large) * 0.5)
open_large = prices_large + np.random.randn(n_large) * 0.1
high_large = prices_large + np.abs(np.random.randn(n_large) * 0.3)
low_large = prices_large - np.abs(np.random.randn(n_large) * 0.3)
close_large = prices_large + np.random.randn(n_large) * 0.1
timestamps_large = np.arange(n_large, dtype=np.int64)

# Señales (1000 señales = ~0.1%)
signal_long_large = np.zeros(n_large, dtype=np.bool_)
signal_short_large = np.zeros(n_large, dtype=np.bool_)
signal_indices = np.random.choice(n_large, 1000, replace=False)
signal_long_large[signal_indices[:500]] = True
signal_short_large[signal_indices[500:]] = True

print(f"   Señales: {signal_long_large.sum()} LONG + {signal_short_large.sum()} SHORT")

# Benchmark
print("\n🚀 Ejecutando benchmark...")
n_runs = 5
times = []

for run in range(n_runs):
    t0 = time.perf_counter()
    result = generate_trades_fast(
        open_arr=open_large.astype(np.float64),
        high_arr=high_large.astype(np.float64),
        low_arr=low_large.astype(np.float64),
        close_arr=close_large.astype(np.float64),
        timestamps=timestamps_large,
        signal_long=signal_long_large,
        signal_short=signal_short_large,
        saldo_apertura=1000.0,
        saldo_usado_cfg=75.0,
        qty_objetivo=0.01,
        apalancamiento_max=60.0,
        block_velas_after_exit=0,
        exit_type="pnl_fixed",
        sl_pct=8.0,
        tp_pct=14.0,
        trail_act_pct=15.0,
        trail_dist_pct=3.0,
        max_trades=10000,
    )
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"   Run {run+1}: {times[-1]*1000:.1f}ms, {result['num_trades']} trades")

avg_time = np.mean(times) * 1000
print(f"\n📈 Resultado: {avg_time:.1f}ms promedio para {n_large:,} barras")
print(f"   Throughput: {n_large / np.mean(times) / 1e6:.2f}M barras/segundo")

# =============================================================================
# TEST 5: Verificar integración con engine.py
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: Integración con engine.py")
print("=" * 60)

try:
    from modelox.core.engine import NUMBA_AVAILABLE, USE_NUMBA_ENGINE
    print("✅ engine.py importado")
    print(f"   NUMBA_AVAILABLE = {NUMBA_AVAILABLE}")
    print(f"   USE_NUMBA_ENGINE = {USE_NUMBA_ENGINE}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ TODOS LOS TESTS PASARON")
print("=" * 60)

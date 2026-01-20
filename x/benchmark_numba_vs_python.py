"""
Benchmark comparativo: Motor Numba vs Motor Python.

Este script compara el rendimiento del motor optimizado con Numba
contra el motor Python puro usando datos reales del proyecto.
"""

import sys
import os
import time

sys.path.insert(0, "/Users/manuel/Desktop/MODELOX")

import numpy as np
import polars as pl

print("=" * 70)
print("BENCHMARK: MOTOR NUMBA vs MOTOR PYTHON")
print("=" * 70)

# =============================================================================
# CARGAR DATOS REALES
# =============================================================================
print("\n📊 Cargando datos BTC 1h...")

data_path = "/Users/manuel/Desktop/MODELOX/data/ohlcv/BTC_ohlcv_1h.feather"
if os.path.exists(data_path):
    df = pl.read_ipc(data_path)
    print(f"   Filas: {len(df):,}")
    print(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
else:
    print(f"❌ No se encontró: {data_path}")
    sys.exit(1)

# =============================================================================
# GENERAR SEÑALES SINTÉTICAS
# =============================================================================
print("\n🎯 Generando señales sintéticas...")

# Simular señales de una estrategia (1% de las barras tienen señal)
n = len(df)
np.random.seed(42)
signal_rate = 0.01

signal_long = np.zeros(n, dtype=bool)
signal_short = np.zeros(n, dtype=bool)

# Generar señales aleatorias
n_signals = int(n * signal_rate)
signal_indices = np.random.choice(n, n_signals, replace=False)
for i, idx in enumerate(signal_indices):
    if i % 2 == 0:
        signal_long[idx] = True
    else:
        signal_short[idx] = True

# Añadir columnas al DataFrame
df = df.with_columns([
    pl.Series("signal_long", signal_long),
    pl.Series("signal_short", signal_short),
])

print(f"   Señales LONG: {signal_long.sum()}")
print(f"   Señales SHORT: {signal_short.sum()}")

# =============================================================================
# SETUP
# =============================================================================
# Parámetros de trading
params = {
    "__saldo_usado": 75.0,
    "__apalancamiento_max": 60.0,
    "__qty_max_activo": 0.01,
    "block_velas_after_exit": 0,
    "__exit_type": "pnl_fixed",
    "__exit_sl_pct": 8.0,
    "__exit_tp_pct": 14.0,
    "__exit_trail_act_pct": 15.0,
    "__exit_trail_dist_pct": 3.0,
}

saldo_apertura = 1000.0

# =============================================================================
# BENCHMARK MOTOR NUMBA
# =============================================================================
print("\n" + "=" * 70)
print("🚀 MOTOR NUMBA")
print("=" * 70)

from modelox.core.engine import generate_trades_fast
from modelox.core.exits import exit_settings_from_params

# Extraer arrays
close = df["close"].to_numpy()
open_ = df["open"].to_numpy()
high = df["high"].to_numpy()
low = df["low"].to_numpy()
timestamps = df["timestamp"].to_numpy()
sig_long = df["signal_long"].to_numpy()
sig_short = df["signal_short"].to_numpy()

exit_settings = exit_settings_from_params(params)

# Primera ejecución (compilación)
print("\n📦 Primera ejecución (incluye compilación JIT)...")
t0 = time.perf_counter()
result_numba = generate_trades_fast(
    open_arr=open_,
    high_arr=high,
    low_arr=low,
    close_arr=close,
    timestamps=timestamps,
    signal_long=sig_long,
    signal_short=sig_short,
    saldo_apertura=saldo_apertura,
    saldo_usado_cfg=float(params["__saldo_usado"]),
    qty_objetivo=float(params["__qty_max_activo"]),
    apalancamiento_max=float(params["__apalancamiento_max"]),
    block_velas_after_exit=int(params["block_velas_after_exit"]),
    exit_type=str(params["__exit_type"]),
    sl_pct=float(params["__exit_sl_pct"]),
    tp_pct=float(params["__exit_tp_pct"]),
    trail_act_pct=float(params["__exit_trail_act_pct"]),
    trail_dist_pct=float(params["__exit_trail_dist_pct"]),
)
t1 = time.perf_counter()
print(f"   Primera ejecución: {(t1-t0)*1000:.1f}ms")
print(f"   Trades generados: {result_numba['num_trades']}")

# Benchmark (ya compilado)
print("\n⏱️  Benchmark (5 ejecuciones)...")
numba_times = []
for i in range(5):
    t0 = time.perf_counter()
    result_numba = generate_trades_fast(
        open_arr=open_,
        high_arr=high,
        low_arr=low,
        close_arr=close,
        timestamps=timestamps,
        signal_long=sig_long,
        signal_short=sig_short,
        saldo_apertura=saldo_apertura,
        saldo_usado_cfg=float(params["__saldo_usado"]),
        qty_objetivo=float(params["__qty_max_activo"]),
        apalancamiento_max=float(params["__apalancamiento_max"]),
        block_velas_after_exit=int(params["block_velas_after_exit"]),
        exit_type=str(params["__exit_type"]),
        sl_pct=float(params["__exit_sl_pct"]),
        tp_pct=float(params["__exit_tp_pct"]),
        trail_act_pct=float(params["__exit_trail_act_pct"]),
        trail_dist_pct=float(params["__exit_trail_dist_pct"]),
    )
    t1 = time.perf_counter()
    numba_times.append(t1 - t0)
    print(f"   Run {i+1}: {numba_times[-1]*1000:.2f}ms")

numba_avg = np.mean(numba_times) * 1000
print(f"\n   PROMEDIO NUMBA: {numba_avg:.2f}ms")

# =============================================================================
# BENCHMARK MOTOR PYTHON
# =============================================================================
print("\n" + "=" * 70)
print("🐍 MOTOR PYTHON (original)")
print("=" * 70)

from modelox.core.engine import _generate_trades_python

# Mock de estrategia (sin salida personalizada)
class MockStrategy:
    name = "MockStrategy"

strategy = MockStrategy()

print("\n⏱️  Benchmark (5 ejecuciones)...")
python_times = []
for i in range(5):
    t0 = time.perf_counter()
    result_python = _generate_trades_python(
        df=df,
        params=params,
        saldo_apertura=saldo_apertura,
        strategy=strategy,
    )
    t1 = time.perf_counter()
    python_times.append(t1 - t0)
    print(f"   Run {i+1}: {python_times[-1]*1000:.2f}ms, trades: {len(result_python)}")

python_avg = np.mean(python_times) * 1000
print(f"\n   PROMEDIO PYTHON: {python_avg:.2f}ms")

# =============================================================================
# COMPARACIÓN
# =============================================================================
print("\n" + "=" * 70)
print("📊 COMPARACIÓN")
print("=" * 70)

speedup = python_avg / numba_avg
print(f"\n   Motor Python:  {python_avg:.2f}ms")
print(f"   Motor Numba:   {numba_avg:.2f}ms")
print(f"\n   🏆 SPEEDUP: {speedup:.1f}x más rápido con Numba")

# Verificar consistencia de resultados
print("\n" + "=" * 70)
print("✅ VERIFICACIÓN DE CONSISTENCIA")
print("=" * 70)

numba_trades = result_numba["num_trades"]
python_trades = len(result_python)

print(f"\n   Trades Numba:  {numba_trades}")
print(f"   Trades Python: {python_trades}")

if numba_trades > 0 and python_trades > 0:
    # Comparar primeros trades
    print("\n   Primer trade NUMBA:")
    print(f"     Entry: idx={result_numba['entry_idx'][0]}, price={result_numba['entry_price'][0]:.2f}")
    print(f"     Exit:  idx={result_numba['exit_idx'][0]}, price={result_numba['exit_price'][0]:.2f}")
    print(f"     Side:  {result_numba['side'][0]}, Salida: {result_numba['tipo_salida'][0]}")
    
    print("\n   Primer trade PYTHON:")
    row = result_python.iloc[0]
    print(f"     Entry: idx={row['entry_idx']}, price={row['entry_price']:.2f}")
    print(f"     Exit:  idx={row['exit_idx']}, price={row['exit_price']:.2f}")
    print(f"     Side:  {row['type']}, Salida: {row['tipo_salida']}")

print("\n" + "=" * 70)
print("✅ BENCHMARK COMPLETADO")
print("=" * 70)

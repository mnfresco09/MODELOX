"""
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS DE OPTIMIZACIÓN DEL SISTEMA MODELOX
═══════════════════════════════════════════════════════════════════════════════

Este documento identifica las áreas que pueden optimizarse para agilizar
el proceso de backtesting, similar a lo hecho con engine_numba.py.

ESTADO ACTUAL DEL SISTEMA DESPUÉS DE OPTIMIZACIÓN NUMBA:
- Engine (generate_trades): 136x más rápido con Numba ✅
- Bottleneck movido a otras áreas del pipeline

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
sys.path.insert(0, "/Users/manuel/Desktop/MODELOX")

import numpy as np
import polars as pl
import pandas as pd

# =============================================================================
# 1. ANÁLISIS DE TIEMPOS POR FASE
# =============================================================================
print("=" * 70)
print("ANÁLISIS DE BOTTLENECKS EN EL PIPELINE DE BACKTESTING")
print("=" * 70)

# Cargar datos de prueba
from modelox.core.data import load_data

print("\n📊 Cargando datos BTC 1h...")
t0 = time.perf_counter()
df = load_data("/Users/manuel/Desktop/MODELOX/data/ohlcv/BTC_ohlcv_1h.feather")
t_load = time.perf_counter() - t0
print(f"   Tiempo carga datos: {t_load*1000:.1f}ms ({len(df):,} filas)")

# =============================================================================
# 2. GENERATE_SIGNALS (Estrategia)
# =============================================================================
print("\n" + "─" * 70)
print("FASE 1: GENERATE_SIGNALS (Estrategia)")
print("─" * 70)

from modelox.strategies.QUANTUM_OSCILLATOR_PRO import EstrategiaQuantumOscillatorPro

strategy = EstrategiaQuantumOscillatorPro()
params = {
    "length_zscore": 21,
    "length_roc": 12,
    "ma_type": "ALMA",
    "smooth_length": 3,
    "alma_offset": 0.85,
    "alma_sigma": 6,
    "zone_threshold": 2.0,
    "exit_reversal_bars": 2,
    "emergency_sl_pct": 0.10,
}

times_signals = []
for i in range(10):
    t0 = time.perf_counter()
    df_signals = strategy.generate_signals(df, params)
    times_signals.append(time.perf_counter() - t0)

avg_signals = np.mean(times_signals) * 1000
print(f"   Promedio: {avg_signals:.2f}ms (10 ejecuciones)")
print(f"   Status: {'✅ Vectorizado con Polars' if avg_signals < 50 else '⚠️ Potencial optimización'}")

# =============================================================================
# 3. GENERATE_TRADES (Motor Numba)
# =============================================================================
print("\n" + "─" * 70)
print("FASE 2: GENERATE_TRADES (Motor Numba)")
print("─" * 70)

from modelox.core.engine import generate_trades, NUMBA_AVAILABLE

class MockStrategy:
    name = "MockStrategy"
    def suggest_params(self, trial):
        return {}

params_rt = dict(params)
params_rt["__saldo_usado"] = 75.0
params_rt["__apalancamiento_max"] = 60.0
params_rt["__qty_max_activo"] = 0.01
params_rt["__exit_type"] = "pnl_fixed"
params_rt["__exit_sl_pct"] = 8.0
params_rt["__exit_tp_pct"] = 14.0

# Añadir señales si no existen
if "signal_long" not in df_signals.columns:
    df_signals = df_signals.with_columns([
        pl.lit(False).alias("signal_long"),
        pl.lit(False).alias("signal_short"),
    ])

times_trades = []
for i in range(10):
    t0 = time.perf_counter()
    trades = generate_trades(df_signals, params_rt, saldo_apertura=1000.0, strategy=MockStrategy())
    times_trades.append(time.perf_counter() - t0)

avg_trades = np.mean(times_trades) * 1000
print(f"   Motor activo: {'Numba' if NUMBA_AVAILABLE else 'Python'}")
print(f"   Promedio: {avg_trades:.2f}ms (10 ejecuciones)")
print(f"   Trades generados: {len(trades) if not trades.empty else 0}")
print("   Status: ✅ Ya optimizado con Numba")

# =============================================================================
# 4. SIMULATE_TRADES
# =============================================================================
print("\n" + "─" * 70)
print("FASE 3: SIMULATE_TRADES")
print("─" * 70)

from modelox.core.engine import simulate_trades
from modelox.core.types import BacktestConfig

config = BacktestConfig(
    saldo_inicial=1000.0,
    saldo_operativo_max=10000.0,
    comision_pct=0.0006,
    comision_sides=2,
    saldo_minimo_operativo=50.0,
    qty_max_activo=0.01,
    saldo_usado=75.0,
    apalancamiento_max=60.0,
)

if not trades.empty:
    times_sim = []
    for i in range(10):
        t0 = time.perf_counter()
        trades_exec, equity_curve = simulate_trades(trades_base=trades, config=config)
        times_sim.append(time.perf_counter() - t0)
    
    avg_sim = np.mean(times_sim) * 1000
    print(f"   Promedio: {avg_sim:.2f}ms (10 ejecuciones)")
    print(f"   Status: {'⚠️ CANDIDATO A NUMBA' if avg_sim > 1 else '✅ Suficientemente rápido'}")
else:
    print("   ⚠️ No hay trades para simular")
    avg_sim = 0

# =============================================================================
# 5. RESUMEN_METRICAS
# =============================================================================
print("\n" + "─" * 70)
print("FASE 4: RESUMEN_METRICAS")
print("─" * 70)

from modelox.core.metrics import resumen_metricas

if not trades.empty and 'trades_exec' in dir():
    times_metrics = []
    for i in range(100):
        t0 = time.perf_counter()
        metricas = resumen_metricas(
            trades_exec,
            saldo_inicial=1000.0,
            equity_curve=equity_curve,
        )
        times_metrics.append(time.perf_counter() - t0)
    
    avg_metrics = np.mean(times_metrics) * 1000
    print(f"   Promedio: {avg_metrics:.2f}ms (100 ejecuciones)")
    print(f"   Status: {'⚠️ CANDIDATO A NUMBA' if avg_metrics > 5 else '✅ Suficientemente rápido'}")
else:
    print("   ⚠️ No hay trades para calcular métricas")
    avg_metrics = 0

# =============================================================================
# 6. SCORE_OPTUNA
# =============================================================================
print("\n" + "─" * 70)
print("FASE 5: SCORE_OPTUNA")
print("─" * 70)

from modelox.core.scoring import score_optuna

if 'metricas' in dir():
    times_score = []
    for i in range(1000):
        t0 = time.perf_counter()
        score = score_optuna(metricas)
        times_score.append(time.perf_counter() - t0)
    
    avg_score = np.mean(times_score) * 1000
    print(f"   Promedio: {avg_score:.4f}ms (1000 ejecuciones)")
    print("   Status: ✅ Ya es muy rápido (pure math)")
else:
    avg_score = 0

# =============================================================================
# 7. REPORTING (Gráficos)
# =============================================================================
print("\n" + "─" * 70)
print("FASE 6: REPORTING (Conversión Polars → Pandas)")
print("─" * 70)

times_convert = []
for i in range(10):
    t0 = time.perf_counter()
    df_pandas = df_signals.to_pandas()
    if "timestamp" in df_pandas.columns:
        df_pandas["timestamp"] = pd.to_datetime(df_pandas["timestamp"], utc=True)
        df_pandas = df_pandas.set_index("timestamp")
    times_convert.append(time.perf_counter() - t0)

avg_convert = np.mean(times_convert) * 1000
print(f"   Conversión Polars→Pandas: {avg_convert:.2f}ms")
print(f"   Status: {'⚠️ CANDIDATO A OPTIMIZAR' if avg_convert > 50 else '✅ Aceptable'}")

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN DE TIEMPOS POR TRIAL")
print("=" * 70)

total_per_trial = avg_signals + avg_trades + avg_sim + avg_metrics + avg_score

print(f"""
   ┌───────────────────────────────────────┬───────────────┬─────────────┐
   │ Fase                                  │ Tiempo (ms)   │ % del Total │
   ├───────────────────────────────────────┼───────────────┼─────────────┤
   │ 1. generate_signals (estrategia)      │ {avg_signals:>10.2f}    │ {100*avg_signals/total_per_trial:>8.1f}%   │
   │ 2. generate_trades (Numba)            │ {avg_trades:>10.2f}    │ {100*avg_trades/total_per_trial:>8.1f}%   │
   │ 3. simulate_trades                    │ {avg_sim:>10.2f}    │ {100*avg_sim/total_per_trial:>8.1f}%   │
   │ 4. resumen_metricas                   │ {avg_metrics:>10.2f}    │ {100*avg_metrics/total_per_trial:>8.1f}%   │
   │ 5. score_optuna                       │ {avg_score:>10.4f}    │ {100*avg_score/total_per_trial:>8.1f}%   │
   ├───────────────────────────────────────┼───────────────┼─────────────┤
   │ TOTAL POR TRIAL                       │ {total_per_trial:>10.2f}    │   100.0%    │
   └───────────────────────────────────────┴───────────────┴─────────────┘
""")

trials_per_sec = 1000 / total_per_trial if total_per_trial > 0 else 0
print(f"   📈 Throughput estimado: {trials_per_sec:.1f} trials/segundo")
print(f"   📈 Tiempo para 1000 trials: {total_per_trial:.0f} segundos = {total_per_trial/60:.1f} minutos")

# =============================================================================
# 9. RECOMENDACIONES
# =============================================================================
print("\n" + "=" * 70)
print("🔧 ÁREAS DE OPTIMIZACIÓN IDENTIFICADAS")
print("=" * 70)

recommendations = []

if avg_signals > 20:
    recommendations.append({
        "area": "generate_signals",
        "impacto": "ALTO" if avg_signals > 50 else "MEDIO",
        "accion": "Pre-compilar indicadores comunes con Polars expressions o Numba",
        "ganancia_potencial": f"{avg_signals - 5:.1f}ms/trial"
    })

if avg_sim > 1:
    recommendations.append({
        "area": "simulate_trades",
        "impacto": "MEDIO",
        "accion": "Ya implementado simulate_trades_fast en engine_numba.py - integrar",
        "ganancia_potencial": f"{avg_sim - 0.5:.1f}ms/trial"
    })

if avg_metrics > 5:
    recommendations.append({
        "area": "resumen_metricas",
        "impacto": "BAJO",
        "accion": "Vectorizar cálculos con NumPy puro, evitar pandas groupby",
        "ganancia_potencial": f"{avg_metrics - 2:.1f}ms/trial"
    })

if avg_convert > 30:
    recommendations.append({
        "area": "Polars → Pandas",
        "impacto": "MEDIO",
        "accion": "Solo convertir cuando se necesite gráfico (ya implementado parcialmente)",
        "ganancia_potencial": "Eliminar en trials sin gráfico"
    })

# Añadir recomendación sobre paralelización
recommendations.append({
    "area": "Optuna Paralelo",
    "impacto": "MUY ALTO",
    "accion": "Usar n_jobs > 1 en OptunaConfig (ya soportado)",
    "ganancia_potencial": "Speedup lineal con núcleos CPU"
})

# Añadir recomendación sobre caching
recommendations.append({
    "area": "Caching de Indicadores",
    "impacto": "ALTO",
    "accion": "Cachear indicadores base (zscore, roc) que no cambian entre trials",
    "ganancia_potencial": "~50% del tiempo de generate_signals"
})

for i, rec in enumerate(recommendations, 1):
    print(f"""
   {i}. {rec['area']}
      Impacto: {rec['impacto']}
      Acción: {rec['accion']}
      Ganancia: {rec['ganancia_potencial']}
""")

print("\n" + "=" * 70)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 70)

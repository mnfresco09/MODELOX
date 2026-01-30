# 🚀 MODELOX C Extensions (Nuclear Engine)

Motor de alto rendimiento para MODELOX usando Cython y paralelización.

## Instalación

```bash
cd cp/
make  # O: python setup.py build_ext --inplace
```

### En Linux/Nix (si GCC no está en PATH):
```bash
export CC=/ruta/a/gcc
make
```

## Funciones Disponibles (v3.0.0)

### Kernels de Simulación
| Función | Descripción | Speedup vs NumPy |
|---------|-------------|------------------|
| `simulate_trades_c()` | Simulación de trades vectorizada | 2-5x |
| `find_exits_c()` | Búsqueda de puntos de salida | 3-8x |
| `compute_metrics_c()` | Cálculo de métricas completo | 5x |

### Kernels de Análisis
| Función | Descripción | Speedup |
|---------|-------------|---------|
| `compute_drawdown_c()` | Máximo drawdown | 2x |
| `compute_sharpe_c()` | Sharpe ratio | 3x |
| `compute_sqn_c()` | System Quality Number | 3x |
| `perturb_returns_c()` | Perturbación gaussiana | 10x |
| `compute_cvar_95_c()` | Conditional Value at Risk | 2x |
| `compute_equity_r2_c()` | R² de curva de equity | 2x |
| `aggregate_neighbor_metrics_c()` | Agregación de métricas | 36x |

## Uso en Código

```python
from cp import (
    C_AVAILABLE,
    simulate_trades_c,
    compute_metrics_c,
    compute_cvar_95_c,
    aggregate_neighbor_metrics_c,
)

if C_AVAILABLE:
    # Usar versión C (más rápida)
    metrics = compute_metrics_c(pnl, equity, n_trades, total_pnl, commission, capital)
else:
    # Fallback a NumPy
    pass
```

## Variables de Entorno

| Variable | Valores | Default | Descripción |
|----------|---------|---------|-------------|
| `MODELOX_PARALLEL` | 0/1 | 1 | Habilitar paralelización global |
| `MODELOX_MAX_WORKERS` | 1-N | 8 | Workers máximos para trials |
| `MODELOX_NEIGHBORHOOD_PARALLEL` | 0/1 | 1 | Paralelizar análisis de vecinos |
| `MODELOX_NEIGHBORHOOD_WORKERS` | 1-N | 6 | Workers para vecinos |

## Benchmark

Ejecutar:
```bash
cd cp/
python benchmark.py
```

Resultados típicos (Apple M1):
```
📦 Extensiones C: ✅ Disponibles (v3.0.0)

BENCHMARK: compute_metrics
✅ C: 0.0064 ms | 🟡 NumPy: 0.0331 ms | ⚡ SPEEDUP: 5.16x

BENCHMARK: compute_cvar_95 + compute_equity_r2
✅ C: 0.2786 ms | 🟡 NumPy: 0.5730 ms | ⚡ SPEEDUP: 2.06x

BENCHMARK: aggregate_neighbor_metrics
✅ C: 0.00222 ms | 🟡 NumPy: 0.08050 ms | ⚡ SPEEDUP: 36.20x
```

## Arquitectura

```
cp/
├── __init__.py          # Auto-detección y fallbacks
├── nuclear_engine.pyx   # Kernels Cython
├── parallel_engine.py   # Multiprocessing
├── setup.py            # Compilación
├── benchmark.py        # Tests de rendimiento
├── Makefile            # Build automatizado
└── README.md           # Esta documentación
```

## Notas de Compilación

### macOS ARM (M1/M2):
- No usa `-march=native` por incompatibilidad
- Optimizaciones: `-O3 -ffast-math`

### Linux x86_64:
- Usa `-march=native` para optimizar CPU específica
- Optimizaciones: `-O3 -ffast-math -funroll-loops`

## Troubleshooting

### Error: "GCC not found"
```bash
# macOS
brew install gcc
# Linux
apt-get install gcc
```

### Error: "numpy/arrayobject.h not found"
```bash
pip install numpy --upgrade
```

### Error al importar
```python
import cp
print(f"C disponible: {cp.C_AVAILABLE}")
print(f"Versión: {cp.get_version()}")
```

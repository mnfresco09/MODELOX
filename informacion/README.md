## MODELOX

Backtester + optimizador (Optuna) para estrategias de trading, con arquitectura modular.

### Requisitos

- **Python >= 3.11**.  
  En macOS, `python3` del sistema suele ser 3.9 y no sirve para estas dependencias.
- Dependencias Python: ver `requirements.txt`.
  - `numba` es **obligatorio** (performance).

### Cómo ejecutar

1) Crea/activa un venv e instala dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ejecuta:

```bash
python ejecutar.py
```

La configuración vive en `general/configuracion.py` (dataset, fechas, N_TRIALS, comisiones, etc).

Para ejecutar solo una combinación, usa el selector numérico en `general/configuracion.py`:
- `COMBINACION_A_EJECUTAR = 0`  (todas)
- `COMBINACION_A_EJECUTAR = 1`  (DPO+ROC)
- `COMBINACION_A_EJECUTAR = 2`  (EMA_CRUCE+SLTP)

### Comprobar dependencias (qué falta)

Este repo ya no incluye scripts auxiliares. La instalación se hace con `pip` y la ejecución con `python ejecutar.py`.

### Dónde está cada cosa (arquitectura)

- **`modelox/core/`**: motor de backtest + métricas + runner de Optuna
  - `engine.py`: genera trades base y simula ejecución (apalancamiento/comisiones/stake)
  - `metrics.py`: métricas (Sharpe/Sortino por retornos, Calmar anualizado, etc.)
  - `runner.py`: loop de optimización (desacoplado de estrategias y reporting)
- **`modelox/strategies/`**: estrategias como plugins (auto-descubiertas)
- **Indicadores**: se calculan dentro de cada estrategia (`modelox/strategies/*.py`)
- **`modelox/reporting/`**: reporters (Rich/Excel/Plotly) como observadores

El repo se limpió para que la “fuente de verdad” sea `modelox/`. Solo se mantiene
`general/configuracion.py` como configuración editable.

### Cómo añadir una estrategia nueva (plug & play)

1) Crea un archivo en `modelox/strategies/`, por ejemplo `modelox/strategies/mi_estrategia.py`
2) Define una clase con:
   - `name: str`
   - `suggest_params(trial) -> dict`
   - `generate_signals(df, params) -> df` (debe añadir `signal_long`/`signal_short`)
   - `decide_exit(...) -> ExitDecision | None`
3) Ejecuta `python ejecutar.py`. El runner la detecta automáticamente.

### Salida de resultados

- Panel en consola: `RichReporter`
- Excel: `resultados/excel/*.xlsx`
- Plots: `resultados/plots/*.html`



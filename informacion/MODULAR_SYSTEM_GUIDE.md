"""
================================================================================
MODELOX - Sistema Ultra Modular de Indicadores y Plots (v7.0)
================================================================================

ARQUITECTURA REVOLUCIONARIA
==========================

Antes (v6.0):
- Hardcoding de indicadores en plot.py
- Cada nuevo indicador requería cambios en múltiples archivos
- Rangos y estilos diseminados por el código
- Imposible de mantener y escalar

Ahora (v7.0):
- Definición CENTRALIZADA de indicadores en indicators_metadata.py
- plot.py COMPLETAMENTE genérico
- Agregar indicador = agregar metadata
- Todo automático


WORKFLOW PARA AGREGAR UN NUEVO INDICADOR
=========================================

PASO 1: Implementar en logic/indicators.py
-----------------------------------------
Ejemplo RSI:

```python
@njit(cache=True, fastmath=True)
def _rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    # Implementación según TradingView
    ...

def rsi(
    df: pl.DataFrame,
    *,
    period: int = 14,
    col: str = "close",
    out: str = "rsi",
) -> pl.DataFrame:
    """Calcular RSI"""
    src = np.ascontiguousarray(df[col].to_numpy().astype(np.float64))
    rsi_vals = _rsi_numba(src, int(period))
    return df.with_columns(pl.Series(out, rsi_vals))
```

PASO 2: Agregar spec en modelox/strategies/indicator_specs.py
-------------------------------------------------------------
Esto describe cómo invocar el indicador:

```python
def cfg_rsi(*, period: int, out: str = "rsi", col: str = "close") -> Dict[str, Any]:
    return {"activo": True, "period": int(period), "col": col, "out": out}
```

PASO 3: Registrar metadata en modelox/indicators_metadata.py
------------------------------------------------------------
Esto define CÓMO se visualiza:

```python
IndicatorMetadata(
    name="rsi",
    display_name="RSI",
    indicator_type="oscillator",  # 'overlay' o 'oscillator'
    color="#60a5fa",              # Color Tailwind
    range_info=IndicatorRange(
        min_value=0,
        max_value=100,
        neutral=50,
        overbought=70,
        oversold=30,
    ),
)
```

PASO 4: ¡LISTO! Plot.py lo detecta automáticamente
--------------------------------------------------

El sistema:
1. Lee el DataFrame del trial
2. Detecta columnas de indicadores
3. Busca metadata en IndicatorRegistry
4. Crea paneles automáticamente con estilos correctos
5. Agregar líneas de overbought/oversold
6. Mostrar parámetros en títulos


ESTRUCTURA DE METADATOS
=======================

class IndicatorMetadata:
    name: str                      # Identificador único (ej: "rsi")
    display_name: str              # Nombre mostrado (ej: "RSI (14)")
    indicator_type: str            # 'overlay' o 'oscillator'
    color: str                     # Hex color Tailwind (ej: "#60a5fa")
    line_style: str                # 'solid', 'dashed', 'dotted'
    line_width: int                # Ancho de línea (default 2)
    range_info: IndicatorRange     # Rango y niveles
    params: Dict[str, Any]         # Parámetros específicos del trial
    column_name: str               # Nombre de columna en DataFrame
    additional_lines: Dict         # Para MACD, Stochastic, etc.


IndicatorRange:
    min_value: float               # Mínimo (ej: 0)
    max_value: float               # Máximo (ej: 100)
    neutral: float                 # Nivel neutral (ej: 50)
    overbought: float              # Nivel overbought (ej: 70)
    oversold: float                # Nivel oversold (ej: 30)
    upper_band: float              # Banda superior opcional
    lower_band: float              # Banda inferior opcional


EJEMPLOS PRÁCTICOS
==================

1. INDICADOR SIMPLE (RSI)
-------------------------
Solo un valor, rango fijo, línea única

IndicatorMetadata(
    name="rsi",
    display_name="RSI",
    indicator_type="oscillator",
    color="#60a5fa",
    range_info=IndicatorRange(
        min_value=0,
        max_value=100,
        neutral=50,
        overbought=70,
        oversold=30,
    ),
)

# Plot automáticamente:
# - Panel "RSI" con línea azul
# - Línea roja en 70 (overbought)
# - Línea verde en 30 (oversold)
# - Línea gris en 50 (neutral)


2. INDICADOR MULTI-LÍNEA (MACD)
-------------------------------
MACD tiene: línea principal, signal, histogram

IndicatorMetadata(
    name="macd",
    display_name="MACD",
    indicator_type="oscillator",
    color="#60a5fa",
    additional_lines={
        "macd_signal": {"color": "#fbbf24", "line_style": "dashed"},
        "macd_hist": {"color": "#60a5fa", "line_style": "solid", "plot_type": "histogram"},
    },
)

# Plot automáticamente:
# - Panel "MACD"
# - Línea azul para MACD
# - Línea amarilla punteada para Signal
# - Histograma azul para MACD_HIST


3. INDICADOR OVERLAY (EMA)
--------------------------
Dibujado en el panel de precios

IndicatorMetadata(
    name="ema",
    display_name="EMA",
    indicator_type="overlay",
    color="#fbbf24",
    range_info=IndicatorRange(min_value=0, max_value=float('inf')),
)

# Plot automáticamente:
# - Línea amarilla en panel de precios
# - Sin líneas de referencia (precio fluido)


CÓMO FUNCIONA plot_modular.py
==============================

1. ModularPlotBuilder.__init__()
   - Detecta indicadores en DataFrame
   - Extrae parámetros de trial_params
   - Determina warmup period

2. add_indicator(indicator_name, column_name)
   - Busca metadata en IndicatorRegistry
   - Determina panel (overlay vs oscillator)
   - Agrega a self.panels

3. build_panel_data(panel_key)
   - Reúne todas las líneas del panel
   - Prepara rango_info, colores, estilos
   - Retorna JSON-serializable

4. export_html(filepath)
   - Genera HTML con Plotly
   - Incrustra parámetros del trial
   - Muestra título con score y parámetros


CARACTERÍSTICAS AVANZADAS
==========================

1. DYNAMIC PANEL NAMING
   El nombre del panel incluye parámetros:
   
   RSI con period=14 → "RSI (period=14)"
   RSI con period=9 → "RSI (period=9)"
   
   Esto permite ver exactamente qué parámetro se usó en cada trial


2. WARMUP DETECTION
   Automáticamente detecta el máximo período entre todos los indicadores
   y descarta barras de warmup del gráfico
   
   Esto evita que veas líneas incompletas al principio


3. TIMESTAMP ALIGNMENT (SATA)
   Single Authoritative Timestamp Array
   
   - Garantiza que todos los datos están alineados
   - Usa búsqueda O(1) con hash
   - Elimina desincronización


4. PARAMETER BINDING
   Los parámetros del trial se leen automáticamente
   y se muestran en:
   - Título del gráfico
   - Nombres de paneles
   - Niveles de referencia (si aplica)


COLORES TAILWIND PARA USAR
===========================

Rojos/Naranjas:
  "#ef4444" - Red
  "#f97316" - Orange
  "#fb923c" - Orange-400
  "#fbbf24" - Amber

Verdes:
  "#22c55e" - Green
  "#10b981" - Emerald
  "#34d399" - Emerald-400

Azules:
  "#60a5fa" - Blue-400
  "#3b82f6" - Blue
  "#06b6d4" - Cyan
  "#38bdf8" - Sky

Púrpuras/Rosas:
  "#a78bfa" - Violet
  "#a855f7" - Violet-600
  "#f472b6" - Pink
  "#ec4899" - Pink-500

Grises/Neutral:
  "#94a3b8" - Slate
  "#6b7280" - Gray


CHECKLIST PARA AGREGAR INDICADOR
================================

[ ] ¿Implementé en logic/indicators.py?
[ ] ¿Seguí exactamente la fórmula de TradingView?
[ ] ¿Agregué cfg_INDICADOR en indicator_specs.py?
[ ] ¿Registré IndicatorMetadata en indicators_metadata.py?
[ ] ¿Elegí color Tailwind diferente para destacar?
[ ] ¿Configuré IndicatorRange correctamente?
[ ] ¿Probé que aparezca automáticamente en el plot?
[ ] ¿El nombre en display_name es legible?
[ ] ¿Si es multi-línea, configuré additional_lines?


TESTING
=======

Para probar tu indicador nuevo:

```python
from logic.indicators import IndicadorFactory
from modelox.strategies.indicator_specs import cfg_rsi
import polars as pl

# Cargar datos
df = pl.read_csv("data/ohlcv/BTC_ohlcv_5m.csv")

# Calcular indicador
cfg = {"rsi": cfg_rsi(period=14)}
df_with_rsi = IndicadorFactory.procesar(df, cfg)

# Verificar columna
print(df_with_rsi.select("rsi"))

# Graficar
from visual.plot_modular import plot_trial
params = {"rsi_period": 14}
filepath = plot_trial(
    df_with_rsi,
    trial_id=1,
    strategy_name="Test",
    trial_params=params,
    score=0.95,
)
print(f"Chart saved to: {filepath}")
```


DEBUGGING
=========

¿No aparece mi indicador?

1. ¿Está la columna en el DataFrame?
   ```
   print(df.columns)
   ```

2. ¿Está registrado en IndicatorRegistry?
   ```
   from modelox.indicators_metadata import IndicatorRegistry
   print(IndicatorRegistry.get_all())
   ```

3. ¿Coincide el nombre?
   ```
   # Columna: "rsi_14"
   # Metadata name: "rsi" ← debe match en patrón regex
   ```

4. ¿Está bien configurado adicional_lines para multi-línea?
   ```
   additional_lines={
       "macd_signal": {...},
   }
   ```


MIGRANDO INDICADORES EXISTENTES
===============================

Si tienes indicadores hardcodeados en plot.py:

1. Verifica que esté en indicators_metadata.py
2. Si NO, agrega IndicatorMetadata
3. Borra el código hardcodeado de plot.py
4. Prueba que funcione automático


CONCLUSIÓN
==========

Este sistema logra:
✓ Cero duplicación de código
✓ Máxima flexibilidad
✓ Escalabilidad infinita
✓ Mantenibilidad perfecta
✓ Plot.py NUNCA necesita cambios
✓ Un solo lugar para actualizar: indicators_metadata.py

Agregar 10 indicadores nuevos = cambiar 10 líneas en metadatos
No vuelves a tocar plot.py nunca más.

"""

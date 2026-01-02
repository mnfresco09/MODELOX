# 🏗️ MODELOX - Análisis de Arquitectura y Documentación Completa

> **Sistema de Backtesting Algorítmico con Optimización Optuna**  
> Análisis completo del flujo de datos, conexiones entre componentes y oportunidades de mejora.

---

## 📋 Índice

1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Flujo de Ejecución Completo](#flujo-de-ejecución-completo)
4. [Componentes Principales](#componentes-principales)
5. [Sistema de Salidas (Exits)](#sistema-de-salidas-exits)
6. [Sistema Multi-Timeframe](#sistema-multi-timeframe)
7. [Problemas Encontrados](#problemas-encontrados)
8. [Mejoras Implementadas](#mejoras-implementadas)
9. [Mejoras Recomendadas](#mejoras-recomendadas)
10. [Guía de Uso](#guía-de-uso)

---

## 🎯 Visión General

### ¿Qué es MODELOX?

MODELOX es un **framework de backtesting algorítmico** optimizado para Mac que:

- ✅ Ejecuta estrategias de trading sobre datos históricos OHLCV
- ✅ Optimiza parámetros usando **Optuna** (algoritmo TPE)
- ✅ Soporta **múltiples timeframes** (entrada ≠ salida)
- ✅ Implementa **salidas configurables** (SL/TP fijo o trailing stop)
- ✅ Genera reportes en **Excel, HTML y consola Rich**
- ✅ Usa **Polars** para máximo rendimiento
- ✅ Incluye **health monitoring** para estabilidad en Mac

---

## 🏛️ Arquitectura del Sistema

### Diagrama de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                      EJECUTAR.PY (Entry Point)              │
│  • Carga configuración                                      │
│  • Inicializa health monitoring                             │
│  • Itera sobre activos y estrategias                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (data.py)                  │
│  • load_data() → Carga Parquet/Feather                      │
│  • Normaliza timestamps a UTC (microsegundos)               │
│  • filter_by_date() → Filtra por rango                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              STRATEGY DISCOVERY (registry.py)               │
│  • Auto-descubre estrategias en modelox/strategies/        │
│  • Valida: name, combinacion_id, métodos requeridos        │
│  • Instancia estrategias seleccionadas                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              OPTIMIZATION RUNNER (runner.py)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  OPTUNA LOOP (N_TRIALS)                              │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ 1. Strategy.suggest_params(trial)              │  │   │
│  │  │    → Genera parámetros candidatos              │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 2. Strategy.generate_signals(df, params)       │  │   │
│  │  │    → Calcula indicadores y señales             │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 3. align_signals_to_base() (si multi-TF)       │  │   │
│  │  │    → Alinea señales al timeframe base          │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 4. generate_trades(df, params, strategy)       │  │   │
│  │  │    → Ejecuta lógica de entrada/salida          │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 5. simulate_trades(trades, config)             │  │   │
│  │  │    → Simula ejecución financiera               │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 6. resumen_metricas(trades, equity_curve)      │  │   │
│  │  │    → Calcula ROI, Sharpe, Drawdown, etc.       │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 7. score_optuna(metrics)                       │  │   │
│  │  │    → Calcula score objetivo (minimizar)        │  │   │
│  │  ├────────────────────────────────────────────────┤  │   │
│  │  │ 8. Reporters.on_trial_end(artifacts)           │  │   │
│  │  │    → Rich Console, Excel, Plots                │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 REPORTING SYSTEM (reporting/)               │
│  • RichReporter → Consola Bloomberg-style                   │
│  • ExcelReporter → Resumen + trades individuales            │
│  • PlotReporter → HTML interactivo (Plotly)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo de Ejecución Completo

### 1. **Inicialización (ejecutar.py)**

```python
# 1.1 Configuración de límites de recursos (Mac-optimized)
os.environ["OMP_NUM_THREADS"] = "1"  # Evita sobre-threads
os.environ["MKL_NUM_THREADS"] = "1"

# 1.2 Carga de configuración
from general.configuracion import CONFIG, ACTIVOS, N_TRIALS

# 1.3 Health monitoring
HealthGuard.check_system_health(ram_threshold=80.0)

# 1.4 Descubrimiento de estrategias
strategies = instantiate_strategies(only_id=COMBINACION_A_EJECUTAR)
```

### 2. **Carga de Datos (data.py)**

```python
# 2.1 Carga lazy con Polars
q = pl.scan_parquet(path)

# 2.2 Normalización temporal
df = q.with_columns([
    pl.col("timestamp")
      .dt.cast_time_unit("us")      # Microsegundos
      .dt.replace_time_zone("UTC")  # UTC explícito
])

# 2.3 Filtrado por fechas
df_filtrado = filter_by_date(df, FECHA_INICIO, FECHA_FIN)
```

### 3. **Loop de Optimización (runner.py)**

#### 3.1 **Sugerencia de Parámetros**
```python
params = strategy.suggest_params(trial)
# Ejemplo: {"rsi_period": 14, "threshold": 0.5}
```

#### 3.2 **Generación de Señales**
```python
df_signals = strategy.generate_signals(df, params)
# Añade columnas: indicadores + signal_long + signal_short
```

#### 3.3 **Alineación Multi-Timeframe** (si aplica)
```python
if entry_tf != base_tf:
    df_signals = align_signals_to_base(df_base, df_signals)
    # join_asof sin lookahead: backward
```

#### 3.4 **Generación de Trades (engine.py)**
```python
trades_base = generate_trades(df_signals, params, strategy=strategy)
# Para cada señal:
#   1. Calcula ATR en vela de entrada
#   2. Calcula SL/TP fijos (o inicializa trailing)
#   3. Escanea velas hasta salida (SL/TP/TIME_EXIT)
#   4. Guarda trade: entry_time, exit_time, prices, tipo_salida
```

#### 3.5 **Simulación Financiera (engine.py)**
```python
trades_exec, equity_curve = simulate_trades(trades_base, config)
# Para cada trade:
#   1. Calcula quantity (apalancamiento, límites)
#   2. Calcula PnL bruto y neto (comisiones)
#   3. Actualiza saldo
#   4. Early exit si saldo < saldo_minimo_operativo
```

#### 3.6 **Cálculo de Métricas (metrics.py)**
```python
metricas = resumen_metricas(trades_exec, equity_curve)
# Calcula: ROI, Sharpe, Sortino, Drawdown, SQN, etc.
```

#### 3.7 **Scoring (scoring.py)**
```python
score = score_optuna(metricas)
# Score multiplicativo: sharpe × sqn × profit_factor × ...
# Penalización fuerte si trades_por_dia < 0.25
```

#### 3.8 **Reporting**
```python
artifacts = TrialArtifacts(
    strategy_name, trial_number, params, score, 
    metrics, df_signals, trades_exec, equity_curve
)

for reporter in reporters:
    reporter.on_trial_end(artifacts)
```

---

## 🧩 Componentes Principales

### **1. ejecutar.py** (Entry Point)
**Responsabilidad:** Orquestar toda la ejecución

**Funciones clave:**
- `main()`: Loop principal sobre activos → estrategias → exit_types
- `HealthGuard`: Monitoreo de RAM y CPU para estabilidad en Mac
- Gestión de caché de timeframes (evita recargas)

**Flujo:**
```
ACTIVOS × ESTRATEGIAS × EXIT_TYPES → OptimizationRunner
```

---

### **2. configuracion.py** (Settings)
**Responsabilidad:** Centralizar toda la configuración

**Variables críticas:**
```python
# Activos y datos
ACTIVO = "GOLD, BTC"
TIMEFRAME = 60  # minutos

# Optimización
N_TRIALS = 150
COMBINACION_A_EJECUTAR = [3]  # IDs de estrategias

# Salidas
EXIT_TYPE = "all"  # "atr_fixed", "trailing", "all"
EXIT_SL_ATR = 1.0
EXIT_TP_ATR = 1.0
EXIT_TRAILING_ATR_MULT = 2.0

# Cuenta
SALDO_INICIAL = 300
APALANCAMIENTO = 50
COMISION_PCT = 0.00043
```

---

### **3. runner.py** (Optimization Orchestrator)
**Responsabilidad:** Gestionar el loop de Optuna y coordinar componentes

**Clase:** `OptimizationRunner`

**Métodos clave:**
- `optimize_strategies()`: Itera sobre lista de estrategias
- `_optimize_one()`: Ejecuta Optuna study para UNA estrategia
- `objetivo()`: Función objetivo de Optuna (1 trial = 1 ejecución)

**Responsabilidades del objetivo():**
1. Sugerir parámetros (strategy + exits + qty)
2. Generar señales
3. Alinear timeframes (si aplica)
4. Ejecutar backtest (trades + simulate)
5. Calcular métricas y score
6. Reportar resultados

---

### **4. engine.py** (Backtest Core)
**Responsabilidad:** Lógica pura de trading

**Funciones:**

#### `generate_trades(df, params, strategy)`
**Entrada:** DataFrame con señales + parámetros
**Salida:** DataFrame de trades base (sin simulación financiera)

**Proceso:**
1. Extrae arrays numpy (close, high, low, signals)
2. Calcula ATR de Wilder
3. Itera sobre señales (long/short)
4. Para cada señal:
   - Calcula SL/TP según `exit_settings`
   - Escanea velas hasta salida
   - Registra trade

**Features:**
- ✅ Intra-bar execution (usa open/high/low)
- ✅ SL tiene prioridad sobre TP (conservador)
- ✅ Gap handling (SL ejecutado en open si gap)
- ✅ Block velas after exit (configurable)

#### `simulate_trades(trades_base, config)`
**Entrada:** Trades base + configuración financiera
**Salida:** Trades ejecutados + curva de equity

**Proceso:**
1. Inicializa saldo
2. Para cada trade:
   - Calcula quantity (stake, apalancamiento, límites)
   - Calcula PnL bruto
   - Aplica comisiones → PnL neto
   - Actualiza saldo
   - **Early exit si saldo < mínimo**
3. Registra equity curve

**Métricas calculadas por trade:**
- `pnl`: PnL bruto
- `pnl_neto`: PnL neto (con comisiones)
- `comision`: Comisiones pagadas
- `saldo_antes/despues`: Estado de cuenta
- `quantity`: Tamaño de posición

---

### **5. exits.py** (Exit Logic - CRÍTICO)
**Responsabilidad:** Centralizar TODA la lógica de salidas

**Clases y Funciones:**

#### `ExitSettings` (dataclass)
```python
@dataclass(frozen=True)
class ExitSettings:
    exit_type: str = "atr_fixed"
    atr_period: int = 14
    sl_atr: float = 1.0
    tp_atr: float = 1.0
    time_stop_bars: int = 260
    trailing_atr_mult: float = 2.0
    emergency_sl_atr_mult: float = 4.0
```

#### `resolve_exit_settings_for_trial(trial, config)`
- Lee configuración
- Si `optimize_exits=True`: sugiere parámetros a Optuna
- Retorna `ExitSettings` para el trial

#### `decide_exit_for_trade(...) → ExitResult`
**Selector de lógica de salida:**
1. ¿Estrategia tiene `decide_exit()`? → Usa método custom
2. Si no, según `exit_type`:
   - `"atr_fixed"` → `decide_exit_atr_fixed_intrabar()`
   - `"trailing"` → `decide_exit_atr_trailing_with_emergency_sl()`

#### `decide_exit_atr_fixed_intrabar(...)`
**Lógica SL/TP Fijos:**
```python
# IMPORTANTE: Se calculan UNA SOLA VEZ
atr_entry = atr[entry_idx]
sl_dist = atr_entry * sl_atr
tp_dist = atr_entry * tp_atr

if side == "LONG":
    stop_loss = entry_price - sl_dist  # FIJO
    take_profit = entry_price + tp_dist  # FIJO

# Escaneo velas: estos valores NO cambian
for j in range(entry_idx+1, end_idx+1):
    if hit_sl_or_tp(j, stop_loss, take_profit):
        return ExitResult(...)
```

#### `decide_exit_atr_trailing_with_emergency_sl(...)`
**Lógica Trailing Stop:**
```python
# SL emergencia: fijo desde entrada
emergency_sl = entry_price ± emergency_sl_atr_mult * atr_entry

# Trailing stop: se actualiza cada vela
trailing_stop = entry_price ± trailing_atr_mult * atr_entry

for j in range(entry_idx+1, end_idx+1):
    # Actualizar trailing siguiendo precio favorable
    if side == "LONG":
        trailing_stop = max(trailing_stop, high[j] - trailing_atr_mult * atr[j])
    
    # Chequear salidas (prioridad: emergency > trailing)
    if hit_emergency_sl or hit_trailing:
        return ExitResult(...)
```

---

### **6. strategies/** (Strategy System)

#### **Estructura de una Estrategia**

```python
class MyStrategy:
    # Metadatos (requeridos)
    combinacion_id: int = 3  # ID único (> 0)
    name: str = "My_Strategy"
    
    # Parámetros Optuna
    parametros_optuna: Dict[str, Any] = {
        "rsi_period": (7, 21, 1),  # (min, max, step)
        "threshold": (0.5, 2.0, 0.1),
    }
    
    # Timeframes (opcional)
    timeframe_entry = None  # None = usa CONFIG.TIMEFRAME
    timeframe_exit = None
    
    def suggest_params(self, trial) -> Dict[str, Any]:
        """Define espacio de búsqueda de Optuna"""
        return {
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "threshold": trial.suggest_float("threshold", 0.5, 2.0),
        }
    
    def generate_signals(self, df: pl.DataFrame, params: Dict) -> pl.DataFrame:
        """Genera señales de trading"""
        # 1. Calcular indicadores
        df = df.with_columns([
            calculate_rsi(pl.col("close"), params["rsi_period"]).alias("rsi")
        ])
        
        # 2. Definir warmup (CRÍTICO)
        params["__warmup_bars"] = params["rsi_period"] + 10
        
        # 3. Generar señales
        df = df.with_columns([
            (pl.col("rsi") < 30).alias("signal_long"),
            (pl.col("rsi") > 70).alias("signal_short"),
        ])
        
        # 4. Metadata para gráficos
        params["__indicators_used"] = ["rsi"]
        
        return df
```

#### **registry.py** (Auto-Discovery)

**Función:** `discover_strategies()`

**Proceso:**
1. Escanea `modelox/strategies/*.py`
2. Busca clases con:
   - `name` (str, no vacío)
   - `combinacion_id` (int > 0)
   - Métodos: `suggest_params`, `generate_signals`
3. Valida IDs únicos
4. Retorna dict: `{name: Strategy_class}`

**Convenión:**
- `combinacion_id = 0` → EXCLUIDO (plantillas)
- `combinacion_id > 0` → INCLUIDO

---

### **7. reporting/** (Output System)

#### **RichReporter** (Consola Bloomberg-style)
```python
class ElegantRichReporter:
    def on_trial_end(self, artifacts: TrialArtifacts):
        # Muestra panel 3 columnas: Performance | Financials | Params
        mostrar_panel_elegante(metrics, params, score, ...)
    
    def on_strategy_end(self, strategy_name, study):
        # Muestra top 5 trials
        mostrar_top_trials(study, n=5)
```

#### **ExcelReporter** (Excel Workbooks)
- **resumen.xlsx**: Un libro por estrategia con sheet por trial
- **trades_TRIAL_X.xlsx**: Archivo individual por trial con todos los trades

#### **PlotReporter** (HTML Interactivo)
- Genera gráficos Plotly con:
  - Precio + indicadores
  - Marcadores de entrada/salida
  - Curva de equity
  - Drawdown
  - Profit/loss por trade

---

### **8. metrics.py** (Financial Metrics)

**Funciones principales:**

```python
def roi_pct(trades, saldo_inicial) -> float
    # ROI porcentual

def winrate_pct(trades) -> float
    # % de trades ganadores

def max_drawdown(equity_curve) -> Tuple[float, float]
    # Drawdown absoluto y porcentual

def sharpe_ratio(trades) -> float
    # Sharpe ratio (annualized)

def sortino_ratio(trades) -> float
    # Sortino ratio (downside deviation)

def sqn(trades) -> float
    # System Quality Number: sqrt(N) × (mean/std)

def profit_factor(trades) -> float
    # Suma(wins) / Suma(losses)

def expectativa(trades) -> float
    # Expectativa matemática: E[PnL] por trade
```

---

### **9. scoring.py** (Objective Function)

**Función:** `score_optuna(metrics) -> float`

**Lógica:**
1. Normaliza métricas a [0, 1]
2. **Penalización crítica:** Si `trades_por_dia < 0.25` → score ≤ 1
3. Score multiplicativo (favorece "todo bien a la vez"):
   ```python
   score = 3000 * sharpe_n * sqn_n * pf_n * roi_n * exp_n * dd_n * trades_n
   ```

**Umbrales de normalización:**
- Sharpe: 0.5 → 1.0 (excelente)
- SQN: 2.0 → 1.0
- ROI: 100% → 1.0
- Expectancia: $20/trade → 1.0
- Profit Factor: 2.0 → 1.0
- Drawdown: 0% → 1.0, 100% → 0.0

---

### **10. data.py** (Data Loading)

**Función:** `load_data(path) -> pl.DataFrame`

**Proceso:**
1. Lazy scan: `pl.scan_parquet()` o `pl.scan_ipc()`
2. Normalización:
   - Renombra columna temporal a `"timestamp"`
   - Cast a microsegundos (`us`)
   - Fuerza UTC (replace o convert)
3. Sort por timestamp
4. Collect (materializa en memoria)

**Formatos soportados:**
- ✅ Parquet
- ✅ Feather (Arrow IPC)

---

## 🚪 Sistema de Salidas (Exits)

### Arquitectura de Salidas

**Centralización:** TODO en `modelox/core/exits.py`

### Tipos de Salida

#### 1. **ATR Fixed (SL/TP Fijos)**
```
Entry: price=100, ATR=2.0, sl_atr=1.5, tp_atr=3.0

LONG:
  SL = 100 - (2.0 × 1.5) = 97.0  ← FIJO (no cambia)
  TP = 100 + (2.0 × 3.0) = 106.0 ← FIJO (no cambia)

SHORT:
  SL = 100 + (2.0 × 1.5) = 103.0 ← FIJO
  TP = 100 - (2.0 × 3.0) = 94.0  ← FIJO
```

**Parámetros optimizables:**
- `exit_sl_atr`: Multiplicador ATR para SL
- `exit_tp_atr`: Multiplicador ATR para TP
- `exit_atr_period`: Período del ATR
- `exit_time_stop_bars`: Máximo de velas antes de TIME_EXIT

#### 2. **Trailing Stop**
```
Entry: price=100, ATR=2.0, trailing=2.0, emergency=4.0

LONG:
  SL emergencia = 100 - (2.0 × 4.0) = 92.0  ← FIJO (protección)
  Trailing inicial = 100 - (2.0 × 2.0) = 96.0
  
  Vela 1: high=102, ATR=2.1
    Trailing = max(96.0, 102 - 2.1×2.0) = 97.8  ← ACTUALIZADO
  
  Vela 2: high=104, ATR=2.0
    Trailing = max(97.8, 104 - 2.0×2.0) = 100.0 ← ACTUALIZADO
  
  Salida si: low < trailing OR low < emergency_sl
```

**Parámetros optimizables:**
- `exit_trailing_atr_mult`: Distancia del trailing stop
- `exit_emergency_sl_atr_mult`: Distancia del SL emergencia
- `exit_atr_period`: Período del ATR
- `exit_time_stop_bars`: Máximo de velas

#### 3. **EXIT_TYPE = "all"**
Ejecuta ambos tipos secuencialmente:
1. 150 trials con `"atr_fixed"`
2. 150 trials con `"trailing"`

Resultados en carpetas separadas:
```
resultados/
  ├── CROSSOVER_HL_MA_ATR_FIXED/
  └── CROSSOVER_HL_MA_TRAILING/
```

---

## ⏱️ Sistema Multi-Timeframe

### Concepto

Permite que:
- **Señales** se generen en un timeframe (ej: 1h)
- **Backtest** se ejecute en otro timeframe (ej: 5m)

### Implementación

#### 1. **En la Estrategia**
```python
class MyStrategy:
    timeframe_entry = "1h"  # Generar señales en 1h
    timeframe_exit = "5m"   # Evaluar salidas en 5m (más precisión)
```

#### 2. **En el Runner**
```python
# Carga múltiples timeframes
df_5m = load_data("BTC_5m.parquet")
df_1h = load_data("BTC_1h.parquet")

df_by_timeframe = {
    "5m": df_5m,
    "1h": df_1h,
}

# Genera señales en 1h
df_signals_1h = strategy.generate_signals(df_1h, params)

# Alinea señales de 1h → 5m (sin lookahead)
df_signals_5m = align_signals_to_base(df_base=df_5m, df_signals=df_signals_1h)

# Backtest en 5m con señales de 1h
trades = generate_trades(df_signals_5m, params, strategy)
```

#### 3. **Alineación sin Lookahead**
```python
def align_signals_to_base(df_base, df_signals):
    # join_asof: backward (no lookahead)
    # Cada vela de 5m toma la señal más reciente de 1h
    return df_base.join_asof(
        df_signals.select(["timestamp", "signal_long", "signal_short"]),
        on="timestamp",
        strategy="backward"
    )
```

---

## ⚠️ Problemas Encontrados

### 1. **BacktestConfig sin exit_type** ✅ RESUELTO
**Problema:** Al implementar EXIT_TYPE="all", el sistema intentaba pasar `exit_type` a `BacktestConfig` pero el campo no existía.

**Solución:** Agregado `exit_type: str = "atr_fixed"` a la dataclass.

### 2. **Falta de Parámetros Trailing en Config** ✅ RESUELTO
**Problema:** `exit_trailing_atr_mult` y `exit_emergency_sl_atr_mult` no estaban en `BacktestConfig`.

**Solución:** Agregados ambos campos con rangos de optimización.

### 3. **Código Duplicado en runner.py** ⚠️ PENDIENTE
**Problema:** Al implementar EXIT_TYPE="all", se creó indentación compleja con try/except duplicados.

**Oportunidad de mejora:** Refactorizar usando un helper method.

### 4. **Cache de Timeframes Redundante** ⚠️ OBSERVACIÓN
**Problema:** Se carga el mismo timeframe múltiples veces si varias estrategias lo usan.

**Estado:** Hay un `tf_cache` pero se reinicia por estrategia, no globalmente.

### 5. **Health Guard Acoplado** ⚠️ OBSERVACIÓN
**Problema:** `HealthGuard` está hardcoded en `ejecutar.py`, difícil de desactivar o configurar.

**Mejora:** Mover a un módulo separado con flag de configuración.

### 6. **Reporting Condicional Complejo** ⚠️ OBSERVACIÓN
```python
# En runner.py líneas 217-227
plot_reporters = [r for r in self.reporters if r.__class__.__name__ == "PlotReporter"]
need_df_for_plot = any(getattr(r, "_should_generate_plot")(score) for r in plot_reporters)
```
**Problema:** Usa introspección de nombres de clase, frágil.

**Mejora:** Interfaz explícita: `reporter.needs_dataframe(score)`.

### 7. **Parámetros Runtime Desordenados** ⚠️ OBSERVACIÓN
**Problema:** Mezcla de `params`, `params_rt`, `params_reporting` con lógica de propagación compleja.

**Mejora:** Clase dedicada para gestionar parámetros de trial.

---

## ✨ Mejoras Implementadas

### 1. **Sistema de Salidas Dual con "all"** ✅
- Implementado EXIT_TYPE = "all"
- Ejecuta ambos tipos secuencialmente
- Carpetas de resultados separadas

### 2. **Banner Profesional de Dos Paneles** ✅
```
╭──────────── ═══ MODELOX ═══ ────────────╮
│       ASSET  ● GOLD                     │
│    STRATEGY  Crossover_HL_MA            │
│   TIMEFRAME  1h                         │
│      PERIOD  2021-01-11 → 2024-08-14    │
╰─────────────────────────────────────────╯

╭──────── Optimization Config ────────────╮
│   EXIT MODE  TP/SL Fijos (ATR)          │
│      TRIALS  150                        │
│      PARAMS  HL_PERIOD · MA_TYPE        │
╰─────────────────────────────────────────╯
```

### 3. **Documentación Explícita de SL/TP Fijos** ✅
Agregado en `exits.py`:
```python
"""
IMPORTANTE: SL/TP SON FIJOS
- Se calculan UNA SOLA VEZ con ATR de vela de entrada
- NO se modifican durante el trade
"""
```

### 4. **Strategy: Crossover HL/2 con MA** ✅
Nueva estrategia implementada:
- Calcula punto medio: (high + low) / 2
- Media móvil configurable: SMA, EMA, ALMA
- Señales por cruce de precio vs MA

---

## 🚀 Mejoras Recomendadas

### **Alta Prioridad**

#### 1. **Refactorizar EXIT_TYPE="all" Loop**
**Problema actual:**
```python
for current_exit_type in exit_types_to_run:
    # 100+ líneas de código duplicado
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["exit_type"] = current_exit_type
    cfg = BacktestConfig(**cfg_dict)
    # ... setup reporters, runner, etc.
```

**Mejora:**
```python
def _run_single_exit_type(self, exit_type, strategy, df_filtrado, ...):
    """Helper method para ejecutar un tipo de salida"""
    # Lógica centralizada
    pass

for current_exit_type in exit_types_to_run:
    _run_single_exit_type(current_exit_type, ...)
```

#### 2. **Clase TrialParameters**
**Problema:** Mezcla de `params`, `params_rt`, `params_reporting`

**Mejora:**
```python
@dataclass
class TrialParameters:
    strategy_params: Dict[str, Any]  # Parámetros de estrategia
    exit_params: Dict[str, Any]      # Parámetros de salida
    runtime_params: Dict[str, Any]   # Config runtime (__xxx)
    
    def to_reporting(self) -> Dict:
        """Filtra y formatea para reporting"""
        pass
```

#### 3. **Interface Reporter Explícita**
**Mejora:**
```python
class Reporter(Protocol):
    def needs_dataframe(self, score: float) -> bool:
        """¿Necesita df_signals convertido a Pandas?"""
        ...
    
    def on_trial_end(self, artifacts: TrialArtifacts) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...
```

#### 4. **Cache Global de Timeframes**
**Mejora:**
```python
class GlobalTimeframeCache:
    _cache: Dict[Tuple[str, str], pl.DataFrame] = {}
    
    @classmethod
    def get_or_load(cls, activo: str, timeframe: str) -> pl.DataFrame:
        key = (activo, timeframe)
        if key not in cls._cache:
            cls._cache[key] = load_data(resolve_archivo_data_tf(activo, timeframe))
        return cls._cache[key]
```

### **Media Prioridad**

#### 5. **Módulo HealthGuard Separado**
```python
# modelox/core/health.py
class HealthMonitor:
    def __init__(self, enabled: bool = True, ram_threshold: float = 80.0):
        self.enabled = enabled
        self.ram_threshold = ram_threshold
    
    def check(self):
        if not self.enabled:
            return
        # lógica actual
```

#### 6. **Validación de Estrategias**
Agregar en `registry.py`:
```python
def validate_strategy(cls) -> List[str]:
    """Retorna lista de errores de validación"""
    errors = []
    if not hasattr(cls, "name") or not cls.name:
        errors.append("Missing 'name' attribute")
    # ...más validaciones
    return errors
```

#### 7. **Logging Estructurado**
Reemplazar `print()` con logging:
```python
import logging
logger = logging.getLogger("modelox")

logger.info(f"Starting optimization for {strategy_name}")
logger.debug(f"Trial {trial_num} score: {score}")
```

### **Baja Prioridad**

#### 8. **Tests Unitarios**
Crear tests para:
- `exits.py`: Lógica de salidas
- `metrics.py`: Cálculo de métricas
- `scoring.py`: Función objetivo
- `data.py`: Normalización de timestamps

#### 9. **Type Hints Completos**
Agregar hints faltantes en:
- `engine.py`
- `runner.py`
- `metrics.py`

#### 10. **Documentación API**
Generar docs con Sphinx:
```bash
sphinx-quickstart docs
sphinx-apidoc -o docs/source modelox
```

---

## 📚 Guía de Uso

### **Instalación**

```bash
# 1. Clonar repositorio
git clone <repo>
cd MODELOX

# 2. Crear entorno virtual
python3.11 -m venv .venv311
source .venv311/bin/activate

# 3. Instalar dependencias
pip install -r x/requirements.txt
```

### **Configuración Básica**

Editar `general/configuracion.py`:

```python
# Activos a testear
ACTIVO = "GOLD, BTC"

# Timeframe base
TIMEFRAME = 60  # minutos (1h)

# Estrategias a ejecutar
COMBINACION_A_EJECUTAR = [3]  # IDs

# Trials de optimización
N_TRIALS = 150

# Tipo de salida
EXIT_TYPE = "atr_fixed"  # o "trailing" o "all"

# Parámetros de salida
EXIT_SL_ATR = 1.0
EXIT_TP_ATR = 2.0
EXIT_TIME_STOP_BARS = 260

# Cuenta
SALDO_INICIAL = 300
APALANCAMIENTO = 50
```

### **Ejecutar Optimización**

```bash
python ejecutar.py
```

### **Crear Nueva Estrategia**

1. **Crear archivo:** `modelox/strategies/my_strategy.py`

```python
from modelox.core.types import Strategy
import polars as pl

class MyStrategy:
    combinacion_id = 10  # ID único
    name = "My_Strategy"
    
    parametros_optuna = {
        "period": (10, 50, 5),
        "threshold": (0.5, 2.0, 0.1),
    }
    
    def suggest_params(self, trial):
        return {
            "period": trial.suggest_int("period", 10, 50, step=5),
            "threshold": trial.suggest_float("threshold", 0.5, 2.0),
        }
    
    def generate_signals(self, df: pl.DataFrame, params):
        # Calcular indicadores
        df = df.with_columns([
            # tu lógica aquí
        ])
        
        # Warmup
        params["__warmup_bars"] = params["period"] + 10
        
        # Señales
        df = df.with_columns([
            pl.lit(False).alias("signal_long"),   # tu condición
            pl.lit(False).alias("signal_short"),  # tu condición
        ])
        
        # Metadata
        params["__indicators_used"] = ["indicator_name"]
        
        return df
```

2. **Configurar ID:**
```python
# En configuracion.py
COMBINACION_A_EJECUTAR = [10]
```

3. **Ejecutar:**
```bash
python ejecutar.py
```

### **Resultados**

```
resultados/
  └── MY_STRATEGY/
      ├── excel/
      │   ├── resumen.xlsx
      │   └── trades_trial_*.xlsx
      └── graficos/
          └── GOLD/
              └── TRIAL-X_SCORE-Y_*.html
```

---

## 📊 Conclusiones

### **Fortalezas del Sistema**

✅ **Arquitectura clara y modular**
- Separación de responsabilidades bien definida
- Componentes independientes y reutilizables

✅ **Performance optimizado**
- Uso de Polars para máxima velocidad
- Early exit en simulación (ahorra CPU)
- Health monitoring para estabilidad

✅ **Flexibilidad**
- Multi-timeframe
- Salidas configurables
- Auto-discovery de estrategias
- Múltiples formatos de output

✅ **Robustez**
- Manejo de timestamps correcto
- Validación de datos
- Error handling apropiado

### **Áreas de Mejora**

⚠️ **Complejidad del runner.py**
- Refactorizar loop EXIT_TYPE="all"
- Extraer helpers para setup de reporters

⚠️ **Gestión de parámetros**
- Unificar params/params_rt/params_reporting
- Clase dedicada TrialParameters

⚠️ **Testing**
- Agregar tests unitarios
- CI/CD pipeline

⚠️ **Documentación**
- API docs con Sphinx
- Más ejemplos de estrategias

### **Resumen de Conexiones**

```
ejecutar.py
    ↓
configuracion.py (settings)
    ↓
data.py (load OHLCV)
    ↓
registry.py (discover strategies)
    ↓
runner.py (OptimizationRunner)
    ├→ strategy.suggest_params(trial)
    ├→ strategy.generate_signals(df, params)
    ├→ timeframes.align_signals_to_base() (si multi-TF)
    ├→ engine.generate_trades(df, params, strategy)
    │   └→ exits.decide_exit_for_trade()
    │       ├→ exits.decide_exit_atr_fixed_intrabar()
    │       └→ exits.decide_exit_atr_trailing_with_emergency_sl()
    ├→ engine.simulate_trades(trades, config)
    ├→ metrics.resumen_metricas(trades, equity)
    ├→ scoring.score_optuna(metrics)
    └→ reporters.on_trial_end(artifacts)
        ├→ RichReporter (consola)
        ├→ ExcelReporter (Excel)
        └→ PlotReporter (HTML)
```

---

## 🎓 Recursos Adicionales

### **Archivos Clave**
- 📄 `ARCHITECTURE.md` (este documento)
- 📄 `README.md` (guía de usuario)
- 📂 `modelox/strategies/ESTRATEGIA_BASE.py` (template)

### **Comandos Útiles**

```bash
# Ver estrategias disponibles
python -c "from modelox.strategies.registry import list_available_strategies; print(list_available_strategies())"

# Limpiar cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Ver métricas de un trial
# (Revisar resultados/STRATEGY/excel/resumen.xlsx)

# Debugging con timings
MODELOX_TIMINGS=1 python ejecutar.py
```

---

**Última actualización:** 2 de enero de 2026  
**Versión:** 1.0.0  
**Autor:** Sistema MODELOX

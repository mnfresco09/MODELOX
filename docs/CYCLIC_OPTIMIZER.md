# 🔄 CYCLIC COORDINATE DESCENT - Optimizador de Descenso de Coordenadas

## 📋 Concepto

El **Descenso de Coordenadas Cíclico** (Cyclic Coordinate Descent - CCD) es una técnica de optimización que:

1. **Optimiza UN parámetro a la vez** mientras mantiene los demás fijos
2. **Cicla** por todos los parámetros secuencialmente (estrategia + SL/TP/Trailing)
3. **Repite** ciclos hasta que los valores **converjan** (no cambien)
4. **Garantiza mínimo 3 ciclos** - NO usa N_TRIALS como criterio de parada

### ⚠️ IMPORTANTE: Este modo ignora N_TRIALS

A diferencia de otros samplers, CYCLIC:
- **NO para por número de trials**
- **Hace todas las vueltas necesarias** hasta convergencia
- **Garantiza mínimo 3 ciclos completos**
- El número total de trials = `(parámetros × trials_per_param × ciclos)`

### 🎸 Analogía: Afinando una Banda de Música

Imagina que estás afinando 7 instrumentos en una banda:

**Vuelta 1 (Ajuste Inicial):**
- Ajustas la **Batería (RSI)**. Suena bien sola.
- Ajustas la **Guitarra (EMA)** con la batería ya fijada.
- Ajustas el **Bajo (MACD)** con batería y guitarra fijas.
- Ajustas el **SL%** para proteger las posiciones.
- Ajustas el **TP%** para cerrar ganancias.
- ... continúas hasta el último instrumento.

**⚠️ El Problema:**
Al ajustar la Batería (RSI) al principio, la Guitarra (EMA) todavía estaba en un valor malo. El RSI que elegiste NO es óptimo para el contexto final.

**Vuelta 2 (Refinamiento):**
- Vuelves a la **Batería (RSI)**, pero ahora con TODOS los instrumentos ya afinados.
- Es muy probable que el RSI óptimo **cambie** ahora que el resto está afinado.

**Vuelta 3+ (Convergencia):**
- Repites ciclos hasta que, al dar una vuelta completa, ningún instrumento necesite reajuste.

---

## ⚙️ Configuración (configuracion.py)

```python
# Activar modo CYCLIC
OPTUNA_SAMPLER = "CYCLIC"

# ═══════════════════════════════════════════════════════════════════
# DOS MODOS DE OPERACIÓN
# ═══════════════════════════════════════════════════════════════════
CYCLIC_USE_N_TRIALS = False         # True = usa N_TRIALS, False = convergencia

# ───────────────────────────────────────────────────────────────────
# MODO N_TRIALS (CYCLIC_USE_N_TRIALS = True)
# ───────────────────────────────────────────────────────────────────
# - Usa TODOS los N_TRIALS configurados
# - NO para por convergencia, sigue hasta acabar trials
# - Útil cuando quieres explorar más a fondo
N_TRIALS = 20000  # Se usarán todos estos trials
CYCLIC_TRIALS_PER_PARAM_FIXED = None  # None = auto-calcular, o poner número fijo (ej: 100)

# ───────────────────────────────────────────────────────────────────
# MODO CONVERGENCIA (CYCLIC_USE_N_TRIALS = False) [DEFAULT]
# ───────────────────────────────────────────────────────────────────
# - Para cuando no hay variación significativa
# - Más eficiente, usa solo los trials necesarios

# Convergencia POR PARÁMETRO
CYCLIC_PARAM_MIN_TRIALS = 20        # Mínimo trials antes de evaluar convergencia
CYCLIC_PARAM_MAX_TRIALS = 200       # Máximo trials por parámetro (seguridad)
CYCLIC_PARAM_PATIENCE = 15          # Trials sin mejora = parámetro convergió
CYCLIC_PARAM_MIN_IMPROVEMENT = 0.001 # Mejora mínima para considerar progreso (0.1%)

# Convergencia ENTRE CICLOS
CYCLIC_CONVERGENCE_THRESHOLD = 0.02 # 2% - Si todos los params cambian menos, convergió

# ═══════════════════════════════════════════════════════════════════
# PARÁMETROS COMUNES A AMBOS MODOS
# ═══════════════════════════════════════════════════════════════════
CYCLIC_MAX_CYCLES = 15              # Máximo de ciclos (seguridad en ambos modos)
CYCLIC_MIN_CYCLES = 3               # MÍNIMO 3 vueltas garantizadas

# SAMPLER INTERNO
CYCLIC_PARAM_SAMPLER = "tpe"        # "tpe" (aprende) o "random" (exploración pura)

# PARÁMETROS DE SALIDA
CYCLIC_INCLUDE_EXITS = True         # Incluir SL/TP/Trailing en optimización cíclica

# VISUALIZACIÓN
CYCLIC_VERBOSE = True               # Mostrar progreso detallado
```

---

## 📊 Diferencia entre Modos

### MODO N_TRIALS (`CYCLIC_USE_N_TRIALS = True`)
```
┌─────────────────────────────────────────────────────────────────┐
│  Ciclo 1    │  Ciclo 2    │  Ciclo 3    │ ... │  Hasta acabar │
│  (P1→P2→..) │  (P1→P2→..) │  (P1→P2→..) │     │  N_TRIALS     │
└─────────────────────────────────────────────────────────────────┘
                                                       │
                           ¡NO PARA POR CONVERGENCIA! ─┘
```

### MODO CONVERGENCIA (`CYCLIC_USE_N_TRIALS = False`)
```
┌─────────────────────────────────────────────────────────────────┐
│  Ciclo 1    │  Ciclo 2    │  Ciclo 3    │  STOP!               │
│  (P1→P2→..) │  (P1→P2→..) │  (P1→P2→..) │  Convergió           │
└─────────────────────────────────────────────────────────────────┘
                                              │
                        Para cuando todos los │
                        params cambian < 2%   ┘
```

### Parámetros Optimizados Automáticamente

Cuando `CYCLIC_INCLUDE_EXITS = True` y `OPTIMIZAR_SALIDAS = True`:

| Tipo | Parámetros |
|------|------------|
| **Estrategia** | Todos los definidos en `suggest_params()` |
| **Stop Loss** | `exit_sl_pct` |
| **Take Profit** | `exit_tp_pct` (si exit_type incluye TP) |
| **Trailing** | `exit_trail_act_pct`, `exit_trail_dist_pct` (si exit_type incluye trailing) |

---

## 🔧 Arquitectura

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│           CyclicCoordinateOptimizer                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. _discover_parameters()                                  │
│     └── Inspecciona strategy.suggest_params()               │
│     └── Añade exit_sl_pct, exit_tp_pct, etc. si aplica     │
│     └── Captura distribuciones (Int, Float, Categorical)    │
│                                                             │
│  2. _initialize_best_params()                               │
│     └── Búsqueda exploratoria inicial (20% de trials)       │
│     └── Establece punto de partida                          │
│                                                             │
│  3. _run_cycle() x N (mínimo 3)                             │
│     │                                                       │
│     │  Para cada param P en [estrategia + exits]:           │
│     │  ┌─────────────────────────────────────────┐          │
│     │  │ _optimize_single_param(P)              │          │
│     │  │  ├── Fija todos los demás en best      │          │
│     │  │  ├── Usa PartialFixedSampler           │          │
│     │  │  ├── Optimiza P con N trials           │          │
│     │  │  └── Actualiza best_params[P] si mejora│          │
│     │  └─────────────────────────────────────────┘          │
│     │                                                       │
│     └── _check_convergence()                                │
│         └── ¿Todos los params cambiaron < threshold?        │
│         └── Solo verifica después de min_cycles (3)         │
│                                                             │
│  4. Resultado Final                                         │
│     └── best_params (estrategia + exits)                    │
│     └── best_score, trajectory                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### PartialFixedSampler

El corazón del algoritmo es el `PartialFixedSampler`:

```python
class PartialFixedSampler(BaseSampler):
    """
    Sampler que fija algunos parámetros y optimiza solo uno.
    
    - Parámetros fijos: Siempre devuelven el valor especificado
    - Parámetro libre: Se optimiza con el sampler interno (TPE/Random)
    """
```

**Funcionamiento:**
```python
fixed_params = {"rsi_period": 14, "ema_length": 200, "atr_mult": 2.5}
free_param = "zlema_fast_len"

sampler = PartialFixedSampler(
    fixed_params=fixed_params,
    free_param=free_param,
    internal_sampler=TPESampler(seed=42),
)

# Cuando Optuna llama a sample_independent():
# - Si param == "rsi_period": return 14 (FIJO)
# - Si param == "ema_length": return 200 (FIJO)  
# - Si param == "atr_mult": return 2.5 (FIJO)
# - Si param == "zlema_fast_len": return TPESampler.suggest() (OPTIMIZA)
```

---

## 📊 Output Esperado

Al ejecutar con `OPTUNA_SAMPLER = "CYCLIC"`:

```
======================================================================
   🔄 CYCLIC COORDINATE DESCENT OPTIMIZER
======================================================================

   📈 Estrategia: StrategyKineticMomentumValidator
   🔁 Max ciclos: 10
   🎯 Trials por parámetro: 100
   📊 Convergencia: 2.0%

   📋 Parámetros descubiertos: 4
      • zlema_fast_len: int[1, 400] step=1
      • zlema_slow_len: int[400, 2500] step=1
      • lookbar: int[0, 500] step=1
      • req_dist_pct: float[0.0, 3.0] step=0.01

   🎯 Inicializando con búsqueda exploratoria...
   ✅ Score inicial: 0.4523

--------------------------------------------------
   🔄 CICLO 1/10
--------------------------------------------------

      🔧 Optimizando: zlema_fast_len
         ✅ 200 → 156 (+22.00%)
         Score: 0.4523 → 0.5012

      🔧 Optimizando: zlema_slow_len
         ✅ 1200 → 1450 (+20.83%)
         Score: 0.5012 → 0.5234

      🔧 Optimizando: lookbar
         ➖ 250 → 250 (sin cambio)
         Score: 0.5234 → 0.5234

      🔧 Optimizando: req_dist_pct
         ✅ 1.50 → 0.85 (-43.33%)
         Score: 0.5234 → 0.5567

   📊 Resumen Ciclo 1:
      • Parámetros mejorados: 3/4
      • Score: 0.4523 → 0.5567
      • Tiempo: 45.2s
      • Trials: 400

--------------------------------------------------
   🔄 CICLO 2/10
--------------------------------------------------

      🔧 Optimizando: zlema_fast_len
         ✅ 156 → 142 (+8.97%)
         Score: 0.5567 → 0.5701

      🔧 Optimizando: zlema_slow_len
         ➖ 1450 → 1445 (-0.34%)
         Score: 0.5701 → 0.5703

      🔧 Optimizando: lookbar
         ➖ 250 → 248 (-0.80%)
         Score: 0.5703 → 0.5705

      🔧 Optimizando: req_dist_pct
         ➖ 0.85 → 0.84 (-1.18%)
         Score: 0.5705 → 0.5708

   📊 Resumen Ciclo 2:
      • Parámetros mejorados: 1/4
      • Score: 0.5567 → 0.5708
      • Tiempo: 44.8s
      • Trials: 400

--------------------------------------------------
   🔄 CICLO 3/10
--------------------------------------------------

      🔧 Optimizando: zlema_fast_len
         ➖ 142 → 143 (+0.70%)
         Score: 0.5708 → 0.5710

      🔧 Optimizando: zlema_slow_len
         ➖ 1445 → 1446 (+0.07%)
         Score: 0.5710 → 0.5711

      🔧 Optimizando: lookbar
         ➖ 248 → 248 (sin cambio)
         Score: 0.5711 → 0.5711

      🔧 Optimizando: req_dist_pct
         ➖ 0.84 → 0.84 (sin cambio)
         Score: 0.5711 → 0.5711

==================================================
   ✅ ¡CONVERGENCIA ALCANZADA EN CICLO 3!
==================================================

======================================================================
   📊 RESUMEN FINAL - CYCLIC COORDINATE DESCENT
======================================================================

   🏆 Mejor Score: 0.5711
   🔁 Ciclos completados: 3
   ✅ Convergió: Sí
   📍 Ciclo de convergencia: 3
   🎯 Total trials: 1220
   ⏱️  Tiempo total: 156.3s

   📋 Mejores parámetros:
      • zlema_fast_len: 143
      • zlema_slow_len: 1446
      • lookbar: 248
      • req_dist_pct: 0.84

   📈 Evolución del Score:
      Ciclo 1: 0.5567 █████████████████████████████
      Ciclo 2: 0.5708 ██████████████████████████████
      Ciclo 3: 0.5711 ██████████████████████████████
```

---

## 🔄 Comparación con otros Samplers

| Aspecto | CYCLIC | PLATEAU | CMA-ES | TPE |
|---------|--------|---------|--------|-----|
| **Enfoque** | 1 param a la vez | Clustering + Refinamiento | Evolutivo | Bayesiano |
| **Interacciones** | ✅ Captura interacciones | ✅ Detecta mesetas | ⚠️ Correlaciones | ⚠️ Limitado |
| **Interpretabilidad** | ✅ Alta | ⚠️ Media | ❌ Baja | ❌ Baja |
| **Convergencia** | ✅ Garantizada | ⚠️ Depende de clusters | ⚠️ Puede estancarse | ⚠️ Exploración infinita |
| **Trials necesarios** | `params × trials_per_param × cycles` | N_TRIALS fijo | N_TRIALS fijo | N_TRIALS fijo |
| **Ideal para** | Estrategias con interacciones complejas | Detectar zonas robustas | Espacios continuos | Exploración general |

---

## 🧪 Casos de Uso

### ✅ Usar CYCLIC cuando:
- Tienes parámetros que **interactúan** entre sí (RSI + EMA)
- Quieres **entender** cómo afecta cada parámetro
- Buscas **convergencia garantizada**
- Tienes tiempo para múltiples ciclos

### ❌ NO usar CYCLIC cuando:
- Tienes **muchos parámetros** (>10) - cada ciclo es costoso
- Los parámetros son **independientes** - no hay ganancia en ciclar
- Necesitas **exploración pura** - mejor usar PLATEAU o Random

---

## 🔗 Integración

El optimizador está completamente integrado con:
- ✅ Sistema de reporters (Rich, CSV, Excel, Plots)
- ✅ Configuración de perturbación (anti-overfitting)
- ✅ Multi-timeframe (MTF)
- ✅ Sistema de salidas (SL/TP/Trailing)
- ✅ Scoring institucional

Solo cambia `OPTUNA_SAMPLER = "CYCLIC"` en `configuracion.py` y ejecuta normalmente:

```bash
python ejecutar.py
```

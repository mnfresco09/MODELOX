# 🏔️ TOPÓGRAFO DE MESETAS - Sistema de Optimización Robusta

## Transformación: De "Buscador de Picos" a "Topógrafo de Mesetas"

Este documento describe la transformación del sistema de optimización de MODELOX
de un enfoque tradicional basado en picos (overfitting) a un enfoque robusto
basado en mesetas (robustez).

---

## 📊 Arquitectura del Sistema

### Problema Original
- **TPE/CMA-ES tradicional** busca el "pico" más alto
- Los picos suelen ser **overfitting**: parámetros muy específicos que funcionan
  solo con datos históricos exactos
- Pequeños cambios en parámetros causan grandes caídas en rendimiento

### Solución: Mesetas
- Una **meseta** es una región donde MÚLTIPLES combinaciones de parámetros
  producen buenos resultados consistentes
- Las mesetas indican **robustez**: el sistema funciona bien incluso si los
  parámetros varían un poco
- El **centroide** de una meseta es el parámetro más robusto

---

## 🔄 Las 3 Fases de Optimización

### FASE 1: Exploración Masiva (40% de trials)
```
┌─────────────────────────────────────────────────────┐
│  RandomSampler dispersa puntos por todo el espacio  │
│  Genera "materia prima" para el clustering          │
│  Evita concentración prematura en un solo punto     │
└─────────────────────────────────────────────────────┘
```

**¿Por qué RandomSampler?**
- TPE y CMA-ES son "greedy" (avariciosos): van directo al primer pico
- RandomSampler llena el espacio uniformemente
- Permite al clustering "ver" todas las zonas prometedoras

### FASE 2: Detección de Mesetas (HDBSCAN)
```
┌─────────────────────────────────────────────────────┐
│  HDBSCAN agrupa puntos cercanos con buen score      │
│  Detecta clusters de DENSIDAD VARIABLE              │
│  Soft clustering: Probabilidades de pertenencia     │
│  Clusters densos = Mesetas = Robustez               │
│  Puntos aislados = Ruido = Overfitting              │
└─────────────────────────────────────────────────────┘
```

**Ventajas de HDBSCAN sobre DBSCAN:**
- No necesita `eps` fijo (radio de vecindad)
- Detecta clusters de **densidad variable** automáticamente
- Proporciona **probabilidades de pertenencia** (soft clustering)
- Los puntos de borde con baja probabilidad se descartan

**Parámetros HDBSCAN:**
- `min_cluster_size`: Tamaño mínimo de cluster (15 por defecto)
- `min_samples`: Puntos para determinar densidad (8 por defecto)
- `min_membership_probability`: Umbral de pertenencia (0.5 por defecto)
- `score_percentile`: Solo analizar top X% de trials (85% por defecto)

### FASE 3: Refinamiento CMA-ES
```
┌─────────────────────────────────────────────────────┐
│  Para cada meseta, lanza CMA-ES local               │
│  Restricción: Solo dentro de los límites de meseta  │
│  Afina la solución en la "zona segura"              │
└─────────────────────────────────────────────────────┘
```

**¿Por qué CMA-ES ahora?**
- CMA-ES es excelente para refinamiento local
- Restringido a la meseta, no puede escapar a un pico de overfitting
- Encuentra el óptimo dentro de la región robusta

---

## ⚙️ Configuración (configuracion.py)

```python
# Seleccionar modo PLATEAU
OPTUNA_SAMPLER = "PLATEAU"  # "PLATEAU", "CMA" o "TPE"

# FASE 1: Exploración
PLATEAU_EXPLORATION_RATIO = 0.40  # 40% de trials
PLATEAU_EXPLORATION_SAMPLER = "random"  # "random" o "tpe"

# FASE 2: HDBSCAN (Clustering jerárquico)
PLATEAU_MIN_CLUSTER_SIZE = 15  # Tamaño mínimo de cluster
PLATEAU_MIN_SAMPLES = 8  # Puntos para determinar densidad
PLATEAU_MIN_MEMBERSHIP_PROB = 0.5  # Umbral probabilidad pertenencia
PLATEAU_SCORE_PERCENTILE = 85.0  # Solo top 15%
PLATEAU_MIN_TRIALS_FOR_MESETA = 10  # Mínimo para meseta válida

# FASE 3: Refinamiento
PLATEAU_MAX_MESETAS = 0  # 0 = refinar TODAS las mesetas encontradas
PLATEAU_MIN_TRIALS_POR_MESETA = 50  # Mínimo trials por meseta

# Selección del representante
PLATEAU_CENTROID_SELECTION = "centroid"  # "centroid", "best", "median", "highest_prob"
```

---

## 🎯 Comparación: Pico vs Meseta

| Característica | Pico (Tradicional) | Meseta (Nuevo) |
|----------------|-------------------|----------------|
| Encontrar óptimo | Rápido | Más lento pero robusto |
| Sensibilidad a cambios | ALTA | BAJA |
| Riesgo de overfitting | ALTO | BAJO |
| Validación forward | Suele fallar | Mejor comportamiento |
| Reproducibilidad | Baja | Alta |

---

## 📁 Archivos Nuevos

### `modelox/core/topology.py`
- `PlateauConfig`: Configuración de DBSCAN
- `PlateauResult`: Resultado de una meseta
- `TopologyAnalysis`: Análisis completo
- `analyze_topology()`: Función principal de detección

### `modelox/core/plateau_optimizer.py`
- `PlateauOptimizerConfig`: Configuración del sistema
- `PlateauOptimizer`: Clase principal del optimizador
- `run_plateau_optimization()`: Función de conveniencia

---

## 🔧 Uso

### Modo PLATEAU (Recomendado)
```python
# En configuracion.py
OPTUNA_SAMPLER = "PLATEAU"
N_TRIALS = 5000  # Más trials = mejor detección de mesetas

# Ejecutar
python ejecutar.py
```

### Modo Clásico (CMA-ES o TPE)
```python
# En configuracion.py
OPTUNA_SAMPLER = "CMA"  # o "TPE"

# Ejecutar (sin cambios)
python ejecutar.py
```

---

## 📈 Output Esperado

### Durante la ejecución:
```
╔═══════════════════════════════════════════════════╗
║      MODELOX OPTIMIZATION PLATEAU                 ║
╠═══════════════════════════════════════════════════╣
║  ASSET         ₿ BTC                              ║
║  TRIALS        5000                               ║
║  PERTURBATION  ON                                 ║
╚═══════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════╗
║  FASE 1: EXPLORACIÓN MASIVA                        ║
║  Trials: 2000 | Sampler: RANDOM                    ║
╚════════════════════════════════════════════════════╝
[■■■■■■■■■■■■■■■■■■■■] 100% 2000/2000

╔════════════════════════════════════════════════════╗
║  FASE 2: DETECCIÓN DE MESETAS                      ║
╚════════════════════════════════════════════════════╝

📊 ANÁLISIS TOPOLÓGICO DE MESETAS
   Trials analizados: 300
   Mesetas encontradas: 3
   Puntos de ruido: 45

🏔️ Mesetas Detectadas
┌─────────┬────────┬──────────────┬───────────┐
│ Cluster │ Trials │ Score Medio  │ Densidad  │
├─────────┼────────┼──────────────┼───────────┤
│    0    │   85   │   342.15     │   12.5    │
│    1    │   62   │   298.43     │   8.7     │
│    2    │   58   │   276.21     │   7.2     │
└─────────┴────────┴──────────────┴───────────┘

╔════════════════════════════════════════════════════╗
║  FASE 3.1: REFINAMIENTO CMA-ES (Meseta 0)          ║
║  Score medio meseta: 342.15 | Trials: 200          ║
╚════════════════════════════════════════════════════╝
[■■■■■■■■■■■■■■■■■■■■] 100% 200/200

╔════════════════════════════════════════════════════╗
║  📊 RESULTADO FINAL                                ║
╠════════════════════════════════════════════════════╣
║  Score Exploración: 298.50                         ║
║  Score Refinado:    367.82 ✓                       ║
║  Mejora:            +69.32                         ║
║  Meseta Ganadora:   Cluster 0                      ║
╚════════════════════════════════════════════════════╝
```

---

## ❓ FAQ

### ¿El scoring.py sigue funcionando igual?
**Sí**, el sistema de scoring institucional no cambia. La diferencia es CÓMO
se explora el espacio de parámetros, no cómo se evalúan los resultados.

### ¿Cuántos trials necesito?
- Mínimo recomendado: 2000
- Óptimo: 5000-10000
- Más trials = mejor detección de mesetas

### ¿Qué pasa si no encuentra mesetas?
El sistema usa el mejor resultado de la fase de exploración como fallback.
Considera:
- Aumentar `PLATEAU_EXPLORATION_RATIO`
- Reducir `PLATEAU_SCORE_PERCENTILE`
- Reducir `PLATEAU_DBSCAN_MIN_SAMPLES`

### ¿Puedo volver al modo clásico?
**Sí**, simplemente cambia `OPTUNA_SAMPLER = "CMA"` o `"TPE"`.

---

## 🎓 Referencias Técnicas

- **DBSCAN**: Ester, M., et al. (1996). "A Density-Based Algorithm for
  Discovering Clusters in Large Spatial Databases with Noise"
- **CMA-ES**: Hansen, N. (2006). "The CMA Evolution Strategy: A Comparing Review"
- **Optuna**: Akiba, T., et al. (2019). "Optuna: A Next-generation
  Hyperparameter Optimization Framework"

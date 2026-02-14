# 📚 GUÍA COMPLETA: CÓMO AÑADIR NUEVOS OPTIMIZADORES A MODELOX

```
═══════════════════════════════════════════════════════════════════════════════
    ██████╗ ██╗   ██╗██╗ █████╗     ██████╗ ███████╗    
   ██╔════╝ ██║   ██║██║██╔══██╗    ██╔══██╗██╔════╝    
   ██║  ███╗██║   ██║██║███████║    ██║  ██║█████╗      
   ██║   ██║██║   ██║██║██╔══██║    ██║  ██║██╔══╝      
   ╚██████╔╝╚██████╔╝██║██║  ██║    ██████╔╝███████╗    
    ╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝    
                                                         
   ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗
  ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══════╝
  ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║   ███╔╝
  ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║  ███╔╝ 
  ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗
   ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝
═══════════════════════════════════════════════════════════════════════════════
```

## 📋 ÍNDICE

1. [Arquitectura del Sistema](#1-arquitectura-del-sistema)
2. [Anatomía de un Optimizador](#2-anatomía-de-un-optimizador)
3. [Sistema de Scoring](#3-sistema-de-scoring)
4. [Paso a Paso: Crear Nuevo Optimizador](#4-paso-a-paso-crear-nuevo-optimizador)
5. [Archivos a Modificar](#5-archivos-a-modificar)
6. [Métricas Disponibles](#6-métricas-disponibles)
7. [Ejemplo Completo: Optimizador Random Search](#7-ejemplo-completo-optimizador-random-search)
8. [Checklist Final](#8-checklist-final)

---

## 1. ARQUITECTURA DEL SISTEMA

### 1.1 Estructura de Archivos

```
modelox/
├── core/                           ← NÚCLEO (NO TOCAR PARA OPTIMIZADORES)
│   ├── engine.py                   ← Motor de backtest Numba
│   ├── metrics.py                  ← Cálculo de métricas (IMPORTANTE)
│   ├── types.py                    ← Tipos y dataclasses
│   ├── exits.py                    ← Configuración de salidas
│   ├── data.py                     ← Carga de datos
│   └── runner.py                   ← SignalGenerator, BacktestEngine
│
└── optimizers/                     ← AQUÍ VAN LOS OPTIMIZADORES
    ├── __init__.py                 ← MODIFICAR: Exportaciones
    ├── cma.py                      ← Referencia: CMA-ES
    ├── tpe.py                      ← Referencia: TPE
    └── tu_optimizador.py           ← CREAR: Tu nuevo optimizador
```

### 1.2 Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FLUJO COMPLETO DE OPTIMIZACIÓN                                              │
│                                                                              │
│  1. ejecutar.py                                                              │
│     └── Carga configuración (general/configuracion.py)                       │
│         └── Define OPTUNA_SAMPLER = "CMA" | "TPE" | "TU_OPTIMIZADOR"        │
│                                                                              │
│  2. modelox/core/runner.py                                                   │
│     └── OptimizationRunner                                                   │
│         └── Usa factory create_study() desde optimizers/__init__.py         │
│                                                                              │
│  3. modelox/optimizers/__init__.py                                           │
│     └── create_study() → Decide qué optimizador usar                         │
│                                                                              │
│  4. modelox/optimizers/tu_optimizador.py                                     │
│     └── TuOptimizer.optimize()                                               │
│         └── Para cada trial:                                                 │
│             a) Sugiere parámetros via trial.suggest_*()                     │
│             b) Ejecuta backtest → obtiene trades                             │
│             c) Calcula métricas via resumen_metricas()                       │
│             d) Calcula score via TuScorer.compute_score()                    │
│             e) Retorna score a Optuna                                        │
│                                                                              │
│  5. Optuna                                                                   │
│     └── Usa el score para guiar la búsqueda (según el sampler)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ANATOMÍA DE UN OPTIMIZADOR

### 2.1 Componentes Necesarios

Cada optimizador tiene **4 componentes principales**:

```python
# 1. CONFIGURACIÓN DEL SCORING
@dataclass
class TuScoringConfig:
    """Parámetros que controlan cómo se calcula el score"""
    SCORE_MIN: float = 1.0
    SCORE_MAX: float = 1000.0
    # ... tus parámetros de scoring

# 2. CONFIGURACIÓN DEL OPTIMIZADOR
@dataclass
class TuOptimizerConfig:
    """Parámetros del sampler de Optuna"""
    SEED: Optional[int] = None
    N_JOBS: int = 1
    # ... parámetros específicos del sampler

# 3. CLASE SCORER
class TuScorer:
    """Calcula el score a partir de las métricas"""
    def compute_score(self, trial, metrics, ...) -> float:
        # Lógica de scoring
        return score

# 4. CLASE OPTIMIZER
class TuOptimizer:
    """Orquesta la optimización completa"""
    def optimize(self, df, strategy, ...) -> optuna.Study:
        # Crea estudio, ejecuta trials, retorna resultados
        return study
```

### 2.2 ¿Qué Hace Cada Componente?

| Componente | Responsabilidad |
|------------|-----------------|
| `ScoringConfig` | Define constantes para el scoring (pesos, umbrales, floors) |
| `OptimizerConfig` | Define configuración de Optuna (seed, n_startup_trials, etc.) |
| `Scorer` | Transforma métricas de backtest → score numérico [1, 1000] |
| `Optimizer` | Crea estudio Optuna, ejecuta trials, gestiona parámetros |

---

## 3. SISTEMA DE SCORING

### 3.1 ¿Qué es el Scoring?

El **scoring** es la función que convierte los resultados de un backtest en un **número único** que Optuna usa para decidir si la configuración es "buena" o "mala".

```
┌────────────────────────────────────────────────────────────────────────────┐
│  BACKTEST RESULT                                                            │
│  ├── sharpe: 1.5                                                           │
│  ├── roi: 45.3%                                                            │
│  ├── drawdown: 12.5%                                                       │
│  ├── n_trades: 234                                                         │
│  └── sqn: 2.8                                                              │
│                                                                             │
│  ─────────────────────── SCORING FUNCTION ───────────────────────          │
│                              │                                              │
│                              ▼                                              │
│                         SCORE: 742.5                                        │
│                                                                             │
│  Este número es lo que Optuna usa para:                                    │
│  • CMA-ES: Adaptar la matriz de covarianza                                 │
│  • TPE: Modelar distribuciones buenas/malas                                │
│  • Tu Optimizador: Lo que tú decidas                                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Filosofías de Scoring

#### CMA-ES (Institucional/Riguroso)
```python
# Score = BaseScore(sharpe) × Penalizaciones
# Rango: [1, 1000]
# Filosofía: Castigar overfitting, premiar robustez

PILARES:
  1. PSR (Probabilistic Sharpe Ratio) - ¿El Sharpe es estadísticamente significativo?
  2. DSR (Deflated Sharpe Ratio) - Corrección por múltiples pruebas
  3. SAM (Stability) - ¿Es estable ante perturbaciones?
  4. Régimen - ¿Funciona en diferentes volatilidades?
  5. Curva - ¿La equity es lineal y consistente?
  6. Decay - Penaliza descubrimientos tardíos
```

#### TPE (Exploratorio/Simple)
```python
# Score = Σ(peso_i × métrica_normalizada_i)
# Rango: [1, 1000]
# Filosofía: Explorar ampliamente, menos penalizaciones

PESOS:
  - Sharpe: 35%
  - SQN: 20%
  - ROI: 20%
  - Drawdown: 15%
  - Trades: 10%
```

### 3.3 ¿Por Qué Importa el Scoring?

| Scoring Malo | Scoring Bueno |
|--------------|---------------|
| Muchos configs con score similar | Diferenciación clara entre configs |
| Overfitting a picos aislados | Premiar soluciones robustas |
| Optimizador no converge | Convergencia hacia buenas regiones |
| Resultados no reproducibles | Resultados consistentes |

---

## 4. PASO A PASO: CREAR NUEVO OPTIMIZADOR

### Paso 1: Crear el Archivo

```bash
# Crear archivo del optimizador
touch modelox/optimizers/mi_optimizador.py
```

### Paso 2: Estructura Básica del Archivo

```python
"""modelox/optimizers/mi_optimizador.py

═══════════════════════════════════════════════════════════════════════════════
    MI OPTIMIZADOR - DESCRIPCIÓN BREVE
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
Explicación de qué hace tu optimizador y cuándo usarlo.

VENTAJAS:
=========
  ✓ Ventaja 1
  ✓ Ventaja 2

FILOSOFÍA DEL SCORING:
======================
Explica cómo calcula el score y por qué.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TU_SAMPLER  # Importa el sampler que uses

# =============================================================================
# IMPORTS INTERNOS (SIEMPRE ESTOS)
# =============================================================================
from modelox.core.engine import BacktestParams, calculate_performance_vectorized_numba
from modelox.core.metrics import resumen_metricas
from modelox.core.types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from modelox.core.exits import resolve_exit_settings_for_trial

# Silenciar warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

### Paso 3: Definir Configuración del Scoring

```python
# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING
# =============================================================================

@dataclass
class MiScoringConfig:
    """
    Configuración del sistema de scoring de MiOptimizador.
    
    PERSONALIZA ESTOS VALORES SEGÚN TU FILOSOFÍA DE SCORING.
    """
    
    # =========================================================================
    # 1.1 RANGO DE SALIDA
    # =========================================================================
    SCORE_MIN: float = 1.0               # NUNCA retornar 0
    SCORE_MAX: float = 1000.0            # Máximo absoluto
    
    # =========================================================================
    # 1.2 PESOS DE MÉTRICAS (DEBEN SUMAR 1.0)
    # =========================================================================
    PESO_SHARPE: float = 0.40            # Peso del Sharpe Ratio
    PESO_SQN: float = 0.20               # Peso del SQN
    PESO_ROI: float = 0.15               # Peso del ROI
    PESO_DRAWDOWN: float = 0.15          # Peso (penalización) del DD
    PESO_TRADES: float = 0.10            # Peso de actividad
    
    # =========================================================================
    # 1.3 UMBRALES DE PENALIZACIÓN
    # =========================================================================
    MIN_TRADES: int = 10                 # Trades mínimos para score válido
    MAX_DRAWDOWN: float = 50.0           # DD máximo aceptable
    MIN_SHARPE: float = -2.0             # Sharpe mínimo (debajo = penalización)
    
    # =========================================================================
    # 1.4 TUS PARÁMETROS PERSONALIZADOS
    # =========================================================================
    MI_PARAMETRO_1: float = 1.0
    MI_PARAMETRO_2: bool = True


# Instancia por defecto
MI_SCORING_CONFIG = MiScoringConfig()
```

### Paso 4: Crear la Clase Scorer

```python
# =============================================================================
# [SECCIÓN 2] CLASE SCORER
# =============================================================================

class MiScorer:
    """
    Scorer personalizado para MiOptimizador.
    
    RESPONSABILIDAD: Convertir métricas → score numérico [1, 1000]
    """
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[MiScoringConfig] = None,
    ):
        self.study = study
        self.config = config or MI_SCORING_CONFIG
    
    # =========================================================================
    # [2.1] FUNCIONES AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """Extrae valor numérico de forma segura (COPIA ESTE MÉTODO)."""
        try:
            val = metrics.get(key, default)
            if val is None:
                return default
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return default
            return f_val
        except Exception:
            return default
    
    # =========================================================================
    # [2.2] NORMALIZACIÓN DE MÉTRICAS
    # =========================================================================
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normaliza Sharpe a [0, 1]."""
        # Ejemplo: Sigmoide simple
        try:
            return 1.0 / (1.0 + math.exp(-1.5 * (sharpe - 1.0)))
        except:
            return 0.5
    
    def _normalize_sqn(self, sqn: float) -> float:
        """Normaliza SQN a [0, 1]."""
        # SQN > 4 = excelente según Van Tharp
        return min(1.0, max(0.0, sqn / 4.0))
    
    def _normalize_roi(self, roi: float) -> float:
        """Normaliza ROI a [0, 1]."""
        if roi <= 0:
            return max(0.0, 0.5 + roi / 200.0)
        return min(1.0, 0.5 + 0.5 * math.log1p(roi) / math.log1p(100))
    
    def _penalize_drawdown(self, drawdown: float) -> float:
        """Penalización por drawdown [0, 1] (1 = sin DD, 0 = DD catastrófico)."""
        cfg = self.config
        if drawdown <= cfg.MAX_DRAWDOWN:
            return 1.0 - 0.5 * (drawdown / cfg.MAX_DRAWDOWN)
        return max(0.1, 0.5 * (1.0 - (drawdown - cfg.MAX_DRAWDOWN) / 50.0))
    
    # =========================================================================
    # [2.3] FUNCIÓN PRINCIPAL: compute_score
    # =========================================================================
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
    ) -> float:
        """
        FUNCIÓN PRINCIPAL DE SCORING.
        
        Args:
            trial: Objeto optuna.Trial (puede ser None para scoring standalone)
            metrics: Diccionario con métricas del backtest
            returns: Array de retornos por trade (opcional, para PSR)
            equity_curve: Curva de equity (opcional, para K-ratio)
        
        Returns:
            Score en rango [SCORE_MIN, SCORE_MAX]
        """
        cfg = self.config
        
        # =================================================================
        # EXTRAER MÉTRICAS (USA MÚLTIPLES KEYS POR COMPATIBILIDAD)
        # =================================================================
        sharpe = self._safe_get(metrics, "sharpe", 0.0)
        if sharpe == 0:
            sharpe = self._safe_get(metrics, "sharpe_ratio", 0.0)
        
        sqn = self._safe_get(metrics, "sqn", 0.0)
        roi = self._safe_get(metrics, "roi", 0.0)
        
        drawdown = self._safe_get(metrics, "drawdown", 50.0)
        if drawdown == 0:
            drawdown = self._safe_get(metrics, "max_drawdown", 50.0)
        
        n_trades = int(self._safe_get(metrics, "n_trades", 0))
        if n_trades == 0:
            n_trades = int(self._safe_get(metrics, "total_trades", 0))
        
        # =================================================================
        # VERIFICACIÓN DE TRADES MÍNIMOS
        # =================================================================
        if n_trades < cfg.MIN_TRADES:
            # Guardar razón en trial para debug
            if trial is not None:
                try:
                    trial.set_user_attr('score_reason', 'insufficient_trades')
                except:
                    pass
            return cfg.SCORE_MIN
        
        # =================================================================
        # NORMALIZAR MÉTRICAS
        # =================================================================
        norm_sharpe = self._normalize_sharpe(sharpe)
        norm_sqn = self._normalize_sqn(sqn)
        norm_roi = self._normalize_roi(roi)
        norm_dd = self._penalize_drawdown(drawdown)
        norm_trades = min(1.0, math.log1p(n_trades) / math.log1p(50))
        
        # =================================================================
        # CALCULAR SCORE PONDERADO
        # =================================================================
        weighted_sum = (
            cfg.PESO_SHARPE * norm_sharpe +
            cfg.PESO_SQN * norm_sqn +
            cfg.PESO_ROI * norm_roi +
            cfg.PESO_DRAWDOWN * norm_dd +
            cfg.PESO_TRADES * norm_trades
        )
        
        # =================================================================
        # ESCALAR A RANGO FINAL
        # =================================================================
        score_range = cfg.SCORE_MAX - cfg.SCORE_MIN
        final_score = cfg.SCORE_MIN + score_range * weighted_sum
        
        # =================================================================
        # GUARDAR ATRIBUTOS PARA AUDITORÍA (OPCIONAL PERO RECOMENDADO)
        # =================================================================
        if trial is not None:
            try:
                trial.set_user_attr('norm_sharpe', float(norm_sharpe))
                trial.set_user_attr('norm_sqn', float(norm_sqn))
                trial.set_user_attr('norm_roi', float(norm_roi))
                trial.set_user_attr('norm_dd', float(norm_dd))
                trial.set_user_attr('weighted_sum', float(weighted_sum))
                trial.set_user_attr('sr_nominal', float(sharpe))
            except:
                pass
        
        # Garantizar rango
        return float(max(cfg.SCORE_MIN, min(cfg.SCORE_MAX, final_score)))
```

### Paso 5: Definir Configuración del Optimizador

```python
# =============================================================================
# [SECCIÓN 3] CONFIGURACIÓN DEL OPTIMIZADOR
# =============================================================================

@dataclass
class MiOptimizerConfig:
    """
    Configuración del optimizador (parámetros de Optuna).
    """
    
    # =========================================================================
    # 3.1 CONFIGURACIÓN GENERAL OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # None = aleatorio cada vez
    N_JOBS: int = 1                       # Workers paralelos (1 recomendado)
    STORAGE: Optional[str] = None         # None = RAM, "sqlite:///..." = persistir
    STUDY_NAME_PREFIX: str = "MODELOX"    # Prefijo para nombres de estudio
    
    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA DE TU SAMPLER
    # =========================================================================
    N_STARTUP_TRIALS: int = 10            # Trials aleatorios antes de usar sampler
    # Añade aquí parámetros específicos de tu sampler


# Instancia por defecto
MI_OPTIMIZER_CONFIG = MiOptimizerConfig()
```

### Paso 6: Crear la Clase Optimizer

```python
# =============================================================================
# [SECCIÓN 4] CLASE OPTIMIZER
# =============================================================================

class MiOptimizer:
    """
    Optimizador personalizado para MODELOX.
    
    USO:
        optimizer = MiOptimizer(config, n_trials=500)
        study = optimizer.optimize(df=df, strategy=strategy)
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[MiOptimizerConfig] = None,
        scoring_config: Optional[MiScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        Args:
            config: Configuración de backtest (saldo, comisiones, etc.)
            n_trials: Número de trials a ejecutar
            reporters: Lista de reporters para resultados
            optimizer_config: Configuración del optimizador
            scoring_config: Configuración del scoring
            activo: Nombre del activo (BTC, GOLD, etc.)
        """
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or MI_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or MI_SCORING_CONFIG
        self.activo = activo
        
        # Estado interno
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[MiScorer] = None
    
    # =========================================================================
    # [4.1] CREAR ESTUDIO OPTUNA
    # =========================================================================
    
    def _create_study(self, strategy_name: str) -> optuna.Study:
        """Crea un estudio Optuna con tu sampler."""
        cfg = self.optimizer_config
        
        # Construir nombre del estudio
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy_name), "MI_OPT"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        # ═══════════════════════════════════════════════════════════════════
        # AQUÍ CONFIGURAS TU SAMPLER
        # ═══════════════════════════════════════════════════════════════════
        # Ejemplos de samplers disponibles en Optuna:
        #   - optuna.samplers.RandomSampler
        #   - optuna.samplers.TPESampler
        #   - optuna.samplers.CmaEsSampler
        #   - optuna.samplers.GridSampler
        #   - optuna.samplers.NSGAIISampler (multi-objetivo)
        #   - optuna.samplers.MOTPESampler (multi-objetivo)
        #   - optuna.samplers.QMCSampler (quasi-Monte Carlo)
        
        from optuna.samplers import RandomSampler  # EJEMPLO
        
        sampler = RandomSampler(
            seed=cfg.SEED,
        )
        
        # Crear estudio
        study = optuna.create_study(
            direction="maximize",  # Siempre maximize (score mayor = mejor)
            sampler=sampler,
            study_name=study_name,
            storage=cfg.STORAGE,
            load_if_exists=False,
        )
        
        # Inicializar scorer
        self._scorer = MiScorer(study=study, config=self.scoring_config)
        
        return study
    
    @staticmethod
    def _slug(s: str) -> str:
        """Genera slug válido para nombres de estudio."""
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    # =========================================================================
    # [4.2] PREPARAR PARÁMETROS
    # =========================================================================
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """
        Prepara parámetros para un trial.
        
        IMPORTANTE: Esta función debe inyectar todos los parámetros
        necesarios para el backtest.
        """
        # Obtener parámetros de la estrategia
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        # ═══════════════════════════════════════════════════════════════════
        # INYECTAR VALORES DE CONFIGURACIÓN (OBLIGATORIO)
        # ═══════════════════════════════════════════════════════════════════
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
        
        # QTY_MAX_ACTIVO
        if self.config.optimize_qty_max_activo:
            qty_min, qty_max, qty_step = self.config.qty_max_activo_range
            qty_optimized = trial.suggest_float(
                "qty_max_activo", qty_min, qty_max, step=qty_step
            )
            params_rt["__qty_max_activo"] = qty_optimized
            params_rt["qty_max_activo"] = qty_optimized
        else:
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(
            getattr(strategy, "SALIDAS_PERSONALIZADAS", False)
        )
        
        # RESOLVER CONFIGURACIÓN DE SALIDA
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # ALIASES PARA COMPATIBILIDAD
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # TIMEFRAMES
        entry_tf = normalize_timeframe_to_suffix(
            getattr(strategy, "timeframe_entry", None) or base_tf
        )
        exit_tf = normalize_timeframe_to_suffix(
            getattr(strategy, "timeframe_exit", None) or base_tf
        )
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    # =========================================================================
    # [4.3] FUNCIÓN OBJETIVO
    # =========================================================================
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        """Crea la función objetivo para el optimizador."""
        
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            t0_total = time.perf_counter()
            
            # Limpieza periódica de memoria
            periodic_cleanup(trial.number)
            
            # Preparar parámetros
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            try:
                # ═══════════════════════════════════════════════════════════
                # 1. GENERAR SEÑALES
                # ═══════════════════════════════════════════════════════════
                signal_gen = SignalGenerator(strategy)
                signals_df = signal_gen.generate(df_entry, params_rt)
                
                if signals_df is None or signals_df.is_empty():
                    return self.scoring_config.SCORE_MIN
                
                # ═══════════════════════════════════════════════════════════
                # 2. EJECUTAR BACKTEST
                # ═══════════════════════════════════════════════════════════
                bt_engine = BacktestEngine(self.config)
                trades_df, equity_curve = bt_engine.run(signals_df, params_rt)
                
                if trades_df is None or trades_df.is_empty():
                    return self.scoring_config.SCORE_MIN
                
                # ═══════════════════════════════════════════════════════════
                # 3. CALCULAR MÉTRICAS
                # ═══════════════════════════════════════════════════════════
                metrics = resumen_metricas(
                    trades=trades_df,
                    saldo_inicial=self.config.saldo_inicial,
                    equity_curve=equity_curve,
                )
                
                # ═══════════════════════════════════════════════════════════
                # 4. CALCULAR SCORE
                # ═══════════════════════════════════════════════════════════
                score = self._scorer.compute_score(
                    trial=trial,
                    metrics=metrics,
                    equity_curve=equity_curve,
                )
                
                return score
                
            except Exception as e:
                # Log del error si lo necesitas
                return self.scoring_config.SCORE_MIN
        
        return objective
    
    # =========================================================================
    # [4.4] MÉTODO PRINCIPAL: optimize
    # =========================================================================
    
    def optimize(
        self,
        df: pl.DataFrame,
        strategy: Strategy,
        df_map: Optional[Dict[str, pl.DataFrame]] = None,
        base_tf: str = "5m",
    ) -> optuna.Study:
        """
        Ejecuta la optimización.
        
        Args:
            df: DataFrame con datos OHLCV
            strategy: Estrategia a optimizar
            df_map: Diccionario de DataFrames por timeframe
            base_tf: Timeframe base
        
        Returns:
            optuna.Study con los resultados
        """
        # Preparar df_map si no se proporciona
        if df_map is None:
            df_map = {base_tf: df}
        
        # Crear estudio
        study = self._create_study(strategy.name)
        
        # Crear objetivo
        objective = self._create_objective(df, df_map, strategy, base_tf)
        
        # ═══════════════════════════════════════════════════════════════════
        # EJECUTAR OPTIMIZACIÓN
        # ═══════════════════════════════════════════════════════════════════
        study.optimize(
            objective,
            n_trials=int(self.n_trials),
            n_jobs=int(self.optimizer_config.N_JOBS),
            gc_after_trial=True,
            catch=(Exception,),
        )
        
        self._last_study = study
        return study
    
    # =========================================================================
    # [4.5] PROPIEDADES
    # =========================================================================
    
    @property
    def last_study(self) -> Optional[optuna.Study]:
        """Retorna el último estudio ejecutado."""
        return self._last_study
    
    @property
    def scorer(self) -> Optional[MiScorer]:
        """Retorna el scorer utilizado."""
        return self._scorer
```

### Paso 7: Añadir Funciones de Utilidad

```python
# =============================================================================
# [SECCIÓN 5] FUNCIONES DE UTILIDAD
# =============================================================================

def create_mi_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Crea un estudio Optuna con tu sampler.
    
    Esta función es usada por el factory en __init__.py
    """
    from optuna.samplers import RandomSampler  # Tu sampler
    
    parts = [study_name_prefix, str(strategy_name), "mi_opt"]
    if activo:
        parts.append(str(activo))
    study_name = re.sub(r'[^a-z0-9]+', '_', "_".join(parts).lower())[:50]
    
    sampler = RandomSampler(seed=seed)
    
    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
    )


def score_mi_optimizador(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
) -> float:
    """
    Función de scoring standalone.
    
    USO:
        score = score_mi_optimizador(metrics)
    """
    scorer = MiScorer()
    return scorer.compute_score(trial=trial, metrics=metrics)


# =============================================================================
# [SECCIÓN 6] EXPORTACIONES
# =============================================================================

__all__ = [
    "MiOptimizer",
    "MiOptimizerConfig",
    "MiScorer",
    "MiScoringConfig",
    "MI_SCORING_CONFIG",
    "MI_OPTIMIZER_CONFIG",
    "create_mi_study",
    "score_mi_optimizador",
]
```

---

## 5. ARCHIVOS A MODIFICAR

### 5.1 `modelox/optimizers/__init__.py`

Añade las importaciones y actualiza el factory:

```python
# =============================================================================
# IMPORTS MI OPTIMIZADOR (AÑADIR)
# =============================================================================
from .mi_optimizador import (
    MiOptimizer,
    MiOptimizerConfig,
    MiScorer,
    MiScoringConfig,
    MI_SCORING_CONFIG,
    MI_OPTIMIZER_CONFIG,
    create_mi_study,
    score_mi_optimizador,
)


# =============================================================================
# FACTORY: CREAR ESTUDIO SEGÚN SAMPLER (MODIFICAR)
# =============================================================================

def create_study(
    sampler: str,
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Factory que crea un estudio Optuna con el sampler indicado.
    """
    sampler_type = sampler.upper() if sampler else "CMA"
    
    if sampler_type == "TPE":
        return create_tpe_study(...)
    elif sampler_type == "MI_OPTIMIZADOR":  # ← AÑADIR ESTE ELIF
        return create_mi_study(
            strategy_name=strategy_name,
            activo=activo,
            seed=seed,
            study_name_prefix=study_name_prefix,
            storage=storage,
        )
    else:
        # Default: CMA-ES
        return create_cma_study(...)


# =============================================================================
# EXPORTACIONES (AÑADIR)
# =============================================================================
__all__ = [
    # ... existentes ...
    
    # MI OPTIMIZADOR
    "MiOptimizer",
    "MiOptimizerConfig",
    "MiScorer",
    "MiScoringConfig",
    "MI_SCORING_CONFIG",
    "MI_OPTIMIZER_CONFIG",
    "create_mi_study",
    "score_mi_optimizador",
]
```

### 5.2 `general/configuracion.py` (Opcional)

Si quieres que sea seleccionable por el usuario:

```python
# Sampler de Optuna: "CMA", "TPE", "MI_OPTIMIZADOR"
OPTUNA_SAMPLER = "MI_OPTIMIZADOR"
```

---

## 6. MÉTRICAS DISPONIBLES

### 6.1 Métricas del `resumen_metricas()`

Estas son las métricas que puedes usar en tu scorer:

| Métrica | Clave | Descripción |
|---------|-------|-------------|
| ROI | `roi` | Retorno sobre inversión (%) |
| Win Rate | `winrate` | Porcentaje de trades ganadores |
| Drawdown | `drawdown` | Máximo drawdown (%) |
| Expectativa | `expectativa` | Esperanza matemática por trade |
| SQN | `sqn` | System Quality Number (Van Tharp) |
| Sharpe | `sharpe` | Sharpe Ratio per-trade |
| Sortino | `sortino` | Sortino Ratio per-trade |
| Profit Factor | `profit_factor` | Ganancias / Pérdidas |
| Payoff Ratio | `payoff_ratio` | Media ganancias / Media pérdidas |
| Calmar | `calmar` | CAGR / Max Drawdown |
| N Trades | `n_trades` | Número total de trades |
| Trades/Día | `trades_por_dia` | Frecuencia de trading |

### 6.2 Cómo Extraer Métricas

```python
# SIEMPRE usa _safe_get con múltiples keys por compatibilidad
sharpe = self._safe_get(metrics, "sharpe", 0.0)
if sharpe == 0:
    sharpe = self._safe_get(metrics, "sharpe_ratio", 0.0)

n_trades = int(self._safe_get(metrics, "n_trades", 0))
if n_trades == 0:
    n_trades = int(self._safe_get(metrics, "total_trades", 0))
```

---

## 7. EJEMPLO COMPLETO: OPTIMIZADOR RANDOM SEARCH

Aquí un ejemplo simplificado de un optimizador Random Search:

```python
"""modelox/optimizers/random_search.py"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import optuna
import polars as pl
from optuna.samplers import RandomSampler

from modelox.core.metrics import resumen_metricas
from modelox.core.types import BacktestConfig, Strategy

@dataclass
class RandomScoringConfig:
    SCORE_MIN: float = 1.0
    SCORE_MAX: float = 1000.0
    PESO_SHARPE: float = 0.5
    PESO_ROI: float = 0.3
    PESO_DD: float = 0.2
    MIN_TRADES: int = 10

RANDOM_SCORING_CONFIG = RandomScoringConfig()


class RandomScorer:
    def __init__(self, config: Optional[RandomScoringConfig] = None):
        self.config = config or RANDOM_SCORING_CONFIG
    
    @staticmethod
    def _safe_get(m: Mapping, k: str, d: float = 0.0) -> float:
        try:
            v = m.get(k, d)
            return float(v) if v is not None and math.isfinite(float(v)) else d
        except:
            return d
    
    def compute_score(self, trial, metrics) -> float:
        cfg = self.config
        
        sharpe = self._safe_get(metrics, "sharpe", 0.0)
        roi = self._safe_get(metrics, "roi", 0.0)
        dd = self._safe_get(metrics, "drawdown", 50.0)
        n = int(self._safe_get(metrics, "n_trades", 0))
        
        if n < cfg.MIN_TRADES:
            return cfg.SCORE_MIN
        
        # Normalizar
        n_sharpe = 1 / (1 + math.exp(-1.5 * (sharpe - 1)))
        n_roi = min(1, max(0, 0.5 + roi / 200))
        n_dd = max(0, 1 - dd / 100)
        
        # Ponderar
        weighted = (
            cfg.PESO_SHARPE * n_sharpe +
            cfg.PESO_ROI * n_roi +
            cfg.PESO_DD * n_dd
        )
        
        return cfg.SCORE_MIN + (cfg.SCORE_MAX - cfg.SCORE_MIN) * weighted


@dataclass
class RandomOptimizerConfig:
    SEED: Optional[int] = None
    N_JOBS: int = 1

RANDOM_OPTIMIZER_CONFIG = RandomOptimizerConfig()


class RandomOptimizer:
    def __init__(self, config: BacktestConfig, n_trials: int, **kwargs):
        self.config = config
        self.n_trials = n_trials
        self._scorer = RandomScorer()
    
    def optimize(self, df, strategy, **kwargs) -> optuna.Study:
        sampler = RandomSampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        # ... implementar objetivo ...
        return study


def create_random_study(strategy_name: str, **kwargs) -> optuna.Study:
    sampler = RandomSampler(seed=kwargs.get("seed"))
    return optuna.create_study(direction="maximize", sampler=sampler)


__all__ = [
    "RandomOptimizer", "RandomOptimizerConfig",
    "RandomScorer", "RandomScoringConfig",
    "create_random_study",
]
```

---

## 8. CHECKLIST FINAL

### ✅ Antes de Commitear

- [ ] **Archivo creado**: `modelox/optimizers/tu_optimizador.py`
- [ ] **ScoringConfig**: Definida con todos los parámetros
- [ ] **OptimizerConfig**: Definida con parámetros del sampler
- [ ] **Scorer**: Implementado con `compute_score()`
- [ ] **Optimizer**: Implementado con `optimize()`
- [ ] **Funciones utilidad**: `create_*_study()` y `score_*()`
- [ ] **`__all__`**: Exportaciones definidas
- [ ] **`__init__.py`**: Importaciones añadidas
- [ ] **`__init__.py`**: Factory `create_study()` actualizado
- [ ] **`__init__.py`**: `__all__` actualizado
- [ ] **Probado**: Ejecutar con tu optimizador

### ✅ Buenas Prácticas

- [ ] Score NUNCA retorna 0 (usa SCORE_MIN = 1.0)
- [ ] Score SIEMPRE en rango [SCORE_MIN, SCORE_MAX]
- [ ] Usar `_safe_get()` para extraer métricas
- [ ] Guardar atributos en `trial.set_user_attr()` para debug
- [ ] Manejar excepciones retornando SCORE_MIN
- [ ] Documentar filosofía de scoring

---

## 🎯 RESUMEN RÁPIDO

```
1. CREAR ARCHIVO:     modelox/optimizers/mi_optimizador.py
2. DEFINIR:           ScoringConfig, OptimizerConfig, Scorer, Optimizer
3. MODIFICAR:         modelox/optimizers/__init__.py
   - Añadir imports
   - Actualizar create_study()
   - Actualizar __all__
4. OPCIONAL:          general/configuracion.py (OPTUNA_SAMPLER)
5. PROBAR:            python ejecutar.py
```

---

**Última actualización**: Febrero 2026  
**Autor**: MODELOX Team

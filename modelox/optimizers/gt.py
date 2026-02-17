"""modelox/optimizers/gt.py

═══════════════════════════════════════════════════════════════════════════════
    ██████╗ ████████╗       ███████╗ ██████╗ ██████╗ ██████╗ ███████╗
   ██╔════╝ ╚══██╔══╝       ██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║  ███╗   ██║   █████╗ ███████╗██║     ██║   ██║██████╔╝█████╗  
   ██║   ██║   ██║   ╚════╝ ╚════██║██║     ██║   ██║██╔══██╗██╔══╝  
   ╚██████╔╝   ██║          ███████║╚██████╗╚██████╔╝██║  ██║███████╗
    ╚═════╝    ╚═╝          ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

    GT-SCORE OPTIMIZER — ANTI-OVERFITTING CON INTELIGENCIA TOPOLÓGICA
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
GT-Score es un optimizador diseñado para ELIMINAR EL SOBREAJUSTE desde la raíz.
Combina CMA-ES como motor de búsqueda con un sistema de puntuación dinámico
que incorpora análisis topológico del espacio de hiperparámetros en tiempo real.

FILOSOFÍA CENTRAL:
==================
  ✗ UNA ESTRATEGIA EN UN PICO AISLADO → SOBREAJUSTE → RECHAZAR
  ✓ UNA ESTRATEGIA EN UNA MESETA ESTABLE → ROBUSTA → PREMIAR

ARQUITECTURA DEL SISTEMA:
=========================
  1. MOTOR DE BÚSQUEDA:
     └── CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
         Mantiene distribución N(m, σ²C), muestrea población cada generación.
         MODULAR: Se puede reemplazar por otro sampler de Optuna fácilmente.

  2. MÉTRICA BASE — GT-SCORE:
     └── Función objetivo compuesta anti-sobreajuste:
         GT = μ × ln(z) × r² × (1/σ_d) × Sharpe × SQN_norm
         • μ     = Rendimiento medio
         • ln(z) = Log del estadístico t (filtra ruido de muestreo)
         • r²    = Coeficiente de determinación de equity (consistencia)
         • σ_d   = Desviación a la baja (riesgo asimétrico)
         • Sharpe = Ratio de Sharpe (calidad riesgo/retorno)
         • SQN   = System Quality Number (calidad del sistema)

  3. INTERCEPTOR DE IA — MOTOR DE PENALIZACIÓN DINÁMICA:
     └── Dentro de cada trial, DESPUÉS de calcular GT-Score base:
         a) Se consulta el historial completo del Study
         b) Se calcula la DISTANCIA PONDERADA a todos los trials previos
         c) Se identifican los K vecinos más cercanos (k-NN)
         d) Se calcula la ESTABILIDAD del score en la vecindad
         e) Se penaliza o premia según estabilidad topológica

  4. DISTANCIA PONDERADA (GOWER MODIFICADA):
     └── D(a,b) = √( Σ wᵢ × (aᵢ - bᵢ)² )
         • Normalización Min-Max al rango [0,1] según límites de Optuna
         • Distancia de Hamming para categóricos (0 si igual, 1 si diferente)
         • Pesos wᵢ calculados con fANOVA periódicamente

  5. PESOS DE IMPORTANCIA (fANOVA):
     └── Functional ANOVA sobre el historial del estudio
         • Calcula la contribución de cada hiperparámetro al score
         • Los pesos se recalculan cada N trials
         • Un cambio del 1% en Stop Loss importa MÁS que en periodo de MA lenta

FLUJO DE DATOS POR TRIAL:
=========================
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRIAL t                                                                │
  │                                                                         │
  │  1. CMA-ES sugiere vector θₜ (hiperparámetros)                         │
  │  2. Backtest con θₜ → métricas crudas (ROI, DD, Sharpe, etc.)          │
  │  3. Calcular GT-Score BASE con las métricas                            │
  │  4. INTERCEPTOR DE IA:                                                  │
  │     a) Recuperar historial {θ₁..θₜ₋₁, score₁..scoreₜ₋₁}              │
  │     b) Calcular distancia ponderada D(θₜ, θᵢ) para todo i             │
  │     c) Seleccionar K vecinos más cercanos                              │
  │     d) Calcular estadísticas de vecindad:                              │
  │        - VARIANZA del score en vecindad                                │
  │        - GRADIENTE LOCAL (¿subiendo o bajando?)                        │
  │        - DESVIACIÓN MEDIA respecto al score actual                     │
  │     e) Determinar: ¿PICO AGUDO o MESETA?                              │
  │  5. Score_final = GT_base - Penalización_topológica                    │
  │  6. Retornar Score_final → CMA-ES actualiza N(m, σ²C)                 │
  └─────────────────────────────────────────────────────────────────────────┘

VENTAJAS:
=========
  ✓ GT-SCORE como primera capa de defensa (métricas anti-overfitting)
  ✓ ANÁLISIS TOPOLÓGICO como segunda capa (estabilidad paramétrica)
  ✓ fANOVA pondera dimensiones por IMPACTO REAL en rendimiento
  ✓ k-NN detecta picos aislados vs mesetas estables
  ✓ CMA-ES aprende de scores "corregidos por robustez"
  ✓ MODULAR: Motor de búsqueda intercambiable fácilmente

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
import os
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import CmaEsSampler

if TYPE_CHECKING:
    pass

# =============================================================================
# IMPORTS INTERNOS
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

# =============================================================================
# SILENCIAR WARNINGS
# =============================================================================
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
#  ██████╗ ████████╗       ███████╗ ██████╗ ██████╗ ██████╗ ███████╗
# ██╔════╝ ╚══██╔══╝       ██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
# ██║  ███╗   ██║   █████╗ ███████╗██║     ██║   ██║██████╔╝█████╗
# ██║   ██║   ██║   ╚════╝ ╚════██║██║     ██║   ██║██╔══██╗██╔══╝
# ╚██████╔╝   ██║          ███████║╚██████╗╚██████╔╝██║  ██║███████╗
#  ╚═════╝    ╚═╝          ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
#
# SISTEMA DE SCORING GT — GT-SCORE + INTERCEPTOR DE IA TOPOLÓGICO
# =============================================================================


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 1: CONFIGURACIÓN DEL GT-SCORE                                ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

@dataclass
class GTScoringConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │              CONFIGURACIÓN DEL GT-SCORE v1.0                            │
    │                                                                         │
    │  GT = μ × ln(z) × r² × (1/σ_d) × Sharpe_norm × SQN_norm              │
    │                                                                         │
    │  LUEGO:  Score_final = GT_base - Penalización_topológica               │
    │                                                                         │
    │  RANGO SALIDA: [1, 1000] - NUNCA CERO ABSOLUTO                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # =========================================================================
    # 1.1 RANGO DE SALIDA DEL SCORE FINAL
    # =========================================================================
    # EL SCORE FINAL SIEMPRE ESTÁ EN [SCORE_MIN, SCORE_MAX].
    # NUNCA RETORNAMOS CERO PARA QUE CMA-ES SIEMPRE TENGA GRADIENTES.
    SCORE_MIN: float = 1.0               # MÍNIMO ABSOLUTO (NUNCA 0)
    SCORE_MAX: float = 1000.0            # MÁXIMO ABSOLUTO

    # =========================================================================
    # 1.2 COMPONENTES DEL GT-SCORE BASE
    # =========================================================================
    # GT-SCORE ES UNA FUNCIÓN COMPUESTA DISEÑADA PARA REDUCIR SOBREAJUSTE.
    # CADA COMPONENTE ACTÚA COMO UN FILTRO INDEPENDIENTE:
    #
    #   μ (RENDIMIENTO MEDIO):
    #     - CAPTURA LA RENTABILIDAD MEDIA POR TRADE
    #     - SE NORMALIZA CON SIGMOIDE PARA EVITAR EXTREMOS
    #
    #   ln(z) (LOGARITMO DEL ESTADÍSTICO t):
    #     - ACTÚA COMO FILTRO DE SIGNIFICANCIA ESTADÍSTICA
    #     - RECHAZA ESTRATEGIAS QUE NO SUPERAN RUIDO DE MUESTREO
    #     - z = μ / (σ / √n) → ESTADÍSTICO t DE STUDENT
    #     - ln(z) COMPRIME VALORES EXTREMOS EVITANDO DOMINANCIA
    #
    #   r² (COEFICIENTE DE DETERMINACIÓN):
    #     - MIDE LA LINEALIDAD/CONSISTENCIA DE LA CURVA DE EQUITY
    #     - r² ALTO → CRECIMIENTO UNIFORME (DESEABLE)
    #     - r² BAJO → CRECIMIENTO ERRÁTICO (SOSPECHOSO)
    #
    #   σ_d (DESVIACIÓN A LA BAJA / DOWNSIDE DEVIATION):
    #     - RIESGO ASIMÉTRICO: SOLO PENALIZA VOLATILIDAD NEGATIVA
    #     - MÁS JUSTO QUE LA DESVIACIÓN ESTÁNDAR TOTAL
    #     - SE USA COMO (1 / (1 + σ_d)) PARA QUE SEA UN FACTOR [0,1]
    #
    #   SHARPE:
    #     - RATIO CLÁSICO DE CALIDAD RIESGO/RETORNO
    #     - SE NORMALIZA CON SIGMOIDE PARA CONTRIBUCIÓN SUAVE
    #
    #   SQN (SYSTEM QUALITY NUMBER):
    #     - MÉTRICA DE VAN THARP PARA CALIDAD DEL SISTEMA
    #     - SQN > 3 = EXCELENTE, SQN > 5 = EXTRAORDINARIO

    # PESOS RELATIVOS DE CADA COMPONENTE EN EL GT-SCORE
    # SUMAN 1.0 PARA QUE EL GT-SCORE ESTÉ EN [0, 1] ANTES DE ESCALAR
    PESO_RENDIMIENTO_MEDIO: float = 0.15       # μ — RENDIMIENTO MEDIO
    PESO_SIGNIFICANCIA_ESTADISTICA: float = 0.20  # ln(z) — ESTADÍSTICO t
    PESO_CONSISTENCIA_EQUITY: float = 0.20     # r² — LINEALIDAD DE EQUITY
    PESO_RIESGO_ASIMETRICO: float = 0.15       # 1/(1+σ_d) — DOWNSIDE DEV
    PESO_SHARPE: float = 0.15                  # SHARPE RATIO NORMALIZADO
    PESO_SQN: float = 0.15                     # SQN NORMALIZADO

    # PARÁMETROS DE NORMALIZACIÓN PARA CADA COMPONENTE
    # RENDIMIENTO MEDIO: SIGMOIDE CON CENTRO EN 0 Y ESCALA AJUSTABLE
    RENDIMIENTO_SIGMOID_CENTER: float = 0.0    # CENTRO: 0% DE RETORNO MEDIO
    RENDIMIENTO_SIGMOID_SCALE: float = 5.0     # PENDIENTE DE LA SIGMOIDE

    # SIGNIFICANCIA: UMBRAL MÍNIMO DEL ESTADÍSTICO t
    SIGNIFICANCIA_T_MIN: float = 1.0           # t < 1 → CONTRIBUCIÓN MUY BAJA
    SIGNIFICANCIA_LN_MAX: float = 3.5          # TECHO PARA NORMALIZAR ln(z)

    # CONSISTENCIA: UMBRAL MÍNIMO DE r²
    CONSISTENCIA_R2_MIN: float = 0.50          # r² < 0.50 → PENALIZACIÓN FUERTE

    # RIESGO: TECHO DE DOWNSIDE DEVIATION
    RIESGO_DOWNSIDE_MAX: float = 0.10          # σ_d > 10% → FACTOR MUY BAJO

    # SHARPE: NORMALIZACIÓN SIGMOIDE
    SHARPE_SIGMOID_CENTER: float = 1.0         # CENTRO DE LA SIGMOIDE
    SHARPE_SIGMOID_SCALE: float = 1.5          # PENDIENTE DE LA SIGMOIDE

    # SQN: VALOR OBJETIVO
    SQN_TARGET: float = 4.0                    # SQN OBJETIVO (>4 = EXCELENTE)

    # =========================================================================
    # 1.3 INTERCEPTOR DE IA — ANÁLISIS TOPOLÓGICO k-NN
    # =========================================================================
    # DESPUÉS DE CALCULAR GT-SCORE BASE, EL INTERCEPTOR ANALIZA LA
    # ESTABILIDAD PARAMÉTRICA USANDO k-NN SOBRE EL HISTORIAL DEL ESTUDIO.
    #
    # ¿QUÉ HACE?
    #   - ENCUENTRA LOS K TRIALS MÁS CERCANOS EN EL ESPACIO DE PARÁMETROS
    #   - CALCULA ESTADÍSTICAS DE ESTABILIDAD SOBRE ESOS VECINOS
    #   - PENALIZA SI EL SCORE ACTUAL ES UN "PICO AGUDO" (OVERFITTING)
    #   - PREMIA SI EL SCORE ACTUAL ESTÁ EN UNA "MESETA" (ROBUSTEZ)

    # ACTIVAR/DESACTIVAR EL INTERCEPTOR TOPOLÓGICO
    INTERCEPTOR_ENABLED: bool = True

    # NÚMERO DE VECINOS MÁS CERCANOS A CONSIDERAR (K EN k-NN)
    INTERCEPTOR_K_NEIGHBORS: int = 7

    # TRIALS MÍNIMOS ANTES DE ACTIVAR EL INTERCEPTOR (FASE DE CALENTAMIENTO)
    # DURANTE EL WARM-UP, LA PENALIZACIÓN TOPOLÓGICA ESTÁ DESACTIVADA,
    # PERMITIENDO A CMA-ES EXPLORAR LIBREMENTE HASTA TENER DENSIDAD DE
    # PUNTOS SUFICIENTE PARA CALCULAR ESTADÍSTICAS LOCALES FIABLES.
    #
    # VALOR DINÁMICO: SE CALCULA AUTOMÁTICAMENTE COMO 15% DE N_TRIALS.
    #   EJEMPLO: 200 TRIALS → WARMUP = 30,  500 TRIALS → WARMUP = 75
    #
    # ESTE DEFAULT (20) SE SOBREESCRIBE EN GTOptimizer._CREATE_STUDY()
    # CON max(10, int(0.15 * n_trials)).
    INTERCEPTOR_WARMUP_TRIALS: int = 20
    INTERCEPTOR_WARMUP_PCT: float = 0.15    # 15% DE LOS TRIALS TOTALES

    # PESO MÁXIMO DE LA PENALIZACIÓN TOPOLÓGICA
    # 0.0 = SIN PENALIZACIÓN, 1.0 = PENALIZACIÓN TOTAL
    # RECOMENDADO: 0.30-0.50 PARA BALANCE ENTRE EXPLORACIÓN Y ROBUSTEZ
    INTERCEPTOR_PENALIZACION_MAX: float = 0.40

    # UMBRAL DE COEFICIENTE DE VARIACIÓN EN VECINDAD
    # SI CV > ESTE UMBRAL → EL PUNTO ES UN PICO AGUDO → PENALIZAR
    INTERCEPTOR_CV_UMBRAL_PICO: float = 0.50

    # UMBRAL DE DEGRADACIÓN MÁXIMA PERMITIDA
    # SI EL SCORE ACTUAL ES MUCHO MAYOR QUE LA MEDIA DE VECINOS,
    # ES SOSPECHOSO (POSIBLE OVERFITTING)
    INTERCEPTOR_DEGRADACION_UMBRAL: float = 1.5

    # =========================================================================
    # 1.4 DISTANCIA PONDERADA — GOWER MODIFICADA
    # =========================================================================
    # LA DISTANCIA ENTRE DOS TRIALS EN EL ESPACIO DE HIPERPARÁMETROS
    # SE CALCULA COMO D(a,b) = √( Σ wᵢ × (aᵢ_norm - bᵢ_norm)² )
    #
    # LOS PESOS wᵢ SE CALCULAN CON fANOVA (FUNCTIONAL ANOVA) PARA QUE
    # DIMENSIONES MÁS IMPORTANTES TENGAN MÁS PESO EN LA DISTANCIA.

    # ACTIVAR fANOVA PARA CALCULAR PESOS DE IMPORTANCIA
    FANOVA_ENABLED: bool = True

    # RECALCULAR PESOS DE fANOVA CADA N TRIALS
    FANOVA_RECALCULATE_EVERY: int = 25

    # TRIALS MÍNIMOS ANTES DE CALCULAR fANOVA POR PRIMERA VEZ
    FANOVA_MIN_TRIALS: int = 30

    # PESO MÍNIMO POR DIMENSIÓN (EVITA QUE ALGUNA DIMENSIÓN SEA 0)
    FANOVA_PESO_MINIMO: float = 0.05

    # PESO MÁXIMO POR DIMENSIÓN (EVITA DOMINANCIA DE UNA SOLA DIMENSIÓN)
    FANOVA_PESO_MAXIMO: float = 0.60

    # =========================================================================
    # 1.5 UMBRALES DE PENALIZACIÓN BÁSICA (SOFT-VETO)
    # =========================================================================
    # PENALIZACIONES QUE SE APLICAN ANTES DEL INTERCEPTOR TOPOLÓGICO.
    # SON FILTROS BÁSICOS QUE REDUCEN EL SCORE PERO NUNCA LO ELIMINAN.

    MIN_TRADES_TOTAL: int = 15             # TRADES MÍNIMOS PARA SCORE VÁLIDO
    MIN_TRADES_POR_DIA: float = 0.10       # ACTIVIDAD MÍNIMA DIARIA (SOFT-VETO)
    MAX_DRAWDOWN_PERMITIDO: float = 60.0   # DRAWDOWN MÁXIMO ACEPTABLE (%)
    MIN_ROI_PERMITIDO: float = -150.0      # ROI MÍNIMO ANTES DE PENALIZAR

    # HARD-KILL: TRADES/DÍA MÍNIMO ABSOLUTO
    # SI TRADES_POR_DIA < ESTE VALOR → SCORE = 0 (SIN EXCEPCIONES)
    # ESTO GARANTIZA QUE NINGUNA ESTRATEGIA CON ACTIVIDAD INSUFICIENTE
    # OBTENGA PUNTUACIÓN, INDEPENDIENTEMENTE DEL RESTO DE MÉTRICAS.
    HARD_MIN_TRADES_POR_DIA: float = 0.20

    # FACTOR DE PENALIZACIÓN CUANDO SE VIOLA UN UMBRAL
    # EL SCORE SE MULTIPLICA POR ESTE FACTOR (SOFT-VETO, NO ELIMINACIÓN)
    UMBRAL_FACTOR_PENALIZACION: float = 0.15

    # =========================================================================
    # 1.6 PSR (PROBABILISTIC SHARPE RATIO) — CAPA ADICIONAL
    # =========================================================================
    # PSR DETERMINA LA PROBABILIDAD DE QUE EL SHARPE OBSERVADO SEA REAL
    # Y NO PRODUCTO DEL AZAR. ACTÚA COMO FILTRO DE SIGNIFICANCIA.

    PSR_ENABLED: bool = True               # ACTIVAR PSR
    PSR_BENCHMARK_SR: float = 0.0          # SR DE REFERENCIA PARA PSR
    PSR_MIN_TRADES: int = 30               # TRADES MÍNIMOS PARA PSR VÁLIDO
    PSR_FLOOR: float = 0.25                # SOFT-VETO: MÍNIMO 25%
    PSR_PESO_EN_FINAL: float = 0.15        # PESO DEL PSR EN SCORE FINAL


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
GT_SCORING_CONFIG = GTScoringConfig()


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 2: CLASE GTScorer — MOTOR DE PUNTUACIÓN GT-SCORE              ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class GTScorer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    SCORER GT-SCORE v1.0                                  │
    │                                                                         │
    │  ARQUITECTURA:                                                          │
    │    1. GT_base = f(μ, ln(z), r², σ_d, Sharpe, SQN)                      │
    │    2. Penalización_topológica = g(k-NN, fANOVA, historial)             │
    │    3. Score_final = GT_base × (1 - Penalización_topológica) × PSR      │
    │                                                                         │
    │  RANGO: [1, 1000] - NUNCA CERO ABSOLUTO                               │
    │  FILOSOFÍA: EL GT-SCORE BASE ES LA PRIMERA DEFENSA.                    │
    │             EL INTERCEPTOR TOPOLÓGICO ES LA SEGUNDA DEFENSA.            │
    │             JUNTOS, ELIMINAN EL SOBREAJUSTE DESDE LA RAÍZ.             │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[GTScoringConfig] = None,
    ):
        """
        INICIALIZA EL SCORER GT-SCORE.

        ARGS:
            study: OBJETO OPTUNA.STUDY PARA ACCEDER AL HISTORIAL DE TRIALS.
                   SE USA PARA EL INTERCEPTOR TOPOLÓGICO (k-NN, fANOVA).
            config: CONFIGURACIÓN PERSONALIZADA DEL SCORING.
                    SI ES NONE, SE USA LA CONFIGURACIÓN POR DEFECTO.
        """
        self.study = study
        self.config = config or GT_SCORING_CONFIG

        # =====================================================================
        # CACHE DE fANOVA
        # =====================================================================
        # LOS PESOS DE IMPORTANCIA DE CADA HIPERPARÁMETRO SE CACHEAN
        # Y SE RECALCULAN CADA N TRIALS (CONFIGURADO EN FANOVA_RECALCULATE_EVERY)
        self._FANOVA_PESOS: Optional[Dict[str, float]] = None
        self._FANOVA_CALCULADO_EN_TRIAL: int = -1

        # =====================================================================
        # CACHE DE PARÁMETROS DEL HISTORIAL
        # =====================================================================
        # SE CACHEAN LOS LÍMITES MIN/MAX DE CADA PARÁMETRO PARA NORMALIZACIÓN
        self._PARAM_BOUNDS: Optional[Dict[str, Tuple[float, float]]] = None
        self._PARAM_BOUNDS_TRIAL: int = -1

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.1: FUNCIONES AUXILIARES — UTILIDADES GENÉRICAS
    #
    # =========================================================================
    # =========================================================================

    @staticmethod
    def _SAFE_GET(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """
        EXTRAE UN VALOR NUMÉRICO DE UN DICCIONARIO DE MÉTRICAS DE FORMA SEGURA.

        MANEJA AUTOMÁTICAMENTE:
          - VALORES NONE → RETORNA DEFAULT
          - VALORES NaN/Inf → RETORNA DEFAULT
          - VALORES NO NUMÉRICOS → RETORNA DEFAULT
          - ERRORES DE CONVERSIÓN → RETORNA DEFAULT

        ARGS:
            metrics: DICCIONARIO DE MÉTRICAS DEL BACKTEST
            key: CLAVE A BUSCAR EN EL DICCIONARIO
            default: VALOR POR DEFECTO SI LA CLAVE NO EXISTE O ES INVÁLIDA

        RETURNS:
            VALOR NUMÉRICO FLOTANTE, SIEMPRE FINITO Y VÁLIDO
        """
        try:
            VAL = metrics.get(key, default)
            if VAL is None:
                return default
            F_VAL = float(VAL)
            if math.isnan(F_VAL) or math.isinf(F_VAL):
                return default
            return F_VAL
        except Exception:
            return default

    @staticmethod
    def _SIGMOID(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        """
        FUNCIÓN SIGMOIDE GENÉRICA PARA NORMALIZAR VALORES A [0, 1].

        FÓRMULA:
            σ(x) = 1 / (1 + exp(-scale × (x - center)))

        PROPIEDADES:
          - x << center → σ(x) ≈ 0
          - x == center → σ(x) = 0.5
          - x >> center → σ(x) ≈ 1
          - TRANSICIÓN SUAVE, DIFERENCIABLE EN TODO EL DOMINIO
          - PROTEGIDA CONTRA OVERFLOW NUMÉRICO

        ARGS:
            x: VALOR DE ENTRADA A NORMALIZAR
            center: PUNTO DONDE σ(x) = 0.5
            scale: CONTROLA LA PENDIENTE (MAYOR = MÁS EMPINADA)

        RETURNS:
            VALOR EN [0, 1]
        """
        try:
            EXPONENT = -scale * (x - center)
            # PROTECCIÓN CONTRA OVERFLOW DE math.exp()
            if EXPONENT > 500:
                return 0.0
            elif EXPONENT < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(EXPONENT))
        except (OverflowError, ValueError):
            return 0.5

    @staticmethod
    def _ERFINV(x: float) -> float:
        """
        APROXIMACIÓN DE LA FUNCIÓN INVERSA DE ERROR (erf⁻¹).

        SE USA INTERNAMENTE PARA CALCULAR EL PSR (PROBABILISTIC SHARPE RATIO).
        LA APROXIMACIÓN DE WINITZKI ES SUFICIENTE PARA NUESTRO PROPÓSITO.

        ARGS:
            x: VALOR EN (-1, 1)

        RETURNS:
            erf⁻¹(x) — INVERSA DE LA FUNCIÓN DE ERROR
        """
        A = 0.147
        SIGN = 1 if x >= 0 else -1
        X_ABS = abs(x)
        if X_ABS >= 1.0:
            return SIGN * float("inf")
        LN_TERM = math.log(1 - X_ABS * X_ABS)
        TERM1 = (2 / (math.pi * A)) + (LN_TERM / 2)
        TERM2 = LN_TERM / A
        RESULT = SIGN * math.sqrt(math.sqrt(TERM1 * TERM1 - TERM2) - TERM1)
        return RESULT

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.2: COMPONENTES DEL GT-SCORE BASE
    #
    #  GT_base = Σ(peso_i × componente_i)
    #
    #  CADA COMPONENTE RETORNA UN VALOR EN [0, 1] QUE REPRESENTA
    #  LA "CALIDAD" DE ESA DIMENSIÓN PARTICULAR DEL BACKTEST.
    #
    # =========================================================================
    # =========================================================================

    def _CALCULAR_COMPONENTE_RENDIMIENTO_MEDIO(
        self,
        RETURNS: np.ndarray,
    ) -> float:
        """
        COMPONENTE μ — RENDIMIENTO MEDIO NORMALIZADO.

        CAPTURA LA RENTABILIDAD MEDIA POR TRADE.
        SE NORMALIZA CON SIGMOIDE PARA QUE VALORES EXTREMOS
        NO DOMINEN EL GT-SCORE.

        CÁLCULO:
          1. MEDIA ARITMÉTICA DE LOS RETORNOS POR TRADE
          2. NORMALIZACIÓN CON SIGMOIDE: σ(μ, center, scale)
          3. RESULTADO EN [0, 1]:
             - μ << 0 → ~0.0 (PÉRDIDA CONSISTENTE)
             - μ ≈ 0  → ~0.5 (BREAK-EVEN)
             - μ >> 0 → ~1.0 (GANANCIA CONSISTENTE)

        ARGS:
            RETURNS: ARRAY DE RETORNOS POR TRADE (PnL / CAPITAL)

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DEL RENDIMIENTO MEDIO AL GT-SCORE
        """
        CFG = self.config

        RETURNS_CLEAN = np.asarray(RETURNS, dtype=np.float64)
        RETURNS_CLEAN = RETURNS_CLEAN[np.isfinite(RETURNS_CLEAN)]

        if len(RETURNS_CLEAN) < 3:
            return 0.0

        MU = float(np.mean(RETURNS_CLEAN))
        NORMALIZADO = self._SIGMOID(
            MU,
            center=CFG.RENDIMIENTO_SIGMOID_CENTER,
            scale=CFG.RENDIMIENTO_SIGMOID_SCALE,
        )
        return float(np.clip(NORMALIZADO, 0.01, 0.99))

    def _CALCULAR_COMPONENTE_SIGNIFICANCIA(
        self,
        RETURNS: np.ndarray,
    ) -> float:
        """
        COMPONENTE ln(z) — LOGARITMO DEL ESTADÍSTICO t DE STUDENT.

        ESTE COMPONENTE ACTÚA COMO FILTRO DE SIGNIFICANCIA ESTADÍSTICA.
        RECHAZA ESTRATEGIAS CUYO RENDIMIENTO NO SUPERA EL RUIDO DE MUESTREO.

        CÁLCULO:
          1. z = μ / (σ / √n) → ESTADÍSTICO t DE STUDENT
          2. SI z < 1.0 → LA SEÑAL NO SUPERA EL RUIDO → PENALIZAR
          3. ln(z) COMPRIME VALORES GRANDES PARA EVITAR DOMINANCIA
          4. NORMALIZAR AL RANGO [0, 1] DIVIDIENDO POR ln(z_max)

        INTERPRETACIÓN:
          - z < 1.0 → EL RENDIMIENTO NO ES ESTADÍSTICAMENTE SIGNIFICATIVO
          - z ≈ 2.0 → SIGNIFICATIVO AL 95% (APROX.)
          - z ≈ 3.0 → SIGNIFICATIVO AL 99%
          - z > 5.0 → MUY SIGNIFICATIVO (PERO ln COMPRIME)

        ARGS:
            RETURNS: ARRAY DE RETORNOS POR TRADE

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DE LA SIGNIFICANCIA AL GT-SCORE
        """
        CFG = self.config

        RETURNS_CLEAN = np.asarray(RETURNS, dtype=np.float64)
        RETURNS_CLEAN = RETURNS_CLEAN[np.isfinite(RETURNS_CLEAN)]

        N = len(RETURNS_CLEAN)
        if N < 5:
            return 0.0

        MU = float(np.mean(RETURNS_CLEAN))
        SIGMA = float(np.std(RETURNS_CLEAN, ddof=1))

        if SIGMA < 1e-10:
            # DESVIACIÓN CERO: SI LA MEDIA ES POSITIVA, PERFECTO; SI NO, MALO
            return 0.9 if MU > 0 else 0.1

        # ESTADÍSTICO t DE STUDENT
        Z = MU / (SIGMA / math.sqrt(N))

        if Z <= 0:
            return 0.0

        if Z < CFG.SIGNIFICANCIA_T_MIN:
            # EL ESTADÍSTICO NO ALCANZA EL UMBRAL MÍNIMO → CONTRIBUCIÓN BAJA
            return float(np.clip(Z / CFG.SIGNIFICANCIA_T_MIN * 0.3, 0.0, 0.3))

        # LOGARITMO NATURAL PARA COMPRIMIR VALORES GRANDES
        LN_Z = math.log(Z)
        LN_MAX = math.log(max(CFG.SIGNIFICANCIA_LN_MAX, 1.01))

        NORMALIZADO = LN_Z / LN_MAX
        return float(np.clip(NORMALIZADO, 0.0, 1.0))

    def _CALCULAR_COMPONENTE_CONSISTENCIA_EQUITY(
        self,
        EQUITY_CURVE: np.ndarray,
    ) -> float:
        """
        COMPONENTE r² — COEFICIENTE DE DETERMINACIÓN DE LA CURVA DE EQUITY.

        MIDE QUÉ TAN LINEAL Y CONSISTENTE ES EL CRECIMIENTO DEL CAPITAL.
        UNA CURVA DE EQUITY PERFECTAMENTE LINEAL TIENE r² = 1.0.

        CÁLCULO:
          1. REGRESIÓN LINEAL SIMPLE: y = m×x + b
          2. r² = 1 - SS_res / SS_tot
          3. SI r² < UMBRAL_MÍNIMO → PENALIZACIÓN PROPORCIONAL

        INTERPRETACIÓN:
          - r² > 0.90 → CRECIMIENTO MUY CONSISTENTE (DESEABLE)
          - r² ≈ 0.70 → CRECIMIENTO MODERADAMENTE CONSISTENTE
          - r² < 0.50 → CRECIMIENTO ERRÁTICO (SOSPECHOSO)

        ARGS:
            EQUITY_CURVE: ARRAY CON LA EVOLUCIÓN DEL CAPITAL

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DE LA CONSISTENCIA AL GT-SCORE
        """
        CFG = self.config

        EQUITY = np.asarray(EQUITY_CURVE, dtype=np.float64)
        EQUITY = EQUITY[np.isfinite(EQUITY)]

        N = len(EQUITY)
        if N < 10:
            return 0.1

        # REGRESIÓN LINEAL MANUAL (RÁPIDA, SIN NUMPY.LINALG)
        X = np.arange(N, dtype=np.float64)
        X_MEAN = np.mean(X)
        Y_MEAN = np.mean(EQUITY)

        SS_TOT = np.sum((EQUITY - Y_MEAN) ** 2)
        if SS_TOT < 1e-10:
            return 1.0  # LÍNEA PERFECTAMENTE PLANA

        NUMERATOR = np.sum((X - X_MEAN) * (EQUITY - Y_MEAN))
        DENOMINATOR = np.sum((X - X_MEAN) ** 2)

        if DENOMINATOR < 1e-10:
            return 0.1

        SLOPE = NUMERATOR / DENOMINATOR
        INTERCEPT = Y_MEAN - SLOPE * X_MEAN
        Y_PRED = SLOPE * X + INTERCEPT
        SS_RES = np.sum((EQUITY - Y_PRED) ** 2)

        R2 = 1.0 - (SS_RES / SS_TOT)
        R2 = float(np.clip(R2, 0.0, 1.0))

        # PENALIZACIÓN SI r² ESTÁ POR DEBAJO DEL UMBRAL MÍNIMO
        if R2 < CFG.CONSISTENCIA_R2_MIN:
            R2 *= (R2 / CFG.CONSISTENCIA_R2_MIN)

        return float(np.clip(R2, 0.0, 1.0))

    def _CALCULAR_COMPONENTE_RIESGO_ASIMETRICO(
        self,
        RETURNS: np.ndarray,
    ) -> float:
        """
        COMPONENTE 1/(1+σ_d) — DOWNSIDE DEVIATION (RIESGO ASIMÉTRICO).

        A DIFERENCIA DE LA DESVIACIÓN ESTÁNDAR TOTAL, LA DOWNSIDE DEVIATION
        SOLO PENALIZA LA VOLATILIDAD NEGATIVA. ES MÁS JUSTO PORQUE NO
        CASTIGA LA VOLATILIDAD AL ALZA (QUE ES DESEABLE).

        CÁLCULO:
          1. FILTRAR SOLO RETORNOS NEGATIVOS (O MENORES QUE UN TARGET)
          2. σ_d = √(Σ min(rᵢ - target, 0)² / N)
          3. FACTOR = 1 / (1 + σ_d / σ_d_max)
          4. RESULTADO EN [0, 1]:
             - σ_d ≈ 0   → ~1.0 (CASI SIN RIESGO A LA BAJA)
             - σ_d = MAX  → ~0.5 (RIESGO MODERADO)
             - σ_d >> MAX → ~0.0 (RIESGO EXCESIVO)

        ARGS:
            RETURNS: ARRAY DE RETORNOS POR TRADE

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DEL RIESGO ASIMÉTRICO AL GT-SCORE
        """
        CFG = self.config

        RETURNS_CLEAN = np.asarray(RETURNS, dtype=np.float64)
        RETURNS_CLEAN = RETURNS_CLEAN[np.isfinite(RETURNS_CLEAN)]

        if len(RETURNS_CLEAN) < 3:
            return 0.0

        # SOLO RETORNOS NEGATIVOS (TARGET = 0)
        NEGATIVE_RETURNS = RETURNS_CLEAN[RETURNS_CLEAN < 0]

        if len(NEGATIVE_RETURNS) == 0:
            # NO HAY RETORNOS NEGATIVOS → RIESGO MÍNIMO
            return 0.99

        # DOWNSIDE DEVIATION
        SIGMA_D = float(np.sqrt(np.mean(NEGATIVE_RETURNS ** 2)))

        # NORMALIZAR CON FACTOR INVERSAMENTE PROPORCIONAL
        FACTOR = 1.0 / (1.0 + SIGMA_D / CFG.RIESGO_DOWNSIDE_MAX)
        return float(np.clip(FACTOR, 0.01, 0.99))

    def _CALCULAR_COMPONENTE_SHARPE(
        self,
        SHARPE_VALOR: float,
    ) -> float:
        """
        COMPONENTE SHARPE — RATIO DE SHARPE NORMALIZADO CON SIGMOIDE.

        EL SHARPE RATIO CLÁSICO MIDE EL EXCESO DE RETORNO POR UNIDAD
        DE RIESGO. SE NORMALIZA CON SIGMOIDE PARA QUE CONTRIBUYA
        SUAVEMENTE AL GT-SCORE.

        MAPEO APROXIMADO:
          - SHARPE -2 → ~0.05 (MUY MALO)
          - SHARPE  0 → ~0.20 (MEDIOCRE)
          - SHARPE  1 → ~0.50 (CENTRO)
          - SHARPE  2 → ~0.80 (BUENO)
          - SHARPE  4 → ~0.95 (EXCELENTE)

        ARGS:
            SHARPE_VALOR: SHARPE RATIO CALCULADO POR EL MOTOR DE MÉTRICAS

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DEL SHARPE AL GT-SCORE
        """
        CFG = self.config
        NORMALIZADO = self._SIGMOID(
            SHARPE_VALOR,
            center=CFG.SHARPE_SIGMOID_CENTER,
            scale=CFG.SHARPE_SIGMOID_SCALE,
        )
        return float(np.clip(NORMALIZADO, 0.01, 0.99))

    def _CALCULAR_COMPONENTE_SQN(
        self,
        SQN_VALOR: float,
    ) -> float:
        """
        COMPONENTE SQN — SYSTEM QUALITY NUMBER NORMALIZADO.

        EL SQN DE VAN THARP MIDE LA CALIDAD GENERAL DEL SISTEMA DE TRADING:
          - SQN < 1.6  → POBRE
          - SQN 1.6-2  → PROMEDIO
          - SQN 2-2.5  → BUENO
          - SQN 2.5-3  → EXCELENTE
          - SQN > 3    → SANTO GRIAL

        SE NORMALIZA LINEALMENTE HASTA EL TARGET, SATURANDO EN 1.0.

        ARGS:
            SQN_VALOR: SQN CALCULADO POR EL MOTOR DE MÉTRICAS

        RETURNS:
            VALOR EN [0, 1] — CONTRIBUCIÓN DEL SQN AL GT-SCORE
        """
        CFG = self.config
        if SQN_VALOR <= 0:
            return 0.0
        NORMALIZADO = SQN_VALOR / CFG.SQN_TARGET
        return float(np.clip(NORMALIZADO, 0.0, 1.0))

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.3: FUNCIÓN DE CÁLCULO DEL GT-SCORE BASE
    #
    #  COMBINA TODOS LOS COMPONENTES EN UN ÚNICO VALOR [0, 1]
    #
    # =========================================================================
    # =========================================================================

    def _CALCULAR_GT_SCORE_BASE(
        self,
        RETURNS: Optional[np.ndarray],
        EQUITY_CURVE: Optional[np.ndarray],
        SHARPE_VALOR: float,
        SQN_VALOR: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        CALCULA EL GT-SCORE BASE COMBINANDO TODOS LOS COMPONENTES.

        GT_base = Σ(peso_i × componente_i)

        DONDE CADA COMPONENTE ESTÁ EN [0, 1] Y LOS PESOS SUMAN 1.0.
        EL RESULTADO GT_base ESTÁ EN [0, 1].

        ARGS:
            RETURNS: ARRAY DE RETORNOS POR TRADE (PUEDE SER NONE)
            EQUITY_CURVE: CURVA DE EQUITY (PUEDE SER NONE)
            SHARPE_VALOR: SHARPE RATIO DEL BACKTEST
            SQN_VALOR: SQN DEL BACKTEST

        RETURNS:
            TUPLA:
              - GT_BASE: VALOR EN [0, 1]
              - DESGLOSE: DICCIONARIO CON EL VALOR DE CADA COMPONENTE
                         (PARA AUDITORÍA Y TRANSPARENCIA)
        """
        CFG = self.config

        # =====================================================================
        # CALCULAR CADA COMPONENTE INDIVIDUALMENTE
        # =====================================================================
        # SI NO HAY RETURNS DISPONIBLES, LOS COMPONENTES QUE LOS NECESITAN
        # SE CALCULAN CON VALORES ESTIMADOS DESDE LAS MÉTRICAS

        if RETURNS is not None and len(RETURNS) >= 5:
            COMP_MU = self._CALCULAR_COMPONENTE_RENDIMIENTO_MEDIO(RETURNS)
            COMP_LN_Z = self._CALCULAR_COMPONENTE_SIGNIFICANCIA(RETURNS)
            COMP_SIGMA_D = self._CALCULAR_COMPONENTE_RIESGO_ASIMETRICO(RETURNS)
        else:
            # FALLBACK: ESTIMAR DESDE SHARPE (MENOS PRECISO PERO FUNCIONAL)
            COMP_MU = self._SIGMOID(SHARPE_VALOR * 0.01, center=0.0, scale=5.0)
            COMP_LN_Z = self._SIGMOID(SHARPE_VALOR, center=1.0, scale=1.0) * 0.7
            COMP_SIGMA_D = 0.5

        if EQUITY_CURVE is not None and len(EQUITY_CURVE) >= 10:
            COMP_R2 = self._CALCULAR_COMPONENTE_CONSISTENCIA_EQUITY(EQUITY_CURVE)
        else:
            # FALLBACK: VALOR NEUTRAL
            COMP_R2 = 0.5

        COMP_SHARPE = self._CALCULAR_COMPONENTE_SHARPE(SHARPE_VALOR)
        COMP_SQN = self._CALCULAR_COMPONENTE_SQN(SQN_VALOR)

        # =====================================================================
        # COMBINAR COMPONENTES CON PESOS
        # =====================================================================
        GT_BASE = (
            CFG.PESO_RENDIMIENTO_MEDIO * COMP_MU
            + CFG.PESO_SIGNIFICANCIA_ESTADISTICA * COMP_LN_Z
            + CFG.PESO_CONSISTENCIA_EQUITY * COMP_R2
            + CFG.PESO_RIESGO_ASIMETRICO * COMP_SIGMA_D
            + CFG.PESO_SHARPE * COMP_SHARPE
            + CFG.PESO_SQN * COMP_SQN
        )

        GT_BASE = float(np.clip(GT_BASE, 0.0, 1.0))

        # =====================================================================
        # DESGLOSE PARA AUDITORÍA
        # =====================================================================
        DESGLOSE = {
            "COMP_RENDIMIENTO_MEDIO_MU": COMP_MU,
            "COMP_SIGNIFICANCIA_LN_Z": COMP_LN_Z,
            "COMP_CONSISTENCIA_R2": COMP_R2,
            "COMP_RIESGO_SIGMA_D": COMP_SIGMA_D,
            "COMP_SHARPE_NORM": COMP_SHARPE,
            "COMP_SQN_NORM": COMP_SQN,
            "GT_BASE_CRUDO": GT_BASE,
        }

        return GT_BASE, DESGLOSE

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.4: PSR (PROBABILISTIC SHARPE RATIO)
    #
    #  CAPA ADICIONAL QUE FILTRA SHARPE RATIOS ESTADÍSTICAMENTE INVÁLIDOS
    #
    # =========================================================================
    # =========================================================================

    def _CALCULAR_PSR(
        self,
        RETURNS: np.ndarray,
        BENCHMARK_SR: Optional[float] = None,
    ) -> float:
        """
        CALCULA EL PROBABILISTIC SHARPE RATIO (PSR).

        EL PSR DETERMINA LA PROBABILIDAD DE QUE EL SHARPE RATIO OBSERVADO
        SEA SUPERIOR A UN UMBRAL DE REFERENCIA, AJUSTANDO POR SKEWNESS
        Y KURTOSIS DE LA DISTRIBUCIÓN DE RETORNOS.

        FÓRMULA:
            PSR(SR*) = Φ( (SR_hat - SR*) × √(n-1) / σ_sr )

        DONDE:
            σ_sr = √( 1 - γ₃×SR + (γ₄-1)/4 × SR² )
            γ₃ = SKEWNESS
            γ₄ = KURTOSIS
            Φ = CDF DE LA DISTRIBUCIÓN NORMAL ESTÁNDAR

        INTERPRETACIÓN:
          - PSR > 0.95 → MUY ALTA PROBABILIDAD DE QUE EL SR SEA REAL
          - PSR ≈ 0.50 → INCERTIDUMBRE SOBRE LA VALIDEZ DEL SR
          - PSR < 0.20 → EL SR PROBABLEMENTE ES PRODUCTO DEL AZAR

        ARGS:
            RETURNS: ARRAY DE RETORNOS POR TRADE
            BENCHMARK_SR: SHARPE RATIO DE REFERENCIA (DEFAULT: 0.0)

        RETURNS:
            PROBABILIDAD [0, 1] DE QUE EL SR REAL SEA > BENCHMARK
        """
        CFG = self.config

        RETURNS_CLEAN = np.asarray(RETURNS, dtype=np.float64)
        RETURNS_CLEAN = RETURNS_CLEAN[np.isfinite(RETURNS_CLEAN)]

        N = len(RETURNS_CLEAN)
        if N < CFG.PSR_MIN_TRADES:
            return 0.1  # MUESTRA INSUFICIENTE → PENALIZACIÓN SEVERA

        # MOMENTOS ESTADÍSTICOS
        MEAN_VAL = float(np.mean(RETURNS_CLEAN))
        STD_VAL = float(np.std(RETURNS_CLEAN, ddof=1))
        if STD_VAL < 1e-10:
            STD_VAL = 1e-10

        # SHARPE RATIO MUESTRAL
        SR = MEAN_VAL / STD_VAL

        # SKEWNESS Y KURTOSIS
        M3 = float(np.mean((RETURNS_CLEAN - MEAN_VAL) ** 3))
        M4 = float(np.mean((RETURNS_CLEAN - MEAN_VAL) ** 4))
        SKEW = M3 / (STD_VAL ** 3) if STD_VAL > 0 else 0.0
        KURT = M4 / (STD_VAL ** 4) if STD_VAL > 0 else 3.0

        # SR DE REFERENCIA
        SR_STAR = BENCHMARK_SR if BENCHMARK_SR is not None else CFG.PSR_BENCHMARK_SR

        # ERROR ESTÁNDAR DEL SHARPE AJUSTADO POR MOMENTOS SUPERIORES
        SR_SQ = SR ** 2
        VARIANCE_FACTOR = 1.0 - SKEW * SR + ((KURT - 1.0) / 4.0) * SR_SQ
        VARIANCE_FACTOR = max(0.01, VARIANCE_FACTOR)

        SIGMA_SR = math.sqrt(VARIANCE_FACTOR / max(1, N - 1))

        if SIGMA_SR < 1e-10:
            return 0.9 if SR > SR_STAR else 0.1

        # Z-SCORE
        Z_SCORE = (SR - SR_STAR) / SIGMA_SR

        # CDF DE NORMAL ESTÁNDAR (USANDO FUNCIÓN DE ERROR)
        PSR_VAL = 0.5 * (1.0 + math.erf(Z_SCORE / math.sqrt(2)))

        return float(np.clip(PSR_VAL, 0.01, 0.99))

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.5: INTERCEPTOR DE IA — ANÁLISIS TOPOLÓGICO k-NN
    #
    #  ESTE ES EL NÚCLEO DIFERENCIAL DEL GT-SCORE.
    #  ANALIZA LA ESTABILIDAD DEL SCORE EN EL ESPACIO DE HIPERPARÁMETROS.
    #
    # =========================================================================
    # =========================================================================

    def _OBTENER_HISTORIAL_TRIALS(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        RECUPERA EL HISTORIAL COMPLETO DE TRIALS DEL ESTUDIO.

        EXTRAE PARA CADA TRIAL COMPLETADO:
          - VECTOR DE PARÁMETROS (HIPERPARÁMETROS SUGERIDOS POR OPTUNA)
          - VALOR OBJETIVO (SCORE RETORNADO A OPTUNA)

        SE FILTRAN TRIALS INCOMPLETOS, FALLIDOS O CON VALORES INVÁLIDOS.

        RETURNS:
            TUPLA:
              - LISTA DE DICCIONARIOS DE PARÁMETROS [{param: valor}, ...]
              - LISTA DE SCORES CORRESPONDIENTES [score_1, score_2, ...]
        """
        if self.study is None:
            return [], []

        PARAMS_LISTA: List[Dict[str, Any]] = []
        SCORES_LISTA: List[float] = []

        for TRIAL in self.study.trials:
            # SOLO TRIALS COMPLETADOS EXITOSAMENTE
            if not TRIAL.state.is_finished():
                continue
            # VERIFICAR QUE TIENE VALOR OBJETIVO VÁLIDO
            if TRIAL.value is None:
                continue
            if not math.isfinite(TRIAL.value):
                continue
            # VERIFICAR QUE TIENE PARÁMETROS
            if not TRIAL.params:
                continue

            PARAMS_LISTA.append(dict(TRIAL.params))
            SCORES_LISTA.append(float(TRIAL.value))

        return PARAMS_LISTA, SCORES_LISTA

    def _CALCULAR_BOUNDS_PARAMETROS(
        self,
        PARAMS_LISTA: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        CALCULA LOS LÍMITES MIN/MAX DE CADA PARÁMETRO PARA NORMALIZACIÓN.

        LA NORMALIZACIÓN MIN-MAX ESCALA CADA PARÁMETRO AL RANGO [0, 1]:
            x_norm = (x - x_min) / (x_max - x_min)

        ESTO ES NECESARIO PARA QUE LA DISTANCIA EUCLIDIANA NO ESTÉ
        DOMINADA POR PARÁMETROS CON RANGOS GRANDES.

        ARGS:
            PARAMS_LISTA: LISTA DE DICCIONARIOS DE PARÁMETROS DE TRIALS PREVIOS

        RETURNS:
            DICCIONARIO {param_name: (min_value, max_value)}
        """
        if not PARAMS_LISTA:
            return {}

        # RECOPILAR TODOS LOS VALORES DE CADA PARÁMETRO
        PARAM_VALUES: Dict[str, List[float]] = defaultdict(list)

        for PARAMS in PARAMS_LISTA:
            for KEY, VAL in PARAMS.items():
                if isinstance(VAL, (int, float)):
                    if math.isfinite(VAL):
                        PARAM_VALUES[KEY].append(float(VAL))
                # CATEGÓRICOS SE MANEJAN APARTE EN LA DISTANCIA

        BOUNDS: Dict[str, Tuple[float, float]] = {}
        for KEY, VALS in PARAM_VALUES.items():
            if len(VALS) < 2:
                BOUNDS[KEY] = (VALS[0] - 1.0, VALS[0] + 1.0)
            else:
                BOUNDS[KEY] = (min(VALS), max(VALS))

        return BOUNDS

    def _CALCULAR_FANOVA_PESOS(
        self,
        PARAMS_LISTA: List[Dict[str, Any]],
        SCORES_LISTA: List[float],
    ) -> Dict[str, float]:
        """
        CALCULA LOS PESOS DE IMPORTANCIA DE CADA HIPERPARÁMETRO USANDO fANOVA.

        fANOVA (FUNCTIONAL ANOVA) DESCOMPONE LA VARIANZA DEL SCORE EN
        CONTRIBUCIONES INDIVIDUALES DE CADA HIPERPARÁMETRO.

        IMPLEMENTACIÓN SIMPLIFICADA:
          1. PARA CADA PARÁMETRO, CALCULAR LA CORRELACIÓN CON EL SCORE
          2. USAR |correlación|² COMO PROXY DE LA IMPORTANCIA
          3. NORMALIZAR PARA QUE SUMEN 1.0

        NOTA: ESTA ES UNA APROXIMACIÓN EFICIENTE DE fANOVA COMPLETO.
        PARA PRODUCCIÓN PUEDE REEMPLAZARSE POR optuna.importance.get_param_importances()

        ARGS:
            PARAMS_LISTA: LISTA DE DICCIONARIOS DE PARÁMETROS
            SCORES_LISTA: LISTA DE SCORES CORRESPONDIENTES

        RETURNS:
            DICCIONARIO {param_name: peso} DONDE LOS PESOS SUMAN 1.0
        """
        CFG = self.config

        if len(PARAMS_LISTA) < CFG.FANOVA_MIN_TRIALS:
            return {}

        SCORES_ARRAY = np.array(SCORES_LISTA, dtype=np.float64)
        SCORES_STD = np.std(SCORES_ARRAY)

        if SCORES_STD < 1e-10:
            return {}

        # INTENTAR USAR OPTUNA.IMPORTANCE PARA fANOVA REAL
        # SI FALLA, USAR APROXIMACIÓN POR CORRELACIÓN
        IMPORTANCIAS: Dict[str, float] = {}

        try:
            # INTENTAR fANOVA REAL DE OPTUNA
            IMPORTANCIAS_OPTUNA = optuna.importance.get_param_importances(
                self.study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )
            IMPORTANCIAS = dict(IMPORTANCIAS_OPTUNA)
        except Exception:
            # FALLBACK: APROXIMACIÓN POR CORRELACIÓN ABSOLUTA
            # PARA CADA PARÁMETRO NUMÉRICO, CALCULAR |corr(param, score)|²
            TODAS_LAS_KEYS = set()
            for P in PARAMS_LISTA:
                TODAS_LAS_KEYS.update(P.keys())

            for KEY in TODAS_LAS_KEYS:
                VALORES = []
                SCORES_FILTRADOS = []

                for I, P in enumerate(PARAMS_LISTA):
                    if KEY in P:
                        VAL = P[KEY]
                        if isinstance(VAL, (int, float)) and math.isfinite(VAL):
                            VALORES.append(float(VAL))
                            SCORES_FILTRADOS.append(SCORES_LISTA[I])

                if len(VALORES) < 10:
                    continue

                V_ARRAY = np.array(VALORES)
                S_ARRAY = np.array(SCORES_FILTRADOS)

                V_STD = np.std(V_ARRAY)
                S_STD = np.std(S_ARRAY)

                if V_STD < 1e-10 or S_STD < 1e-10:
                    IMPORTANCIAS[KEY] = CFG.FANOVA_PESO_MINIMO
                    continue

                CORR = float(np.corrcoef(V_ARRAY, S_ARRAY)[0, 1])
                if not math.isfinite(CORR):
                    IMPORTANCIAS[KEY] = CFG.FANOVA_PESO_MINIMO
                    continue

                # IMPORTANCIA = CORRELACIÓN AL CUADRADO
                IMPORTANCIAS[KEY] = CORR ** 2

        # APLICAR CLIPPING Y NORMALIZAR
        if not IMPORTANCIAS:
            return {}

        for KEY in IMPORTANCIAS:
            IMPORTANCIAS[KEY] = max(CFG.FANOVA_PESO_MINIMO,
                                    min(CFG.FANOVA_PESO_MAXIMO, IMPORTANCIAS[KEY]))

        SUMA = sum(IMPORTANCIAS.values())
        if SUMA > 0:
            for KEY in IMPORTANCIAS:
                IMPORTANCIAS[KEY] /= SUMA

        return IMPORTANCIAS

    def _CALCULAR_DISTANCIA_PONDERADA(
        self,
        PARAMS_A: Dict[str, Any],
        PARAMS_B: Dict[str, Any],
        BOUNDS: Dict[str, Tuple[float, float]],
        PESOS: Dict[str, float],
    ) -> float:
        """
        CALCULA LA DISTANCIA PONDERADA ENTRE DOS VECTORES DE HIPERPARÁMETROS.

        IMPLEMENTA UNA DISTANCIA DE GOWER MODIFICADA QUE COMBINA:
          - DISTANCIA EUCLIDIANA NORMALIZADA PARA PARÁMETROS NUMÉRICOS
          - DISTANCIA DE HAMMING PARA PARÁMETROS CATEGÓRICOS
          - PONDERACIÓN POR IMPORTANCIA (fANOVA) DE CADA DIMENSIÓN

        FÓRMULA:
            D(a, b) = √( Σ wᵢ × dᵢ(aᵢ, bᵢ)² )

        DONDE:
            dᵢ = |aᵢ_norm - bᵢ_norm| PARA NUMÉRICOS (NORMALIZADO A [0,1])
            dᵢ = 0 SI aᵢ == bᵢ, 1 SI aᵢ ≠ bᵢ PARA CATEGÓRICOS
            wᵢ = PESO DE IMPORTANCIA DE LA DIMENSIÓN i

        ARGS:
            PARAMS_A: PARÁMETROS DEL TRIAL ACTUAL
            PARAMS_B: PARÁMETROS DE UN TRIAL PREVIO
            BOUNDS: LÍMITES {param: (min, max)} PARA NORMALIZACIÓN
            PESOS: PESOS DE IMPORTANCIA {param: peso}

        RETURNS:
            DISTANCIA PONDERADA ≥ 0
        """
        TODAS_LAS_KEYS = set(PARAMS_A.keys()) | set(PARAMS_B.keys())
        # FILTRAR PARÁMETROS INTERNOS (QUE EMPIEZAN CON __)
        KEYS_VALIDAS = [K for K in TODAS_LAS_KEYS if not K.startswith("__")]

        if not KEYS_VALIDAS:
            return float("inf")

        SUMA_DISTANCIAS = 0.0
        SUMA_PESOS = 0.0

        for KEY in KEYS_VALIDAS:
            # OBTENER VALORES (DEFAULT = NONE SI NO EXISTE EN ESE TRIAL)
            VAL_A = PARAMS_A.get(KEY, None)
            VAL_B = PARAMS_B.get(KEY, None)

            # SI ALGUNO NO TIENE EL PARÁMETRO, DISTANCIA MÁXIMA EN ESA DIMENSIÓN
            if VAL_A is None or VAL_B is None:
                DIST_I = 1.0
            elif isinstance(VAL_A, (int, float)) and isinstance(VAL_B, (int, float)):
                # ─────────────────────────────────────────────────────────────
                # PARÁMETROS NUMÉRICOS: DISTANCIA EUCLIDIANA NORMALIZADA
                # ─────────────────────────────────────────────────────────────
                if KEY in BOUNDS:
                    BMIN, BMAX = BOUNDS[KEY]
                    RANGO = BMAX - BMIN
                    if RANGO < 1e-10:
                        DIST_I = 0.0
                    else:
                        A_NORM = (float(VAL_A) - BMIN) / RANGO
                        B_NORM = (float(VAL_B) - BMIN) / RANGO
                        DIST_I = abs(A_NORM - B_NORM)
                else:
                    # SIN BOUNDS: USAR DIFERENCIA ABSOLUTA NORMALIZADA
                    DIFF = abs(float(VAL_A) - float(VAL_B))
                    SCALE = max(abs(float(VAL_A)), abs(float(VAL_B)), 1.0)
                    DIST_I = DIFF / SCALE
            else:
                # ─────────────────────────────────────────────────────────────
                # PARÁMETROS CATEGÓRICOS: DISTANCIA DE HAMMING
                # ─────────────────────────────────────────────────────────────
                DIST_I = 0.0 if VAL_A == VAL_B else 1.0

            # OBTENER PESO PARA ESTA DIMENSIÓN
            PESO_I = PESOS.get(KEY, 1.0 / max(1, len(KEYS_VALIDAS)))

            SUMA_DISTANCIAS += PESO_I * (DIST_I ** 2)
            SUMA_PESOS += PESO_I

        if SUMA_PESOS < 1e-10:
            return float("inf")

        # RAÍZ CUADRADA DE LA SUMA PONDERADA
        DISTANCIA = math.sqrt(SUMA_DISTANCIAS)
        return DISTANCIA

    def _ENCONTRAR_K_VECINOS(
        self,
        PARAMS_ACTUAL: Dict[str, Any],
        PARAMS_LISTA: List[Dict[str, Any]],
        SCORES_LISTA: List[float],
        K: int,
        BOUNDS: Dict[str, Tuple[float, float]],
        PESOS: Dict[str, float],
    ) -> Tuple[List[float], List[float]]:
        """
        ENCUENTRA LOS K VECINOS MÁS CERCANOS EN EL ESPACIO DE HIPERPARÁMETROS.

        IMPLEMENTA k-NN USANDO LA DISTANCIA PONDERADA GOWER MODIFICADA.

        PARA CADA TRIAL PREVIO:
          1. CALCULA LA DISTANCIA PONDERADA AL TRIAL ACTUAL
          2. ORDENA POR DISTANCIA ASCENDENTE
          3. SELECCIONA LOS K MÁS CERCANOS

        ARGS:
            PARAMS_ACTUAL: PARÁMETROS DEL TRIAL ACTUAL
            PARAMS_LISTA: LISTA DE PARÁMETROS DE TRIALS PREVIOS
            SCORES_LISTA: SCORES DE TRIALS PREVIOS
            K: NÚMERO DE VECINOS A ENCONTRAR
            BOUNDS: LÍMITES PARA NORMALIZACIÓN
            PESOS: PESOS DE IMPORTANCIA POR DIMENSIÓN

        RETURNS:
            TUPLA:
              - DISTANCIAS_VECINOS: DISTANCIAS A LOS K VECINOS MÁS CERCANOS
              - SCORES_VECINOS: SCORES DE LOS K VECINOS MÁS CERCANOS
        """
        if not PARAMS_LISTA or not SCORES_LISTA:
            return [], []

        # CALCULAR DISTANCIA A CADA TRIAL PREVIO
        DISTANCIAS_Y_SCORES: List[Tuple[float, float]] = []

        for I, PARAMS_PREVIO in enumerate(PARAMS_LISTA):
            DIST = self._CALCULAR_DISTANCIA_PONDERADA(
                PARAMS_ACTUAL, PARAMS_PREVIO, BOUNDS, PESOS
            )
            if math.isfinite(DIST):
                DISTANCIAS_Y_SCORES.append((DIST, SCORES_LISTA[I]))

        if not DISTANCIAS_Y_SCORES:
            return [], []

        # ORDENAR POR DISTANCIA ASCENDENTE
        DISTANCIAS_Y_SCORES.sort(key=lambda X: X[0])

        # SELECCIONAR K MÁS CERCANOS
        K_REAL = min(K, len(DISTANCIAS_Y_SCORES))
        SELECCIONADOS = DISTANCIAS_Y_SCORES[:K_REAL]

        DISTANCIAS_VECINOS = [X[0] for X in SELECCIONADOS]
        SCORES_VECINOS = [X[1] for X in SELECCIONADOS]

        return DISTANCIAS_VECINOS, SCORES_VECINOS

    def _CALCULAR_PENALIZACION_TOPOLOGICA(
        self,
        SCORE_ACTUAL: float,
        DISTANCIAS_VECINOS: List[float],
        SCORES_VECINOS: List[float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        CALCULA LA PENALIZACIÓN TOPOLÓGICA BASADA EN LA VECINDAD k-NN.

        ESTE ES EL CORAZÓN DEL INTERCEPTOR DE IA.
        DETERMINA SI EL SCORE ACTUAL ES UN "PICO AGUDO" (OVERFITTING)
        O UNA "MESETA ESTABLE" (ROBUSTEZ).

        ESTADÍSTICAS CALCULADAS SOBRE LA VECINDAD:
          1. MEDIA DE SCORES VECINOS
          2. VARIANZA DE SCORES VECINOS
          3. COEFICIENTE DE VARIACIÓN (CV = σ/μ)
          4. RATIO DE DEGRADACIÓN (SCORE_ACTUAL / MEDIA_VECINOS)
          5. GRADIENTE LOCAL (PENDIENTE SCORE vs DISTANCIA)

        CRITERIOS DE PENALIZACIÓN:
          - CV ALTO → REGIÓN INESTABLE → PENALIZAR
          - RATIO DE DEGRADACIÓN ALTO → PICO AISLADO → PENALIZAR
          - GRADIENTE MUY EMPINADO → PICO AGUDO → PENALIZAR

        ARGS:
            SCORE_ACTUAL: GT-SCORE BASE DEL TRIAL ACTUAL
            DISTANCIAS_VECINOS: DISTANCIAS A LOS K VECINOS
            SCORES_VECINOS: SCORES DE LOS K VECINOS

        RETURNS:
            TUPLA:
              - PENALIZACION: VALOR EN [0, PENALIZACION_MAX] QUE SE RESTA
              - DESGLOSE: DICCIONARIO CON DETALLES DE LA PENALIZACIÓN
        """
        CFG = self.config

        if not SCORES_VECINOS or len(SCORES_VECINOS) < 2:
            return 0.0, {"MOTIVO": "VECINOS_INSUFICIENTES", "PENALIZACION": 0.0}

        # =====================================================================
        # ESTADÍSTICAS DE LA VECINDAD
        # =====================================================================
        SCORES_ARR = np.array(SCORES_VECINOS, dtype=np.float64)
        MEDIA_VECINOS = float(np.mean(SCORES_ARR))
        STD_VECINOS = float(np.std(SCORES_ARR))
        MIN_VECINOS = float(np.min(SCORES_ARR))
        MAX_VECINOS = float(np.max(SCORES_ARR))

        # =====================================================================
        # 1. COEFICIENTE DE VARIACIÓN (CV)
        # =====================================================================
        # CV ALTO → LOS VECINOS TIENEN SCORES MUY DISPARES → REGIÓN INESTABLE
        if abs(MEDIA_VECINOS) > 1e-10:
            CV = STD_VECINOS / abs(MEDIA_VECINOS)
        else:
            CV = STD_VECINOS

        # =====================================================================
        # 2. RATIO DE DEGRADACIÓN
        # =====================================================================
        # SI EL SCORE ACTUAL ES MUCHO MAYOR QUE SUS VECINOS,
        # ES PROBABLE QUE SEA UN PICO AISLADO (OVERFITTING)
        if MEDIA_VECINOS > 1e-10:
            RATIO_DEGRADACION = SCORE_ACTUAL / MEDIA_VECINOS
        else:
            RATIO_DEGRADACION = 1.0

        # =====================================================================
        # 3. DESVIACIÓN MEDIA RESPECTO AL ACTUAL
        # =====================================================================
        # ¿CUÁNTO SE DESVÍAN LOS VECINOS DEL SCORE ACTUAL?
        DESVIACION_MEDIA = float(np.mean(np.abs(SCORES_ARR - SCORE_ACTUAL)))
        if SCORE_ACTUAL > 1e-10:
            DESVIACION_RELATIVA = DESVIACION_MEDIA / SCORE_ACTUAL
        else:
            DESVIACION_RELATIVA = 0.0

        # =====================================================================
        # 4. GRADIENTE LOCAL (PENDIENTE SCORE vs DISTANCIA)
        # =====================================================================
        # UN GRADIENTE MUY EMPINADO INDICA QUE PEQUEÑOS CAMBIOS EN PARAMS
        # CAUSAN GRANDES CAMBIOS EN SCORE → PICO AGUDO
        GRADIENTE = 0.0
        if len(DISTANCIAS_VECINOS) >= 2:
            DIST_ARR = np.array(DISTANCIAS_VECINOS)
            if np.std(DIST_ARR) > 1e-10:
                CORR_DIST_SCORE = np.corrcoef(DIST_ARR, SCORES_ARR)[0, 1]
                if math.isfinite(CORR_DIST_SCORE):
                    # GRADIENTE NEGATIVO = SCORE CAE CON LA DISTANCIA = PICO
                    GRADIENTE = -CORR_DIST_SCORE

        # =====================================================================
        # CALCULAR PENALIZACIÓN COMPUESTA
        # =====================================================================
        PENALIZACION = 0.0

        # COMPONENTE 1: INESTABILIDAD POR CV (40% DEL PESO)
        if CV > CFG.INTERCEPTOR_CV_UMBRAL_PICO:
            CV_EXCESO = (CV - CFG.INTERCEPTOR_CV_UMBRAL_PICO) / CFG.INTERCEPTOR_CV_UMBRAL_PICO
            PENALIZACION += 0.40 * min(1.0, CV_EXCESO)

        # COMPONENTE 2: PICO AISLADO POR RATIO DE DEGRADACIÓN (30% DEL PESO)
        if RATIO_DEGRADACION > CFG.INTERCEPTOR_DEGRADACION_UMBRAL:
            RATIO_EXCESO = (RATIO_DEGRADACION - CFG.INTERCEPTOR_DEGRADACION_UMBRAL)
            PENALIZACION += 0.30 * min(1.0, RATIO_EXCESO)

        # COMPONENTE 3: GRADIENTE EMPINADO (20% DEL PESO)
        if GRADIENTE > 0.3:
            PENALIZACION += 0.20 * min(1.0, (GRADIENTE - 0.3) / 0.7)

        # COMPONENTE 4: DESVIACIÓN RELATIVA ALTA (10% DEL PESO)
        if DESVIACION_RELATIVA > 0.5:
            PENALIZACION += 0.10 * min(1.0, (DESVIACION_RELATIVA - 0.5) / 0.5)

        # ESCALAR AL MÁXIMO PERMITIDO
        PENALIZACION = PENALIZACION * CFG.INTERCEPTOR_PENALIZACION_MAX
        PENALIZACION = float(np.clip(PENALIZACION, 0.0, CFG.INTERCEPTOR_PENALIZACION_MAX))

        # =====================================================================
        # BONUS: SI LA VECINDAD ES MUY ESTABLE, REDUCIR LA PENALIZACIÓN
        # =====================================================================
        # (UNA MESETA ESTABLE MERECE SER PREMIADA)
        if CV < 0.15 and DESVIACION_RELATIVA < 0.2:
            # VECINDAD MUY ESTABLE → REDUCIR PENALIZACIÓN A LA MITAD
            PENALIZACION *= 0.5

        DESGLOSE = {
            "MEDIA_VECINOS": MEDIA_VECINOS,
            "STD_VECINOS": STD_VECINOS,
            "CV_VECINOS": CV,
            "RATIO_DEGRADACION": RATIO_DEGRADACION,
            "DESVIACION_RELATIVA": DESVIACION_RELATIVA,
            "GRADIENTE_LOCAL": GRADIENTE,
            "PENALIZACION_FINAL": PENALIZACION,
            "N_VECINOS_USADOS": len(SCORES_VECINOS),
            "MIN_VECINOS": MIN_VECINOS,
            "MAX_VECINOS": MAX_VECINOS,
        }

        return PENALIZACION, DESGLOSE

    # =========================================================================
    # =========================================================================
    #
    #  SECCIÓN 2.6: FUNCIÓN MAESTRA — COMPUTE SCORE
    #
    #  ORQUESTA TODO EL PIPELINE:
    #    GT_base → Interceptor Topológico → PSR → Score Final
    #
    # =========================================================================
    # =========================================================================

    def COMPUTE_SCORE(
        self,
        TRIAL: Optional[optuna.Trial],
        METRICS: Mapping[str, Any],
        RETURNS: Optional[np.ndarray] = None,
        EQUITY_CURVE: Optional[np.ndarray] = None,
    ) -> float:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │           FUNCIÓN PRINCIPAL DE SCORING GT-SCORE v1.0                │
        │                                                                     │
        │  PIPELINE COMPLETO:                                                 │
        │                                                                     │
        │  1. EXTRAER MÉTRICAS DEL BACKTEST                                  │
        │  2. APLICAR PENALIZACIONES BÁSICAS (SOFT-VETO)                     │
        │  3. CALCULAR GT-SCORE BASE (μ, ln(z), r², σ_d, Sharpe, SQN)       │
        │  4. EJECUTAR INTERCEPTOR TOPOLÓGICO (k-NN + fANOVA)               │
        │  5. CALCULAR PSR (PROBABILISTIC SHARPE RATIO)                      │
        │  6. COMBINAR: Score_final = GT_base × (1 - Penal) × PSR × Vetos   │
        │  7. ESCALAR A RANGO [1, 1000]                                      │
        │                                                                     │
        │  RANGO: [1, 1000] - NUNCA CERO ABSOLUTO                           │
        └────────────────────────────────────────────────────────────────────┘

        ARGS:
            TRIAL: OBJETO OPTUNA.TRIAL DEL TRIAL ACTUAL (PUEDE SER NONE)
            METRICS: DICCIONARIO DE MÉTRICAS DEL BACKTEST
            RETURNS: ARRAY DE RETORNOS POR TRADE (OPCIONAL, SE INTENTA
                     EXTRAER DE METRICS SI NO SE PROPORCIONA)
            EQUITY_CURVE: CURVA DE EQUITY (OPCIONAL, SE INTENTA
                          EXTRAER DE METRICS SI NO SE PROPORCIONA)

        RETURNS:
            SCORE FINAL EN RANGO [1, 1000]
        """
        CFG = self.config

        # =====================================================================
        # PASO 1: EXTRAER MÉTRICAS BASE DEL DICCIONARIO
        # =====================================================================

        # TRADES POR DÍA (BUSCAR MÚLTIPLES NOMBRES DE CLAVE)
        TRADES_DIA = self._SAFE_GET(METRICS, "trades_por_dia", 0.0)
        if TRADES_DIA == 0:
            TRADES_DIA = self._SAFE_GET(METRICS, "trades_dia", 0.0)
        if TRADES_DIA == 0:
            TRADES_DIA = self._SAFE_GET(METRICS, "trades_per_day", 0.0)

        # ─────────────────────────────────────────────────────────────────
        # HARD-KILL: SI TRADES/DÍA < 0.20 → SCORE = 0 (SIN EXCEPCIONES)
        # ─────────────────────────────────────────────────────────────────
        if TRADES_DIA < CFG.HARD_MIN_TRADES_POR_DIA:
            if TRIAL is not None:
                try:
                    TRIAL.set_user_attr("HARD_KILL", True)
                    TRIAL.set_user_attr("HARD_KILL_RAZON", f"trades_dia={TRADES_DIA:.3f} < {CFG.HARD_MIN_TRADES_POR_DIA}")
                    TRIAL.set_user_attr("SCORE_FINAL", float(CFG.SCORE_MIN))
                except Exception:
                    pass
            return float(CFG.SCORE_MIN)

        # NÚMERO TOTAL DE TRADES
        N_TRADES = int(self._SAFE_GET(METRICS, "n_trades", 0))
        if N_TRADES == 0:
            N_TRADES = int(self._SAFE_GET(METRICS, "total_trades", 0))

        # DRAWDOWN MÁXIMO
        DRAWDOWN = self._SAFE_GET(METRICS, "drawdown", 50.0)
        if DRAWDOWN == 0:
            DRAWDOWN = self._SAFE_GET(METRICS, "max_drawdown", 50.0)

        # ROI
        ROI = self._SAFE_GET(METRICS, "roi", 0.0)

        # SHARPE RATIO
        SHARPE_NOMINAL = self._SAFE_GET(METRICS, "sharpe", 0.0)
        if SHARPE_NOMINAL == 0:
            SHARPE_NOMINAL = self._SAFE_GET(METRICS, "sharpe_ratio", 0.0)

        # SQN (SYSTEM QUALITY NUMBER)
        SQN_NOMINAL = self._SAFE_GET(METRICS, "sqn", 0.0)

        # =====================================================================
        # PASO 1B: INTENTAR RECUPERAR RETURNS Y EQUITY_CURVE DESDE METRICS
        # =====================================================================
        if RETURNS is None:
            RAW_RETURNS = METRICS.get("returns", None)
            if RAW_RETURNS is None:
                RAW_RETURNS = METRICS.get("trade_returns", None)
            if RAW_RETURNS is not None:
                try:
                    RETURNS = np.asarray(RAW_RETURNS, dtype=np.float64)
                except Exception:
                    RETURNS = None

        if EQUITY_CURVE is None:
            RAW_EQUITY = METRICS.get("equity_curve", None)
            if RAW_EQUITY is None:
                RAW_EQUITY = METRICS.get("equity", None)
            if RAW_EQUITY is not None:
                try:
                    EQUITY_CURVE = np.asarray(RAW_EQUITY, dtype=np.float64)
                except Exception:
                    EQUITY_CURVE = None

        # =====================================================================
        # PASO 2: PENALIZACIONES BÁSICAS (SOFT-VETO)
        # =====================================================================
        # ESTAS PENALIZACIONES REDUCEN EL SCORE PERO NUNCA LO ELIMINAN.
        # ACTÚAN COMO FILTROS DE CALIDAD MÍNIMA.

        FACTOR_VETO = 1.0

        if TRADES_DIA < CFG.MIN_TRADES_POR_DIA:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if N_TRADES < CFG.MIN_TRADES_TOTAL:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if DRAWDOWN > CFG.MAX_DRAWDOWN_PERMITIDO:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        if ROI < CFG.MIN_ROI_PERMITIDO:
            FACTOR_VETO *= CFG.UMBRAL_FACTOR_PENALIZACION

        # FLOOR DEL FACTOR VETO (NUNCA LLEGAR A 0)
        FACTOR_VETO = max(0.02, FACTOR_VETO)

        # =====================================================================
        # PASO 3: CALCULAR GT-SCORE BASE
        # =====================================================================
        GT_BASE, DESGLOSE_GT = self._CALCULAR_GT_SCORE_BASE(
            RETURNS=RETURNS,
            EQUITY_CURVE=EQUITY_CURVE,
            SHARPE_VALOR=SHARPE_NOMINAL,
            SQN_VALOR=SQN_NOMINAL,
        )

        # =====================================================================
        # PASO 4: EJECUTAR INTERCEPTOR TOPOLÓGICO (k-NN + fANOVA)
        # =====================================================================
        PENALIZACION_TOPO = 0.0
        DESGLOSE_TOPO: Dict[str, Any] = {"INTERCEPTOR_ACTIVO": False}

        if (
            CFG.INTERCEPTOR_ENABLED
            and self.study is not None
            and TRIAL is not None
        ):
            # OBTENER HISTORIAL
            PARAMS_HIST, SCORES_HIST = self._OBTENER_HISTORIAL_TRIALS()
            N_HIST = len(PARAMS_HIST)

            if N_HIST < CFG.INTERCEPTOR_WARMUP_TRIALS:
                # FASE DE CALENTAMIENTO — CMA-ES EXPLORA LIBREMENTE
                DESGLOSE_TOPO["WARMUP"] = True
                DESGLOSE_TOPO["WARMUP_RESTANTE"] = CFG.INTERCEPTOR_WARMUP_TRIALS - N_HIST
                DESGLOSE_TOPO["WARMUP_TOTAL"] = CFG.INTERCEPTOR_WARMUP_TRIALS

            if N_HIST >= CFG.INTERCEPTOR_WARMUP_TRIALS:
                DESGLOSE_TOPO["INTERCEPTOR_ACTIVO"] = True
                DESGLOSE_TOPO["N_TRIALS_HISTORIAL"] = N_HIST

                # ─────────────────────────────────────────────────────────────
                # CALCULAR O RECUPERAR BOUNDS
                # ─────────────────────────────────────────────────────────────
                TRIAL_NUMBER = getattr(TRIAL, "number", 0)

                if (
                    self._PARAM_BOUNDS is None
                    or TRIAL_NUMBER - self._PARAM_BOUNDS_TRIAL > 10
                ):
                    self._PARAM_BOUNDS = self._CALCULAR_BOUNDS_PARAMETROS(PARAMS_HIST)
                    self._PARAM_BOUNDS_TRIAL = TRIAL_NUMBER

                # ─────────────────────────────────────────────────────────────
                # CALCULAR O RECUPERAR PESOS fANOVA
                # ─────────────────────────────────────────────────────────────
                if CFG.FANOVA_ENABLED:
                    NECESITA_RECALCULAR = (
                        self._FANOVA_PESOS is None
                        or N_HIST - self._FANOVA_CALCULADO_EN_TRIAL >= CFG.FANOVA_RECALCULATE_EVERY
                    )
                    if NECESITA_RECALCULAR and N_HIST >= CFG.FANOVA_MIN_TRIALS:
                        self._FANOVA_PESOS = self._CALCULAR_FANOVA_PESOS(
                            PARAMS_HIST, SCORES_HIST
                        )
                        self._FANOVA_CALCULADO_EN_TRIAL = N_HIST
                        DESGLOSE_TOPO["FANOVA_RECALCULADO"] = True

                PESOS_ACTIVOS = self._FANOVA_PESOS or {}

                # ─────────────────────────────────────────────────────────────
                # ENCONTRAR K VECINOS MÁS CERCANOS
                # ─────────────────────────────────────────────────────────────
                PARAMS_ACTUAL = dict(TRIAL.params) if TRIAL.params else {}

                DIST_VECINOS, SCORES_VECINOS = self._ENCONTRAR_K_VECINOS(
                    PARAMS_ACTUAL=PARAMS_ACTUAL,
                    PARAMS_LISTA=PARAMS_HIST,
                    SCORES_LISTA=SCORES_HIST,
                    K=CFG.INTERCEPTOR_K_NEIGHBORS,
                    BOUNDS=self._PARAM_BOUNDS or {},
                    PESOS=PESOS_ACTIVOS,
                )

                # ─────────────────────────────────────────────────────────────
                # CALCULAR PENALIZACIÓN TOPOLÓGICA
                # ─────────────────────────────────────────────────────────────
                # NOTA: EL SCORE ACTUAL QUE PASAMOS ES EL GT_BASE ESCALADO,
                # PARA QUE LA COMPARACIÓN CON VECINOS SEA CONSISTENTE
                SCORE_ACTUAL_ESCALADO = GT_BASE * (CFG.SCORE_MAX - CFG.SCORE_MIN)

                PENALIZACION_TOPO, DESGLOSE_PENAL = self._CALCULAR_PENALIZACION_TOPOLOGICA(
                    SCORE_ACTUAL=SCORE_ACTUAL_ESCALADO,
                    DISTANCIAS_VECINOS=DIST_VECINOS,
                    SCORES_VECINOS=SCORES_VECINOS,
                )

                DESGLOSE_TOPO.update(DESGLOSE_PENAL)
                DESGLOSE_TOPO["FANOVA_PESOS"] = PESOS_ACTIVOS

        # =====================================================================
        # PASO 5: CALCULAR PSR (PROBABILISTIC SHARPE RATIO)
        # =====================================================================
        PSR_FACTOR = 1.0

        if CFG.PSR_ENABLED and RETURNS is not None and len(RETURNS) >= CFG.PSR_MIN_TRADES:
            PSR_VAL = self._CALCULAR_PSR(RETURNS)
            # APLICAR FLOOR (SOFT-VETO)
            PSR_VAL = max(CFG.PSR_FLOOR, min(1.0, PSR_VAL))
            # COMBINAR CON PESO CONFIGURADO
            PSR_FACTOR = 1.0 - CFG.PSR_PESO_EN_FINAL * (1.0 - PSR_VAL)
        else:
            PSR_VAL = 0.5  # VALOR NEUTRAL CUANDO NO HAY SUFICIENTES DATOS

        # =====================================================================
        # PASO 6: COMBINAR TODO EN SCORE FINAL
        # =====================================================================
        #
        # FÓRMULA:
        #   SCORE_CRUDO = GT_BASE × (1 - PENALIZACIÓN_TOPOLÓGICA) × PSR_FACTOR
        #   SCORE_CON_VETO = SCORE_CRUDO × FACTOR_VETO
        #   SCORE_FINAL = ESCALAR A [SCORE_MIN, SCORE_MAX]

        SCORE_CRUDO = GT_BASE * (1.0 - PENALIZACION_TOPO) * PSR_FACTOR
        SCORE_CON_VETO = SCORE_CRUDO * FACTOR_VETO

        # ESCALAR DE [0, 1] A [SCORE_MIN, SCORE_MAX]
        SCORE_RANGO = CFG.SCORE_MAX - CFG.SCORE_MIN
        SCORE_FINAL = CFG.SCORE_MIN + SCORE_RANGO * SCORE_CON_VETO

        # GARANTIZAR RANGO ABSOLUTO
        SCORE_FINAL = max(CFG.SCORE_MIN, min(CFG.SCORE_MAX, SCORE_FINAL))

        # =====================================================================
        # PASO 7: GUARDAR ATRIBUTOS EN EL TRIAL PARA AUDITORÍA
        # =====================================================================
        if TRIAL is not None:
            try:
                # GT-SCORE BASE Y SUS COMPONENTES
                TRIAL.set_user_attr("GT_BASE", float(GT_BASE))
                for KEY, VAL in DESGLOSE_GT.items():
                    TRIAL.set_user_attr(KEY, float(VAL))

                # INTERCEPTOR TOPOLÓGICO
                TRIAL.set_user_attr("INTERCEPTOR_ACTIVO", DESGLOSE_TOPO.get("INTERCEPTOR_ACTIVO", False))
                TRIAL.set_user_attr("PENALIZACION_TOPOLOGICA", float(PENALIZACION_TOPO))
                if "MEDIA_VECINOS" in DESGLOSE_TOPO:
                    TRIAL.set_user_attr("MEDIA_VECINOS", float(DESGLOSE_TOPO["MEDIA_VECINOS"]))
                    TRIAL.set_user_attr("CV_VECINOS", float(DESGLOSE_TOPO["CV_VECINOS"]))
                    TRIAL.set_user_attr("RATIO_DEGRADACION", float(DESGLOSE_TOPO["RATIO_DEGRADACION"]))
                    TRIAL.set_user_attr("GRADIENTE_LOCAL", float(DESGLOSE_TOPO["GRADIENTE_LOCAL"]))
                    TRIAL.set_user_attr("N_VECINOS_USADOS", int(DESGLOSE_TOPO["N_VECINOS_USADOS"]))

                # PSR
                TRIAL.set_user_attr("PSR", float(PSR_VAL))
                TRIAL.set_user_attr("PSR_FACTOR", float(PSR_FACTOR))

                # VETOS Y FINAL
                TRIAL.set_user_attr("FACTOR_VETO", float(FACTOR_VETO))
                TRIAL.set_user_attr("SCORE_FINAL", float(SCORE_FINAL))
                TRIAL.set_user_attr("SR_NOMINAL", float(SHARPE_NOMINAL))
            except Exception:
                pass

        return float(SCORE_FINAL)


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 3: CONFIGURACIÓN DEL OPTIMIZADOR GT                          ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

@dataclass
class GTOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │              CONFIGURACIÓN DEL OPTIMIZADOR GT v1.0                      │
    │                                                                         │
    │  MOTOR DE BÚSQUEDA: CMA-ES (MODULAR — SE PUEDE CAMBIAR)               │
    │  SCORING: GT-SCORE + INTERCEPTOR TOPOLÓGICO                            │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # =========================================================================
    # 3.1 CONFIGURACIÓN DE OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # SEMILLA ALEATORIA (NONE = VARIEDAD)
    N_JOBS: int = 1                       # WORKERS PARALELOS (1 = SECUENCIAL)
    STORAGE: Optional[str] = None         # NONE = EJECUCIÓN EN RAM
    STUDY_NAME_PREFIX: str = "MODELOX"    # PREFIJO PARA NOMBRES DE ESTUDIO

    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA DEL MOTOR DE BÚSQUEDA
    # =========================================================================
    # POR DEFECTO USAMOS CMA-ES PERO EL SISTEMA ES MODULAR:
    # PARA CAMBIAR A OTRO SAMPLER, MODIFICAR SOLO _create_sampler()

    N_STARTUP_TRIALS: int = 15            # TRIALS ALEATORIOS INICIALES
    #  (MÁS QUE CMA PURO PORQUE EL INTERCEPTOR TOPOLÓGICO NECESITA
    #   SUFICIENTE HISTORIAL PARA FUNCIONAR CORRECTAMENTE)

    WARN_INDEPENDENT_SAMPLING: bool = False
    CONSIDER_PRUNED_TRIALS: bool = False

    # =========================================================================
    # 3.3 SELECCIÓN DEL MOTOR DE BÚSQUEDA
    # =========================================================================
    # MOTOR DISPONIBLE: "CMA-ES" (DEFAULT), "TPE", "RANDOM"
    # ESTO PERMITE CAMBIAR EL SAMPLER SIN TOCAR EL SCORING
    MOTOR_BUSQUEDA: str = "CMA-ES"


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
GT_OPTIMIZER_CONFIG = GTOptimizerConfig()


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 4: CLASE OPTIMIZADOR GT — ORQUESTADOR PRINCIPAL              ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████

class GTOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       OPTIMIZADOR GT v1.0                                │
    │                                                                         │
    │  GT-SCORE + INTERCEPTOR DE IA TOPOLÓGICO + CMA-ES (MODULAR)            │
    │                                                                         │
    │  CARACTERÍSTICAS:                                                       │
    │    ✓ GT-SCORE COMO MÉTRICA BASE ANTI-OVERFITTING                       │
    │    ✓ INTERCEPTOR TOPOLÓGICO k-NN CON fANOVA                            │
    │    ✓ DISTANCIA GOWER MODIFICADA (NUMÉRICOS + CATEGÓRICOS)              │
    │    ✓ PSR COMO FILTRO DE SIGNIFICANCIA                                  │
    │    ✓ CMA-ES COMO MOTOR DE BÚSQUEDA (INTERCAMBIABLE)                   │
    │    ✓ AUDITORÍA COMPLETA EN CADA TRIAL                                  │
    │                                                                         │
    │  FILOSOFÍA:                                                             │
    │    PREMIAR MESETAS ESTABLES, PENALIZAR PICOS AISLADOS                  │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[GTOptimizerConfig] = None,
        scoring_config: Optional[GTScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        INICIALIZA EL OPTIMIZADOR GT.

        ARGS:
            config: CONFIGURACIÓN DE BACKTEST (SALDO, COMISIONES, SALIDAS, ETC.)
            n_trials: NÚMERO DE TRIALS A EJECUTAR
            reporters: LISTA DE REPORTERS PARA NOTIFICAR RESULTADOS
            optimizer_config: CONFIGURACIÓN DEL OPTIMIZADOR GT
            scoring_config: CONFIGURACIÓN DEL GT-SCORE
            activo: NOMBRE DEL ACTIVO (EJ: "BTC", "GOLD")
        """
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or GT_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or GT_SCORING_CONFIG
        self.activo = activo

        # ESTADO INTERNO
        self._LAST_STUDY: Optional[optuna.Study] = None
        self._SCORER: Optional[GTScorer] = None

    # =========================================================================
    # [4.1] CREAR SAMPLER — MOTOR DE BÚSQUEDA MODULAR
    # =========================================================================

    def _CREATE_SAMPLER(self) -> optuna.samplers.BaseSampler:
        """
        CREA EL SAMPLER DE OPTUNA SEGÚN LA CONFIGURACIÓN.

        EL SISTEMA ES MODULAR: PARA AÑADIR UN NUEVO MOTOR DE BÚSQUEDA,
        SIMPLEMENTE AGREGAR UN NUEVO BLOQUE elif AQUÍ.

        MOTORES DISPONIBLES:
          - "CMA-ES": COVARIANCE MATRIX ADAPTATION (DEFAULT, RECOMENDADO)
          - "TPE":    TREE-STRUCTURED PARZEN ESTIMATOR
          - "RANDOM": RANDOM SEARCH (PARA BASELINE)

        RETURNS:
            INSTANCIA DE optuna.samplers.BaseSampler
        """
        CFG = self.optimizer_config
        MOTOR = CFG.MOTOR_BUSQUEDA.upper().replace("-", "").replace("_", "")

        if MOTOR in ("CMAES", "CMA"):
            # ─────────────────────────────────────────────────────────────────
            # CMA-ES: MOTOR RECOMENDADO
            # MANTIENE DISTRIBUCIÓN N(m, σ²C) Y LA ADAPTA CADA GENERACIÓN
            # ─────────────────────────────────────────────────────────────────
            return CmaEsSampler(
                seed=CFG.SEED,
                n_startup_trials=CFG.N_STARTUP_TRIALS,
                warn_independent_sampling=CFG.WARN_INDEPENDENT_SAMPLING,
                consider_pruned_trials=CFG.CONSIDER_PRUNED_TRIALS,
            )
        elif MOTOR == "TPE":
            # ─────────────────────────────────────────────────────────────────
            # TPE: MOTOR EXPLORATORIO
            # ─────────────────────────────────────────────────────────────────
            from optuna.samplers import TPESampler
            return TPESampler(
                seed=CFG.SEED,
                n_startup_trials=CFG.N_STARTUP_TRIALS,
            )
        elif MOTOR == "RANDOM":
            # ─────────────────────────────────────────────────────────────────
            # RANDOM: BASELINE PARA COMPARACIÓN
            # ─────────────────────────────────────────────────────────────────
            from optuna.samplers import RandomSampler
            return RandomSampler(seed=CFG.SEED)
        else:
            # DEFAULT: CMA-ES
            return CmaEsSampler(
                seed=CFG.SEED,
                n_startup_trials=CFG.N_STARTUP_TRIALS,
                warn_independent_sampling=CFG.WARN_INDEPENDENT_SAMPLING,
                consider_pruned_trials=CFG.CONSIDER_PRUNED_TRIALS,
            )

    # =========================================================================
    # [4.2] CREAR ESTUDIO OPTUNA
    # =========================================================================

    def _CREATE_STUDY(self, STRATEGY_NAME: str) -> optuna.Study:
        """
        CREA UN ESTUDIO OPTUNA CON EL SAMPLER CONFIGURADO Y EL SCORER GT.

        PASOS:
          1. CONSTRUIR NOMBRE DEL ESTUDIO
          2. CREAR SAMPLER (CMA-ES POR DEFECTO)
          3. CREAR ESTUDIO OPTUNA (DIRECCIÓN: MAXIMIZAR)
          4. INICIALIZAR SCORER GT CON REFERENCIA AL ESTUDIO

        ARGS:
            STRATEGY_NAME: NOMBRE DE LA ESTRATEGIA (PARA IDENTIFICAR EL ESTUDIO)

        RETURNS:
            OBJETO OPTUNA.STUDY CONFIGURADO Y LISTO PARA OPTIMIZAR
        """
        CFG = self.optimizer_config

        # CONSTRUIR NOMBRE DEL ESTUDIO
        PARTS = [CFG.STUDY_NAME_PREFIX, "GT", str(STRATEGY_NAME)]
        if self.activo:
            PARTS.append(str(self.activo))
        STUDY_NAME = self._SLUG("_".join(PARTS))

        # CREAR SAMPLER
        SAMPLER = self._CREATE_SAMPLER()

        # CREAR ESTUDIO
        STUDY = optuna.create_study(
            direction="maximize",
            sampler=SAMPLER,
            study_name=STUDY_NAME,
            storage=CFG.STORAGE,
            load_if_exists=False,
        )

        # INICIALIZAR SCORER GT CON REFERENCIA AL ESTUDIO
        # (EL SCORER NECESITA EL ESTUDIO PARA EL INTERCEPTOR TOPOLÓGICO)
        self._SCORER = GTScorer(study=STUDY, config=self.scoring_config)

        # ─────────────────────────────────────────────────────────────────
        # WARM-UP DINÁMICO: 15% DE N_TRIALS
        # DURANTE ESTA FASE EL INTERCEPTOR TOPOLÓGICO ESTÁ DESACTIVADO,
        # PERMITIENDO A CMA-ES EXPLORAR LIBREMENTE SIN PENALIZACIÓN.
        # ─────────────────────────────────────────────────────────────────
        WARMUP_DINAMICO = max(10, int(self.scoring_config.INTERCEPTOR_WARMUP_PCT * self.n_trials))
        self._SCORER.config.INTERCEPTOR_WARMUP_TRIALS = WARMUP_DINAMICO

        return STUDY

    @staticmethod
    def _SLUG(s: str) -> str:
        """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO OPTUNA."""
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")[:60]

    # =========================================================================
    # [4.3] PREPARAR PARÁMETROS
    # =========================================================================

    def _PREPARE_PARAMS(
        self,
        TRIAL: optuna.Trial,
        STRATEGY: Strategy,
        BASE_TF: str,
    ) -> Dict[str, Any]:
        """
        PREPARA EL DICCIONARIO COMPLETO DE PARÁMETROS PARA UN TRIAL.

        COMBINA:
          - PARÁMETROS DE LA ESTRATEGIA (suggest_params)
          - CONFIGURACIÓN DE BACKTEST (__saldo_inicial, __comision_pct, etc.)
          - CONFIGURACIÓN DE SALIDAS (exit_type, sl_pct, tp_pct, etc.)
          - TIMEFRAMES (entry, exit)
          - QTY_MAX_ACTIVO (OPTIMIZABLE O FIJO)

        ARGS:
            TRIAL: TRIAL ACTUAL DE OPTUNA
            STRATEGY: ESTRATEGIA QUE DEFINE LOS PARÁMETROS A OPTIMIZAR
            BASE_TF: TIMEFRAME BASE (EJ: "1m", "5m")

        RETURNS:
            DICCIONARIO COMPLETO DE PARÁMETROS LISTO PARA BACKTEST
        """
        PARAMS_PUROS = STRATEGY.suggest_params(TRIAL)
        PARAMS_RT = dict(PARAMS_PUROS)

        # INYECTAR VALORES DE CONFIGURACIÓN DE BACKTEST
        PARAMS_RT["__activo"] = self.activo
        PARAMS_RT["__saldo_inicial"] = float(self.config.saldo_inicial)
        PARAMS_RT["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)


        PARAMS_RT["__comision_pct"] = float(self.config.comision_pct)
        PARAMS_RT["__comision_sides"] = int(self.config.comision_sides)
        PARAMS_RT["__saldo_usado"] = float(self.config.saldo_usado)
        PARAMS_RT["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        PARAMS_RT["__strategy_exit_enabled"] = bool(
            getattr(STRATEGY, "SALIDAS_PERSONALIZADAS", False)
        )

        # RESOLVER CONFIGURACIÓN DE SALIDA
        EXIT_SETTINGS = resolve_exit_settings_for_trial(trial=TRIAL, config=self.config)
        PARAMS_RT["__exit_type"] = EXIT_SETTINGS.exit_type
        PARAMS_RT["__exit_sl_pct"] = EXIT_SETTINGS.sl_pct
        PARAMS_RT["__exit_tp_pct"] = EXIT_SETTINGS.tp_pct
        PARAMS_RT["__exit_trail_act_pct"] = EXIT_SETTINGS.trail_act_pct
        PARAMS_RT["__exit_trail_dist_pct"] = EXIT_SETTINGS.trail_dist_pct

        # ALIASES PARA COMPATIBILIDAD
        PARAMS_RT["exit_type"] = EXIT_SETTINGS.exit_type
        PARAMS_RT["exit_sl_pct"] = EXIT_SETTINGS.sl_pct
        PARAMS_RT["exit_tp_pct"] = EXIT_SETTINGS.tp_pct
        PARAMS_RT["exit_trail_act_pct"] = EXIT_SETTINGS.trail_act_pct
        PARAMS_RT["exit_trail_dist_pct"] = EXIT_SETTINGS.trail_dist_pct

        # TIMEFRAMES
        ENTRY_TF = normalize_timeframe_to_suffix(
            getattr(STRATEGY, "timeframe_entry", None) or BASE_TF
        )
        EXIT_TF = normalize_timeframe_to_suffix(
            getattr(STRATEGY, "timeframe_exit", None) or BASE_TF
        )
        PARAMS_RT["__timeframe_base"] = BASE_TF
        PARAMS_RT["__timeframe_entry"] = ENTRY_TF
        PARAMS_RT["__timeframe_exit"] = EXIT_TF

        return PARAMS_RT

    # =========================================================================
    # [4.4] FUNCIÓN OBJETIVO — EL CORAZÓN DEL OPTIMIZER
    # =========================================================================

    def _CREATE_OBJECTIVE(
        self,
        DF_BASE: pl.DataFrame,
        DF_MAP: Dict[str, pl.DataFrame],
        STRATEGY: Strategy,
        BASE_TF: str,
    ) -> Callable[[optuna.Trial], float]:
        """
        CREA LA FUNCIÓN OBJETIVO QUE OPTUNA LLAMA EN CADA TRIAL.

        ESTA FUNCIÓN ENCAPSULA TODO EL PIPELINE:
          1. PREPARAR PARÁMETROS (CON suggest_params DE LA ESTRATEGIA)
          2. GENERAR SEÑALES DE TRADING
          3. EJECUTAR BACKTEST
          4. CALCULAR MÉTRICAS
          5. CALCULAR GT-SCORE (CON INTERCEPTOR TOPOLÓGICO)
          6. CREAR ARTIFACTS PARA REPORTERS
          7. RETORNAR SCORE A OPTUNA

        ARGS:
            DF_BASE: DATAFRAME CON DATOS OHLCV BASE
            DF_MAP: DICT CON DATAFRAMES POR TIMEFRAME
            STRATEGY: ESTRATEGIA A OPTIMIZAR
            BASE_TF: TIMEFRAME BASE

        RETURNS:
            FUNCIÓN objective(trial) → float
        """
        # IMPORTAR COMPONENTES NECESARIOS (LAZY IMPORT)
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup

        def OBJECTIVE(TRIAL: optuna.Trial) -> float:
            """
            FUNCIÓN OBJETIVO PARA CADA TRIAL.

            ESTA FUNCIÓN SE EJECUTA UNA VEZ POR CADA TRIAL DE OPTUNA.
            EL VALOR RETORNADO ES EL GT-SCORE FINAL (CON PENALIZACIONES).
            """
            T0_TOTAL = time.perf_counter()

            # ─────────────────────────────────────────────────────────────────
            # LIMPIEZA PERIÓDICA DE MEMORIA
            # ─────────────────────────────────────────────────────────────────
            periodic_cleanup(TRIAL.number)

            # ─────────────────────────────────────────────────────────────────
            # PREPARAR PARÁMETROS
            # ─────────────────────────────────────────────────────────────────
            PARAMS_RT = self._PREPARE_PARAMS(TRIAL, STRATEGY, BASE_TF)
            ENTRY_TF = PARAMS_RT["__timeframe_entry"]
            DF_ENTRY = DF_MAP.get(ENTRY_TF, DF_BASE)

            # ─────────────────────────────────────────────────────────────────
            # GENERAR SEÑALES DE TRADING
            # ─────────────────────────────────────────────────────────────────
            SIGNALS_DF = SignalGenerator.generate_signals(
                DF_ENTRY, STRATEGY, PARAMS_RT, DF_MAP
            )

            # ─────────────────────────────────────────────────────────────────
            # EJECUTAR BACKTEST
            # ─────────────────────────────────────────────────────────────────
            TRADES_DF, EQUITY_CURVE, METRICS = BacktestEngine.run_backtest(
                DF_ENTRY, SIGNALS_DF, self.config, PARAMS_RT, STRATEGY,
            )

            # ─────────────────────────────────────────────────────────────────
            # VERIFICAR QUE HAY TRADES
            # ─────────────────────────────────────────────────────────────────
            if TRADES_DF.is_empty():
                return 0.0

            TRIAL.set_user_attr("metricas", METRICS)

            # ─────────────────────────────────────────────────────────────────
            # CALCULAR GT-SCORE CON INTERCEPTOR TOPOLÓGICO
            # ─────────────────────────────────────────────────────────────────
            SCORE = self._SCORER.COMPUTE_SCORE(
                TRIAL=TRIAL,
                METRICS=METRICS,
                EQUITY_CURVE=np.array(EQUITY_CURVE) if EQUITY_CURVE else None,
            )

            # ─────────────────────────────────────────────────────────────────
            # CREAR ARTIFACTS PARA REPORTERS
            # ─────────────────────────────────────────────────────────────────
            ARTIFACTS = TrialArtifacts(
                strategy_name=STRATEGY.name,
                trial_number=TRIAL.number,
                params=PARAMS_RT,
                params_reporting=PARAMS_RT,
                score=SCORE,
                metrics=METRICS,
                df_signals=None,
                trades=TRADES_DF.to_pandas(),
                equity_curve=EQUITY_CURVE,
                indicators_used=PARAMS_RT.get("__indicators_used", []),
            )

            # NOTIFICAR A REPORTERS
            for REPORTER in self.reporters:
                REPORTER.on_trial_end(ARTIFACTS)

            return SCORE

        return OBJECTIVE

    # =========================================================================
    # [4.5] OPTIMIZAR — PUNTO DE ENTRADA PRINCIPAL
    # =========================================================================

    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.Study:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │              EJECUTAR OPTIMIZACIÓN GT-SCORE                         │
        │                                                                     │
        │  MOTOR: CMA-ES (MODULAR)                                           │
        │  SCORING: GT-SCORE + INTERCEPTOR TOPOLÓGICO k-NN + fANOVA          │
        │  OBJETIVO: MAXIMIZAR ROBUSTEZ, ELIMINAR SOBREAJUSTE                │
        └────────────────────────────────────────────────────────────────────┘

        ARGS:
            df: DATAFRAME CON DATOS OHLCV (POLARS)
            strategy: ESTRATEGIA A OPTIMIZAR (DEBE IMPLEMENTAR Strategy)
            df_by_timeframe: DICT {timeframe: DataFrame} PARA MULTI-TF
            base_timeframe: TIMEFRAME BASE (DEFAULT: "1m")

        RETURNS:
            OBJETO OPTUNA.STUDY CON TODOS LOS RESULTADOS Y ATRIBUTOS
        """
        BASE_TF = base_timeframe or "1m"
        DF_MAP = df_by_timeframe or {BASE_TF: df}
        DF_BASE = DF_MAP.get(BASE_TF, df)

        # CREAR ESTUDIO CON SAMPLER Y SCORER
        STUDY = self._CREATE_STUDY(strategy.name)

        # CREAR FUNCIÓN OBJETIVO
        OBJECTIVE = self._CREATE_OBJECTIVE(DF_BASE, DF_MAP, strategy, BASE_TF)

        # EJECUTAR OPTIMIZACIÓN
        STUDY.optimize(
            OBJECTIVE,
            n_trials=int(self.n_trials),
            n_jobs=int(self.optimizer_config.N_JOBS),
            gc_after_trial=True,
            catch=(Exception,),
        )

        self._LAST_STUDY = STUDY
        return STUDY

    # =========================================================================
    # [4.6] PROPIEDADES
    # =========================================================================

    @property
    def last_study(self) -> Optional[optuna.Study]:
        """RETORNA EL ÚLTIMO ESTUDIO EJECUTADO."""
        return self._LAST_STUDY

    @property
    def scorer(self) -> Optional[GTScorer]:
        """RETORNA EL SCORER GT UTILIZADO."""
        return self._SCORER


# █████████████████████████████████████████████████████████████████████████████
# ██                                                                         ██
# ██   SECCIÓN 5: FUNCIONES DE UTILIDAD — STANDALONE                         ██
# ██                                                                         ██
# █████████████████████████████████████████████████████████████████████████████


def _SLUG(s: str) -> str:
    """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_gt_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    n_startup_trials: int = 15,
    storage: Optional[str] = None,
    motor_busqueda: str = "CMA-ES",
) -> optuna.Study:
    """
    CREA UN ESTUDIO OPTUNA CON SCORING GT-SCORE.

    FUNCIÓN DE FÁBRICA STANDALONE PARA CREAR UN ESTUDIO OPTUNA
    CONFIGURADO CON EL MOTOR DE BÚSQUEDA DESEADO.
    POR DEFECTO USA CMA-ES PERO ES MODULAR.

    ARGS:
        strategy_name: NOMBRE DE LA ESTRATEGIA
        activo: NOMBRE DEL ACTIVO (OPCIONAL)
        seed: SEMILLA ALEATORIA (NONE = VARIEDAD)
        study_name_prefix: PREFIJO PARA EL NOMBRE DEL ESTUDIO
        n_startup_trials: TRIALS ALEATORIOS INICIALES
        storage: URI DE ALMACENAMIENTO (NONE = RAM)
        motor_busqueda: "CMA-ES" (DEFAULT), "TPE", "RANDOM"

    RETURNS:
        OPTUNA.STUDY CONFIGURADO CON GT-SCORE Y EL SAMPLER INDICADO
    """
    # CONSTRUIR NOMBRE
    PARTS = [study_name_prefix, "GT", str(strategy_name)]
    if activo:
        PARTS.append(str(activo))
    STUDY_NAME = _SLUG("_".join(PARTS))

    # CREAR SAMPLER SEGÚN MOTOR ELEGIDO
    MOTOR = motor_busqueda.upper().replace("-", "").replace("_", "")

    if MOTOR in ("CMAES", "CMA"):
        SAMPLER = CmaEsSampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            warn_independent_sampling=False,
            consider_pruned_trials=False,
        )
    elif MOTOR == "TPE":
        from optuna.samplers import TPESampler
        SAMPLER = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    elif MOTOR == "RANDOM":
        from optuna.samplers import RandomSampler
        SAMPLER = RandomSampler(seed=seed)
    else:
        SAMPLER = CmaEsSampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            warn_independent_sampling=False,
            consider_pruned_trials=False,
        )

    # CREAR ESTUDIO
    STUDY = optuna.create_study(
        direction="maximize",
        sampler=SAMPLER,
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=False,
    )

    return STUDY


def score_gt(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    FUNCIÓN DE SCORING GT-SCORE STANDALONE.

    CALCULA EL GT-SCORE SIN NECESIDAD DE UN OPTIMIZADOR COMPLETO.
    ÚTIL PARA EVALUAR MÉTRICAS DE FORMA INDEPENDIENTE.

    NOTA: SIN UN ESTUDIO OPTUNA, EL INTERCEPTOR TOPOLÓGICO NO SE ACTIVA.
    SOLO SE CALCULA EL GT-SCORE BASE.

    USO:
        from modelox.optimizers.gt import score_gt

        SCORE = score_gt(METRICS)
        SCORE = score_gt(METRICS, equity_curve=EQUITY)

    ARGS:
        metrics: DICCIONARIO DE MÉTRICAS DEL BACKTEST
        trial: TRIAL DE OPTUNA (OPCIONAL)
        equity_curve: CURVA DE EQUITY (OPCIONAL)

    RETURNS:
        SCORE EN RANGO [1, 1000]
    """
    SCORER = GTScorer()
    return SCORER.COMPUTE_SCORE(
        TRIAL=trial,
        METRICS=metrics,
        EQUITY_CURVE=np.array(equity_curve) if equity_curve else None,
    )


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # CLASE OPTIMIZADOR
    "GTOptimizer",
    # CONFIGURACIONES
    "GTOptimizerConfig",
    "GTScoringConfig",
    # INSTANCIAS DEFAULT
    "GT_SCORING_CONFIG",
    "GT_OPTIMIZER_CONFIG",
    # CLASE SCORER
    "GTScorer",
    # FUNCIONES STANDALONE
    "create_gt_study",
    "score_gt",
]

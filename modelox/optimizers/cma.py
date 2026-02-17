"""modelox/optimizers/cma.py

═══════════════════════════════════════════════════════════════════════════════
    ██████╗███╗   ███╗ █████╗       ███████╗███████╗
   ██╔════╝████╗ ████║██╔══██╗      ██╔════╝██╔════╝
   ██║     ██╔████╔██║███████║█████╗█████╗  ███████╗
   ██║     ██║╚██╔╝██║██╔══██║╚════╝██╔══╝  ╚════██║
   ╚██████╗██║ ╚═╝ ██║██║  ██║      ███████╗███████║
    ╚═════╝╚═╝     ╚═╝╚═╝  ╚═╝      ╚══════╝╚══════╝
    
    COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
CMA-ES es un algoritmo evolutivo que adapta la matriz de covarianza para
explorar el espacio de parámetros de forma inteligente.

VENTAJAS:
=========
  ✓ Aprende de los scores para adaptar la búsqueda
  ✓ Converge hacia regiones de alta calidad
  ✓ Ideal para encontrar "mesetas de parámetros" (soluciones robustas)
  ✓ Penaliza implícitamente los picos aislados (overfitting)
  ✓ RECOMENDADO para trading cuantitativo

FILOSOFÍA DEL SCORING CMA:
==========================
El scoring de CMA-ES usa una arquitectura SIGMOIDE con SOFT-VETO:
  - Score Base: Mapeo sigmoide de Sharpe → [100, 900]
  - Penalizaciones: Multiplicadores con floor mínimo (nunca colapsa a 0)
  - Rango Final: [1, 1000] - NUNCA cero absoluto

PILARES DEL SCORING:
====================
  1. PSR/DSR - Inferencia Probabilística (floor: 0.30)
  2. ESTABILIDAD - Consistencia de Vecindario (floor: 0.25)
  3. RÉGIMEN - Consistencia en Volatilidad (floor: 0.30)
  4. CURVA - Calidad K-Ratio/R² (floor: 0.20)
  5. DECAY - Factor de Exploración (floor: 0.40)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

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
# ███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
# ██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝ 
# ███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
# ╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
# ███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
# ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
#                                                         
# SISTEMA DE SCORING CMA-ES - SIGMOIDE CON SOFT-VETO
# =============================================================================


# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING CMA
# =============================================================================

@dataclass
class CMAScoringConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │              CONFIGURACIÓN DEL SCORING CMA-ES v2.0                      │
    │                                                                         │
    │  ARQUITECTURA: Sigmoide Base × Penalizaciones Soft-Veto                │
    │  RANGO SALIDA: [1, 1000] - NUNCA CERO ABSOLUTO                        │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 1.1 RANGO DE SALIDA DEL SCORE
    # =========================================================================
    SCORE_MIN: float = 1.0               # MÍNIMO ABSOLUTO (NUNCA 0)
    SCORE_MAX: float = 1000.0            # MÁXIMO ABSOLUTO
    
    # =========================================================================
    # 1.2 SIGMOIDE BASE (MAPEA SHARPE A SCORE BASE)
    # =========================================================================
    # LA SIGMOIDE TRANSFORMA EL SHARPE RATIO A UN SCORE BASE EN [100, 900]
    # FÓRMULA: sigmoid(x) = 1 / (1 + exp(-k * (x - x0)))
    #
    # MAPEO APROXIMADO:
    #   SHARPE -2  → ~50   (MUY MALO)
    #   SHARPE  0  → ~200  (MEDIOCRE)
    #   SHARPE  1  → ~500  (CENTRO)
    #   SHARPE  2  → ~800  (BUENO)
    #   SHARPE  4  → ~950  (EXCELENTE)
    
    SIGMOID_K: float = 1.5               # PENDIENTE DE LA SIGMOIDE
    SIGMOID_X0: float = 1.0              # CENTRO (SHARPE DONDE SCORE=50%)
    SIGMOID_FLOOR: float = 0.15          # MÍNIMO DE LA SIGMOIDE NORMALIZADA
    SIGMOID_CEILING: float = 0.95        # MÁXIMO DE LA SIGMOIDE NORMALIZADA
    
    # =========================================================================
    # 1.3 PILAR 1: PSR/DSR - INFERENCIA PROBABILÍSTICA
    # =========================================================================
    # PSR = PROBABILISTIC SHARPE RATIO
    # DETERMINA LA PROBABILIDAD DE QUE EL SHARPE OBSERVADO SEA REAL
    # DSR = DEFLATED SHARPE RATIO (CORRIGE POR MÚLTIPLES PRUEBAS)
    
    PSR_BENCHMARK_SR: float = 0.0        # SR DE REFERENCIA PARA PSR
    DSR_WARMUP_TRIALS: int = 30          # TRIALS ANTES DE ACTIVAR DSR
    MIN_TRADES_FOR_PSR: int = 30         # TRADES MÍNIMOS PARA PSR VÁLIDO
    PSR_FLOOR: float = 0.30              # SOFT-VETO: MÍNIMO 30%
    
    # =========================================================================
    # 1.4 PILAR 2: ESTABILIDAD DE VECINDARIO (SAM)
    # =========================================================================
    # SAM = SHARPNESS-AWARE MINIMIZATION
    # EVALÚA SI EL RENDIMIENTO ES SENSIBLE A PERTURBACIONES
    # UNA ESTRATEGIA ROBUSTA MANTIENE RENDIMIENTO SIMILAR EN VECINDAD
    
    SAM_ENABLED: bool = True             # ACTIVAR ANÁLISIS DE VECINDARIO
    SAM_RADIUS_PCT: float = 0.05         # RADIO DE PERTURBACIÓN (5% DEL RANGO)
    SAM_N_NEIGHBORS: int = 5             # NÚMERO DE PUNTOS PERTURBADOS
    SAM_MIN_STABILITY: float = 0.80      # SCORE MÍNIMO EN VECINDAD VS ORIGINAL
    SAM_FLOOR: float = 0.25              # SOFT-VETO: MÍNIMO 25%
    
    # =========================================================================
    # 1.5 PILAR 3: RÉGIMEN - CONSISTENCIA EN VOLATILIDAD
    # =========================================================================
    # DIVIDE DATOS EN 3 CUBETAS DE VOLATILIDAD (BAJA, MEDIA, ALTA)
    # MIDE SI EL DESEMPEÑO ES CONSISTENTE ENTRE ELLAS
    
    REGIME_ENABLED: bool = True
    REGIME_PERCENTILES: Tuple[float, float] = (33, 66)  # CORTES DE VOLATILIDAD
    REGIME_MIN_SAMPLES: int = 10         # MÍNIMO DE SAMPLES POR RÉGIMEN
    REGIME_FLOOR: float = 0.30           # SOFT-VETO: MÍNIMO 30%
    
    # =========================================================================
    # 1.6 PILAR 4: CURVA - CALIDAD DE EQUITY (K-RATIO Y R²)
    # =========================================================================
    # K-RATIO: MIDE CONSISTENCIA DEL CRECIMIENTO (SLOPE/SE × SQRT(N))
    # R²: MIDE LINEALIDAD DEL CRECIMIENTO DEL CAPITAL
    
    CURVE_K_RATIO_TARGET: float = 2.0    # K-RATIO OBJETIVO (> 2.0 = EXCELENTE)
    CURVE_R2_MIN: float = 0.70           # R² MÍNIMO ACEPTABLE
    CURVE_FLOOR: float = 0.20            # SOFT-VETO: MÍNIMO 20%
    
    # =========================================================================
    # 1.7 PILAR 5: DECAY - FACTOR DE EXPLORACIÓN
    # =========================================================================
    # PENALIZA DESCUBRIMIENTOS TARDÍOS PARA COMBATIR DATA SNOOPING
    # UN SR ENCONTRADO EN TRIAL #10 ES MÁS CONFIABLE QUE EN TRIAL #500
    
    DECAY_ENABLED: bool = True
    DECAY_BASE: float = math.e           # BASE DEL LOGARITMO DE DECAIMIENTO
    DECAY_FLOOR: float = 0.40            # SOFT-VETO: MÍNIMO 40%
    
    # =========================================================================
    # 1.8 UMBRALES DE PENALIZACIÓN (SOFT-VETO, NO ELIMINACIÓN)
    # =========================================================================
    # ESTOS UMBRALES APLICAN PENALIZACIONES FUERTES PERO NO ELIMINAN
    
    MIN_TRADES_PER_DAY: float = 0.10     # PENALIZACIÓN FUERTE SI < 0.10
    MIN_TOTAL_TRADES: int = 15           # PENALIZACIÓN FUERTE SI < 15
    MAX_DRAWDOWN_ALLOWED: float = 60.0   # PENALIZACIÓN FUERTE SI > 60%
    MIN_ROI: float = -150.0              # PENALIZACIÓN FUERTE SI < -150%
    
    # FACTOR DE PENALIZACIÓN PARA UMBRALES (EN LUGAR DE SUSPENSO TOTAL)
    THRESHOLD_PENALTY: float = 0.15      # MULTIPLICA POR 0.15 SI VIOLA UMBRAL


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
CMA_SCORING_CONFIG = CMAScoringConfig()


# =============================================================================
# [SECCIÓN 2] CLASE SCORER CMA-ES
# =============================================================================

class CMAScorer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    SCORER INSTITUCIONAL CMA-ES v2.0                     │
    │                                                                         │
    │  ARQUITECTURA: Score = BaseScore(sharpe) × P_psr × P_stab × P_reg ×   │
    │                        P_curve × P_decay × P_threshold                  │
    │                                                                         │
    │  FILOSOFÍA: Gradientes preservados para que CMA-ES pueda aprender      │
    │  RANGO: [1, 1000] - NUNCA CERO ABSOLUTO                               │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[CMAScoringConfig] = None,
    ):
        """
        INICIALIZA EL SCORER CMA-ES.
        
        ARGS:
            study: OBJETO OPTUNA.STUDY PARA ACCEDER AL HISTORIAL
            config: CONFIGURACIÓN PERSONALIZADA (USA DEFAULT SI NONE)
        """
        self.study = study
        self.config = config or CMA_SCORING_CONFIG
        
        # CACHE DE ESTADÍSTICAS DEL ESTUDIO
        self._cached_sr_stats: Optional[Dict[str, float]] = None
        self._cached_at_trial: int = -1
    
    # =========================================================================
    # [2.1] FUNCIÓN SIGMOIDE BASE
    # =========================================================================
    
    def _sigmoid(self, x: float, k: float = 1.5, x0: float = 1.0) -> float:
        """
        FUNCIÓN SIGMOIDE PARA MAPEAR SHARPE RATIO A [0, 1].
        
        ARGS:
            x: VALOR DE ENTRADA (SHARPE RATIO)
            k: PENDIENTE DE LA SIGMOIDE (MAYOR = MÁS EMPINADA)
            x0: CENTRO DE LA SIGMOIDE (SHARPE DONDE OUTPUT = 0.5)
        
        RETURNS:
            VALOR EN [0, 1]
        """
        try:
            exponent = -k * (x - x0)
            # PROTECCIÓN CONTRA OVERFLOW
            if exponent > 500:
                return 0.0
            elif exponent < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5
    
    def _compute_base_score(self, sharpe: float) -> float:
        """
        CALCULA EL SCORE BASE USANDO SIGMOIDE.
        
        MAPEO:
            SHARPE -2 → ~50
            SHARPE 0  → ~200  
            SHARPE 1  → ~500 (CENTRO)
            SHARPE 2  → ~800
            SHARPE 4  → ~950
        
        RETURNS:
            SCORE BASE EN RANGO [SCORE_MIN, SCORE_MAX]
        """
        cfg = self.config
        
        # SIGMOID RAW VALUE EN [0, 1]
        sig_raw = self._sigmoid(sharpe, k=cfg.SIGMOID_K, x0=cfg.SIGMOID_X0)
        
        # APLICAR FLOOR Y CEILING PARA EVITAR EXTREMOS
        sig_normalized = cfg.SIGMOID_FLOOR + (cfg.SIGMOID_CEILING - cfg.SIGMOID_FLOOR) * sig_raw
        
        # ESCALAR A RANGO [SCORE_MIN, SCORE_MAX]
        score_range = cfg.SCORE_MAX - cfg.SCORE_MIN
        base_score = cfg.SCORE_MIN + score_range * sig_normalized
        
        return float(base_score)
    
    # =========================================================================
    # [2.2] FUNCIONES AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """EXTRAE VALOR NUMÉRICO DE FORMA SEGURA."""
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
    
    @staticmethod
    def _get_moments(returns: np.ndarray) -> Dict[str, float]:
        """CALCULA LOS MOMENTOS ESTADÍSTICOS PARA PSR."""
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 3:
            return {'mean': 0.0, 'std': 1.0, 'skew': 0.0, 'kurt': 3.0}
        
        mean_val = float(np.mean(returns))
        std_val = float(np.std(returns, ddof=1))
        
        if std_val < 1e-10:
            std_val = 1e-10
        
        # CALCULAR SKEWNESS Y KURTOSIS MANUALMENTE
        n = len(returns)
        m3 = np.mean((returns - mean_val) ** 3)
        m4 = np.mean((returns - mean_val) ** 4)
        skew_val = m3 / (std_val ** 3) if std_val > 0 else 0.0
        kurt_val = m4 / (std_val ** 4) if std_val > 0 else 3.0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'skew': skew_val,
            'kurt': kurt_val
        }
    
    # =========================================================================
    # [2.3] PILAR 1: PSR (PROBABILISTIC SHARPE RATIO)
    # =========================================================================
    
    def calculate_psr(
        self,
        returns: np.ndarray,
        benchmark_sr: Optional[float] = None,
    ) -> float:
        """
        CALCULA EL PROBABILISTIC SHARPE RATIO.
        
        EL PSR DETERMINA LA PROBABILIDAD DE QUE EL SHARPE RATIO OBSERVADO
        SEA SUPERIOR A UN UMBRAL DE REFERENCIA, AJUSTANDO POR LA CALIDAD
        DE LA DISTRIBUCIÓN (SKEWNESS Y KURTOSIS).
        
        FÓRMULA:
            PSR(SR*) = Z((SR_hat - SR*) * sqrt(n-1) / sigma_sr)
            
        DONDE sigma_sr = sqrt(1 - γ3×SR + (γ4-1)/4 × SR²)
        
        RETURNS:
            PROBABILIDAD [0, 1] DE QUE EL SR REAL SEA > BENCHMARK
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        if n < self.config.MIN_TRADES_FOR_PSR:
            # MUESTRA MUY PEQUEÑA: PENALIZACIÓN SEVERA
            return 0.1
        
        m = self._get_moments(returns)
        sr = m['mean'] / m['std'] if m['std'] > 1e-10 else 0.0
        
        # SR DE REFERENCIA
        sr_star = benchmark_sr if benchmark_sr is not None else self.config.PSR_BENCHMARK_SR
        
        # ERROR ESTÁNDAR DEL ESTIMADOR SHARPE AJUSTADO
        sr_sq = sr ** 2
        variance_factor = 1 - m['skew'] * sr + ((m['kurt'] - 1) / 4) * sr_sq
        variance_factor = max(0.01, variance_factor)
        
        sigma_sr = math.sqrt(variance_factor / max(1, n - 1))
        
        if sigma_sr < 1e-10:
            return 0.9 if sr > sr_star else 0.1
        
        z_score = (sr - sr_star) / sigma_sr
        
        # CDF DE NORMAL ESTÁNDAR (APROXIMACIÓN)
        psr = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2)))
        
        return float(np.clip(psr, 0.01, 0.99))
    
    # =========================================================================
    # [2.4] PILAR 1B: DSR (DEFLATED SHARPE RATIO)
    # =========================================================================
    
    def _update_sr_cache(self) -> None:
        """ACTUALIZA EL CACHE DE ESTADÍSTICAS DE SR DEL ESTUDIO."""
        if self.study is None:
            return
        
        current_trial = len(self.study.trials)
        if current_trial == self._cached_at_trial:
            return
        
        # RECOPILAR TODOS LOS SR NOMINALES DE TRIALS COMPLETADOS
        all_srs = []
        for t in self.study.trials:
            if t.state.is_finished():
                sr = t.user_attrs.get('sr_nominal', None)
                if sr is not None and math.isfinite(sr):
                    all_srs.append(sr)
        
        if len(all_srs) < 2:
            self._cached_sr_stats = None
        else:
            self._cached_sr_stats = {
                'mean': float(np.mean(all_srs)),
                'var': float(np.var(all_srs)),
                'n': len(all_srs)
            }
        
        self._cached_at_trial = current_trial
    
    def calculate_dsr_threshold(self) -> float:
        """
        CALCULA EL UMBRAL DE SHARPE RATIO DESINFLADO.
        
        EL DSR CORRIGE EL SESGO DE SELECCIÓN CUANDO SE PRUEBAN MUCHAS
        CONFIGURACIONES. EL UMBRAL SR₀ SE ELEVA AUTOMÁTICAMENTE CON MÁS TRIALS.
        
        RETURNS:
            UMBRAL DE SR PARA EL DSR (SR₀)
        """
        self._update_sr_cache()
        
        if self._cached_sr_stats is None:
            return 0.0
        
        n_trials = self._cached_sr_stats['n']
        var_sr = self._cached_sr_stats['var']
        mean_sr = self._cached_sr_stats['mean']
        
        if n_trials < self.config.DSR_WARMUP_TRIALS:
            return 0.0
        
        if var_sr < 1e-10:
            return 0.0
        
        # CONSTANTE DE EULER-MASCHERONI
        emc = 0.5772156649
        
        # CALCULAR Z^(-1) (QUANTILES DE NORMAL ESTÁNDAR)
        p1 = max(0.001, 1 - 1 / n_trials)
        p2 = max(0.001, 1 - 1 / (n_trials * math.e))
        z1 = math.sqrt(2) * self._erfinv(2 * p1 - 1)
        z2 = math.sqrt(2) * self._erfinv(2 * p2 - 1)
        
        # MAX SR ESPERADO TRAS N PRUEBAS INDEPENDIENTES
        std_sr = math.sqrt(var_sr)
        sr0 = mean_sr + std_sr * ((1 - emc) * z1 + emc * z2)
        
        return max(0.0, sr0)
    
    @staticmethod
    def _erfinv(x: float) -> float:
        """APROXIMACIÓN DE LA FUNCIÓN INVERSA DE ERROR."""
        a = 0.147
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        if x >= 1:
            return sign * float('inf')
        
        ln_term = math.log(1 - x * x)
        term1 = (2 / (math.pi * a)) + (ln_term / 2)
        term2 = ln_term / a
        
        result = sign * math.sqrt(math.sqrt(term1 * term1 - term2) - term1)
        return result
    
    def calculate_deflated_psr(self, returns: np.ndarray) -> float:
        """CALCULA EL PSR CON UMBRAL AJUSTADO POR DSR."""
        sr_deflated = self.calculate_dsr_threshold()
        return self.calculate_psr(returns, benchmark_sr=sr_deflated)
    
    # =========================================================================
    # [2.5] PILAR 2: ESTABILIDAD (SAM)
    # =========================================================================
    
    def calculate_stability_score(
        self,
        original_score: float,
        neighbor_scores: List[float],
    ) -> float:
        """
        CALCULA EL FACTOR DE ESTABILIDAD DE PARÁMETROS (SAM).
        
        EVALÚA SI EL RENDIMIENTO ES SENSIBLE A PEQUEÑAS PERTURBACIONES.
        
        RETURNS:
            FACTOR DE ESTABILIDAD [0, 1]
        """
        if not self.config.SAM_ENABLED:
            return 1.0
        
        if not neighbor_scores or len(neighbor_scores) == 0:
            return 0.5  # SIN DATOS DE VECINDARIO: PENALIZACIÓN MODERADA
        
        if original_score <= 0:
            return 0.1
        
        min_neighbor = min(neighbor_scores)
        epsilon_safe = 1e-6
        
        # DEGRADACIÓN RELATIVA
        degradation = (original_score - min_neighbor) / (original_score + epsilon_safe)
        
        # FACTOR DE ESTABILIDAD
        stability = max(0.0, 1.0 - degradation)
        
        # PENALIZACIÓN ADICIONAL SI DEGRADACIÓN ES CATASTRÓFICA (> 50%)
        if degradation > 0.5:
            stability *= 0.5
        
        return float(np.clip(stability, 0.01, 1.0))
    
    # =========================================================================
    # [2.6] PILAR 3: RÉGIMEN (CONSISTENCIA EN VOLATILIDAD)
    # =========================================================================
    
    def calculate_regime_score(
        self,
        returns: np.ndarray,
        volatility_series: np.ndarray,
    ) -> float:
        """
        EVALÚA LA CONSISTENCIA DEL RENDIMIENTO EN DIFERENTES REGÍMENES.
        
        DIVIDE LOS DATOS EN 3 CUBETAS (BAJA, MEDIA, ALTA VOLATILIDAD)
        Y MIDE SI EL DESEMPEÑO ES CONSISTENTE ENTRE ELLAS.
        
        RETURNS:
            FACTOR DE RÉGIMEN [0, 1]
        """
        if not self.config.REGIME_ENABLED:
            return 1.0
        
        returns = np.asarray(returns, dtype=np.float64)
        volatility_series = np.asarray(volatility_series, dtype=np.float64)
        
        # ASEGURAR MISMA LONGITUD
        min_len = min(len(returns), len(volatility_series))
        if min_len < self.config.REGIME_MIN_SAMPLES * 3:
            return 0.7  # DATOS INSUFICIENTES
        
        returns = returns[:min_len]
        volatility_series = volatility_series[:min_len]
        
        # FILTRAR NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(volatility_series)
        returns = returns[valid_mask]
        volatility_series = volatility_series[valid_mask]
        
        if len(returns) < self.config.REGIME_MIN_SAMPLES * 3:
            return 0.7
        
        # CALCULAR PERCENTILES DE VOLATILIDAD
        low_t, high_t = np.percentile(volatility_series, self.config.REGIME_PERCENTILES)
        
        # DIVIDIR EN CUBETAS
        low_mask = volatility_series <= low_t
        mid_mask = (volatility_series > low_t) & (volatility_series <= high_t)
        high_mask = volatility_series > high_t
        
        # CALCULAR SR POR RÉGIMEN
        regime_srs = []
        for mask in [low_mask, mid_mask, high_mask]:
            segment = returns[mask]
            if len(segment) >= self.config.REGIME_MIN_SAMPLES:
                std = np.std(segment)
                if std > 1e-10:
                    sr = np.mean(segment) / std
                else:
                    sr = 0.0
                regime_srs.append(sr)
            else:
                regime_srs.append(0.0)
        
        # COEFICIENTE DE VARIACIÓN
        arr = np.array(regime_srs)
        mean_sr = np.mean(arr)
        
        if abs(mean_sr) < 1e-10:
            cv = float(np.std(arr))
        else:
            cv = float(np.std(arr) / abs(mean_sr))
        
        # SCORE EXPONENCIAL NEGATIVO
        regime_score = float(np.exp(-abs(cv)))
        
        # PENALIZACIÓN SI ALGÚN RÉGIMEN ES NEGATIVO Y OTROS POSITIVOS
        signs = np.sign(arr)
        if np.any(signs < 0) and np.any(signs > 0):
            regime_score *= 0.7
        
        return float(np.clip(regime_score, 0.01, 1.0))
    
    # =========================================================================
    # [2.7] PILAR 4: CURVA (K-RATIO Y R²)
    # =========================================================================
    
    def calculate_k_ratio(self, equity_curve: np.ndarray) -> float:
        """
        CALCULA EL K-RATIO DE LARS KESTNER.
        
        EVALÚA LA CONSISTENCIA DEL CRECIMIENTO DEL CAPITAL.
        K-RATIO > 2.0 INDICA CONSISTENCIA EXCEPCIONAL.
        
        RETURNS:
            K-RATIO NORMALIZADO [0, 1]
        """
        equity = np.asarray(equity_curve, dtype=np.float64)
        equity = equity[np.isfinite(equity) & (equity > 0)]
        
        n = len(equity)
        if n < 10:
            return 0.1
        
        # VAMI (VALUE ADDED MONTHLY INDEX)
        vami = equity
        if equity[0] != 1.0 and equity[0] > 0:
            vami = equity / equity[0]
        
        # LOG VAMI
        log_vami = np.log(np.maximum(vami, 1e-10))
        
        # REGRESIÓN LINEAL MANUAL
        x = np.arange(n, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(log_vami)
        
        numerator = np.sum((x - x_mean) * (log_vami - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator < 1e-10:
            return 0.1
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        predicted = slope * x + intercept
        residuals = log_vami - predicted
        
        # ERROR ESTÁNDAR DE LA PENDIENTE
        if n > 2:
            mse = np.sum(residuals ** 2) / (n - 2)
            x_var = np.var(x) * n
            if x_var > 0:
                std_err = np.sqrt(mse / x_var)
            else:
                std_err = 1.0
        else:
            std_err = 1.0
        
        if std_err < 1e-10:
            std_err = 1e-10
        
        # K-RATIO AJUSTADO
        k_ratio = (slope / std_err) * np.sqrt(n) / n
        
        # NORMALIZAR A [0, 1] USANDO TARGET
        k_normalized = min(1.0, max(0.0, k_ratio / self.config.CURVE_K_RATIO_TARGET))
        
        return float(k_normalized)
    
    def calculate_r2_score(self, equity_curve: np.ndarray) -> float:
        """
        CALCULA EL R² DE LA CURVA DE EQUITY.
        
        MIDE QUÉ TAN LINEAL ES EL CRECIMIENTO DEL CAPITAL.
        
        RETURNS:
            R² [0, 1]
        """
        equity = np.asarray(equity_curve, dtype=np.float64)
        equity = equity[np.isfinite(equity)]
        
        n = len(equity)
        if n < 10:
            return 0.1
        
        x = np.arange(n, dtype=np.float64)
        y = equity
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot < 1e-10:
            return 1.0  # LÍNEA PERFECTAMENTE PLANA
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator < 1e-10:
            return 0.1
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return float(np.clip(r2, 0.0, 1.0))
    
    def calculate_curve_quality(self, equity_curve: np.ndarray) -> float:
        """COMBINA K-RATIO Y R² PARA EVALUAR CALIDAD DE CURVA."""
        k_ratio = self.calculate_k_ratio(equity_curve)
        r2 = self.calculate_r2_score(equity_curve)
        
        # PROMEDIO GEOMÉTRICO (MÁS EXIGENTE)
        curve_quality = math.sqrt(k_ratio * r2)
        
        # PENALIZACIÓN SI R² MUY BAJO
        if r2 < self.config.CURVE_R2_MIN:
            curve_quality *= (r2 / self.config.CURVE_R2_MIN)
        
        return float(np.clip(curve_quality, 0.01, 1.0))
    
    # =========================================================================
    # [2.8] PILAR 5: DECAY (FACTOR DE DECAIMIENTO)
    # =========================================================================
    
    def calculate_decay_factor(self, trial_number: int) -> float:
        """
        CALCULA EL FACTOR DE DECAIMIENTO POR EXPLORACIÓN.
        
        PENALIZA DESCUBRIMIENTOS TARDÍOS PARA COMBATIR DATA SNOOPING.
        
        RETURNS:
            FACTOR DE DECAIMIENTO [DECAY_FLOOR, 1.0]
        """
        if not self.config.DECAY_ENABLED:
            return 1.0
        
        t = max(1, trial_number)
        decay = 1.0 / math.log(t + self.config.DECAY_BASE)
        
        # SOFT-VETO: APLICAR FLOOR
        return float(max(self.config.DECAY_FLOOR, min(1.0, decay)))
    
    # =========================================================================
    # [2.9] FUNCIÓN SOFT-VETO
    # =========================================================================
    
    def _apply_soft_penalty(self, value: float, floor: float) -> float:
        """
        APLICA PENALIZACIÓN CON FLOOR MÍNIMO (SOFT-VETO).
        
        EN LUGAR DE PERMITIR QUE VALUE LLEGUE A 0, SE MANTIENE UN MÍNIMO.
        ESTO EVITA EL "PLATEAU DE CEROS" QUE IMPIDE EL APRENDIZAJE.
        """
        return float(max(floor, min(1.0, value)))
    
    # =========================================================================
    # [2.10] FUNCIÓN MAESTRA: COMPUTE SCORE
    # =========================================================================
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        volatility_series: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
        neighbor_scores: Optional[List[float]] = None,
    ) -> float:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │              FUNCIÓN PRINCIPAL DE SCORING CMA-ES                    │
        │                                                                     │
        │  Score = BaseScore(sharpe) × P_psr × P_stab × P_reg ×             │
        │          P_curve × P_decay × P_threshold                           │
        │                                                                     │
        │  RANGO: [1, 1000] - NUNCA CERO                                     │
        └────────────────────────────────────────────────────────────────────┘
        """
        cfg = self.config
        
        # =================================================================
        # EXTRAER MÉTRICAS BASE
        # =================================================================
        trades_dia = self._safe_get(metrics, "trades_por_dia", 0.0)
        if trades_dia == 0:
            trades_dia = self._safe_get(metrics, "trades_dia", 0.0)
        if trades_dia == 0:
            trades_dia = self._safe_get(metrics, "trades_per_day", 0.0)
        
        n_trades = int(self._safe_get(metrics, "n_trades", 0))
        if n_trades == 0:
            n_trades = int(self._safe_get(metrics, "total_trades", 0))
        
        drawdown = self._safe_get(metrics, "drawdown", 50.0)
        if drawdown == 0:
            drawdown = self._safe_get(metrics, "max_drawdown", 50.0)
        
        roi = self._safe_get(metrics, "roi", 0.0)
        
        # SHARPE NOMINAL PARA BASE SCORE
        sharpe_nominal = self._safe_get(metrics, "sharpe", 0.0)
        if sharpe_nominal == 0:
            sharpe_nominal = self._safe_get(metrics, "sharpe_ratio", 0.0)
        
        # =================================================================
        # INTENTAR RECUPERAR returns Y equity_curve DESDE metrics
        # =================================================================
        if returns is None:
            raw_returns = metrics.get("returns", None)
            if raw_returns is None:
                raw_returns = metrics.get("trade_returns", None)
            if raw_returns is not None:
                try:
                    returns = np.asarray(raw_returns, dtype=np.float64)
                except Exception:
                    returns = None
        
        if equity_curve is None:
            raw_equity = metrics.get("equity_curve", None)
            if raw_equity is None:
                raw_equity = metrics.get("equity", None)
            if raw_equity is not None:
                try:
                    equity_curve = np.asarray(raw_equity, dtype=np.float64)
                except Exception:
                    equity_curve = None
        
        if volatility_series is None:
            raw_vol = metrics.get("volatility_series", None)
            if raw_vol is None:
                raw_vol = metrics.get("volatility", None)
            if raw_vol is not None:
                try:
                    volatility_series = np.asarray(raw_vol, dtype=np.float64)
                except Exception:
                    volatility_series = None
        
        # =================================================================
        # PENALIZACIÓN DE UMBRALES (SOFT-VETO)
        # =================================================================
        threshold_penalty_multiplier = 1.0
        
        if trades_dia < cfg.MIN_TRADES_PER_DAY:
            threshold_penalty_multiplier *= cfg.THRESHOLD_PENALTY
        
        if n_trades < cfg.MIN_TOTAL_TRADES:
            threshold_penalty_multiplier *= cfg.THRESHOLD_PENALTY
        
        if drawdown > cfg.MAX_DRAWDOWN_ALLOWED:
            threshold_penalty_multiplier *= cfg.THRESHOLD_PENALTY
        
        if roi < cfg.MIN_ROI:
            threshold_penalty_multiplier *= cfg.THRESHOLD_PENALTY
        
        # FLOOR PARA THRESHOLD_PENALTY_MULTIPLIER
        threshold_penalty_multiplier = max(0.02, threshold_penalty_multiplier)
        
        # =================================================================
        # OBTENER NÚMERO DE TRIAL
        # =================================================================
        trial_number = 1
        if trial is not None:
            trial_number = getattr(trial, 'number', 1) + 1
        
        # =================================================================
        # SCORE BASE: SIGMOIDE DE SHARPE
        # =================================================================
        base_score = self._compute_base_score(sharpe_nominal)
        
        # =================================================================
        # PILAR 1: PSR DEFLATED (CON SOFT-VETO FLOOR)
        # =================================================================
        if returns is not None and len(returns) >= cfg.MIN_TRADES_FOR_PSR:
            psr_val = self.calculate_deflated_psr(returns)
        else:
            psr_val = min(0.95, max(0.10, 0.5 + sharpe_nominal * 0.15))
        
        psr_penalty = self._apply_soft_penalty(psr_val, cfg.PSR_FLOOR)
        
        # =================================================================
        # PILAR 2: ESTABILIDAD (SAM) (CON SOFT-VETO FLOOR)
        # =================================================================
        if neighbor_scores is not None and len(neighbor_scores) > 0:
            original_score = psr_val
            stability_val = self.calculate_stability_score(original_score, neighbor_scores)
        else:
            stability_val = 1.0
        
        stability_penalty = self._apply_soft_penalty(stability_val, cfg.SAM_FLOOR)
        
        # =================================================================
        # PILAR 3: CONSISTENCIA DE RÉGIMEN (CON SOFT-VETO FLOOR)
        # =================================================================
        if returns is not None and volatility_series is not None:
            regime_val = self.calculate_regime_score(returns, volatility_series)
        else:
            regime_val = 1.0
        
        regime_penalty = self._apply_soft_penalty(regime_val, cfg.REGIME_FLOOR)
        
        # =================================================================
        # PILAR 4: CALIDAD DE CURVA (CON SOFT-VETO FLOOR)
        # =================================================================
        if equity_curve is not None and len(equity_curve) > 10:
            curve_val = self.calculate_curve_quality(np.array(equity_curve))
        else:
            curve_val = 1.0
        
        curve_penalty = self._apply_soft_penalty(curve_val, cfg.CURVE_FLOOR)
        
        # =================================================================
        # PILAR 5: DECAIMIENTO (YA INCLUYE FLOOR INTERNO)
        # =================================================================
        decay_penalty = self.calculate_decay_factor(trial_number)
        
        # =================================================================
        # GUARDAR ATRIBUTOS PARA AUDITORÍA
        # =================================================================
        if trial is not None:
            try:
                trial.set_user_attr('base_score', float(base_score))
                trial.set_user_attr('psr', float(psr_val))
                trial.set_user_attr('psr_penalty', float(psr_penalty))
                trial.set_user_attr('stability', float(stability_val))
                trial.set_user_attr('stability_penalty', float(stability_penalty))
                trial.set_user_attr('regime', float(regime_val))
                trial.set_user_attr('regime_penalty', float(regime_penalty))
                trial.set_user_attr('curve_quality', float(curve_val))
                trial.set_user_attr('curve_penalty', float(curve_penalty))
                trial.set_user_attr('decay', float(decay_penalty))
                trial.set_user_attr('threshold_penalty', float(threshold_penalty_multiplier))
                trial.set_user_attr('sr_nominal', float(sharpe_nominal))
            except Exception:
                pass
        
        # =================================================================
        # SCORE FINAL: BASE × PENALIZACIONES CON SOFT-VETO
        # =================================================================
        final_score = (
            base_score 
            * psr_penalty 
            * stability_penalty 
            * regime_penalty 
            * curve_penalty 
            * decay_penalty
            * threshold_penalty_multiplier
        )
        
        # GARANTIZAR RANGO [SCORE_MIN, SCORE_MAX]
        final_score = max(cfg.SCORE_MIN, min(cfg.SCORE_MAX, final_score))
        
        return float(final_score)


# =============================================================================
# ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
# ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗
# ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝
# ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗
# ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
#  ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
#
# [SECCIÓN 3] CLASE OPTIMIZADOR CMA-ES
# =============================================================================


@dataclass
class CMAOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                  CONFIGURACIÓN DEL OPTIMIZADOR CMA-ES                   │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 3.1 CONFIGURACIÓN DE OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # SEMILLA ALEATORIA (NONE = VARIEDAD)
    N_JOBS: int = 1                       # NÚMERO DE WORKERS PARALELOS
    STORAGE: Optional[str] = None         # NONE = EJECUCIÓN EN RAM
    STUDY_NAME_PREFIX: str = "MODELOX"    # PREFIJO PARA NOMBRES DE ESTUDIO
    
    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA CMA-ES
    # =========================================================================
    N_STARTUP_TRIALS: int = 10            # TRIALS ALEATORIOS INICIALES
    WARN_INDEPENDENT_SAMPLING: bool = False
    CONSIDER_PRUNED_TRIALS: bool = False


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
CMA_OPTIMIZER_CONFIG = CMAOptimizerConfig()


class CMAOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       OPTIMIZADOR CMA-ES                                │
    │                                                                         │
    │  COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY                        │
    │                                                                         │
    │  CARACTERÍSTICAS:                                                       │
    │    ✓ APRENDE DE LOS SCORES PARA ADAPTAR LA BÚSQUEDA                   │
    │    ✓ CONVERGE HACIA REGIONES DE ALTA CALIDAD                          │
    │    ✓ IDEAL PARA ENCONTRAR "MESETAS DE PARÁMETROS"                     │
    │    ✓ PENALIZA IMPLÍCITAMENTE LOS PICOS AISLADOS                       │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[CMAOptimizerConfig] = None,
        scoring_config: Optional[CMAScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        INICIALIZA EL OPTIMIZADOR CMA-ES.
        
        ARGS:
            config: CONFIGURACIÓN DE BACKTEST
            n_trials: NÚMERO DE TRIALS A EJECUTAR
            reporters: LISTA DE REPORTERS PARA RESULTADOS
            optimizer_config: CONFIGURACIÓN DEL OPTIMIZADOR
            scoring_config: CONFIGURACIÓN DEL SCORING
            activo: NOMBRE DEL ACTIVO (OPCIONAL)
        """
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or CMA_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or CMA_SCORING_CONFIG
        self.activo = activo
        
        # ESTADO INTERNO
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[CMAScorer] = None
    
    # =========================================================================
    # [3.1] CREAR ESTUDIO OPTUNA
    # =========================================================================
    
    def _create_study(self, strategy_name: str) -> optuna.Study:
        """
        CREA UN ESTUDIO OPTUNA CON SAMPLER CMA-ES.
        
        RETURNS:
            OBJETO OPTUNA.STUDY CONFIGURADO
        """
        cfg = self.optimizer_config
        
        # CONSTRUIR NOMBRE DEL ESTUDIO
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy_name)]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        # CREAR SAMPLER CMA-ES
        sampler = CmaEsSampler(
            seed=cfg.SEED,
            n_startup_trials=cfg.N_STARTUP_TRIALS,
            warn_independent_sampling=cfg.WARN_INDEPENDENT_SAMPLING,
            consider_pruned_trials=cfg.CONSIDER_PRUNED_TRIALS,
        )
        
        # CREAR ESTUDIO
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=cfg.STORAGE,
            load_if_exists=False,
        )
        
        # INICIALIZAR SCORER CON EL ESTUDIO
        self._scorer = CMAScorer(study=study, config=self.scoring_config)
        
        return study
    
    @staticmethod
    def _slug(s: str) -> str:
        """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    # =========================================================================
    # [3.2] PREPARAR PARÁMETROS
    # =========================================================================
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """PREPARA PARÁMETROS PARA UN TRIAL."""
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        # INYECTAR VALORES DE CONFIGURACIÓN
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)

        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
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
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    # =========================================================================
    # [3.3] FUNCIÓN OBJETIVO
    # =========================================================================
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        """CREA LA FUNCIÓN OBJETIVO PARA CMA-ES."""
        
        # IMPORTAR COMPONENTES NECESARIOS
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            t0_total = time.perf_counter()
            
            # LIMPIEZA PERIÓDICA
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            # GENERAR SEÑALES
            signals_df = SignalGenerator.generate_signals(df_entry, strategy, params_rt, df_map)
            
            # EJECUTAR BACKTEST
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_entry, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                return 0.0
            
            trial.set_user_attr("metricas", metrics)
            
            # CALCULAR SCORE CON SCORER CMA
            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
            )
            
            # CREAR ARTIFACTS
            artifacts = TrialArtifacts(
                strategy_name=strategy.name,
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_rt,
                score=score,
                metrics=metrics,
                df_signals=None,
                trades=trades_df.to_pandas(),
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
            )
            
            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)
            
            return score
        
        return objective
    
    # =========================================================================
    # [3.4] OPTIMIZAR
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
        │                    EJECUTAR OPTIMIZACIÓN CMA-ES                     │
        └────────────────────────────────────────────────────────────────────┘
        
        ARGS:
            df: DATAFRAME CON DATOS OHLCV
            strategy: ESTRATEGIA A OPTIMIZAR
            df_by_timeframe: DICT CON DATAFRAMES POR TIMEFRAME
            base_timeframe: TIMEFRAME BASE
        
        RETURNS:
            OBJETO OPTUNA.STUDY CON RESULTADOS
        """
        base_tf = base_timeframe or "1m"
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # CREAR ESTUDIO
        study = self._create_study(strategy.name)
        
        # CREAR OBJETIVO
        objective = self._create_objective(df_base, df_map, strategy, base_tf)
        
        # EJECUTAR OPTIMIZACIÓN
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
    # [3.5] PROPIEDADES
    # =========================================================================
    
    @property
    def last_study(self) -> Optional[optuna.Study]:
        """RETORNA EL ÚLTIMO ESTUDIO EJECUTADO."""
        return self._last_study
    
    @property
    def scorer(self) -> Optional[CMAScorer]:
        """RETORNA EL SCORER UTILIZADO."""
        return self._scorer


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def _slug(s: str) -> str:
    """Genera un slug válido para nombres de estudio."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_cma_study(
    strategy_name: str,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    n_startup_trials: int = 10,
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Crea un estudio Optuna con sampler CMA-ES.
    
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy):
    - Aprende de los scores para adaptar la matriz de covarianza
    - Ideal para encontrar "mesetas de parámetros" (soluciones robustas)
    - Penaliza implícitamente los picos aislados (overfitting)
    - RECOMENDADO para scoring institucional
    
    Args:
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria
        study_name_prefix: Prefijo para el nombre del estudio
        n_startup_trials: Trials aleatorios iniciales
        storage: URI de almacenamiento (None = RAM)
    
    Returns:
        optuna.Study configurado con CMA-ES
    """
    # Construir nombre del estudio
    parts = [study_name_prefix, str(strategy_name)]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    # Crear sampler CMA-ES
    sampler = CmaEsSampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        warn_independent_sampling=False,
        consider_pruned_trials=False,
    )
    
    # Crear estudio
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
    )
    
    return study


def score_cma(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    FUNCIÓN DE SCORING CMA-ES STANDALONE.
    
    USO:
        score = score_cma(metrics)
    """
    scorer = CMAScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
    )


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    "CMAOptimizer",
    "CMAOptimizerConfig",
    "CMAScorer",
    "CMAScoringConfig",
    "CMA_SCORING_CONFIG",
    "CMA_OPTIMIZER_CONFIG",
    "create_cma_study",
    "score_cma",
]

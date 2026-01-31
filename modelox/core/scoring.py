"""modelox/core/scoring.py

═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE SCORING INSTITUCIONAL v2.0 - NORMALIZACIÓN SIGMOIDEA CONTINUA
═══════════════════════════════════════════════════════════════════════════════

FILOSOFÍA v2.0 - GRADIENTES PRESERVADOS:
El Score usa una arquitectura de SOFT-VETO con penalizaciones suaves que
PRESERVAN EL GRADIENTE para que Optuna pueda aprender y distinguir entre
estrategias "desastrosas" y "mediocres con potencial".

    Score = ScoreBase(sharpe) × P_psr × P_stability × P_regime × P_curve × P_decay
    
RANGO DE SALIDA: [1, 1000] - NUNCA CERO ABSOLUTO

CAMBIOS CLAVE vs v1.0:
1. SIGMOIDE BASE: Sharpe 0 → ~200, Sharpe 2 → ~800
2. SOFT-VETO: Penalizaciones con FLOOR mínimo (0.2-0.3), no colapso a 0
3. GRADIENTE PRESERVADO: Optuna siempre puede distinguir mejor de peor
4. RECUPERACIÓN DE DATOS: Intenta extraer returns/equity de métricas

PILARES DEL SISTEMA (ahora con suelos):
1. PSR/DSR - Inferencia Probabilística (floor: 0.3)
2. SAM - Estabilidad de Vecindario (floor: 0.25)
3. Régimen - Consistencia de Volatilidad (floor: 0.3)
4. Curva - Calidad de Equity K-Ratio/R² (floor: 0.2)
5. Decay - Factor de Exploración (floor: 0.4)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Importaciones con fallback
try:
    from scipy.stats import norm, skew, kurtosis
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    
try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

if TYPE_CHECKING:
    import optuna


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA INSTITUCIONAL
# =============================================================================

@dataclass
class InstitutionalScoringConfig:
    """
    Configuración maestra del sistema de scoring institucional v2.0.
    
    NUEVO: Incluye floors (suelos) para soft-veto y parámetros de sigmoide.
    """
    # =========================================================================
    # RANGO DE SALIDA
    # =========================================================================
    score_min: float = 1.0               # Mínimo absoluto (nunca 0)
    score_max: float = 1000.0            # Máximo absoluto
    
    # =========================================================================
    # SIGMOIDE BASE (mapea Sharpe a Score Base)
    # =========================================================================
    sigmoid_k: float = 1.5               # Pendiente de la sigmoide
    sigmoid_x0: float = 1.0              # Centro de la sigmoide (Sharpe donde score=50%)
    sigmoid_floor: float = 0.15          # Mínimo de la sigmoide normalizada
    sigmoid_ceiling: float = 0.95        # Máximo de la sigmoide normalizada
    
    # =========================================================================
    # PILAR 1: PSR/DSR - Inferencia Probabilística
    # =========================================================================
    psr_benchmark_sr: float = 0.0        # SR de referencia para PSR
    dsr_warmup_trials: int = 30          # Trials antes de activar DSR
    min_trades_for_psr: int = 30         # Trades mínimos para PSR válido
    psr_floor: float = 0.30              # SOFT-VETO: mínimo 30% (no colapso)
    
    # =========================================================================
    # PILAR 2: SAM - Estabilidad de Vecindario
    # =========================================================================
    sam_enabled: bool = True             # Activar análisis de vecindario
    sam_radius_pct: float = 0.05         # Radio de perturbación (5% del rango)
    sam_n_neighbors: int = 5             # Número de puntos perturbados
    sam_min_stability: float = 0.80      # Score mínimo en vecindad vs original
    sam_floor: float = 0.25              # SOFT-VETO: mínimo 25%
    
    # =========================================================================
    # PILAR 3: RÉGIMEN - Consistencia de Volatilidad
    # =========================================================================
    regime_enabled: bool = True
    regime_percentiles: Tuple[float, float] = (33, 66)  # Cortes de volatilidad
    regime_min_samples: int = 10         # Mínimo de samples por régimen
    regime_floor: float = 0.30           # SOFT-VETO: mínimo 30%
    
    # =========================================================================
    # PILAR 4: CURVA - Calidad de Equity
    # =========================================================================
    curve_k_ratio_target: float = 2.0    # K-Ratio objetivo (> 2.0 = excelente)
    curve_r2_min: float = 0.70           # R² mínimo aceptable
    curve_floor: float = 0.20            # SOFT-VETO: mínimo 20%
    
    # =========================================================================
    # PILAR 5: DECAY - Factor de Exploración
    # =========================================================================
    decay_enabled: bool = True
    decay_base: float = math.e           # Base del logaritmo de decaimiento
    decay_floor: float = 0.40            # SOFT-VETO: mínimo 40%
    
    # =========================================================================
    # UMBRALES DE SUSPENSO (ahora son penalizaciones fuertes, no eliminación)
    # =========================================================================
    min_trades_per_day: float = 0.10     # Penalización fuerte si < 0.10
    min_total_trades: int = 15           # Penalización fuerte si < 15
    max_drawdown_allowed: float = 60.0   # Penalización fuerte si > 60%
    min_roi: float = -150.0              # Penalización fuerte si < -150%
    
    # Factor de penalización para umbrales (en lugar de suspenso total)
    threshold_penalty: float = 0.15      # Multiplica por 0.15 si viola umbral


# Configuración por defecto
DEFAULT_CONFIG = InstitutionalScoringConfig()


# =============================================================================
# CLASE PRINCIPAL: SCORER INSTITUCIONAL
# =============================================================================

class InstitutionalScorer:
    """
    Sistema de Scoring Institucional v2.0 para Optimización Bayesiana.
    
    ARQUITECTURA SIGMOIDE CON SOFT-VETO:
    =====================================
    - Score Base: Mapeo sigmoide de Sharpe a rango [100, 900]
    - Penalizaciones: Multiplicadores con floor mínimo (0.2-0.4)
    - Rango Final: [1, 1000] - NUNCA cero absoluto
    
    FILOSOFÍA:
    ----------
    - Un mal Sharpe (-1) → ~50
    - Un Sharpe mediocre (0.5) → ~200
    - Un Sharpe bueno (1.5) → ~500  
    - Un Sharpe excelente (3.0) → ~850
    
    Multiplicado por penalizaciones SUAVES que nunca bajan de 0.2
    """
    
    def __init__(
        self,
        study: Optional["optuna.Study"] = None,
        config: Optional[InstitutionalScoringConfig] = None,
    ):
        """
        Inicializa el sistema de scoring institucional.
        
        Args:
            study: Objeto optuna.Study para acceder al historial de trials
            config: Configuración personalizada (usa DEFAULT si None)
        """
        self.study = study
        self.config = config or DEFAULT_CONFIG
        
        # Cache de estadísticas del estudio
        self._cached_sr_stats: Optional[Dict[str, float]] = None
        self._cached_at_trial: int = -1
    
    # =========================================================================
    # NUEVA FUNCIÓN: SIGMOIDE PARA BASE SCORE
    # =========================================================================
    
    def _sigmoid(self, x: float, k: float = 1.5, x0: float = 1.0) -> float:
        """
        Función sigmoide para mapear Sharpe Ratio a [0, 1].
        
        Args:
            x: Valor de entrada (Sharpe Ratio)
            k: Pendiente de la sigmoide (mayor = más empinada)
            x0: Centro de la sigmoide (Sharpe donde output = 0.5)
        
        Returns:
            Valor en [0, 1]
        """
        try:
            exponent = -k * (x - x0)
            # Protección contra overflow
            if exponent > 500:
                return 0.0
            elif exponent < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5
    
    def _compute_base_score(self, sharpe: float) -> float:
        """
        Calcula el score base usando sigmoide.
        
        Mapeo:
            Sharpe -2 → ~50
            Sharpe 0  → ~200  
            Sharpe 1  → ~500 (centro)
            Sharpe 2  → ~800
            Sharpe 4  → ~950
        
        Args:
            sharpe: Sharpe Ratio del backtest
        
        Returns:
            Score base en rango [score_min, score_max]
        """
        cfg = self.config
        
        # Sigmoid raw value en [0, 1]
        sig_raw = self._sigmoid(sharpe, k=cfg.sigmoid_k, x0=cfg.sigmoid_x0)
        
        # Aplicar floor y ceiling para evitar extremos
        sig_normalized = cfg.sigmoid_floor + (cfg.sigmoid_ceiling - cfg.sigmoid_floor) * sig_raw
        
        # Escalar a rango [score_min, score_max]
        score_range = cfg.score_max - cfg.score_min
        base_score = cfg.score_min + score_range * sig_normalized
        
        return float(base_score)
    
    # =========================================================================
    # FUNCIONES AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """Extrae valor numérico de forma segura."""
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
        """Calcula los momentos estadísticos necesarios para PSR."""
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 3:
            return {'mean': 0.0, 'std': 1.0, 'skew': 0.0, 'kurt': 3.0}
        
        mean_val = float(np.mean(returns))
        std_val = float(np.std(returns, ddof=1))
        
        if std_val < 1e-10:
            std_val = 1e-10
        
        if _SCIPY_AVAILABLE:
            skew_val = float(skew(returns, bias=False))
            kurt_val = float(kurtosis(returns, fisher=False, bias=False))
        else:
            # Implementación manual si scipy no está disponible
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
    # PILAR 1: PSR (Probabilistic Sharpe Ratio)
    # =========================================================================
    
    def calculate_psr(
        self,
        returns: np.ndarray,
        benchmark_sr: Optional[float] = None,
    ) -> float:
        """
        Calcula el Probabilistic Sharpe Ratio.
        
        El PSR determina la probabilidad de que el Sharpe Ratio observado
        sea superior a un umbral de referencia, ajustando por la calidad
        de la distribución (skewness y kurtosis).
        
        Fórmula:
            PSR(SR*) = Z((SR_hat - SR*) * sqrt(n-1) / sigma_sr)
            
        donde sigma_sr = sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR²)
        
        Args:
            returns: Array de retornos (no log-retornos, sino retornos simples)
            benchmark_sr: Sharpe Ratio de referencia (default: config.psr_benchmark_sr)
        
        Returns:
            Probabilidad [0, 1] de que el SR real sea > benchmark
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        if n < self.config.min_trades_for_psr:
            # Muestra muy pequeña: penalización severa
            return 0.1
        
        m = self._get_moments(returns)
        sr = m['mean'] / m['std'] if m['std'] > 1e-10 else 0.0
        
        # SR de referencia
        sr_star = benchmark_sr if benchmark_sr is not None else self.config.psr_benchmark_sr
        
        # Error estándar del estimador Sharpe ajustado por asimetría y curtosis
        # sigma_sr = sqrt((1 - skew*SR + (kurt-1)/4 * SR²) / (n-1))
        sr_sq = sr ** 2
        variance_factor = 1 - m['skew'] * sr + ((m['kurt'] - 1) / 4) * sr_sq
        variance_factor = max(0.01, variance_factor)  # Evitar valores negativos
        
        sigma_sr = math.sqrt(variance_factor / max(1, n - 1))
        
        if sigma_sr < 1e-10:
            return 0.9 if sr > sr_star else 0.1
        
        z_score = (sr - sr_star) / sigma_sr
        
        # CDF de normal estándar
        if _SCIPY_AVAILABLE:
            psr = float(norm.cdf(z_score))
        else:
            # Aproximación manual de CDF normal
            psr = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2)))
        
        return float(np.clip(psr, 0.01, 0.99))
    
    # =========================================================================
    # PILAR 1B: DSR (Deflated Sharpe Ratio)
    # =========================================================================
    
    def _update_sr_cache(self) -> None:
        """Actualiza el cache de estadísticas de SR del estudio."""
        if self.study is None:
            return
        
        current_trial = len(self.study.trials)
        if current_trial == self._cached_at_trial:
            return
        
        # Recopilar todos los SR nominales de trials completados
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
        Calcula el umbral de Sharpe Ratio desinflado basado en múltiples pruebas.
        
        El DSR corrige el sesgo de selección cuando se prueban muchas
        configuraciones. El umbral SR₀ se eleva automáticamente con más trials.
        
        Fórmula aproximada del SR máximo esperado bajo H₀:
            E[max(SR)] ≈ E[SR] + sqrt(V) * ((1-γ)*Z⁻¹(1-1/N) + γ*Z⁻¹(1-1/(Ne)))
        
        donde γ = constante de Euler-Mascheroni ≈ 0.5772
        
        Returns:
            Umbral de SR para el DSR (SR₀)
        """
        self._update_sr_cache()
        
        if self._cached_sr_stats is None:
            return 0.0
        
        n_trials = self._cached_sr_stats['n']
        var_sr = self._cached_sr_stats['var']
        mean_sr = self._cached_sr_stats['mean']
        
        if n_trials < self.config.dsr_warmup_trials:
            return 0.0
        
        if var_sr < 1e-10:
            return 0.0
        
        # Constante de Euler-Mascheroni
        emc = 0.5772156649
        
        # Calcular Z^(-1) (quantiles de normal estándar)
        if _SCIPY_AVAILABLE:
            z1 = float(norm.ppf(max(0.001, 1 - 1 / n_trials)))
            z2 = float(norm.ppf(max(0.001, 1 - 1 / (n_trials * math.e))))
        else:
            # Aproximación simple
            p1 = max(0.001, 1 - 1 / n_trials)
            p2 = max(0.001, 1 - 1 / (n_trials * math.e))
            z1 = math.sqrt(2) * _erfinv(2 * p1 - 1)
            z2 = math.sqrt(2) * _erfinv(2 * p2 - 1)
        
        # Max SR esperado tras N pruebas independientes
        std_sr = math.sqrt(var_sr)
        sr0 = mean_sr + std_sr * ((1 - emc) * z1 + emc * z2)
        
        return max(0.0, sr0)
    
    def calculate_deflated_psr(self, returns: np.ndarray) -> float:
        """
        Calcula el PSR con umbral ajustado por DSR.
        
        Combina PSR y DSR para obtener una medida de significancia
        que corrige tanto por distribución como por múltiples pruebas.
        
        Args:
            returns: Array de retornos
        
        Returns:
            PSR deflated [0, 1]
        """
        sr_deflated = self.calculate_dsr_threshold()
        return self.calculate_psr(returns, benchmark_sr=sr_deflated)
    
    # =========================================================================
    # PILAR 2: SAM (Sharpness-Aware Minimization / Stability)
    # =========================================================================
    
    def calculate_stability_score(
        self,
        original_score: float,
        neighbor_scores: List[float],
    ) -> float:
        """
        Calcula el factor de estabilidad de parámetros (SAM).
        
        Evalúa si el rendimiento es sensible a pequeñas perturbaciones
        en los parámetros. Una estrategia robusta debe mantener
        rendimiento similar en la vecindad de sus parámetros.
        
        Fórmula:
            S_stability = max(0, 1 - (Score_original - min(Score_neighbors)) / 
                                     (Score_original + epsilon))
        
        Args:
            original_score: Score con parámetros originales
            neighbor_scores: Scores con parámetros perturbados
        
        Returns:
            Factor de estabilidad [0, 1]
        """
        if not self.config.sam_enabled:
            return 1.0
        
        if not neighbor_scores or len(neighbor_scores) == 0:
            return 0.5  # Sin datos de vecindario: penalización moderada
        
        if original_score <= 0:
            return 0.1
        
        min_neighbor = min(neighbor_scores)
        epsilon_safe = 1e-6
        
        # Degradación relativa
        degradation = (original_score - min_neighbor) / (original_score + epsilon_safe)
        
        # Factor de estabilidad
        stability = max(0.0, 1.0 - degradation)
        
        # Penalización adicional si degradación es catastrófica (> 50%)
        if degradation > 0.5:
            stability *= 0.5
        
        return float(np.clip(stability, 0.01, 1.0))
    
    # =========================================================================
    # PILAR 3: RÉGIMEN (Consistencia en Cubetas de Volatilidad)
    # =========================================================================
    
    def calculate_regime_score(
        self,
        returns: np.ndarray,
        volatility_series: np.ndarray,
    ) -> float:
        """
        Evalúa la consistencia del rendimiento en diferentes regímenes de volatilidad.
        
        Divide los datos en 3 cubetas (baja, media, alta volatilidad)
        y mide si el desempeño es consistente entre ellas.
        
        Fórmula:
            S_regime = exp(-CV(SR_low, SR_med, SR_high))
        
        donde CV = coeficiente de variación (std/mean)
        
        Args:
            returns: Array de retornos por trade/período
            volatility_series: Serie de volatilidad correspondiente
        
        Returns:
            Factor de régimen [0, 1]
        """
        if not self.config.regime_enabled:
            return 1.0
        
        returns = np.asarray(returns, dtype=np.float64)
        volatility_series = np.asarray(volatility_series, dtype=np.float64)
        
        # Asegurar misma longitud
        min_len = min(len(returns), len(volatility_series))
        if min_len < self.config.regime_min_samples * 3:
            return 0.7  # Datos insuficientes: penalización leve
        
        returns = returns[:min_len]
        volatility_series = volatility_series[:min_len]
        
        # Filtrar NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(volatility_series)
        returns = returns[valid_mask]
        volatility_series = volatility_series[valid_mask]
        
        if len(returns) < self.config.regime_min_samples * 3:
            return 0.7
        
        # Calcular percentiles de volatilidad
        low_t, high_t = np.percentile(volatility_series, self.config.regime_percentiles)
        
        # Dividir en cubetas
        low_mask = volatility_series <= low_t
        mid_mask = (volatility_series > low_t) & (volatility_series <= high_t)
        high_mask = volatility_series > high_t
        
        # Calcular SR por régimen
        regime_srs = []
        for mask in [low_mask, mid_mask, high_mask]:
            segment = returns[mask]
            if len(segment) >= self.config.regime_min_samples:
                std = np.std(segment)
                if std > 1e-10:
                    sr = np.mean(segment) / std
                else:
                    sr = 0.0
                regime_srs.append(sr)
            else:
                regime_srs.append(0.0)
        
        # Coeficiente de variación
        arr = np.array(regime_srs)
        mean_sr = np.mean(arr)
        
        if abs(mean_sr) < 1e-10:
            # Media cercana a cero: usar varianza absoluta
            cv = float(np.std(arr))
        else:
            cv = float(np.std(arr) / abs(mean_sr))
        
        # Score exponencial negativo
        regime_score = float(np.exp(-abs(cv)))
        
        # Penalización adicional si algún régimen es negativo y otros positivos
        signs = np.sign(arr)
        if np.any(signs < 0) and np.any(signs > 0):
            regime_score *= 0.7  # Penalizar inconsistencia de signo
        
        return float(np.clip(regime_score, 0.01, 1.0))
    
    # =========================================================================
    # PILAR 4: CURVA (K-Ratio y R²)
    # =========================================================================
    
    def calculate_k_ratio(self, equity_curve: np.ndarray) -> float:
        """
        Calcula el K-Ratio de Lars Kestner (revisión 2013).
        
        Evalúa la consistencia del crecimiento del capital midiendo
        la pendiente de regresión sobre el error estándar.
        
        Fórmula:
            K-Ratio = (Slope de log(VAMI)) / (SE(Slope) × n) × sqrt(n)
        
        Un K-Ratio > 2.0 indica consistencia excepcional.
        
        Args:
            equity_curve: Curva de equity acumulada (VAMI)
        
        Returns:
            K-Ratio normalizado [0, 1]
        """
        equity = np.asarray(equity_curve, dtype=np.float64)
        equity = equity[np.isfinite(equity) & (equity > 0)]
        
        n = len(equity)
        if n < 10:
            return 0.1
        
        # VAMI (Value Added Monthly Index)
        # Si equity ya está normalizada a 1, usarla directamente
        vami = equity
        if equity[0] != 1.0 and equity[0] > 0:
            vami = equity / equity[0]
        
        # Log VAMI
        log_vami = np.log(np.maximum(vami, 1e-10))
        
        # Regresión lineal
        x = np.arange(n).reshape(-1, 1)
        
        if _SKLEARN_AVAILABLE:
            model = LinearRegression().fit(x, log_vami)
            slope = float(model.coef_[0])
            residuals = log_vami - model.predict(x)
        else:
            # Regresión manual
            x_flat = np.arange(n, dtype=np.float64)
            x_mean = np.mean(x_flat)
            y_mean = np.mean(log_vami)
            
            numerator = np.sum((x_flat - x_mean) * (log_vami - y_mean))
            denominator = np.sum((x_flat - x_mean) ** 2)
            
            if denominator < 1e-10:
                return 0.1
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            predicted = slope * x_flat + intercept
            residuals = log_vami - predicted
        
        # Error estándar de la pendiente
        if n > 2:
            mse = np.sum(residuals ** 2) / (n - 2)
            x_var = np.var(np.arange(n)) * n
            if x_var > 0:
                std_err = np.sqrt(mse / x_var)
            else:
                std_err = 1.0
        else:
            std_err = 1.0
        
        if std_err < 1e-10:
            std_err = 1e-10
        
        # K-Ratio ajustado
        k_ratio = (slope / std_err) * np.sqrt(n) / n
        
        # Normalizar a [0, 1] usando target
        k_normalized = min(1.0, max(0.0, k_ratio / self.config.curve_k_ratio_target))
        
        return float(k_normalized)
    
    def calculate_r2_score(self, equity_curve: np.ndarray) -> float:
        """
        Calcula el R² de la curva de equity.
        
        Mide qué tan lineal es el crecimiento del capital.
        Un R² alto indica generación consistente de alpha.
        
        Args:
            equity_curve: Curva de equity acumulada
        
        Returns:
            R² [0, 1]
        """
        equity = np.asarray(equity_curve, dtype=np.float64)
        equity = equity[np.isfinite(equity)]
        
        n = len(equity)
        if n < 10:
            return 0.1
        
        x = np.arange(n, dtype=np.float64)
        y = equity
        
        # Regresión lineal manual
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot < 1e-10:
            return 1.0  # Línea perfectamente plana
        
        # Calcular pendiente y residuos
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
        """
        Combina K-Ratio y R² para evaluar calidad de curva.
        
        Args:
            equity_curve: Curva de equity acumulada
        
        Returns:
            Factor de calidad de curva [0, 1]
        """
        k_ratio = self.calculate_k_ratio(equity_curve)
        r2 = self.calculate_r2_score(equity_curve)
        
        # Promedio geométrico (más exigente que aritmético)
        curve_quality = math.sqrt(k_ratio * r2)
        
        # Penalización si R² muy bajo (curva errática)
        if r2 < self.config.curve_r2_min:
            curve_quality *= (r2 / self.config.curve_r2_min)
        
        return float(np.clip(curve_quality, 0.01, 1.0))
    
    # =========================================================================
    # PILAR 5: DECAY (Factor de Decaimiento por Exploración)
    # =========================================================================
    
    def calculate_decay_factor(self, trial_number: int) -> float:
        """
        Calcula el factor de decaimiento por exploración.
        
        Penaliza descubrimientos tardíos para combatir data snooping.
        Un SR encontrado en trial #10 es más confiable que en trial #500.
        
        Fórmula:
            F_decay = 1 / log(trial + e)
        
        Con SOFT-VETO: aplica floor mínimo de config.decay_floor
        
        Args:
            trial_number: Número del trial actual
        
        Returns:
            Factor de decaimiento [decay_floor, 1.0]
        """
        if not self.config.decay_enabled:
            return 1.0
        
        t = max(1, trial_number)
        decay = 1.0 / math.log(t + self.config.decay_base)
        
        # SOFT-VETO: aplicar floor
        return float(max(self.config.decay_floor, min(1.0, decay)))
    
    # =========================================================================
    # FUNCIÓN SOFT-VETO: Aplicar penalización con floor
    # =========================================================================
    
    def _apply_soft_penalty(self, value: float, floor: float) -> float:
        """
        Aplica penalización con floor mínimo (soft-veto).
        
        En lugar de permitir que value llegue a 0, se mantiene un mínimo.
        Esto evita el "plateau de ceros" que impide el aprendizaje.
        
        Args:
            value: Valor del multiplicador de penalización [0, 1]
            floor: Mínimo permitido (soft-veto floor)
        
        Returns:
            max(floor, value)
        """
        return float(max(floor, min(1.0, value)))
    
    # =========================================================================
    # FUNCIÓN MAESTRA: SCORE INSTITUCIONAL CON SIGMOIDE Y SOFT-VETO
    # =========================================================================
    
    def compute_master_score(
        self,
        trial: Optional["optuna.Trial"],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        volatility_series: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
        neighbor_scores: Optional[List[float]] = None,
    ) -> float:
        """
        Función principal de scoring institucional v2.0.
        
        NUEVA ARQUITECTURA SIGMOIDE CON SOFT-VETO:
        ==========================================
        
        Score = BaseScore(sharpe) × P_psr × P_stability × P_regime × P_curve × P_decay
        
        Donde:
        - BaseScore: Sigmoide de Sharpe → [100, 900]
        - Cada P_x: Penalización con floor (nunca baja de 0.2-0.4)
        - Resultado Final: Rango [1, 1000], NUNCA cero
        
        Args:
            trial: Trial de Optuna (para guardar atributos y obtener número)
            metrics: Diccionario de métricas del backtest
            returns: Array de retornos por trade
            volatility_series: Serie de volatilidad (para régimen)
            equity_curve: Curva de equity acumulada
            neighbor_scores: Scores de parámetros perturbados (para SAM)
        
        Returns:
            Score institucional final [1, 1000]
        """
        cfg = self.config
        
        # =====================================================================
        # EXTRAER MÉTRICAS BASE
        # =====================================================================
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
        
        # Sharpe nominal para base score
        sharpe_nominal = self._safe_get(metrics, "sharpe", 0.0)
        if sharpe_nominal == 0:
            sharpe_nominal = self._safe_get(metrics, "sharpe_ratio", 0.0)
        
        # =====================================================================
        # INTENTAR RECUPERAR returns y equity_curve desde metrics
        # =====================================================================
        if returns is None:
            # Intentar extraer de metrics
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
        
        # =====================================================================
        # PENALIZACIÓN DE UMBRALES (Soft-veto en lugar de suspenso total)
        # =====================================================================
        threshold_penalty_multiplier = 1.0
        
        if trades_dia < cfg.min_trades_per_day:
            threshold_penalty_multiplier *= cfg.threshold_penalty
        
        if n_trades < cfg.min_total_trades:
            threshold_penalty_multiplier *= cfg.threshold_penalty
        
        if drawdown > cfg.max_drawdown_allowed:
            threshold_penalty_multiplier *= cfg.threshold_penalty
        
        if roi < cfg.min_roi:
            threshold_penalty_multiplier *= cfg.threshold_penalty
        
        # Floor para threshold_penalty_multiplier
        threshold_penalty_multiplier = max(0.02, threshold_penalty_multiplier)
        
        # =====================================================================
        # OBTENER NÚMERO DE TRIAL
        # =====================================================================
        trial_number = 1
        if trial is not None:
            trial_number = getattr(trial, 'number', 1) + 1
        
        # =====================================================================
        # SCORE BASE: Sigmoide de Sharpe
        # =====================================================================
        base_score = self._compute_base_score(sharpe_nominal)
        
        # =====================================================================
        # PILAR 1: PSR DEFLATED (con soft-veto floor)
        # =====================================================================
        if returns is not None and len(returns) >= cfg.min_trades_for_psr:
            psr_val = self.calculate_deflated_psr(returns)
        else:
            # Sin retornos: usar Sharpe normalizado como proxy
            # Mapear sharpe a [0.3, 0.9] aproximadamente
            psr_val = min(0.95, max(0.10, 0.5 + sharpe_nominal * 0.15))
        
        # SOFT-VETO: aplicar floor
        psr_penalty = self._apply_soft_penalty(psr_val, cfg.psr_floor)
        
        # =====================================================================
        # PILAR 2: ESTABILIDAD (SAM) (con soft-veto floor)
        # =====================================================================
        if neighbor_scores is not None and len(neighbor_scores) > 0:
            # Calcular score original comparable (usar PSR como base)
            original_score = psr_val
            stability_val = self.calculate_stability_score(original_score, neighbor_scores)
        else:
            # Sin análisis de vecindario: valor NEUTRO (no penalizar por falta de datos)
            stability_val = 1.0
        
        # SOFT-VETO: aplicar floor
        stability_penalty = self._apply_soft_penalty(stability_val, cfg.sam_floor)
        
        # =====================================================================
        # PILAR 3: CONSISTENCIA DE RÉGIMEN (con soft-veto floor)
        # =====================================================================
        if returns is not None and volatility_series is not None:
            regime_val = self.calculate_regime_score(returns, volatility_series)
        else:
            # Sin datos de régimen: valor NEUTRO (no penalizar por falta de datos)
            regime_val = 1.0
        
        # SOFT-VETO: aplicar floor
        regime_penalty = self._apply_soft_penalty(regime_val, cfg.regime_floor)
        
        # =====================================================================
        # PILAR 4: CALIDAD DE CURVA (con soft-veto floor)
        # =====================================================================
        if equity_curve is not None and len(equity_curve) > 10:
            curve_val = self.calculate_curve_quality(np.array(equity_curve))
        else:
            # Sin curva: valor NEUTRO (no penalizar por falta de datos)
            curve_val = 1.0
        
        # SOFT-VETO: aplicar floor
        curve_penalty = self._apply_soft_penalty(curve_val, cfg.curve_floor)
        
        # =====================================================================
        # PILAR 5: DECAIMIENTO (ya incluye floor interno)
        # =====================================================================
        decay_penalty = self.calculate_decay_factor(trial_number)
        
        # =====================================================================
        # GUARDAR ATRIBUTOS PARA AUDITORÍA (si hay trial)
        # =====================================================================
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
        
        # =====================================================================
        # SCORE FINAL: Base × Penalizaciones con Soft-Veto
        # =====================================================================
        final_score = (
            base_score 
            * psr_penalty 
            * stability_penalty 
            * regime_penalty 
            * curve_penalty 
            * decay_penalty
            * threshold_penalty_multiplier
        )
        
        # Garantizar rango [score_min, score_max]
        final_score = max(cfg.score_min, min(cfg.score_max, final_score))
        
        return float(final_score)


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def _erfinv(x: float) -> float:
    """Aproximación de la función inversa de error."""
    # Aproximación de Winitzki
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


# =============================================================================
# INSTANCIA GLOBAL Y FUNCIONES DE COMPATIBILIDAD
# =============================================================================

_GLOBAL_SCORER: Optional[InstitutionalScorer] = None


def get_global_scorer(study: Optional["optuna.Study"] = None) -> InstitutionalScorer:
    """Obtiene o crea el scorer institucional global."""
    global _GLOBAL_SCORER
    
    if _GLOBAL_SCORER is None or (study is not None and _GLOBAL_SCORER.study != study):
        _GLOBAL_SCORER = InstitutionalScorer(study=study)
    
    return _GLOBAL_SCORER


def set_study_for_scorer(study: "optuna.Study") -> None:
    """Configura el estudio para el scorer global."""
    global _GLOBAL_SCORER
    _GLOBAL_SCORER = InstitutionalScorer(study=study)


# =============================================================================
# FUNCIÓN PRINCIPAL DE SCORING PARA OPTUNA
# =============================================================================

def score_optuna(
    metrics: Mapping[str, Any],
    trial: Optional["optuna.Trial"] = None,
    returns: Optional[np.ndarray] = None,
    volatility_series: Optional[np.ndarray] = None,
    equity_curve: Optional[List[float]] = None,
    neighbor_scores: Optional[List[float]] = None,
) -> float:
    """
    Función principal de scoring para Optuna v2.0.
    
    ARQUITECTURA SIGMOIDE CON SOFT-VETO:
    ====================================
    - Rango de salida: [1, 1000]
    - NUNCA retorna 0 (evita plateau de ceros)
    - Base sigmoide sobre Sharpe
    - Penalizaciones con floor mínimo
    
    Args:
        metrics: Diccionario de métricas del backtest
        trial: Trial de Optuna (opcional pero recomendado)
        returns: Array de retornos por trade (opcional)
        volatility_series: Serie de volatilidad (opcional)
        equity_curve: Curva de equity (opcional)
        neighbor_scores: Scores de vecinos perturbados (opcional)
    
    Returns:
        Score institucional [1, 1000]
    """
    # Obtener estudio del trial si está disponible
    study = None
    if trial is not None:
        try:
            study = trial.study
        except Exception:
            pass
    
    scorer = get_global_scorer(study)
    
    # Convertir equity_curve a numpy si es lista
    equity_arr = None
    if equity_curve is not None:
        equity_arr = np.array(equity_curve, dtype=np.float64)
    
    return scorer.compute_master_score(
        trial=trial,
        metrics=metrics,
        returns=returns,
        volatility_series=volatility_series,
        equity_curve=equity_arr,
        neighbor_scores=neighbor_scores,
    )


def score_unified(
    metrics: Mapping[str, Any],
    neighborhood_result: Optional[Mapping[str, Any]] = None,
    trial_number: int = 0,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    Alias de compatibilidad con el sistema anterior.
    
    Convierte el formato antiguo al nuevo sistema institucional.
    """
    # Extraer neighbor_scores si hay resultado de vecindario
    neighbor_scores = None
    if neighborhood_result is not None:
        neighbor_scores = neighborhood_result.get('neighbor_scores', [])
    
    return score_optuna(
        metrics=metrics,
        trial=None,
        returns=None,
        volatility_series=None,
        equity_curve=equity_curve,
        neighbor_scores=neighbor_scores if neighbor_scores else None,
    )


def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """Score sin análisis de vecindario."""
    return score_optuna(metrics)


def nsga2_objectives(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """
    Objetivos para NSGA-II (quality, drawdown).
    
    Quality ahora en rango [1, 1000].
    """
    quality = score_optuna(metrics)
    drawdown = InstitutionalScorer._safe_get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = InstitutionalScorer._safe_get(metrics, "max_drawdown", 50.0)
    return (max(1.0, quality), max(0.0, min(100.0, drawdown)))


# =============================================================================
# CLASES LEGACY PARA COMPATIBILIDAD
# =============================================================================

@dataclass
class NeighborhoodConfig:
    """Configuración del sistema de vecindario (versión institucional)."""
    n_neighbors: int = 5
    max_dispersion: float = 0.40
    perturbation_std: float = 0.05
    lambda_penalty: float = 1.5
    seed: Optional[int] = 42
    exclude_prefixes: Tuple[str, ...] = ("__", "exit_", "cantidad")
    enabled: bool = True
    min_trades_per_day: float = 0.15
    min_profit_factor: float = 1.1
    min_sharpe: float = 0.5


@dataclass
class NeighborhoodResult:
    """Resultado del test de robustez institucional."""
    
    original_metrics: Dict[str, Any] = field(default_factory=dict)
    original_score: float = 0.0
    
    n_neighbors_tested: int = 0
    n_neighbors_successful: int = 0
    neighbor_metrics: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_params: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_scores: List[float] = field(default_factory=list)
    
    dispersions: Dict[str, float] = field(default_factory=dict)
    avg_dispersion: float = 1.0
    incertidumbre: float = 1.0
    
    robustness_approved: bool = False
    max_dispersion_allowed: float = 0.50
    
    aggregated_score: float = 0.0
    
    mean_score: float = 0.0
    std_score: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    neighbor_sharpes: List[float] = field(default_factory=list)
    neighbor_cvars: List[float] = field(default_factory=list)
    neighbor_r2s: List[float] = field(default_factory=list)
    
    execution_time_ms: float = 0.0
    trial_number: int = 0
    skip_reason: str = ""
    
    original_sharpe: float = 0.0
    original_cvar: float = 0.0
    original_r2: float = 0.0
    robust_dsr: float = 0.0
    worst_case_cvar: float = 100.0
    equity_stability_r2: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "n_neighbors_tested": self.n_neighbors_tested,
            "n_neighbors_successful": self.n_neighbors_successful,
            "robustness_approved": self.robustness_approved,
            "avg_dispersion": self.avg_dispersion,
            "incertidumbre": self.incertidumbre,
            "dispersions": dict(self.dispersions),
            "max_dispersion_allowed": self.max_dispersion_allowed,
            "aggregated_score": self.aggregated_score,
            "neighbor_scores": list(self.neighbor_scores),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "execution_time_ms": self.execution_time_ms,
            "trial_number": self.trial_number,
            "original_score": self.original_score,
            "sr_nominal": self.original_sharpe,
        }


DEFAULT_NEIGHBORHOOD_CONFIG = NeighborhoodConfig()


# =============================================================================
# FUNCIONES LEGACY
# =============================================================================

def format_score(score: float) -> str:
    """
    Formatea el score para display.
    
    Nuevo rango [1, 1000]:
        - score >= 100: mostrar como entero
        - score < 100: mostrar con 1 decimal
        - score < 10: mostrar con 2 decimales
    """
    if score >= 100:
        return f"{score:.0f}"
    elif score >= 10:
        return f"{score:.1f}"
    else:
        return f"{score:.2f}"


def check_robustness_requirements(
    metrics: Mapping[str, Any],
    cfg: Optional[NeighborhoodConfig] = None,
) -> Tuple[bool, str]:
    """Verifica requisitos para test de robustez."""
    if cfg is None:
        cfg = DEFAULT_NEIGHBORHOOD_CONFIG
    
    pf = InstitutionalScorer._safe_get(metrics, "profit_factor", 0.0)
    if pf == 0:
        pf = InstitutionalScorer._safe_get(metrics, "pf", 0.0)
    
    tpd = InstitutionalScorer._safe_get(metrics, "trades_por_dia", 0.0)
    if tpd == 0:
        tpd = InstitutionalScorer._safe_get(metrics, "trades_per_day", 0.0)
    
    sharpe = InstitutionalScorer._safe_get(metrics, "sharpe", 0.0)
    
    if pf < cfg.min_profit_factor:
        return False, f"PF ({pf:.2f}) < {cfg.min_profit_factor}"
    
    if tpd < cfg.min_trades_per_day:
        return False, f"TPD ({tpd:.2f}) < {cfg.min_trades_per_day}"
    
    if sharpe < cfg.min_sharpe:
        return False, f"Sharpe ({sharpe:.2f}) < {cfg.min_sharpe}"
    
    return True, "OK"


def generate_gaussian_neighbors(
    params: Dict[str, Any],
    n_neighbors: int,
    perturbation_std: float,
    exclude_prefixes: Tuple[str, ...],
    seed: Optional[int] = None,
    trial_number: int = 0,
) -> List[Dict[str, Any]]:
    """Genera N vecinos usando ruido gaussiano."""
    actual_seed = (seed or 42) + trial_number * 1000
    rng = np.random.default_rng(actual_seed)
    
    neighbors = []
    
    perturbable = {}
    for key, value in params.items():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            perturbable[key] = value
    
    if not perturbable:
        return [dict(params) for _ in range(n_neighbors)]
    
    for _ in range(n_neighbors):
        neighbor = dict(params)
        
        for key, original in perturbable.items():
            if abs(original) < 1e-10:
                sigma = perturbation_std
            else:
                sigma = abs(original) * perturbation_std
            
            noise = rng.normal(0, sigma)
            new_val = original + noise
            
            if isinstance(original, int):
                new_val = int(round(new_val))
                if original > 0:
                    new_val = max(1, new_val)
            else:
                if original > 0:
                    new_val = max(1e-10, new_val)
            
            neighbor[key] = new_val
        
        neighbors.append(neighbor)
    
    return neighbors


def run_neighborhood_analysis(
    *,
    strategy: Any,
    df: Any,
    params: Dict[str, Any],
    original_metrics: Dict[str, Any],
    original_score: float,
    equity_curve: Optional[List[float]],
    config: Any,
    neighborhood_config: NeighborhoodConfig,
    trial_number: int,
    run_backtest_fn: Callable,
    generate_signals_fn: Callable,
) -> NeighborhoodResult:
    """Ejecuta análisis de vecindario institucional."""
    t_start = time.perf_counter()
    
    cfg = neighborhood_config
    result = NeighborhoodResult()
    result.trial_number = trial_number
    result.original_score = original_score
    result.original_metrics = dict(original_metrics)
    
    # Verificar requisitos
    ok, reason = check_robustness_requirements(original_metrics, cfg)
    if not ok:
        result.n_neighbors_tested = -1
        result.skip_reason = reason
        result.aggregated_score = score_optuna(original_metrics)
        result.execution_time_ms = (time.perf_counter() - t_start) * 1000
        return result
    
    # Generar vecinos
    neighbors = generate_gaussian_neighbors(
        params=params,
        n_neighbors=cfg.n_neighbors,
        perturbation_std=cfg.perturbation_std,
        exclude_prefixes=cfg.exclude_prefixes,
        seed=cfg.seed,
        trial_number=trial_number,
    )
    
    result.n_neighbors_tested = len(neighbors)
    all_scores = [original_score]
    
    for neighbor_params in neighbors:
        try:
            neighbor_signals = generate_signals_fn(df, strategy, neighbor_params)
            trades_df, neighbor_equity, neighbor_metrics = run_backtest_fn(
                df, neighbor_signals, config, neighbor_params, strategy
            )
            
            if trades_df.is_empty():
                continue
            
            n_score = float(score_optuna(neighbor_metrics))
            all_scores.append(n_score)
            result.neighbor_scores.append(n_score)
            result.n_neighbors_successful += 1
            
        except Exception:
            continue
    
    # Calcular score final con sistema institucional
    result.aggregated_score = score_optuna(
        original_metrics,
        equity_curve=equity_curve,
        neighbor_scores=result.neighbor_scores if result.neighbor_scores else None,
    )
    
    result.execution_time_ms = (time.perf_counter() - t_start) * 1000
    
    return result


def calculate_deflated_sharpe_score(
    sharpe: float,
    trial_number: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR Score para compatibilidad."""
    scorer = get_global_scorer()
    
    if n_trades < 5:
        return 5.0
    
    # Generar retornos sintéticos con los momentos dados
    returns = np.random.normal(sharpe * 0.01, 0.01, n_trades)
    
    psr = scorer.calculate_psr(returns, benchmark_sr=0)
    return float(psr * 100)


def deflated_sharpe_ratio(sharpe_obs: float, n_trials: int, n_trades: int, **kw) -> float:
    """Legacy: Retorna probabilidad [0, 1]."""
    return calculate_deflated_sharpe_score(sharpe_obs, n_trials, n_trades) / 100.0


def shutdown_neighbor_pool():
    """Legacy: No hace nada."""
    pass


def cleanup_parallel_resources():
    """Limpia recursos."""
    global _GLOBAL_SCORER
    _GLOBAL_SCORER = None


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # Clase principal
    "InstitutionalScorer",
    "InstitutionalScoringConfig",
    
    # Funciones principales
    "score_optuna",
    "score_unified",
    "score_quality_only",
    "set_study_for_scorer",
    "get_global_scorer",
    
    # Funciones de vecindario
    "run_neighborhood_analysis",
    "generate_gaussian_neighbors",
    "check_robustness_requirements",
    
    # Clases legacy
    "NeighborhoodConfig",
    "NeighborhoodResult",
    "DEFAULT_NEIGHBORHOOD_CONFIG",
    
    # Utilidades
    "format_score",
    "nsga2_objectives",
    "cleanup_parallel_resources",
]

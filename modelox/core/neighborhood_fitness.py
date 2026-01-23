"""modelox/core/neighborhood_fitness.py

═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE AGREGACIÓN DE FITNESS VECINAL (Neighborhood Fitness Aggregation)
═══════════════════════════════════════════════════════════════════════════════

Implementación basada en el paper de optimización robusta que rechaza evaluar
parámetros de forma aislada. En su lugar, evalúa la TOPOLOGÍA LOCAL alrededor
de los parámetros propuestos.

METODOLOGÍA:
1. Generación de Vecinos: Para cada parámetro base, genera K variaciones 
   aleatorias usando ruido gaussiano.
2. Ejecución Múltiple: Ejecuta K+1 backtests por trial (original + vecinos).
3. Fórmula de Penalización: Score = μ_M - λ·σ_M
   - μ_M: Media del rendimiento de todos los vecinos
   - σ_M: Desviación estándar (variabilidad entre vecinos)
   - λ: Factor de penalización por inestabilidad

TRINIDAD DE OBJETIVOS (NSGA-II):
1. Robust_DSR: Sharpe ajustado por vecinos y número de trials
2. Worst_Case_CVaR: El peor CVaR entre todos los vecinos  
3. Equity_Stability_R2: R² promedio de la curva de equity

FILOSOFÍA:
- "Picos de aguja" (alta ganancia pero vecinos malos) → Score bajo
- "Mesetas" (rendimiento estable en todo el vecindario) → Score alto
- Penalización por data mining según número de trials probados
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy import stats as scipy_stats


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA VECINAL
# =============================================================================

@dataclass
class NeighborhoodConfig:
    """
    Configuración del sistema de Agregación de Fitness Vecinal.
    
    PARÁMETROS CLAVE:
    - n_neighbors: Número de vecinos a generar (K). Más vecinos = más robusto pero más lento.
    - perturbation_std: Desviación estándar del ruido gaussiano (% del valor del parámetro).
    - lambda_penalty: Factor de penalización por varianza (aversión a la inestabilidad).
    
    NOTA: El costo computacional aumenta linealmente con n_neighbors.
    """
    
    # Número de vecinos a generar por trial
    n_neighbors: int = 5
    
    # Desviación estándar del ruido gaussiano (como % del valor del parámetro)
    # 0.05 = 5% de perturbación gaussiana
    perturbation_std: float = 0.05
    
    # Factor de penalización por varianza (λ en Score = μ - λ·σ)
    # Mayor λ = más aversión a la inestabilidad
    lambda_penalty: float = 1.5
    
    # Semilla base para reproducibilidad
    seed: Optional[int] = 42
    
    # Parámetros a excluir de la perturbación (internos, salidas, etc.)
    exclude_prefixes: Tuple[str, ...] = ("__", "exit_", "cantidad")
    
    # Activar/desactivar el sistema
    enabled: bool = True


@dataclass
class NeighborhoodResult:
    """
    Resultado del análisis de vecindario para un trial.
    
    Contiene todas las métricas necesarias para la Trinidad de Objetivos.
    """
    
    # Métricas del trial original
    original_score: float = 0.0
    original_sharpe: float = 0.0
    original_cvar: float = 0.0
    original_r2: float = 0.0
    original_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Métricas agregadas del vecindario
    neighbor_scores: List[float] = field(default_factory=list)
    neighbor_sharpes: List[float] = field(default_factory=list)
    neighbor_cvars: List[float] = field(default_factory=list)
    neighbor_r2s: List[float] = field(default_factory=list)
    neighbor_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scores finales de la Trinidad
    robust_dsr: float = 0.0           # Objetivo 1: MAXIMIZAR
    worst_case_cvar: float = 100.0    # Objetivo 2: MINIMIZAR
    equity_stability_r2: float = 0.0  # Objetivo 3: MAXIMIZAR
    
    # Score agregado final (para TPE single-objective)
    aggregated_score: float = 0.0
    
    # Estadísticas del vecindario
    mean_score: float = 0.0
    std_score: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    
    # Metadatos
    n_neighbors_tested: int = 0
    n_neighbors_successful: int = 0
    execution_time_ms: float = 0.0
    trial_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_score": self.original_score,
            "original_sharpe": self.original_sharpe,
            "original_cvar": self.original_cvar,
            "original_r2": self.original_r2,
            "robust_dsr": self.robust_dsr,
            "worst_case_cvar": self.worst_case_cvar,
            "equity_stability_r2": self.equity_stability_r2,
            "aggregated_score": self.aggregated_score,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "n_neighbors_tested": self.n_neighbors_tested,
            "n_neighbors_successful": self.n_neighbors_successful,
            "execution_time_ms": self.execution_time_ms,
            "trial_number": self.trial_number,
            # Listas de métricas de vecinos (necesarias para el reporter)
            "neighbor_scores": list(self.neighbor_scores),
            "neighbor_sharpes": list(self.neighbor_sharpes),
            "neighbor_cvars": list(self.neighbor_cvars),
            "neighbor_r2s": list(self.neighbor_r2s),
            "neighbor_metrics": [dict(m) for m in self.neighbor_metrics],
        }


# =============================================================================
# FUNCIONES DE MÉTRICAS ROBUSTAS
# =============================================================================

def _safe_get(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Extrae un valor numérico de forma segura."""
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


def calculate_sharpe_ratio(metrics: Dict[str, Any]) -> float:
    """
    Calcula el Sharpe Ratio a partir de las métricas.
    Si ya existe en metrics, lo usa directamente.
    """
    # Intentar usar Sharpe existente
    sharpe = _safe_get(metrics, "sharpe", 0.0)
    if sharpe != 0.0:
        return sharpe
    
    sharpe = _safe_get(metrics, "sharpe_ratio", 0.0)
    if sharpe != 0.0:
        return sharpe
    
    # Calcular desde ROI y volatilidad si están disponibles
    roi = _safe_get(metrics, "roi", 0.0)
    volatility = _safe_get(metrics, "volatility", 0.0)
    
    if volatility > 0.001:
        return roi / volatility
    
    return 0.0


def calculate_cvar_95(metrics: Dict[str, Any], equity_curve: Optional[List[float]] = None) -> float:
    """
    Calcula el CVaR 95% (Conditional Value at Risk).
    
    CVaR mide la pérdida promedio esperada en el peor 5% de los casos.
    Un valor MÁS BAJO es MEJOR (menos riesgo de cola).
    """
    # Si ya está calculado, usarlo
    cvar = _safe_get(metrics, "cvar_95", 0.0)
    if cvar != 0.0:
        return abs(cvar)
    
    cvar = _safe_get(metrics, "cvar", 0.0)
    if cvar != 0.0:
        return abs(cvar)
    
    # Calcular desde equity curve si está disponible
    if equity_curve and len(equity_curve) > 20:
        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
        returns = returns[np.isfinite(returns)]
        
        if len(returns) > 10:
            # CVaR 95% = promedio del peor 5%
            sorted_returns = np.sort(returns)
            n_tail = max(1, int(len(sorted_returns) * 0.05))
            cvar = -np.mean(sorted_returns[:n_tail])  # Negativo porque son pérdidas
            return max(0.0, cvar * 100)  # Convertir a porcentaje
    
    # Fallback: usar drawdown como proxy
    drawdown = _safe_get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _safe_get(metrics, "max_drawdown", 50.0)
    
    return abs(drawdown)


def calculate_equity_r2(equity_curve: Optional[List[float]] = None, metrics: Dict[str, Any] = None) -> float:
    """
    Calcula el R² de la curva de equity (ajuste lineal).
    
    Mide qué tan "recta" es la curva de ganancias.
    Un R² cercano a 1.0 indica crecimiento consistente.
    """
    if metrics:
        r2 = _safe_get(metrics, "equity_r2", 0.0)
        if r2 != 0.0:
            return r2
        r2 = _safe_get(metrics, "r2", 0.0)
        if r2 != 0.0:
            return r2
    
    if not equity_curve or len(equity_curve) < 10:
        return 0.0
    
    try:
        equity_arr = np.array(equity_curve, dtype=np.float64)
        
        # Usar log para estabilidad
        equity_log = np.log(np.maximum(equity_arr, 1e-10))
        
        # Ajustar línea recta
        x = np.arange(len(equity_log))
        
        # Calcular R²
        correlation_matrix = np.corrcoef(x, equity_log)
        correlation = correlation_matrix[0, 1]
        r2 = correlation ** 2
        
        return max(0.0, min(1.0, r2))
    except Exception:
        return 0.0


def deflated_sharpe_ratio(
    sharpe_obs: float,
    trial_number: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    DEFLATED SHARPE RATIO (DSR) - Detector de Suerte.
    
    Descuenta el Sharpe observado por el número de trials probados.
    Si encontraste una estrategia en el trial 2000, necesita un Sharpe
    mucho más alto que una encontrada en el trial 10.
    
    Basado en: Bailey & López de Prado (2014)
    """
    if n_trades < 10 or trial_number < 1:
        return 0.0
    
    if sharpe_obs <= 0:
        return 0.0
    
    # Ajuste por no-normalidad
    sr_var_factor = 1.0 + 0.5 * sharpe_obs**2 - skewness * sharpe_obs + (kurtosis - 3) / 4 * sharpe_obs**2
    sr_var_factor = max(sr_var_factor, 0.5)
    
    sr_std = math.sqrt(sr_var_factor / n_trades)
    
    # Sharpe esperado bajo selección múltiple
    # Usamos trial_number como proxy del número de trials probados
    n_effective_trials = max(1, trial_number + 1)
    expected_max_sr = sr_std * math.sqrt(2 * math.log(n_effective_trials))
    
    # DSR = P(SR_obs > SR_esperado_por_azar)
    if sr_std > 0:
        z_score = (sharpe_obs - expected_max_sr) / sr_std
        dsr = float(scipy_stats.norm.cdf(z_score))
    else:
        dsr = 0.5
    
    return max(0.0, min(1.0, dsr))


# =============================================================================
# GENERACIÓN DE VECINOS (PERTURBACIÓN GAUSSIANA)
# =============================================================================

def generate_gaussian_neighbors(
    params: Dict[str, Any],
    n_neighbors: int,
    perturbation_std: float,
    exclude_prefixes: Tuple[str, ...],
    seed: Optional[int] = None,
    trial_number: int = 0,
) -> List[Dict[str, Any]]:
    """
    Genera K vecinos usando ruido gaussiano alrededor de los parámetros base.
    
    Para cada parámetro numérico, añade ruido N(0, σ) donde σ = valor * perturbation_std.
    
    Args:
        params: Parámetros originales del trial
        n_neighbors: Número de vecinos a generar
        perturbation_std: Desviación estándar como fracción del valor (ej: 0.05 = 5%)
        exclude_prefixes: Prefijos de parámetros a excluir
        seed: Semilla para reproducibilidad
        trial_number: Número del trial (para generar semillas únicas)
    
    Returns:
        Lista de diccionarios con parámetros perturbados
    """
    # Semilla única por trial
    actual_seed = (seed or 42) + trial_number * 1000
    rng = np.random.default_rng(actual_seed)
    
    neighbors = []
    
    # Identificar parámetros perturbables
    perturbable_params = {}
    for key, value in params.items():
        # Excluir por prefijo
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Solo numéricos (no booleanos)
        if isinstance(value, bool):
            continue
        
        if isinstance(value, (int, float)):
            perturbable_params[key] = value
    
    if not perturbable_params:
        # Sin parámetros perturbables, devolver copias del original
        return [dict(params) for _ in range(n_neighbors)]
    
    # Generar K vecinos
    for _ in range(n_neighbors):
        neighbor = dict(params)  # Copia del original
        
        for key, original_value in perturbable_params.items():
            # Calcular desviación estándar para este parámetro
            if abs(original_value) < 1e-10:
                # Si el valor es ~0, usar perturbación absoluta pequeña
                sigma = perturbation_std
            else:
                sigma = abs(original_value) * perturbation_std
            
            # Generar ruido gaussiano
            noise = rng.normal(0, sigma)
            new_value = original_value + noise
            
            # Mantener tipo original
            if isinstance(original_value, int):
                new_value = int(round(new_value))
                # Asegurar que ints sean al menos 1 si el original era positivo
                if original_value > 0:
                    new_value = max(1, new_value)
            else:
                # Asegurar que floats no sean negativos si el original era positivo
                if original_value > 0:
                    new_value = max(1e-10, new_value)
            
            neighbor[key] = new_value
        
        neighbors.append(neighbor)
    
    return neighbors


# =============================================================================
# CÁLCULO DE LA TRINIDAD DE OBJETIVOS
# =============================================================================

def calculate_robust_dsr(
    sharpes: List[float],
    trial_number: int,
    n_trades: int,
    lambda_penalty: float,
) -> float:
    """
    Calcula el Robust DSR (Objective 1: MAXIMIZAR).
    
    Fórmula: Robust_DSR = DSR(μ_sharpe - λ·σ_sharpe)
    
    1. Calcula media y desviación estándar de Sharpes del vecindario
    2. Aplica penalización por varianza: μ - λ·σ
    3. Aplica DSR para penalizar por número de trials
    """
    if not sharpes:
        return 0.0
    
    sharpes_arr = np.array(sharpes)
    
    # Media y std del vecindario
    mu_sharpe = float(np.mean(sharpes_arr))
    sigma_sharpe = float(np.std(sharpes_arr)) if len(sharpes_arr) > 1 else 0.0
    
    # Sharpe penalizado por varianza vecinal
    penalized_sharpe = mu_sharpe - lambda_penalty * sigma_sharpe
    
    # Aplicar DSR para penalizar por data mining
    robust_dsr = deflated_sharpe_ratio(
        sharpe_obs=penalized_sharpe,
        trial_number=trial_number,
        n_trades=n_trades,
    )
    
    return robust_dsr


def calculate_worst_case_cvar(cvars: List[float]) -> float:
    """
    Calcula el Worst-Case CVaR (Objective 2: MINIMIZAR).
    
    Filosofía: "Tu estrategia es tan segura como su versión vecina más peligrosa."
    
    Retorna el MÁXIMO CVaR (el peor caso) entre todos los vecinos.
    """
    if not cvars:
        return 100.0  # Peor caso si no hay datos
    
    # El peor CVaR es el más alto (más riesgo)
    return float(max(cvars))


def calculate_avg_equity_r2(r2s: List[float]) -> float:
    """
    Calcula el R² promedio de equity (Objective 3: MAXIMIZAR).
    
    Filosofía: "Prefiero ganar menos pero dormir tranquilo."
    """
    if not r2s:
        return 0.0
    
    return float(np.mean(r2s))


# =============================================================================
# FUNCIÓN PRINCIPAL: EJECUTAR ANÁLISIS DE VECINDARIO
# =============================================================================

def run_neighborhood_analysis(
    *,
    strategy: Any,
    df: Any,
    params: Dict[str, Any],
    original_metrics: Dict[str, Any],
    original_score: float,
    equity_curve: Optional[List[float]],
    config: Any,  # BacktestConfig
    neighborhood_config: NeighborhoodConfig,
    trial_number: int,
    run_backtest_fn: Callable,
    generate_signals_fn: Callable,
) -> NeighborhoodResult:
    """
    Ejecuta el análisis completo de vecindario para un trial.
    
    PROCESO:
    1. Extrae métricas del trial original
    2. Genera K vecinos con ruido gaussiano
    3. Ejecuta backtest para cada vecino
    4. Calcula la Trinidad de Objetivos
    5. Calcula el score agregado final
    
    Args:
        strategy: Estrategia a evaluar
        df: DataFrame con datos OHLCV
        params: Parámetros originales del trial
        original_metrics: Métricas del backtest original
        original_score: Score original del trial
        equity_curve: Curva de equity del trial original
        config: Configuración del backtest
        neighborhood_config: Configuración del análisis vecinal
        trial_number: Número del trial actual
        run_backtest_fn: Función para ejecutar backtest
        generate_signals_fn: Función para generar señales
    
    Returns:
        NeighborhoodResult con todas las métricas y scores
    """
    t_start = time.perf_counter()
    
    cfg = neighborhood_config
    result = NeighborhoodResult()
    result.trial_number = trial_number
    
    # =========================================================================
    # PASO 1: Extraer métricas del trial original
    # =========================================================================
    result.original_score = original_score
    result.original_sharpe = calculate_sharpe_ratio(original_metrics)
    result.original_cvar = calculate_cvar_95(original_metrics, equity_curve)
    result.original_r2 = calculate_equity_r2(equity_curve, original_metrics)
    result.original_metrics = dict(original_metrics)
    
    n_trades = int(_safe_get(original_metrics, "n_trades", 0))
    if n_trades == 0:
        n_trades = int(_safe_get(original_metrics, "total_trades", 0))
    
    # Inicializar listas con valores del original
    all_sharpes = [result.original_sharpe]
    all_cvars = [result.original_cvar]
    all_r2s = [result.original_r2]
    all_scores = [original_score]
    
    # =========================================================================
    # PASO 2: Generar y evaluar vecinos
    # =========================================================================
    if cfg.enabled and cfg.n_neighbors > 0:
        neighbors = generate_gaussian_neighbors(
            params=params,
            n_neighbors=cfg.n_neighbors,
            perturbation_std=cfg.perturbation_std,
            exclude_prefixes=cfg.exclude_prefixes,
            seed=cfg.seed,
            trial_number=trial_number,
        )
        
        result.n_neighbors_tested = len(neighbors)
        
        for neighbor_params in neighbors:
            try:
                # Generar señales con params del vecino
                neighbor_signals = generate_signals_fn(df, strategy, neighbor_params)
                
                # Ejecutar backtest
                trades_df, neighbor_equity, neighbor_metrics = run_backtest_fn(
                    df, neighbor_signals, config, neighbor_params, strategy
                )
                
                if trades_df.is_empty():
                    # Vecino sin trades - contar como fallido
                    continue
                
                # Extraer métricas del vecino
                neighbor_sharpe = calculate_sharpe_ratio(neighbor_metrics)
                neighbor_cvar = calculate_cvar_95(neighbor_metrics, neighbor_equity)
                neighbor_r2 = calculate_equity_r2(neighbor_equity, neighbor_metrics)
                
                # Calcular score del vecino (usando la misma función que el original)
                from .scoring import score_optuna
                neighbor_score = float(score_optuna(neighbor_metrics))
                
                # Agregar a las listas
                all_sharpes.append(neighbor_sharpe)
                all_cvars.append(neighbor_cvar)
                all_r2s.append(neighbor_r2)
                all_scores.append(neighbor_score)
                
                result.neighbor_sharpes.append(neighbor_sharpe)
                result.neighbor_cvars.append(neighbor_cvar)
                result.neighbor_r2s.append(neighbor_r2)
                result.neighbor_scores.append(neighbor_score)
                result.neighbor_metrics.append(neighbor_metrics)
                result.n_neighbors_successful += 1
                
            except Exception:
                # Vecino falló - ignorar
                continue
    
    # =========================================================================
    # PASO 3: Calcular estadísticas del vecindario
    # =========================================================================
    result.mean_score = float(np.mean(all_scores))
    result.std_score = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
    result.mean_sharpe = float(np.mean(all_sharpes))
    result.std_sharpe = float(np.std(all_sharpes)) if len(all_sharpes) > 1 else 0.0
    
    # =========================================================================
    # PASO 4: Calcular la Trinidad de Objetivos
    # =========================================================================
    
    # Objetivo 1: Robust_DSR (MAXIMIZAR)
    result.robust_dsr = calculate_robust_dsr(
        sharpes=all_sharpes,
        trial_number=trial_number,
        n_trades=n_trades,
        lambda_penalty=cfg.lambda_penalty,
    )
    
    # Objetivo 2: Worst_Case_CVaR (MINIMIZAR)
    result.worst_case_cvar = calculate_worst_case_cvar(all_cvars)
    
    # Objetivo 3: Equity_Stability_R2 (MAXIMIZAR)
    result.equity_stability_r2 = calculate_avg_equity_r2(all_r2s)
    
    # =========================================================================
    # PASO 5: Calcular Score Agregado (para TPE single-objective)
    # =========================================================================
    # Fórmula: Score = μ_M - λ·σ_M
    result.aggregated_score = result.mean_score - cfg.lambda_penalty * result.std_score
    
    # Asegurar que no sea negativo
    result.aggregated_score = max(0.0, result.aggregated_score)
    
    result.execution_time_ms = (time.perf_counter() - t_start) * 1000
    
    return result


def nsga2_objectives_robust(result: NeighborhoodResult) -> Tuple[float, float, float]:
    """
    Retorna la Trinidad de Objetivos para NSGA-II.
    
    Returns:
        (robust_dsr, worst_case_cvar, equity_stability_r2)
        
        - robust_dsr: MAXIMIZAR (mayor es mejor)
        - worst_case_cvar: MINIMIZAR (menor es mejor)
        - equity_stability_r2: MAXIMIZAR (mayor es mejor)
    """
    return (
        result.robust_dsr,
        result.worst_case_cvar,
        result.equity_stability_r2,
    )


# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================

DEFAULT_NEIGHBORHOOD_CONFIG = NeighborhoodConfig()

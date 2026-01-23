from __future__ import annotations
from typing import Any, Mapping, List, Dict, Optional, Callable
import math
import numpy as np
from scipy import stats as scipy_stats


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA DE ROBUSTEZ
# =============================================================================

# Pesos para scoring robusto
WEIGHT_DSR = 0.35          # Deflated Sharpe Ratio (anti-suerte)
WEIGHT_CVAR = 0.25         # CVaR 95% (control de riesgo de cola)
WEIGHT_R2 = 0.25           # Equity R² (estabilidad)
WEIGHT_QUALITY = 0.15      # Calidad base tradicional

# Configuración de análisis de vecindario
NEIGHBOR_PERTURBATION = 0.10  # ±10% de perturbación para vecinos
N_NEIGHBORS_PER_PARAM = 2     # Vecinos por lado por parámetro
PLATEAU_STABILITY_THRESHOLD = 0.70  # % mínimo de vecinos exitosos para "meseta"


def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Helper seguro para extraer valores numéricos de las métricas."""
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


def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SCORE DE CALIDAD PURA (Para optimización Multi-Objetivo)
    ═══════════════════════════════════════════════════════════════════════════

    OBJETIVO:
    Medir exclusivamente la "Fuerza" y "Robustez" de la estrategia para ganar dinero.

    FILTROS HARD-CUT:
    1. Frecuencia Mínima: Si trades_por_dia < 0.25 -> Score 0.1 (Descarte inmediato)

    IMPORTANTE:
    NO penalizamos Drawdown aquí. El Drawdown se minimiza como un
    OBJETIVO INDEPENDIENTE en el optimizador NSGA-II.
    ═══════════════════════════════════════════════════════════════════════════
    """

    # 1. Extracción de Datos Básicos
    n_trades = _get(metrics, "n_trades", 0.0)
    if n_trades == 0:
        n_trades = _get(metrics, "total_trades", 0.0)

    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)

    # ═════════════════════════════════════════════════════════════════════════
    # HARD CUT: FILTRO DE ACTIVIDAD MÍNIMA
    # ═════════════════════════════════════════════════════════════════════════
    if trades_dia < 0.25:
        return 0.1
    # ═════════════════════════════════════════════════════════════════════════

    winrate_raw = _get(metrics, "winrate", 0.0)
    # Normalizar winrate: si viene como porcentaje (>1) dividir por 100
    # Ejemplo: 65.0 -> 0.65, pero 0.65 se mantiene como 0.65
    if winrate_raw > 1.0:
        winrate = winrate_raw / 100.0
    else:
        winrate = winrate_raw
    
    # Validar rango de winrate [0, 1]
    winrate = max(0.0, min(1.0, winrate))

    payoff_ratio = _get(metrics, "payoff_ratio", 0.0)
    # Protección contra payoff_ratio inválido (NaN o muy grande)
    if math.isnan(payoff_ratio) or payoff_ratio < 0 or payoff_ratio > 100:
        payoff_ratio = 0.0
    
    profit_factor = _get(metrics, "profit_factor", 0.0)
    # Protección contra profit_factor inválido
    if math.isnan(profit_factor) or profit_factor < 0:
        profit_factor = 0.0
    
    sqn_val = _get(metrics, "sqn", 0.0)

    max_ganancia = _get(metrics, "max_ganancia", 0.0)
    pnl_neto = _get(metrics, "pnl_neto", 0.0)
    if pnl_neto == 0:
        pnl_neto = _get(metrics, "net_pnl", 0.0)

    roi = _get(metrics, "roi", 0.0)
    if roi == 0:
        roi = _get(metrics, "roi_pct", 0.0)

    # 2. Cálculo de Ventaja Estadística (Edge)
    # Edge = Expectativa Normalizada = (Win% * Payoff) - (Loss% * 1.0)
    # Donde Loss% = 1 - Win%, y asumimos que la pérdida promedio normalizada es 1.0
    # 
    # Fórmula equivalente derivada de Kelly:
    #   E[R] = P(win) * avg_win - P(loss) * avg_loss
    #   Edge = E[R] / avg_loss = P(win) * (avg_win/avg_loss) - P(loss)
    #        = winrate * payoff_ratio - (1 - winrate)
    #
    # Interpretación:
    #   Edge > 0: Estrategia con ventaja estadística
    #   Edge = 0: Estrategia sin ventaja (break-even antes de costos)
    #   Edge < 0: Estrategia con desventaja
    if payoff_ratio > 0 and not math.isnan(payoff_ratio):
        edge = (winrate * payoff_ratio) - (1.0 - winrate)
    else:
        # Sin payoff válido, no hay edge calculable
        edge = -1.0

    # 3. Factores base
    evidence_factor = n_trades / (n_trades + 50.0)
    sqn_score = math.log(1.0 + max(0.0, sqn_val) * 2.0) if sqn_val > 0 else 0.0
    
    # Profit factor bonus: limitar entre 0.1 y 3.0
    # Si profit_factor es 0 o NaN, usar 1.0 (neutro)
    if profit_factor > 0:
        pf_bonus = min(max(profit_factor, 0.1), 3.0)
    else:
        pf_bonus = 1.0

    # 4. Si la estrategia tiene edge negativo, usar PF como base
    # Esto permite diferenciar entre estrategias "malas" y "muy malas"
    if edge <= 0:
        # Score basado en profit_factor cuando edge es negativo
        # PF=0.5 -> score ~5, PF=0.8 -> score ~8, PF=1.0 -> score ~10
        base_score = profit_factor * 10.0 * evidence_factor
        # Añadir pequeño bonus por ROI si es positivo
        if roi > 0:
            base_score += min(roi / 10.0, 5.0)
        return max(0.1, min(base_score, 15.0))  # Cap en 15 para estrategias perdedoras

    # 5. Penalización de Concentración (anti-overfitting)
    concentration = 0.0
    if pnl_neto > 0 and max_ganancia > 0:
        concentration = max_ganancia / pnl_neto

    conc_penalty = 1.0
    if concentration > 0.30:
        conc_penalty = math.exp(-3.0 * (concentration - 0.30))

    # 6. Cálculo Final para estrategias con edge positivo
    raw_score = (edge * 100.0) * pf_bonus * (1.0 + sqn_score) * evidence_factor * conc_penalty

    return max(0.1, float(raw_score))


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """
    LEGACY: Mantenido por compatibilidad si se usa TPE (Single Objective).
    Usa una versión simplificada pero robusta.
    """
    quality = score_quality_only(metrics)

    # Si quality devolvió 0.1 (Hard Cut), devolvemos eso directamente
    if quality <= 0.101:
        return quality

    drawdown = _get(metrics, "drawdown", 100.0)

    # Penalización suave de drawdown para single-objective
    dd_factor = 1.0 / (1.0 + (drawdown / 20.0))

    return quality * dd_factor


def nsga2_objectives(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    OBJETIVOS PARA NSGA-II (Multi-Objetivo)
    ═══════════════════════════════════════════════════════════════════════════

    Retorna una tupla de dos valores:
    1. quality (MAXIMIZAR): Score de calidad pura de la estrategia
    2. drawdown (MINIMIZAR): Drawdown máximo en porcentaje

    NSGA-II optimiza ambos simultáneamente, buscando el Frente de Pareto.
    ═══════════════════════════════════════════════════════════════════════════
    """
    quality = score_quality_only(metrics)
    drawdown = _get(metrics, "drawdown", 100.0)

    # Asegurar valores válidos
    quality = max(0.1, float(quality))
    drawdown = max(0.0, min(100.0, float(drawdown)))

    return (quality, drawdown)


# =============================================================================
# MÉTRICAS ROBUSTAS AVANZADAS (Anti-Overfitting)
# =============================================================================

def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    DEFLATED SHARPE RATIO (DSR) - "El Detector de Suerte"
    ═══════════════════════════════════════════════════════════════════════════

    Descuenta el Sharpe Ratio observado por el sesgo de selección múltiple.

    Basado en: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
    and Non-Normality"

    Args:
        sharpe_obs: Sharpe Ratio observado de la estrategia
        n_trials: Número de backtests/configuraciones probadas
        n_trades: Número de trades en el backtest
        skewness: Asimetría de los retornos (0 = normal)
        kurtosis: Curtosis de los retornos (3 = normal)

    Returns:
        DSR: Probabilidad [0, 1] de que el Sharpe sea genuino y no fruto del azar.
             Valores > 0.95 indican alta confianza estadística.
    ═══════════════════════════════════════════════════════════════════════════
    """
    if n_trades < 10 or n_trials < 1:
        return 0.0

    if sharpe_obs <= 0:
        return 0.0

    # Ajuste por no-normalidad de retornos
    sr_var_factor = 1.0 + 0.5 * sharpe_obs**2 - skewness * sharpe_obs + (kurtosis - 3) / 4 * sharpe_obs**2
    sr_var_factor = max(sr_var_factor, 0.5)

    sr_std = math.sqrt(sr_var_factor / n_trades)

    # Sharpe esperado bajo selección múltiple (aproximación de Bonferroni)
    if n_trials > 1:
        expected_max_sr = sr_std * math.sqrt(2 * math.log(n_trials))
    else:
        expected_max_sr = 0.0

    # DSR = P(SR_obs > SR_esperado_por_azar)
    if sr_std > 0:
        z_score = (sharpe_obs - expected_max_sr) / sr_std
        dsr = float(scipy_stats.norm.cdf(z_score))
    else:
        dsr = 0.5

    return max(0.0, min(1.0, dsr))


def cvar_95(returns: np.ndarray) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    CONDITIONAL VALUE AT RISK 95% (CVaR) - "El Cinturón de Seguridad"
    ═══════════════════════════════════════════════════════════════════════════

    Mide la pérdida promedio en el peor 5% de los casos.

    Args:
        returns: Array de retornos por trade (en decimales)

    Returns:
        CVaR como valor positivo (menor es mejor/más seguro).
    ═══════════════════════════════════════════════════════════════════════════
    """
    if len(returns) < 5:
        return 0.0

    returns = returns[np.isfinite(returns)]
    if len(returns) < 5:
        return 0.0

    var_5 = np.percentile(returns, 5)
    tail_returns = returns[returns <= var_5]

    if len(tail_returns) == 0:
        return abs(var_5) if var_5 < 0 else 0.0

    cvar = -float(np.mean(tail_returns))
    return max(0.0, cvar)


def equity_r_squared(equity_curve: List[float]) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    EQUITY R² - "La Prueba de la Regla"
    ═══════════════════════════════════════════════════════════════════════════

    Mide qué tan bien se ajusta la curva de equity a una línea recta perfecta.

    Returns:
        R² entre 0 y 1:
        - R² > 0.95: Curva casi perfectamente recta (ideal)
        - R² > 0.80: Buena consistencia
        - R² < 0.50: Alta volatilidad en los resultados
    ═══════════════════════════════════════════════════════════════════════════
    """
    if equity_curve is None or (hasattr(equity_curve, '__len__') and len(equity_curve) < 3):
        return 0.0

    y = np.asarray(equity_curve, dtype=np.float64)
    n = len(y)
    x = np.arange(n, dtype=np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return max(0.0, min(1.0, float(r2)))


def compute_robust_metrics(
    metrics: Mapping[str, Any],
    equity_curve: Optional[List[float]] = None,
    returns: Optional[np.ndarray] = None,
    n_trials: int = 1,
) -> Dict[str, float]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MÉTRICAS ROBUSTAS COMBINADAS
    ═══════════════════════════════════════════════════════════════════════════

    Calcula DSR, CVaR 95%, y Equity R² para evaluación anti-overfitting.
    """
    result = {
        "dsr": 0.0,
        "cvar_95": 0.0,
        "equity_r2": 0.0,
    }

    n_trades = int(_get(metrics, "n_trades", 0))
    if n_trades == 0:
        n_trades = int(_get(metrics, "total_trades", 0))

    if n_trades < 5:
        return result

    # 1. Calcular DSR
    sharpe = _get(metrics, "sharpe", 0.0)
    if sharpe > 0 and n_trades >= 10:
        # Aproximar skewness y kurtosis si no los tenemos
        result["dsr"] = deflated_sharpe_ratio(
            sharpe_obs=sharpe,
            n_trials=max(1, n_trials),
            n_trades=n_trades,
            skewness=0.0,  # Asumir distribución simétrica
            kurtosis=3.0,  # Asumir distribución normal
        )

    # 2. Calcular CVaR si tenemos retornos
    if returns is not None and len(returns) >= 5:
        result["cvar_95"] = cvar_95(returns)

    # 3. Calcular Equity R²
    if equity_curve and len(equity_curve) >= 3:
        result["equity_r2"] = equity_r_squared(equity_curve)

    return result


# =============================================================================
# SCORING ROBUSTO ANTI-OVERFITTING
# =============================================================================

def score_robust(
    metrics: Mapping[str, Any],
    equity_curve: Optional[List[float]] = None,
    returns: Optional[np.ndarray] = None,
    n_trials: int = 1,
) -> float:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SCORE ROBUSTO COMBINADO (Anti-Overfitting)
    ═══════════════════════════════════════════════════════════════════════════

    Combina múltiples métricas para penalizar overfitting:

    1. DSR (Deflated Sharpe): ¿Es el resultado suerte o skill?
    2. CVaR 95%: ¿Cuánto perdemos en el peor 5%?
    3. Equity R²: ¿Es la curva estable o errática?
    4. Quality base: Métricas tradicionales (edge, PF, SQN)

    Fórmula:
        Score = w_DSR*DSR + w_R2*R² + w_CVaR*(1-CVaR_norm) + w_Q*Quality_norm

    Returns:
        Score [0, 100] donde mayor es mejor y más robusto.
    ═══════════════════════════════════════════════════════════════════════════
    """
    # 1. Filtro básico
    n_trades = _get(metrics, "n_trades", 0)
    if n_trades == 0:
        n_trades = _get(metrics, "total_trades", 0)

    trades_dia = _get(metrics, "trades_por_dia", 0)

    if trades_dia < 0.25 or n_trades < 10:
        return 0.1

    # 2. Calcular métricas robustas
    robust = compute_robust_metrics(metrics, equity_curve, returns, n_trials)

    dsr = robust["dsr"]
    cvar = robust["cvar_95"]
    r2 = robust["equity_r2"]

    # 3. Normalizar CVaR (queremos minimizar, así que invertimos)
    # CVaR típico entre 0% y 20%, normalizado a [0, 1]
    cvar_score = max(0.0, 1.0 - (cvar / 0.20))

    # 4. Calcular calidad base normalizada
    quality_raw = score_quality_only(metrics)
    quality_norm = min(1.0, quality_raw / 100.0)  # Normalizar a [0, 1]

    # 5. Si no tenemos métricas robustas, usar calidad base
    if dsr == 0 and r2 == 0:
        return quality_raw

    # 6. Combinar ponderadamente
    combined = (
        WEIGHT_DSR * dsr * 100 +
        WEIGHT_R2 * r2 * 100 +
        WEIGHT_CVAR * cvar_score * 100 +
        WEIGHT_QUALITY * quality_norm * 100
    )

    # 7. Factor de evidencia (más trades = más confianza)
    evidence = n_trades / (n_trades + 50.0)
    combined *= (0.6 + 0.4 * evidence)

    return max(0.1, float(combined))


# =============================================================================
# ANÁLISIS DE VECINDARIO (Búsqueda de Mesetas)
# =============================================================================

def perturb_params(
    params: Dict[str, Any],
    param_ranges: Dict[str, tuple],
    perturbation_pct: float = NEIGHBOR_PERTURBATION,
) -> List[Dict[str, Any]]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    GENERADOR DE VECINOS - Perturbación de Parámetros
    ═══════════════════════════════════════════════════════════════════════════

    Genera vecinos perturbando cada parámetro numérico ±X%.
    Esto permite verificar si estamos en una meseta o en un punto aislado.

    Args:
        params: Parámetros originales del trial
        param_ranges: Rangos válidos {param: (min, max, step)}
        perturbation_pct: Porcentaje de perturbación (default 10%)

    Returns:
        Lista de diccionarios con parámetros perturbados
    ═══════════════════════════════════════════════════════════════════════════
    """
    neighbors = []

    for param_name, value in params.items():
        # Ignorar parámetros internos
        if param_name.startswith("__") or param_name.startswith("exit_"):
            continue

        # Solo perturbar numéricos
        if not isinstance(value, (int, float)):
            continue

        # Obtener rango si existe
        if param_name in param_ranges:
            p_min, p_max, p_step = param_ranges[param_name]
        else:
            # Estimar rango basado en el valor
            p_min = value * 0.5
            p_max = value * 1.5
            p_step = (p_max - p_min) / 20

        # Calcular perturbación
        delta = abs(value) * perturbation_pct
        if delta < p_step:
            delta = p_step

        # Generar vecinos hacia arriba y abajo
        for direction in [-1, 1]:
            for i in range(1, N_NEIGHBORS_PER_PARAM + 1):
                new_value = value + direction * delta * i

                # Clip al rango válido
                new_value = max(p_min, min(p_max, new_value))

                # Respetar step si es entero
                if isinstance(value, int):
                    new_value = int(round(new_value / p_step) * p_step)

                # Crear vecino
                neighbor = params.copy()
                neighbor[param_name] = new_value
                neighbors.append(neighbor)

    return neighbors


def analyze_neighborhood_stability(
    scores: List[float],
    threshold: float = PLATEAU_STABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    ANÁLISIS DE ESTABILIDAD DEL VECINDARIO
    ═══════════════════════════════════════════════════════════════════════════

    Determina si estamos en una "meseta" (zona robusta) o en un "pico" aislado.

    Una meseta es deseable porque indica que pequeños cambios en los parámetros
    no destruyen el rendimiento -> ROBUSTO.

    Un pico aislado es peligroso porque cualquier desviación del parámetro
    óptimo causa caída de performance -> OVERFITTING.

    Args:
        scores: Lista de scores del centro + vecinos
        threshold: % mínimo de vecinos con score aceptable

    Returns:
        Dict con métricas de estabilidad:
        - is_plateau: True si es una meseta estable
        - stability_score: [0, 1] donde 1 = perfectamente estable
        - mean_score: Promedio de todos los scores
        - std_score: Desviación estándar
        - cv: Coeficiente de variación
    ═══════════════════════════════════════════════════════════════════════════
    """
    if not scores or len(scores) < 2:
        return {
            "is_plateau": False,
            "stability_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "cv": float("inf"),
        }

    arr = np.array(scores)
    center_score = arr[0]  # Primer elemento es el centro
    neighbor_scores = arr[1:] if len(arr) > 1 else arr

    mean_score = float(np.mean(arr))
    std_score = float(np.std(arr))

    # Coeficiente de variación (menor es más estable)
    cv = std_score / mean_score if mean_score > 0 else float("inf")

    # % de vecinos con score >= 70% del centro
    if center_score > 0:
        good_neighbors = np.sum(neighbor_scores >= 0.7 * center_score)
        stability = good_neighbors / len(neighbor_scores) if len(neighbor_scores) > 0 else 0.0
    else:
        stability = 0.0

    # Es meseta si stability >= threshold
    is_plateau = stability >= threshold

    return {
        "is_plateau": is_plateau,
        "stability_score": float(stability),
        "mean_score": mean_score,
        "std_score": std_score,
        "cv": cv,
    }

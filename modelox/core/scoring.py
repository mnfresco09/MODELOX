"""modelox/core/scoring.py

═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE SCORING UNIFICADO v8.0 - CALIDAD + ROBUSTEZ (SUMA)
═══════════════════════════════════════════════════════════════════════════════

FILOSOFÍA:
El Score ES UNA SUMA de dos componentes:

    Score = Puntos_Calidad + Puntos_Robustez
    
    Máximo total: 1000 puntos (600 calidad + 400 robustez)

1. COMPONENTE DE CALIDAD (Puntos_Calidad):
   - Rango: [0, 600]
   - Se normaliza Sharpe, SQN, ROI, Drawdown usando funciones tanh (sigmoides)
   - Rendimientos decrecientes: mejorar Sharpe de 2→3 da más puntos que 5→6
   - Factor de actividad se aplica como multiplicador sobre la calidad
   - Escalas ampliadas para dificultar alcanzar el máximo

2. COMPONENTE DE ACTIVIDAD (Factor sobre Calidad):
   - Función: Sigmoide logística centrada en 0.25 trades/día
   - Si trades/día < 0.25 → el factor cae suavemente hacia 0
   - Si trades/día >= 0.5 → factor ~1.0
   - Se aplica multiplicando los puntos de calidad

3. COMPONENTE DE ROBUSTEZ (Puntos_Robustez):
   - Rango: [0, 400]
   
   - CASO A (Sin test de vecindario): 0 puntos
     → Techo de cristal: máximo 600 puntos sin test de robustez
   
   - CASO B (Con test de vecindario):
     → Puntos = 400 × e^(-3.0 × Incertidumbre)
     
     Mecánica:
     - Dispersión 0% → 400 puntos (total posible: 1000)
     - Dispersión ~15% → ~255 puntos
     - Dispersión ~30% → ~162 puntos
     - Dispersión ~50% → ~89 puntos

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

# =============================================================================
# CONFIGURACIÓN IMPORTADA
# =============================================================================
try:
    from general.configuracion import (
        VECINDARIO_N_NEIGHBORS,
        VECINDARIO_MAX_DISPERSION,
        VECINDARIO_PERTURBATION_STD,
        VECINDARIO_LAMBDA_PENALTY,
        VECINDARIO_SEED,
        VECINDARIO_MIN_TRADES_DIA,
        VECINDARIO_MIN_PROFIT_FACTOR,
        VECINDARIO_MIN_SHARPE,
    )
    _CONFIG_LOADED = True
except ImportError:
    VECINDARIO_N_NEIGHBORS = 8
    VECINDARIO_MAX_DISPERSION = 0.40
    VECINDARIO_PERTURBATION_STD = 0.05
    VECINDARIO_LAMBDA_PENALTY = 1.5
    VECINDARIO_SEED = 42
    VECINDARIO_MIN_TRADES_DIA = 0.25
    VECINDARIO_MIN_PROFIT_FACTOR = 1.1
    VECINDARIO_MIN_SHARPE = 1.25
    _CONFIG_LOADED = False


# =============================================================================
# ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗██████╗  ██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██╔════╝
# ██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   ██████╔╝██║   ██║███████╗
# ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║   ██║╚════██║
# ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝███████║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
#                                                                                      
# ═══════════════════════════════════════════════════════════════════════════════════
# AJUSTA ESTOS VALORES SEGÚN TUS DATOS - SCORING v8.0
# ═══════════════════════════════════════════════════════════════════════════════════
#
# FÓRMULA DEL SCORE:
#   Score = (Puntos_Calidad × Factor_Actividad) + Puntos_Robustez
#
# ═══════════════════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------------
# 1. DISTRIBUCIÓN DE PUNTOS (deben sumar 1000)
# -----------------------------------------------------------------------------
# ¿Cuántos puntos máximos para cada componente?
#   - Calidad: métricas de rendimiento (Sharpe, SQN, ROI, Drawdown)
#   - Robustez: estabilidad en el test de vecindario

MAX_PUNTOS_CALIDAD = 600.0   # Máximo por buenas métricas (0-600)
MAX_PUNTOS_ROBUSTEZ = 400.0  # Máximo por estabilidad (0-400)
MAX_SCORE = 1000.0           # Total (no cambiar)

# -----------------------------------------------------------------------------
# 2. PESOS DE MÉTRICAS DE CALIDAD (deben sumar 1.0)
# -----------------------------------------------------------------------------
# ¿Qué importancia tiene cada métrica dentro de los puntos de calidad?
# Ejemplo: PESO_SHARPE = 0.35 significa que Sharpe vale el 35% de la calidad

PESO_SHARPE = 0.33     # 35% - Ratio riesgo/retorno
PESO_SQN = 0.27        # 30% - System Quality Number (calidad del sistema)
PESO_DRAWDOWN = 0.17   # 20% - Máxima caída (menor es mejor)
PESO_ROI = 0.23        # 15% - Retorno total

# -----------------------------------------------------------------------------
# 3. ESCALAS DE NORMALIZACIÓN (¡AJUSTAR SEGÚN TUS DATOS!)
# -----------------------------------------------------------------------------
# Cada escala define qué valor de la métrica da ~80% de sus puntos.
# Fórmula: escala = valor_máximo_esperado × 1.2 (aprox)
#
# CÓMO AJUSTAR:
#   1. Mira los máximos de tus datos reales
#   2. Multiplica por 1.2 para dejar margen
#   3. Valores mayores = más difícil alcanzar el máximo
#
# EJEMPLO con tus datos actuales:
#   - Tu mejor Sharpe es 2.0 → escala = 2.4 (para que 2.0 de ~80%)
#   - Tu mejor ROI es 52% → escala = 62 (para que 52% de ~80%)

SHARPE_SCALE = 2.25    # Sharpe 2.0 → ~80%, Sharpe 4.7 → ~95%
SQN_SCALE = 6.0      # SQN 2.4 → ~80%, SQN 5.5 → ~95%
ROI_SCALE = 500.0       # ROI 52% → ~80%, ROI 120% → ~95%
DD_SCALE = 10.0        # DD < 15% para buena puntuación (menor es mejor)

# -----------------------------------------------------------------------------
# 4. FACTOR DE ACTIVIDAD (penaliza estrategias con pocos trades)
# -----------------------------------------------------------------------------
# Sigmoide que penaliza si hay muy pocos trades por día.
# - trades/día < centro → factor baja hacia 0
# - trades/día = centro → factor = 0.5
# - trades/día > centro → factor sube hacia 1.0

ACTIVIDAD_CENTRO = 0.25      # Centro de la sigmoide (trades/día)
ACTIVIDAD_PENDIENTE = 12.0   # Qué tan abrupta es la transición (mayor = más brusca)

# -----------------------------------------------------------------------------
# 5. UMBRALES DE SUSPENSO (estrategias que ni se evalúan)
# -----------------------------------------------------------------------------
# Si una estrategia no cumple estos mínimos, obtiene score ~0

REF_ROI_SUSPENSO = 100.0     # Si ROI < -15% → suspenso
REF_TRADES_DIA_MIN = 0.15    # Si trades/día < 0.15 → suspenso

# -----------------------------------------------------------------------------
# 6. ROBUSTEZ (decaimiento por incertidumbre en vecindario)
# -----------------------------------------------------------------------------
# Puntos_Robustez = MAX_PUNTOS_ROBUSTEZ × e^(-DECAY_FACTOR × incertidumbre)
# Mayor DECAY_FACTOR = penaliza más la dispersión

DECAY_FACTOR = 3.0  # Dispersión 0% → 400p, 15% → ~255p, 30% → ~162p

# ═══════════════════════════════════════════════════════════════════════════════════
# FIN DE PARÁMETROS AJUSTABLES
# ═══════════════════════════════════════════════════════════════════════════════════


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA DE VECINDARIO
# =============================================================================

@dataclass
class NeighborhoodConfig:
    """Configuración del sistema de test de robustez."""
    
    n_neighbors: int = VECINDARIO_N_NEIGHBORS
    max_dispersion: float = VECINDARIO_MAX_DISPERSION
    perturbation_std: float = VECINDARIO_PERTURBATION_STD
    lambda_penalty: float = VECINDARIO_LAMBDA_PENALTY
    seed: Optional[int] = VECINDARIO_SEED
    exclude_prefixes: Tuple[str, ...] = ("__", "exit_", "cantidad")
    enabled: bool = True
    # Criterios para hacer test de vecindario (v7.1)
    min_trades_per_day: float = VECINDARIO_MIN_TRADES_DIA       # > 0.25
    min_profit_factor: float = VECINDARIO_MIN_PROFIT_FACTOR     # > 1.1
    min_sharpe: float = VECINDARIO_MIN_SHARPE                   # > 1.25


@dataclass
class NeighborhoodResult:
    """Resultado del test de robustez."""
    
    original_metrics: Dict[str, Any] = field(default_factory=dict)
    original_score: float = 0.0
    
    n_neighbors_tested: int = 0
    n_neighbors_successful: int = 0
    neighbor_metrics: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_params: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_scores: List[float] = field(default_factory=list)
    
    dispersions: Dict[str, float] = field(default_factory=dict)
    avg_dispersion: float = 1.0
    incertidumbre: float = 1.0  # Nueva: dispersión agregada para v7.0
    
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
    
    # Legacy campos (para compatibilidad)
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
            "neighbor_metrics": [dict(m) for m in self.neighbor_metrics],
            "neighbor_params": [dict(p) for p in self.neighbor_params],
            "neighbor_scores": list(self.neighbor_scores),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "execution_time_ms": self.execution_time_ms,
            "trial_number": self.trial_number,
            "skip_reason": self.skip_reason,
            "original_score": self.original_score,
            "original_sharpe": self.original_sharpe,
            "original_cvar": self.original_cvar,
            "original_r2": self.original_r2,
            "robust_dsr": self.robust_dsr,
            "worst_case_cvar": self.worst_case_cvar,
            "equity_stability_r2": self.equity_stability_r2,
            "neighbor_sharpes": list(self.neighbor_sharpes),
            "neighbor_cvars": list(self.neighbor_cvars),
            "neighbor_r2s": list(self.neighbor_r2s),
        }


DEFAULT_NEIGHBORHOOD_CONFIG = NeighborhoodConfig()


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Helper seguro para extraer valores numéricos."""
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


def _calculate_cv(values: List[float]) -> float:
    """
    Calcula el Coeficiente de Variación (CV = std / |mean|).
    """
    if not values or len(values) < 2:
        return 0.0
    
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    
    if len(arr) < 2:
        return 0.0
    
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    if abs(mean_val) < 1e-10:
        return 1.0 if std_val > 0.1 else 0.0
    
    return float(std_val / abs(mean_val))


# =============================================================================
# COMPONENTE 1: CALIDAD_RAW (0-1000)
# =============================================================================

def _tanh_normalize(value: float, scale: float, min_val: float = 0.0) -> float:
    """
    Normaliza un valor usando tanh para rendimientos decrecientes.
    Retorna valor entre 0 y 1.
    
    tanh((value - min_val) / scale) → [0, ~1]
    """
    if value <= min_val:
        return 0.0
    
    normalized = (value - min_val) / scale
    return float(max(0.0, math.tanh(normalized)))


def calculate_calidad_raw(
    sharpe: float,
    sqn: float,
    drawdown: float,
    roi: float,
) -> float:
    """
    Calcula el componente de CALIDAD (0-600).
    
    Usa funciones tanh para normalizar cada métrica con rendimientos decrecientes.
    Escalas calibradas al 118% del máximo observado en datos reales.
    
    Args:
        sharpe: Sharpe ratio (típico: -0.5 a 2.0)
        sqn: System Quality Number (típico: -1 a 2.5)
        drawdown: Máximo drawdown en % (típico: 30 a 70)
        roi: Retorno total en % (típico: -5 a 50)
    
    Returns:
        Puntos de calidad [0, 600]
    """
    # Sharpe (35%): escala 2.4, max observado ~2.0
    # Sharpe 2.0 → ~80%, Sharpe 4.7 → ~95%
    sharpe_norm = _tanh_normalize(sharpe, SHARPE_SCALE, min_val=-0.5)
    
    # SQN (30%): escala 2.8, max observado ~2.4
    # SQN 2.4 → ~80%, SQN 5.5 → ~95%
    sqn_norm = _tanh_normalize(sqn, SQN_SCALE, min_val=0.0)
    
    # ROI (15%): escala 62%, max observado ~52%
    # ROI 52% → ~80%, ROI 120% → ~95%
    roi_norm = _tanh_normalize(roi, ROI_SCALE, min_val=0.0)
    
    # Drawdown (20%): escala 15%, muy exigente
    # DD 0% → 100%, DD 15% → ~24%, DD 30% → ~4%
    dd_norm = 1.0 - math.tanh(max(0.0, drawdown) / DD_SCALE)
    dd_norm = max(0.0, dd_norm)
    
    # Combinar con pesos
    calidad = (
        PESO_SHARPE * sharpe_norm +
        PESO_SQN * sqn_norm +
        PESO_DRAWDOWN * dd_norm +
        PESO_ROI * roi_norm
    )
    
    # Escalar a 0-600 (MAX_PUNTOS_CALIDAD)
    return float(calidad * MAX_PUNTOS_CALIDAD)


# =============================================================================
# COMPONENTE 2: FACTOR_ACTIVIDAD (0-1)
# =============================================================================

def calculate_factor_actividad(trades_dia: float) -> float:
    """
    Calcula el factor de actividad usando sigmoide logística.
    
    Sigmoide centrada en ACTIVIDAD_CENTRO (0.25 trades/día):
    - trades_dia << 0.25 → factor ~0
    - trades_dia = 0.25 → factor = 0.5
    - trades_dia >> 0.25 → factor ~1
    
    Fórmula: 1 / (1 + e^(-k*(x - centro)))
    donde k = ACTIVIDAD_PENDIENTE
    
    Args:
        trades_dia: Número de trades por día
    
    Returns:
        Factor de actividad [0, 1]
    """
    if trades_dia <= 0:
        return 0.0
    
    # Sigmoide logística
    exponent = -ACTIVIDAD_PENDIENTE * (trades_dia - ACTIVIDAD_CENTRO)
    factor = 1.0 / (1.0 + math.exp(exponent))
    
    return float(max(0.0, min(1.0, factor)))


# =============================================================================
# COMPONENTE 3: FACTOR_ROBUSTEZ (0-1)
# =============================================================================

def calculate_incertidumbre(dispersions: Dict[str, float]) -> float:
    """
    Calcula la incertidumbre agregada a partir de las dispersiones de métricas.
    
    Promedio ponderado de dispersiones de ROI, Sharpe y Drawdown.
    
    Args:
        dispersions: Dict con CV de cada métrica {metrica: cv}
    
    Returns:
        Incertidumbre agregada [0, 1+]
    """
    if not dispersions:
        return 1.0
    
    # Usar las dispersiones más relevantes: ROI, Sharpe, Drawdown
    relevant = []
    weights = []
    
    if "roi" in dispersions:
        relevant.append(dispersions["roi"])
        weights.append(0.4)  # ROI tiene más peso
    
    if "sharpe" in dispersions:
        relevant.append(dispersions["sharpe"])
        weights.append(0.35)
    
    if "drawdown" in dispersions:
        relevant.append(dispersions["drawdown"])
        weights.append(0.25)
    
    if not relevant:
        # Fallback: promedio simple de todas
        return float(np.mean(list(dispersions.values())))
    
    # Normalizar pesos
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Promedio ponderado
    incertidumbre = sum(d * w for d, w in zip(relevant, weights))
    
    return float(incertidumbre)


def calculate_puntos_robustez(
    neighborhood_result: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Calcula los PUNTOS de robustez (0-400) según el test de vecindario.
    
    CASO A (Sin test): 0 puntos
        → Techo de cristal: Score máximo = 600 (solo calidad)
    
    CASO B (Con test): Puntos = 400 × e^(-DECAY_FACTOR * incertidumbre)
        - Incertidumbre 0% → 400 puntos (total: 1000)
        - Incertidumbre ~15% → ~255 puntos
        - Incertidumbre ~30% → ~162 puntos
        - Incertidumbre ~50% → ~89 puntos
    
    Args:
        neighborhood_result: Resultado del test de vecindario (dict o None)
    
    Returns:
        (puntos: float [0, 400], descripcion: str)
    """
    # CASO A: Sin test de vecindario → 0 puntos de robustez
    if not neighborhood_result:
        return 0.0, "sin_test"
    
    n_tested = neighborhood_result.get("n_neighbors_tested", 0)
    
    # Códigos especiales: -1 = skip por mal ROI, -2 = skip por baja frecuencia
    if n_tested < 0:
        return 0.0, f"skip_codigo_{n_tested}"
    
    if n_tested == 0:
        return 0.0, "no_vecinos"
    
    # CASO B: Test realizado - calcular incertidumbre
    dispersions = neighborhood_result.get("dispersions", {})
    
    if not dispersions:
        return 0.0, "sin_dispersiones"
    
    # Calcular incertidumbre agregada
    incertidumbre = neighborhood_result.get("incertidumbre")
    if incertidumbre is None:
        incertidumbre = calculate_incertidumbre(dispersions)
    
    # Decaimiento exponencial: puntos = 400 × e^(-3.0 * incertidumbre)
    factor = math.exp(-DECAY_FACTOR * incertidumbre)
    puntos = MAX_PUNTOS_ROBUSTEZ * factor
    
    # Clasificar resultado
    if incertidumbre <= 0.05:
        desc = "excelente"
    elif incertidumbre <= 0.15:
        desc = "bueno"
    elif incertidumbre <= 0.30:
        desc = "aceptable"
    elif incertidumbre <= 0.40:
        desc = "marginal"
    else:
        desc = "inestable"
    
    return float(max(0.0, min(MAX_PUNTOS_ROBUSTEZ, puntos))), desc


# Alias para compatibilidad
def calculate_factor_robustez(
    neighborhood_result: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, str]:
    """Legacy: retorna factor [0, 1] en lugar de puntos."""
    puntos, desc = calculate_puntos_robustez(neighborhood_result)
    factor = puntos / MAX_PUNTOS_ROBUSTEZ if MAX_PUNTOS_ROBUSTEZ > 0 else 0.0
    return factor, desc


# =============================================================================
# FUNCIÓN PRINCIPAL: SCORE UNIFICADO v7.0
# =============================================================================

def score_unified(
    metrics: Mapping[str, Any],
    neighborhood_result: Optional[Mapping[str, Any]] = None,
    trial_number: int = 0,
    equity_curve: Optional[List[float]] = None,
) -> float:
    """
    SCORE UNIFICADO v8.0
    
    Fórmula: Score = (Calidad_Raw × Factor_Actividad) + Puntos_Robustez
    
    RANGO: [0, 1000] = [0, 600] calidad + [0, 400] robustez
    
    COMPORTAMIENTO:
    - Sin test de robustez → máximo 600 (techo de cristal)
    - Con test y dispersión baja → hasta 1000 (600 + 400)
    - Con test y dispersión alta → 600 + pocos puntos de robustez
    
    Args:
        metrics: Métricas del trial original
        neighborhood_result: Resultado del test de robustez (vecindario)
        trial_number: Número del trial (para logging)
        equity_curve: Curva de equity (opcional)
    
    Returns:
        Score final [0, 1000]
    """
    
    # =========================================================================
    # EXTRACCIÓN DE MÉTRICAS BASE
    # =========================================================================
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_per_day", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "tpd", 0.0)
    
    # Suspenso total si trades/día muy bajo
    if trades_dia < REF_TRADES_DIA_MIN:
        return 0.001  # Score mínimo no-cero para Optuna
    
    sqn = _get(metrics, "sqn", 0.0)
    
    sharpe = _get(metrics, "sharpe", 0.0)
    if sharpe == 0:
        sharpe = _get(metrics, "sharpe_ratio", 0.0)
    
    drawdown = _get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _get(metrics, "max_drawdown", 50.0)
    
    roi = _get(metrics, "roi", 0.0)
    if roi == 0:
        roi = _get(metrics, "roi_pct", 0.0)
    
    # =========================================================================
    # COMPONENTE 1: PUNTOS_CALIDAD (0-600)
    # =========================================================================
    calidad_raw = calculate_calidad_raw(
        sharpe=sharpe,
        sqn=sqn,
        drawdown=drawdown,
        roi=roi,
    )
    
    # =========================================================================
    # COMPONENTE 2: FACTOR_ACTIVIDAD (0-1) - se aplica sobre calidad
    # =========================================================================
    factor_actividad = calculate_factor_actividad(trades_dia)
    puntos_calidad = calidad_raw * factor_actividad
    
    # =========================================================================
    # COMPONENTE 3: PUNTOS_ROBUSTEZ (0-400)
    # =========================================================================
    puntos_robustez, _ = calculate_puntos_robustez(neighborhood_result)
    
    # =========================================================================
    # SCORE FINAL = SUMA
    # =========================================================================
    score = puntos_calidad + puntos_robustez
    
    return float(max(0.001, min(MAX_SCORE, score)))


def format_score(score: float) -> str:
    """Formatea el score: X.XX si >=1, 0.XXX si <1."""
    if score >= 1.0:
        return f"{score:.2f}"
    else:
        return f"{score:.3f}"


# =============================================================================
# VERIFICACIÓN DE REQUISITOS PARA TEST DE ROBUSTEZ
# =============================================================================

def check_robustness_requirements(metrics: Mapping[str, Any], cfg: Optional[NeighborhoodConfig] = None) -> Tuple[bool, str]:
    """
    Verifica si un trial cumple los requisitos para hacer el test de robustez.
    
    Criterios v7.1:
    - Profit Factor > 1.1
    - Trades/día > 0.25  
    - Sharpe > 1.25
    
    Returns:
        (puede_hacer_test: bool, razon_suspenso: str)
    """
    if cfg is None:
        cfg = DEFAULT_NEIGHBORHOOD_CONFIG
    
    # Extraer métricas
    profit_factor = _get(metrics, "profit_factor", 0.0)
    if profit_factor == 0:
        profit_factor = _get(metrics, "pf", 0.0)
    
    trades_dia = _get(metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "trades_per_day", 0.0)
    if trades_dia == 0:
        trades_dia = _get(metrics, "tpd", 0.0)
    
    sharpe = _get(metrics, "sharpe", 0.0)
    
    # Verificar criterios
    if profit_factor < cfg.min_profit_factor:
        return False, f"PF ({profit_factor:.2f}) < {cfg.min_profit_factor}"
    
    if trades_dia < cfg.min_trades_per_day:
        return False, f"Trades/día ({trades_dia:.2f}) < {cfg.min_trades_per_day}"
    
    if sharpe < cfg.min_sharpe:
        return False, f"Sharpe ({sharpe:.2f}) < {cfg.min_sharpe}"
    
    return True, "OK"


# =============================================================================
# GENERACIÓN DE VECINOS
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
    Genera N vecinos usando ruido gaussiano alrededor de los parámetros base.
    """
    actual_seed = (seed or 42) + trial_number * 1000
    rng = np.random.default_rng(actual_seed)
    
    neighbors = []
    
    perturbable_params = {}
    for key, value in params.items():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        if isinstance(value, bool):
            continue
        
        if isinstance(value, (int, float)):
            perturbable_params[key] = value
    
    if not perturbable_params:
        return [dict(params) for _ in range(n_neighbors)]
    
    for _ in range(n_neighbors):
        neighbor = dict(params)
        
        for key, original_value in perturbable_params.items():
            if abs(original_value) < 1e-10:
                sigma = perturbation_std
            else:
                sigma = abs(original_value) * perturbation_std
            
            noise = rng.normal(0, sigma)
            new_value = original_value + noise
            
            if isinstance(original_value, int):
                new_value = int(round(new_value))
                if original_value > 0:
                    new_value = max(1, new_value)
            else:
                if original_value > 0:
                    new_value = max(1e-10, new_value)
            
            neighbor[key] = new_value
        
        neighbors.append(neighbor)
    
    return neighbors


# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS DE VECINDARIO
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
    Ejecuta el test de robustez completo.
    
    PROCESO:
    1. Verificar requisitos (ROI >= -15%, trades/día >= 0.15)
    2. Si no cumple → retornar sin test (Factor = 0.30)
    3. Generar N vecinos
    4. Ejecutar backtest para cada vecino
    5. Calcular dispersiones e incertidumbre
    6. Calcular score final con v7.0
    """
    t_start = time.perf_counter()
    
    cfg = neighborhood_config
    result = NeighborhoodResult()
    result.trial_number = trial_number
    result.original_score = original_score
    result.original_metrics = dict(original_metrics)
    result.max_dispersion_allowed = cfg.max_dispersion
    
    # =========================================================================
    # PASO 1: Extraer métricas del original
    # =========================================================================
    roi = _get(original_metrics, "roi", 0.0)
    trades_dia = _get(original_metrics, "trades_por_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(original_metrics, "trades_dia", 0.0)
    if trades_dia == 0:
        trades_dia = _get(original_metrics, "trades_per_day", 0.0)
    
    sharpe = _get(original_metrics, "sharpe", 0.0)
    sqn = _get(original_metrics, "sqn", 0.0)
    drawdown = _get(original_metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _get(original_metrics, "max_drawdown", 50.0)
    
    profit_factor = _get(original_metrics, "profit_factor", 0.0)
    if profit_factor == 0:
        profit_factor = _get(original_metrics, "pf", 0.0)
    
    result.original_sharpe = sharpe
    
    # =========================================================================
    # PASO 2: Verificar requisitos para hacer el test (v7.1: 3 criterios)
    # - Profit Factor > 1.1
    # - Trades/día > 0.25
    # - Sharpe > 1.25
    # =========================================================================
    if profit_factor < cfg.min_profit_factor:
        result.n_neighbors_tested = -1
        result.skip_reason = f"PF ({profit_factor:.2f}) < {cfg.min_profit_factor}"
        result.robustness_approved = False
        result.incertidumbre = 1.0
        result.aggregated_score = _calculate_score_without_robustness(original_metrics)
        result.execution_time_ms = (time.perf_counter() - t_start) * 1000
        return result
    
    if trades_dia < cfg.min_trades_per_day:
        result.n_neighbors_tested = -2
        result.skip_reason = f"Trades/día ({trades_dia:.2f}) < {cfg.min_trades_per_day}"
        result.robustness_approved = False
        result.incertidumbre = 1.0
        result.aggregated_score = _calculate_score_without_robustness(original_metrics)
        result.execution_time_ms = (time.perf_counter() - t_start) * 1000
        return result
    
    if sharpe < cfg.min_sharpe:
        result.n_neighbors_tested = -3
        result.skip_reason = f"Sharpe ({sharpe:.2f}) < {cfg.min_sharpe}"
        result.robustness_approved = False
        result.incertidumbre = 1.0
        result.aggregated_score = _calculate_score_without_robustness(original_metrics)
        result.execution_time_ms = (time.perf_counter() - t_start) * 1000
        return result
    
    # =========================================================================
    # PASO 3: Generar vecinos
    # =========================================================================
    neighbors = generate_gaussian_neighbors(
        params=params,
        n_neighbors=cfg.n_neighbors,
        perturbation_std=cfg.perturbation_std,
        exclude_prefixes=cfg.exclude_prefixes,
        seed=cfg.seed,
        trial_number=trial_number,
    )
    
    # =========================================================================
    # PASO 4: Ejecutar backtest para cada vecino
    # =========================================================================
    result.n_neighbors_tested = len(neighbors)
    
    all_rois = [roi]
    all_sqns = [sqn]
    all_sharpes = [sharpe]
    all_drawdowns = [drawdown]
    all_scores = [original_score]
    
    for i, neighbor_params in enumerate(neighbors):
        try:
            neighbor_signals = generate_signals_fn(df, strategy, neighbor_params)
            trades_df, neighbor_equity, neighbor_metrics = run_backtest_fn(
                df, neighbor_signals, config, neighbor_params, strategy
            )
            
            if trades_df.is_empty():
                continue
            
            n_roi = _get(neighbor_metrics, "roi", 0.0)
            n_sqn = _get(neighbor_metrics, "sqn", 0.0)
            n_sharpe = _get(neighbor_metrics, "sharpe", 0.0)
            n_dd = _get(neighbor_metrics, "drawdown", 50.0)
            if n_dd == 0:
                n_dd = _get(neighbor_metrics, "max_drawdown", 50.0)
            
            n_score = float(score_quality_only(neighbor_metrics))
            
            all_rois.append(n_roi)
            all_sqns.append(n_sqn)
            all_sharpes.append(n_sharpe)
            all_drawdowns.append(n_dd)
            all_scores.append(n_score)
            
            result.neighbor_scores.append(n_score)
            result.neighbor_sharpes.append(n_sharpe)
            result.neighbor_metrics.append(dict(neighbor_metrics))
            
            clean_params = {
                k: v for k, v in neighbor_params.items()
                if not str(k).startswith("__") and not str(k).startswith("exit_")
            }
            result.neighbor_params.append(clean_params)
            
            result.n_neighbors_successful += 1
            
        except Exception:
            continue
    
    # =========================================================================
    # PASO 5: Calcular dispersiones e incertidumbre
    # =========================================================================
    result.dispersions = {
        "roi": _calculate_cv(all_rois),
        "sqn": _calculate_cv(all_sqns),
        "sharpe": _calculate_cv(all_sharpes),
        "drawdown": _calculate_cv(all_drawdowns),
    }
    
    result.avg_dispersion = float(np.mean(list(result.dispersions.values())))
    result.incertidumbre = calculate_incertidumbre(result.dispersions)
    
    result.mean_score = float(np.mean(all_scores)) if all_scores else 0.0
    result.std_score = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
    result.mean_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0
    result.std_sharpe = float(np.std(all_sharpes)) if len(all_sharpes) > 1 else 0.0
    
    # =========================================================================
    # PASO 6: Determinar aprobación (legacy, pero útil para reporting)
    # =========================================================================
    all_pass = all(d <= cfg.max_dispersion for d in result.dispersions.values())
    result.robustness_approved = all_pass and result.n_neighbors_successful > 0
    
    # =========================================================================
    # PASO 7: Calcular score final con v7.0
    # =========================================================================
    result.aggregated_score = score_unified(
        metrics=original_metrics,
        neighborhood_result=result.to_dict(),
    )
    
    result.execution_time_ms = (time.perf_counter() - t_start) * 1000
    
    return result


def _calculate_score_without_robustness(metrics: Dict[str, Any]) -> float:
    """Calcula score solo con calidad (sin robustez) → techo de cristal."""
    return score_unified(metrics, neighborhood_result=None)


# =============================================================================
# FUNCIONES LEGACY PARA COMPATIBILIDAD
# =============================================================================

def score_quality_only(metrics: Mapping[str, Any]) -> float:
    """Score sin robustez (máximo ~300 debido al techo de cristal)."""
    return score_unified(metrics, neighborhood_result=None, trial_number=0)


def score_optuna(metrics: Mapping[str, Any]) -> float:
    """Alias para compatibilidad."""
    return score_unified(metrics, neighborhood_result=None, trial_number=0)


def nsga2_objectives(metrics: Mapping[str, Any]) -> Tuple[float, float]:
    """Objetivos para NSGA-II (quality, drawdown)."""
    quality = score_unified(metrics, neighborhood_result=None, trial_number=0)
    drawdown = _get(metrics, "drawdown", 50.0)
    if drawdown == 0:
        drawdown = _get(metrics, "max_drawdown", 50.0)
    return (max(0.001, quality), max(0.0, min(100.0, drawdown)))


def calculate_deflated_sharpe_score(
    sharpe: float,
    trial_number: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR Score [0, 100] para compatibilidad."""
    if n_trades < 5 or sharpe <= 0:
        return max(5.0, 15.0 + sharpe * 10.0) if sharpe > -1 else 5.0
    
    sr_var = max(0.5, 1.0 + 0.5 * sharpe**2)
    sr_std = math.sqrt(sr_var / max(n_trades, 10))
    trial_factor = math.sqrt(2 * math.log(max(2, trial_number + 1)))
    expected_sr = sr_std * trial_factor
    
    if sr_std > 0.001:
        z = (sharpe - expected_sr) / sr_std
        prob = 0.5 * (1.0 + math.erf(z / 1.414))
    else:
        prob = 0.9 if sharpe > 0.5 else 0.5
    
    return min(100.0, max(0.1, prob * 100.0))


def deflated_sharpe_ratio(sharpe_obs: float, n_trials: int, n_trades: int, **kw) -> float:
    """Legacy: Retorna probabilidad [0, 1]."""
    return calculate_deflated_sharpe_score(sharpe_obs, n_trials, n_trades) / 100.0


def score_robust(metrics: Mapping[str, Any], **kw) -> float:
    """Legacy."""
    return score_unified(metrics, neighborhood_result=None, trial_number=kw.get("n_trials", 1))


def check_trial_approved(
    metrics: Mapping[str, Any],
    neighborhood_result: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, float]:
    """
    Verifica si un trial está APROBADO en robustez.
    
    Returns:
        (aprobado: bool, stability_score: float 0-1)
    """
    if not neighborhood_result:
        return False, 0.0
    
    n_tested = neighborhood_result.get("n_neighbors_tested", 0)
    if n_tested <= 0:
        return False, 0.0
    
    aprobado = neighborhood_result.get("robustness_approved", False)
    incertidumbre = neighborhood_result.get("incertidumbre", 1.0)
    
    stability_score = max(0.0, 1.0 - incertidumbre)
    
    return aprobado, stability_score


# Legacy: calcular dispersión
def calculate_dispersion(values: List[float]) -> float:
    """Alias para _calculate_cv."""
    return _calculate_cv(values)


def check_robustness_approval(
    dispersions: Dict[str, float],
    max_dispersion: float = None,
) -> Tuple[bool, float]:
    """Legacy: verifica aprobación basada en dispersiones."""
    if max_dispersion is None:
        max_dispersion = VECINDARIO_MAX_DISPERSION
    
    if not dispersions:
        return False, 1.0
    
    all_approved = all(d <= max_dispersion for d in dispersions.values())
    avg_dispersion = float(np.mean(list(dispersions.values())))
    
    return all_approved, avg_dispersion


def calculate_robustness_points(
    aprobado: bool,
    avg_dispersion: float,
    max_dispersion: float = None,
) -> float:
    """Legacy: puntos de robustez (ya no se usa en v7.0)."""
    if not aprobado:
        return 0.0
    
    incertidumbre = avg_dispersion
    factor = math.exp(-DECAY_FACTOR * incertidumbre)
    return factor * MAX_SCORE * 0.7  # Aproximación legacy


def calculate_quality_points(
    sqn: float,
    sharpe: float,
    drawdown: float,
    roi: float,
) -> float:
    """Legacy: puntos de calidad (ahora usa calculate_calidad_raw)."""
    return calculate_calidad_raw(sharpe, sqn, drawdown, roi) * 0.3


# Legacy funciones del neighborhood_fitness
def shutdown_neighbor_pool():
    """Legacy: No hace nada en v7.0."""
    pass


def nsga2_objectives_robust(result: NeighborhoodResult) -> Tuple[float, float, float]:
    """Legacy: Retorna objetivos para NSGA-II."""
    return (
        result.robust_dsr,
        result.worst_case_cvar,
        result.equity_stability_r2,
    )

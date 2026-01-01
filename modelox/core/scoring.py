import math
import numpy as np
from scipy.stats import norm, skew, kurtosis
from typing import Any, Dict

def estimate_moments(winrate: float, payoff: float, loss: float) -> tuple[float, float]:
    n_sim = 1000
    w = max(0.0, min(100.0, winrate))
    n_wins = int(n_sim * (w / 100.0))
    n_losses = max(0, n_sim - n_wins)
    if n_wins + n_losses == 0: return 0.0, 3.0
    returns = np.array([payoff] * n_wins + [-abs(loss)] * n_losses)
    if len(returns) < 2 or np.std(returns) == 0: return 0.0, 3.0
    return skew(returns), kurtosis(returns)

def score_optuna(metricas: Dict[str, Any]) -> float:
    """
    MODELOX v14 - Penatización Ultra-Fuerte de Frecuencia.
    Optimizado para filtrar 'Lucky Trials' con pocos trades.
    """
    def get_val(key: str, default: float = 0.0) -> float:
        val = metricas.get(key, default)
        if val is None or not math.isfinite(float(val)): return default
        return float(val)

    # --- 1. DATOS DE FRECUENCIA ---
    n_trades = get_val("n_trades", 0)
    n_days = max(0.001, get_val("days_total", 1))
    trades_per_day = n_trades / n_days

    # --- 2. PENALIZACIÓN SIGMOIDAL AGRESIVA (La 'Barrera 0.4') ---
    # Centramos la caída en 0.4. 
    # Usamos una pendiente (k) de 15.0 para que sea un 'acantilado'.
    target_freq = 0.4 
    k_steepness = 15.0
    # Esta función devuelve 0.5 en 0.4 trades/día, y colapsa hacia 0.0001 por debajo.
    frequency_gate = 1 / (1 + math.exp(-k_steepness * (trades_per_day - target_freq)))

    # --- 3. RENDIMIENTO SEGURO (Capped Sharpe) ---
    sharpe = get_val("sharpe", -1.0)
    winrate = get_val("winrate", 0.0)
    profit_factor = get_val("profit_factor", 1.0)
    max_dd_pct = abs(get_val("drawdown", 100.0))
    estabilidad = max(0.01, get_val("estabilidad", 0.1))
    payoff = abs(get_val("riesgo_beneficio", 1.0))

    # --- 4. PROBABILIDAD ESTADÍSTICA (PSR) ---
    # Castiga fuertemente el bajo número de trades en la desviación estándar del Sharpe
    est_skew, est_kurt = estimate_moments(winrate, payoff, 1.0)
    # Si n_trades es 6, el divisor es muy pequeño, sr_std será gigante, prob_skill será baja.
    divisor_std = max(2.0, n_trades - 1)
    sr_std = np.sqrt(abs(1 - est_skew * sharpe + ((est_kurt - 1) / 4) * sharpe**2) / divisor_std)
    sr_std = max(0.001, sr_std)
    prob_skill = max(1e-9, norm.cdf(sharpe / sr_std))

    # --- 5. COMPOSICIÓN DEL SCORE ---
    # Performance core con e^Sharpe pero limitado para no distorsionar
    performance_core = math.exp(max(-3.0, min(sharpe, 5.0)))
    
    # Factor de cantidad de trades (logarítmico para premiar volumen de datos)
    # 6 trades = 1.94 | 50 trades = 3.93
    count_factor = math.log1p(n_trades)
    
    # Penalización por dolor (Drawdown)
    dd_pain = (1 + (max_dd_pct / 100.0)) ** 4 # Exponente 4 para mayor castigo al DD

    # FÓRMULA MAESTRA MODELOX V14
    # Multiplicamos por frequency_gate para que si freq < 0.4, todo el score tienda a 0.
    raw_score = (performance_core * prob_skill * count_factor * estabilidad * frequency_gate) / dd_pain

    # Booster para estrategias consistentes (solo si superan el umbral de frecuencia)
    if profit_factor > 1.1 and trades_per_day >= target_freq:
        raw_score *= math.log1p(profit_factor) * 2

    # Aseguramos que nunca sea 0 absoluto para que Optuna tenga gradiente
    return float(max(1e-6, raw_score))
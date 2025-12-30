from __future__ import annotations

"""
Estrategia 2 — Mean Reversion OU (5m)

Estrategia cuantitativa 100% determinista basada en un proceso de
Ornstein–Uhlenbeck (OU) discreto calibrado en ventana rodante sobre
log-precios.

Variable de estado:
    X_t = log(P_t) - media_rodante_{W_calib}(log(P))

Modelo continuo:
    dX_t = mu * (theta - X_t) dt + sigma dB_t

Discretización exacta (paso 1 barra = 5m):
    X_{t+1} = a * X_t + b + eps_t
    a = exp(-mu)
    b = theta * (1 - a)

A partir de (a, b, Var[eps_t]) estimados por OLS se reconstruyen
(mu, theta, sigma) y se definen los umbrales:
    d* = theta - k_entry * sigma      (entrada)
    b* = theta + k_exit * sigma       (take profit)
    L  = d*    - k_sl    * sigma      (stop-loss)

Reglas operativas (solo lado LONG):
    - FLAT & X_t <= d*  -> abrir LONG
    - LONG & X_t >= b*  -> cerrar por take profit
    - LONG & X_t <= L   -> cerrar por stop-loss

Además, si la intensidad de reversión estimada mu_t <= mu_min,
la estrategia se desactiva en esa ventana (no se abren nuevas
operaciones).
"""

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from numba import njit

from modelox.core.types import ExitDecision


class Strategy2OUMeanReversion:
    """Estrategia de reversión a la media basada en OU discreto (solo LONG)."""

    # Identificación
    combinacion_id = 2
    name = "OU_MeanReversion_Long"

    # Indicadores que se quieren ver en el plot
    # ou_x: estado OU (desviación logarítmica respecto a su media de equilibrio)
    __indicators_used = ["ou_x"]

    @classmethod
    def get_indicators_used(cls):
        return cls.__indicators_used

    # Espacio de parámetros para Optuna
    # Formato: nombre -> (min, max, step)
    parametros_optuna = {
        # Ventana de calibración OU (en barras de 5m)
        "W_calib": (30, 300, 5),
        # Coeficientes de umbral en sigma
        "k_entry": (1.0, 3.0, 0.1),
        "k_exit": (0.2, 1.5, 0.1),
        "k_sl": (1.5, 4.5, 0.1),
        # Intensidad mínima de reversión
        "mu_min": (0.01, 0.05, 0.005),
        # Máximo de barras en posición (protección adicional)
        "max_holding_bars": (20, 100, 5),
    }

    # Límite de seguridad absoluto
    TIMEOUT_BARS = 300

    # ------------------------------------------------------------------
    # Sugerencia de parámetros para Optuna
    # ------------------------------------------------------------------
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        W_calib = trial.suggest_int("W_calib", 30, 300, step=5)

        k_exit = trial.suggest_float("k_exit", 0.2, 1.5, step=0.1)
        k_entry = trial.suggest_float("k_entry", 1.0, 3.0, step=0.1)
        # Restricción dura: k_entry > k_exit
        if k_entry <= k_exit:
            k_entry = min(k_exit + 0.2, 3.0)

        k_sl = trial.suggest_float("k_sl", 1.5, 4.5, step=0.1)
        # Restricción dura: k_sl > k_entry
        if k_sl <= k_entry:
            k_sl = min(k_entry + 0.5, 4.5)

        mu_min = trial.suggest_float("mu_min", 0.01, 0.05, step=0.005)
        max_holding_bars = trial.suggest_int("max_holding_bars", 20, 100, step=5)

        return {
            "W_calib": W_calib,
            "k_entry": float(k_entry),
            "k_exit": float(k_exit),
            "k_sl": float(k_sl),
            "mu_min": float(mu_min),
            "max_holding_bars": int(max_holding_bars),
        }

    # ------------------------------------------------------------------
    # Generación de señales (100% determinista)
    # ------------------------------------------------------------------
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        # Para reporting/plot modular
        params["__indicators_used"] = self.get_indicators_used()

        W_calib = int(params["W_calib"])
        k_entry = float(params["k_entry"])
        k_exit = float(params["k_exit"])
        k_sl = float(params["k_sl"])
        mu_min = float(params["mu_min"])

        # Warm-up robusto: necesitamos al menos W_calib barras para estimar OU
        params["__warmup_bars"] = max(W_calib + 5, 50)

        close = df["close"].to_numpy().astype(np.float64)

        (
            ou_x,
            ou_theta,
            ou_sigma,
            ou_mu,
            ou_d_entry,
            ou_b_exit,
            ou_L_stop,
            ou_active,
        ) = _compute_ou_state_and_thresholds(
            close=close,
            window=W_calib,
            k_entry=k_entry,
            k_exit=k_exit,
            k_sl=k_sl,
            mu_min=mu_min,
        )

        n = len(close)
        assert len(ou_x) == n

        # Añadimos columnas OU al DataFrame (para debug/plot si se desea)
        df = df.with_columns(
            [
                pl.Series("ou_x", ou_x),
                pl.Series("ou_mu", ou_mu),
                pl.Series("ou_sigma", ou_sigma),
                pl.Series("ou_theta", ou_theta),
                pl.Series("ou_d_entry", ou_d_entry),
                pl.Series("ou_b_exit", ou_b_exit),
                pl.Series("ou_L_stop", ou_L_stop),
                pl.Series("ou_active", ou_active),
            ]
        )

        # Señal LONG determinista: sólo se activa cuando el modelo es válido
        signal_long = (
            (pl.col("ou_active"))
            & pl.col("ou_x").is_not_null()
            & pl.col("ou_d_entry").is_not_null()
            & (pl.col("ou_x") <= pl.col("ou_d_entry"))
        ).fill_null(False)

        # Esta estrategia no abre cortos (estrictamente LONG)
        signal_short = pl.lit(False)

        return df.with_columns([
            signal_long.alias("signal_long"),
            signal_short.alias("signal_short"),
        ])

    # ------------------------------------------------------------------
    # Gestión de salidas
    # ------------------------------------------------------------------
    def decide_exit(
        self,
        df: pl.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        side: str,
        *,
        saldo_apertura: float,
    ) -> Optional[ExitDecision]:
        """Aplica reglas de salida puramente basadas en OU.

        - Take profit cuando X_t >= b*
        - Stop-loss cuando X_t <= L
        - Time-out si supera max_holding_bars
        """

        n = len(df)
        if entry_idx >= n - 1:
            return None

        # Sólo implementamos lado LONG según la especificación
        is_long = side.upper() == "LONG"
        if not is_long:
            # Cerrar inmediatamente cualquier posición que no sea LONG
            return ExitDecision(exit_idx=min(entry_idx + 1, n - 1), reason="NON_LONG_FORCED_EXIT")

        ou_x = df["ou_x"].to_numpy().astype(np.float64)
        ou_d = df["ou_d_entry"].to_numpy().astype(np.float64)
        ou_b = df["ou_b_exit"].to_numpy().astype(np.float64)
        ou_L = df["ou_L_stop"].to_numpy().astype(np.float64)
        ou_active = df["ou_active"].to_numpy().astype(np.bool_)

        max_holding_bars = int(params.get("max_holding_bars", 50))
        timeout_limit = min(self.TIMEOUT_BARS, max_holding_bars)

        exit_idx, reason_code = _decide_exit_ou_numba(
            entry_idx,
            ou_x,
            ou_d,
            ou_b,
            ou_L,
            ou_active,
            timeout_limit,
        )

        reason_map = {
            0: None,
            1: "TAKE_PROFIT_OU",   # X_t >= b*
            2: "STOP_LOSS_OU",     # X_t <= L
            3: "TIME_EXIT_OU",     # timeout
            4: "MODEL_INVALID_OU", # mu_t <= mu_min durante la operación
        }

        if exit_idx >= 0 and reason_code in reason_map and reason_map[reason_code] is not None:
            return ExitDecision(exit_idx=exit_idx, reason=reason_map[reason_code])

        # Fallback de seguridad
        fallback_idx = min(entry_idx + timeout_limit, n - 1)
        return ExitDecision(exit_idx=fallback_idx, reason="TIME_EXIT_OU")


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Media móvil simple (trailing) implementada en numpy puro."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n == 0 or n < window:
        return out

    cumsum = np.cumsum(arr)
    for i in range(window - 1, n):
        start = i - window + 1
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = total / window
    return out


def _compute_ou_state_and_thresholds(
    *,
    close: np.ndarray,
    window: int,
    k_entry: float,
    k_exit: float,
    k_sl: float,
    mu_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibra un OU discreto X_{t+1} = a X_t + b + eps_t en ventana rodante.

    Devuelve arrays alineados con las barras originales:
        - ou_x:      estado X_t
        - ou_theta:  media de equilibrio theta_t
        - ou_sigma:  sigma_t del proceso continuo
        - ou_mu:     intensidad de reversión mu_t
        - ou_d:      umbral de entrada d*_t
        - ou_b:      umbral de salida b*_t
        - ou_L:      stop-loss L_t
        - ou_active: bool de modelo válido (mu_t > mu_min, 0 < a < 1)
    """

    n = len(close)
    ou_x = np.full(n, np.nan, dtype=np.float64)
    ou_theta = np.full(n, np.nan, dtype=np.float64)
    ou_sigma = np.full(n, np.nan, dtype=np.float64)
    ou_mu = np.zeros(n, dtype=np.float64)
    ou_d = np.full(n, np.nan, dtype=np.float64)
    ou_b = np.full(n, np.nan, dtype=np.float64)
    ou_L = np.full(n, np.nan, dtype=np.float64)
    ou_active = np.zeros(n, dtype=np.bool_)

    if n == 0 or window <= 2:
        return ou_x, ou_theta, ou_sigma, ou_mu, ou_d, ou_b, ou_L, ou_active

    # 1) Definir X_t = log(P_t) - media_rodante_{W_calib}(log(P))
    log_p = np.log(close)
    log_ma = _rolling_mean(log_p, window)
    ou_x = log_p - log_ma

    # 2) Calibrar OU en cada ventana [t-window+1, t]
    min_pairs = max(10, window // 3)

    for t in range(window, n):
        start = t - window + 1
        end = t
        x_win = ou_x[start : end + 1]

        # Necesitamos pares (X_{i-1}, X_i) sin NaN
        x_prev = x_win[:-1]
        x_curr = x_win[1:]
        mask = np.isfinite(x_prev) & np.isfinite(x_curr)
        if mask.sum() < min_pairs:
            continue

        xp = x_prev[mask]
        xc = x_curr[mask]
        m_x = xp.mean()
        m_y = xc.mean()
        s_xx = np.sum((xp - m_x) ** 2)
        if s_xx <= 0.0:
            continue
        s_xy = np.sum((xp - m_x) * (xc - m_y))
        a = s_xy / s_xx
        # Estacionariedad OU: 0 < a < 1
        if not (0.0 < a < 0.999):
            continue

        b = m_y - a * m_x

        # Residuos eps_t = X_t - (a X_{t-1} + b)
        eps = xc - (a * xp + b)
        dof = max(1, eps.size - 2)
        var_eps = np.sum(eps ** 2) / dof
        if not np.isfinite(var_eps) or var_eps <= 0.0:
            continue

        # Reconstruir parámetros continuos (Delta t = 1 barra)
        mu = -np.log(a)
        if not np.isfinite(mu) or mu <= 0.0:
            continue
        # Condición de validez del modelo: mu > mu_min
        if mu <= mu_min:
            # mu ≈ 0  => paseo aleatorio, sin reversión clara
            continue

        theta = b / (1.0 - a)
        # sigma^2 = var_eps * 2 mu / (1 - a^2)
        denom = 1.0 - a * a
        if denom <= 0.0:
            continue
        sigma_sq = var_eps * 2.0 * mu / denom
        if sigma_sq <= 0.0 or not np.isfinite(sigma_sq):
            continue
        sigma = float(np.sqrt(sigma_sq))

        ou_mu[t] = mu
        ou_theta[t] = theta
        ou_sigma[t] = sigma

        # Umbrales en espacio X_t
        d_star = theta - k_entry * sigma
        b_star = theta + k_exit * sigma
        L = d_star - k_sl * sigma

        ou_d[t] = d_star
        ou_b[t] = b_star
        ou_L[t] = L
        ou_active[t] = True

    return ou_x, ou_theta, ou_sigma, ou_mu, ou_d, ou_b, ou_L, ou_active


@njit(cache=True, fastmath=True)
def _decide_exit_ou_numba(
    entry_idx: int,
    ou_x: np.ndarray,
    ou_d: np.ndarray,
    ou_b: np.ndarray,
    ou_L: np.ndarray,
    ou_active: np.ndarray,
    max_holding_bars: int,
) -> tuple:
    """Recorre barras posteriores a la entrada y aplica reglas OU.

    Prioridad de salida:
        1) STOP LOSS (X_t <= L)
        2) TAKE PROFIT (X_t >= b*)
        3) MODEL_INVALID (mu_t <= mu_min durante la operación)
        4) TIMEOUT (supera max_holding_bars)
    """

    n = len(ou_x)
    last_allowed = min(n - 1, entry_idx + max_holding_bars)

    for i in range(entry_idx + 1, last_allowed + 1):
        x = ou_x[i]
        d = ou_d[i]
        b = ou_b[i]
        L = ou_L[i]

        if np.isnan(x) or np.isnan(d) or np.isnan(b) or np.isnan(L):
            continue

        # 1) STOP LOSS coherente en espacio X_t
        if x <= L:
            return i, 2

        # 2) TAKE PROFIT cuando X_t revierte hacia theta y cruza b*
        if x >= b:
            return i, 1

        # 3) Si el modelo deja de ser válido durante la vida del trade
        if not ou_active[i]:
            return i, 4

    # 4) TIMEOUT si no se ha salido antes
    if last_allowed > entry_idx:
        return last_allowed, 3

    return -1, 0

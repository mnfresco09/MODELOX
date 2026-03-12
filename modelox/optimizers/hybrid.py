"""modelox/optimizers/hybrid.py

═══════════════════════════════════════════════════════════════════════════════
    OPTIMIZADOR HÍBRIDO (QMC → TPE)
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
Este optimizador divide los trials al 50%. La primera mitad utiliza QMCSampler
para asegurar una cobertura equidistante del espacio de búsqueda. La segunda
mitad utiliza TPESampler que aprende de los trials anteriores de QMC para
concentrarse en las zonas más prometedoras, aprovechando lo que ya encontró.

VENTAJAS:
=========
  ✓ EXPLORACIÓN PERFECTA inicial garantizada por QMC (Sobol)
  ✓ REFINAMIENTO RÁPIDO gracias al árbol bayesiano de TPE
  ✓ SIN OVERFITTING en la fase inicial, ya que QMC es determinista y no se sesga
  ✓ LA MEJOR COMBINACIÓN para espacios grandes sin caer en mínimos locales.

FILOSOFÍA DEL SCORING:
======================
  Score ∈ [0, 100] = media de N_PARTS slices temporales del periodo.

  Cada slice:
    raw_score  = E_score(E) + DD_score(DD)
    slice_score = clamp(raw_score, 0) / E_MAX × T_score(tpd) × 100

  Componentes:
    E = WR × (AvgWin / AvgLoss_abs) − LR       (R-Multiple; equilibrio en 0)

    E_score = E_MAX × tanh(E)           si E ≥ 0   → progresivo hasta E_MAX
            = E_MAX × tanh(E × 2.0)     si E < 0   → penalización más dura

    DD = Max Drawdown % desde pico del equity
    DD_score = 0                         si DD ≤ DD_THRESHOLD   (neutro)
             = −MAX_DD_PENALTY × ((DD − DD_THRESHOLD) / (100 − DD_THRESHOLD))^1.5

    T_score  = 1.0                       si tpd ≥ TPD_THRESHOLD  (multiplicativo)
             = (tpd / TPD_THRESHOLD)^3   si tpd < TPD_THRESHOLD

  Anti-overfitting: un slice sin trades puntúa 0.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler, QMCSampler
from .storage import resolve_storage_for_strategy

# =============================================================================
# IMPORTS INTERNOS
# =============================================================================
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
# HELPER: SERIALIZACIÓN COMPACTA DE TRADES (para dashboard analítico)
# =============================================================================

def _serialize_trades_compact(trades_df: "pl.DataFrame", max_trades: int = 5000) -> list:
    """
    Serializa trades a una lista compacta de dicts para almacenamiento en Optuna DB.
    Usado por el dashboard analítico institucional.

    Formato por trade:
        t   → type ("long"/"short")
        ep  → entry_price (float, 2 decimales)
        xp  → exit_price  (float, 2 decimales)
        pnl → pnl_neto    (float, 4 decimales)
        pct → pnl_pct     (float, 4 decimales)
        r   → reason int  (0=EndData 1=SL 2=TP 3=Trail 4=Time 5=Signal)
        ent → entry_time  str (ISO, 19 chars)
        ext → exit_time   str (ISO, 19 chars)
    """
    try:
        n = len(trades_df)
        if n == 0:
            return []
        if n > max_trades:
            trades_df = trades_df.head(max_trades)

        cols = set(trades_df.columns)
        select_map = {
            "type":        "t",
            "entry_price": "ep",
            "exit_price":  "xp",
            "pnl_neto":    "pnl",
            "pnl_pct":     "pct",
            "reason":      "r",
            "entry_time":  "ent",
            "exit_time":   "ext",
        }
        available = {src: dst for src, dst in select_map.items() if src in cols}
        if not available:
            return []

        df_small = trades_df.select(list(available.keys())).rename(available)

        # Redondear floats
        for c, decimals in [("ep", 2), ("xp", 2), ("pnl", 4), ("pct", 4)]:
            if c in df_small.columns:
                df_small = df_small.with_columns(pl.col(c).round(decimals))

        # Timestamps → string truncado
        for c in ("ent", "ext"):
            if c in df_small.columns:
                try:
                    df_small = df_small.with_columns(
                        pl.col(c).cast(pl.String).str.slice(0, 19).alias(c)
                    )
                except Exception:
                    pass

        return df_small.to_dicts()
    except Exception:
        return []


# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING HYBRID
# =============================================================================

@dataclass
class HybridScoringConfig:
    """
    Configuración del scoring híbrido.

    Score ∈ [0, 100] = media de N_PARTS slices temporales.

    Cada slice:
        raw        = E_score(E) + DD_score(DD)
        slice_score = clamp(raw, 0) / E_MAX × T_score(tpd) × 100

    E_score: E_MAX × tanh(E)       si E ≥ 0
             E_MAX × tanh(E × 2.0) si E < 0   (más severo)
    DD_score: 0                     si DD ≤ DD_THRESHOLD
              −MAX_DD_PENALTY × ((DD − DD_THRESHOLD) / (100 − DD_THRESHOLD))^1.5
    T_score:  1.0                   si tpd ≥ TPD_THRESHOLD
              (tpd / TPD_THRESHOLD)^3
    """
    E_MAX: float          = 2.0   # Techo de E_score (y divisor de normalización)
    DD_THRESHOLD: float   = 27.5  # Umbral de drawdown en % (por encima → penaliza)
    MAX_DD_PENALTY: float = 2.0   # Penalización máxima de DD (en unidades de E_score)
    TPD_THRESHOLD: float  = 0.10  # Trades/día mínimos sin penalización
    N_PARTS: int          = 4     # Slices temporales para anti-overfitting
    # BASE_CAPITAL: se usa saldo_inicial del BacktestConfig, no hay valor fijo aquí


# Instancia por defecto
HYBRID_SCORING_CONFIG = HybridScoringConfig()


# =============================================================================
# [SECCIÓN 2] CLASE SCORER HYBRID
# =============================================================================

class HybridScorer:
    """
    Scorer para Optimizador Híbrido QMC/TPE.

    Score ∈ [0, 100] = media de N_PARTS slices temporales.

    Cada slice puntúa por [Expectancy, Drawdown, Frecuencia].
    Un slice sin trades puntúa 0 (penaliza inconsistencia temporal).
    """

    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[HybridScoringConfig] = None,
    ):
        self.study = study
        self.config = config or HYBRID_SCORING_CONFIG

    # =========================================================================
    # HELPERS MATEMÁTICOS (ESTÁTICOS)
    # =========================================================================

    @staticmethod
    def _compute_expectancy(pnl: np.ndarray) -> float:
        """
        E = WR × ganancia_media − (1−WR) × pérdida_media_abs

        MISMA FÓRMULA que metrics.py:expectativa().
        Resultado en unidades monetarias (dollars/pesos).

        Para el scoring con tanh se normaliza por avg_loss_abs internamente,
        lo que es equivalente a:  E_norm = WR × (AvgWin/AvgLoss) − LR
        Esto mantiene la calibración del tanh sin cambiar el score behavior.
        """
        n = len(pnl)
        if n == 0:
            return -1.0

        winners = pnl[pnl > 0]
        losers  = pnl[pnl < 0]

        wr           = len(winners) / n
        avg_win      = float(np.mean(winners))         if len(winners) > 0 else 0.0
        avg_loss_abs = float(np.mean(np.abs(losers)))  if len(losers)  > 0 else 1e-10

        # Fórmula canónica (dollar) — normalizada por avg_loss_abs para el tanh
        # E_dollar = wr * avg_win - (1-wr) * avg_loss_abs
        # E_norm   = E_dollar / avg_loss_abs = wr * (avg_win/avg_loss_abs) - (1-wr)
        rr = avg_win / max(avg_loss_abs, 1e-10)
        return float(wr * rr - (1.0 - wr))

    @staticmethod
    def _compute_max_dd_pct(pnl: np.ndarray, base_capital: float) -> float:
        """
        Max Drawdown % desde el pico del equity.
        equity = base_capital + cumsum(pnl)
        """
        if len(pnl) == 0:
            return 0.0
        equity = base_capital + np.cumsum(pnl)
        peak   = np.maximum.accumulate(equity)
        dd_pct = np.where(peak > 0, (peak - equity) / peak * 100.0, 0.0)
        return float(np.max(dd_pct))

    # =========================================================================
    # COMPONENTES DEL SCORE
    # =========================================================================

    def _e_score(self, E: float) -> float:
        """
        E ≥ 0 → E_MAX × tanh(E)          progresivo hasta E_MAX
        E < 0 → E_MAX × tanh(E × 2.0)    penalización más dura
        """
        cfg = self.config
        if E >= 0.0:
            return cfg.E_MAX * math.tanh(E)
        else:
            return cfg.E_MAX * math.tanh(E * 2.0)

    def _dd_score(self, dd_pct: float) -> float:
        """
        DD ≤ DD_THRESHOLD → 0.0           (neutro, sin bonus ni penalización)
        DD > DD_THRESHOLD → penalización progresiva creciente [−MAX_DD_PENALTY, 0)
        """
        cfg = self.config
        if dd_pct <= cfg.DD_THRESHOLD:
            return 0.0
        excess_norm = (dd_pct - cfg.DD_THRESHOLD) / max(100.0 - cfg.DD_THRESHOLD, 1e-10)
        excess_norm = min(1.0, excess_norm)
        return -cfg.MAX_DD_PENALTY * (excess_norm ** 1.5)

    def _t_score(self, tpd: float) -> float:
        """
        tpd ≥ TPD_THRESHOLD → 1.0              (sin efecto)
        tpd < TPD_THRESHOLD → (tpd / threshold)^3   (colapsa hacia 0)
        Multiplicativo: destruye el score si la estrategia no opera suficiente.
        """
        cfg = self.config
        if tpd >= cfg.TPD_THRESHOLD:
            return 1.0
        return (tpd / cfg.TPD_THRESHOLD) ** 3.0

    # =========================================================================
    # SCORE POR SLICE
    # =========================================================================

    def _score_slice(self, pnl_slice: np.ndarray, days_slice: float, saldo_inicial: float) -> float:
        """
        Score [0, 100] para un slice temporal.
        Slice vacío → 0.0  (penaliza periodos sin actividad).
        """
        return self._score_slice_detailed(pnl_slice, days_slice, saldo_inicial)["score"]

    def _score_slice_detailed(self, pnl_slice: np.ndarray, days_slice: float, saldo_inicial: float) -> dict:
        """
        Score [0, 100] + desglose de componentes para un slice temporal.
        Devuelve dict con: score, e_score, dd_score, t_score, E, dd_pct, tpd, n
        """
        cfg = self.config
        n = len(pnl_slice)
        if n == 0 or days_slice <= 0.0:
            return {"score": 0.0, "e_score": 0.0, "dd_score": 0.0, "t_score": 0.0,
                    "E": -1.0, "dd_pct": 0.0, "tpd": 0.0, "n": 0}

        E      = self._compute_expectancy(pnl_slice)
        dd_pct = self._compute_max_dd_pct(pnl_slice, saldo_inicial)
        tpd    = n / days_slice

        e_s  = self._e_score(E)
        dd_s = self._dd_score(dd_pct)
        t_s  = self._t_score(tpd)

        raw     = e_s + dd_s
        clamped = max(0.0, raw)
        norm    = clamped / cfg.E_MAX
        score   = norm * t_s * 100.0

        return {
            "score":    round(score, 4),
            "e_score":  round(e_s,   4),
            "dd_score": round(dd_s,  4),
            "t_score":  round(t_s,   4),
            "E":        round(E,     4),
            "dd_pct":   round(dd_pct, 3),
            "tpd":      round(tpd,   4),
            "n":        int(n),
        }

    # =========================================================================
    # FUNCIÓN PÚBLICA PRINCIPAL
    # =========================================================================

    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        equity_curve: Optional[np.ndarray] = None,
        trades_pnl: Optional[np.ndarray] = None,
        total_days: Optional[float] = None,
        saldo_inicial: Optional[float] = None,
    ) -> float:
        """
        Score ∈ [0, 100] = media de N_PARTS slices del periodo.

        Divide trades_pnl en N_PARTS chunks iguales por índice.
        Cada chunk cubre total_days / N_PARTS días.
        Un chunk sin trades puntúa 0.

        saldo_inicial: capital de partida del backtest (BacktestConfig.saldo_inicial).
                       Se usa como referencia para calcular el Drawdown %.
                       Si no se pasa, se infiere de equity_curve[0].
        """
        cfg = self.config

        # ── 1. Extracción de datos ─────────────────────────────────────────
        if trades_pnl is None:
            raw = metrics.get("_trades_pnl_array", None)
            if raw is not None:
                trades_pnl = np.asarray(raw, dtype=np.float64)

        if trades_pnl is None or trades_pnl.size == 0:
            return 0.0

        if total_days is None or total_days <= 0.0:
            td = float(metrics.get("_total_days", 0.0) or 0.0)
            if td <= 0.0:
                tpd = float(metrics.get("trades_por_dia", 0.0) or 0.0)
                n_t = int(metrics.get("n_trades", 0) or metrics.get("total_trades", 0))
                td = float(n_t) / tpd if tpd > 0 else 1.0
            total_days = td

        # Resolver capital de referencia para DD
        if saldo_inicial is None or saldo_inicial <= 0.0:
            if equity_curve is not None and len(equity_curve) > 0:
                saldo_inicial = float(equity_curve[0])
            else:
                saldo_inicial = 1_000.0  # fallback mínimo

        trades_pnl = np.asarray(trades_pnl, dtype=np.float64)
        n_trades   = len(trades_pnl)
        if n_trades == 0 or total_days <= 0.0:
            return 0.0

        # ── 2. Anti-overfitting: N_PARTS slices temporales ────────────────
        n_parts        = max(1, cfg.N_PARTS)
        slices         = np.array_split(trades_pnl, n_parts)
        days_slice     = total_days / n_parts
        slice_details  = [self._score_slice_detailed(s, days_slice, saldo_inicial) for s in slices]
        slice_scores   = [d["score"] for d in slice_details]

        final_score = float(np.mean(slice_scores))
        if not math.isfinite(final_score):
            final_score = 0.0

        # ── 3. Auditoría en Optuna ─────────────────────────────────────────
        if trial is not None:
            try:
                E_global   = self._compute_expectancy(trades_pnl)
                dd_global  = self._compute_max_dd_pct(trades_pnl, saldo_inicial)
                tpd_global = n_trades / total_days
                trial.set_user_attr("final_score",       float(final_score))
                trial.set_user_attr("expectancy",        float(E_global))
                trial.set_user_attr("max_dd_pct",        float(dd_global))
                trial.set_user_attr("trades_per_day",    float(tpd_global))
                trial.set_user_attr("slice_scores",      [round(s, 4) for s in slice_scores])
                trial.set_user_attr("slice_components",  slice_details)
                trial.set_user_attr("n_trades",          int(n_trades))
            except Exception:
                pass

        return final_score


# =============================================================================
# [SECCIÓN 3] CLASE OPTIMIZADOR HYBRID
# =============================================================================

@dataclass
class HybridOptimizerConfig:
    """Configuración del optimizador Híbrido."""
    SEED: Optional[int] = None           
    N_JOBS: int = 1                       
    CREATE_DATABASE: bool = True          # True = crear SQLite por estrategia (IDx.db)
    STUDY_NAME_PREFIX: str = "MODELOX"    
    SCRAMBLE: bool = True

HYBRID_OPTIMIZER_CONFIG = HybridOptimizerConfig()

class HybridOptimizer:
    """
    Optimizador Híbrido
    QMC primero (50% de trials) -> Puntos equidistantes puros
    TPE segundo (50% de trials) -> Aprende del score de QMC en la misma base de datos Optuna
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[HybridOptimizerConfig] = None,
        scoring_config: Optional[HybridScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or HYBRID_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or HYBRID_SCORING_CONFIG
        self.activo = activo
        
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[HybridScorer] = None
    
    @staticmethod
    def _slug(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        params_rt["__exit_time_bars"] = exit_settings.time_stop_bars

        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        params_rt["exit_time_bars"] = exit_settings.time_stop_bars

        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        # time_bars: usa el mismo TF de entrada para contar barras (no 1m)
        exit_tf = entry_tf if exit_settings.exit_type == "BARS" else "1m"
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            signals_df = SignalGenerator.generate_signals(df_entry, strategy, params_rt, df_map)
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_entry, signals_df, self.config, params_rt, strategy,
            )

            if trades_df.is_empty():
                return 0.0

            trial.set_user_attr("metricas", metrics)

            # ── Dashboard analytics: trades compactos + equity curve ──────
            try:
                trial.set_user_attr("trades_data", _serialize_trades_compact(trades_df))
            except Exception:
                pass
            try:
                if equity_curve and len(equity_curve) <= 10_000:
                    trial.set_user_attr("equity_curve", [round(v, 4) for v in equity_curve])
            except Exception:
                pass

            # Extraer pnl_neto array
            if isinstance(trades_df, pl.DataFrame):
                _pnl_arr = trades_df["pnl_neto"].to_numpy().astype(np.float64)
            else:
                _pnl_arr = trades_df["pnl_neto"].to_numpy(dtype=np.float64)

            # total_days desde timestamps del DataFrame de entrada
            _total_days = 0.0
            if "timestamp" in df_entry.columns and len(df_entry) > 0:
                try:
                    _ts0 = df_entry["timestamp"][0]
                    _ts1 = df_entry["timestamp"][-1]
                    _delta = pl.DataFrame({"s": [_ts0], "e": [_ts1]}).select(
                        ((pl.col("e") - pl.col("s")).dt.total_seconds() / 86400.0).alias("d")
                    )
                    _total_days = max(1.0, float(_delta["d"][0]))
                except Exception:
                    _total_days = 1.0

            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
                trades_pnl=_pnl_arr,
                total_days=_total_days,
                saldo_inicial=float(self.config.saldo_inicial),
            )
            
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
    
    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.Study:
        
        base_tf = base_timeframe or "1m"
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        cfg = self.optimizer_config
        
        storage = resolve_storage_for_strategy(
            create_database=bool(cfg.CREATE_DATABASE),
            strategy_id=getattr(strategy, "combinacion_id", None),
        )
            
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy.name), "HYBRID"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        self._scorer = HybridScorer(config=self.scoring_config)
        objective = self._create_objective(df_base, df_map, strategy, base_tf)
        
        # Fase 1: QMC (Exactamente 50%)
        n_qmc = int(math.ceil(self.n_trials / 2))
        n_tpe = self.n_trials - n_qmc
        
        if n_qmc > 0:
            sampler_qmc = QMCSampler(
                seed=cfg.SEED, 
                scramble=getattr(cfg, "SCRAMBLE", True),
                warn_independent_sampling=False
            )
            study_qmc = optuna.create_study(
                direction="maximize",
                sampler=sampler_qmc,
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            self._scorer.study = study_qmc
            
            study_qmc.optimize(
                objective,
                n_trials=n_qmc,
                n_jobs=int(cfg.N_JOBS),
                gc_after_trial=True,
                catch=(Exception,),
            )
            
            self._last_study = study_qmc
            
        # Fase 2: TPE (Aprovecha los trials anteriores guardados en storage)
        if n_tpe > 0:
            sampler_tpe = TPESampler(seed=cfg.SEED, n_startup_trials=0, multivariate=True)
            study_tpe = optuna.create_study(
                direction="maximize",
                sampler=sampler_tpe,
                study_name=study_name,
                storage=storage,
                load_if_exists=True, # Aquí engancha los scores previos de QMC
            )
            self._scorer.study = study_tpe
            
            study_tpe.optimize(
                objective,
                n_trials=n_tpe,
                n_jobs=int(cfg.N_JOBS),
                gc_after_trial=True,
                catch=(Exception,),
            )
            
            self._last_study = study_tpe
            return study_tpe
            
        return self._last_study


def create_hybrid_study(
    strategy_name: str,
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    create_database: bool = True,
    storage: Optional[str] = None,
    phase: str = "QMC" # Opcional si se quiere forzar un sampler
) -> optuna.Study:
    """Función de factoría básica por si otros la importan."""
    parts = [study_name_prefix, str(strategy_name), "hybrid"]
    if activo:
        parts.append(str(activo))
    study_name = HybridOptimizer._slug("_".join(parts))
    
    if phase == "QMC":
        sampler = QMCSampler(seed=seed, scramble=True, warn_independent_sampling=False)
    else:
        sampler = TPESampler(seed=seed, n_startup_trials=0, multivariate=True)
    
    if storage is None:
        storage = resolve_storage_for_strategy(
            create_database=create_database,
            strategy_id=strategy_id,
        )
        
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    return study

def score_hybrid(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
    trades_pnl: Optional[np.ndarray] = None,
    total_days: Optional[float] = None,
    saldo_inicial: Optional[float] = None,
) -> float:
    scorer = HybridScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
        trades_pnl=trades_pnl,
        total_days=total_days,
        saldo_inicial=saldo_inicial,
    )

__all__ = [
    "HybridOptimizer",
    "HybridOptimizerConfig",
    "HybridScorer",
    "HybridScoringConfig",
    "HYBRID_SCORING_CONFIG",
    "HYBRID_OPTIMIZER_CONFIG",
    "create_hybrid_study",
    "score_hybrid",
]
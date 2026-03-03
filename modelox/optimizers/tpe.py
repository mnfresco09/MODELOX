"""modelox.optimizers.tpe — Optimizador TPE (Tree-structured Parzen Estimator)."""

from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from .storage import resolve_storage_for_strategy

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



@dataclass
class TPEScoringConfig:
    SCORE_MIN: float = 1.0
    SCORE_MAX: float = 1000.0
    WEIGHT_SHARPE: float = 0.35
    WEIGHT_SQN: float = 0.20
    WEIGHT_ROI: float = 0.20
    WEIGHT_DRAWDOWN: float = 0.15
    WEIGHT_TRADES: float = 0.10
    SHARPE_CENTER: float = 1.0
    SHARPE_SCALE: float = 1.5
    SQN_TARGET: float = 4.0
    ROI_TARGET: float = 100.0
    DRAWDOWN_MAX_ACCEPTABLE: float = 30.0
    MIN_TRADES_TARGET: int = 50
    MIN_TRADES_FOR_VALID: int = 10
    MIN_TRADES_PER_DAY: float = 0.05
    MAX_DRAWDOWN_LIMIT: float = 80.0
    PSR_ENABLED: bool = True
    PSR_BENCHMARK: float = 0.0
    PSR_MIN_TRADES: int = 30
    PSR_WEIGHT: float = 0.15


TPE_SCORING_CONFIG = TPEScoringConfig()


class TPEScorer:
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[TPEScoringConfig] = None,
    ):
        self.study = study
        self.config = config or TPE_SCORING_CONFIG

    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
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
    def _sigmoid(x: float, center: float = 1.0, scale: float = 1.5) -> float:
        """FUNCIÓN SIGMOIDE SIMPLE."""
        try:
            exponent = -scale * (x - center)
            if exponent > 500:
                return 0.0
            elif exponent < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        cfg = self.config
        normalized = self._sigmoid(sharpe, cfg.SHARPE_CENTER, cfg.SHARPE_SCALE)
        return float(np.clip(normalized, 0.01, 0.99))

    def _normalize_sqn(self, sqn: float) -> float:
        cfg = self.config
        normalized = sqn / cfg.SQN_TARGET
        return float(np.clip(normalized, 0.0, 1.0))

    def _normalize_roi(self, roi: float) -> float:
        cfg = self.config
        if roi <= 0:
            normalized = max(0.0, 0.5 + (roi / 200.0))
        else:
            log_roi = math.log1p(roi)
            log_target = math.log1p(cfg.ROI_TARGET)
            normalized = 0.5 + 0.5 * min(1.0, log_roi / log_target)
        return float(np.clip(normalized, 0.01, 0.99))

    def _normalize_drawdown(self, drawdown: float) -> float:
        cfg = self.config
        if drawdown <= cfg.DRAWDOWN_MAX_ACCEPTABLE:
            penalty = 1.0 - (drawdown / cfg.DRAWDOWN_MAX_ACCEPTABLE) * 0.5
        else:
            excess = drawdown - cfg.DRAWDOWN_MAX_ACCEPTABLE
            max_excess = cfg.MAX_DRAWDOWN_LIMIT - cfg.DRAWDOWN_MAX_ACCEPTABLE
            penalty = 0.5 * (1.0 - min(1.0, excess / max_excess))
        return float(np.clip(penalty, 0.01, 1.0))

    def _normalize_trades(self, n_trades: int, days: float) -> float:
        cfg = self.config
        if n_trades < cfg.MIN_TRADES_FOR_VALID:
            return 0.1
        log_trades = math.log1p(n_trades)
        log_target = math.log1p(cfg.MIN_TRADES_TARGET)
        normalized = min(1.0, log_trades / log_target)
        return float(np.clip(normalized, 0.1, 1.0))

    def _calculate_simple_psr(self, returns: np.ndarray) -> float:
        cfg = self.config
        
        if not cfg.PSR_ENABLED:
            return 1.0
        
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        
        if n < cfg.PSR_MIN_TRADES:
            return 0.5  # SIN PENALIZACIÓN SEVERA
        
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        
        if std_r < 1e-10:
            return 0.5
        
        sr = mean_r / std_r
        
        # PSR SIMPLE: Z-SCORE
        std_err = 1.0 / math.sqrt(max(1, n - 1))
        z_score = (sr - cfg.PSR_BENCHMARK) / std_err
        
        # CDF NORMAL
        psr = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2)))
        
        return float(np.clip(psr, 0.1, 0.99))
    
    # =========================================================================
    # [2.4] FUNCIÓN MAESTRA: COMPUTE SCORE
    # =========================================================================
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
        trades_pnl: Optional[np.ndarray] = None,
        total_days: Optional[float] = None,
    ) -> float:
        """
        SCORING TPE — Fórmula única (misma que Hybrid).
        
        Fórmula:
            Score = (Esperanza / (Max_Racha_Perdedora + 1)) × R² × P_lineal(x)
        
        Donde:
            Esperanza = Beneficio Neto Total / N trades
                        (amortiguado si n_trades < 10)
            R²        = Coef. determinación de capital acumulado vs nº trade
            P_lineal  = min(1, trades_per_day / 0.5)
        """
        # ── 1. Extracción y validación de datos ────────────────────────
        if trades_pnl is None:
            raw = metrics.get("_trades_pnl_array", None)
            if raw is not None:
                trades_pnl = np.asarray(raw, dtype=np.float64)

        if trades_pnl is None or trades_pnl.size == 0:
            return 0.0

        if total_days is None or total_days <= 0.0:
            td = self._safe_get(metrics, "_total_days", 0.0)
            if td <= 0.0:
                tpd = self._safe_get(metrics, "trades_por_dia", 0.0)
                n_t = int(self._safe_get(metrics, "n_trades", 0))
                if n_t == 0:
                    n_t = int(self._safe_get(metrics, "total_trades", 0))
                td = float(n_t) / tpd if tpd > 0 else 1.0
            total_days = td

        n_trades = len(trades_pnl)
        if n_trades == 0 or total_days <= 0.0:
            return 0.0

        # ── 2. Cálculos Matemáticos Core ───────────────────────────────

        # A. Esperanza = Beneficio Neto Total / N trades
        #    Con amortiguación para pocos trades (< 10)
        MIN_TRADES_FOR_RELIABLE = 10
        esperanza_raw = float(np.sum(trades_pnl)) / n_trades
        if n_trades < MIN_TRADES_FOR_RELIABLE:
            trade_factor = n_trades / MIN_TRADES_FOR_RELIABLE
            esperanza = esperanza_raw * trade_factor
        else:
            esperanza = esperanza_raw

        # B. Racha Perdedora máxima (Vectorizada)
        es_negativo = (trades_pnl < 0).astype(int)
        padded = np.pad(es_negativo, (1, 1), 'constant', constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_racha = float(np.max(ends - starts)) if len(starts) > 0 else 0.0

        # C. R² (Estabilidad de la curva de capital)
        #    Eje X = Número de trade (1, 2, 3, ...)
        #    Eje Y = Capital acumulado (cumsum de PnL por trade)
        if n_trades > 1:
            x = np.arange(1, n_trades + 1)
            y = np.cumsum(trades_pnl)
            if np.std(y) == 0:
                r_squared = 0.0
            else:
                correlacion = np.corrcoef(x, y)[0, 1]
                r_squared = float(correlacion**2) if not np.isnan(correlacion) else 0.0
        else:
            r_squared = 0.0

        # D. Penalización Lineal — P_lineal(x)
        #    Si x >= 0.5 → 1.0; Si x < 0.5 → x / 0.5
        trades_per_day = n_trades / total_days
        target_tpd = 0.5
        p_lineal = min(1.0, trades_per_day / target_tpd) if target_tpd > 0 else 1.0

        # ── 3. Fórmula Final ───────────────────────────────────────────
        #    Score = (Esperanza / (Max_Racha_Perdedora + 1)) × R² × P_lineal × 100
        final_score = (esperanza / (max_racha + 1.0)) * r_squared * p_lineal * 100.0

        # ── SEGURO: proteger contra NaN / Inf por edge cases numéricos ──
        if not math.isfinite(final_score):
            final_score = 0.0

        # ── 4. Auditoría en Optuna ─────────────────────────────────────
        if trial is not None:
            try:
                trial.set_user_attr('final_score', float(final_score))
                trial.set_user_attr('esperanza', float(esperanza))
                trial.set_user_attr('esperanza_raw', float(esperanza_raw))
                trial.set_user_attr('max_racha', float(max_racha))
                trial.set_user_attr('r_squared', float(r_squared))
                trial.set_user_attr('p_lineal', float(p_lineal))
                trial.set_user_attr('n_trades', int(n_trades))
            except Exception:
                pass

        return float(final_score)


# =============================================================================
# ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
# ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗
# ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝
# ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗
# ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
#  ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
#
# [SECCIÓN 3] CLASE OPTIMIZADOR TPE
# =============================================================================


@dataclass
class TPEOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                  CONFIGURACIÓN DEL OPTIMIZADOR TPE                      │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 3.1 CONFIGURACIÓN DE OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # SEMILLA ALEATORIA (NONE = VARIEDAD)
    N_JOBS: int = 1                       # NÚMERO DE WORKERS PARALELOS
    CREATE_DATABASE: bool = True          # True = crear SQLite por estrategia (IDx.db)
    STUDY_NAME_PREFIX: str = "MODELOX"    # PREFIJO PARA NOMBRES DE ESTUDIO
    
    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA TPE
    # =========================================================================
    N_STARTUP_TRIALS: int = 10            # TRIALS ALEATORIOS INICIALES
    N_EI_CANDIDATES: int = 24             # CANDIDATOS PARA EXPECTED IMPROVEMENT
    MULTIVARIATE: bool = True             # CONSIDERAR DEPENDENCIAS ENTRE PARAMS
    GROUP: bool = True                     # AGRUPAR PARÁMETROS RELACIONADOS
    CONSTANT_LIAR: bool = True             # MEJORAR PARALELISMO


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
TPE_OPTIMIZER_CONFIG = TPEOptimizerConfig()


class TPEOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       OPTIMIZADOR TPE                                   │
    │                                                                         │
    │  TREE-STRUCTURED PARZEN ESTIMATOR                                       │
    │                                                                         │
    │  CARACTERÍSTICAS:                                                       │
    │    ✓ EXPLORACIÓN AMPLIA DEL ESPACIO                                   │
    │    ✓ DIVERSIDAD DE SOLUCIONES                                         │
    │    ✓ IDEAL PARA ESPACIOS GRANDES Y DISCONTINUOS                       │
    │    ✓ MANEJA PARÁMETROS CATEGÓRICOS                                    │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[TPEOptimizerConfig] = None,
        scoring_config: Optional[TPEScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        INICIALIZA EL OPTIMIZADOR TPE.
        
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
        self.optimizer_config = optimizer_config or TPE_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or TPE_SCORING_CONFIG
        self.activo = activo
        
        # ESTADO INTERNO
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[TPEScorer] = None
    
    # =========================================================================
    # [3.1] CREAR ESTUDIO OPTUNA
    # =========================================================================
    
    def _create_study(self, strategy_name: str, strategy_id: Optional[int] = None) -> optuna.Study:
        """
        CREA UN ESTUDIO OPTUNA CON SAMPLER TPE.
        
        RETURNS:
            OBJETO OPTUNA.STUDY CONFIGURADO
        """
        cfg = self.optimizer_config
        
        # CONSTRUIR NOMBRE DEL ESTUDIO
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy_name), "TPE"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        # CREAR SAMPLER TPE
        sampler = TPESampler(
            seed=cfg.SEED,
            n_startup_trials=cfg.N_STARTUP_TRIALS,
            n_ei_candidates=cfg.N_EI_CANDIDATES,
            multivariate=cfg.MULTIVARIATE,
            group=cfg.GROUP,
            constant_liar=cfg.CONSTANT_LIAR,
        )
        
        storage = resolve_storage_for_strategy(
            create_database=bool(cfg.CREATE_DATABASE),
            strategy_id=strategy_id,
        )

        # CREAR ESTUDIO
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=False,
        )
        
        # INICIALIZAR SCORER CON EL ESTUDIO
        self._scorer = TPEScorer(study=study, config=self.scoring_config)
        
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
        # FORZAR salidas SIEMPRE en 1m para máxima precisión (SL/TP/Trailing)
        exit_tf = "1m"
        
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
        """CREA LA FUNCIÓN OBJETIVO PARA TPE."""
        
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
            
            # Extraer pnl_neto array y total_days para score_universal
            if isinstance(trades_df, pl.DataFrame):
                _pnl_arr = trades_df["pnl_neto"].to_numpy().astype(np.float64)
            else:
                _pnl_arr = trades_df["pnl_neto"].to_numpy(dtype=np.float64)
            
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
            
            # CALCULAR SCORE CON SCORER TPE (SCORE UNIVERSAL)
            score = self._scorer.compute_score(
                trial=trial,
                metrics=metrics,
                equity_curve=np.array(equity_curve) if equity_curve else None,
                trades_pnl=_pnl_arr,
                total_days=_total_days,
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
        │                    EJECUTAR OPTIMIZACIÓN TPE                        │
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
        study = self._create_study(
            strategy.name,
            strategy_id=getattr(strategy, "combinacion_id", None),
        )
        
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
    def scorer(self) -> Optional[TPEScorer]:
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


def create_tpe_study(
    strategy_name: str,
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    n_startup_trials: int = 10,
    multivariate: bool = True,
    group: bool = True,
    create_database: bool = True,
    storage: Optional[str] = None,
) -> optuna.Study:
    """
    Crea un estudio Optuna con sampler TPE.
    
    TPE (Tree-structured Parzen Estimator):
    - Algoritmo clásico de Optuna
    - Exploración amplia del espacio de parámetros
    - Bueno para espacios mixtos con variables categóricas
    - Ideal para fase inicial de exploración
    
    Args:
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria
        study_name_prefix: Prefijo para el nombre del estudio
        n_startup_trials: Trials aleatorios iniciales
        multivariate: Considerar dependencias entre parámetros
        group: Agrupar parámetros relacionados
        create_database: True para usar SQLite por estrategia (IDx.db)
        storage: URI de almacenamiento (si se pasa, tiene prioridad)
    
    Returns:
        optuna.Study configurado con TPE
    """
    # Construir nombre del estudio
    parts = [study_name_prefix, str(strategy_name), "tpe"]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    # Crear sampler TPE
    sampler = TPESampler(
        seed=seed,
        multivariate=multivariate,
        group=group,
    )
    
    if storage is None:
        storage = resolve_storage_for_strategy(
            create_database=create_database,
            strategy_id=strategy_id,
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


def score_tpe(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
    trades_pnl: Optional[np.ndarray] = None,
    total_days: Optional[float] = None,
) -> float:
    """
    FUNCIÓN DE SCORING TPE STANDALONE.
    
    USO:
        score = score_tpe(metrics)
    """
    scorer = TPEScorer()
    return scorer.compute_score(
        trial=trial,
        metrics=metrics,
        equity_curve=np.array(equity_curve) if equity_curve else None,
        trades_pnl=trades_pnl,
        total_days=total_days,
    )


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    "TPEOptimizer",
    "TPEOptimizerConfig",
    "TPEScorer",
    "TPEScoringConfig",
    "TPE_SCORING_CONFIG",
    "TPE_OPTIMIZER_CONFIG",
    "create_tpe_study",
    "score_tpe",
]

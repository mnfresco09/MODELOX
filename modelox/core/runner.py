"""modelox/core/runner.py

Runner principal (V2) basado en:
- DataLoader: carga/normaliza datos
- SignalGenerator: genera señales (multi-timeframe opcional)
- BacktestEngine: ejecuta el vector engine

Este archivo reemplaza runner_v2.py para que el sistema use el nombre estable
`modelox.core.runner`.

Nota: se mantiene `OptunaConfig` y `create_study_for_strategy` aquí mismo para
no depender de módulos separados.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import re
import warnings

import optuna
import polars as pl

from optuna.exceptions import ExperimentalWarning

from .engine import BacktestParams, calculate_performance_vectorized_numba
from .metrics import resumen_metricas
from .scoring import score_optuna
from .types import BacktestConfig, Reporter, Strategy, TrialArtifacts, normalize_timeframe_to_suffix
from .data_blender import prepare_multitimeframe_data
from .exits import resolve_exit_settings_for_trial

# Silenciar warnings experimentales de Optuna (multivariate/group en TPESampler)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Debug timings (activar con env MODELOX_TIMINGS_VERBOSE=1)
_TIMINGS_VERBOSE = os.environ.get("MODELOX_TIMINGS_VERBOSE", "0") in {"1", "true", "True", "YES", "yes"}
_TIMINGS_PRINT_EVERY = int(os.environ.get("MODELOX_TIMINGS_PRINT_EVERY", "1"))


@dataclass(frozen=True)
class OptunaConfig:
    seed: Optional[int] = None
    n_jobs: int = 1
    storage: Optional[str] = None
    study_name_prefix: str = "MODELOX"
    sampler: str = "tpe"


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_study_for_strategy(
    *,
    cfg: OptunaConfig,
    strategy_name: str,
    activo: Optional[str] = None,
) -> optuna.study.Study:
    sampler: optuna.samplers.BaseSampler
    if str(cfg.sampler).lower() == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=cfg.seed,
            multivariate=True,
            group=True,
        )
    else:
        raise ValueError(f"Sampler no soportado: {cfg.sampler}")

    parts = [str(cfg.study_name_prefix), str(strategy_name)]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))

    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=None,
        load_if_exists=False,
    )


@dataclass
class DataLoader:
    """Handles data loading and normalization."""

    @staticmethod
    def load_data(file_path: str) -> pl.DataFrame:
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pl.read_csv(file_path)
        elif file_path.endswith(".feather") or file_path.endswith(".arrow"):
            df = pl.read_ipc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if "timestamp" not in df.columns and "datetime" in df.columns:
            df = df.rename({"datetime": "timestamp"})

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' or 'datetime' column")

        return df


@dataclass
class SignalGenerator:
    """Executes strategy and returns a DataFrame with signals."""

    @staticmethod
    def generate_signals(
        df: pl.DataFrame,
        strategy: Strategy,
        params: Dict[str, Any],
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> pl.DataFrame:
        base_tf = normalize_timeframe_to_suffix(params.get("__timeframe_base", "1h"))

        if hasattr(strategy, "get_required_timeframes") and callable(strategy.get_required_timeframes):
            required_tfs = strategy.get_required_timeframes(params)
            if required_tfs and df_by_timeframe:
                df = prepare_multitimeframe_data(
                    df,
                    required_tfs,
                    base_tf=base_tf,
                    anti_lookahead=True,
                )

        signals_df = strategy.generate_signals(df, params)

        if "signal_long" not in signals_df.columns:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_long"))
        if "signal_short" not in signals_df.columns:
            signals_df = signals_df.with_columns(pl.lit(False).alias("signal_short"))

        return signals_df


@dataclass
class BacktestEngine:
    """Takes signals + prices and returns metrics, without knowing about Optuna."""

    @staticmethod
    def run_backtest(
        df: pl.DataFrame,
        signals: pl.DataFrame,
        config: BacktestConfig,
        params: Dict[str, Any],
        strategy: Strategy,
    ) -> tuple[pl.DataFrame, list[float], Dict[str, Any]]:
        backtest_params = BacktestParams.from_config_and_params(config, params)

        trades_df, equity_curve = calculate_performance_vectorized_numba(
            df=df,
            signals=signals,
            params=backtest_params,
            strategy=strategy,
        )

        metrics: Dict[str, Any]
        if not trades_df.is_empty():
            metrics = resumen_metricas(
                trades_df,
                saldo_inicial=config.saldo_inicial,
                equity_curve=equity_curve,
            )
        else:
            metrics = {}

        return trades_df, equity_curve, metrics


@dataclass
class OptimizationRunner:
    """Optimization runner using vectorized engine (V2)."""

    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = OptunaConfig()
    activo: Optional[str] = None

    # Para UI / post-processing
    _last_study: Optional[optuna.study.Study] = None

    def optimize_strategies(
        self,
        *,
        df: pl.DataFrame,
        strategies: Sequence[Strategy],
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for strat in strategies:
            study = self._optimize_one(
                df=df,
                strategy=strat,
                df_by_timeframe=df_by_timeframe,
                base_timeframe=base_timeframe,
            )
            results[strat.name] = study
            self._last_study = study

            # Llamar on_strategy_end para mostrar resumen (top 5 trials, etc.)
            for reporter in self.reporters:
                if hasattr(reporter, "on_strategy_end"):
                    try:
                        reporter.on_strategy_end(strat.name, study)
                    except Exception:
                        pass  # Nunca romper el flujo por errores de reporting
        return results

    def _optimize_one(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.study.Study:
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1h")
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)

        def objective(trial: optuna.trial.Trial) -> float:
            t0_total = time.perf_counter()
            
            params_puros = strategy.suggest_params(trial)
            params_rt = dict(params_puros)

            params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
            params_rt["__comision_pct"] = float(self.config.comision_pct)
            params_rt["__comision_sides"] = int(self.config.comision_sides)
            params_rt["__saldo_usado"] = float(self.config.saldo_usado)
            params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)

            # Reporting/UI: marcar si la estrategia controla salidas
            params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))

            # Usar resolve_exit_settings_for_trial (soporta optimize_exits)
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
            params_rt["__exit_type"] = exit_settings.exit_type
            params_rt["__exit_sl_pct"] = exit_settings.sl_pct
            params_rt["__exit_tp_pct"] = exit_settings.tp_pct
            params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
            params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
            
            # Copiar valores visibles para mostrar en params (usados por Rich)
            params_rt["exit_type"] = exit_settings.exit_type
            params_rt["exit_sl_pct"] = exit_settings.sl_pct
            params_rt["exit_tp_pct"] = exit_settings.tp_pct
            params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
            params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct

            entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
            exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)

            params_rt["__timeframe_base"] = base_tf
            params_rt["__timeframe_entry"] = entry_tf
            params_rt["__timeframe_exit"] = exit_tf

            df_entry = df_map.get(entry_tf, df_base)

            t1_signals = time.perf_counter()
            signals_df = SignalGenerator.generate_signals(
                df_entry,
                strategy,
                params_rt,
                df_by_timeframe,
            )
            t2_signals = time.perf_counter()

            t1_backtest = time.perf_counter()
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_base,
                signals_df,
                self.config,
                params_rt,
                strategy,
            )
            t2_backtest = time.perf_counter()

            if trades_df.is_empty():
                return 0.0

            trial.set_user_attr("metricas", metrics)
            score = float(score_optuna(metrics))
            
            t_total = time.perf_counter() - t0_total
            
            # Debug timings
            if _TIMINGS_VERBOSE and (trial.number % _TIMINGS_PRINT_EVERY == 0):
                print(
                    f"  ⏱ TRIAL {trial.number:3d} │ "
                    f"signals {(t2_signals - t1_signals)*1000:6.1f}ms │ "
                    f"backtest {(t2_backtest - t1_backtest)*1000:6.1f}ms │ "
                    f"total {t_total*1000:6.1f}ms │ "
                    f"trades {len(trades_df):5d}"
                )

            # Determinar si algún reporter necesita df_signals (ej: PlotReporter para top-K)
            # Si lo necesita, combinamos OHLC de df_base con señales+indicadores de signals_df
            df_signals_for_artifacts = None
            for reporter in self.reporters:
                if hasattr(reporter, "needs_dataframe") and reporter.needs_dataframe(score):
                    # Columnas OHLC base
                    ohlc_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    base_cols = [c for c in ohlc_cols if c in df_base.columns]
                    # Columnas de señales/indicadores (sin duplicar las que ya están en base)
                    signal_cols = [c for c in signals_df.columns if c not in base_cols]
                    # hstack es seguro porque ambos DFs tienen mismo nº de filas (mismo origen)
                    df_signals_for_artifacts = df_base.select(base_cols).hstack(
                        signals_df.select(signal_cols)
                    )
                    break

            artifacts = TrialArtifacts(
                strategy_name=strategy.name,
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_rt,
                score=score,
                metrics=metrics,
                df_signals=df_signals_for_artifacts,
                trades=trades_df.to_pandas(),
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
            )

            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)

            return score

        study = create_study_for_strategy(cfg=self.optuna, strategy_name=strategy.name, activo=self.activo)

        study.optimize(
            objective,
            n_trials=int(self.n_trials),
            n_jobs=int(getattr(self.optuna, "n_jobs", 1)),
            gc_after_trial=True,
            catch=(Exception,),
        )

        return study

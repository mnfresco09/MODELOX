from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import optuna
import pandas as pd
import polars as pl

from modelox.core.engine import generate_trades, simulate_trades
from modelox.core.metrics import resumen_metricas
from modelox.core.optuna_config import OptunaConfig
from modelox.core.optuna_utils import create_study_for_strategy
from modelox.core.scoring import score_optuna
from modelox.core.types import BacktestConfig, Reporter, Strategy, TrialArtifacts


@dataclass
class OptimizationRunner:
    """Ejecuta la optimización de Optuna usando Polars para el backtest."""

    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = OptunaConfig()
    activo: Optional[str] = None

    def optimize_strategies(
        self, *, df: pl.DataFrame, strategies: Sequence[Strategy]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for strat in strategies:
            study = self._optimize_one(df=df, strategy=strat)
            results[strat.name] = study
        return results

    def _optimize_one(self, *, df: pl.DataFrame, strategy: Strategy):
        saldo_apertura = float(self.config.saldo_inicial)
        best_score_so_far: float = float("-inf")

        def objetivo(trial: optuna.trial.Trial) -> float:
            nonlocal best_score_so_far
            params_puros = strategy.suggest_params(trial)

            # 1. Generación de señales en Polars
            df_signals = strategy.generate_signals(df, params_puros)

            # 2. Generación y simulación de trades
            trades_base = generate_trades(
                df_signals,
                params_puros,
                saldo_apertura=saldo_apertura,
                strategy=strategy,
            )
            trades_exec, equity_curve = simulate_trades(
                trades_base=trades_base, config=self.config
            )

            if trades_exec is None or trades_exec.empty:
                return -9999.0

            # 3. Métricas y Scoring
            metricas = resumen_metricas(
                trades_exec, saldo_inicial=saldo_apertura, equity_curve=equity_curve
            )
            score = float(score_optuna(metricas))
            if score > best_score_so_far:
                best_score_so_far = score

            # ⭐ STORE METRICS IN TRIAL USER_ATTRS (Critical for mostrar_top_trials)
            trial.set_user_attr("metricas", metricas)

            # 4. PREPARACIÓN PARA REPORTING (Conversión Polars -> Pandas Indexado)
            # Esto evita el TypeError: '>=' not supported between instances of 'numpy.ndarray' and 'Timestamp'
            df_pandas = df_signals.to_pandas()

            if "timestamp" in df_pandas.columns:
                df_pandas["timestamp"] = pd.to_datetime(
                    df_pandas["timestamp"], utc=True
                )
                df_pandas = df_pandas.set_index("timestamp")
            elif "datetime" in df_pandas.columns:
                # Fallback: a veces la columna se llama datetime
                df_pandas["datetime"] = pd.to_datetime(df_pandas["datetime"], utc=True)
                df_pandas = df_pandas.set_index("datetime")
                df_pandas.index.name = "timestamp"

            indicators_used = sorted(
                [
                    c
                    for c in df_signals.columns
                    if c
                    not in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "signal_long",
                        "signal_short",
                        "timestamp",
                    }
                ]
            )


            # Modular: always set params_reporting and ensure indicator list is from strategy class
            params_reporting = dict(params_puros)
            # Use the class property if available, fallback to detected columns
            if hasattr(strategy, 'get_indicators_used'):
                params_reporting["__indicators_used"] = strategy.get_indicators_used()
            else:
                params_reporting["__indicators_used"] = indicators_used
            params_reporting["__best_score_so_far"] = best_score_so_far

            artifacts = TrialArtifacts(
                strategy_name=strategy.name,
                trial_number=trial.number,
                params=params_reporting,
                params_reporting=params_reporting,
                score=score,
                metrics=metricas,
                df_signals=df_pandas,
                trades=trades_exec,
                equity_curve=equity_curve,
                indicators_used=indicators_used,
            )

            for r in self.reporters:
                r.on_trial_end(artifacts)
            return score

        study = create_study_for_strategy(
            cfg=self.optuna, strategy_name=strategy.name, activo=self.activo
        )
        study.optimize(
            objetivo,
            n_trials=int(self.n_trials),
            n_jobs=max(1, int(self.optuna.n_jobs)),
            gc_after_trial=True,
        )
        for r in self.reporters:
            r.on_strategy_end(strategy.name, study)
        return study

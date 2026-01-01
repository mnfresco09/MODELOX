from __future__ import annotations

from dataclasses import dataclass, replace
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

            # Runtime params: include config so strategies can do sizing-aware exits if needed.
            # Kept under __ prefix so reporters can ignore them.
            params_rt = dict(params_puros)
            params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
            params_rt["__apalancamiento"] = float(self.config.apalancamiento)
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
            params_rt["__comision_pct"] = float(self.config.comision_pct)
            params_rt["__comision_sides"] = int(self.config.comision_sides)

            # Qty cap (per asset): optionally optimized by Optuna
            optimize_qty = bool(getattr(self.config, "optimize_qty_max_activo", False))
            qty_max_activo = float(getattr(self.config, "qty_max_activo", float("inf")))
            if optimize_qty:
                q_rng = tuple(getattr(self.config, "qty_max_activo_range", (0.01, 5.0, 0.01)))
                q_min = float(q_rng[0])
                q_max = float(q_rng[1])
                q_step = float(q_rng[2]) if len(q_rng) >= 3 else None
                if q_step is None or q_step <= 0:
                    qty_max_activo = float(trial.suggest_float("qty_max_activo", q_min, q_max))
                else:
                    qty_max_activo = float(trial.suggest_float("qty_max_activo", q_min, q_max, step=q_step))

            # Ensure engine sees the chosen qty cap
            params_rt["__qty_max_activo"] = float(qty_max_activo)

            # simulate_trades reads qty cap from config, so create a per-trial config
            config_trial = self.config if not optimize_qty else replace(self.config, qty_max_activo=float(qty_max_activo))

            # Global exits (engine-owned)
            optimize_exits = bool(getattr(self.config, "optimize_exits", False))

            exit_atr_period = int(getattr(self.config, "exit_atr_period", 14))
            exit_sl_atr = float(getattr(self.config, "exit_sl_atr", 1.0))
            exit_tp_atr = float(getattr(self.config, "exit_tp_atr", 1.0))
            exit_time_stop_bars = int(getattr(self.config, "exit_time_stop_bars", 260))

            if optimize_exits:
                p_rng = tuple(getattr(self.config, "exit_atr_period_range", (7, 30, 1)))
                sl_rng = tuple(getattr(self.config, "exit_sl_atr_range", (0.5, 5.0, 0.1)))
                tp_rng = tuple(getattr(self.config, "exit_tp_atr_range", (0.5, 8.0, 0.1)))
                ts_rng = tuple(getattr(self.config, "exit_time_stop_bars_range", (50, 800, 10)))

                p_min, p_max, p_step = (int(p_rng[0]), int(p_rng[1]), int(p_rng[2]) if len(p_rng) >= 3 else 1)
                ts_min, ts_max, ts_step = (
                    int(ts_rng[0]),
                    int(ts_rng[1]),
                    int(ts_rng[2]) if len(ts_rng) >= 3 else 1,
                )
                sl_min, sl_max, sl_step = (
                    float(sl_rng[0]),
                    float(sl_rng[1]),
                    float(sl_rng[2]) if len(sl_rng) >= 3 else 0.1,
                )
                tp_min, tp_max, tp_step = (
                    float(tp_rng[0]),
                    float(tp_rng[1]),
                    float(tp_rng[2]) if len(tp_rng) >= 3 else 0.1,
                )

                exit_atr_period = int(trial.suggest_int("exit_atr_period", p_min, p_max, step=max(1, p_step)))
                exit_sl_atr = float(trial.suggest_float("exit_sl_atr", sl_min, sl_max, step=sl_step))
                exit_tp_atr = float(trial.suggest_float("exit_tp_atr", tp_min, tp_max, step=tp_step))
                exit_time_stop_bars = int(
                    trial.suggest_int("exit_time_stop_bars", ts_min, ts_max, step=max(1, ts_step))
                )

            params_rt["__exit_atr_period"] = int(exit_atr_period)
            params_rt["__exit_sl_atr"] = float(exit_sl_atr)
            params_rt["__exit_tp_atr"] = float(exit_tp_atr)
            params_rt["__exit_time_stop_bars"] = int(exit_time_stop_bars)

            # 1. Generación de señales en Polars
            df_signals = strategy.generate_signals(df, params_rt)

            # 2. Generación y simulación de trades
            trades_base = generate_trades(
                df_signals,
                params_rt,
                saldo_apertura=saldo_apertura,
                strategy=strategy,
            )
            trades_exec, equity_curve = simulate_trades(
                trades_base=trades_base, config=config_trial
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
            # Expose global exits in reports (so you can see what Optuna picked)
            params_reporting["exit_atr_period"] = int(exit_atr_period)
            params_reporting["exit_sl_atr"] = float(exit_sl_atr)
            params_reporting["exit_tp_atr"] = float(exit_tp_atr)
            params_reporting["exit_time_stop_bars"] = int(exit_time_stop_bars)
            params_reporting["qty_max_activo"] = float(qty_max_activo)
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

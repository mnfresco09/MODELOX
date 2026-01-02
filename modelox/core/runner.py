from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence

import os
import time

import optuna
import pandas as pd
import polars as pl

from modelox.core.engine import generate_trades, simulate_trades
from modelox.core.metrics import resumen_metricas
from modelox.core.optuna_config import OptunaConfig
from modelox.core.optuna_utils import create_study_for_strategy
from modelox.core.scoring import score_optuna
from modelox.core.types import BacktestConfig, Reporter, Strategy, TrialArtifacts
from modelox.core.exits import resolve_exit_settings_for_trial
from modelox.core.timeframes import (
    align_signals_to_base,
    convert_warmup_bars_to_base,
    normalize_timeframe_to_suffix,
)


@dataclass
class OptimizationRunner:
    """Ejecuta la optimización de Optuna usando Polars para el backtest."""

    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    optuna: OptunaConfig = OptunaConfig()
    activo: Optional[str] = None

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
        return results

    def _optimize_one(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ):
        saldo_apertura = float(self.config.saldo_inicial)
        best_score_so_far: float = float("-inf")

        base_tf = normalize_timeframe_to_suffix(base_timeframe)
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)

        # Periodo del backtest (ya filtrado por FECHA_INICIO/FECHA_FIN en ejecutar.py).
        # Se usa para métricas como trades/día, con fin capado al último trade ejecutado.
        period_start = None
        period_end = None
        try:
            ts_col = "timestamp" if "timestamp" in df_base.columns else ("datetime" if "datetime" in df_base.columns else None)
            if ts_col:
                period_start = df_base.select(pl.col(ts_col).min()).item()
                period_end = df_base.select(pl.col(ts_col).max()).item()
        except Exception:
            period_start = None
            period_end = None

        timings_enabled = os.environ.get("MODELOX_TIMINGS", "0") in {"1", "true", "True", "YES", "yes"}
        timings_acc: Dict[str, float] = {
            "generate_signals_s": 0.0,
            "align_signals_s": 0.0,
            "generate_trades_s": 0.0,
            "simulate_trades_s": 0.0,
            "reporting_s": 0.0,
            "trials": 0.0,
        }

        def objetivo(trial: optuna.trial.Trial) -> float:
            nonlocal best_score_so_far
            params_puros = strategy.suggest_params(trial)

            # Runtime params: include config so strategies can do sizing-aware exits if needed.
            # Kept under __ prefix so reporters can ignore them.
            params_rt = dict(params_puros)
            params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
            params_rt["__comision_pct"] = float(self.config.comision_pct)
            params_rt["__comision_sides"] = int(self.config.comision_sides)
            
            # Position Sizing: SALDO_USADO fijo, QTY fija, APALANCAMIENTO variable
            params_rt["__saldo_usado"] = float(self.config.saldo_usado)
            params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)

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

            # Global exits (centralizados): `modelox/core/exits.py`
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=config_trial)


            # IMPORTANT: engine reads exits ONLY from params.
            # Sistema porcentual - inyectar parámetros de salida
            params_rt["__exit_type"] = str(exit_settings.exit_type)
            params_rt["__exit_sl_pct"] = float(exit_settings.sl_pct)
            params_rt["__exit_tp_pct"] = float(exit_settings.tp_pct)
            params_rt["__exit_trail_act_pct"] = float(exit_settings.trail_act_pct)
            params_rt["__exit_trail_dist_pct"] = float(exit_settings.trail_dist_pct)

            # ===== TIMEFRAMES (entry/exit) =====
            entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
            exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)

            params_rt["__timeframe_base"] = base_tf
            params_rt["__timeframe_entry"] = entry_tf
            params_rt["__timeframe_exit"] = exit_tf

            # Expose multi-timeframe dataframes to strategy (optional usage)
            if isinstance(df_map, dict) and df_map:
                params_rt["__df_by_timeframe"] = df_map
            if exit_tf in df_map:
                params_rt["__df_exit_tf"] = df_map[exit_tf]

            # 1. Generación de señales en Polars (en timeframe de entrada)
            df_entry = df_map.get(entry_tf, df_base)
            t_sig0 = time.perf_counter()
            df_signals_entry = strategy.generate_signals(df_entry, params_rt)
            t_sig1 = time.perf_counter()
            if timings_enabled:
                dt = float(t_sig1 - t_sig0)
                timings_acc["generate_signals_s"] += dt
                trial.set_user_attr("timing_generate_signals_s", dt)

            # Si la estrategia generó señales en otro TF, alinear al TF base (sin lookahead)
            if entry_tf != base_tf:
                t_al0 = time.perf_counter()
                # Ajustar warmup a base TF si la estrategia lo definió en el TF de entrada
                warmup_entry = params_rt.get("__warmup_bars", None)
                if warmup_entry is not None:
                    try:
                        params_rt["__warmup_bars"] = convert_warmup_bars_to_base(
                            warmup_bars=int(warmup_entry),
                            from_tf=entry_tf,
                            to_tf=base_tf,
                        )
                    except Exception:
                        pass

                df_signals = align_signals_to_base(df_base=df_base, df_signals=df_signals_entry)
                t_al1 = time.perf_counter()
                if timings_enabled:
                    dt = float(t_al1 - t_al0)
                    timings_acc["align_signals_s"] += dt
                    trial.set_user_attr("timing_align_signals_s", dt)
            else:
                df_signals = df_signals_entry

            # 2. Generación y simulación de trades
            t_gt0 = time.perf_counter()
            trades_base = generate_trades(
                df_signals,
                params_rt,
                saldo_apertura=saldo_apertura,
                strategy=strategy,
            )
            t_gt1 = time.perf_counter()
            if timings_enabled:
                dt = float(t_gt1 - t_gt0)
                timings_acc["generate_trades_s"] += dt
                trial.set_user_attr("timing_generate_trades_s", dt)

            t_sim0 = time.perf_counter()
            trades_exec, equity_curve = simulate_trades(
                trades_base=trades_base, config=config_trial
            )
            t_sim1 = time.perf_counter()
            if timings_enabled:
                dt = float(t_sim1 - t_sim0)
                timings_acc["simulate_trades_s"] += dt
                trial.set_user_attr("timing_simulate_trades_s", dt)

            if trades_exec is None or trades_exec.empty:
                if timings_enabled:
                    timings_acc["trials"] += 1.0
                return 0.0

            # 3. Métricas y Scoring
            metricas = resumen_metricas(
                trades_exec,
                saldo_inicial=saldo_apertura,
                equity_curve=equity_curve,
                period_start=period_start,
                period_end=period_end,
            )
            score = float(score_optuna(metricas))
            if score > best_score_so_far:
                best_score_so_far = score

            # ⭐ STORE METRICS IN TRIAL USER_ATTRS (Critical for mostrar_top_trials)
            trial.set_user_attr("metricas", metricas)

            # 4. REPORTING (OPTIMIZADO)
            # Convertir Polars -> Pandas SOLO si algún reporter lo necesita.
            # Usa el método needs_dataframe() de la interfaz BaseReporter.
            df_pandas = None
            try:
                need_df_for_plot = any(
                    r.needs_dataframe(score)
                    for r in self.reporters
                )
            except Exception:
                need_df_for_plot = False

            if need_df_for_plot:
                # Esto evita el TypeError: '>=' not supported entre ndarray y Timestamp
                df_pandas = df_signals.to_pandas()

                if "timestamp" in df_pandas.columns:
                    df_pandas["timestamp"] = pd.to_datetime(df_pandas["timestamp"], utc=True)
                    df_pandas = df_pandas.set_index("timestamp")
                elif "datetime" in df_pandas.columns:
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


            # Modular: always set params_reporting and ensure indicator list is from the *actual trial*.
            # NOTE: strategies are allowed to mutate `params_rt` inside generate_signals()
            # to include things like __indicators_used / __warmup_bars.
            params_reporting = dict(params_puros)
            # Expose global exits in reports (so you can see what Optuna picked)
            params_reporting["exit_type"] = str(exit_settings.exit_type)
            params_reporting["exit_sl_pct"] = float(exit_settings.sl_pct)
            params_reporting["exit_tp_pct"] = float(exit_settings.tp_pct)
            params_reporting["exit_time_stop_bars"] = int(exit_settings.time_stop_bars)
            params_reporting["exit_trail_act_pct"] = float(exit_settings.trail_act_pct)
            params_reporting["exit_trail_dist_pct"] = float(exit_settings.trail_dist_pct)
            params_reporting["qty_max_activo"] = float(qty_max_activo)
            # Expose active asset to reporters (for filesystem routing, e.g. Excel folders)
            if getattr(self, "activo", None) is not None:
                params_reporting["__activo"] = str(self.activo)
            # Prefer what the strategy computed for THIS trial (dynamic), else fallback.
            indicators_used_trial = params_rt.get("__indicators_used", None)
            if isinstance(indicators_used_trial, list) and indicators_used_trial:
                params_reporting["__indicators_used"] = indicators_used_trial
            elif hasattr(strategy, "get_indicators_used"):
                params_reporting["__indicators_used"] = strategy.get_indicators_used()
            else:
                params_reporting["__indicators_used"] = indicators_used

            # Strategy-driven plot metadata (optional): reference lines, panel overrides, etc.
            # These are consumed by `visual/grafico.py` and should require zero hardcoding there.
            for k in ("__indicator_bounds", "__indicator_specs", "__indicator_panels"):
                v = params_rt.get(k, None)
                if isinstance(v, dict) and v:
                    params_reporting[k] = v
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

            t_rep0 = time.perf_counter()
            for r in self.reporters:
                r.on_trial_end(artifacts)
            t_rep1 = time.perf_counter()
            if timings_enabled:
                dt = float(t_rep1 - t_rep0)
                timings_acc["reporting_s"] += dt
                trial.set_user_attr("timing_reporting_s", dt)
                timings_acc["trials"] += 1.0
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

        if timings_enabled:
            trials_done = max(1.0, float(timings_acc.get("trials", 0.0) or 0.0))
            study.set_user_attr(
                "timings_avg_s",
                {
                    "generate_signals_s": float(timings_acc["generate_signals_s"] / trials_done),
                    "align_signals_s": float(timings_acc["align_signals_s"] / trials_done),
                    "generate_trades_s": float(timings_acc["generate_trades_s"] / trials_done),
                    "simulate_trades_s": float(timings_acc["simulate_trades_s"] / trials_done),
                    "reporting_s": float(timings_acc["reporting_s"] / trials_done),
                    "trials": int(trials_done),
                },
            )
        for r in self.reporters:
            r.on_strategy_end(strategy.name, study)
        return study

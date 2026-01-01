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

            # Global exits (centralizados): `modelox/core/exits.py`
            exit_settings = resolve_exit_settings_for_trial(trial=trial, config=config_trial)

            params_rt["__exit_atr_period"] = int(exit_settings.atr_period)
            params_rt["__exit_sl_atr"] = float(exit_settings.sl_atr)
            params_rt["__exit_tp_atr"] = float(exit_settings.tp_atr)
            params_rt["__exit_time_stop_bars"] = int(exit_settings.time_stop_bars)

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
            df_signals_entry = strategy.generate_signals(df_entry, params_rt)

            # Si la estrategia generó señales en otro TF, alinear al TF base (sin lookahead)
            if entry_tf != base_tf:
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
            else:
                df_signals = df_signals_entry

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
                return 0.0

            # 3. Métricas y Scoring
            metricas = resumen_metricas(
                trades_exec, saldo_inicial=saldo_apertura, equity_curve=equity_curve
            )
            score = float(score_optuna(metricas))
            if score > best_score_so_far:
                best_score_so_far = score

            # ⭐ STORE METRICS IN TRIAL USER_ATTRS (Critical for mostrar_top_trials)
            trial.set_user_attr("metricas", metricas)

            # 4. REPORTING (OPTIMIZADO)
            # Convertir Polars -> Pandas SOLO si algún reporter lo necesita.
            # En la práctica, el único que lo necesita es PlotReporter (para el HTML).
            df_pandas = None
            try:
                plot_reporters = [
                    r
                    for r in self.reporters
                    if r.__class__.__name__ == "PlotReporter"
                ]
                need_df_for_plot = any(
                    getattr(r, "_should_generate_plot")(score)  # type: ignore[misc]
                    for r in plot_reporters
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
            params_reporting["exit_atr_period"] = int(exit_settings.atr_period)
            params_reporting["exit_sl_atr"] = float(exit_settings.sl_atr)
            params_reporting["exit_tp_atr"] = float(exit_settings.tp_atr)
            params_reporting["exit_time_stop_bars"] = int(exit_settings.time_stop_bars)
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

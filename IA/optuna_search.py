"""
================================================================================
IA/OPTUNA_SEARCH.PY — OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS
================================================================================
Usa Optuna (TPE sampler) para buscar:
  - gru_units   : [32, 64, 128, 256]
  - batch_size  : [16, 32, 64, 128]
  - lr          : [5e-5, 5e-4]  (log-uniform)
  - dropout     : [0.1, 0.5]
  - lookback    : [12, 24, 48, 72]
  - alpha_loss  : [1.5, 4.0]

Métrica objetivo: val_loss en el primer fold walk-forward.
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, MofNCompleteColumn

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from IA.config import (
    N_OPTUNA_TRIALS, OPTUNA_TIMEOUT,
    OPTUNA_GRU_UNITS, OPTUNA_BATCH_SIZES,
    OPTUNA_LR_RANGE, OPTUNA_DROPOUT_RANGE, OPTUNA_LOOKBACK,
    QUICK_MODE, QUICK_TRIALS, QUICK_MAX_EPOCHS, MAX_EPOCHS,
)
from IA.model import build_model
from IA.trainer import train_model
from IA.walk_forward import WalkForwardFold
from IA.data_pipeline import build_sequences

console = Console()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparams(
    fold: WalkForwardFold,
    n_trials:   int  = N_OPTUNA_TRIALS,
    timeout:    int  = OPTUNA_TIMEOUT,
    verbose:    bool = True,
) -> dict:
    """
    Ejecuta búsqueda Bayesiana sobre el primer fold walk-forward.
    Devuelve los mejores hiperparámetros encontrados.

    Args:
      fold     : WalkForwardFold con X_train/y_train/X_val/y_val
      n_trials : Número de trials Optuna
      timeout  : Timeout en segundos
      verbose  : Mostrar progreso Rich

    Returns:
      best_params dict con: gru_units, batch_size, lr, dropout, lookback, alpha_loss
    """
    if QUICK_MODE:
        n_trials = QUICK_TRIALS
        max_ep   = QUICK_MAX_EPOCHS
    else:
        max_ep   = min(50, MAX_EPOCHS)  # reducido para Optuna (velocidad)

    # ── Trial results para mostrar ────────────────────────────────────
    trial_results = []

    def objective(trial: optuna.Trial) -> float:
        # ── Sugerir hiperparámetros ───────────────────────────────────
        gru_units  = trial.suggest_categorical("gru_units",  OPTUNA_GRU_UNITS)
        batch_size = trial.suggest_categorical("batch_size", OPTUNA_BATCH_SIZES)
        lr         = trial.suggest_float("lr",     *OPTUNA_LR_RANGE, log=True)
        dropout    = trial.suggest_float("dropout", *OPTUNA_DROPOUT_RANGE)
        lookback   = trial.suggest_categorical("lookback", OPTUNA_LOOKBACK)
        alpha_loss = trial.suggest_float("alpha_loss", 1.5, 4.0)

        # ── Reconstruir secuencias si lookback cambió ─────────────────
        from IA.data_pipeline import FEATURE_COLS
        import IA.config as cfg

        # Necesitamos re-crear sequences con nuevo lookback
        # Usamos los datos ya cargados del fold
        n_f = fold.X_train.shape[2] if fold.X_train is not None else cfg.N_FEATURES

        try:
            model = build_model(
                n_features = n_f,
                gru_units  = gru_units,
                n_layers   = cfg.N_GRU_LAYERS,
                dropout    = dropout,
                fc_units   = cfg.FC_UNITS,
            )

            result = train_model(
                model      = model,
                X_train    = fold.X_train,
                y_train    = fold.y_train,
                X_val      = fold.X_val,
                y_val      = fold.y_val,
                fold_label = f"Optuna T{trial.number+1}",
                batch_size = batch_size,
                lr         = lr,
                max_epochs = max_ep,
                patience   = 8,   # patience reducido para trials Optuna
                pos_weight = cfg.CLASS_WEIGHT_MINORITY,
                verbose    = False,
            )
            val_loss = result["best_val_loss"]

            trial_results.append({
                "trial"     : trial.number + 1,
                "val_loss"  : val_loss,
                "gru_units" : gru_units,
                "batch_size": batch_size,
                "lr"        : lr,
                "dropout"   : dropout,
            })

        except Exception as e:
            val_loss = float("inf")

        finally:
            # Liberar memoria GPU/CPU
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return val_loss

    # ── Crear estudio Optuna ──────────────────────────────────────────
    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    if verbose:
        console.print(Panel(
            f"[cyan]Lanzando [bold]{n_trials}[/bold] trials Optuna...\n"
            f"Timeout: [bold]{timeout}s[/bold] | Sampler: TPE | Max épocas/trial: [bold]{max_ep}[/bold][/cyan]",
            title="[bold white]🔮 OPTIMIZACIÓN BAYESIANA (OPTUNA)[/bold white]",
            border_style="magenta",
        ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Trials Optuna", total=n_trials)

        def _callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
            progress.advance(task)
            best_v = study.best_value
            n_done = len(study.trials)
            progress.update(task,
                description=f"[bold magenta]Trials Optuna | Best val_loss: {best_v:.5f}")

        study.optimize(
            objective,
            n_trials  = n_trials,
            timeout   = timeout,
            callbacks = [_callback],
            show_progress_bar = False,
        )

    best = study.best_params
    best["best_val_loss"] = study.best_value

    # ── Mostrar top 5 trials ──────────────────────────────────────────
    if verbose:
        from rich.table import Table
        tbl = Table(
            show_header=True, header_style="bold magenta",
            border_style="magenta",
        )
        tbl.add_column("Trial",      width=7,  justify="right")
        tbl.add_column("Val Loss",   width=12, justify="right")
        tbl.add_column("GRU Units",  width=10, justify="right")
        tbl.add_column("Batch",      width=7,  justify="right")
        tbl.add_column("LR",         width=12, justify="right")
        tbl.add_column("Dropout",    width=9,  justify="right")

        sorted_trials = sorted(
            study.trials, key=lambda t: t.value if t.value is not None else float("inf")
        )[:5]

        for t in sorted_trials:
            if t.value is None:
                continue
            tbl.add_row(
                str(t.number + 1),
                f"{t.value:.5f}",
                str(t.params.get("gru_units",  "—")),
                str(t.params.get("batch_size", "—")),
                f"{t.params.get('lr', 0):.2e}",
                f"{t.params.get('dropout', 0):.2f}",
            )

        console.print(Panel(tbl,
            title="[bold white]🏆 TOP 5 TRIALS OPTUNA[/bold white]",
            border_style="magenta"))

    return best

"""
================================================================================
IA/TRAINER.PY — BUCLE DE ENTRENAMIENTO CON RICH LIVE DISPLAY
================================================================================
Gestiona:
  - DataLoaders con class weights
  - Early stopping (patience=15, monitor=val_loss)
  - Ajuste de LR con ReduceLROnPlateau
  - Display Rich ultra-detallado en tiempo real
================================================================================
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from IA.config import (
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE,
    CLASS_WEIGHT_MINORITY, ALPHA_LOSS,
)
from IA.model import GRUTradingModel, AsymmetricDirectionalLoss, DEVICE, build_model

console = Console()


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Para el entrenamiento cuando val_loss no mejora en `patience` épocas.
    Guarda el mejor modelo en memoria.
    """

    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-5) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.best_state = None
        self.counter    = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
            return False   # continuar
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True   # parar
            return False

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =============================================================================
# DATASET Y DATALOADERS
# =============================================================================

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders con WeightedRandomSampler para manejar desbalance de clases.
    """
    # ── Convertir a tensores ──────────────────────────────────────────
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).float()
    X_vl = torch.from_numpy(X_val).float()
    y_vl = torch.from_numpy(y_val).float()

    # ── Class weights para WeightedRandomSampler ─────────────────────
    n_0 = float((y_train == 0).sum())
    n_1 = float((y_train == 1).sum())
    n   = float(len(y_train))

    w_0 = n / (2.0 * n_0) if n_0 > 0 else 1.0
    w_1 = n / (2.0 * n_1) if n_1 > 0 else 1.0

    sample_weights = np.where(y_train == 1, w_1, w_0)
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(y_train),
        replacement = True,
    )

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_vl, y_vl)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 4, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader


# =============================================================================
# MÉTRICAS DE ENTRENAMIENTO
# =============================================================================

def _epoch_metrics(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    is_train:   bool = True,
) -> Dict[str, float]:
    """
    Ejecuta una época de train o val.
    Devuelve: loss, accuracy, directional_accuracy.
    """
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    n_correct   = 0
    n_dir_ok    = 0
    n_samples   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            pred_label  = (pred > 0.5).float()
            n_correct  += (pred_label == y_batch).sum().item()
            n_dir_ok   += (pred_label == y_batch).sum().item()
            n_samples  += bs

    return {
        "loss"     : total_loss / n_samples if n_samples > 0 else 0.0,
        "accuracy" : 100.0 * n_correct / n_samples if n_samples > 0 else 0.0,
        "dir_acc"  : 100.0 * n_dir_ok  / n_samples if n_samples > 0 else 0.0,
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

def train_model(
    model:        GRUTradingModel,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    X_val:        np.ndarray,
    y_val:        np.ndarray,
    fold_label:   str  = "Fold ?",
    batch_size:   int  = BATCH_SIZE,
    lr:           float = LEARNING_RATE,
    max_epochs:   int  = MAX_EPOCHS,
    patience:     int  = PATIENCE,
    pos_weight:   float = CLASS_WEIGHT_MINORITY,
    verbose:      bool  = True,
) -> Dict:
    """
    Entrena el modelo GRU con:
      - Pérdida asimétrica direccional
      - WeightedRandomSampler para desbalance de clases
      - Early stopping
      - ReduceLROnPlateau
      - Rich Live Display (tabla de épocas en tiempo real)

    Returns dict con historial y mejor val_loss.
    """
    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    criterion = AsymmetricDirectionalLoss(alpha=ALPHA_LOSS, pos_weight=pos_weight).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    stopper   = EarlyStopping(patience=patience)

    # ── Historial ────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "dir_acc":    [], "lr":       [],
    }

    best_epoch = 0
    epoch_rows = []   # para la tabla Rich (últimas N épocas)

    # ── Tabla Rich inicial ────────────────────────────────────────────
    def _make_table(rows: List) -> Table:
        tbl = Table(
            show_header=True, header_style="bold cyan",
            border_style="bright_black", expand=True,
        )
        tbl.add_column("Época",       style="dim",          width=6,  justify="right")
        tbl.add_column("Train Loss",  style="yellow",       width=12, justify="right")
        tbl.add_column("Val Loss",    style="bright_yellow",width=12, justify="right")
        tbl.add_column("Train Acc",   style="cyan",         width=11, justify="right")
        tbl.add_column("Val Acc",     style="bright_cyan",  width=10, justify="right")
        tbl.add_column("Dir Acc",     style="magenta",      width=10, justify="right")
        tbl.add_column("LR",          style="dim",          width=12, justify="right")
        tbl.add_column("Estado",      width=8,              justify="center")

        for r in rows[-20:]:   # mostrar últimas 20 épocas
            tbl.add_row(*r)
        return tbl

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    start_ts   = time.time()
    epoch_task = progress.add_task(f"[{fold_label}]", total=max_epochs)

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        for epoch in range(1, max_epochs + 1):
            t_m = _epoch_metrics(model, train_loader, criterion, optimizer, is_train=True)
            v_m = _epoch_metrics(model, val_loader,   criterion, optimizer=None, is_train=False)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(v_m["loss"])
            stopped    = stopper(v_m["loss"], model)
            is_best    = stopper.counter == 0

            if is_best:
                best_epoch = epoch

            # ── Guardar historial ─────────────────────────────────
            history["train_loss"].append(t_m["loss"])
            history["val_loss"].append(v_m["loss"])
            history["train_acc"].append(t_m["accuracy"])
            history["val_acc"].append(v_m["accuracy"])
            history["dir_acc"].append(v_m["dir_acc"])
            history["lr"].append(current_lr)

            # ── Fila para la tabla ─────────────────────────────────
            status_icon = "[green]✓ BEST[/green]" if is_best else (
                f"[red]{stopper.counter}/{patience}[/red]"
            )
            epoch_rows.append((
                str(epoch),
                f"{t_m['loss']:.5f}",
                f"{v_m['loss']:.5f}",
                f"{t_m['accuracy']:.1f}%",
                f"{v_m['accuracy']:.1f}%",
                f"{v_m['dir_acc']:.1f}%",
                f"{current_lr:.2e}",
                status_icon,
            ))

            # ── Actualizar Live ───────────────────────────────────
            progress.advance(epoch_task)
            tbl = _make_table(epoch_rows)
            live.update(Group(
                Panel(tbl, title=f"[bold white]🤖 {fold_label} — Entrenamiento GRU[/bold white]",
                      subtitle=f"[dim]Época {epoch}/{max_epochs} | Mejor: {best_epoch} | "
                               f"Val Loss: {stopper.best_loss:.5f}[/dim]",
                      border_style="blue"),
                progress,
            ))

            if stopped:
                console.print(
                    f"  [yellow]⚡ Early Stopping[/yellow] en época {epoch} "
                    f"(sin mejora {patience} épocas). Mejor: época {best_epoch}."
                )
                break

    # ── Restaurar mejor modelo ────────────────────────────────────────
    stopper.restore(model)

    elapsed = time.time() - start_ts
    return {
        "history"      : history,
        "best_epoch"   : best_epoch,
        "best_val_loss": stopper.best_loss,
        "elapsed_sec"  : elapsed,
        "n_epochs"     : epoch,
    }

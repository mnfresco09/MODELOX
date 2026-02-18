"""
================================================================================
IA/MAIN.PY — ORQUESTADOR PRINCIPAL DEL PIPELINE IA BTC
================================================================================

USO:
  python IA/main.py                  # Walk-forward completo
  python IA/main.py --quick          # Modo rápido (datos 2022-2025)
  python IA/main.py --optimize       # Walk-forward + Optuna
  python IA/main.py --folds 3        # Máximo 3 folds

PIPELINE:
  1. Cargar y preprocesar datos BTC 1m
  2. Computar features + etiquetas TP/SL $500
  3. Generar folds walk-forward
  4. (Opcional) Optimizar hiperparámetros con Optuna en fold 1
  5. Para cada fold:
     a. Entrenar modelo GRU (con rich live display)
     b. Generar señales con filtros de entropía y anomalía
     c. Ejecutar backtest TP/SL $500
     d. Calcular métricas: SQN, ROI, Drawdown, WR, Longs, Shorts...
     e. Mostrar resultados completos
  6. Resumen consolidado de todos los folds
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Añadir raíz del proyecto al path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


# =============================================================================
# COMPROBACIÓN DE DEPENDENCIAS
# =============================================================================

def _check_deps() -> bool:
    """Verifica que PyTorch está instalado."""
    try:
        import torch
        return True
    except ImportError:
        console.print(
            "[bold red]❌ PyTorch no está instalado.[/bold red]\n"
            "[yellow]Instalar con:[/yellow] [cyan]pip install torch[/cyan]"
        )
        return False


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(
    optimize:   bool  = False,
    quick:      bool  = False,
    max_folds:  int   = 10,
    date_start: Optional[str] = None,
    date_end:   Optional[str] = None,
) -> None:
    """Pipeline completo de IA para BTC trading."""

    t0 = time.time()

    # ── Aplicar modo rápido ────────────────────────────────────────────
    import IA.config as cfg
    if quick:
        cfg.QUICK_MODE   = True
        date_start       = date_start or cfg.QUICK_DATE_START
        date_end         = date_end   or cfg.QUICK_DATE_END
        cfg.MAX_EPOCHS   = cfg.QUICK_MAX_EPOCHS
        cfg.STRIDE       = cfg.QUICK_STRIDE
        console.print("[yellow]⚡ MODO RÁPIDO activado[/yellow]")

    # ── Imports principales ────────────────────────────────────────────
    from IA.display import (
        print_banner, print_config, print_data_summary,
        print_folds_table, print_fold_result, print_trades_table,
        print_walkforward_summary, print_model_summary, print_optuna_result,
        console as disp_console,
    )
    from IA.data_pipeline import prepare_data, data_summary
    from IA.walk_forward   import generate_folds, fold_summary
    from IA.model          import build_model, model_summary, DEVICE
    from IA.trainer        import train_model
    from IA.signals        import predict_batch, generate_signals, signals_summary
    from IA.backtest       import run_backtest, compute_backtest_metrics, trades_to_dataframe

    # ──────────────────────────────────────────────────────────────────
    # 1. BANNER + CONFIG
    # ──────────────────────────────────────────────────────────────────
    print_banner()
    print_config(cfg)

    # ──────────────────────────────────────────────────────────────────
    # 2. CARGAR Y PREPROCESAR DATOS
    # ──────────────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]📂 CARGANDO DATOS BTC 1M[/bold cyan]"))
    console.print()

    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as prog:
        t_load = prog.add_task("Cargando feather + preprocesando...", total=None)
        feat_df, labels, raw_df = prepare_data(date_start, date_end, verbose=False)
        prog.stop()

    d_sum = data_summary(feat_df, labels)
    print_data_summary(d_sum)
    console.print(f"  [dim]Dispositivo PyTorch: [bold]{DEVICE}[/bold][/dim]")
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # 3. ETIQUETADO (info)
    # ──────────────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]🏷  ETIQUETADO TP/SL 0.65%[/bold cyan]"))
    valid_labels = labels[labels >= 0]
    n_long  = (valid_labels == 1).sum()
    n_short = (valid_labels == 0).sum()
    pct_long = 100.0 * n_long / max(len(valid_labels), 1)
    console.print(
        f"  Labels válidos: [bold]{len(valid_labels):,}[/bold] | "
        f"LONG: [green]{n_long:,} ({pct_long:.1f}%)[/green] | "
        f"SHORT: [red]{n_short:,} ({100-pct_long:.1f}%)[/red]"
    )
    console.print(
        f"  TP = precio ± [bold yellow]{cfg.TP_PCT:.2f}%[/bold yellow] | "
        f"SL = precio ∓ [bold red]{cfg.SL_PCT:.2f}%[/bold red] | "
        f"Max lookforward: [dim]{cfg.MAX_FORWARD_CANDLES} velas[/dim]"
    )
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # 4. GENERAR FOLDS WALK-FORWARD
    # ──────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]📅 GENERANDO FOLDS WALK-FORWARD[/bold yellow]"))
    console.print()

    with Progress(
        SpinnerColumn(), TextColumn("[bold yellow]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as prog:
        t_folds = prog.add_task("Generando folds...", total=None)
        folds   = generate_folds(feat_df, labels, max_folds=max_folds)
        prog.stop()

    if not folds:
        console.print("[bold red]❌ No se pudieron generar folds suficientes.[/bold red]")
        console.print("[yellow]  Prueba un rango de fechas mayor o reduce TRAIN_YEARS.[/yellow]")
        return

    fold_summaries = [fold_summary(f) for f in folds]
    print_folds_table(fold_summaries)
    console.print(f"  Total folds generados: [bold yellow]{len(folds)}[/bold yellow]")
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # 5. MODELO BASE
    # ──────────────────────────────────────────────────────────────────
    n_features = folds[0].X_train.shape[2] if folds[0].X_train is not None else cfg.N_FEATURES
    m_sum      = model_summary(build_model(n_features=n_features))
    print_model_summary(m_sum)
    console.print()

    # ──────────────────────────────────────────────────────────────────
    # 6. (OPCIONAL) OPTUNA EN FOLD 1
    # ──────────────────────────────────────────────────────────────────
    best_params = {}
    if optimize:
        console.print(Rule("[bold magenta]🔮 OPTIMIZACIÓN BAYESIANA (FOLD 1)[/bold magenta]"))
        from IA.optuna_search import optimize_hyperparams
        best_params = optimize_hyperparams(folds[0], verbose=True)
        print_optuna_result(
            {k: v for k, v in best_params.items() if k != "best_val_loss"},
            best_params.get("best_val_loss", 0.0),
        )
        console.print()

    # ──────────────────────────────────────────────────────────────────
    # 7. WALK-FORWARD: ENTRENAMIENTO + BACKTEST POR FOLD
    # ──────────────────────────────────────────────────────────────────
    console.print(Rule("[bold green]🚀 WALK-FORWARD: ENTRENAMIENTO Y BACKTEST[/bold green]"))
    console.print()

    all_fold_metrics : List[dict]       = []
    all_trades       : List            = []
    all_equity_curves: List[np.ndarray] = []

    close_arr = raw_df["close"].values.astype(np.float64)

    for fold in folds:
        fold_n = fold.fold_n

        console.print(Rule(
            f"[bold blue]FOLD {fold_n}/{len(folds)} | "
            f"Train: {fold.train_start.date()} → {fold.train_end.date()} | "
            f"Val: {fold.val_start.date()} → {fold.val_end.date()}[/bold blue]"
        ))

        fs = fold_summaries[fold_n - 1]
        console.print(
            f"  [dim]Secuencias train: [bold]{fs['n_train']:,}[/bold] | "
            f"val: [bold]{fs['n_val']:,}[/bold][/dim]"
        )
        console.print()

        # ── Hiperparámetros (base o de Optuna) ────────────────────────
        gru_units  = int(best_params.get("gru_units",  cfg.GRU_UNITS))
        batch_size = int(best_params.get("batch_size", cfg.BATCH_SIZE))
        lr_val     = float(best_params.get("lr",       cfg.LEARNING_RATE))
        dropout    = float(best_params.get("dropout",  cfg.DROPOUT))

        # ── Construir modelo fresco para cada fold ────────────────────
        model = build_model(
            n_features = n_features,
            gru_units  = gru_units,
            n_layers   = cfg.N_GRU_LAYERS,
            dropout    = dropout,
        )

        # ── Calcular class weight del fold ────────────────────────────
        n_1   = float((fold.y_train == 1).sum())
        n_0   = float((fold.y_train == 0).sum())
        n_tot = float(len(fold.y_train))
        pw    = (n_tot / (2.0 * n_1)) * cfg.CLASS_WEIGHT_MINORITY if n_1 > 0 else cfg.CLASS_WEIGHT_MINORITY

        # ── Entrenar ──────────────────────────────────────────────────
        train_result = train_model(
            model      = model,
            X_train    = fold.X_train,
            y_train    = fold.y_train,
            X_val      = fold.X_val,
            y_val      = fold.y_val,
            fold_label = f"Fold {fold_n}/{len(folds)}",
            batch_size = batch_size,
            lr         = lr_val,
            max_epochs = cfg.MAX_EPOCHS,
            patience   = cfg.PATIENCE,
            pos_weight = pw,
        )

        console.print(
            f"  [dim]✓ Entrenamiento completado | "
            f"Mejor época: [bold]{train_result['best_epoch']}[/bold] | "
            f"Val Loss: [bold]{train_result['best_val_loss']:.5f}[/bold] | "
            f"Tiempo: [bold]{train_result['elapsed_sec']:.1f}s[/bold][/dim]"
        )
        console.print()

        # ── Generar predicciones en validación ────────────────────────
        console.print("  [cyan]Generando predicciones en validación...[/cyan]")
        probs = predict_batch(model, fold.X_val, batch_size=512)

        # ── Generar señales con filtros ───────────────────────────────
        # Obtener precios de cierre para el período de validación
        val_feat_slice = feat_df.loc[
            (feat_df.index >= fold.val_start) & (feat_df.index < fold.val_end)
        ]
        val_close = val_feat_slice["close_raw"].values if "close_raw" in val_feat_slice.columns else close_arr

        # Índices en feat_df (globales) de las muestras de validación
        val_global_indices = fold.val_indices  # índices relativos al subset de val

        signals = generate_signals(
            probs           = probs,
            close_prices    = close_arr,
            indices         = val_global_indices,
            prob_threshold  = cfg.PROB_THRESHOLD,
            short_threshold = cfg.SHORT_THRESHOLD,
            use_entropy     = True,
            use_anomaly     = True,
        )

        sig_sum = signals_summary(signals)
        console.print(
            f"  [dim]Señales generadas | "
            f"LONG: [green]{sig_sum['n_long']}[/green] | "
            f"SHORT: [red]{sig_sum['n_short']}[/red] | "
            f"Sin señal: {sig_sum['n_none']} | "
            f"(P>{cfg.PROB_THRESHOLD:.0%} o P<{cfg.SHORT_THRESHOLD:.0%})[/dim]"
        )
        console.print()

        # ── Alinear índices de señales con feat_df ────────────────────
        # Los signal_indices del fold son relativos al val subset
        # Necesitamos mapearlos a índices en feat_df global
        # Para el backtest, necesitamos timestamps en feat_df

        # Crear sub-feat_df para este período de val
        val_feat_df = feat_df.loc[
            (feat_df.index >= fold.val_start) & (feat_df.index < fold.val_end)
        ]

        # Los val_indices son posiciones dentro del val subset de feat_df
        # que fueron usadas para crear secuencias (con stride)
        # Ajustar para que sean posiciones absolutas en val_feat_df
        signal_abs_indices = np.array([
            min(int(idx), len(val_feat_df) - 1)
            for idx in fold.val_indices
        ], dtype=np.int64)

        # ── Ejecutar backtest ─────────────────────────────────────────
        console.print(f"  [cyan]Ejecutando backtest TP/SL {cfg.TP_PCT:.2f}%...[/cyan]")

        # Construir feat_df alineado para el backtest usando val_feat_df
        # signal_abs_indices son posiciones en val_feat_df
        trades, equity_curve = run_backtest(
            raw_df         = raw_df.loc[
                (raw_df.index >= fold.val_start) & (raw_df.index < fold.val_end)
            ],
            feat_df        = val_feat_df,
            signals        = signals,
            signal_indices = signal_abs_indices,
            saldo_inicial  = cfg.SALDO_INICIAL,
            apalancamiento = cfg.APALANCAMIENTO,
            saldo_usado    = cfg.SALDO_USADO,
            comision_pct   = cfg.COMISION_PCT,
            comision_sides = cfg.COMISION_SIDES,
            tp_pct         = cfg.TP_PCT,
            sl_pct         = cfg.SL_PCT,
            max_forward    = cfg.MAX_FORWARD_CANDLES,
        )

        console.print(
            f"  [dim]✓ Backtest completado | "
            f"Trades ejecutados: [bold]{len(trades)}[/bold][/dim]"
        )
        console.print()

        # ── Métricas financieras ──────────────────────────────────────
        metrics = compute_backtest_metrics(trades, equity_curve, cfg.SALDO_INICIAL)

        all_fold_metrics.append(metrics)
        all_trades.extend(trades)
        all_equity_curves.append(equity_curve)

        # ── Mostrar resultados del fold ───────────────────────────────
        print_fold_result(fold_n, metrics, equity_curve)
        print_trades_table(trades, max_rows=25)
        console.print()

    # ──────────────────────────────────────────────────────────────────
    # 8. RESUMEN CONSOLIDADO
    # ──────────────────────────────────────────────────────────────────
    if all_fold_metrics:
        print_walkforward_summary(
            fold_metrics   = all_fold_metrics,
            fold_summaries = fold_summaries[:len(all_fold_metrics)],
            all_trades     = all_trades,
            equity_curves  = all_equity_curves,
            saldo_inicial  = cfg.SALDO_INICIAL,
        )

    # ──────────────────────────────────────────────────────────────────
    # 9. GUARDAR RESULTADOS
    # ──────────────────────────────────────────────────────────────────
    if all_trades:
        import json
        from IA.backtest import trades_to_dataframe
        trades_df = trades_to_dataframe(all_trades)
        output_csv = cfg.RESULTS_DIR / "trades_walkforward.csv"
        trades_df.to_csv(output_csv, index=False)
        console.print(f"\n  [dim]📁 Trades guardados en: [bold]{output_csv}[/bold][/dim]")

        # Guardar métricas por fold
        metrics_path = cfg.RESULTS_DIR / "metricas_walkforward.json"
        with open(metrics_path, "w") as f:
            import math
            def _clean(obj):
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                return obj
            clean_metrics = [{k: _clean(v) for k, v in m.items()} for m in all_fold_metrics]
            json.dump(clean_metrics, f, indent=2)
        console.print(f"  [dim]📁 Métricas guardadas en: [bold]{metrics_path}[/bold][/dim]")

    elapsed = time.time() - t0
    console.print()
    console.print(Rule(f"[bold green]✅ Pipeline completado en {elapsed:.1f}s[/bold green]"))
    console.print()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="🤖 GRU Trading AI — Pipeline completo BTC 1M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python IA/main.py                    # Pipeline completo
  python IA/main.py --quick            # Modo rápido (2022-2025, pocas épocas)
  python IA/main.py --optimize         # Con optimización Optuna
  python IA/main.py --quick --optimize # Rápido + Optuna
  python IA/main.py --folds 3          # Solo 3 folds
  python IA/main.py --start 2021-01-01 --end 2024-01-01  # Rango personalizado
        """,
    )
    parser.add_argument("--quick",    action="store_true", help="Modo rápido (datos reducidos)")
    parser.add_argument("--optimize", action="store_true", help="Activar optimización Optuna")
    parser.add_argument("--folds",    type=int, default=10, metavar="N", help="Máximo de folds")
    parser.add_argument("--start",    type=str, default=None, metavar="YYYY-MM-DD", help="Fecha inicio datos")
    parser.add_argument("--end",      type=str, default=None, metavar="YYYY-MM-DD", help="Fecha fin datos")
    args = parser.parse_args()

    if not _check_deps():
        sys.exit(1)

    run_pipeline(
        optimize   = args.optimize,
        quick      = args.quick,
        max_folds  = args.folds,
        date_start = args.start,
        date_end   = args.end,
    )


if __name__ == "__main__":
    main()

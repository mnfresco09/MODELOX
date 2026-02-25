"""
================================================================================
VISUAL/MONTECARLO_DB.PY — MONTE CARLO DE EQUITY CURVES DESDE DB OPTUNA
================================================================================

PROPÓSITO:
    Lee la base de datos SQLite de Optuna y grafica las curvas de equity
    de todos los trials completados en estilo Monte Carlo.

    Cada curva representa un trial de optimización. Se colorean de rojo
    (score bajo) a verde (score alto). La mejor curva se resalta en dorado,
    la mediana en azul claro y la peor en rojo semitransparente.

USO:
    # Opción A — arrastrar el .db al comando:
    python visual/montecarlo.py DATABASE/ID2.db

    # Opción B — ejecutar sin argumentos y arrastrar cuando lo pide:
    python visual/montecarlo.py

REQUISITOS:
    - optuna
    - plotly (recomendado, genera HTML interactivo) o matplotlib (PNG)
    - GUARDAR_EQUITY_EN_DB = True en general/configuracion.py
      (es necesario haber ejecutado al menos una optimización con esa opción)

================================================================================
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple


# =============================================================================
# 1. SELECCIÓN DE ARCHIVO
# =============================================================================

def _ask_db_path() -> Optional[str]:
    """
    Pide la ruta del .db por terminal.
    En macOS/Linux puedes arrastrar el archivo directamente sobre
    la ventana del terminal para autocompletar la ruta, y luego Enter.
    """
    print()
    print("  Arrastra el archivo .db sobre esta ventana y pulsa Enter")
    print("  (o escribe/pega la ruta completa):")
    print()
    try:
        raw = input("  → ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    # macOS añade comillas simples a rutas con espacios al arrastrar
    path = raw.strip("'\"").strip()
    return path if path else None


# =============================================================================
# 2. LECTURA DE DATOS DESDE OPTUNA
# =============================================================================

def _load_equity_curves(
    db_path: str,
) -> Tuple[List[List[float]], List[float], List[int], str, float]:
    """
    Carga las equity curves de todos los trials completados en la DB.

    Returns
    -------
    curves        : Lista de listas de floats (una por trial)
    scores        : Score Optuna de cada trial
    trial_numbers : Número de trial de cada curva
    study_name    : Nombre del estudio dentro de la DB
    saldo_inicial : Capital inicial detectado (primer valor de la 1ª curva)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    storage = f"sqlite:///{db_path}"

    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        raise ValueError(f"No se encontraron estudios en: {db_path}")

    # Usar el estudio con más trials completados
    summaries.sort(key=lambda s: s.n_trials, reverse=True)
    study_name = summaries[0].study_name

    study = optuna.load_study(study_name=study_name, storage=storage)

    curves: List[List[float]] = []
    scores: List[float] = []
    trial_numbers: List[int] = []

    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        equity = trial.user_attrs.get("equity_curve")
        if not equity or len(equity) < 2:
            continue
        curves.append([float(x) for x in equity])
        scores.append(float(trial.value) if trial.value is not None else 0.0)
        trial_numbers.append(trial.number)

    saldo_inicial = curves[0][0] if curves else 1000.0
    return curves, scores, trial_numbers, study_name, saldo_inicial


# =============================================================================
# 3. HELPERS
# =============================================================================

def _score_to_rgb(score: float, min_s: float, max_s: float) -> str:
    """Mapea score → color RGB: rojo (bajo) → amarillo (medio) → verde (alto)."""
    t = (score - min_s) / (max_s - min_s + 1e-12)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        r = 220
        g = int(220 * t * 2)
        b = 40
    else:
        r = int(220 * (1.0 - t) * 2)
        g = 200
        b = 40
    return f"rgb({r},{g},{b})"


def _normalize_curves(
    curves: List[List[float]], n_points: int = 200
) -> "list[list[float]]":
    """Interpola cada curva a n_points para que sean comparables."""
    import numpy as np

    result = []
    for c in curves:
        if len(c) < 2:
            result.append(c)
            continue
        x_orig = np.linspace(0.0, 100.0, len(c))
        x_new = np.linspace(0.0, 100.0, n_points)
        result.append(np.interp(x_new, x_orig, c).tolist())
    return result


# =============================================================================
# 4. PLOTLY (HTML interactivo)
# =============================================================================

def _plot_plotly(
    curves: List[List[float]],
    scores: List[float],
    trial_numbers: List[int],
    study_name: str,
    saldo_inicial: float,
    output_path: str,
    max_curves: int = 600,
) -> None:
    import numpy as np
    import plotly.graph_objects as go

    n_orig = len(curves)

    # Subsamplear preservando siempre el mejor y el peor
    if n_orig > max_curves:
        arr_scores = list(enumerate(scores))
        arr_scores.sort(key=lambda x: x[1])
        idx_worst_global = arr_scores[0][0]
        idx_best_global = arr_scores[-1][0]

        middle = [i for i in range(n_orig) if i not in (idx_worst_global, idx_best_global)]
        rng = np.random.default_rng(42)
        sampled = rng.choice(middle, size=min(max_curves - 2, len(middle)), replace=False).tolist()
        keep = [idx_worst_global, idx_best_global] + sampled

        curves = [curves[i] for i in keep]
        scores = [scores[i] for i in keep]
        trial_numbers = [trial_numbers[i] for i in keep]

    n = len(curves)
    min_s, max_s = min(scores), max(scores)

    N_POINTS = 200
    norm = _normalize_curves(curves, N_POINTS)
    x_axis = list(range(N_POINTS))  # índice 0-199 (trade interpolado)

    arr = np.array(norm)
    median_curve = np.median(arr, axis=0).tolist()
    p10 = np.percentile(arr, 10, axis=0).tolist()
    p90 = np.percentile(arr, 90, axis=0).tolist()

    idx_best = int(np.argmax(scores))
    idx_worst = int(np.argmin(scores))

    rois = [(c[-1] / saldo_inicial - 1) * 100 for c in norm]
    n_positive = sum(1 for r in rois if r > 0)
    survival_pct = n_positive / n * 100 if n else 0.0

    best_roi = rois[idx_best]
    worst_roi = rois[idx_worst]
    median_roi = (median_curve[-1] / saldo_inicial - 1) * 100

    fig = go.Figure()

    # ── Curvas individuales ──
    for i, (curve, score, tnum) in enumerate(zip(norm, scores, trial_numbers)):
        if i in (idx_best, idx_worst):
            continue  # Se dibujan aparte con más detalle
        color = _score_to_rgb(score, min_s, max_s)
        roi_i = (curve[-1] / saldo_inicial - 1) * 100
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=curve,
            mode="lines",
            line=dict(color=color, width=0.7),
            opacity=0.13,
            hovertemplate=(
                f"Trial #{tnum}<br>"
                f"Score: {score:.4f}<br>"
                f"ROI final: {roi_i:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # ── Banda P10-P90 ──
    fig.add_trace(go.Scatter(
        x=x_axis + x_axis[::-1],
        y=p90 + p10[::-1],
        fill="toself",
        fillcolor="rgba(100, 160, 255, 0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Banda P10–P90",
        hoverinfo="skip",
    ))

    # ── Curva Peor ──
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=norm[idx_worst],
        mode="lines",
        line=dict(color="rgba(255, 80, 70, 0.85)", width=1.8, dash="dot"),
        name=f"Peor — Trial #{trial_numbers[idx_worst]} (ROI {worst_roi:+.1f}%)",
        opacity=0.9,
    ))

    # ── Mediana ──
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=median_curve,
        mode="lines",
        line=dict(color="rgba(160, 180, 255, 0.95)", width=2.2),
        name=f"Mediana (ROI {median_roi:+.1f}%)",
    ))

    # ── Curva Mejor ──
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=norm[idx_best],
        mode="lines",
        line=dict(color="rgba(255, 215, 0, 1.0)", width=2.5),
        name=f"Mejor — Trial #{trial_numbers[idx_best]} (ROI {best_roi:+.1f}%)",
    ))

    # ── Línea de capital inicial ──
    fig.add_hline(
        y=saldo_inicial,
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
        annotation_text=f"Capital inicial ${saldo_inicial:,.0f}",
        annotation_font_color="rgba(255,255,255,0.45)",
        annotation_position="bottom right",
    )

    # ── Estadísticas en el título ──
    stats = (
        f"{n:,} trials   |   "
        f"Supervivencia: {survival_pct:.1f}%   |   "
        f"Mejor ROI: {best_roi:+.1f}%   |   "
        f"Mediana ROI: {median_roi:+.1f}%   |   "
        f"Peor ROI: {worst_roi:+.1f}%"
    )
    if n_orig > max_curves:
        stats += f"   |   (mostrando {n:,} de {n_orig:,})"

    fig.update_layout(
        title=dict(
            text=f"<b>Monte Carlo Equity — {study_name}</b><br><sub>{stats}</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="white"),
        ),
        paper_bgcolor="rgb(13, 13, 22)",
        plot_bgcolor="rgb(18, 18, 32)",
        font=dict(color="rgba(210,210,230,0.9)", family="monospace, sans-serif"),
        xaxis=dict(
            title="Trade interpolado (índice normalizado)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickcolor="rgba(200,200,220,0.4)",
        ),
        yaxis=dict(
            title="Equity ($)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickprefix="$",
            tickformat=",.0f",
            tickcolor="rgba(200,200,220,0.4)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
        ),
        hovermode="x unified",
        margin=dict(l=80, r=50, t=110, b=60),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"✓ Gráfico guardado: {output_path}")

    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


# =============================================================================
# 5. MATPLOTLIB (PNG fallback)
# =============================================================================

def _plot_matplotlib(
    curves: List[List[float]],
    scores: List[float],
    trial_numbers: List[int],
    study_name: str,
    saldo_inicial: float,
    output_path: str,
    max_curves: int = 400,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_orig = len(curves)

    if n_orig > max_curves:
        rng = np.random.default_rng(42)
        keep = rng.choice(n_orig, size=max_curves, replace=False).tolist()
        curves = [curves[i] for i in keep]
        scores = [scores[i] for i in keep]
        trial_numbers = [trial_numbers[i] for i in keep]

    n = len(curves)
    min_s, max_s = min(scores), max(scores)

    N_POINTS = 200
    norm = _normalize_curves(curves, N_POINTS)
    x_axis = list(range(N_POINTS))

    arr = np.array(norm)
    median_curve = np.median(arr, axis=0)
    idx_best = int(np.argmax(scores))
    idx_worst = int(np.argmin(scores))

    rois = [(c[-1] / saldo_inicial - 1) * 100 for c in norm]
    n_positive = sum(1 for r in rois if r > 0)
    survival_pct = n_positive / n * 100 if n else 0.0

    cmap = cm.RdYlGn

    fig, ax = plt.subplots(figsize=(15, 7), facecolor="#0d0d16")
    ax.set_facecolor("#121220")

    for i, (curve, score) in enumerate(zip(norm, scores)):
        if i in (idx_best, idx_worst):
            continue
        t = (score - min_s) / (max_s - min_s + 1e-12)
        ax.plot(x_axis, curve, color=cmap(t), alpha=0.09, linewidth=0.6)

    worst_roi = rois[idx_worst]
    ax.plot(
        x_axis, norm[idx_worst],
        color="#ff5050", alpha=0.85, linewidth=1.8, linestyle="--",
        label=f"Peor — Trial #{trial_numbers[idx_worst]} (ROI {worst_roi:+.1f}%)",
    )

    median_roi = (median_curve[-1] / saldo_inicial - 1) * 100
    ax.plot(
        x_axis, median_curve,
        color="#a0b4ff", alpha=0.92, linewidth=2.3,
        label=f"Mediana (ROI {median_roi:+.1f}%)",
    )

    best_roi = rois[idx_best]
    ax.plot(
        x_axis, norm[idx_best],
        color="#ffd700", alpha=1.0, linewidth=2.5,
        label=f"Mejor — Trial #{trial_numbers[idx_best]} (ROI {best_roi:+.1f}%)",
    )

    ax.axhline(saldo_inicial, color="white", alpha=0.2, linewidth=1, linestyle="--",
               label=f"Capital inicial ${saldo_inicial:,.0f}")

    title = (
        f"Monte Carlo Equity — {study_name}\n"
        f"{n:,} trials   |   Supervivencia: {survival_pct:.1f}%   |   "
        f"Mejor: {best_roi:+.1f}%   |   Mediana: {median_roi:+.1f}%"
    )
    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.set_xlabel("Trade interpolado (índice normalizado)", color="#aaaacc", fontsize=10)
    ax.set_ylabel("Equity ($)", color="#aaaacc", fontsize=10)
    ax.tick_params(colors="#aaaacc")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#333355")
    ax.grid(True, color="white", alpha=0.04)
    ax.legend(
        facecolor="#1a1a2e", edgecolor="#333355",
        labelcolor="white", fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Gráfico guardado: {output_path}")
    plt.show()


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    # ── Obtener ruta de la DB ──
    # Opción 1: arrastrar el .db directamente al comando en terminal
    #   python visual/montecarlo.py DATABASE/ID2.db
    # Opción 2: ejecutar sin argumentos y arrastrar cuando lo pide
    if len(sys.argv) > 1:
        db_path = os.path.abspath(sys.argv[1].strip("'\""))
    else:
        db_path = _ask_db_path()

    if not db_path:
        print("No se seleccionó ningún archivo. Saliendo.")
        sys.exit(0)

    if not os.path.exists(db_path):
        print(f"Error: el archivo no existe → {db_path}")
        sys.exit(1)

    print(f"\n  Base de datos : {db_path}")

    # ── Cargar datos ──
    try:
        curves, scores, trial_numbers, study_name, saldo_inicial = _load_equity_curves(db_path)
    except Exception as exc:
        print(f"Error al leer la base de datos: {exc}")
        sys.exit(1)

    if not curves:
        print(
            "\n  No se encontraron curvas de equity.\n"
            "  Asegúrate de tener GUARDAR_EQUITY_EN_DB = True en general/configuracion.py\n"
            "  y vuelve a ejecutar una optimización."
        )
        sys.exit(0)

    print(f"  Estudio       : {study_name}")
    print(f"  Trials válidos: {len(curves):,}")
    print(f"  Capital inicial detectado: ${saldo_inicial:,.2f}")

    # ── Directorio de salida (junto al .db) ──
    out_dir = os.path.dirname(db_path)
    base_name = os.path.splitext(os.path.basename(db_path))[0]

    # ── Renderizar ──
    try:
        import plotly  # noqa: F401
        output_path = os.path.join(out_dir, f"montecarlo_{base_name}.html")
        _plot_plotly(curves, scores, trial_numbers, study_name, saldo_inicial, output_path)
    except ImportError:
        try:
            import matplotlib  # noqa: F401
            output_path = os.path.join(out_dir, f"montecarlo_{base_name}.png")
            _plot_matplotlib(curves, scores, trial_numbers, study_name, saldo_inicial, output_path)
        except ImportError:
            print(
                "\n  Error: se necesita plotly o matplotlib.\n"
                "  Instala con:  pip install plotly\n"
                "            o:  pip install matplotlib"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()

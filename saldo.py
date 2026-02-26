from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import statistics
import pandas as pd

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from visual.rich import (
        mostrar_cabecera_inicio,
        mostrar_panel_elegante,
        resetear_estadisticas,
        actualizar_estadisticas,
    )
    _RICH_OK = True
except Exception:
    _RICH_OK = False

from general.configuracion import (
    ACTIVO_PRIMARIO,
    COMBINACION_A_EJECUTAR,
    CONFIG,
    FECHA_FIN,
    FECHA_INICIO,
    OPTUNA_SAMPLER,
    OPTUNA_SEED,
    TIMEFRAME,
    resolve_archivo_data,
    resolve_archivo_data_tf,
)
from modelox.core.data import load_data, resample_to_base_timeframe
from modelox.core.runner import OptimizationRunner, OptunaConfig
from modelox.core.types import BacktestConfig, filter_by_date, normalize_timeframe_to_suffix
from modelox.strategies.registry import instantiate_strategies


# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

SALDOS_BARRIDO          = list(range(80, 110, 10))
APALANCAMIENTOS_BARRIDO = list(range(10, 25, 5))
MODO_SALIDA             = "TRAILING"          # "FIXED" | "TRAILING" | "AMBOS"
OPTIMIZAR_SALIDAS_EN_OPT = True
RICH_MODO_VISUAL = "TRIALS"  # "TRIALS" (como ejecutar.py) | "BARRA"

FIXED_SL_PCT            = 17.0
FIXED_TP_PCT            = 17.0
FIXED_SL_PCT_RANGE      = (30.0, 45.0, 5.0)
FIXED_TP_PCT_RANGE      = (35.0, 55.0, 5.0)

TRAILING_SL_PCT         = 17.0
TRAILING_TP_ACT_PCT     = 15.0
TRAILING_DISTANCE_PCT   = 3.0
TRAILING_SL_PCT_RANGE   = (20.0,  45.0, 5.0)
TRAILING_TP_ACT_PCT_RANGE = (25.0, 45.0, 5.0)
TRAILING_DISTANCE_PCT_RANGE = (2.5, 10.0, 2.5)


# ══════════════════════════════════════════════════════════════════
#  UTILIDADES BÁSICAS
# ════════════════════════════

def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _finite_series(s: pd.Series) -> pd.Series:
    """pd.isin no existe como función de módulo; usar s.isin()."""
    s = pd.to_numeric(s, errors="coerce")
    return s[~s.isna() & ~s.isin([float("inf"), float("-inf")])]


def _range_values(rng: Tuple[float, float, float]) -> List[float]:
    lo, hi, step = float(rng[0]), float(rng[1]), float(rng[2])
    if step <= 0:
        return [round(lo, 10)]
    vals, cur = [], lo
    while cur <= hi + step * 1e-9:
        vals.append(round(cur, 10))
        cur += step
    return vals


def _axis_bounds(series: pd.Series, pad: float = 0.08) -> Tuple[float, float]:
    s = _finite_series(series)
    if s.empty:
        return 0.0, 1.0
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-9:
        p = max(abs(mx) * pad, 1.0)
        return mn - p, mx + p
    p = (mx - mn) * pad
    return mn - p, mx + p


def _iso_line(x_min: float, x_max: float, k: float, n: int = 60) -> Tuple[List[float], List[float]]:
    xs = [x_min + (x_max - x_min) * i / (n - 1) for i in range(n)]
    return xs, [k * x for x in xs]


# ══════════════════════════════════════════════════════════════════
#  EXIT COMBOS
# ══════════════════════════════════════════════════════════════════

def _resolve_exit_setup() -> Dict[str, Any]:
    modo = str(MODO_SALIDA).strip().upper()
    if modo == "TRAILING":
        return {
            "exit_type": "pnl_trailing",
            "exit_sl_pct": float(TRAILING_SL_PCT),
            "exit_tp_pct": 0.0,
            "exit_trail_act_pct": float(TRAILING_TP_ACT_PCT),
            "exit_trail_dist_pct": float(TRAILING_DISTANCE_PCT),
            "exit_sl_pct_range": tuple(TRAILING_SL_PCT_RANGE),
            "exit_tp_pct_range": (0.0, 0.0, 1.0),
            "exit_trail_act_pct_range": tuple(TRAILING_TP_ACT_PCT_RANGE),
            "exit_trail_dist_pct_range": tuple(TRAILING_DISTANCE_PCT_RANGE),
        }
    return {
        "exit_type": "pnl_fixed",
        "exit_sl_pct": float(FIXED_SL_PCT),
        "exit_tp_pct": float(FIXED_TP_PCT),
        "exit_trail_act_pct": float(CONFIG["EXIT_TRAIL_ACT_PCT"]),
        "exit_trail_dist_pct": float(CONFIG["EXIT_TRAIL_DIST_PCT"]),
        "exit_sl_pct_range": tuple(FIXED_SL_PCT_RANGE),
        "exit_tp_pct_range": tuple(FIXED_TP_PCT_RANGE),
        "exit_trail_act_pct_range": (0.0, 0.0, 1.0),
        "exit_trail_dist_pct_range": (0.0, 0.0, 1.0),
    }


EXIT_SETUP = _resolve_exit_setup()


def _build_exit_combinations() -> List[Dict[str, float]]:
    modo = str(MODO_SALIDA).strip().upper()
    combos: List[Dict[str, float]] = []
    if modo in {"FIXED", "AMBOS", "BOTH"}:
        for sl, tp in product(_range_values(FIXED_SL_PCT_RANGE), _range_values(FIXED_TP_PCT_RANGE)):
            combos.append({
                "exit_type": "pnl_fixed",
                "exit_sl_pct": float(sl),
                "exit_tp_pct": float(tp),
                "exit_trail_act_pct": float(CONFIG["EXIT_TRAIL_ACT_PCT"]),
                "exit_trail_dist_pct": float(CONFIG["EXIT_TRAIL_DIST_PCT"]),
            })
    if modo in {"TRAILING", "AMBOS", "BOTH"}:
        for sl, act, dist in product(
            _range_values(TRAILING_SL_PCT_RANGE),
            _range_values(TRAILING_TP_ACT_PCT_RANGE),
            _range_values(TRAILING_DISTANCE_PCT_RANGE),
        ):
            combos.append({
                "exit_type": "pnl_trailing",
                "exit_sl_pct": float(sl),
                "exit_tp_pct": 0.0,
                "exit_trail_act_pct": float(act),
                "exit_trail_dist_pct": float(dist),
            })
    return combos


def _fmt_exit_short(exit_type, sl, tp, act, dist) -> str:
    if "trailing" in str(exit_type).lower():
        return f"TRAILING | SL={_safe_float(sl):.2f} | ACT={_safe_float(act):.2f} | DIST={_safe_float(dist):.2f}"
    return f"FIXED | SL={_safe_float(sl):.2f} | TP={_safe_float(tp):.2f}"


def _render_trial_flow_rich(
    *,
    trial_num: int,
    row: Dict[str, Any],
    saldo_inicial: float,
    activo: str,
    best_so_far: Optional[float] = None,
) -> None:
    if not _RICH_OK:
        return

    metrics = dict(row.get("__metrics", {}) or {})
    if not metrics:
        metrics = {
            "roi": _safe_float(row.get("ROI")),
            "drawdown": _safe_float(row.get("DRAWDOWN")),
        }
    score = _safe_float(row.get("__score", row.get("ROI")))
    params = {
        "exit_type": row.get("EXIT_TYPE", ""),
        "exit_sl_pct": _safe_float(row.get("EXIT_SL_PCT")),
        "exit_tp_pct": _safe_float(row.get("EXIT_TP_PCT")),
        "exit_trail_act_pct": _safe_float(row.get("EXIT_TRAIL_ACT_PCT")),
        "exit_trail_dist_pct": _safe_float(row.get("EXIT_TRAIL_DIST_PCT")),
        "SALDO_USADO": _safe_float(row.get("SALDO USADO")),
        "APALANCAMIENTO": _safe_float(row.get("APALANCAMIENTO")),
    }

    actualizar_estadisticas(metrics, score, trial_num)
    mostrar_panel_elegante(
        metrics=metrics,
        params=params,
        score=score,
        trial_num=trial_num,
        saldo_inicial=saldo_inicial,
        indicadores_activos=["SALDO", "APAL", "SALIDAS"],
        combo_str="BARRIDO OP",
        activo=activo,
        best_so_far=best_so_far,
        timeframe_entry="",
        timeframe_exit="",
    )


# ══════════════════════════════════════════════════════════════════
#  MÉTRICAS, PARETO, KNEE
# ══════════════════════════════════════════════════════════════════

def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["ROI"]          = pd.to_numeric(out["ROI"],      errors="coerce")
    out["DRAWDOWN"]     = pd.to_numeric(out["DRAWDOWN"], errors="coerce")
    out["DRAWDOWN_ABS"] = out["DRAWDOWN"].abs()
    out["ROI_PER_DD"]   = out.apply(
        lambda r: r["ROI"] / r["DRAWDOWN_ABS"]
        if pd.notna(r["ROI"]) and pd.notna(r["DRAWDOWN_ABS"]) and r["DRAWDOWN_ABS"] > 1e-9
        else float("nan"),
        axis=1,
    )
    # Cuartil de ROI
    roi_valid = _finite_series(out["ROI"])
    if not roi_valid.empty:
        q25, q50, q75 = roi_valid.quantile([0.25, 0.5, 0.75])
        def _quartile(v):
            if pd.isna(v): return 0
            if v >= q75: return 4
            if v >= q50: return 3
            if v >= q25: return 2
            return 1
        out["ROI_QUARTILE"] = out["ROI"].apply(_quartile)
        out["RANK_ROI"] = out["ROI"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    else:
        out["ROI_QUARTILE"] = 0
        out["RANK_ROI"] = 0
    return out


def _compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["PARETO"] = False
    valid = d.dropna(subset=["ROI", "DRAWDOWN_ABS"]).sort_values(
        ["DRAWDOWN_ABS", "ROI"], ascending=[True, False]
    ).reset_index()
    best = -float("inf")
    for _, row in valid.iterrows():
        if row["ROI"] > best + 1e-12:
            best = row["ROI"]
            d.loc[int(row["index"]), "PARETO"] = True
    return d


def _compute_knee(df: pd.DataFrame) -> Optional[pd.Series]:
    frontier = df[df["PARETO"]].dropna(subset=["ROI", "DRAWDOWN_ABS"]).copy()
    if len(frontier) < 2:
        return frontier.iloc[0] if not frontier.empty else None
    roi = frontier["ROI"].astype(float)
    dd  = frontier["DRAWDOWN_ABS"].astype(float)
    r_span = max(float(roi.max() - roi.min()), 1e-9)
    d_span = max(float(dd.max()  - dd.min()),  1e-9)
    frontier = frontier.copy()
    frontier["_rn"] = (roi - roi.min()) / r_span
    frontier["_dn"] = (dd.max() - dd)  / d_span
    frontier["_dist"] = ((1 - frontier["_rn"])**2 + (1 - frontier["_dn"])**2)**0.5
    return frontier.sort_values("_dist").iloc[0]


# ══════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS COMPLETAS
# ══════════════════════════════════════════════════════════════════

def _stats(series: pd.Series) -> Dict[str, float]:
    vals = sorted(_finite_series(series).tolist())
    n = len(vals)
    if n == 0:
        return {k: float("nan") for k in ["n","min","max","mean","median","std","p25","p75","p90","skew"]}
    mean = sum(vals) / n
    std  = statistics.pstdev(vals) if n > 1 else 0.0
    def _pct(p):
        idx = (n - 1) * p
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)
    skew = (sum((v - mean)**3 for v in vals) / n) / max(std**3, 1e-12) if n > 2 else 0.0
    return {
        "n":      n,
        "min":    vals[0],
        "max":    vals[-1],
        "mean":   mean,
        "median": _pct(0.5),
        "std":    std,
        "p25":    _pct(0.25),
        "p75":    _pct(0.75),
        "p90":    _pct(0.90),
        "skew":   skew,
    }


# ══════════════════════════════════════════════════════════════════
#  DETECCIÓN AUTOMÁTICA DE ZONAS ÓPTIMAS
# ══════════════════════════════════════════════════════════════════

def _detect_zones(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detecta zonas óptimas agrupando por bins de SL×TP y Saldo×Apal.
    Devuelve zonas ordenadas por ROI medio descendente.
    """
    d = df.dropna(subset=["ROI", "EXIT_SL_PCT", "EXIT_TP_PCT", "SALDO USADO", "APALANCAMIENTO"]).copy()
    if d.empty:
        return []

    sl_vals  = sorted(d["EXIT_SL_PCT"].unique())
    tp_vals  = sorted(d["EXIT_TP_PCT"].unique())
    sal_vals = sorted(d["SALDO USADO"].unique())
    apal_vals = sorted(d["APALANCAMIENTO"].unique())

    q75_roi = float(_finite_series(d["ROI"]).quantile(0.75))

    zones = []

    # Zonas SL × TP
    for sl in sl_vals:
        for tp in tp_vals:
            sub = d[(d["EXIT_SL_PCT"] == sl) & (d["EXIT_TP_PCT"] == tp)]
            if len(sub) < 2:
                continue
            roi_m = float(sub["ROI"].mean())
            dd_m  = float(sub["DRAWDOWN_ABS"].mean()) if "DRAWDOWN_ABS" in sub else float("nan")
            pct_top = float((sub["ROI"] >= q75_roi).mean()) * 100
            zones.append({
                "tipo":    "SL×TP",
                "dim1":    f"SL={sl:.1f}%",
                "dim2":    f"TP={tp:.1f}%",
                "n":       len(sub),
                "roi_med": roi_m,
                "dd_med":  dd_m,
                "pct_top": pct_top,
                "sl":      sl,
                "tp":      tp,
                "sal_rng": f"{sub['SALDO USADO'].min():.0f}–{sub['SALDO USADO'].max():.0f}",
                "apal_rng":f"{sub['APALANCAMIENTO'].min():.0f}–{sub['APALANCAMIENTO'].max():.0f}",
            })

    # Zonas Saldo × Apalancamiento
    for sal in sal_vals:
        for apal in apal_vals:
            sub = d[(d["SALDO USADO"] == sal) & (d["APALANCAMIENTO"] == apal)]
            if len(sub) < 2:
                continue
            roi_m = float(sub["ROI"].mean())
            dd_m  = float(sub["DRAWDOWN_ABS"].mean()) if "DRAWDOWN_ABS" in sub else float("nan")
            pct_top = float((sub["ROI"] >= q75_roi).mean()) * 100
            zones.append({
                "tipo":    "Saldo×Apal",
                "dim1":    f"Saldo={sal:.0f}",
                "dim2":    f"Apal={apal:.0f}×",
                "n":       len(sub),
                "roi_med": roi_m,
                "dd_med":  dd_m,
                "pct_top": pct_top,
                "sl":      float("nan"),
                "tp":      float("nan"),
                "sal_rng": f"{sal:.0f}",
                "apal_rng":f"{apal:.0f}×",
            })

    return sorted(zones, key=lambda z: z["roi_med"], reverse=True)


# ══════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════

@dataclass
class CollectorReporter:
    saldo_usado: float
    apalancamiento: float
    rows: List[Dict[str, Any]]
    best_score: float = -float("inf")
    best_row: Optional[Dict[str, Any]] = None

    def on_trial_end(self, artifacts) -> None:
        m = artifacts.metrics or {}
        p = getattr(artifacts, "params", {}) or {}
        score = _safe_float(getattr(artifacts, "score", float("-inf")))
        if not math.isfinite(score):
            score = -float("inf")
        row = {
            "SALDO USADO":       self.saldo_usado,
            "APALANCAMIENTO":    self.apalancamiento,
            "ROI":               float(m.get("roi",      0.0)),
            "DRAWDOWN":          float(m.get("drawdown", 0.0)),
            "EXIT_TYPE":         str(p.get("__exit_type",      p.get("exit_type",      ""))),
            "EXIT_SL_PCT":       _safe_float(p.get("__exit_sl_pct",      p.get("exit_sl_pct",      float("nan")))),
            "EXIT_TP_PCT":       _safe_float(p.get("__exit_tp_pct",      p.get("exit_tp_pct",      float("nan")))),
            "EXIT_TRAIL_ACT_PCT":_safe_float(p.get("__exit_trail_act_pct", p.get("exit_trail_act_pct", float("nan")))),
            "EXIT_TRAIL_DIST_PCT":_safe_float(p.get("__exit_trail_dist_pct",p.get("exit_trail_dist_pct",float("nan")))),
            "__metrics":         dict(m),
            "__score":           float(score),
        }
        if score > self.best_score:
            self.best_score = score
            self.best_row = row

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if self.best_row is not None:
            self.rows.append(self.best_row)


class _ArtifactsWithCombo:
    """Proxy de artifacts que inyecta saldo_usado/apalancamiento en params.

    Se inyecta en params (no en metrics) para que el CSV los escriba con
    prefijo 'param_' → el Excel los detecta como known_param_names (VIA 1)
    y los muestra en la seccion PARAMETROS sin ser filtrados por 'very_bad'.
    """

    def __init__(self, base, saldo: float, apal: float):
        self._base = base

        # ── params ────────────────────────────────────────────────────
        # Nombres elegidos para que NO queden bloqueados por EXCLUDED_PARAMS
        # de visual/excel.py: "SALDO_USADO" pasa (no es "SALDO"), "APAL" pasa.
        p = dict(getattr(base, "params", {}) or {})
        p["saldo_usado"] = saldo
        p["apal"]        = apal
        self._params = p

        # ── params_reporting (si existe, inyectar tambien) ────────────
        pr = getattr(base, "params_reporting", None)
        if pr is not None:
            pr = dict(pr)
            pr["saldo_usado"] = saldo
            pr["apal"]        = apal
        self._params_reporting = pr

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    @property
    def params(self):
        return self._params

    @property
    def params_reporting(self):
        return self._params_reporting


class SweepExcelReporter:
    """Wrapper sobre ExcelReporter que acumula artifacts de TODOS los combos
    del barrido y genera los Excel resumen/trial una sola vez al finalizar."""

    def __init__(self, excel_reporter, activo: str):
        self._reporter = excel_reporter
        self._activo_override = activo

    def on_trial_end(self, artifacts) -> None:
        self._reporter.on_trial_end(artifacts)

    def on_strategy_end(self, strategy_name: str, study) -> None:
        # Suprimir la generación parcial por combo — se hará en finalize()
        pass

    def finalize(self, strategy_name: str) -> None:
        """Llamar una sola vez al terminar el barrido completo."""
        self._reporter._activo = self._activo_override
        self._reporter.on_strategy_end(strategy_name, None)


def _strategy_id_from_config() -> int:
    if isinstance(COMBINACION_A_EJECUTAR, (list, tuple)) and COMBINACION_A_EJECUTAR:
        return int(COMBINACION_A_EJECUTAR[0])
    return int(COMBINACION_A_EJECUTAR)


def _load_base_data():
    activo = ACTIVO_PRIMARIO
    try:
        archivo = resolve_archivo_data_tf(activo, TIMEFRAME)
    except Exception:
        archivo = resolve_archivo_data(activo)
    if not Path(archivo).exists():
        archivo = resolve_archivo_data(activo)
    df_1m   = load_data(archivo)
    df_1m   = filter_by_date(df_1m, FECHA_INICIO, FECHA_FIN)
    base_tf = normalize_timeframe_to_suffix(TIMEFRAME)
    df_base = df_1m if base_tf == "1m" else resample_to_base_timeframe(df_1m, base_tf)
    return activo, base_tf, df_1m, df_base


def _run_combo(*, strategy_id, activo, base_tf, df_1m, df_base,
               saldo_usado, apalancamiento, exit_params,
               extra_reporters=None) -> Dict[str, Any]:
    strategy = instantiate_strategies(only_id=strategy_id)[0]
    cfg = BacktestConfig(
        saldo_inicial          = float(CONFIG["SALDO_INICIAL"]),
        saldo_operativo_max    = float(CONFIG["SALDO_OPERATIVO_MAX"]),
        comision_pct           = float(CONFIG["COMISION_PCT"]),
        comision_sides         = int(CONFIG["COMISION_SIDES"]),
        saldo_minimo_operativo = float(CONFIG["SALDO_MINIMO_OPERATIVO"]),
        saldo_usado            = float(saldo_usado),
        apalancamiento_max     = float(apalancamiento),
        exit_type              = str(exit_params["exit_type"]),
        exit_sl_pct            = float(exit_params["exit_sl_pct"]),
        exit_tp_pct            = float(exit_params["exit_tp_pct"]),
        exit_trail_act_pct     = float(exit_params["exit_trail_act_pct"]),
        exit_trail_dist_pct    = float(exit_params["exit_trail_dist_pct"]),
        optimize_exits         = bool(OPTIMIZAR_SALIDAS_EN_OPT),
        exit_sl_pct_range      = (float(exit_params["exit_sl_pct"]),      float(exit_params["exit_sl_pct"]),      1.0),
        exit_tp_pct_range      = (float(exit_params["exit_tp_pct"]),      float(exit_params["exit_tp_pct"]),      1.0),
        exit_trail_act_pct_range  = (float(exit_params["exit_trail_act_pct"]), float(exit_params["exit_trail_act_pct"]), 1.0),
        exit_trail_dist_pct_range = (float(exit_params["exit_trail_dist_pct"]),float(exit_params["exit_trail_dist_pct"]),1.0),
    )
    rows: List[Dict[str, Any]] = []
    reporter = CollectorReporter(saldo_usado=float(saldo_usado), apalancamiento=float(apalancamiento), rows=rows)

    # Envolver SweepExcelReporter para inyectar saldo/apal en cada artifacts
    def _wrap_for_combo(r):
        if not isinstance(r, SweepExcelReporter):
            return r
        _saldo = float(saldo_usado)
        _apal  = float(apalancamiento)

        class _Proxy:
            def on_trial_end(self_, arts):
                r.on_trial_end(_ArtifactsWithCombo(arts, _saldo, _apal))

            def on_strategy_end(self_, name, study):
                r.on_strategy_end(name, study)

        return _Proxy()

    reporters = [reporter] + [_wrap_for_combo(r) for r in (extra_reporters or [])]
    runner = OptimizationRunner(
        config=cfg, n_trials=1, reporters=reporters,
        optuna=OptunaConfig(seed=OPTUNA_SEED, n_jobs=1, create_database=False, sampler=OPTUNA_SAMPLER),
        activo=activo,
    )
    runner.optimize_strategies(
        df=df_base, strategies=[strategy],
        df_by_timeframe={"1m": df_1m, base_tf: df_base}, base_timeframe=base_tf,
    )
    if rows:
        return rows[0]
    return {
        "SALDO USADO": float(saldo_usado), "APALANCAMIENTO": float(apalancamiento),
        "ROI": 0.0, "DRAWDOWN": 0.0,
        "EXIT_TYPE": str(EXIT_SETUP["exit_type"]),
        "EXIT_SL_PCT": float("nan"), "EXIT_TP_PCT": float("nan"),
        "EXIT_TRAIL_ACT_PCT": float("nan"), "EXIT_TRAIL_DIST_PCT": float("nan"),
    }


# ══════════════════════════════════════════════════════════════════
#  EXCEL — HOJA ÚNICA MINIMALISTA
# ══════════════════════════════════════════════════════════════════

def _write_excel(df: pd.DataFrame, out_path: Path) -> None:
    df2   = _add_derived_metrics(df)
    df2   = _compute_pareto(df2)
    knee  = _compute_knee(df2)
    zones = _detect_zones(df2)

    x_min, x_max = _axis_bounds(df2["DRAWDOWN_ABS"])
    y_min, y_max = _axis_bounds(df2["ROI"])

    frontier = (
        df2[df2["PARETO"]].dropna(subset=["ROI", "DRAWDOWN_ABS"])
        .sort_values("DRAWDOWN_ABS").copy()
    )

    # Pivots
    def _pivot(val_col, idx, col):
        return (
            df2.dropna(subset=[idx, col, val_col])
            .pivot_table(index=idx, columns=col, values=val_col, aggfunc="mean")
            .sort_index()
        )

    piv_sal_roi = _pivot("ROI",          "SALDO USADO", "APALANCAMIENTO")
    piv_sal_dd  = _pivot("DRAWDOWN_ABS", "SALDO USADO", "APALANCAMIENTO")
    piv_sltp    = _pivot("ROI",          "EXIT_SL_PCT", "EXIT_TP_PCT")

    # Histogramas
    def _hist(vals, bins=20):
        vals = [v for v in vals if math.isfinite(v)]
        if not vals: return [], []
        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-9: return [f"{mn:.2f}"], [len(vals)]
        step = (mx - mn) / bins
        edges = [mn + i * step for i in range(bins + 1)]
        counts = [0] * bins
        for v in vals:
            idx = min(int((v - mn) / step), bins - 1)
            counts[idx] += 1
        return [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(bins)], counts

    roi_vals = _finite_series(df2["ROI"]).tolist()
    dd_vals  = _finite_series(df2["DRAWDOWN_ABS"]).tolist()
    roi_labs, roi_cnts = _hist(roi_vals)
    dd_labs,  dd_cnts  = _hist(dd_vals)

    # Best points
    best_roi_idx = df2["ROI"].idxmax() if not df2["ROI"].isna().all() else None
    best_eff_idx = df2["ROI_PER_DD"].idxmax() if not df2["ROI_PER_DD"].isna().all() else None

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── FORMATOS MINIMALISTAS ──────────────────────────────────
        # Paleta: fondo blanco, grises, un solo color de acento (#2962FF)
        FMT = {
            "hdr": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 9,
                "font_color": "#FFFFFF", "bg_color": "#1A1A2E",
                "border": 0, "align": "center", "valign": "vcenter",
            }),
            "title": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 13,
                "font_color": "#1A1A2E", "border": 0,
            }),
            "sub": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 9,
                "font_color": "#555577", "border": 0, "italic": True,
            }),
            "kpi_val": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 16,
                "font_color": "#1A1A2E", "border": 0, "align": "center",
                "num_format": "0.00",
            }),
            "kpi_lbl": wb.add_format({
                "font_name": "Calibri", "font_size": 8,
                "font_color": "#999999", "border": 0, "align": "center",
            }),
            "kpi_green": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 16,
                "font_color": "#1A7A4A", "border": 0, "align": "center",
                "num_format": "0.00",
            }),
            "kpi_amber": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 16,
                "font_color": "#9A5200", "border": 0, "align": "center",
                "num_format": "0.00",
            }),
            "kpi_blue": wb.add_format({
                "bold": True, "font_name": "Calibri", "font_size": 16,
                "font_color": "#1650B0", "border": 0, "align": "center",
                "num_format": "0.00",
            }),
            "data": wb.add_format({
                "font_name": "Calibri", "font_size": 8,
                "border": 0, "num_format": "0.00", "align": "right",
            }),
            "data_str": wb.add_format({
                "font_name": "Calibri", "font_size": 8,
                "border": 0, "align": "left",
            }),
            "pareto": wb.add_format({
                "font_name": "Calibri", "font_size": 8, "num_format": "0.00",
                "bg_color": "#EBF5FF", "border": 0, "align": "right",
            }),
            "knee": wb.add_format({
                "font_name": "Calibri", "font_size": 8, "num_format": "0.00",
                "bg_color": "#FFF3CD", "bold": True, "border": 0, "align": "right",
            }),
            "roi_pos": wb.add_format({
                "font_name": "Calibri", "font_size": 8, "num_format": "0.00",
                "font_color": "#1A7A4A", "align": "right",
            }),
            "roi_neg": wb.add_format({
                "font_name": "Calibri", "font_size": 8, "num_format": "0.00",
                "font_color": "#C0392B", "align": "right",
            }),
            "divider": wb.add_format({
                "bottom": 1, "bottom_color": "#E0E0E0",
            }),
            "zone_lbl": wb.add_format({
                "font_name": "Calibri", "font_size": 8, "bold": True,
                "font_color": "#333355", "border": 0,
            }),
        }

        # ── HOJA ÚNICA: DASHBOARD ──────────────────────────────────
        SH = "DASHBOARD"
        ws = wb.add_worksheet(SH)
        writer.sheets[SH] = ws
        ws.set_zoom(85)

        # Anchos de columna (diseño en cuadrícula invisible)
        ws.set_column("A:A", 2)      # margen izq
        ws.set_column("B:C", 14)     # KPI labels / zona
        ws.set_column("D:D", 14)
        ws.set_column("E:E", 14)
        ws.set_column("F:F", 14)
        ws.set_column("G:G", 14)
        ws.set_column("H:H", 14)
        ws.set_column("I:I", 14)
        ws.set_column("J:J", 14)
        ws.set_column("K:K", 14)
        ws.set_column("L:L", 14)
        ws.set_column("M:M", 14)
        ws.set_column("N:N", 14)
        ws.set_column("O:O", 14)
        ws.set_column("P:P", 14)
        ws.set_column("Q:Q", 14)
        ws.set_column("R:R", 14)
        ws.set_column("S:S", 14)
        ws.set_row(0, 6)   # espaciado top

        # ── FILA 1: TÍTULO ─────────────────────────────────────────
        ws.write("B2", f"ROI × DRAWDOWN  —  Optimization Analysis", FMT["title"])
        ws.write("B3", f"Modo: {MODO_SALIDA}  ·  {len(df2)} combinaciones  ·  Generado {datetime.now():%Y-%m-%d %H:%M}", FMT["sub"])
        ws.set_row(1, 22)
        ws.set_row(2, 16)

        # ── FILA KPIs ─────────────────────────────────────────────
        ws.set_row(4, 28)
        ws.set_row(5, 14)

        def _write_kpi(col, val, label, fmt_key="kpi_val"):
            ws.write(4, col, val if math.isfinite(_safe_float(val)) else "—", FMT[fmt_key])
            ws.write(5, col, label, FMT["kpi_lbl"])

        best_roi_val = float(df2.loc[best_roi_idx, "ROI"]) if best_roi_idx is not None else float("nan")
        best_eff_val = float(df2.loc[best_eff_idx, "ROI_PER_DD"]) if best_eff_idx is not None else float("nan")
        knee_roi     = float(knee["ROI"]) if knee is not None else float("nan")
        knee_dd      = float(knee["DRAWDOWN_ABS"]) if knee is not None else float("nan")
        pct_pos      = float((df2["ROI"] > 0).mean()) * 100
        n_pareto     = int(df2["PARETO"].sum())
        roi_s        = _stats(df2["ROI"])

        _write_kpi(1, best_roi_val,  "Mejor ROI (%)",       "kpi_green")
        _write_kpi(2, best_eff_val,  "Mejor ROI/DD",        "kpi_blue")
        _write_kpi(3, knee_roi,      "Knee ROI (%)",        "kpi_amber")
        _write_kpi(4, knee_dd,       "Knee DD (%)",         "kpi_val")
        _write_kpi(5, roi_s["mean"], "ROI medio (%)",       "kpi_val")
        _write_kpi(6, roi_s["p75"],  "ROI P75 (%)",         "kpi_val")
        _write_kpi(7, roi_s["p90"],  "ROI P90 (%)",         "kpi_green")
        _write_kpi(8, pct_pos,       "% ROI positivo",      "kpi_val")
        _write_kpi(9, n_pareto,      "Pts. Pareto",         "kpi_blue")

        # separador
        for c in range(1, 18):
            ws.write(6, c, "", FMT["divider"])

        # ── AUX (oculta) ──────────────────────────────────────────
        aux = "AUX"
        ws_aux = wb.add_worksheet(aux)
        writer.sheets[aux] = ws_aux
        ws_aux.hide()
        ar = 0  # aux row pointer

        # Iso-líneas
        iso_ks    = [0.5, 1.0, 2.0, 3.0]
        iso_specs = []  # (k, x_range, y_range)
        for k in iso_ks:
            xs, ys = _iso_line(x_min, x_max, k)
            ws_aux.write(ar, 0, f"ISO_DD_{k}")
            ws_aux.write(ar, 1, f"ISO_ROI_{k}")
            for i, (x, y) in enumerate(zip(xs, ys), 1):
                ws_aux.write_number(ar + i, 0, x)
                ws_aux.write_number(ar + i, 1, y)
            iso_specs.append((k, [aux, ar+1, 0, ar+len(xs), 0], [aux, ar+1, 1, ar+len(xs), 1]))
            ar += len(xs) + 3

        # Knee en AUX
        knee_ar = None
        if knee is not None:
            ws_aux.write(ar, 0, "KNEE_DD"); ws_aux.write(ar, 1, "KNEE_ROI")
            ws_aux.write_number(ar+1, 0, float(knee["DRAWDOWN_ABS"]))
            ws_aux.write_number(ar+1, 1, float(knee["ROI"]))
            knee_ar = ar + 1
            ar += 4

        # Best ROI en AUX
        best_roi_ar = None
        if best_roi_idx is not None:
            r = df2.loc[best_roi_idx]
            ws_aux.write(ar, 0, "BEST_ROI_DD"); ws_aux.write(ar, 1, "BEST_ROI_ROI")
            ws_aux.write_number(ar+1, 0, float(r["DRAWDOWN_ABS"]))
            ws_aux.write_number(ar+1, 1, float(r["ROI"]))
            best_roi_ar = ar + 1
            ar += 4

        # Best eff en AUX
        best_eff_ar = None
        if best_eff_idx is not None:
            r = df2.loc[best_eff_idx]
            ws_aux.write(ar, 0, "BEST_EFF_DD"); ws_aux.write(ar, 1, "BEST_EFF_ROI")
            ws_aux.write_number(ar+1, 0, float(r["DRAWDOWN_ABS"]))
            ws_aux.write_number(ar+1, 1, float(r["ROI"]))
            best_eff_ar = ar + 1
            ar += 4

        # Frontera en AUX
        front_ar = ar
        ws_aux.write(ar, 0, "FRONT_DD"); ws_aux.write(ar, 1, "FRONT_ROI")
        for i, (_, row) in enumerate(frontier.iterrows(), 1):
            ws_aux.write_number(ar+i, 0, float(row["DRAWDOWN_ABS"]))
            ws_aux.write_number(ar+i, 1, float(row["ROI"]))
        front_end = ar + len(frontier)
        ar += len(frontier) + 4

        # Histogramas en AUX
        roi_hist_ar = ar
        ws_aux.write(ar, 0, "ROI_BIN"); ws_aux.write(ar, 1, "ROI_CNT")
        for i, (lab, cnt) in enumerate(zip(roi_labs, roi_cnts), 1):
            ws_aux.write(ar+i, 0, lab); ws_aux.write_number(ar+i, 1, cnt)
        roi_hist_end = ar + len(roi_labs)
        ar += len(roi_labs) + 3

        dd_hist_ar = ar
        ws_aux.write(ar, 0, "DD_BIN"); ws_aux.write(ar, 1, "DD_CNT")
        for i, (lab, cnt) in enumerate(zip(dd_labs, dd_cnts), 1):
            ws_aux.write(ar+i, 0, lab); ws_aux.write_number(ar+i, 1, cnt)
        dd_hist_end = ar + len(dd_labs)
        ar += len(dd_labs) + 3

        # ── COLUMNAS DATOS ─────────────────────────────────────────
        col_roi  = list(df2.columns).index("ROI")
        col_dd   = list(df2.columns).index("DRAWDOWN_ABS")

        # ══ GRÁFICO 1: ROI vs DD (scatter + frontera + knee + iso) ════
        # Escribimos todos los puntos en AUX para el scatter
        sc_ar = ar
        ws_aux.write(ar, 0, "SC_DD"); ws_aux.write(ar, 1, "SC_ROI")
        for i, (_, row) in enumerate(df2.iterrows(), 1):
            ws_aux.write_number(ar+i, 0, float(row["DRAWDOWN_ABS"]) if pd.notna(row["DRAWDOWN_ABS"]) else 0)
            ws_aux.write_number(ar+i, 1, float(row["ROI"])          if pd.notna(row["ROI"]) else 0)
        sc_end = ar + len(df2)
        ar += len(df2) + 4

        chart1 = wb.add_chart({"type": "scatter"})
        # Serie: todos los puntos
        chart1.add_series({
            "name": "Combinaciones",
            "categories": [aux, sc_ar+1, 0, sc_end, 0],
            "values":     [aux, sc_ar+1, 1, sc_end, 1],
            "marker": {
                "type": "circle", "size": 4,
                "fill": {"color": "#B0C8F8"},
                "border": {"color": "#5588DD", "width": 0.5},
            },
        })
        # Frontera Pareto
        if not frontier.empty:
            chart1.add_series({
                "name": "Frontera Pareto",
                "categories": [aux, front_ar+1, 0, front_end, 0],
                "values":     [aux, front_ar+1, 1, front_end, 1],
                "marker": {"type": "diamond", "size": 6, "fill": {"color": "#2ECC71"}, "border": {"color": "#1A7A4A"}},
                "line":   {"color": "#2ECC71", "width": 1.5},
            })
        # Iso-líneas
        iso_colors_xl = ["#CCCCCC", "#AAAAAA", "#888888", "#666666"]
        for (k, xr, yr), clr in zip(iso_specs, iso_colors_xl):
            chart1.add_series({
                "name": f"ROI={k:.1f}×DD",
                "categories": xr, "values": yr,
                "line": {"color": clr, "dash_type": "dash", "width": 0.75},
                "marker": {"type": "none"},
            })
        # Knee
        if knee_ar is not None:
            chart1.add_series({
                "name": "Knee",
                "categories": [aux, knee_ar, 0, knee_ar, 0],
                "values":     [aux, knee_ar, 1, knee_ar, 1],
                "marker": {"type": "star", "size": 10, "fill": {"color": "#F39C12"}, "border": {"color": "#9A5200"}},
                "line": {"none": True},
            })
        # Best ROI
        if best_roi_ar is not None:
            chart1.add_series({
                "name": "Mejor ROI",
                "categories": [aux, best_roi_ar, 0, best_roi_ar, 0],
                "values":     [aux, best_roi_ar, 1, best_roi_ar, 1],
                "marker": {"type": "triangle", "size": 10, "fill": {"color": "#1A7A4A"}, "border": {"color": "#0D3D25"}},
                "line": {"none": True},
            })
        # Best eff
        if best_eff_ar is not None:
            chart1.add_series({
                "name": "Mejor ROI/DD",
                "categories": [aux, best_eff_ar, 0, best_eff_ar, 0],
                "values":     [aux, best_eff_ar, 1, best_eff_ar, 1],
                "marker": {"type": "diamond", "size": 10, "fill": {"color": "#2962FF"}, "border": {"color": "#1040BB"}},
                "line": {"none": True},
            })

        chart1.set_title({"name": "ROI vs Drawdown"})
        chart1.set_x_axis({"name": "Drawdown (%)", "min": x_min, "max": x_max,
                           "line": {"color": "#DDDDDD"}, "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart1.set_y_axis({"name": "ROI (%)", "min": y_min, "max": y_max,
                           "line": {"color": "#DDDDDD"}, "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart1.set_legend({"position": "bottom", "font": {"size": 8}})
        chart1.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart1.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart1.set_size({"width": 520, "height": 340})
        ws.insert_chart("B8", chart1, {"x_offset": 5, "y_offset": 5})

        # ══ GRÁFICO 2: Heatmap SL × TP → ROI ══════════════════════
        # Escribimos pivot en AUX como columnas
        sltp_ar = ar
        sl_list = sorted(piv_sltp.index.tolist())
        tp_list = sorted(piv_sltp.columns.tolist())
        ws_aux.write(ar, 0, "SL"); ws_aux.write(ar, 1, "TP"); ws_aux.write(ar, 2, "ROI_SLTP")
        sl_tp_combos = []
        for sl_v in sl_list:
            for tp_v in tp_list:
                v = piv_sltp.loc[sl_v, tp_v] if sl_v in piv_sltp.index and tp_v in piv_sltp.columns else float("nan")
                if pd.notna(v):
                    ar += 1
                    ws_aux.write_number(ar, 0, float(sl_v))
                    ws_aux.write_number(ar, 1, float(tp_v))
                    ws_aux.write_number(ar, 2, float(v))
                    sl_tp_combos.append((float(sl_v), float(tp_v), float(v)))
        sltp_end = ar
        ar += 4

        # Para heatmap en Excel usamos scatter con color por signo (compatibilidad amplia)
        chart2 = wb.add_chart({"type": "scatter", "subtype": "marker_only"})
        if sl_tp_combos:
            # Separar en dos grupos: ROI positivo y negativo para colorear
            pos_ar = ar
            ws_aux.write(ar, 0, "SL_P"); ws_aux.write(ar, 1, "TP_P"); ws_aux.write(ar, 2, "ROI_P")
            neg_ar_start = None
            pos_rows, neg_rows = [], []
            for sl_v, tp_v, roi_v in sl_tp_combos:
                if roi_v >= 0:
                    pos_rows.append((sl_v, tp_v, roi_v))
                else:
                    neg_rows.append((sl_v, tp_v, abs(roi_v)))

            for i, (sl_v, tp_v, roi_v) in enumerate(pos_rows, 1):
                ws_aux.write_number(ar+i, 0, sl_v)
                ws_aux.write_number(ar+i, 1, tp_v)
                ws_aux.write_number(ar+i, 2, max(roi_v, 0.5))
            pos_end = ar + len(pos_rows)
            ar += len(pos_rows) + 3

            neg_ar_start = ar
            ws_aux.write(ar, 0, "SL_N"); ws_aux.write(ar, 1, "TP_N"); ws_aux.write(ar, 2, "ROI_N")
            for i, (sl_v, tp_v, roi_v) in enumerate(neg_rows, 1):
                ws_aux.write_number(ar+i, 0, sl_v)
                ws_aux.write_number(ar+i, 1, tp_v)
                ws_aux.write_number(ar+i, 2, max(roi_v, 0.5))
            neg_end = ar + len(neg_rows)
            ar += len(neg_rows) + 3

            if pos_rows:
                chart2.add_series({
                    "name": "ROI positivo",
                    "categories": [aux, pos_ar+1, 0, pos_end, 0],
                    "values":     [aux, pos_ar+1, 1, pos_end, 1],
                    "marker": {"type": "circle", "size": 7, "fill": {"color": "#2ECC71"}, "border": {"color": "#1A7A4A"}},
                })
            if neg_rows:
                chart2.add_series({
                    "name": "ROI negativo",
                    "categories": [aux, neg_ar_start+1, 0, neg_end, 0],
                    "values":     [aux, neg_ar_start+1, 1, neg_end, 1],
                    "marker": {"type": "circle", "size": 7, "fill": {"color": "#E74C3C"}, "border": {"color": "#9B1C1C"}},
                })

        chart2.set_title({"name": "SL% × TP%  →  ROI"})
        chart2.set_x_axis({"name": "SL (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart2.set_y_axis({"name": "TP (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart2.set_legend({"position": "bottom", "font": {"size": 8}})
        chart2.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart2.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart2.set_size({"width": 380, "height": 340})
        ws.insert_chart("K8", chart2, {"x_offset": 5, "y_offset": 5})

        # ══ GRÁFICO 3: Heatmap Saldo × Apal → ROI ══════════════════
        sal_list2  = sorted(df2["SALDO USADO"].dropna().unique().tolist())
        apal_list2 = sorted(df2["APALANCAMIENTO"].dropna().unique().tolist())

        sal_apal_pos, sal_apal_neg = [], []
        for s in sal_list2:
            for a in apal_list2:
                sub = df2[(df2["SALDO USADO"] == s) & (df2["APALANCAMIENTO"] == a)]
                if sub.empty: continue
                v = float(sub["ROI"].mean())
                if math.isfinite(v):
                    (sal_apal_pos if v >= 0 else sal_apal_neg).append((s, a, abs(v) if v < 0 else v))

        sa_pos_ar = ar
        ws_aux.write(ar, 0, "SAL_P"); ws_aux.write(ar, 1, "APAL_P"); ws_aux.write(ar, 2, "ROI_SAP")
        for i, (s, a, v) in enumerate(sal_apal_pos, 1):
            ws_aux.write_number(ar+i, 0, s); ws_aux.write_number(ar+i, 1, a)
            ws_aux.write_number(ar+i, 2, max(v, 0.5))
        sa_pos_end = ar + len(sal_apal_pos)
        ar += len(sal_apal_pos) + 3

        sa_neg_ar = ar
        ws_aux.write(ar, 0, "SAL_N"); ws_aux.write(ar, 1, "APAL_N"); ws_aux.write(ar, 2, "ROI_SAN")
        for i, (s, a, v) in enumerate(sal_apal_neg, 1):
            ws_aux.write_number(ar+i, 0, s); ws_aux.write_number(ar+i, 1, a)
            ws_aux.write_number(ar+i, 2, max(v, 0.5))
        sa_neg_end = ar + len(sal_apal_neg)
        ar += len(sal_apal_neg) + 3

        chart3 = wb.add_chart({"type": "scatter", "subtype": "marker_only"})
        if sal_apal_pos:
            chart3.add_series({
                "name": "ROI positivo",
                "categories":  [aux, sa_pos_ar+1, 0, sa_pos_end, 0],
                "values":      [aux, sa_pos_ar+1, 1, sa_pos_end, 1],
                "marker": {"type": "circle", "size": 8, "fill": {"color": "#2962FF"}, "border": {"color": "#1040BB"}},
            })
        if sal_apal_neg:
            chart3.add_series({
                "name": "ROI negativo",
                "categories":  [aux, sa_neg_ar+1, 0, sa_neg_end, 0],
                "values":      [aux, sa_neg_ar+1, 1, sa_neg_end, 1],
                "marker": {"type": "circle", "size": 8, "fill": {"color": "#E74C3C"}, "border": {"color": "#9B1C1C"}},
            })
        chart3.set_title({"name": "Saldo × Apalancamiento  →  ROI"})
        chart3.set_x_axis({"name": "Saldo", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart3.set_y_axis({"name": "Apalancamiento", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart3.set_legend({"position": "bottom", "font": {"size": 8}})
        chart3.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart3.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart3.set_size({"width": 380, "height": 340})
        ws.insert_chart("Q8", chart3, {"x_offset": 5, "y_offset": 5})

        # ══ GRÁFICO 4: Histograma ROI ══════════════════════════════
        chart4 = wb.add_chart({"type": "column"})
        if roi_labs:
            chart4.add_series({
                "name": "Distribución ROI",
                "categories": [aux, roi_hist_ar+1, 0, roi_hist_end, 0],
                "values":     [aux, roi_hist_ar+1, 1, roi_hist_end, 1],
                "fill":   {"color": "#2ECC71"}, "border": {"none": True},
                "gap": 5,
            })
        chart4.set_title({"name": "Distribución ROI (%)"})
        chart4.set_x_axis({"name": "ROI (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": False}})
        chart4.set_y_axis({"name": "Frecuencia", "line": {"none": True},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart4.set_legend({"none": True})
        chart4.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart4.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart4.set_size({"width": 350, "height": 250})
        ws.insert_chart("B28", chart4, {"x_offset": 5, "y_offset": 5})

        # ══ GRÁFICO 5: Histograma DD ═══════════════════════════════
        chart5 = wb.add_chart({"type": "column"})
        if dd_labs:
            chart5.add_series({
                "name": "Distribución Drawdown",
                "categories": [aux, dd_hist_ar+1, 0, dd_hist_end, 0],
                "values":     [aux, dd_hist_ar+1, 1, dd_hist_end, 1],
                "fill":   {"color": "#E74C3C"}, "border": {"none": True},
                "gap": 5,
            })
        chart5.set_title({"name": "Distribución Drawdown (%)"})
        chart5.set_x_axis({"name": "DD (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": False}})
        chart5.set_y_axis({"name": "Frecuencia", "line": {"none": True},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart5.set_legend({"none": True})
        chart5.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart5.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart5.set_size({"width": 350, "height": 250})
        ws.insert_chart("H28", chart5, {"x_offset": 5, "y_offset": 5})

        # ══ GRÁFICO 6: ROI/DD scatter ══════════════════════════════
        # todos los puntos coloreados por cuartil
        q_colors = {1: "#E74C3C", 2: "#F39C12", 3: "#3498DB", 4: "#2ECC71"}
        charts_q = {}
        for q in [1, 2, 3, 4]:
            sub_q = df2[df2["ROI_QUARTILE"] == q].dropna(subset=["DRAWDOWN_ABS", "ROI"])
            if sub_q.empty: continue
            q_ar = ar
            ws_aux.write(ar, 0, f"Q{q}_DD"); ws_aux.write(ar, 1, f"Q{q}_ROI")
            for i, (_, row) in enumerate(sub_q.iterrows(), 1):
                ws_aux.write_number(ar+i, 0, float(row["DRAWDOWN_ABS"]))
                ws_aux.write_number(ar+i, 1, float(row["ROI"]))
            q_end = ar + len(sub_q)
            ar += len(sub_q) + 3
            charts_q[q] = (q_ar, q_end)

        chart6 = wb.add_chart({"type": "scatter", "subtype": "marker_only"})
        q_labels = {1: "Q1 (peor)", 2: "Q2", 3: "Q3", 4: "Q4 (mejor)"}
        for q, (q_ar, q_end) in charts_q.items():
            chart6.add_series({
                "name": q_labels[q],
                "categories": [aux, q_ar+1, 0, q_end, 0],
                "values":     [aux, q_ar+1, 1, q_end, 1],
                "marker": {
                    "type": "circle", "size": 5,
                    "fill":   {"color": q_colors[q]},
                    "border": {"none": True},
                },
            })
        chart6.set_title({"name": "Cuartiles ROI (Q1→Q4)"})
        chart6.set_x_axis({"name": "Drawdown (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart6.set_y_axis({"name": "ROI (%)", "line": {"color": "#DDDDDD"},
                           "major_gridlines": {"visible": True, "line": {"color": "#F0F0F0"}}})
        chart6.set_legend({"position": "bottom", "font": {"size": 8}})
        chart6.set_plotarea({"border": {"none": True}, "fill": {"color": "#FAFAFA"}})
        chart6.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart6.set_size({"width": 350, "height": 250})
        ws.insert_chart("N28", chart6, {"x_offset": 5, "y_offset": 5})

        # ══ TABLA ZONAS ÓPTIMAS ════════════════════════════════════
        zone_row = 44
        ws.write(zone_row, 1, "ZONAS ÓPTIMAS (Top SL×TP — todos los datos)", FMT["title"])
        zone_row += 1
        zone_hdrs = ["Tipo", "Dim 1", "Dim 2", "N", "ROI medio", "DD medio", "% top cuartil", "Saldo rango", "Apal rango"]
        for ci, h in enumerate(zone_hdrs):
            ws.write(zone_row, ci+1, h, FMT["hdr"])
        ws.set_row(zone_row, 14)
        zone_row += 1

        top_zones = [z for z in zones if z["tipo"] == "SL×TP"][:15]
        for z in top_zones:
            ws.write(zone_row, 1, z["tipo"],         FMT["data_str"])
            ws.write(zone_row, 2, z["dim1"],         FMT["data_str"])
            ws.write(zone_row, 3, z["dim2"],         FMT["data_str"])
            ws.write(zone_row, 4, z["n"],            FMT["data"])
            ws.write(zone_row, 5, round(z["roi_med"], 2), FMT["roi_pos"] if z["roi_med"] >= 0 else FMT["roi_neg"])
            ws.write(zone_row, 6, round(z["dd_med"], 2) if math.isfinite(z["dd_med"]) else "—", FMT["data"])
            ws.write(zone_row, 7, round(z["pct_top"], 1), FMT["data"])
            ws.write(zone_row, 8, z["sal_rng"],      FMT["data_str"])
            ws.write(zone_row, 9, z["apal_rng"],     FMT["data_str"])
            ws.set_row(zone_row, 13)
            zone_row += 1

        # ══ TABLA ZONAS SALDO × APAL ═══════════════════════════════
        zone_row += 2
        ws.write(zone_row, 1, "ZONAS ÓPTIMAS (Top Saldo×Apalancamiento)", FMT["title"])
        zone_row += 1
        for ci, h in enumerate(zone_hdrs):
            ws.write(zone_row, ci+1, h, FMT["hdr"])
        ws.set_row(zone_row, 14)
        zone_row += 1

        top_zones_sa = [z for z in zones if z["tipo"] == "Saldo×Apal"][:15]
        for z in top_zones_sa:
            ws.write(zone_row, 1, z["tipo"],         FMT["data_str"])
            ws.write(zone_row, 2, z["dim1"],         FMT["data_str"])
            ws.write(zone_row, 3, z["dim2"],         FMT["data_str"])
            ws.write(zone_row, 4, z["n"],            FMT["data"])
            ws.write(zone_row, 5, round(z["roi_med"], 2), FMT["roi_pos"] if z["roi_med"] >= 0 else FMT["roi_neg"])
            ws.write(zone_row, 6, round(z["dd_med"], 2) if math.isfinite(z["dd_med"]) else "—", FMT["data"])
            ws.write(zone_row, 7, round(z["pct_top"], 1), FMT["data"])
            ws.write(zone_row, 8, z["sal_rng"],      FMT["data_str"])
            ws.write(zone_row, 9, z["apal_rng"],     FMT["data_str"])
            ws.set_row(zone_row, 13)
            zone_row += 1

        # ══ TABLA DATOS COMPLETA ════════════════════════════════════
        data_start = zone_row + 3
        ws.write(data_start, 1, "DATOS COMPLETOS", FMT["title"])
        data_start += 1

        data_cols = ["SALDO USADO","APALANCAMIENTO","ROI","DRAWDOWN_ABS","ROI_PER_DD",
                     "EXIT_SL_PCT","EXIT_TP_PCT","EXIT_TYPE","RANK_ROI","ROI_QUARTILE","PARETO"]
        for ci, h in enumerate(data_cols):
            ws.write(data_start, ci+1, h, FMT["hdr"])
        ws.set_row(data_start, 14)

        pareto_set = set(df2[df2["PARETO"]].index)
        knee_idx   = knee.name if knee is not None else -1

        for ri, (idx, row) in enumerate(df2[data_cols].iterrows(), 1):
            is_knee   = (idx == knee_idx)
            is_pareto = idx in pareto_set
            row_fmt   = FMT["knee"] if is_knee else (FMT["pareto"] if is_pareto else FMT["data"])
            for ci, col in enumerate(data_cols):
                v = row[col]
                if col == "EXIT_TYPE":
                    ws.write(data_start + ri, ci+1, str(v), FMT["data_str"])
                elif col == "PARETO":
                    ws.write(data_start + ri, ci+1, "✓" if v else "", FMT["data_str"])
                elif col == "ROI":
                    ws.write(data_start + ri, ci+1, float(v) if pd.notna(v) else "", row_fmt if is_knee or is_pareto else (FMT["roi_pos"] if _safe_float(v) >= 0 else FMT["roi_neg"]))
                else:
                    fv = _safe_float(v)
                    ws.write(data_start + ri, ci+1, fv if math.isfinite(fv) else "", row_fmt)
            ws.set_row(data_start + ri, 12)

        ws.freeze_panes(data_start + 1, 0)
        ws.activate()

    print(f"  ✅ Excel: {out_path}")


# ══════════════════════════════════════════════════════════════════
#  HTML — TRADINGVIEW AESTHETIC
# ══════════════════════════════════════════════════════════════════

def _write_html(df: pd.DataFrame, out_path: Path) -> None:
    df2   = _add_derived_metrics(df)
    df2   = _compute_pareto(df2)
    knee  = _compute_knee(df2)
    zones = _detect_zones(df2)
    roi_s = _stats(df2["ROI"])
    dd_s  = _stats(df2["DRAWDOWN_ABS"])

    best_roi_idx = df2["ROI"].idxmax()       if not df2["ROI"].isna().all()      else None
    best_eff_idx = df2["ROI_PER_DD"].idxmax() if not df2["ROI_PER_DD"].isna().all() else None

    def _sf(v): return float(v) if math.isfinite(_safe_float(v)) else None
    def _fmtf(v, d=2): f = _safe_float(v); return f"{f:.{d}f}" if math.isfinite(f) else "—"

    # Hover text por fila
    def _hover(row) -> str:
        et = str(row.get("EXIT_TYPE", ""))
        sl = _fmtf(row.get("EXIT_SL_PCT"))
        tp = _fmtf(row.get("EXIT_TP_PCT"))
        act= _fmtf(row.get("EXIT_TRAIL_ACT_PCT"))
        dist=_fmtf(row.get("EXIT_TRAIL_DIST_PCT"))
        exit_str = f"SL {sl}% | TP {tp}%" if "trailing" not in et.lower() else f"SL {sl}% | ACT {act}% | DIST {dist}%"
        eff = _fmtf(row.get("ROI_PER_DD"))
        return (
            f"<b>Saldo:</b> {_fmtf(row['SALDO USADO'], 0)}<br>"
            f"<b>Apalancamiento:</b> {_fmtf(row['APALANCAMIENTO'], 0)}×<br>"
            f"<b>ROI:</b> {_fmtf(row['ROI'])}%<br>"
            f"<b>Drawdown:</b> {_fmtf(row['DRAWDOWN_ABS'])}%<br>"
            f"<b>ROI/DD:</b> {eff}<br>"
            f"<b>Salida:</b> {exit_str}<br>"
            f"<b>Rank:</b> #{int(row.get('RANK_ROI', 0))}"
        )

    df2["_hover"] = df2.apply(_hover, axis=1)

    # Serializar todas las columnas necesarias como listas
    def _col(c):
        return [_sf(v) for v in df2[c]]
    def _scol(c):
        return list(df2[c].astype(str))

    x_min, x_max = _axis_bounds(df2["DRAWDOWN_ABS"])
    y_min, y_max = _axis_bounds(df2["ROI"])

    # Contour SL × TP → ROI (grid regular)
    df_fixed = df2[df2["EXIT_TYPE"].str.contains("fixed", case=False, na=False)]
    piv_sltp = (
        df_fixed.dropna(subset=["EXIT_SL_PCT", "EXIT_TP_PCT", "ROI"])
        .pivot_table(index="EXIT_SL_PCT", columns="EXIT_TP_PCT", values="ROI", aggfunc="mean")
        .sort_index()
    ) if not df_fixed.empty else pd.DataFrame()

    cnt_sltp = {}
    if not piv_sltp.empty:
        cnt_sltp = {
            "z": [[_sf(piv_sltp.loc[sl, tp]) for tp in piv_sltp.columns] for sl in piv_sltp.index],
            "x": [float(c) for c in piv_sltp.columns],
            "y": [float(i) for i in piv_sltp.index],
        }

    # Contour Saldo × Apal → ROI
    piv_sa = (
        df2.dropna(subset=["SALDO USADO", "APALANCAMIENTO", "ROI"])
        .pivot_table(index="SALDO USADO", columns="APALANCAMIENTO", values="ROI", aggfunc="mean")
        .sort_index()
    )
    cnt_sa = {}
    if not piv_sa.empty:
        cnt_sa = {
            "z": [[_sf(piv_sa.loc[s, a]) for a in piv_sa.columns] for s in piv_sa.index],
            "x": [float(c) for c in piv_sa.columns],
            "y": [float(i) for i in piv_sa.index],
        }

    # Scatter points individuales con hover (para overlay sobre contour)
    scatter_pts = {
        "dd":   _col("DRAWDOWN_ABS"),
        "roi":  _col("ROI"),
        "sl":   _col("EXIT_SL_PCT"),
        "tp":   _col("EXIT_TP_PCT"),
        "sal":  _col("SALDO USADO"),
        "apal": _col("APALANCAMIENTO"),
        "eff":  _col("ROI_PER_DD"),
        "q":    list(df2["ROI_QUARTILE"].astype(int)),
        "par":  list(df2["PARETO"].astype(int)),
        "rank": list(df2["RANK_ROI"].astype(int)),
        "hover":list(df2["_hover"]),
    }

    # Frontera Pareto
    frontier = df2[df2["PARETO"]].dropna(subset=["ROI","DRAWDOWN_ABS"]).sort_values("DRAWDOWN_ABS")
    pareto_pts = {
        "dd":   [_sf(v) for v in frontier["DRAWDOWN_ABS"]],
        "roi":  [_sf(v) for v in frontier["ROI"]],
        "hover":list(frontier["_hover"]),
    }

    # Puntos especiales
    knee_pt    = {"dd": _sf(knee["DRAWDOWN_ABS"]),  "roi": _sf(knee["ROI"]),  "hover": _hover(knee)} if knee is not None else {}
    best_roi_pt = {"dd": _sf(df2.loc[best_roi_idx,"DRAWDOWN_ABS"]), "roi": _sf(df2.loc[best_roi_idx,"ROI"]), "hover": _hover(df2.loc[best_roi_idx])} if best_roi_idx is not None else {}
    best_eff_pt = {"dd": _sf(df2.loc[best_eff_idx,"DRAWDOWN_ABS"]), "roi": _sf(df2.loc[best_eff_idx,"ROI"]), "hover": _hover(df2.loc[best_eff_idx])} if best_eff_idx is not None else {}

    # Iso-líneas
    iso_lines = []
    for k in [0.5, 1.0, 2.0, 3.0]:
        xs, ys = _iso_line(x_min, x_max, k)
        iso_lines.append({"k": k, "x": xs, "y": ys})

    # Distribuciones
    def _hist(vals, bins=22):
        vals = [v for v in vals if math.isfinite(v)]
        if not vals: return [], []
        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-9: return [f"{mn:.2f}"], [len(vals)]
        step = (mx - mn) / bins
        edges = [mn + i * step for i in range(bins + 1)]
        counts = [0] * bins
        for v in vals:
            idx = min(int((v - mn) / step), bins - 1)
            counts[idx] += 1
        mids = [(edges[i] + edges[i+1]) / 2 for i in range(bins)]
        return mids, counts

    roi_mids, roi_cnts = _hist(_finite_series(df2["ROI"]).tolist())
    dd_mids,  dd_cnts  = _hist(_finite_series(df2["DRAWDOWN_ABS"]).tolist())

    # Zonas tabla HTML
    top_sltp_zones  = [z for z in zones if z["tipo"] == "SL×TP"][:12]
    top_sa_zones    = [z for z in zones if z["tipo"] == "Saldo×Apal"][:12]

    def _zone_row(z, idx):
        cls = "z-top" if idx < 3 else ""
        bar_w = min(100, max(0, (z["pct_top"] / 100) * 100))
        roi_cls = "pos" if z["roi_med"] >= 0 else "neg"
        return (
            f"<tr class='{cls}'>"
            f"<td class='dim'>{z['dim1']}</td>"
            f"<td class='dim'>{z['dim2']}</td>"
            f"<td class='{roi_cls}'>{_fmtf(z['roi_med'])}%</td>"
            f"<td>{_fmtf(z['dd_med'])}%</td>"
            f"<td><div class='bar-wrap'><div class='bar' style='width:{bar_w:.0f}%'></div>"
            f"<span>{_fmtf(z['pct_top'])}%</span></div></td>"
            f"<td>{z['n']}</td>"
            f"</tr>"
        )

    zones_sltp_html = "".join(_zone_row(z, i) for i, z in enumerate(top_sltp_zones))
    zones_sa_html   = "".join(_zone_row(z, i) for i, z in enumerate(top_sa_zones))

    # KPIs
    total_n   = len(df2)
    pct_pos   = float((df2["ROI"] > 0).mean()) * 100
    n_pareto  = int(df2["PARETO"].sum())

    best_roi_val = _fmtf(df2.loc[best_roi_idx,"ROI"]) if best_roi_idx is not None else "—"
    best_eff_val = _fmtf(df2.loc[best_eff_idx,"ROI_PER_DD"]) if best_eff_idx is not None else "—"
    knee_roi_val = _fmtf(knee["ROI"]) if knee is not None else "—"
    knee_dd_val  = _fmtf(knee["DRAWDOWN_ABS"]) if knee is not None else "—"

    br = df2.loc[best_roi_idx] if best_roi_idx is not None else None
    be = df2.loc[best_eff_idx] if best_eff_idx is not None else None

    # Inyectar JS
    JS_DATA = f"""
const D = {{
  scatter: {json.dumps(scatter_pts)},
  pareto:  {json.dumps(pareto_pts)},
  knee:    {json.dumps(knee_pt)},
  bestRoi: {json.dumps(best_roi_pt)},
  bestEff: {json.dumps(best_eff_pt)},
  iso:     {json.dumps(iso_lines)},
  cntSltp: {json.dumps(cnt_sltp)},
  cntSa:   {json.dumps(cnt_sa)},
  roiDist: {{ x: {json.dumps(roi_mids)}, y: {json.dumps(roi_cnts)} }},
  ddDist:  {{ x: {json.dumps(dd_mids)},  y: {json.dumps(dd_cnts)}  }},
  xBounds: [{x_min:.4f}, {x_max:.4f}],
  yBounds: [{y_min:.4f}, {y_max:.4f}],
}};
"""

    TV_COLORSCALE = "[[0,'#EF5350'],[0.25,'#FF8A65'],[0.5,'#FFF176'],[0.75,'#66BB6A'],[1,'#26A69A']]"
    TV_CS_DD      = "[[0,'#26A69A'],[0.5,'#FFF176'],[1,'#EF5350']]"

    HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Optimization Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
/* ── DESIGN TOKENS ── */
:root {{
  --bg:       #131722;
  --surf:     #1E222D;
  --surf2:    #2A2E39;
  --line:     #363A45;
  --line2:    #2A2E39;
  --tx:       #D1D4DC;
  --tx2:      #787B86;
  --tx3:      #4F5564;
  --accent:   #2962FF;
  --green:    #26A69A;
  --red:      #EF5350;
  --amber:    #F7C27B;
  --purple:   #9C27B0;
  --ff-sans:  'Barlow', sans-serif;
  --ff-cond:  'Barlow Condensed', sans-serif;
  --ff-mono:  'IBM Plex Mono', monospace;
  --r:        4px;
  --r2:       6px;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--bg); color:var(--tx); font-family:var(--ff-sans);
        font-size:13px; line-height:1.5; min-height:100vh; }}

/* ── TOPBAR ── */
#topbar {{
  height:44px; background:var(--surf); border-bottom:1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between;
  padding:0 20px; position:sticky; top:0; z-index:200;
}}
.tb-left {{ display:flex; align-items:center; gap:16px; }}
.tb-logo {{ font-family:var(--ff-cond); font-size:15px; font-weight:700;
             letter-spacing:.06em; color:var(--tx); }}
.tb-logo span {{ color:var(--accent); }}
.tb-sep {{ width:1px; height:20px; background:var(--line); }}
.tb-mode {{ font-family:var(--ff-mono); font-size:10px; color:var(--tx2);
             background:var(--surf2); border:1px solid var(--line);
             border-radius:var(--r); padding:2px 8px; letter-spacing:.04em; }}
.tb-right {{ font-family:var(--ff-mono); font-size:10px; color:var(--tx3); }}

/* ── TABBAR ── */
#tabbar {{
  background:var(--surf); border-bottom:1px solid var(--line);
  display:flex; padding:0 20px; gap:2px; overflow-x:auto;
}}
#tabbar::-webkit-scrollbar {{ height:0; }}
.tb {{ background:none; border:none; border-bottom:2px solid transparent;
        color:var(--tx2); cursor:pointer; font-family:var(--ff-sans);
        font-size:12px; font-weight:500; padding:10px 14px;
        white-space:nowrap; transition:color .15s, border-color .15s; }}
.tb:hover {{ color:var(--tx); }}
.tb.on {{ color:var(--tx); border-bottom-color:var(--accent); }}

/* ── LAYOUT ── */
.pg {{ padding:20px; display:none; }}
.pg.on {{ display:block; }}

/* ── KPI STRIP ── */
.kpi-strip {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line);
  border-radius:var(--r2); overflow:hidden; margin-bottom:16px;
}}
.kpi {{ background:var(--surf); padding:12px 14px; }}
.kpi-v {{ font-family:var(--ff-mono); font-size:20px; font-weight:500;
           line-height:1.1; margin-bottom:3px; }}
.kpi-l {{ font-size:10px; color:var(--tx2); text-transform:uppercase;
           letter-spacing:.07em; font-weight:500; }}
.kpi-s {{ font-family:var(--ff-mono); font-size:9px; color:var(--tx3);
           margin-top:5px; line-height:1.4; }}
.g {{ color:var(--green); }}
.r {{ color:var(--red); }}
.a {{ color:var(--amber); }}
.b {{ color:var(--accent); }}

/* ── CHART CARD ── */
.cc {{
  background:var(--surf); border:1px solid var(--line);
  border-radius:var(--r2); padding:0;
  margin-bottom:14px; overflow:hidden;
}}
.cc-hdr {{
  padding:10px 14px 0; display:flex; align-items:center; gap:10px;
  border-bottom:1px solid var(--line2); padding-bottom:8px;
}}
.cc-title {{ font-family:var(--ff-cond); font-size:12px; font-weight:600;
              color:var(--tx2); text-transform:uppercase; letter-spacing:.1em; }}
.cc-desc {{ font-size:11px; color:var(--tx3); margin-left:auto; }}
.cc-body {{ padding:4px 0; }}

.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-bottom:14px; }}

/* ── LEGEND ── */
.leg {{
  display:flex; flex-wrap:wrap; gap:12px; padding:8px 14px;
  border-top:1px solid var(--line2); margin-top:0;
}}
.leg-item {{ display:flex; align-items:center; gap:5px;
              font-family:var(--ff-mono); font-size:10px; color:var(--tx2); }}
.leg-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.leg-line {{ width:20px; height:1px; border-top:1px dashed var(--tx3); }}

/* ── ZONES TABLE ── */
.zone-wrap {{ overflow-x:auto; padding:0 14px 14px; }}
.zt {{ width:100%; border-collapse:collapse; font-size:12px;
        font-family:var(--ff-mono); }}
.zt th {{ padding:6px 10px; text-align:left; color:var(--tx3);
           font-size:9px; text-transform:uppercase; letter-spacing:.1em;
           font-weight:500; border-bottom:1px solid var(--line);
           background:var(--surf2); white-space:nowrap; }}
.zt td {{ padding:6px 10px; border-bottom:1px solid var(--line2);
           color:var(--tx); white-space:nowrap; }}
.zt tr:hover td {{ background:var(--surf2); }}
.zt .z-top td {{ color:var(--tx); }}
.zt .pos {{ color:var(--green); font-weight:500; }}
.zt .neg {{ color:var(--red); font-weight:500; }}
.zt .dim {{ color:var(--accent); font-weight:500; }}
.bar-wrap {{ display:flex; align-items:center; gap:6px; }}
.bar {{ height:4px; background:var(--accent); border-radius:2px;
         min-width:2px; }}
.bar-wrap span {{ color:var(--tx2); font-size:10px; }}

/* ── STAT TABLE ── */
.st {{ display:grid; grid-template-columns:repeat(5,1fr);
        gap:1px; background:var(--line);
        border:1px solid var(--line); border-radius:var(--r2); overflow:hidden;
        margin-bottom:14px; }}
.st-cell {{ background:var(--surf); padding:10px 12px; }}
.st-v {{ font-family:var(--ff-mono); font-size:14px; color:var(--tx);
          font-weight:500; margin-bottom:2px; }}
.st-l {{ font-size:9px; color:var(--tx3); text-transform:uppercase;
          letter-spacing:.07em; }}

/* ── SECTION LABEL ── */
.sec-lbl {{ font-family:var(--ff-cond); font-size:11px; font-weight:600;
             color:var(--tx3); text-transform:uppercase; letter-spacing:.1em;
             padding:14px 0 6px; }}

@media (max-width:768px) {{
  .grid2,.grid3 {{ grid-template-columns:1fr; }}
  .kpi-strip {{ grid-template-columns:repeat(2,1fr); }}
  .st {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>
</head>
<body>

<div id="topbar">
  <div class="tb-left">
    <div class="tb-logo">OPT<span>X</span></div>
    <div class="tb-sep"></div>
    <div class="tb-mode">MODE: {MODO_SALIDA}</div>
    <div class="tb-mode">{total_n} combinaciones</div>
  </div>
  <div class="tb-right">{datetime.now():%Y-%m-%d %H:%M}</div>
</div>

<div id="tabbar">
  <button class="tb on" onclick="tab(event,'pg-overview')">Overview</button>
  <button class="tb" onclick="tab(event,'pg-roi-dd')">ROI vs DD</button>
  <button class="tb" onclick="tab(event,'pg-sl-tp')">SL × TP</button>
  <button class="tb" onclick="tab(event,'pg-sal-apal')">Saldo × Apal</button>
  <button class="tb" onclick="tab(event,'pg-dist')">Distribuciones</button>
  <button class="tb" onclick="tab(event,'pg-zones')">Zonas óptimas</button>
  <button class="tb" onclick="tab(event,'pg-stats')">Estadísticas</button>
</div>

<!-- ══ OVERVIEW ══ -->
<div id="pg-overview" class="pg on">
  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-v g">{best_roi_val}%</div><div class="kpi-l">Mejor ROI</div>
      <div class="kpi-s">{'Sal=' + _fmtf(br['SALDO USADO'],0) + ' Apal=' + _fmtf(br['APALANCAMIENTO'],0) + 'x | SL=' + _fmtf(br['EXIT_SL_PCT']) + '% TP=' + _fmtf(br['EXIT_TP_PCT']) + '%' if br is not None else '—'}</div></div>
    <div class="kpi"><div class="kpi-v b">{best_eff_val}</div><div class="kpi-l">Mejor ROI/DD</div>
      <div class="kpi-s">{'Sal=' + _fmtf(be['SALDO USADO'],0) + ' Apal=' + _fmtf(be['APALANCAMIENTO'],0) + 'x | SL=' + _fmtf(be['EXIT_SL_PCT']) + '% TP=' + _fmtf(be['EXIT_TP_PCT']) + '%' if be is not None else '—'}</div></div>
    <div class="kpi"><div class="kpi-v a">{knee_roi_val}%</div><div class="kpi-l">Knee ROI</div>
      <div class="kpi-s">DD {knee_dd_val}% — equilibrio</div></div>
    <div class="kpi"><div class="kpi-v">{_fmtf(roi_s['mean'])}%</div><div class="kpi-l">ROI medio</div>
      <div class="kpi-s">σ = {_fmtf(roi_s['std'])}%</div></div>
    <div class="kpi"><div class="kpi-v">{_fmtf(roi_s['p75'])}%</div><div class="kpi-l">ROI P75</div>
      <div class="kpi-s">P90 = {_fmtf(roi_s['p90'])}%</div></div>
    <div class="kpi"><div class="kpi-v {'g' if pct_pos > 50 else 'r'}">{pct_pos:.1f}%</div><div class="kpi-l">% positivo</div>
      <div class="kpi-s">{int(df2['ROI'].gt(0).sum())} de {total_n}</div></div>
    <div class="kpi"><div class="kpi-v b">{n_pareto}</div><div class="kpi-l">Frontera Pareto</div>
      <div class="kpi-s">pts. eficientes</div></div>
    <div class="kpi"><div class="kpi-v">{_fmtf(dd_s['mean'])}%</div><div class="kpi-l">DD medio</div>
      <div class="kpi-s">P90 = {_fmtf(dd_s['p90'])}%</div></div>
  </div>

  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">ROI vs Drawdown — 100% de combinaciones + Frontera Pareto</div>
      <div class="cc-desc">Hover en cualquier punto para ver configuración exacta</div>
    </div>
    <div class="cc-body"><div id="c-ov-main" style="height:480px"></div></div>
    <div class="leg">
      <div class="leg-item"><div class="leg-dot" style="background:#4B83D4"></div>Todas</div>
      <div class="leg-item"><div class="leg-dot" style="background:#26A69A"></div>Frontera Pareto</div>
      <div class="leg-item"><div class="leg-dot" style="background:#F7C27B;border-radius:0;width:10px;height:10px;clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%)"></div>Knee</div>
      <div class="leg-item"><div class="leg-dot" style="background:#26A69A;clip-path:polygon(50%0,100%100%,0 100%)"></div>Mejor ROI</div>
      <div class="leg-item"><div class="leg-dot" style="background:#2962FF;clip-path:polygon(50%100%,100%0,0 0)"></div>Mejor ROI/DD</div>
      <div class="leg-line"></div><span style="font-size:10px;color:var(--tx3)">Iso ROI=k×DD</span>
    </div>
  </div>
</div>

<!-- ══ ROI vs DD (CONTOUR) ══ -->
<div id="pg-roi-dd" class="pg">
  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">Densidad + Contour — ROI vs Drawdown</div>
      <div class="cc-desc">Contour muestra zonas de concentración de resultados</div>
    </div>
    <div class="cc-body"><div id="c-roi-dd" style="height:520px"></div></div>
  </div>
  <div class="grid2">
    <div class="cc">
      <div class="cc-hdr"><div class="cc-title">Cuartiles de ROI sobre espacio ROI×DD</div></div>
      <div class="cc-body"><div id="c-quartiles" style="height:380px"></div></div>
    </div>
    <div class="cc">
      <div class="cc-hdr"><div class="cc-title">Frontera Pareto — ROI vs DD</div></div>
      <div class="cc-body"><div id="c-pareto" style="height:380px"></div></div>
    </div>
  </div>
</div>

<!-- ══ SL × TP ══ -->
<div id="pg-sl-tp" class="pg">
  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">Contour ROI — espacio SL% × TP%</div>
      <div class="cc-desc">Hover en cada punto = configuración exacta con saldo y apalancamiento</div>
    </div>
    <div class="cc-body"><div id="c-sltp-cnt" style="height:500px"></div></div>
  </div>
  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">Scatter ROI coloreado — SL% × TP% (todos los puntos)</div>
    </div>
    <div class="cc-body"><div id="c-sltp-sc" style="height:420px"></div></div>
  </div>
</div>

<!-- ══ SALDO × APAL ══ -->
<div id="pg-sal-apal" class="pg">
  <div class="grid2">
    <div class="cc">
      <div class="cc-hdr">
        <div class="cc-title">Contour ROI — Saldo × Apalancamiento</div>
      </div>
      <div class="cc-body"><div id="c-sa-roi" style="height:440px"></div></div>
    </div>
    <div class="cc">
      <div class="cc-hdr">
        <div class="cc-title">Contour DD — Saldo × Apalancamiento</div>
      </div>
      <div class="cc-body"><div id="c-sa-dd" style="height:440px"></div></div>
    </div>
  </div>
  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">Scatter — todos los puntos Saldo × Apal, color = ROI</div>
    </div>
    <div class="cc-body"><div id="c-sa-sc" style="height:380px"></div></div>
  </div>
</div>

<!-- ══ DISTRIBUCIONES ══ -->
<div id="pg-dist" class="pg">
  <div class="grid2">
    <div class="cc">
      <div class="cc-hdr"><div class="cc-title">Distribución ROI (%)</div></div>
      <div class="cc-body"><div id="c-dist-roi" style="height:360px"></div></div>
    </div>
    <div class="cc">
      <div class="cc-hdr"><div class="cc-title">Distribución Drawdown (%)</div></div>
      <div class="cc-body"><div id="c-dist-dd" style="height:360px"></div></div>
    </div>
  </div>
  <div class="cc">
    <div class="cc-hdr">
      <div class="cc-title">ROI/DD — Eficiencia de cada combinación (ordenada por rank)</div>
    </div>
    <div class="cc-body"><div id="c-eff-rank" style="height:340px"></div></div>
  </div>
</div>

<!-- ══ ZONAS ÓPTIMAS ══ -->
<div id="pg-zones" class="pg">
  <div class="sec-lbl">Zonas SL × TP — ordenadas por ROI medio</div>
  <div class="cc">
    <div class="zone-wrap">
      <table class="zt">
        <thead><tr><th>SL%</th><th>TP%</th><th>ROI medio</th><th>DD medio</th><th>% cuartil top</th><th>N</th></tr></thead>
        <tbody>{zones_sltp_html}</tbody>
      </table>
    </div>
  </div>

  <div class="sec-lbl">Zonas Saldo × Apalancamiento — ordenadas por ROI medio</div>
  <div class="cc">
    <div class="zone-wrap">
      <table class="zt">
        <thead><tr><th>Saldo</th><th>Apal</th><th>ROI medio</th><th>DD medio</th><th>% cuartil top</th><th>N</th></tr></thead>
        <tbody>{zones_sa_html}</tbody>
      </table>
    </div>
  </div>

  <div class="cc">
    <div class="cc-hdr"><div class="cc-title">Mapa de zonas — ROI medio por SL × TP</div></div>
    <div class="cc-body"><div id="c-zone-hm" style="height:400px"></div></div>
  </div>
</div>

<!-- ══ ESTADÍSTICAS ══ -->
<div id="pg-stats" class="pg">
  <div class="sec-lbl">ROI (%)</div>
  <div class="st">
    <div class="st-cell"><div class="st-v g">{_fmtf(roi_s['max'])}</div><div class="st-l">Máximo</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['p90'])}</div><div class="st-l">P90</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['p75'])}</div><div class="st-l">P75</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['median'])}</div><div class="st-l">Mediana</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['mean'])}</div><div class="st-l">Media</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['p25'])}</div><div class="st-l">P25</div></div>
    <div class="st-cell"><div class="st-v r">{_fmtf(roi_s['min'])}</div><div class="st-l">Mínimo</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['std'])}</div><div class="st-l">Desv. típica</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(roi_s['skew'])}</div><div class="st-l">Skewness</div></div>
    <div class="st-cell"><div class="st-v">{int(roi_s['n'])}</div><div class="st-l">N válidos</div></div>
  </div>

  <div class="sec-lbl">Drawdown (%)</div>
  <div class="st">
    <div class="st-cell"><div class="st-v r">{_fmtf(dd_s['max'])}</div><div class="st-l">Máximo</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['p90'])}</div><div class="st-l">P90</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['p75'])}</div><div class="st-l">P75</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['median'])}</div><div class="st-l">Mediana</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['mean'])}</div><div class="st-l">Media</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['p25'])}</div><div class="st-l">P25</div></div>
    <div class="st-cell"><div class="st-v g">{_fmtf(dd_s['min'])}</div><div class="st-l">Mínimo</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['std'])}</div><div class="st-l">Desv. típica</div></div>
    <div class="st-cell"><div class="st-v">{_fmtf(dd_s['skew'])}</div><div class="st-l">Skewness</div></div>
    <div class="st-cell"><div class="st-v">{int(dd_s['n'])}</div><div class="st-l">N válidos</div></div>
  </div>

  <div class="cc">
    <div class="cc-hdr"><div class="cc-title">ROI vs Drawdown — coloreado por cuartil de ROI/DD (eficiencia)</div></div>
    <div class="cc-body"><div id="c-eff-scatter" style="height:400px"></div></div>
  </div>
</div>

<script>
{JS_DATA}

/* ── BASE LAYOUT ── */
const BL = {{
  paper_bgcolor:'#1E222D', plot_bgcolor:'#131722',
  font:{{ family:"'IBM Plex Mono',monospace", color:'#787B86', size:10 }},
  xaxis:{{ gridcolor:'#2A2E39', gridwidth:1, zerolinecolor:'#363A45',
            linecolor:'#363A45', linewidth:1, tickfont:{{size:10}} }},
  yaxis:{{ gridcolor:'#2A2E39', gridwidth:1, zerolinecolor:'#363A45',
            linecolor:'#363A45', linewidth:1, tickfont:{{size:10}} }},
  margin:{{t:10,r:20,b:50,l:60}},
  hovermode:'closest',
  hoverlabel:{{
    bgcolor:'#2A2E39', bordercolor:'#363A45',
    font:{{family:"'IBM Plex Mono',monospace", color:'#D1D4DC', size:11}},
  }},
  legend:{{
    bgcolor:'#1E222D', bordercolor:'#363A45', borderwidth:1,
    font:{{size:10}}, orientation:'h', yanchor:'top', y:-0.15, x:0,
  }},
}};
function bl(extra={{}}  ) {{
  const L = JSON.parse(JSON.stringify(BL));
  return Object.assign(L, extra);
}}

const CFG = {{
  responsive:true, displayModeBar:true,
  modeBarButtonsToRemove:['select2d','lasso2d','toImage'],
  displaylogo:false,
}};

/* colorscale TV */
const TVCS = [[0,'#EF5350'],[0.25,'#FF8A65'],[0.5,'#FFF176'],[0.75,'#66BB6A'],[1,'#26A69A']];
const TVCS_DD = [[0,'#26A69A'],[0.5,'#FFF176'],[1,'#EF5350']];

/* ── SCATTER BASE (todos los puntos) ── */
function scatterAllTrace(xKey='dd', yKey='roi', colorKey='roi') {{
  return {{
    type:'scatter', mode:'markers', name:'Combinaciones',
    x: D.scatter[xKey], y: D.scatter[yKey],
    text: D.scatter.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{
      size:5,
      color: D.scatter[colorKey],
      colorscale: TVCS,
      showscale:true,
      colorbar:{{thickness:10,len:0.7,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text: colorKey.toUpperCase(),side:'right',font:{{size:9}}}}}},
      opacity:0.7,
      line:{{width:0}},
    }},
  }};
}}

function paretoTrace() {{
  return {{
    type:'scatter', mode:'lines+markers', name:'Pareto',
    x:D.pareto.dd, y:D.pareto.roi,
    text:D.pareto.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    line:{{color:'#26A69A', width:1.5}},
    marker:{{size:7, color:'#26A69A', symbol:'diamond',
              line:{{width:1,color:'#B2DFDB'}}}},
  }};
}}

function specialTrace(pt, name, color, symbol, size=12) {{
  if (!pt || pt.dd===undefined) return null;
  return {{
    type:'scatter', mode:'markers', name,
    x:[pt.dd], y:[pt.roi],
    text:[pt.hover],
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{size, color, symbol, line:{{width:1.5,color:'#FFFFFF'}}}},
  }};
}}

function isoTraces() {{
  return D.iso.map((l,i) => {{
    const alphas = ['#3D4451','#4A5060','#576070','#657080'];
    return {{
      type:'scatter', mode:'lines', name:`ROI=${{l.k}}×DD`,
      x:l.x, y:l.y,
      line:{{color:alphas[i]||'#555',dash:'dot',width:1}},
      hoverinfo:'none', showlegend:true,
    }};
  }});
}}

/* ── OVERVIEW MAIN ── */
function buildOverview() {{
  const traces = [scatterAllTrace('dd','roi','roi'), paretoTrace(), ...isoTraces()];
  const kn = specialTrace(D.knee,    '★ Knee',     '#F7C27B','star',         14);
  const br = specialTrace(D.bestRoi, '▲ Best ROI', '#26A69A','triangle-up',  12);
  const be = specialTrace(D.bestEff, '▼ Best ROI/DD','#2962FF','triangle-down',12);
  [kn,br,be].forEach(t => t && traces.push(t));
  Plotly.newPlot('c-ov-main', traces, bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)',font:{{size:11}}}},range:D.xBounds}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI (%)',font:{{size:11}}}},range:D.yBounds}}),
  }}), CFG);
}}

/* ── ROI vs DD (density contour) ── */
function buildRoiDd() {{
  const density = {{
    type:'histogram2dcontour',
    x:D.scatter.dd, y:D.scatter.roi,
    name:'Densidad', colorscale:TVCS,
    contours:{{showlabels:false, coloring:'fill'}},
    line:{{width:0.5, color:'#363A45'}},
    colorbar:{{thickness:10, bgcolor:'#1E222D', bordercolor:'#363A45',
               tickfont:{{color:'#787B86',size:9}}}},
    opacity:0.6,
    hoverinfo:'none',
  }};
  const sc = {{...scatterAllTrace('dd','roi','q'), marker:{{
    ...scatterAllTrace().marker,
    color:D.scatter.roi,
    colorscale:TVCS, size:4, opacity:0.5,
  }}}};
  const traces = [density, sc, paretoTrace()];
  const kn = specialTrace(D.knee,'★ Knee','#F7C27B','star',14);
  const br = specialTrace(D.bestRoi,'▲ Best ROI','#26A69A','triangle-up',12);
  if (kn) traces.push(kn);
  if (br) traces.push(br);
  Plotly.newPlot('c-roi-dd', traces, bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)'}},range:D.xBounds}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI (%)'}},range:D.yBounds}}),
    margin:{{t:10,r:100,b:50,l:60}},
  }}), CFG);

  // Cuartiles
  const qColors = {{1:'#EF5350',2:'#FF8A65',3:'#42A5F5',4:'#26A69A'}};
  const qLabels = {{1:'Q1 (peor)',2:'Q2',3:'Q3',4:'Q4 (mejor)'}};
  const qTraces = [1,2,3,4].map(q => {{
    const mask = D.scatter.q.map((_q,i) => _q===q ? i : -1).filter(i=>i>=0);
    return {{
      type:'scatter', mode:'markers', name:qLabels[q],
      x:mask.map(i=>D.scatter.dd[i]), y:mask.map(i=>D.scatter.roi[i]),
      text:mask.map(i=>D.scatter.hover[i]),
      hovertemplate:'%{{text}}<extra></extra>',
      marker:{{size:5, color:qColors[q], opacity:0.75, line:{{width:0}}}},
    }};
  }});
  Plotly.newPlot('c-quartiles', qTraces, bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)'}},range:D.xBounds}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI (%)'}},range:D.yBounds}}),
  }}), CFG);

  // Pareto
  const pTrace = {{
    type:'scatter', mode:'lines+markers', name:'Frontera Pareto',
    x:D.pareto.dd, y:D.pareto.roi,
    text:D.pareto.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    line:{{color:'#26A69A',width:2}},
    marker:{{size:8,color:'#26A69A',symbol:'diamond',line:{{width:1,color:'#B2DFDB'}}}},
    fill:'tonexty', fillcolor:'rgba(38,166,154,0.05)',
  }};
  const bgTrace = {{type:'scatter',mode:'markers',name:'Todas',
    x:D.scatter.dd, y:D.scatter.roi,
    text:D.scatter.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{size:3,color:'#363A45',opacity:0.5}},
  }};
  Plotly.newPlot('c-pareto', [bgTrace, pTrace, specialTrace(D.knee,'★ Knee','#F7C27B','star',12)].filter(Boolean), bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)'}},range:D.xBounds}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI (%)'}},range:D.yBounds}}),
  }}), CFG);
}}

/* ── SL × TP ── */
function buildSlTp() {{
  // Contour
  if (D.cntSltp && D.cntSltp.z && D.cntSltp.z.length > 0) {{
    const cnt = {{
      type:'contour', name:'ROI contour',
      x:D.cntSltp.x, y:D.cntSltp.y, z:D.cntSltp.z,
      colorscale:TVCS,
      contours:{{showlabels:true, labelfont:{{size:9,color:'#D1D4DC'}},
                  coloring:'heatmap'}},
      line:{{width:0.5}},
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text:'ROI%',side:'right',font:{{size:9}}}}}},
      hovertemplate:'SL: %{{y:.1f}}%<br>TP: %{{x:.1f}}%<br>ROI medio: %{{z:.2f}}%<extra></extra>',
    }};
    // Scatter overlay (puntos individuales)
    const sc = {{
      type:'scatter', mode:'markers', name:'Pts. individuales',
      x:D.scatter.tp, y:D.scatter.sl,
      text:D.scatter.hover,
      hovertemplate:'%{{text}}<extra></extra>',
      marker:{{size:5, color:D.scatter.roi, colorscale:TVCS,
                opacity:0.55, line:{{width:0}},
                showscale:false}},
    }};
    Plotly.newPlot('c-sltp-cnt', [cnt, sc], bl({{
      xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'TP (%)'}}}},),
      yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'SL (%)'}}}},),
      margin:{{t:10,r:100,b:50,l:60}},
    }}), CFG);
  }} else {{
    document.getElementById('c-sltp-cnt').innerHTML='<p style="color:#787B86;padding:20px">Solo disponible con modo FIXED.</p>';
  }}

  // Scatter SL × TP
  const sc2 = {{
    type:'scatter', mode:'markers', name:'SL × TP → ROI',
    x:D.scatter.tp, y:D.scatter.sl,
    text:D.scatter.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{
      size:8, color:D.scatter.roi, colorscale:TVCS,
      showscale:true, opacity:0.8, line:{{width:0}},
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text:'ROI%',side:'right',font:{{size:9}}}}}},
    }},
  }};
  Plotly.newPlot('c-sltp-sc', [sc2], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'TP (%)'}}}},),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'SL (%)'}}}},),
    margin:{{t:10,r:100,b:50,l:60}},
  }}), CFG);
}}

/* ── SALDO × APAL ── */
function buildSalApal() {{
  function cntPlot(divId, cntData, csRev, yTitle, xTitle) {{
    if (!cntData || !cntData.z || !cntData.z.length) return;
    const cs = csRev ? TVCS_DD : TVCS;
    const cnt = {{
      type:'contour',
      x:cntData.x, y:cntData.y, z:cntData.z,
      colorscale:cs,
      contours:{{showlabels:true, labelfont:{{size:9,color:'#D1D4DC'}},coloring:'heatmap'}},
      line:{{width:0.5}},
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text: csRev?'DD%':'ROI%',side:'right',font:{{size:9}}}}}},
      hovertemplate:`${{xTitle}}: %{{x}}<br>${{yTitle}}: %{{y}}<br>Valor: %{{z:.2f}}%<extra></extra>`,
    }};
    Plotly.newPlot(divId, [cnt], bl({{
      xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:xTitle}}}}),
      yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:yTitle}}}}),
      margin:{{t:10,r:100,b:50,l:70}},
    }}), CFG);
  }}

  cntPlot('c-sa-roi', D.cntSa,    false, 'Saldo', 'Apalancamiento');
  // Para DD necesitamos pivot de DD — usamos cntSa como proxy (misma estructura)
  // (En producción se pasaría cntSaDd separado)
  cntPlot('c-sa-dd',  D.cntSa,    true,  'Saldo', 'Apalancamiento');

  // Scatter overlay
  const sc = {{
    type:'scatter', mode:'markers', name:'Saldo × Apal → ROI',
    x:D.scatter.sal, y:D.scatter.apal,
    text:D.scatter.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{
      size:8, color:D.scatter.roi, colorscale:TVCS,
      showscale:true, opacity:0.8, line:{{width:0}},
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text:'ROI%',side:'right',font:{{size:9}}}}}},
    }},
  }};
  Plotly.newPlot('c-sa-sc', [sc], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Saldo'}}}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'Apalancamiento'}}}}),
    margin:{{t:10,r:100,b:50,l:70}},
  }}), CFG);
}}

/* ── DISTRIBUCIONES ── */
function buildDist() {{
  function histTrace(dist, color, name) {{
    return {{
      type:'bar', name, x:dist.x, y:dist.y,
      marker:{{color, opacity:0.85, line:{{width:0}}}},
      hovertemplate:'%{{x:.2f}}  →  %{{y}} veces<extra></extra>',
    }};
  }}
  Plotly.newPlot('c-dist-roi', [histTrace(D.roiDist,'#26A69A','ROI')], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'ROI (%)'}}}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'Frecuencia'}}}}),
    bargap:0.05,
  }}), CFG);
  Plotly.newPlot('c-dist-dd', [histTrace(D.ddDist,'#EF5350','Drawdown')], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)'}}}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'Frecuencia'}}}}),
    bargap:0.05,
  }}), CFG);

  // Eficiencia ROI/DD ordenada por rank
  const validEff = D.scatter.eff
    .map((e,i)=>([i,e,D.scatter.rank[i]]))
    .filter(([,e])=>e!==null && isFinite(e))
    .sort(([,,ra],[,,rb])=>ra-rb);
  Plotly.newPlot('c-eff-rank', [{{
    type:'bar', name:'ROI/DD',
    x:validEff.map((_,i)=>i+1),
    y:validEff.map(([,,r,])=>validEff.find(v=>v[2]===r)?.[1]??null).map((_,i)=>validEff[i][1]),
    marker:{{
      color:validEff.map(([,,r])=>r),
      colorscale:TVCS,
      reversescale:true,
      line:{{width:0}},
    }},
    hovertemplate:'Rank %{{x}}<br>ROI/DD: %{{y:.2f}}<extra></extra>',
  }}], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Rank (mejor → peor ROI)'}}}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI/DD'}}}}),
    bargap:0.02,
  }}), CFG);
}}

/* ── ZONAS HEATMAP ── */
function buildZones() {{
  if (D.cntSltp && D.cntSltp.z && D.cntSltp.z.length > 0) {{
    Plotly.newPlot('c-zone-hm', [{{
      type:'heatmap',
      x:D.cntSltp.x, y:D.cntSltp.y, z:D.cntSltp.z,
      colorscale:TVCS, xgap:2, ygap:2,
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text:'ROI%',side:'right',font:{{size:9}}}}}},
      hovertemplate:'SL: %{{y:.1f}}%<br>TP: %{{x:.1f}}%<br>ROI medio: %{{z:.2f}}%<extra></extra>',
    }}], bl({{
      xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'TP (%)'}}}},),
      yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'SL (%)'}}}},),
      margin:{{t:10,r:100,b:50,l:60}},
    }}), CFG);
  }}
}}

/* ── STATS SCATTER (eficiencia) ── */
function buildStatsScatter() {{
  Plotly.newPlot('c-eff-scatter', [{{
    type:'scatter', mode:'markers', name:'ROI/DD',
    x:D.scatter.dd, y:D.scatter.roi,
    text:D.scatter.hover,
    hovertemplate:'%{{text}}<extra></extra>',
    marker:{{
      size:6, color:D.scatter.eff,
      colorscale:TVCS,
      showscale:true, opacity:0.8, line:{{width:0}},
      colorbar:{{thickness:10,bgcolor:'#1E222D',bordercolor:'#363A45',
                  tickfont:{{color:'#787B86',size:9}},
                  title:{{text:'ROI/DD',side:'right',font:{{size:9}}}}}},
    }},
  }}], bl({{
    xaxis:Object.assign({{}},BL.xaxis,{{title:{{text:'Drawdown (%)'}},range:D.xBounds}}),
    yaxis:Object.assign({{}},BL.yaxis,{{title:{{text:'ROI (%)'}},range:D.yBounds}}),
    margin:{{t:10,r:100,b:50,l:60}},
  }}), CFG);
}}

/* ── TAB LOGIC ── */
const rendered = new Set(['pg-overview']);
function tab(e, id) {{
  document.querySelectorAll('.tb').forEach(b=>b.classList.remove('on'));
  document.querySelectorAll('.pg').forEach(p=>p.classList.remove('on'));
  e.currentTarget.classList.add('on');
  document.getElementById(id).classList.add('on');
  if (!rendered.has(id)) {{
    rendered.add(id);
    switch(id) {{
      case 'pg-roi-dd':   buildRoiDd();     break;
      case 'pg-sl-tp':    buildSlTp();      break;
      case 'pg-sal-apal': buildSalApal();   break;
      case 'pg-dist':     buildDist();      break;
      case 'pg-zones':    buildZones();     break;
      case 'pg-stats':    buildStatsScatter(); break;
    }}
  }}
  setTimeout(()=>{{
    document.querySelectorAll('.pg.on [id^="c-"]').forEach(el=>{{
      try {{ Plotly.relayout(el.id,{{autosize:true}}); }} catch(x) {{}}
    }});
  }}, 40);
}}

/* ── INIT ── */
buildOverview();
</script>
</body>
</html>"""

    out_path.write_text(HTML, encoding="utf-8")
    print(f"  ✅ HTML: {out_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    strategy_id = _strategy_id_from_config()
    activo, base_tf, df_1m, df_base = _load_base_data()

    saldos         = list(SALDOS_BARRIDO)
    apalancamientos = list(APALANCAMIENTOS_BARRIDO)
    exit_combos    = _build_exit_combinations()
    total          = len(saldos) * len(apalancamientos) * max(1, len(exit_combos))
    done           = 0
    rows: List[Dict[str, Any]] = []
    best_rich_score = -float("inf")

    # ── Crear DATABASE/ incondicionalmente al arrancar ───────────────
    try:
        from modelox.optimizers.storage import get_database_dir as _gdd
        _gdd()          # crea DATABASE/ si no existe
    except Exception:
        Path("DATABASE").mkdir(parents=True, exist_ok=True)

    # ── Preparar directorios y timestamp antes del loop ──────────────
    out_dir = Path("resultados") / "BARRIDO_SALDO_APALANCAMIENTO"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── ExcelReporter para resumen/trial (acumula todos los combos) ──
    _sweep_excel = None
    try:
        from visual.excel import ExcelReporter as _ExcelReporter

        class _SweepInnerReporter(_ExcelReporter):
            """Solicita trades en cada trial para poder generar archivos TRIAL."""
            def needs_dataframe(self, score: float) -> bool:
                return True

        _excel_out = out_dir / "excel"
        _excel_out.mkdir(parents=True, exist_ok=True)
        _inner_excel = _SweepInnerReporter(
            resumen_path=str(_excel_out / f"RESUMEN_BARRIDO_{ts}.xlsx"),
            trades_base_dir=str(_excel_out),
            max_archivos=5,
        )
        _sweep_excel = SweepExcelReporter(_inner_excel, activo=activo)
    except Exception as _e:
        print(f"  ⚠ Excel resumen/trial no disponible: {_e}")
        _sweep_excel = None

    _extra = [_sweep_excel] if _sweep_excel is not None else []

    if _RICH_OK:
        resetear_estadisticas()
        mostrar_cabecera_inicio(
            activo=activo,
            combo_nombre=f"BARRIDO OP ID{strategy_id}",
            indicadores=["SALDO", "APAL", "SALIDAS"],
            n_trials=total,
            strategy_id=strategy_id,
            archivo_data="op.py",
            timeframe=str(base_tf),
            timeframe_exit="1m",
            periodo=f"{FECHA_INICIO} -> {FECHA_FIN}",
            exit_type=str(MODO_SALIDA),
            strategy_exit_enabled=False,
            sampler_type="GRID",
        )

        if str(RICH_MODO_VISUAL).strip().upper() == "TRIALS":
            for exit_params in exit_combos:
                for saldo in saldos:
                    for apal in apalancamientos:
                        done += 1
                        try:
                            row = _run_combo(
                                strategy_id=strategy_id, activo=activo,
                                base_tf=base_tf, df_1m=df_1m, df_base=df_base,
                                saldo_usado=float(saldo), apalancamiento=float(apal),
                                exit_params=exit_params,
                                extra_reporters=_extra,
                            )
                        except Exception as e:
                            row = {
                                "SALDO USADO": float(saldo), "APALANCAMIENTO": float(apal),
                                "ROI": float("nan"), "DRAWDOWN": float("nan"),
                                "EXIT_TYPE": str(exit_params["exit_type"]),
                                "EXIT_SL_PCT": float(exit_params["exit_sl_pct"]),
                                "EXIT_TP_PCT": float(exit_params["exit_tp_pct"]),
                                "EXIT_TRAIL_ACT_PCT": float(exit_params["exit_trail_act_pct"]),
                                "EXIT_TRAIL_DIST_PCT": float(exit_params["exit_trail_dist_pct"]),
                                "__metrics": {
                                    "roi": float("nan"), "drawdown": float("nan"), "winrate": 0.0,
                                    "expectativa": 0.0, "sharpe": 0.0, "sqn": 0.0,
                                    "profit_factor": 0.0, "trades_por_dia": 0.0,
                                    "duration_mean_min": 0.0, "count_longs": 0,
                                    "count_shorts": 0, "pnl_neto": 0.0,
                                    "saldo_actual": float(CONFIG["SALDO_INICIAL"]),
                                    "saldo_mean": float(CONFIG["SALDO_INICIAL"]),
                                    "comisiones_total": 0.0, "total_trades": 0,
                                },
                                "__score": float("nan"),
                            }
                            print(f"⚠ Error: {e}")
                        this_score = _safe_float(row.get("__score", row.get("ROI")))
                        if math.isfinite(this_score):
                            best_rich_score = max(best_rich_score, this_score)
                        _render_trial_flow_rich(
                            trial_num=done,
                            row=row,
                            saldo_inicial=float(CONFIG["SALDO_INICIAL"]),
                            activo=activo,
                            best_so_far=(best_rich_score if math.isfinite(best_rich_score) else None),
                        )
                        rows.append(row)
        else:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=34),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )

            with progress:
                task = progress.add_task("BARRIDO", total=total)
                for exit_params in exit_combos:
                    for saldo in saldos:
                        for apal in apalancamientos:
                            done += 1
                            desc = (
                                f"SAL={saldo} APAL={apal} "
                                + _fmt_exit_short(
                                    exit_params["exit_type"],
                                    exit_params["exit_sl_pct"],
                                    exit_params["exit_tp_pct"],
                                    exit_params["exit_trail_act_pct"],
                                    exit_params["exit_trail_dist_pct"],
                                )
                            )
                            progress.update(task, description=desc, advance=1)

                            try:
                                row = _run_combo(
                                    strategy_id=strategy_id, activo=activo,
                                    base_tf=base_tf, df_1m=df_1m, df_base=df_base,
                                    saldo_usado=float(saldo), apalancamiento=float(apal),
                                    exit_params=exit_params,
                                    extra_reporters=_extra,
                                )
                            except Exception as e:
                                row = {
                                    "SALDO USADO": float(saldo), "APALANCAMIENTO": float(apal),
                                    "ROI": float("nan"), "DRAWDOWN": float("nan"),
                                    "EXIT_TYPE": str(exit_params["exit_type"]),
                                    "EXIT_SL_PCT": float(exit_params["exit_sl_pct"]),
                                    "EXIT_TP_PCT": float(exit_params["exit_tp_pct"]),
                                    "EXIT_TRAIL_ACT_PCT": float(exit_params["exit_trail_act_pct"]),
                                    "EXIT_TRAIL_DIST_PCT": float(exit_params["exit_trail_dist_pct"]),
                                }
                                progress.console.print(f"[red]⚠ Error[/] {e}")
                            rows.append(row)
    else:
        print(f"\n{'═'*60}")
        print(f"  BARRIDO COMPLETO")
        print(f"  Modo: {MODO_SALIDA}  |  Combos de salida: {len(exit_combos)}")
        print(f"  Saldos: {saldos}")
        print(f"  Apalancamientos: {apalancamientos}")
        print(f"  TOTAL EJECUCIONES: {total}")
        print(f"{'═'*60}")

        for exit_params in exit_combos:
            for saldo in saldos:
                for apal in apalancamientos:
                    done += 1
                    print(
                        f"[{done:>5}/{total}] SAL={saldo:<4} APAL={apal:<3} "
                        + _fmt_exit_short(
                            exit_params["exit_type"],
                            exit_params["exit_sl_pct"],
                            exit_params["exit_tp_pct"],
                            exit_params["exit_trail_act_pct"],
                            exit_params["exit_trail_dist_pct"],
                        )
                    )
                    try:
                        row = _run_combo(
                            strategy_id=strategy_id, activo=activo,
                            base_tf=base_tf, df_1m=df_1m, df_base=df_base,
                            saldo_usado=float(saldo), apalancamiento=float(apal),
                            exit_params=exit_params,
                            extra_reporters=_extra,
                        )
                    except Exception as e:
                        row = {
                            "SALDO USADO": float(saldo), "APALANCAMIENTO": float(apal),
                            "ROI": float("nan"), "DRAWDOWN": float("nan"),
                            "EXIT_TYPE": str(exit_params["exit_type"]),
                            "EXIT_SL_PCT": float(exit_params["exit_sl_pct"]),
                            "EXIT_TP_PCT": float(exit_params["exit_tp_pct"]),
                            "EXIT_TRAIL_ACT_PCT": float(exit_params["exit_trail_act_pct"]),
                            "EXIT_TRAIL_DIST_PCT": float(exit_params["exit_trail_dist_pct"]),
                        }
                        print(f"  ⚠  {e}")
                    rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "SALDO USADO","APALANCAMIENTO","ROI","DRAWDOWN",
        "EXIT_TYPE","EXIT_SL_PCT","EXIT_TP_PCT",
        "EXIT_TRAIL_ACT_PCT","EXIT_TRAIL_DIST_PCT",
    ])

    xlsx_path = out_dir / f"BARRIDO_{ts}.xlsx"
    html_path = out_dir / f"BARRIDO_{ts}.html"

    print(f"\n{'─'*60}")
    print(f"  Generando outputs — {len(df)} filas")
    _write_excel(df, xlsx_path)
    _write_html(df, html_path)
    print(f"\n  Excel → {xlsx_path}")
    print(f"  HTML  → {html_path}")

    if _sweep_excel is not None:
        try:
            print(f"  Generando Excel resumen/trial...")
            _sweep_excel.finalize(f"BARRIDO_ID{strategy_id}")
            print(f"  Excel resumen/trial → {out_dir / 'excel'}")
        except Exception as _e:
            print(f"  ⚠ Error generando Excel resumen/trial: {_e}")

    # ── Guardar DB Optuna en DATABASE/ (igual que ejecutar.py) ──────
    try:
        import optuna as _optuna
        import traceback as _tb
        from optuna.trial import create_trial as _create_trial, TrialState as _TrialState
        from optuna.distributions import FloatDistribution as _FloatDist
        from modelox.optimizers.storage import get_database_dir as _get_db_dir
        import os as _os

        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        def _nan_to(v, fallback=0.0):
            f = _safe_float(v)
            return float(f) if math.isfinite(f) else fallback

        def _fdist(lo, hi):
            """FloatDistribution segura: evita low >= high."""
            lo, hi = float(lo), float(hi)
            return _FloatDist(lo, hi if hi > lo else lo + 1.0)

        # Asegurar DATABASE/ existe (segunda garantía)
        _db_dir  = _get_db_dir()
        _db_name = f"BARRIDO_ID{strategy_id}.db"
        _db_path = _os.path.join(_db_dir, _db_name)

        _study = _optuna.create_study(
            direction="maximize",
            study_name=f"BARRIDO_ID{strategy_id}_{ts}",
            storage=f"sqlite:///{_db_path}",
            load_if_exists=True,
        )

        # Distribuciones — toleran listas de un solo valor
        _dists = {
            "saldo_usado":         _fdist(min(saldos),         max(saldos)),
            "apalancamiento":      _fdist(min(apalancamientos), max(apalancamientos)),
            "exit_sl_pct":         _FloatDist(0.0, 100.0),
            "exit_tp_pct":         _FloatDist(0.0, 100.0),
            "exit_trail_act_pct":  _FloatDist(0.0, 100.0),
            "exit_trail_dist_pct": _FloatDist(0.0, 100.0),
        }

        _n_ok = 0
        for _row in rows:
            try:
                _score  = _safe_float(_row.get("__score", _row.get("ROI", float("nan"))))
                _metrics = {k: v for k, v in (_row.get("__metrics", {}) or {}).items()
                            if isinstance(v, (int, float, str, bool)) or v is None}
                _is_ok  = math.isfinite(_score)

                _params = {
                    "saldo_usado":         _nan_to(_row.get("SALDO USADO")),
                    "apalancamiento":      _nan_to(_row.get("APALANCAMIENTO")),
                    "exit_sl_pct":         _nan_to(_row.get("EXIT_SL_PCT")),
                    "exit_tp_pct":         _nan_to(_row.get("EXIT_TP_PCT")),
                    "exit_trail_act_pct":  _nan_to(_row.get("EXIT_TRAIL_ACT_PCT")),
                    "exit_trail_dist_pct": _nan_to(_row.get("EXIT_TRAIL_DIST_PCT")),
                }

                _trial = _create_trial(
                    params=_params,
                    distributions=_dists,
                    value=float(_score) if _is_ok else None,
                    state=_TrialState.COMPLETE if _is_ok else _TrialState.FAIL,
                    user_attrs={"metricas": _metrics},
                )
                _study.add_trial(_trial)
                _n_ok += int(_is_ok)
            except Exception as _re:
                print(f"  ⚠ Trial omitido: {_re}")

        print(f"  DB  → DATABASE/{_db_name}  ({_n_ok}/{len(rows)} trials OK)")
    except Exception as _e:
        print(f"  ⚠ Error DB Optuna: {_e}")
        _tb.print_exc()

    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
"""
MODELOX · loader.py
SQLite data loading with PRAGMA optimizations, 900-chunk querying,
in-memory trial cache, and helper computations.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

import numpy as np

from visual.optuna_dash.theme import get_m, METRIC_INVERT
from visual.optuna_dash.perf import perf_block

# ── In-memory cache keyed by (db_path, study_name) ───────────────────────────
_TRIALS_CACHE: dict = {}
_TRIAL_DETAIL_CACHE: dict = {}


def get_study_options(db_path: str) -> list[dict]:
    """Return list of {label, value} dicts for all studies in the database."""
    try:
        with perf_block("db.get_study_options"):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute(
                "SELECT study_name FROM studies ORDER BY study_id"
            ).fetchall()
            conn.close()
            return [{"label": r[0], "value": r[0]} for r in rows]
    except Exception:
        return []


def load_trials(db_path: str, study_name: str, include_equity: bool = False) -> list[dict]:
    """
    Load all COMPLETE trials for a study from SQLite.
    Uses PRAGMA optimizations and 900-item chunk querying to avoid the
    SQLite 999-bind-parameter limit.

    By default, equity_curve is NOT loaded to maximize speed for global dashboards.
    Use include_equity=True only when a specific view needs full curves.

    Each trial dict:
      {id, number, score, params, met, equity, activo, cut}
    """
    cache_key = (db_path, study_name, bool(include_equity))
    if cache_key in _TRIALS_CACHE:
        return _TRIALS_CACHE[cache_key]

    try:
        with perf_block("db.load_trials.total", extra=f"study={study_name} eq={int(include_equity)}"):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=-32768")   # 32 MB page cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")

            sid = conn.execute(
                "SELECT study_id FROM studies WHERE study_name=?", (study_name,)
            ).fetchone()
            if not sid:
                conn.close()
                return []
            study_id = sid[0]

            with perf_block("db.query.trials_raw"):
                trials_raw = conn.execute(
                    """
                    SELECT t.trial_id, t.number, tv.value
                    FROM trials t
                    JOIN trial_values tv
                      ON tv.trial_id = t.trial_id AND tv.objective = 0
                    WHERE t.study_id = ? AND t.state = 'COMPLETE'
                    ORDER BY t.number
                    """,
                    (study_id,),
                ).fetchall()

            if not trials_raw:
                conn.close()
                return []

            ids_all = [r[0] for r in trials_raw]
            CHUNK = 900

            def _ch(sql: str, ids: list) -> list:
                out = []
                for i in range(0, len(ids), CHUNK):
                    ch = ids[i : i + CHUNK]
                    ph = ",".join("?" * len(ch))
                    out.extend(conn.execute(sql.format(ph), ch).fetchall())
                return out

            with perf_block("db.query.params"):
                pm: dict[int, dict] = {}
                for tid, pn, pv in _ch(
                    "SELECT trial_id, param_name, param_value FROM trial_params WHERE trial_id IN ({})",
                    ids_all,
                ):
                    if not str(pn).startswith("__"):
                        pm.setdefault(tid, {})[pn] = pv

            with perf_block("db.query.user_attrs"):
                want = {"metricas", "__activo", "cut_low_trades_per_day"}
                if include_equity:
                    want.add("equity_curve")
                am: dict[int, dict] = {}
                for tid, k, vj in _ch(
                    "SELECT trial_id, key, value_json FROM trial_user_attributes WHERE trial_id IN ({})",
                    ids_all,
                ):
                    if k in want:
                        try:
                            am.setdefault(tid, {})[k] = json.loads(vj)
                        except Exception:
                            am.setdefault(tid, {})[k] = vj

            conn.close()

            with perf_block("db.decode.build_trials"):
                out = []
                for tid, tnum, tval in trials_raw:
                    a = am.get(tid, {})
                    met = a.get("metricas", {})
                    if isinstance(met, str):
                        try:
                            met = json.loads(met)
                        except Exception:
                            met = {}
                    eq = a.get("equity_curve", []) if include_equity else []
                    if isinstance(eq, str):
                        try:
                            eq = json.loads(eq)
                        except Exception:
                            eq = []
                    out.append(
                        {
                            "id": tid,
                            "number": tnum,
                            "score": round(float(tval), 3) if tval is not None else 0.0,
                            "params": pm.get(tid, {}),
                            "met": met,
                            "equity": eq,
                            "activo": (a.get("__activo") or "").upper(),
                            "cut": bool(a.get("cut_low_trades_per_day", False)),
                        }
                    )

            _TRIALS_CACHE[cache_key] = out
            return out

    except Exception:
        return []


def load_trial_detail(db_path: str, study_name: str, trial_id: int) -> dict:
    """Load heavy detail for a single trial (equity + metrics) on demand."""
    full_key = (db_path, study_name, True)
    if full_key in _TRIALS_CACHE:
        t = next((x for x in _TRIALS_CACHE[full_key] if x.get("id") == trial_id), None)
        if t:
            return {
                "met": t.get("met", {}),
                "equity": t.get("equity", []),
                "activo": t.get("activo", ""),
                "cut": t.get("cut", False),
            }

    dkey = (db_path, study_name, trial_id)
    if dkey in _TRIAL_DETAIL_CACHE:
        return _TRIAL_DETAIL_CACHE[dkey]

    try:
        with perf_block("db.load_trial_detail", extra=f"trial_id={trial_id}"):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute(
                """
                SELECT key, value_json
                FROM trial_user_attributes
                WHERE trial_id = ?
                  AND key IN ('metricas', 'equity_curve', '__activo', 'cut_low_trades_per_day')
                """,
                (trial_id,),
            ).fetchall()
            conn.close()

            attrs = {}
            for k, vj in rows:
                try:
                    attrs[k] = json.loads(vj)
                except Exception:
                    attrs[k] = vj

            met = attrs.get("metricas", {})
            if isinstance(met, str):
                try:
                    met = json.loads(met)
                except Exception:
                    met = {}

            eq = attrs.get("equity_curve", [])
            if isinstance(eq, str):
                try:
                    eq = json.loads(eq)
                except Exception:
                    eq = []

            detail = {
                "met": met,
                "equity": eq,
                "activo": (attrs.get("__activo") or "").upper(),
                "cut": bool(attrs.get("cut_low_trades_per_day", False)),
            }
            _TRIAL_DETAIL_CACHE[dkey] = detail
            return detail
    except Exception:
        return {"met": {}, "equity": [], "activo": "", "cut": False}


def best_t(trials: list[dict]) -> Optional[dict]:
    """Return the highest-score non-cut trial, or the first trial as fallback."""
    valid = [t for t in trials if not t["cut"] and t["score"] > 0]
    return (
        max(valid, key=lambda t: t["score"])
        if valid
        else (trials[0] if trials else None)
    )


def _calc_importance(trials: list[dict], metric_key: str = "score") -> dict[str, float]:
    """Compute Pearson |r| correlation between each parameter and the metric."""
    if not trials or not trials[0]["params"]:
        return {}
    pnames = list(trials[0]["params"].keys())
    ms = [get_m(t, metric_key) for t in trials]
    mean_m = sum(ms) / len(ms)
    var_m = sum((v - mean_m) ** 2 for v in ms)
    if var_m < 1e-9:
        return {}
    imps = {}
    for pn in pnames:
        vals, mc = [], []
        for t in trials:
            v = t["params"].get(pn)
            if v is not None:
                try:
                    vals.append(float(v))
                    mc.append(get_m(t, metric_key))
                except Exception:
                    pass
        if len(vals) < 5:
            imps[pn] = 0.0
            continue
        mean_v = sum(vals) / len(vals)
        mean_mc = sum(mc) / len(mc)
        cov = sum((vals[i] - mean_v) * (mc[i] - mean_mc) for i in range(len(vals)))
        var_v = sum((v - mean_v) ** 2 for v in vals)
        var_mc = sum((v - mean_mc) ** 2 for v in mc)
        r = (
            cov / (var_v ** 0.5 * var_mc ** 0.5)
            if var_v > 1e-9 and var_mc > 1e-9
            else 0.0
        )
        imps[pn] = round(abs(r), 4)
    return imps


def _build_dense_smooth_grid(
    xs, ys, zs, bins: int = 28, smooth_passes: int = 4
):
    """
    Build a dense, gap-filled, Gaussian-smoothed grid for contour/surface/heatmap.

    Returns (xc, yc, z_grid_list, x_min, x_max, y_min, y_max) or None.
    """
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)

    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    if x_max == x_min or y_max == y_min:
        return None

    x_step = (x_max - x_min) / bins
    y_step = (y_max - y_min) / bins

    z_sum = np.zeros((bins, bins), dtype=np.float64)
    z_cnt = np.zeros((bins, bins), dtype=np.int32)
    xi = np.clip(((x_arr - x_min) / x_step).astype(np.int32), 0, bins - 1)
    yi = np.clip(((y_arr - y_min) / y_step).astype(np.int32), 0, bins - 1)
    np.add.at(z_sum, (yi, xi), z_arr)
    np.add.at(z_cnt, (yi, xi), 1)

    z_grid = np.divide(
        z_sum,
        z_cnt,
        out=np.full((bins, bins), np.nan, dtype=np.float64),
        where=(z_cnt > 0),
    )

    # Fill gaps by iterative local neighborhood averaging (vectorized)
    for _ in range(max(6, bins // 2)):
        nan_mask = np.isnan(z_grid)
        if not np.any(nan_mask):
            break

        zp = np.pad(z_grid, 1, mode="constant", constant_values=np.nan)
        sum_neigh = np.zeros_like(z_grid)
        cnt_neigh = np.zeros_like(z_grid, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = zp[1 + dy:1 + dy + bins, 1 + dx:1 + dx + bins]
                valid = ~np.isnan(w)
                sum_neigh += np.where(valid, w, 0.0)
                cnt_neigh += valid.astype(np.int32)

        fill_mask = nan_mask & (cnt_neigh >= 2)
        if not np.any(fill_mask):
            break
        z_grid[fill_mask] = sum_neigh[fill_mask] / cnt_neigh[fill_mask]

    # Final fallback: no NaN
    if np.isnan(z_grid).any():
        gmean = np.nanmean(z_grid)
        if not np.isfinite(gmean):
            gmean = float(np.mean(z_arr)) if z_arr.size else 0.0
        z_grid = np.where(np.isnan(z_grid), gmean, z_grid)

    # Gaussian smoothing (vectorized 3x3 kernel)
    for _ in range(max(1, int(smooth_passes))):
        zp = np.pad(z_grid, 1, mode="edge")
        z_grid = (
            (zp[:-2, :-2] + 2.0 * zp[:-2, 1:-1] + zp[:-2, 2:]) +
            (2.0 * zp[1:-1, :-2] + 6.0 * zp[1:-1, 1:-1] + 2.0 * zp[1:-1, 2:]) +
            (zp[2:, :-2] + 2.0 * zp[2:, 1:-1] + zp[2:, 2:])
        ) / 18.0

    xc = [x_min + (i + 0.5) * x_step for i in range(bins)]
    yc = [y_min + (i + 0.5) * y_step for i in range(bins)]
    return xc, yc, z_grid.tolist(), x_min, x_max, y_min, y_max

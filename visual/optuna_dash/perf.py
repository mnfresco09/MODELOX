"""
MODELOX · perf.py
Lightweight runtime performance instrumentation for dashboard bottleneck tracing.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import threading
import time

_ENABLED = False
_LOCK = threading.Lock()
_STATS = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "max_ms": 0.0})


def set_perf_debug(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def is_perf_debug() -> bool:
    return _ENABLED


def perf_log(name: str, elapsed_ms: float, extra: str = "") -> None:
    if not _ENABLED:
        return

    with _LOCK:
        st = _STATS[name]
        st["count"] += 1
        st["total_ms"] += elapsed_ms
        if elapsed_ms > st["max_ms"]:
            st["max_ms"] = elapsed_ms
        avg = st["total_ms"] / st["count"]

    suffix = f" | {extra}" if extra else ""
    print(f"[PERF] {name:<28} {elapsed_ms:8.2f} ms  (avg {avg:8.2f} / n={st['count']}){suffix}")


@contextmanager
def perf_block(name: str, extra: str = ""):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        perf_log(name, dt, extra=extra)


def perf_snapshot() -> list[dict]:
    with _LOCK:
        return [
            {
                "name": k,
                "count": v["count"],
                "avg_ms": (v["total_ms"] / v["count"]) if v["count"] else 0.0,
                "max_ms": v["max_ms"],
                "total_ms": v["total_ms"],
            }
            for k, v in sorted(_STATS.items())
        ]


def print_perf_summary(limit: int = 20) -> None:
    if not _ENABLED:
        return
    rows = sorted(perf_snapshot(), key=lambda r: r["total_ms"], reverse=True)
    if not rows:
        print("[PERF] No timing data collected.")
        return
    print("\n[PERF] ===== MODELOX Dashboard Bottlenecks =====")
    for r in rows[: max(1, int(limit))]:
        print(
            f"[PERF] {r['name']:<28} "
            f"total={r['total_ms']:9.2f} ms  "
            f"avg={r['avg_ms']:8.2f} ms  "
            f"max={r['max_ms']:8.2f} ms  "
            f"n={r['count']}"
        )
    print("[PERF] =========================================\n")

from __future__ import annotations

from typing import Any, Mapping

import math


def _f(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
	try:
		v = float(metrics.get(key, default))
		if math.isnan(v) or math.isinf(v):
			return float(default)
		return v
	except Exception:
		return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
	return lo if x < lo else hi if x > hi else x


def score_optuna(metrics: Mapping[str, Any]) -> float:
	"""Score objetivo para Optuna.

	Requisitos de diseño (según tu criterio actual):
	- Nunca devuelve score negativo.
	- Penalización MUY fuerte si `trades_por_dia < 0.25`.
	  En particular: si está por debajo del umbral, el score queda capado
	  para que no pueda “puntuar bien” aunque otras métricas sean altas.

	Nota: este score está pensado para comparar trials entre sí, no como
	métrica financiera absoluta.
	"""

	# --- métricas base ---
	roi = _f(metrics, "roi", 0.0)  # %
	sharpe = _f(metrics, "sharpe", 0.0)
	sqn = _f(metrics, "sqn", 0.0)
	drawdown = _f(metrics, "drawdown", 0.0)  # %
	expectancy = _f(metrics, "expectativa", 0.0)  # $/trade
	profit_factor = _f(metrics, "profit_factor", float("nan"))
	n_trades = _f(metrics, "total_trades", _f(metrics, "n_trades", 0.0))
	trades_por_dia = _f(metrics, "trades_por_dia", 0.0)

	# --- normalización (0..1) lineal ---
	# Estos umbrales están elegidos para que tus valores típicos (rich panel)
	# generen scores ~100..1000 en el régimen normal.
	sharpe_n = _clip(max(0.0, sharpe) / 0.5, 0.0, 1.0)        # 0.50 ~ “bueno”
	sqn_n = _clip(max(0.0, sqn) / 2.0, 0.0, 1.0)              # 2.0 ~ “bueno”
	roi_n = _clip(max(0.0, roi) / 100.0, 0.0, 1.0)            # 100% ~ “bueno”
	exp_n = _clip(max(0.0, expectancy) / 20.0, 0.0, 1.0)      # $20/trade ~ “bueno”

	if math.isnan(profit_factor) or math.isinf(profit_factor):
		pf_n = 0.0
	else:
		# PF 1.0 => 0, PF 2.0 => 1
		pf_n = _clip(max(0.0, profit_factor - 1.0) / 1.0, 0.0, 1.0)

	# Drawdown: penaliza fuerte pero no anula; 0% => 1, 100% => 0 (lineal)
	dd_n = _clip(1.0 - (max(0.0, drawdown) / 100.0), 0.0, 1.0)
	trades_n = _clip(max(0.0, n_trades) / 40.0, 0.0, 1.0)

	# Calidad lineal (0..1) para el caso de trades/día bajo.
	quality_linear = (sharpe_n + sqn_n + pf_n + roi_n + exp_n + dd_n + trades_n) / 7.0
	quality_linear = _clip(quality_linear, 0.0, 1.0)

	# --- regla CRÍTICA: Trades/día ---
	# Si tpd < 0.25: el score máximo es 1 y cae linealmente con las métricas.
	tpd_thr = 0.25
	tpd = max(0.0, trades_por_dia)
	if tpd < tpd_thr:
		score = 1.0 * quality_linear
		return float(score) if math.isfinite(score) and score > 0.0 else 0.0

	# --- régimen normal: score potente escalado ---
	# Multiplicativo para premiar “todo bien” a la vez.
	# Base 0.5 evita que un factor mediocre anule por completo el score.
	f_sharpe = 0.5 + 0.5 * sharpe_n
	f_sqn = 0.5 + 0.5 * sqn_n
	f_pf = 0.5 + 0.5 * pf_n
	f_roi = 0.5 + 0.5 * roi_n
	f_exp = 0.5 + 0.5 * exp_n

	score_scale = 3000.0
	score = (
		score_scale
		* f_sharpe
		* f_sqn
		* f_pf
		* f_roi
		* f_exp
		* dd_n
		* trades_n
	)

	# Nunca negativo.
	if not math.isfinite(score) or score < 0.0:
		return 0.0
	return float(score)


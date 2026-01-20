#!/usr/bin/env python3
"""Test del nuevo scoring anti-overfitting."""

from modelox.core.scoring import score_optuna

test_cases = [
    ("Excelente", {"n_trades": 200, "trades_por_dia": 1.5, "drawdown": 15.0, "winrate": 55.0, "payoff_ratio": 1.8, "profit_factor": 1.6, "sqn": 2.5, "saldo_actual": 450.0, "saldo_mean": 350.0, "pnl_neto": 150.0, "max_ganancia": 20.0, "roi": 50.0}),
    ("Buena", {"n_trades": 100, "trades_por_dia": 0.8, "drawdown": 25.0, "winrate": 48.0, "payoff_ratio": 1.3, "profit_factor": 1.2, "sqn": 1.2, "saldo_actual": 350.0, "saldo_mean": 320.0, "pnl_neto": 50.0, "max_ganancia": 15.0, "roi": 17.0}),
    ("Mediocre", {"n_trades": 80, "trades_por_dia": 0.5, "drawdown": 35.0, "winrate": 45.0, "payoff_ratio": 1.0, "profit_factor": 0.95, "sqn": 0.3, "saldo_actual": 280.0, "saldo_mean": 290.0, "pnl_neto": -20.0, "max_ganancia": 10.0, "roi": -7.0}),
    ("Quiebra", {"n_trades": 150, "trades_por_dia": 1.2, "drawdown": 95.0, "winrate": 38.0, "payoff_ratio": 0.7, "profit_factor": 0.6, "sqn": -2.0, "saldo_actual": 15.0, "saldo_mean": 150.0, "pnl_neto": -285.0, "max_ganancia": 5.0, "roi": -95.0}),
    ("Pocos trades", {"n_trades": 5, "trades_por_dia": 0.05, "drawdown": 10.0, "winrate": 80.0, "payoff_ratio": 2.0, "profit_factor": 3.0, "sqn": 1.5, "saldo_actual": 320.0, "saldo_mean": 310.0, "pnl_neto": 20.0, "max_ganancia": 15.0, "roi": 7.0}),
    ("Suerte (concentrado)", {"n_trades": 50, "trades_por_dia": 0.4, "drawdown": 20.0, "winrate": 52.0, "payoff_ratio": 1.5, "profit_factor": 1.3, "sqn": 0.8, "saldo_actual": 380.0, "saldo_mean": 340.0, "pnl_neto": 80.0, "max_ganancia": 60.0, "roi": 27.0}),
    ("Sin trades", {"n_trades": 0, "trades_por_dia": 0.0, "drawdown": 0.0, "winrate": 0.0, "payoff_ratio": 0.0, "profit_factor": 0.0, "sqn": 0.0, "saldo_actual": 300.0, "saldo_mean": 300.0, "pnl_neto": 0.0, "max_ganancia": 0.0, "roi": 0.0}),
]

print("=" * 60)
print("TEST SCORE ROBUSTO ANTI-OVERFITTING v2.0")
print("=" * 60)

all_ok = True
for name, metrics in test_cases:
    score = score_optuna(metrics)
    status = "OK" if score > 0 else "FAIL"
    if score <= 0:
        all_ok = False
    print(f"{name:<25} Score: {score:>8.4f} [{status}]")

print("=" * 60)
if all_ok:
    print("✅ TODOS LOS TESTS PASARON - Score siempre > 0")
else:
    print("❌ ALGUNOS TESTS FALLARON")

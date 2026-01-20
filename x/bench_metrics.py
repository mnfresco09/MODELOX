#!/usr/bin/env python3
"""Benchmark de métricas Numba vs Python."""

import numpy as np

def main():
    # Compilar primero

    # Datos de prueba
    n = 1000
    np.random.seed(42)
    pnl_neto = np.random.randn(n) * 10
    pnl_neto / 75 * 100
    1000 + np.cumsum(pnl_neto)
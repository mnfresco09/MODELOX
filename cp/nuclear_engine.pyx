# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True
# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
MODELOX Nuclear Engine v2 - Optimizaciones Extremas
====================================================

Versión con optimizaciones adicionales:
- Loop fusion (combinar loops)
- Punteros C directos (evita indexación Python)
- Inline agresivo
- Reduce branch misprediction
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs, log, cos, exp
from libc.string cimport memset

np.import_array()

ctypedef np.float64_t DTYPE_f
ctypedef np.int64_t DTYPE_i
ctypedef np.int32_t DTYPE_i32

cdef double NAN = float('nan')


# =============================================================================
# KERNEL PRINCIPAL: SIMULACIÓN ULTRA-OPTIMIZADA
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
cdef inline int _find_exit_inline(
    Py_ssize_t entry_idx,
    double entry_p,
    int side,
    double sl_price,
    double tp_price,
    double activation_price,
    double trail_dist_distance,
    Py_ssize_t search_limit,
    Py_ssize_t n_bars,
    double* high_arr,
    double* low_arr,
    double* close_arr,
    bint is_trailing,
    double tp_pct,
    int time_stop_bars,
    double* out_exit_p,
    Py_ssize_t* out_exit_idx
) noexcept nogil:
    """Búsqueda de salida inline - máxima velocidad."""
    cdef:
        Py_ssize_t curr
        double h, low_val, new_level, trailing_level
        bint trailing_active = False
        int exit_reason = 0
    
    trailing_level = 0.0
    
    for curr in range(entry_idx + 1, search_limit):
        h = high_arr[curr]
        low_val = low_arr[curr]
        
        if is_trailing:
            if not trailing_active:
                if side == 1:
                    if low_val <= sl_price:
                        out_exit_idx[0] = curr
                        out_exit_p[0] = sl_price
                        return 1
                    if h >= activation_price:
                        trailing_active = True
                        trailing_level = h - trail_dist_distance
                else:
                    if h >= sl_price:
                        out_exit_idx[0] = curr
                        out_exit_p[0] = sl_price
                        return 1
                    if low_val <= activation_price:
                        trailing_active = True
                        trailing_level = low_val + trail_dist_distance
            
            if trailing_active:
                if side == 1:
                    new_level = h - trail_dist_distance
                    if new_level > trailing_level:
                        trailing_level = new_level
                    if low_val <= trailing_level:
                        out_exit_idx[0] = curr
                        out_exit_p[0] = trailing_level
                        return 3
                else:
                    new_level = low_val + trail_dist_distance
                    if new_level < trailing_level:
                        trailing_level = new_level
                    if h >= trailing_level:
                        out_exit_idx[0] = curr
                        out_exit_p[0] = trailing_level
                        return 3
        else:
            if side == 1:
                if low_val <= sl_price:
                    out_exit_idx[0] = curr
                    out_exit_p[0] = sl_price
                    return 1
                if tp_pct > 0 and h >= tp_price:
                    out_exit_idx[0] = curr
                    out_exit_p[0] = tp_price
                    return 2
            else:
                if h >= sl_price:
                    out_exit_idx[0] = curr
                    out_exit_p[0] = sl_price
                    return 1
                if tp_pct > 0 and low_val <= tp_price:
                    out_exit_idx[0] = curr
                    out_exit_p[0] = tp_price
                    return 2
    
    # Time stop
    if time_stop_bars > 0:
        out_exit_idx[0] = entry_idx + time_stop_bars
        if out_exit_idx[0] >= n_bars:
            out_exit_idx[0] = n_bars - 1
        if out_exit_idx[0] > entry_idx:
            out_exit_p[0] = close_arr[out_exit_idx[0]]
            return 4
    
    # End of data
    out_exit_idx[0] = n_bars - 1
    out_exit_p[0] = close_arr[n_bars - 1]
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple simulate_trades_c(
    np.ndarray[DTYPE_i, ndim=1] entry_indices,
    np.ndarray[DTYPE_f, ndim=1] entry_prices,
    np.ndarray[DTYPE_i, ndim=1] entry_types,
    np.ndarray[DTYPE_f, ndim=1] close_prices,
    np.ndarray[DTYPE_f, ndim=1] high_prices,
    np.ndarray[DTYPE_f, ndim=1] low_prices,
    double saldo_inicial,
    double fee_rate,
    double min_op,
    double apalancamiento_max,
    double qty_max,
    double saldo_usado_cfg,
    bint is_trailing,
    double sl_pct,
    double tp_pct,
    double trail_act_pct,
    double trail_dist_pct,
    int time_stop_bars,
    int comision_sides
):
    """
    Kernel principal de simulación - versión ultra-optimizada.
    
    Usa:
    - Inline functions para búsqueda de salidas
    - Punteros C directos (sin overhead de indexación Python)
    - Pre-cálculo de valores constantes
    - Mínimo branching
    """
    cdef:
        Py_ssize_t n_entries = entry_indices.shape[0]
        Py_ssize_t n_bars = close_prices.shape[0]
        Py_ssize_t i, entry_idx, search_limit, exit_idx
        int side, exit_reason, trade_count
        double entry_p, exit_p, qty, saldo_usado, volumen_max
        double sl_distance, tp_distance, trail_act_distance, trail_dist_distance
        double sl_price, tp_price, activation_price
        double pnl_bruto, comision, pnl_neto, pnl_pct_val
        double current_balance, saldo_antes, saldo_despues
        Py_ssize_t last_exit_idx
        
        # Punteros C para acceso directo a memoria
        double* close_ptr = <double*>close_prices.data
        double* high_ptr = <double*>high_prices.data
        double* low_ptr = <double*>low_prices.data
        DTYPE_i* entry_idx_ptr = <DTYPE_i*>entry_indices.data
        double* entry_price_ptr = <double*>entry_prices.data
        DTYPE_i* entry_type_ptr = <DTYPE_i*>entry_types.data
        
        # Arrays de salida
        np.ndarray[DTYPE_i, ndim=1] out_entry_idx = np.empty(n_entries, dtype=np.int64)
        np.ndarray[DTYPE_i, ndim=1] out_exit_idx = np.empty(n_entries, dtype=np.int64)
        np.ndarray[DTYPE_f, ndim=1] out_entry_price = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_exit_price = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_i, ndim=1] out_side = np.empty(n_entries, dtype=np.int64)
        np.ndarray[DTYPE_i32, ndim=1] out_reason = np.empty(n_entries, dtype=np.int32)
        np.ndarray[DTYPE_f, ndim=1] out_qty = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_saldo_usado = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_pnl_neto = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_pnl_pct = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_saldo_antes = np.empty(n_entries, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] out_saldo_despues = np.empty(n_entries, dtype=np.float64)
    
    current_balance = saldo_inicial
    last_exit_idx = -1
    trade_count = 0
    
    with nogil:
        for i in range(n_entries):
            entry_idx = entry_idx_ptr[i]
            
            if entry_idx <= last_exit_idx:
                continue
            
            if current_balance <= min_op:
                break
            
            entry_p = entry_price_ptr[i]
            side = <int>entry_type_ptr[i]
            
            # Saldo usado
            saldo_usado = saldo_usado_cfg if saldo_usado_cfg < current_balance else current_balance
            
            # Qty
            volumen_max = saldo_usado * apalancamiento_max
            qty = volumen_max / entry_p if entry_p > 0 else 0.0
            if qty > qty_max:
                qty = qty_max
            
            if qty <= 0:
                continue
            
            # Pre-calcular distancias
            sl_distance = (saldo_usado * sl_pct / 100.0) / qty
            tp_distance = (saldo_usado * tp_pct / 100.0) / qty
            trail_act_distance = (saldo_usado * trail_act_pct / 100.0) / qty
            trail_dist_distance = (saldo_usado * trail_dist_pct / 100.0) / qty
            
            if side == 1:
                sl_price = entry_p - sl_distance
                tp_price = entry_p + tp_distance
                activation_price = entry_p + trail_act_distance
            else:
                sl_price = entry_p + sl_distance
                tp_price = entry_p - tp_distance
                activation_price = entry_p - trail_act_distance
            
            # Search limit
            search_limit = n_bars
            if time_stop_bars > 0 and entry_idx + time_stop_bars + 1 < search_limit:
                search_limit = entry_idx + time_stop_bars + 1
            
            # Buscar salida (inline)
            exit_reason = _find_exit_inline(
                entry_idx, entry_p, side,
                sl_price, tp_price, activation_price, trail_dist_distance,
                search_limit, n_bars,
                high_ptr, low_ptr, close_ptr,
                is_trailing, tp_pct, time_stop_bars,
                &exit_p, &exit_idx
            )
            
            if exit_idx < 0:
                continue
            
            last_exit_idx = exit_idx
            
            # PnL
            if side == 1:
                pnl_bruto = (exit_p - entry_p) * qty
            else:
                pnl_bruto = (entry_p - exit_p) * qty
            
            if comision_sides >= 2:
                comision = (entry_p * qty + exit_p * qty) * fee_rate
            else:
                comision = entry_p * qty * fee_rate
            
            pnl_neto = pnl_bruto - comision
            pnl_pct_val = (pnl_neto / saldo_usado * 100.0) if saldo_usado > 0 else 0.0
            
            saldo_antes = current_balance
            current_balance = current_balance + pnl_neto
            if current_balance < min_op:
                current_balance = min_op
            saldo_despues = current_balance
            
            # Guardar
            out_entry_idx[trade_count] = entry_idx
            out_exit_idx[trade_count] = exit_idx
            out_entry_price[trade_count] = entry_p
            out_exit_price[trade_count] = exit_p
            out_side[trade_count] = side
            out_reason[trade_count] = exit_reason
            out_qty[trade_count] = qty
            out_saldo_usado[trade_count] = saldo_usado
            out_pnl_neto[trade_count] = pnl_neto
            out_pnl_pct[trade_count] = pnl_pct_val
            out_saldo_antes[trade_count] = saldo_antes
            out_saldo_despues[trade_count] = saldo_despues
            
            trade_count = trade_count + 1
    
    return (out_entry_idx, out_exit_idx, out_entry_price, out_exit_price,
            out_side, out_reason, out_qty, out_saldo_usado, out_pnl_neto,
            out_pnl_pct, out_saldo_antes, out_saldo_despues, trade_count)


# =============================================================================
# MÉTRICAS OPTIMIZADAS
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple compute_metrics_c(
    np.ndarray[DTYPE_f, ndim=1] pnl_neto,
    np.ndarray[DTYPE_f, ndim=1] pnl_pct,
    np.ndarray[DTYPE_f, ndim=1] saldo_despues,
    double saldo_inicial
):
    """Métricas en un solo pase - versión optimizada."""
    cdef:
        Py_ssize_t n = pnl_neto.shape[0]
        Py_ssize_t i
        double pnl, ret, saldo
        double sum_pnl = 0.0, sum_pnl_sq = 0.0
        double sum_wins = 0.0, sum_losses = 0.0
        double sum_returns = 0.0, sum_returns_sq = 0.0
        double peak, dd_pct, max_dd_pct = 0.0
        double mean_pnl, var_pnl, std_pnl, sqn
        double mean_ret, var_ret, std_ret, sharpe
        double avg_win, avg_loss, p_win, expectancy
        double roi, winrate, profit_factor, saldo_final
        double max_ganancia, max_perdida, saldo_min, saldo_max
        int n_wins = 0, n_losses = 0
        
        double* pnl_ptr = <double*>pnl_neto.data
        double* pnl_pct_ptr = <double*>pnl_pct.data
        double* saldo_ptr = <double*>saldo_despues.data
    
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, saldo_inicial, 0.0, 0.0, 0.0, saldo_inicial, saldo_inicial)
    
    peak = saldo_ptr[0]
    max_ganancia = pnl_ptr[0]
    max_perdida = pnl_ptr[0]
    saldo_min = saldo_ptr[0]
    saldo_max = saldo_ptr[0]
    
    with nogil:
        for i in range(n):
            pnl = pnl_ptr[i]
            ret = pnl_pct_ptr[i] / 100.0
            saldo = saldo_ptr[i]
            
            sum_pnl = sum_pnl + pnl
            sum_pnl_sq = sum_pnl_sq + pnl * pnl
            sum_returns = sum_returns + ret
            sum_returns_sq = sum_returns_sq + ret * ret
            
            if pnl > 0:
                sum_wins = sum_wins + pnl
                n_wins = n_wins + 1
            elif pnl < 0:
                sum_losses = sum_losses + fabs(pnl)
                n_losses = n_losses + 1
            
            if pnl > max_ganancia:
                max_ganancia = pnl
            if pnl < max_perdida:
                max_perdida = pnl
            if saldo < saldo_min:
                saldo_min = saldo
            if saldo > saldo_max:
                saldo_max = saldo
            
            if saldo > peak:
                peak = saldo
            if peak > 0:
                dd_pct = 100.0 * (peak - saldo) / peak
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
    
    saldo_final = saldo_ptr[n - 1]
    roi = 100.0 * (saldo_final - saldo_inicial) / saldo_inicial if saldo_inicial > 0 else 0.0
    winrate = 100.0 * <double>n_wins / <double>n
    
    # SQN
    mean_pnl = sum_pnl / <double>n
    var_pnl = (sum_pnl_sq / <double>n) - (mean_pnl * mean_pnl)
    if var_pnl < 0:
        var_pnl = 0.0
    std_pnl = sqrt(var_pnl * <double>n / (<double>n - 1.0)) if n > 1 else 0.0
    sqn = sqrt(<double>n) * (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
    
    # Sharpe
    mean_ret = sum_returns / <double>n
    var_ret = (sum_returns_sq / <double>n) - (mean_ret * mean_ret)
    if var_ret < 0:
        var_ret = 0.0
    std_ret = sqrt(var_ret * <double>n / (<double>n - 1.0)) if n > 1 else 0.0
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    
    # Expectancy & Profit Factor
    avg_win = sum_wins / <double>n_wins if n_wins > 0 else 0.0
    avg_loss = sum_losses / <double>n_losses if n_losses > 0 else 0.0
    p_win = <double>n_wins / <double>n
    expectancy = p_win * avg_win - (1.0 - p_win) * avg_loss
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0
    
    return (roi, winrate, max_dd_pct, sharpe, sqn, expectancy,
            n_wins, n_losses, saldo_final, profit_factor,
            max_ganancia, max_perdida, saldo_min, saldo_max)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double compute_drawdown_c(np.ndarray[DTYPE_f, ndim=1] equity_curve):
    """Drawdown optimizado."""
    cdef:
        Py_ssize_t n = equity_curve.shape[0]
        Py_ssize_t i
        double peak, val, dd_pct, max_dd_pct = 0.0
        double* eq_ptr = <double*>equity_curve.data
    
    if n < 2:
        return 0.0
    
    peak = eq_ptr[0]
    
    with nogil:
        for i in range(1, n):
            val = eq_ptr[i]
            if val > peak:
                peak = val
            if peak > 0:
                dd_pct = 100.0 * (peak - val) / peak
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
    
    return max_dd_pct


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double compute_sharpe_c(np.ndarray[DTYPE_f, ndim=1] returns):
    """Sharpe ratio optimizado."""
    cdef:
        Py_ssize_t n = returns.shape[0]
        Py_ssize_t i
        double sum_ret = 0.0, sum_ret_sq = 0.0
        double mean_ret, var_ret, std_ret
        double* ret_ptr = <double*>returns.data
    
    if n < 2:
        return 0.0
    
    with nogil:
        for i in range(n):
            sum_ret = sum_ret + ret_ptr[i]
            sum_ret_sq = sum_ret_sq + ret_ptr[i] * ret_ptr[i]
    
    mean_ret = sum_ret / <double>n
    var_ret = (sum_ret_sq / <double>n) - (mean_ret * mean_ret)
    if var_ret < 0:
        var_ret = 0.0
    std_ret = sqrt(var_ret * <double>n / (<double>n - 1.0))
    
    return mean_ret / std_ret if std_ret > 0 else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double compute_sqn_c(np.ndarray[DTYPE_f, ndim=1] pnl_array):
    """SQN optimizado."""
    cdef:
        Py_ssize_t n = pnl_array.shape[0]
        Py_ssize_t i
        double sum_pnl = 0.0, sum_pnl_sq = 0.0
        double mean_pnl, var_pnl, std_pnl
        double* pnl_ptr = <double*>pnl_array.data
    
    if n < 2:
        return 0.0
    
    with nogil:
        for i in range(n):
            sum_pnl = sum_pnl + pnl_ptr[i]
            sum_pnl_sq = sum_pnl_sq + pnl_ptr[i] * pnl_ptr[i]
    
    mean_pnl = sum_pnl / <double>n
    var_pnl = (sum_pnl_sq / <double>n) - (mean_pnl * mean_pnl)
    if var_pnl < 0:
        var_pnl = 0.0
    std_pnl = sqrt(var_pnl * <double>n / (<double>n - 1.0))
    
    return sqrt(<double>n) * (mean_pnl / std_pnl) if std_pnl > 0 else 0.0


# =============================================================================
# FUNCIÓN AUXILIAR PARA BÚSQUEDA DE SALIDAS (BATCH)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple find_exits_c(
    np.ndarray[DTYPE_i, ndim=1] entry_indices,
    np.ndarray[DTYPE_f, ndim=1] entry_prices,
    np.ndarray[DTYPE_i, ndim=1] entry_types,
    np.ndarray[DTYPE_f, ndim=1] entry_qty,
    np.ndarray[DTYPE_f, ndim=1] entry_stake,
    np.ndarray[DTYPE_f, ndim=1] close_prices,
    np.ndarray[DTYPE_f, ndim=1] high_prices,
    np.ndarray[DTYPE_f, ndim=1] low_prices,
    bint is_trailing,
    double sl_pct,
    double tp_pct,
    double trail_act_pct,
    double trail_dist_pct,
    int time_stop_bars
):
    """Busca salidas para múltiples trades en batch."""
    cdef:
        Py_ssize_t n_entries = entry_indices.shape[0]
        Py_ssize_t n_bars = close_prices.shape[0]
        Py_ssize_t i, entry_idx, search_limit, exit_idx
        double entry_price, qty, stake
        double sl_distance, tp_distance, trail_act_distance, trail_dist_distance
        double sl_price, tp_price, activation_price, exit_p
        int side, exit_reason
        
        double* close_ptr = <double*>close_prices.data
        double* high_ptr = <double*>high_prices.data
        double* low_ptr = <double*>low_prices.data
        
        np.ndarray[DTYPE_i, ndim=1] exit_indices = np.full(n_entries, -1, dtype=np.int64)
        np.ndarray[DTYPE_f, ndim=1] exit_prices = np.full(n_entries, NAN, dtype=np.float64)
        np.ndarray[DTYPE_i32, ndim=1] exit_reasons = np.zeros(n_entries, dtype=np.int32)
    
    with nogil:
        for i in range(n_entries):
            entry_idx = entry_indices[i]
            entry_price = entry_prices[i]
            side = <int>entry_types[i]
            qty = entry_qty[i]
            stake = entry_stake[i]
            
            if qty <= 0 or stake <= 0:
                continue
            
            sl_distance = (stake * sl_pct / 100.0) / qty
            tp_distance = (stake * tp_pct / 100.0) / qty
            trail_act_distance = (stake * trail_act_pct / 100.0) / qty
            trail_dist_distance = (stake * trail_dist_pct / 100.0) / qty
            
            if side == 1:
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
                activation_price = entry_price + trail_act_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
                activation_price = entry_price - trail_act_distance
            
            search_limit = n_bars
            if time_stop_bars > 0 and entry_idx + time_stop_bars + 1 < search_limit:
                search_limit = entry_idx + time_stop_bars + 1
            
            exit_reason = _find_exit_inline(
                entry_idx, entry_price, side,
                sl_price, tp_price, activation_price, trail_dist_distance,
                search_limit, n_bars,
                high_ptr, low_ptr, close_ptr,
                is_trailing, tp_pct, time_stop_bars,
                &exit_p, &exit_idx
            )
            
            exit_indices[i] = exit_idx
            exit_prices[i] = exit_p
            exit_reasons[i] = exit_reason
    
    return exit_indices, exit_prices, exit_reasons


# =============================================================================
# NUEVAS FUNCIONES: PERTURBACIÓN Y ANÁLISIS VECINAL
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[DTYPE_f, ndim=1] perturb_returns_c(
    np.ndarray[DTYPE_f, ndim=1] close_prices,
    double noise_factor,
    unsigned long seed
):
    """
    Perturba retornos con ruido gaussiano calibrado a volatilidad.
    Método profesional para validación de robustez.
    
    Returns:
        Array de precios close perturbados
    """
    cdef:
        Py_ssize_t n = close_prices.shape[0]
        Py_ssize_t i
        double* close_ptr = <double*>close_prices.data
        np.ndarray[DTYPE_f, ndim=1] log_returns = np.empty(n - 1, dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=1] new_close = np.empty(n, dtype=np.float64)
        double* log_ret_ptr
        double* new_close_ptr
        double sum_ret = 0.0, sum_ret_sq = 0.0
        double mean_ret, volatility, noise_std
        double cumsum = 0.0
        # LCG random state
        unsigned long rand_state = seed
        double u1, u2, z
        double PI = 3.14159265358979323846
    
    if n < 2:
        return close_prices.copy()
    
    log_ret_ptr = <double*>log_returns.data
    new_close_ptr = <double*>new_close.data
    
    # Calcular log-returns
    with nogil:
        for i in range(n - 1):
            if close_ptr[i] > 1e-10:
                log_ret_ptr[i] = (close_ptr[i + 1] - close_ptr[i]) / close_ptr[i]
            else:
                log_ret_ptr[i] = 0.0
            sum_ret = sum_ret + log_ret_ptr[i]
            sum_ret_sq = sum_ret_sq + log_ret_ptr[i] * log_ret_ptr[i]
        
        # Volatilidad
        mean_ret = sum_ret / <double>(n - 1)
        volatility = sqrt((sum_ret_sq / <double>(n - 1)) - mean_ret * mean_ret)
        if volatility < 1e-10:
            volatility = 0.001
        
        noise_std = volatility * noise_factor
        
        # Añadir ruido gaussiano (Box-Muller transform con LCG)
        for i in range(n - 1):
            # LCG para generar uniformes
            rand_state = (rand_state * 1103515245 + 12345) & 0x7fffffff
            u1 = <double>rand_state / <double>0x7fffffff
            rand_state = (rand_state * 1103515245 + 12345) & 0x7fffffff
            u2 = <double>rand_state / <double>0x7fffffff
            
            # Box-Muller
            if u1 < 1e-10:
                u1 = 1e-10
            z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
            
            log_ret_ptr[i] = log_ret_ptr[i] + z * noise_std
        
        # Reconstruir precios
        new_close_ptr[0] = close_ptr[0]
        cumsum = 0.0
        for i in range(n - 1):
            cumsum = cumsum + log_ret_ptr[i]
            new_close_ptr[i + 1] = close_ptr[0] * (1.0 + cumsum)
            if new_close_ptr[i + 1] < 1e-10:
                new_close_ptr[i + 1] = 1e-10
    
    return new_close


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double compute_cvar_95_c(np.ndarray[DTYPE_f, ndim=1] equity_curve):
    """
    Calcula CVaR 95% (Conditional Value at Risk).
    Promedio de las peores pérdidas (5% peor).
    """
    cdef:
        Py_ssize_t n = equity_curve.shape[0]
        Py_ssize_t i, n_returns, n_tail
        double* eq_ptr = <double*>equity_curve.data
        np.ndarray[DTYPE_f, ndim=1] returns
        double* ret_ptr
        double sum_tail = 0.0
    
    if n < 20:
        return 50.0  # Default alto si no hay suficientes datos
    
    n_returns = n - 1
    returns = np.empty(n_returns, dtype=np.float64)
    ret_ptr = <double*>returns.data
    
    # Calcular retornos
    with nogil:
        for i in range(n_returns):
            if eq_ptr[i] > 1e-10:
                ret_ptr[i] = (eq_ptr[i + 1] - eq_ptr[i]) / eq_ptr[i]
            else:
                ret_ptr[i] = 0.0
    
    # Ordenar (usamos numpy sort fuera de nogil)
    returns = np.sort(returns)
    ret_ptr = <double*>returns.data
    
    # CVaR = promedio del peor 5%
    n_tail = max(1, <Py_ssize_t>(n_returns * 0.05))
    
    with nogil:
        for i in range(n_tail):
            sum_tail = sum_tail + ret_ptr[i]
    
    return -100.0 * sum_tail / <double>n_tail  # Negativo y en %


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double compute_equity_r2_c(np.ndarray[DTYPE_f, ndim=1] equity_curve):
    """
    Calcula R² de la curva de equity (ajuste lineal).
    Mide consistencia del crecimiento.
    """
    cdef:
        Py_ssize_t n = equity_curve.shape[0]
        Py_ssize_t i
        double* eq_ptr = <double*>equity_curve.data
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0
        double sum_x2 = 0.0, sum_y2 = 0.0
        double x_val, y_val
        double mean_x, mean_y
        double cov_xy, var_x, var_y
        double correlation, r2
    
    if n < 10:
        return 0.0
    
    # Usar log para estabilidad
    with nogil:
        for i in range(n):
            x_val = <double>i
            y_val = eq_ptr[i]
            if y_val > 1e-10:
                y_val = log(y_val)
            else:
                y_val = -23.0  # log(1e-10)
            
            sum_x = sum_x + x_val
            sum_y = sum_y + y_val
            sum_xy = sum_xy + x_val * y_val
            sum_x2 = sum_x2 + x_val * x_val
            sum_y2 = sum_y2 + y_val * y_val
        
        mean_x = sum_x / <double>n
        mean_y = sum_y / <double>n
        
        cov_xy = (sum_xy / <double>n) - (mean_x * mean_y)
        var_x = (sum_x2 / <double>n) - (mean_x * mean_x)
        var_y = (sum_y2 / <double>n) - (mean_y * mean_y)
        
        if var_x > 1e-10 and var_y > 1e-10:
            correlation = cov_xy / sqrt(var_x * var_y)
            r2 = correlation * correlation
        else:
            r2 = 0.0
        
        if r2 < 0.0:
            r2 = 0.0
        if r2 > 1.0:
            r2 = 1.0
    
    return r2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple aggregate_neighbor_metrics_c(
    np.ndarray[DTYPE_f, ndim=1] scores,
    np.ndarray[DTYPE_f, ndim=1] sharpes,
    np.ndarray[DTYPE_f, ndim=1] cvars,
    np.ndarray[DTYPE_f, ndim=1] r2s,
    double lambda_penalty
):
    """
    Agrega métricas de vecinos para calcular la Trinidad de Objetivos.
    
    Returns:
        (robust_score, mean_score, std_score, worst_cvar, avg_r2)
    """
    cdef:
        Py_ssize_t n = scores.shape[0]
        Py_ssize_t i
        double* score_ptr = <double*>scores.data
        double* sharpe_ptr = <double*>sharpes.data
        double* cvar_ptr = <double*>cvars.data
        double* r2_ptr = <double*>r2s.data
        double sum_scores = 0.0, sum_scores_sq = 0.0
        double sum_sharpes = 0.0, sum_sharpes_sq = 0.0
        double sum_r2 = 0.0
        double worst_cvar = 0.0
        double mean_score, std_score, mean_sharpe, std_sharpe, avg_r2
        double robust_score
    
    if n == 0:
        return (0.0, 0.0, 0.0, 100.0, 0.0)
    
    with nogil:
        for i in range(n):
            sum_scores = sum_scores + score_ptr[i]
            sum_scores_sq = sum_scores_sq + score_ptr[i] * score_ptr[i]
            sum_sharpes = sum_sharpes + sharpe_ptr[i]
            sum_sharpes_sq = sum_sharpes_sq + sharpe_ptr[i] * sharpe_ptr[i]
            sum_r2 = sum_r2 + r2_ptr[i]
            
            if cvar_ptr[i] > worst_cvar:
                worst_cvar = cvar_ptr[i]
        
        mean_score = sum_scores / <double>n
        mean_sharpe = sum_sharpes / <double>n
        avg_r2 = sum_r2 / <double>n
        
        if n > 1:
            std_score = sqrt((sum_scores_sq / <double>n) - (mean_score * mean_score))
            std_sharpe = sqrt((sum_sharpes_sq / <double>n) - (mean_sharpe * mean_sharpe))
            if std_score < 0:
                std_score = 0.0
            if std_sharpe < 0:
                std_sharpe = 0.0
        else:
            std_score = 0.0
            std_sharpe = 0.0
        
        # Score robusto = media - lambda * desviación
        robust_score = mean_score - lambda_penalty * std_score
        if robust_score < 0:
            robust_score = 0.0
    
    return (robust_score, mean_score, std_score, worst_cvar, avg_r2)

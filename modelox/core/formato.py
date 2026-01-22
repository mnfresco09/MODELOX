"""
modelox/core/formato.py
========================
Formateo consistente de métricas para todo el sistema MODELOX.

FORMATO ESTÁNDAR:
- Números >= 1: X,XX (ej: 25,43)
- Números < 1:  0,XXX (ej: 0,543)
- Porcentajes siempre con 2 decimales
- NaN/Inf -> "---"

Este módulo centraliza TODO el formateo de métricas para:
- Consola Rich
- Excel
- CSV
- Gráficos
"""

from __future__ import annotations

import math
from typing import Any, Dict


def formatear_numero(valor: Any, decimales: int = 2, forzar_signo: bool = False) -> str:
    """
    Formatea un número con el estándar MODELOX.
    
    Args:
        valor: Número a formatear (puede ser None, str, float, int)
        decimales: Decimales a mostrar (default 2)
        forzar_signo: Si True, añade + a números positivos
    
    Returns:
        String formateado según reglas MODELOX
    
    Ejemplos:
        formatear_numero(25.432)     -> "25,43"
        formatear_numero(0.5432)     -> "0,543"
        formatear_numero(-15.7)      -> "-15,70"
        formatear_numero(None)       -> "---"
        formatear_numero(float('inf')) -> "---"
    """
    # Caso nulo
    if valor is None:
        return "---"
    
    # Convertir a float
    try:
        num = float(valor)
    except (ValueError, TypeError):
        return str(valor)
    
    # Caso NaN o Infinito
    if math.isnan(num) or math.isinf(num):
        return "---"
    
    # Determinar decimales basado en magnitud
    abs_num = abs(num)
    if abs_num < 1 and abs_num > 0:
        # Para números menores a 1, usar 3 decimales
        dec = 3
    else:
        dec = decimales
    
    # Formatear
    if forzar_signo and num > 0:
        resultado = f"+{num:,.{dec}f}"
    else:
        resultado = f"{num:,.{dec}f}"
    
    # Convertir separadores: punto decimal a coma, comas a puntos
    # Ejemplo: 1,234.56 -> 1.234,56
    resultado = resultado.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    
    return resultado


def formatear_porcentaje(valor: Any, decimales: int = 2, con_simbolo: bool = True) -> str:
    """
    Formatea un valor como porcentaje.
    
    Args:
        valor: Número (ya debe ser porcentaje, no decimal)
        decimales: Decimales a mostrar
        con_simbolo: Si True, añade '%' al final
    
    Returns:
        String formateado como porcentaje
    
    Ejemplos:
        formatear_porcentaje(25.5)     -> "25,50%"
        formatear_porcentaje(-5.123)   -> "-5,12%"
        formatear_porcentaje(0.5, con_simbolo=False) -> "0,50"
    """
    if valor is None:
        return "---"
    
    try:
        num = float(valor)
    except (ValueError, TypeError):
        return str(valor)
    
    if math.isnan(num) or math.isinf(num):
        return "---"
    
    resultado = f"{num:,.{decimales}f}"
    resultado = resultado.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    
    if con_simbolo:
        resultado += "%"
    
    return resultado


def formatear_entero(valor: Any, separador_miles: bool = True) -> str:
    """
    Formatea un número como entero.
    
    Args:
        valor: Número a formatear
        separador_miles: Si True, usa separador de miles
    
    Returns:
        String formateado como entero
    
    Ejemplos:
        formatear_entero(1500)      -> "1.500"
        formatear_entero(50)        -> "50"
        formatear_entero(None)      -> "---"
    """
    if valor is None:
        return "---"
    
    try:
        num = int(float(valor))
    except (ValueError, TypeError):
        return str(valor)
    
    if separador_miles:
        resultado = f"{num:,d}"
        resultado = resultado.replace(",", ".")
    else:
        resultado = str(num)
    
    return resultado


def formatear_moneda(valor: Any, simbolo: str = "$", decimales: int = 2) -> str:
    """
    Formatea un número como moneda.
    
    Args:
        valor: Número a formatear
        simbolo: Símbolo de moneda (default $)
        decimales: Decimales a mostrar
    
    Returns:
        String formateado como moneda
    
    Ejemplos:
        formatear_moneda(1500.50)   -> "$1.500,50"
        formatear_moneda(-250.5)    -> "-$250,50"
    """
    if valor is None:
        return "---"
    
    try:
        num = float(valor)
    except (ValueError, TypeError):
        return str(valor)
    
    if math.isnan(num) or math.isinf(num):
        return "---"
    
    if num < 0:
        resultado = f"-{simbolo}{abs(num):,.{decimales}f}"
    else:
        resultado = f"{simbolo}{num:,.{decimales}f}"
    
    resultado = resultado.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    
    return resultado


def formatear_metricas(metricas: Dict[str, Any]) -> Dict[str, str]:
    """
    Formatea un diccionario completo de métricas según el estándar MODELOX.
    
    Args:
        metricas: Diccionario con métricas crudas
    
    Returns:
        Diccionario con métricas formateadas como strings
    """
    # Mapeo de métricas a tipo de formato
    FORMATO_PORCENTAJE = {
        "roi", "roi_pct", "winrate", "drawdown", "max_drawdown", "porc_ganadoras",
        "porc_perdedoras", "estabilidad"
    }
    
    FORMATO_ENTERO = {
        "n_trades", "total_trades", "num_trades", "trades_totales",
        "count_longs", "num_longs", "n_longs",
        "count_shorts", "num_shorts", "n_shorts",
        "racha_ganadora", "racha_perdedora", "win_streak", "loss_streak"
    }
    
    FORMATO_MONEDA = {
        "saldo_actual", "saldo_final", "saldo_inicial", "saldo_min", "saldo_max",
        "pnl_neto", "net_pnl", "max_ganancia", "max_perdida", "comisiones_total"
    }
    
    resultado = {}
    
    for clave, valor in metricas.items():
        clave_lower = clave.lower()
        
        if clave_lower in FORMATO_PORCENTAJE:
            resultado[clave] = formatear_porcentaje(valor)
        elif clave_lower in FORMATO_ENTERO:
            resultado[clave] = formatear_entero(valor)
        elif clave_lower in FORMATO_MONEDA:
            resultado[clave] = formatear_moneda(valor)
        else:
            resultado[clave] = formatear_numero(valor)
    
    return resultado


def obtener_valor_seguro(
    metricas: Dict[str, Any],
    clave: str,
    default: float = 0.0,
    claves_alternativas: tuple = ()
) -> float:
    """
    Obtiene un valor numérico de forma segura del diccionario de métricas.
    
    Args:
        metricas: Diccionario de métricas
        clave: Clave principal a buscar
        default: Valor por defecto si no se encuentra
        claves_alternativas: Tupla de claves alternativas a probar
    
    Returns:
        Valor numérico (float)
    """
    # Intentar clave principal
    valor = metricas.get(clave)
    if valor is not None:
        try:
            num = float(valor)
            if not (math.isnan(num) or math.isinf(num)):
                return num
        except (ValueError, TypeError):
            pass
    
    # Intentar claves alternativas
    for alt in claves_alternativas:
        valor = metricas.get(alt)
        if valor is not None:
            try:
                num = float(valor)
                if not (math.isnan(num) or math.isinf(num)):
                    return num
            except (ValueError, TypeError):
                pass
    
    return default


# Alias para compatibilidad
fmt_num = formatear_numero
fmt_pct = formatear_porcentaje
fmt_int = formatear_entero
fmt_mon = formatear_moneda

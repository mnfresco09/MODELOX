"""
TRADE LOGGER - REGISTRO DE TRADES EN EXCEL
============================================
Guarda cada trade cerrado en un archivo Excel con formato profesional.
El archivo se crea si no existe y se agregan filas nuevas.

Estilo basado en visual/excel.py de MODELOX.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURACION
# =============================================================================

EXCEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resultados")
EXCEL_FILENAME = "TRADES_REAL.xlsx"
SHEET_NAME = "Trades"

# Estilo MODELOX
COLORS = {
    "header_bg": "1A5276",
    "text_white": "FFFFFF",
    "text_dark": "212F3D",
    "border": "BDC3C7",
    "profit_bg": "D5F5E3",
    "profit_text": "006100",
    "loss_bg": "FADBD8",
    "loss_text": "922B21",
}

FONT_HEADER = "Arial"
FONT_BODY = "Arial"
FONT_SIZE = 12

# Columnas en orden
COLUMNS = [
    "FECHA",
    "DURACION",
    "ACTIVO",
    "DIRECCION",
    "CANTIDAD",
    "PRECIO APERTURA",
    "PRECIO SALIDA",
    "PNL BRUTO",
    "COMISIONES",
    "PNL NETO",
    "ROI %",
    "CIERRE",
    "SL %",
    "TP %",
    "TRAIL ACT %",
    "TRAIL DIST %",
    "ESTRATEGIA",
]


def _get_excel_path() -> str:
    """Retorna la ruta completa del archivo Excel."""
    os.makedirs(EXCEL_DIR, exist_ok=True)
    return os.path.join(EXCEL_DIR, EXCEL_FILENAME)


def _get_display_name(symbol: str) -> str:
    """Convierte simbolo BingX a nombre corto."""
    from real.config import DISPLAY_NAMES
    return DISPLAY_NAMES.get(symbol, symbol.replace("-USDT", ""))


def _determine_close_reason(trade: Any) -> str:
    """Determina el motivo de cierre."""
    reason = getattr(trade, 'reason', None) or 'MANUAL'
    reason_map = {
        "SL": "SL",
        "TP": "TP",
        "TRAILING": "TRAILING",
        "EXCHANGE_CLOSE": "MANUAL",
    }
    return reason_map.get(reason, "MANUAL")


def log_trade(
    trade: Any,
    config: Any,
    strategy_name: str = "",
) -> None:
    """
    Registra un trade cerrado en el archivo Excel.
    Crea el archivo si no existe, agrega fila si ya existe.
    
    Args:
        trade: TradeRecord con datos del trade
        config: TradingConfig con configuracion
        strategy_name: Nombre de la estrategia usada
    """
    try:
        excel_path = _get_excel_path()
        
        # Datos del trade
        pnl_bruto = getattr(trade, 'pnl', 0) or 0
        trading_fees = getattr(trade, 'trading_fees', 0) or 0
        funding_fees = getattr(trade, 'funding_fees', 0) or 0
        comisiones = trading_fees + funding_fees
        pnl_neto = getattr(trade, 'total_pnl', None) or (pnl_bruto + comisiones)
        quantity = getattr(trade, 'quantity', 0) or 0
        leverage = getattr(trade, 'leverage', 1) or 1
        
        # ROI sobre margen
        margin = (quantity * trade.entry_price) / leverage if trade.entry_price > 0 else 0
        roi = (pnl_neto / margin) * 100 if margin > 0 else 0
        
        # Motivo de cierre
        close_reason = _determine_close_reason(trade)
        
        # SL/TP/Trailing desde config
        sl_pct = getattr(config, 'stop_loss_pct', None)
        tp_pct = getattr(config, 'take_profit_pct', None)
        trailing_act = None
        trailing_dist = None
        
        trailing_enabled = getattr(config, 'trailing_stop_enabled', False)
        if trailing_enabled:
            trailing_act = getattr(config, 'trailing_stop_activation_pct', None)
            trailing_dist = getattr(config, 'trailing_stop_distance_pct', None)
            # Si cerro por trailing, no mostrar TP
            if close_reason == "TRAILING":
                tp_pct = None
        
        # Si cerro por TP normal, no mostrar trailing
        if close_reason == "TP" and not trailing_enabled:
            trailing_act = None
            trailing_dist = None
        
        # Fecha
        exit_time = getattr(trade, 'exit_time', None) or datetime.now()
        fecha = exit_time.strftime("%d/%m/%Y") if isinstance(exit_time, datetime) else datetime.now().strftime("%d/%m/%Y")
        
        # Duracion
        open_time = getattr(trade, 'timestamp', None)
        if open_time and isinstance(exit_time, datetime) and isinstance(open_time, datetime):
            delta = exit_time - open_time
            total_secs = int(delta.total_seconds())
            hours = total_secs // 3600
            minutes = (total_secs % 3600) // 60
            seconds = total_secs % 60
            duracion = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            duracion = "--:--:--"
        
        # Construir fila
        row = {
            "FECHA": fecha,
            "DURACION": duracion,
            "ACTIVO": _get_display_name(trade.symbol),
            "DIRECCION": trade.side,
            "CANTIDAD": round(quantity, 6),
            "PRECIO APERTURA": round(trade.entry_price, 2),
            "PRECIO SALIDA": round(trade.exit_price, 2) if trade.exit_price else 0,
            "PNL BRUTO": round(pnl_bruto, 4),
            "COMISIONES": round(comisiones, 4),
            "PNL NETO": round(pnl_neto, 4),
            "ROI %": round(roi, 2),
            "CIERRE": close_reason,
            "SL %": sl_pct,
            "TP %": tp_pct if close_reason != "TRAILING" else None,
            "TRAIL ACT %": trailing_act,
            "TRAIL DIST %": trailing_dist,
            "ESTRATEGIA": strategy_name,
        }
        
        # Verificar si el archivo existe
        if os.path.exists(excel_path):
            _append_row(excel_path, row)
        else:
            _create_new_excel(excel_path, row)
            
    except Exception as e:
        # No romper el flujo por un error de logging
        import logging
        logging.getLogger(__name__).warning(f"Error guardando trade en Excel: {e}")


def _create_new_excel(path: str, row: dict) -> None:
    """Crea un nuevo archivo Excel con headers y la primera fila."""
    df = pd.DataFrame([row], columns=COLUMNS)
    df.to_excel(path, index=False, sheet_name=SHEET_NAME)
    _apply_style(path)


def _append_row(path: str, row: dict) -> None:
    """Agrega una fila al archivo Excel existente."""
    wb = load_workbook(path)
    ws = wb[SHEET_NAME] if SHEET_NAME in wb.sheetnames else wb.active
    
    next_row = ws.max_row + 1
    
    for col_idx, col_name in enumerate(COLUMNS, 1):
        cell = ws.cell(row=next_row, column=col_idx)
        cell.value = row.get(col_name)
        
        # Aplicar estilo de datos
        cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["text_dark"])
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = Border(
            left=Side(style='thin', color=COLORS["border"]),
            right=Side(style='thin', color=COLORS["border"]),
            top=Side(style='thin', color=COLORS["border"]),
            bottom=Side(style='thin', color=COLORS["border"]),
        )
        
        # Formato numerico
        header = col_name.upper()
        if isinstance(cell.value, (int, float)):
            if "PNL" in header or "COMISIONES" in header:
                cell.number_format = '#,##0.0000'
            elif "PRECIO" in header:
                cell.number_format = '#,##0.00'
            elif "ROI" in header or "%" in header:
                cell.number_format = '0.00'
            elif "CANTIDAD" in header:
                cell.number_format = '0.000000'
    
    # Color condicional en PNL NETO
    pnl_col = COLUMNS.index("PNL NETO") + 1
    pnl_cell = ws.cell(row=next_row, column=pnl_col)
    if pnl_cell.value is not None:
        if float(pnl_cell.value) >= 0:
            pnl_cell.fill = PatternFill("solid", fgColor=COLORS["profit_bg"])
            pnl_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["profit_text"], bold=True)
        else:
            pnl_cell.fill = PatternFill("solid", fgColor=COLORS["loss_bg"])
            pnl_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["loss_text"], bold=True)
    
    # Color condicional en ROI
    roi_col = COLUMNS.index("ROI %") + 1
    roi_cell = ws.cell(row=next_row, column=roi_col)
    if roi_cell.value is not None:
        if float(roi_cell.value) >= 0:
            roi_cell.fill = PatternFill("solid", fgColor=COLORS["profit_bg"])
            roi_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["profit_text"], bold=True)
        else:
            roi_cell.fill = PatternFill("solid", fgColor=COLORS["loss_bg"])
            roi_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["loss_text"], bold=True)
    
    wb.save(path)


def _apply_style(path: str) -> None:
    """Aplica el estilo profesional MODELOX al archivo Excel."""
    wb = load_workbook(path)
    ws = wb.active
    ws.sheet_view.showGridLines = False
    
    max_col = ws.max_column
    max_row = ws.max_row
    
    border = Border(
        left=Side(style='thin', color=COLORS["border"]),
        right=Side(style='thin', color=COLORS["border"]),
        top=Side(style='thin', color=COLORS["border"]),
        bottom=Side(style='thin', color=COLORS["border"]),
    )
    
    # HEADERS (fila 1)
    for col in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(name=FONT_HEADER, size=FONT_SIZE, bold=True, color=COLORS["text_white"])
        cell.fill = PatternFill("solid", fgColor=COLORS["header_bg"])
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border
        
        # Ancho de columna
        val_len = len(str(cell.value)) if cell.value else 8
        ws.column_dimensions[get_column_letter(col)].width = max(12, val_len + 4)
    
    ws.row_dimensions[1].height = 28
    
    # DATOS
    for r in range(2, max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["text_dark"])
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
            
            # Formato numerico
            header = str(ws.cell(1, c).value).upper()
            if isinstance(cell.value, (int, float)):
                if "PNL" in header or "COMISIONES" in header:
                    cell.number_format = '#,##0.0000'
                elif "PRECIO" in header:
                    cell.number_format = '#,##0.00'
                elif "ROI" in header or "%" in header:
                    cell.number_format = '0.00'
                elif "CANTIDAD" in header:
                    cell.number_format = '0.000000'
        
        # Color PNL NETO
        pnl_col = COLUMNS.index("PNL NETO") + 1
        pnl_cell = ws.cell(row=r, column=pnl_col)
        if pnl_cell.value is not None:
            try:
                val = float(pnl_cell.value)
                if val >= 0:
                    pnl_cell.fill = PatternFill("solid", fgColor=COLORS["profit_bg"])
                    pnl_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["profit_text"], bold=True)
                else:
                    pnl_cell.fill = PatternFill("solid", fgColor=COLORS["loss_bg"])
                    pnl_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["loss_text"], bold=True)
            except (ValueError, TypeError):
                pass
        
        # Color ROI
        roi_col = COLUMNS.index("ROI %") + 1
        roi_cell = ws.cell(row=r, column=roi_col)
        if roi_cell.value is not None:
            try:
                val = float(roi_cell.value)
                if val >= 0:
                    roi_cell.fill = PatternFill("solid", fgColor=COLORS["profit_bg"])
                    roi_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["profit_text"], bold=True)
                else:
                    roi_cell.fill = PatternFill("solid", fgColor=COLORS["loss_bg"])
                    roi_cell.font = Font(name=FONT_BODY, size=FONT_SIZE, color=COLORS["loss_text"], bold=True)
            except (ValueError, TypeError):
                pass
    
    # Ajustar anchos especificos
    special_widths = {
        "FECHA": 14,
        "DURACION": 14,
        "ACTIVO": 12,
        "DIRECCION": 12,
        "CANTIDAD": 14,
        "PRECIO APERTURA": 18,
        "PRECIO SALIDA": 16,
        "PNL BRUTO": 14,
        "COMISIONES": 14,
        "PNL NETO": 14,
        "ROI %": 10,
        "CIERRE": 12,
        "SL %": 10,
        "TP %": 10,
        "TRAIL ACT %": 14,
        "TRAIL DIST %": 14,
        "ESTRATEGIA": 20,
    }
    
    for col_idx, col_name in enumerate(COLUMNS, 1):
        if col_name in special_widths:
            ws.column_dimensions[get_column_letter(col_idx)].width = special_widths[col_name]
    
    wb.save(path)

"""
================================================================================
VISUAL/EXCEL.PY — DASHBOARD QUANT EN EXCEL  (v7.1 — ANCHO 10 XL)
================================================================================

CAMBIOS v7.1:
  - Gráfico: anclado en col B, ancho 1000px (ancho 10)
  - Tabla:   columna R (índice 17, 0-based), fuente 13pt, fila 2x
  - Sin solape: chart 1000px, tabla desplazada a la derecha
================================================================================
"""

import os
import re
import math
import datetime
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import csv

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import DataBarRule, ColorScaleRule

try:
    import xlsxwriter
    from xlsxwriter.utility import xl_col_to_name
    _HAS_XW = True
except ImportError:
    _HAS_XW = False

logger = logging.getLogger(__name__)

# ==============================================================================
# COLORES Y CONSTANTES
# ==============================================================================

COLORS = {
    "header_bg_metrics": "1A1A2E",
    "header_bg_params":  "16213E",
    "header_bg_id":      "0F3460",
    "text_white":        "FFFFFF",
    "text_dark":         "1A1A2E",
    "border_color":      "E8EDF5",
    "success_bg":        "E8F5E9",
    "danger_bg":         "FFEBEE",
    "accent_green":      "00897B",
    "accent_red":        "C62828",
    "row_alt":           "F7F9FC",
    "section_border":    "3A86FF",
    "table_header_bg":   "E3EAF6",
}

FONT_TITLE = "Arial"
FONT_BODY  = "Arial"

METRICS_ORDER = [
    "TOTAL_TRADES", "TRADES_DIA", "LONG", "SHORT",
    "PROFIT_FACTOR", "ROI_PCT", "WINRATE_PCT", "MAX_DD_PCT",
    "SHARPE", "SQN", "EXPECTATIVA"
]
ID_COLS = ["TRIAL", "ESTRATEGIA", "SCORE"]

EXCLUDED_PARAMS = {
    "NOMBRE_COMBO", "EXIT_TYPE", "CANTIDAD",
    "ACTIVO", "TIMEFRAME", "TF", "ASSET", "SYMBOL",
    "RESULTADO", "METRICS", "COMBO", "ESTATEGIA",
    "SALDO", "VOLUMEN", "APALANCAMIENTO",
}

METRIC_KEYWORDS_TO_DROP = [
    "PROFIT", "LOSS", "PNL", "NET", "GROSS", "SALDO", "BALANCE", "RETORNO", "RETURN",
    "ROI", "BENEFICIO", "RIESGO", "RISK", "REWARD", "COMISION", "FEES",
    "WIN", "GANADORA", "PERDEDORA", "ACIERTO", "RATE", "PCT", "PORC_", "PERCENT",
    "DRAWDOWN", "DD", "RACHA", "STREAK", "UNDERWATER",
    "RATIO", "FACTOR", "SHARPE", "SORTINO", "CALMAR", "SQN", "EXPECTATIVA", "KELLY",
    "AVG", "MEAN", "MEDIAN", "STD", "VAR", "MAX", "MIN", "SUM", "TOTAL",
    "ESTABILIDAD", "COUNT", "NUM_", "N_", "TRADES", "LONGS", "SHORTS", "CANTIDAD_OP",
    "METRIC", "RESULT", "BEST", "WORST", "DIA_OPERADO", "DURATION", "TIME"
]

PREFIXES_TO_CLEAN = [
    "ESTRATEGIA_PARAMS_", "STRATEGY_PARAMS_", "PARAM_", "PARAMS_",
    "INDICATOR_", "CONFIG_", "METRICS_"
]

# ==============================================================================
# EXCEL REPORTER
# ==============================================================================

class ReporterProtocol(Protocol):
    def needs_dataframe(self, score: float) -> bool: ...
    def on_trial_end(self, artifacts: Any) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


@dataclass
class ExcelReporter:
    resumen_path: str = "resultados/excel/resumen.xlsx"
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5
    use_fast_mode: bool = True

    _csv_resumen_path: Optional[str] = field(default=None, init=False, repr=False)
    _resumen_rows: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _trade_candidates: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _min_candidate_score: float = field(default=float("-inf"), init=False, repr=False)
    _activo: Optional[str] = field(default=None, init=False, repr=False)
    _final_excel_path: Optional[str] = field(default=None, init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        return False

    @staticmethod
    def _safe_activo_name(activo: str) -> str:
        return str(activo).strip().replace(" ", "_").upper() if activo else "DEFAULT"

    def _update_min_score(self):
        self._min_candidate_score = (
            min(c["score"] for c in self._trade_candidates)
            if self._trade_candidates else float("-inf")
        )

    def on_trial_end(self, artifacts) -> None:
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        activo = None
        if isinstance(params_src, dict):
            activo = params_src.get("__activo") or params_src.get("ACTIVO") or params_src.get("activo")

        self._activo = activo
        score  = artifacts.score if artifacts.score is not None else 0.0
        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name

        self._resumen_rows.append({
            "trial_number":  artifacts.trial_number,
            "score":         score,
            "metrics":       deepcopy(artifacts.metrics) if artifacts.metrics else {},
            "params":        {k: v for k, v in params.items() if not str(k).startswith("__")},
            "strategy_name": artifacts.strategy_name,
        })

        try:
            base_dir = self.trades_base_dir
            os.makedirs(base_dir, exist_ok=True)
            if not self._csv_resumen_path:
                self._csv_resumen_path = os.path.join(base_dir, "RESUMEN.csv")
            self._write_resumen_csv(self._csv_resumen_path)
        except Exception:
            pass

        is_candidate = (
            len(self._trade_candidates) < self.max_archivos or
            score > self._min_candidate_score
        )
        if is_candidate and artifacts.trades is not None:
            self._trade_candidates.append({
                "score":        score,
                "trial_number": artifacts.trial_number,
                "trades":       artifacts.trades,
                "params":       params,
                "metrics":      artifacts.metrics,
            })
            if len(self._trade_candidates) > self.max_archivos:
                self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
                self._trade_candidates.pop()
            self._update_min_score()

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if not self._resumen_rows:
            return

        activo   = self._activo
        base_dir = self.trades_base_dir
        os.makedirs(base_dir, exist_ok=True)

        activo_safe = self._safe_activo_name(str(activo) if activo else "DEFAULT")
        csv_path    = os.path.join(base_dir, "RESUMEN.csv")
        self._write_resumen_csv(csv_path)

        self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
        for candidate in self._trade_candidates[:self.max_archivos]:
            try:
                self._write_trades_excel(base_dir, candidate)
            except Exception as e:
                logger.warning(f"Error guardando trades trial {candidate['trial_number']}: {e}")

        try:
            self._final_excel_path = convertir_resumen_csv_a_excel(
                csv_path=csv_path,
                strategy_name=strategy_name,
                activo=activo_safe,
                output_dir=base_dir,
                excel_path=self.resumen_path
            )
        except Exception as e:
            logger.warning(f"Error generando Dashboard Excel: {e}")

        self._resumen_rows        = []
        self._trade_candidates    = []
        self._min_candidate_score = float("-inf")

    def _write_resumen_csv(self, csv_path: str):
        if not self._resumen_rows:
            return
        all_keys = {"trial", "score", "strategy"}
        for row in self._resumen_rows:
            if row.get("metrics"):
                all_keys.update(row["metrics"].keys())
            if row.get("params"):
                all_keys.update(f"param_{k}" for k in row["params"].keys())

        columns = ["trial", "score", "strategy"]
        columns.extend(sorted(k for k in all_keys if k not in {"trial","score","strategy"} and not k.startswith("param_")))
        columns.extend(sorted(k for k in all_keys if k.startswith("param_")))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self._resumen_rows:
                csv_row = {"trial": row["trial_number"], "score": row["score"], "strategy": row["strategy_name"]}
                if row.get("metrics"):
                    csv_row.update(row["metrics"])
                if row.get("params"):
                    csv_row.update({f"param_{k}": v for k, v in row["params"].items()})
                writer.writerow(csv_row)

    def _write_trades_excel(self, trades_dir: str, candidate: Dict[str, Any]):
        trades = candidate["trades"]
        if trades is None or (hasattr(trades, "empty") and trades.empty):
            return

        df_trades = trades.to_pandas() if hasattr(trades, "to_pandas") else trades

        # --- Limpiar timezone ---
        try:
            df_trades = df_trades.copy()
            if isinstance(df_trades.index, pd.DatetimeIndex) and df_trades.index.tz is not None:
                df_trades.index = df_trades.index.tz_localize(None)
            for col in list(df_trades.columns):
                try:
                    if isinstance(df_trades[col].dtype, pd.DatetimeTZDtype):
                        df_trades[col] = df_trades[col].dt.tz_localize(None)
                except Exception:
                    continue
            for col in list(df_trades.columns):
                if df_trades[col].dtype != object:
                    continue
                s = df_trades[col]
                try:
                    sample = next((v for v in s.head(50).tolist() if v is not None), None)
                    if sample and getattr(sample, "tzinfo", None):
                        df_trades[col] = s.apply(
                            lambda v: v.replace(tzinfo=None) if hasattr(v, "tzinfo") and v.tzinfo else v
                        )
                except Exception:
                    continue
        except Exception:
            pass

        df_export = _preparar_df_trades(df_trades)

        saldo = candidate['params'].get('__saldo_usado') or 0
        apal  = candidate['params'].get('__apalancamiento_max') or 0
        vol   = saldo * apal

        filename = f"TRIAL {candidate['trial_number']}.xlsx"
        filepath = os.path.join(trades_dir, filename)

        try:
            _escribir_trades_xlsxwriter(filepath, df_export, saldo, vol, apal)
        except Exception as e:
            logger.warning(f"Error xlsxwriter, fallback openpyxl: {e}")
            _escribir_trades_openpyxl_fallback(filepath, df_export, saldo, vol, apal)


# ==============================================================================
# PREPARACIÓN DEL DATAFRAME
# ==============================================================================

def _preparar_df_trades(df_trades: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "entry_time": "ENTRY_TIME", "exit_time": "EXIT_TIME",
        "type": "POSICIÓN", "entry_price": "ENTRY_PRICE", "exit_price": "EXIT_PRICE",
        "qty": "CANTIDAD", "saldo_usado": "SALDO",
        "pnl_bruto": "PNL BRUTO", "comision": "COMISIONES",
        "pnl_neto": "PNL NETO", "pnl_pct": "ROI",
        "saldo_antes": "BALANCE_PRE", "saldo_despues": "BALANCE",
        "reason": "EXIT_REASON", "trail_act_price": "TRAIL_ACT_PRICE",
        "trail_act_time": "TRAIL_ACT_TIME"
    }
    df = df_trades.rename(columns=rename_map)

    cols_to_drop = [
        "ENTRY_IDX", "EXIT_IDX", "SIDE_INT", "entry_idx", "exit_idx", "side_int",
        "BALANCE_PRE", "TRAIL_ACT_IDX", "trail_act_idx",
        "ROI", "SALDO", "VOLUMEN", "APALANCAMIENTO", "CANTIDAD"
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    reason_map = {1: "SL", 2: "TP", 3: "TRAIL", 4: "TIME", 0: "END", 5: "CUSTOM"}
    if "EXIT_REASON" in df.columns:
        df["EXIT_REASON"] = df["EXIT_REASON"].map(reason_map).fillna("UNKNOWN")

    for c in ["POSICIÓN", "EXIT_REASON"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()

    df.columns = [c.upper() for c in df.columns]

    order = [
        "ENTRY_TIME", "ENTRY_PRICE", "TRAIL_ACT_PRICE", "TRAIL_ACT_TIME",
        "EXIT_TIME", "EXIT_PRICE", "POSICIÓN", "EXIT_REASON",
        "PNL BRUTO", "COMISIONES", "PNL NETO", "BALANCE"
    ]
    # Truncar precios a 2 decimales según petición del usuario
    for price_col in ["ENTRY_PRICE", "EXIT_PRICE", "TRAIL_ACT_PRICE"]:
        if price_col in df.columns:
            df[price_col] = df[price_col].apply(lambda x: int(x * 100) / 100.0 if pd.notnull(x) else x)

    return df[[c for c in order if c in df.columns]]


# ==============================================================================
# HELPER: major_unit "bonito" para el eje Y
# ==============================================================================

def _nice_major_unit(data_range: float, target_ticks: int = 6) -> float:
    """Calcula un major_unit legible para el eje Y."""
    if data_range <= 0:
        return 1.0
    raw = data_range / target_ticks
    magnitude = math.pow(10, math.floor(math.log10(raw)))
    normalized = raw / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


# ==============================================================================
# ESCRITURA ULTRA RÁPIDA CON XLSXWRITER  (v7 — posición y escalado corregidos)
# ==============================================================================

def _escribir_trades_xlsxwriter(
    filepath: str,
    df: pd.DataFrame,
    val_saldo: float = 0,
    val_volumen: float = 0,
    val_apal: float = 0,
):
    """
    Layout final:
      · Datos:  A1 → L(max_row+1)
      · Gráfico: anclado en B(max_row+3), ancho 500 px × alto 430 px
                 → ocupa aproximadamente B → I (8 cols × ~65 px = 520 px)
      · Tabla:  anclada en J(max_row+3), inmediatamente junto al gráfico
                fuente 13 pt, filas 28 px (tamaño 2x)
    """
    n_rows = len(df)
    cols   = list(df.columns)

    wb = xlsxwriter.Workbook(filepath, {
        'nan_inf_to_errors': True,
        'strings_to_numbers': False,
        # Necesario para que xlsxwriter serialice datetime correctamente
        'default_date_format': 'dd/mm/yy hh:mm',
    })
    ws = wb.add_worksheet('Trades')
    ws.hide_gridlines(2)
    ws.freeze_panes(1, 0)
    ws.set_row(0, 28)

    # ── BASE de formato ──────────────────────────────────────────────────────
    _BASE = dict(
        font_name='Arial', font_size=10, font_color='#1A1A2E',
        align='center', valign='vcenter',
        border=1, border_color='#E8EDF5'
    )

    def _fmt(bg='#FFFFFF', num_format=None, **kw):
        p = {**_BASE, 'bg_color': bg}
        if num_format:
            p['num_format'] = num_format
        p.update(kw)
        return wb.add_format(p)

    hdr_fmt = wb.add_format({
        'bold': True, 'font_name': 'Arial', 'font_size': 11,
        'font_color': '#FFFFFF', 'bg_color': '#1A1A2E',
        'align': 'center', 'valign': 'vcenter',
        'border': 1, 'border_color': '#E8EDF5', 'text_wrap': True,
    })

    # Pares de formato (fila normal, fila alternada)
    FMT = {
        'gen':   (_fmt('#FFFFFF'),                          _fmt('#F7F9FC')),
        'dt':    (_fmt('#FFFFFF', 'dd/mm/yy hh:mm'),        _fmt('#F7F9FC', 'dd/mm/yy hh:mm')),
        'price': (_fmt('#FFFFFF', '#,##0.00'),               _fmt('#F7F9FC', '#,##0.00')),
        'money': (_fmt('#FFFFFF', '#,##0.00'),               _fmt('#F7F9FC', '#,##0.00')),
        'pct':   (_fmt('#FFFFFF', '0.00%'),                  _fmt('#F7F9FC', '0.00%')),
        'apal':  (_fmt('#FFFFFF', '0.00"x"'),                _fmt('#F7F9FC', '0.00"x"')),
    }

    def _fmt_key(hdr: str) -> str:
        h = hdr.upper()
        if 'TIME' in h or 'DATE' in h:                          return 'dt'
        if 'PRICE' in h:                                         return 'price'
        if 'PNL' in h or 'BALANCE' in h or 'COMISIONES' in h:  return 'money'
        if h == 'ROI':                                           return 'pct'
        if 'APALANCAMIENTO' in h:                                return 'apal'
        return 'gen'

    col_keys = [_fmt_key(c) for c in cols]

    # Índices de columnas especiales
    pnl_neto_idx   = next((i for i, c in enumerate(cols) if 'PNL' in c and 'NETO' in c), None)
    balance_idx    = next((i for i, c in enumerate(cols) if c == 'BALANCE'), None)
    entry_time_idx = next((i for i, c in enumerate(cols) if c == 'ENTRY_TIME'), None)

    # ── Anchos de columna: muestrea 20 filas, O(cols) ───────────────────────
    for i, col_name in enumerate(cols):
        max_len = len(str(col_name))
        if n_rows > 0:
            sample_len = df.iloc[:20, i].astype(str).str.len().max()
            if pd.notna(sample_len):
                max_len = max(max_len, int(sample_len))
        ws.set_column(i, i, min((max_len + 2) * 1.15, 28))

    # ── HEADERS ─────────────────────────────────────────────────────────────
    for i, col_name in enumerate(cols):
        ws.write(0, i, col_name, hdr_fmt)

    # ── DATOS: O(n) en C nativo, ~20-50x más rápido que openpyxl ───────────
    for r_idx, row_data in enumerate(df.itertuples(index=False), start=1):
        alt = (r_idx % 2 == 0)
        ws.set_row(r_idx, 18)

        for c_idx, val in enumerate(row_data):
            fmt = FMT[col_keys[c_idx]][1 if alt else 0]

            if val is None or (isinstance(val, float) and pd.isna(val)):
                ws.write_blank(r_idx, c_idx, None, fmt)
            elif isinstance(val, (pd.Timestamp, datetime.datetime)):
                try:
                    dt = val.to_pydatetime() if hasattr(val, 'to_pydatetime') else val
                    dt = dt.replace(tzinfo=None)
                    ws.write_datetime(r_idx, c_idx, dt, fmt)
                except Exception:
                    ws.write_string(r_idx, c_idx, str(val), fmt)
            elif isinstance(val, bool):
                ws.write_boolean(r_idx, c_idx, val, fmt)
            elif isinstance(val, (int, float)):
                ws.write_number(r_idx, c_idx, float(val), fmt)
            else:
                ws.write_string(r_idx, c_idx, str(val), fmt)

    # ── FORMATO CONDICIONAL PNL NETO ────────────────────────────────────────
    if pnl_neto_idx is not None and n_rows > 0:
        fmt_green = wb.add_format({**_BASE, 'bg_color': '#E8F5E9',
                                    'font_color': '#00897B', 'bold': True})
        fmt_red   = wb.add_format({**_BASE, 'bg_color': '#FFEBEE',
                                    'font_color': '#C62828', 'bold': True})
        ws.conditional_format(1, pnl_neto_idx, n_rows, pnl_neto_idx,
                               {'type': 'cell', 'criteria': '>', 'value': 0, 'format': fmt_green})
        ws.conditional_format(1, pnl_neto_idx, n_rows, pnl_neto_idx,
                               {'type': 'cell', 'criteria': '<', 'value': 0, 'format': fmt_red})

    # ── FILA DE SEPARACIÓN entre datos y gráfico/tabla ──────────────────────
    BLOCK_ROW = n_rows + 2   # fila 0-indexed donde empieza bloque inferior

    # ── GRÁFICO: col B (índice 1), 500px ancho × 430px alto ─────────────────
    # 500px ≈ 8 columnas × ~62px → termina antes de col J
    if balance_idx is not None and n_rows > 0:
        bal_series = df.iloc[:, balance_idx].dropna()
        y_min_raw  = float(bal_series.min())
        y_max_raw  = float(bal_series.max())
        data_range = y_max_raw - y_min_raw if y_max_raw != y_min_raw else max(abs(y_max_raw) * 0.1, 1.0)

        # Margen del 3% arriba y abajo
        margin     = data_range * 0.03
        y_min_axis = y_min_raw - margin
        y_max_axis = y_max_raw + margin
        major_unit = _nice_major_unit(data_range, target_ticks=6)

        chart = wb.add_chart({'type': 'line'})

        bal_col_letter = xl_col_to_name(balance_idx)
        series_cfg = {
            'values': f"=Trades!${bal_col_letter}$2:${bal_col_letter}${n_rows + 1}",
            'line':   {'color': '#3A86FF', 'width': 1.75, 'smooth': True},
            'marker': {'type': 'none'},
        }
        if entry_time_idx is not None:
            et_letter = xl_col_to_name(entry_time_idx)
            series_cfg['categories'] = f"=Trades!${et_letter}$2:${et_letter}${n_rows + 1}"

        chart.add_series(series_cfg)
        chart.set_title({'name': 'EVOLUCIÓN DEL BALANCE', 'name_font': {'size': 12, 'bold': True}})

        chart.set_y_axis({
            'name':            'Balance ($)',
            'name_font':       {'size': 9, 'bold': False},
            'num_format':      '#,##0.00',
            'num_font':        {'size': 8},
            'min':             y_min_axis,
            'max':             y_max_axis,
            'major_unit':      major_unit,
            'major_gridlines': {'visible': False},
            'minor_gridlines': {'visible': False},
            'line':            {'none': True},
        })
        chart.set_x_axis({
            'num_format':      'dd/mm/yy',
            'num_font':        {'size': 7},
            'major_gridlines': {'visible': False},
            'major_tick_mark': 'outside',
            'line':            {'color': '#CCCCCC'},
        })
        chart.set_legend({'none': True})
        chart.set_plotarea({'border': {'none': True}})
        chart.set_chartarea({'border': {'color': '#E0E0E0'}, 'fill': {'color': '#FAFBFF'}})

        # Tamaño: 1000 × 430 px (Ancho 10)
        chart.set_size({'width': 1000, 'height': 430})

        # Anclar: fila=BLOCK_ROW (0-indexed), columna=1 (B)
        ws.insert_chart(BLOCK_ROW, 1, chart, {'x_offset': 2, 'y_offset': 5})

    # ── TABLA DE PARÁMETROS: Col B, debajo del gráfico ──────────────────────
    # El gráfico ocupa ~22-23 filas (430px). Ponemos la tabla en la 24.
    TABLE_COL = 1    # B (0-indexed)
    TABLE_ROW = BLOCK_ROW + 24

    # Fuente 13pt = visualmente "2x" respecto al cuerpo de 10pt
    _TBL = dict(font_name='Arial', font_size=13, valign='vcenter', indent=1)

    title_fmt = wb.add_format({**_TBL,
        'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#1A1A2E',
        'align': 'left',
        'left': 5,   'left_color':   '#3A86FF',
        'top': 5,    'top_color':    '#3A86FF',
        'right': 5,  'right_color':  '#3A86FF',
        'bottom': 1, 'bottom_color': '#E8EDF5',
    })

    def _lbl_fmt(is_last=False):
        return wb.add_format({**_TBL,
            'bold': True, 'font_color': '#1A1A2E', 'bg_color': '#E3EAF6',
            'align': 'left',
            'left': 5,  'left_color':   '#3A86FF',
            'top': 1,   'top_color':    '#E8EDF5',
            'right': 1, 'right_color':  '#E8EDF5',
            'bottom': 5 if is_last else 1,
            'bottom_color': '#3A86FF' if is_last else '#E8EDF5',
        })

    def _val_fmt(num_fmt='#,##0.00', is_last=False):
        return wb.add_format({**_TBL,
            'bold': True, 'font_color': '#0F3460', 'bg_color': '#FFFFFF',
            'align': 'right', 'num_format': num_fmt,
            'left': 1,   'left_color':   '#E8EDF5',
            'top': 1,    'top_color':    '#E8EDF5',
            'right': 5,  'right_color':  '#3A86FF',
            'bottom': 5 if is_last else 1,
            'bottom_color': '#3A86FF' if is_last else '#E8EDF5',
        })

    items = [
        ("SALDO USADO",    val_saldo,   '#,##0.00 $', False),
        ("VOLUMEN MÁX.",   val_volumen, '#,##0.00 $', False),
        ("APALANCAMIENTO", val_apal,    '0.00"x"',    True),
    ]

    # Título de la tarjeta (fusionado 2 columnas)
    ws.set_row(TABLE_ROW, 32)
    ws.merge_range(TABLE_ROW, TABLE_COL, TABLE_ROW, TABLE_COL + 1,
                   '⬛  PARÁMETROS DEL TRIAL', title_fmt)

    # Filas de datos
    for i, (label, value, num_fmt, is_last) in enumerate(items):
        r = TABLE_ROW + i + 1
        ws.set_row(r, 30)   # 30 px → visible grande
        ws.write(r, TABLE_COL,     label, _lbl_fmt(is_last))
        ws.write(r, TABLE_COL + 1, value, _val_fmt(num_fmt, is_last))

    # Ancho de columnas de la tabla
    ws.set_column(TABLE_COL,     TABLE_COL,     26)   # etiqueta
    ws.set_column(TABLE_COL + 1, TABLE_COL + 1, 20)   # valor

    wb.close()


# ==============================================================================
# FALLBACK OPENPYXL
# ==============================================================================

def _escribir_trades_openpyxl_fallback(
    filepath: str,
    df: pd.DataFrame,
    val_saldo: float = 0,
    val_volumen: float = 0,
    val_apal: float = 0,
):
    from openpyxl.formatting.rule import CellIsRule

    df.to_excel(filepath, index=False, sheet_name="Trades", engine='openpyxl')
    wb = load_workbook(filepath)
    ws = wb.active
    ws.sheet_view.showGridLines = False

    max_col = ws.max_column
    max_row = ws.max_row

    border   = _make_border_op(COLORS["border_color"])
    fill_hdr = PatternFill("solid", fgColor=COLORS["header_bg_metrics"])
    fill_alt = PatternFill("solid", fgColor=COLORS["row_alt"])
    fill_w   = PatternFill("solid", fgColor="FFFFFF")
    font_hdr = Font(name=FONT_TITLE, size=11, bold=True, color="FFFFFF")
    font_b   = Font(name=FONT_BODY,  size=10, color=COLORS["text_dark"])
    align_c  = Alignment(horizontal='center', vertical='center')
    align_cw = Alignment(horizontal='center', vertical='center', wrap_text=True)

    col_headers = {c: str(ws.cell(1, c).value or "").upper() for c in range(1, max_col + 1)}
    col_fmt = {}
    pnl_col = None
    for c, hdr in col_headers.items():
        if "TIME" in hdr or "DATE" in hdr:
            col_fmt[c] = "DD/MM/YY HH:MM"
        elif "PRICE" in hdr:
            col_fmt[c] = "#,##0.00"
        elif "PNL" in hdr or "BALANCE" in hdr or "COMISIONES" in hdr:
            col_fmt[c] = "#,##0.00"
        else:
            col_fmt[c] = "General"
        if "PNL" in hdr and "NETO" in hdr:
            pnl_col = c

    for col in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = font_hdr; cell.alignment = align_cw
        cell.fill = fill_hdr; cell.border = border
    ws.row_dimensions[1].height = 28

    for r in range(2, max_row + 1):
        rf = fill_alt if r % 2 == 0 else fill_w
        ws.row_dimensions[r].height = 18
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font_b; cell.alignment = align_c
            cell.border = border; cell.fill = rf
            cell.number_format = col_fmt.get(c, "General")

    if pnl_col:
        col_letter = get_column_letter(pnl_col)
        rng = f"{col_letter}2:{col_letter}{max_row}"
        ws.conditional_formatting.add(rng, CellIsRule(
            operator='greaterThan', formula=['0'],
            fill=PatternFill("solid", fgColor=COLORS["success_bg"]),
            font=Font(name=FONT_BODY, size=10, color=COLORS["accent_green"], bold=True)
        ))
        ws.conditional_formatting.add(rng, CellIsRule(
            operator='lessThan', formula=['0'],
            fill=PatternFill("solid", fgColor=COLORS["danger_bg"]),
            font=Font(name=FONT_BODY, size=10, color=COLORS["accent_red"], bold=True)
        ))

    _col_widths_fast_op(ws, max_col, max_row)
    ws.freeze_panes = ws.cell(row=2, column=1)

    # Gráfico openpyxl con eje Y escalado
    col_map = {str(ws.cell(1, c).value or "").upper().strip(): c for c in range(1, max_col + 1)}
    if "BALANCE" in col_map:
        from openpyxl.chart import LineChart, Reference
        bal_col_idx = col_map["BALANCE"]
        bal_data = [ws.cell(r, bal_col_idx).value for r in range(2, max_row + 1)
                    if isinstance(ws.cell(r, bal_col_idx).value, (int, float))]
        if bal_data:
            data_range = max(bal_data) - min(bal_data) or 1
            margin = data_range * 0.03
            chart2 = LineChart()
            chart2.title  = "EVOLUCIÓN DEL BALANCE"
            chart2.legend = None
            chart2.y_axis.numFmt = '#,##0.00'
            chart2.y_axis.scaling.min = min(bal_data) - margin
            chart2.y_axis.scaling.max = max(bal_data) + margin
            chart2.y_axis.majorGridlines = None
            chart2.x_axis.majorGridlines = None
            data = Reference(ws, min_col=bal_col_idx, min_row=1, max_row=max_row)
            chart2.add_data(data, titles_from_data=True)
            s1 = chart2.series[0]
            s1.graphicalProperties.line.solidFill = "3A86FF"
            s1.graphicalProperties.line.width = 22860
            s1.smooth = True
            s1.marker.symbol = "none"
            chart2.height = 14; chart2.width = 44  # Ampliado (ancho 10 eq)
            ws.add_chart(chart2, f"B{max_row + 3}")

    # Tabla fallback debajo del gráfico (B)
    TABLE_COL_OP = 2   # openpyxl 1-indexed = B
    TABLE_ROW_OP = max_row + 33  # Justo debajo del gráfico (que es alto)
    for i, (label, value) in enumerate([("SALDO", val_saldo), ("VOLUMEN", val_volumen), ("APALANCAMIENTO", val_apal)]):
        r = TABLE_ROW_OP + i
        ws.cell(r, TABLE_COL_OP).value = label
        ws.cell(r, TABLE_COL_OP + 1).value = value

    wb.save(filepath)


# ==============================================================================
# FUNCIÓN PRINCIPAL (CSV → DASHBOARD RESUMEN)
# ==============================================================================

def convertir_resumen_csv_a_excel(
    *,
    csv_path: str,
    strategy_name: str,
    activo: Optional[str] = None,
    timeframe: Optional[str] = None,
    saldo_inicial: float = 300.0,
    excel_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    if output_dir is None and excel_path:
        output_dir = os.path.dirname(str(excel_path)) or None

    activo    = activo or "UNKNOWN"
    timeframe = timeframe or "UNKNOWN"

    df = pd.read_csv(csv_path)
    df = _normalizar_nombres(df, strategy_name)
    df_final, cols_metrics, cols_params = _organizar_y_filtrar_columnas(df)
    df_final = _ordenar_filas(df_final)

    final_excel_path = _generar_nombre_archivo(csv_path, output_dir, str(activo), strategy_name, str(timeframe))
    os.makedirs(os.path.dirname(final_excel_path) or ".", exist_ok=True)

    with pd.ExcelWriter(final_excel_path, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, startrow=1)

    _aplicar_estilo_avanzado(final_excel_path, df_final, cols_metrics, cols_params, saldo_inicial)

    path_to_return = final_excel_path
    if excel_path:
        import shutil
        try:
            if os.path.abspath(excel_path) != os.path.abspath(final_excel_path):
                shutil.move(final_excel_path, excel_path)
                path_to_return = excel_path
        except Exception:
            pass

    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    except Exception:
        pass

    return path_to_return


# ==============================================================================
# LÓGICA DE PROCESAMIENTO
# ==============================================================================

def _normalizar_nombres(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    df.columns = [str(c).upper().strip() for c in df.columns]

    rename_map = {
        "STRATEGY": "ESTRATEGIA", "SHARPE_RATIO": "SHARPE", "TRADES_POR_DIA": "TRADES_DIA",
        "N_TRADES": "TOTAL_TRADES", "NUM_TRADES": "TOTAL_TRADES", "COUNT_TRADES": "TOTAL_TRADES",
        "AVG_TRADE_DURATION": "AVG_TRADE", "DURATION_MEAN_MIN": "AVG_TRADE",
        "RETORNO_PROMEDIO": "AVG_TRADE",
        "WIN_RATE_PCT": "WINRATE_PCT", "WINRATE": "WINRATE_PCT",
        "PORC_GANADORAS": "WINRATE_PCT", "WIN_RATE": "WINRATE_PCT",
        "RACHA_GANADORA": "WIN_STREAK", "RACHA_PERDEDORA": "LOSS_STREAK",
        "MAX_DRAWDOWN_PCT": "MAX_DD_PCT", "MAX_DRAWDOWN": "MAX_DD_PCT",
        "DRAWDOWN": "MAX_DD_PCT", "DD": "MAX_DD_PCT", "DD_PCT": "MAX_DD_PCT", "MAX_DD": "MAX_DD_PCT",
        "RETURN_PCT": "ROI_PCT", "ROI": "ROI_PCT",
        "COUNT_LONGS": "LONG", "N_LONGS": "LONG", "LONGS": "LONG", "NUM_LONGS": "LONG",
        "N_TRADES_LONG": "LONG", "TRADES_LONG": "LONG",
        "COUNT_SHORTS": "SHORT", "N_SHORTS": "SHORT", "SHORTS": "SHORT", "NUM_SHORTS": "SHORT",
        "N_TRADES_SHORT": "SHORT", "TRADES_SHORT": "SHORT",
        "EXIT_SL_PCT": "SL", "P_SL": "SL", "SL_PCT": "SL",
        "EXIT_TP_PCT": "TP", "P_TP": "TP", "TP_PCT": "TP",
        "EXIT_TRAIL_ACT_PCT": "ACT", "TRAIL_ACT": "ACT",
        "EXIT_TRAIL_DIST_PCT": "DIST", "TRAIL_DIST": "DIST", "DISTANCE": "DIST",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
        elif old in df.columns and new in df.columns:
            df.drop(columns=[old], inplace=True)

    if "ESTRATEGIA" not in df.columns:
        df.insert(0, "ESTRATEGIA", strategy_name.upper())
    if "TRIAL" not in df.columns and df.index.name != "TRIAL":
        df.insert(0, "TRIAL", range(len(df)))

    new_cols = []
    for col in df.columns:
        if col in METRICS_ORDER or col in ID_COLS or col in {"SL", "TP", "ACT", "DIST"}:
            new_cols.append(col)
            continue
        clean = col
        for p in PREFIXES_TO_CLEAN:
            if clean.startswith(p):
                clean = clean[len(p):]
        new_cols.append(clean.replace("_PCT", "%").replace("PERCENTAGE", "%"))
    df.columns = new_cols
    return df


def _organizar_y_filtrar_columnas(df: pd.DataFrame):
    cols = list(df.columns)
    current_ids = [c for c in ID_COLS if c in cols]

    current_metrics = []
    for m in METRICS_ORDER:
        for t in [m, m.replace("_PCT", "%")]:
            if t in cols:
                current_metrics.append(t)
                break
    current_metrics = list(dict.fromkeys(current_metrics))

    excluded = set(current_ids + current_metrics)
    candidates = [c for c in cols if c not in excluded]

    exit_cols = {"SL", "TP", "ACT", "DIST"}
    has_trailing_data = False
    for t_col in ["ACT", "DIST"]:
        if t_col in df.columns:
            try:
                if (pd.to_numeric(df[t_col], errors='coerce').fillna(0) > 0).any():
                    has_trailing_data = True; break
            except Exception:
                pass

    very_bad = {"PROFIT", "WIN", "SALDO", "BALANCE", "DRAWDOWN", "DD",
                "ROI", "RETORNO", "NUM_", "COUNT", "TRADES", "RESULT", "METRIC"}
    exceptions = {"STOP", "SL", "TP", "TRAIL", "TIME", "PERIOD", "LEN", "FAST",
                  "SLOW", "SIGNAL", "LIMIT", "THRESHOLD", "SIGMA", "OFFSET", "ATR", "ACT", "DIST"}

    current_params = []
    for c in candidates:
        if df[c].astype(str).str.strip().eq("").all() or c.startswith("__"):
            continue
        if c in EXCLUDED_PARAMS or c.replace("%", "_PCT") in EXCLUDED_PARAMS:
            continue
        if c in exit_cols:
            if c in {"ACT", "DIST"} and not has_trailing_data:
                continue
            current_params.append(c); continue
        c_upper = c.upper()
        is_garbage = False
        for kw in METRIC_KEYWORDS_TO_DROP:
            if kw in c_upper:
                if any(bw in c_upper for bw in very_bad):
                    is_garbage = True; break
                if not any(ex in c_upper for ex in exceptions):
                    is_garbage = True; break
        if not is_garbage:
            current_params.append(c)

    current_params.sort()
    final_cols = current_ids + current_metrics + current_params
    return df[final_cols], current_metrics, current_params


def _ordenar_filas(df: pd.DataFrame) -> pd.DataFrame:
    if "SCORE" in df.columns:
        return df.sort_values("SCORE", ascending=False).reset_index(drop=True)
    if "SALDO_ACTUAL" in df.columns:
        return df.sort_values("SALDO_ACTUAL", ascending=False).reset_index(drop=True)
    return df


def _generar_nombre_archivo(csv, out_dir, activo, est, tf) -> str:
    fname = (f"RESUMEN_{activo}_{re.sub(r'[^A-Z0-9]', '', est.upper())}"
             f"_{re.sub(r'[^a-zA-Z0-9]', '', tf.lower())}.xlsx")
    return os.path.join(out_dir or os.path.dirname(csv), fname)


# ==============================================================================
# HELPERS OPENPYXL
# ==============================================================================

def _make_border_op(color: str, style: str = 'thin') -> Border:
    s = Side(style=style, color=color)
    return Border(left=s, right=s, top=s, bottom=s)


def _col_widths_fast_op(ws, max_col: int, max_row: int, header_row: int = 1, sample: int = 20):
    for col in range(1, max_col + 1):
        col_letter = get_column_letter(col)
        max_len = len(str(ws.cell(row=header_row, column=col).value or ""))
        for r in range(header_row + 1, min(header_row + sample + 1, max_row + 1)):
            try:
                v = ws.cell(row=r, column=col).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min((max_len + 2) * 1.15, 28)


# ==============================================================================
# ESTILOS DASHBOARD RESUMEN
# ==============================================================================

def _aplicar_estilo_avanzado(filepath, df, metrics_cols, params_cols, saldo_ini):
    wb = load_workbook(filepath)
    ws = wb.active
    ws.sheet_view.showGridLines = False

    max_col = ws.max_column
    max_row = ws.max_row

    n_ids         = len([c for c in ID_COLS if c in df.columns])
    n_metrics     = len(metrics_cols)
    n_params      = len(params_cols)
    start_metrics = n_ids + 1
    end_metrics   = start_metrics + n_metrics - 1
    start_params  = end_metrics + 1
    end_params    = start_params + n_params - 1

    border   = _make_border_op(COLORS["border_color"])
    fill_id  = PatternFill("solid", fgColor=COLORS["header_bg_id"])
    fill_met = PatternFill("solid", fgColor=COLORS["header_bg_metrics"])
    fill_par = PatternFill("solid", fgColor=COLORS["header_bg_params"])
    fill_alt = PatternFill("solid", fgColor=COLORS["row_alt"])
    fill_w   = PatternFill("solid", fgColor="FFFFFF")

    font_grp  = Font(name=FONT_TITLE, size=11, bold=True, color="FFFFFF")
    font_hdr  = Font(name=FONT_TITLE, size=11, bold=True, color="FFFFFF")
    font_body = Font(name=FONT_BODY,  size=10, color=COLORS["text_dark"])
    align_c   = Alignment(horizontal='center', vertical='center')
    align_cw  = Alignment(horizontal='center', vertical='center', wrap_text=True)

    def _section(c_start, c_end, label, fill):
        c = ws.cell(row=1, column=c_start)
        c.value = label; c.fill = fill; c.font = font_grp; c.alignment = align_c
        if c_end > c_start:
            ws.merge_cells(start_row=1, start_column=c_start, end_row=1, end_column=c_end)

    if n_ids > 0:
        _section(1, n_ids, "DATOS", fill_id)
    if n_metrics > 0:
        _section(start_metrics, end_metrics, "MÉTRICAS", fill_met)
    if n_params > 0:
        _section(start_params, end_params, "PARÁMETROS", fill_par)

    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 30

    for col in range(1, max_col + 1):
        cell = ws.cell(row=2, column=col)
        cell.font = font_hdr; cell.alignment = align_cw; cell.border = border
        cell.fill = fill_id if col < start_metrics else (fill_met if col <= end_metrics else fill_par)

    col_hdrs = {c: str(ws.cell(2, c).value or "").upper() for c in range(1, max_col + 1)}

    for r in range(3, max_row + 1):
        rf = fill_alt if r % 2 == 0 else fill_w
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font_body; cell.alignment = align_c
            cell.border = border; cell.fill = rf
            hdr = col_hdrs[c]
            if "DIA" in hdr and "TRADES" in hdr:
                cell.number_format = "0.00"
            elif "TRADES" in hdr:
                cell.number_format = "0"
            elif any(k in hdr for k in ("%", "PCT", "WINRATE", "ROI")):
                cell.number_format = "0.00%"
                try:
                    cell.value = float(cell.value) / 100.0
                except Exception:
                    pass
            elif any(k in hdr for k in ("SCORE", "SHARPE", "FACTOR", "SQN")):
                cell.number_format = "0.00"

    _col_widths_fast_op(ws, max_col, max_row, header_row=2, sample=20)

    col_map = {str(ws.cell(2, c).value or "").strip(): get_column_letter(c)
               for c in range(1, max_col + 1)}

    if "ROI%" in col_map:
        ws.conditional_formatting.add(
            f"{col_map['ROI%']}3:{col_map['ROI%']}{max_row}",
            DataBarRule(start_type='min', end_type='max', color="3A86FF", showValue=True))

    if "SCORE" in col_map:
        ws.conditional_formatting.add(
            f"{col_map['SCORE']}3:{col_map['SCORE']}{max_row}",
            ColorScaleRule(start_type='min', start_color='FFFFFF',
                           mid_type='percentile', mid_value=50, mid_color='E3EAF6',
                           end_type='max', end_color='B3C9F7'))

    ws.freeze_panes = ws.cell(row=3, column=start_metrics)
    wb.save(filepath)


# ==============================================================================
# EXPORTACIÓN RÁPIDA
# ==============================================================================

def exportar_trades_excel_rapido(
    df_trades: pd.DataFrame,
    resumen_csv_path: str,
    metrics: dict,
    params: dict,
    trial_number: int,
    trades_actual_base: str = "trades_trial",
    score: float = None,
    max_archivos: int = 5,
    skip_trades_file: bool = False,
):
    params = dict(params or {})
    fila = {
        "TRIAL":      trial_number,
        "SCORE":      score if score is not None else 0,
        "ESTRATEGIA": params.get("NOMBRE_COMBO", "UNKNOWN"),
    }
    for k, v in metrics.items():
        fila[k.upper()] = v

    def _aplanar(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(_aplanar(v, f"{prefix}{k.upper()}_"))
            else:
                out[f"{prefix}{k.upper()}"] = v
        return out

    fila.update(_aplanar(params))
    df_fila = pd.DataFrame([fila])

    if int(trial_number) == 0 and os.path.exists(resumen_csv_path):
        try:
            os.remove(resumen_csv_path)
        except Exception:
            pass

    mode = "w" if not os.path.exists(resumen_csv_path) else "a"
    df_fila.to_csv(resumen_csv_path, index=False, mode=mode, header=(mode == "w"))

    if not skip_trades_file:
        _gestionar_archivos_trades(df_trades, trades_actual_base, trial_number, score, max_archivos, params=params)


def _gestionar_archivos_trades(df, base_path, trial, score, max_files, params=None):
    trades_dir = os.path.dirname(base_path) or "."
    os.makedirs(trades_dir, exist_ok=True)

    s_val = score if score is not None else -999
    fpath = os.path.join(trades_dir, f"TRADES_TRIAL{trial}_SCORE{s_val:.2f}.xlsx")

    df_export = df.copy()
    for col in df_export.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        if hasattr(df_export[col].dt, "tz") and df_export[col].dt.tz is not None:
            df_export[col] = df_export[col].dt.tz_localize(None)

    df_export = _preparar_df_trades(df_export)

    saldo = (params or {}).get('__saldo_usado') or 0
    apal  = (params or {}).get('__apalancamiento_max') or 0

    try:
        _escribir_trades_xlsxwriter(fpath, df_export, saldo, saldo * apal, apal)
    except Exception:
        _escribir_trades_openpyxl_fallback(fpath, df_export, saldo, saldo * apal, apal)

    files = [f for f in os.listdir(trades_dir) if f.startswith("TRADES_TRIAL") and f.endswith(".xlsx")]
    if len(files) > max_files:
        scored = []
        for f in files:
            try:
                s = float(re.search(r"SCORE(-?\d+\.?\d*)", f).group(1))
                scored.append((s, f))
            except Exception:
                scored.append((-9999, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, f_del in scored[max_files:]:
            try:
                os.remove(os.path.join(trades_dir, f_del))
            except Exception:
                pass
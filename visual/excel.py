"""
================================================================================
VISUAL/EXCEL.PY — DASHBOARD QUANT EN EXCEL
================================================================================

PROPÓSITO:
    Generación de reportes Excel profesionales con:
    1. Filtrado inteligente de métricas (elimina ruido).
    2. Separación clara entre INPUTS (Parámetros) y OUTPUTS (Métricas).
    3. Formato condicional automático (Barras de datos, Escalas de color).

FUNCIONALIDAD:
    - `ExcelReporter`: Clase principal que se integra con el Runner.
    - Genera `RESUMEN.csv` incremental (seguridad contra crash).
    - Convierte a `.xlsx` con estilos al finalizar.

================================================================================
"""

import os
import re
from typing import Optional

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import DataBarRule, ColorScaleRule

# ==============================================================================
# CONFIGURACIÓN DE ESTILO
# ==============================================================================

COLORS = {
    "header_bg_metrics": "1A5276", # Azul Oscuro (Métricas)
    "header_bg_params":  "566573", # Gris Plomo (Parámetros)
    "header_bg_id":      "212F3D", # Negro Azulado (IDs)
    "text_white":        "FFFFFF",
    "text_dark":         "212F3D",
    "border_color":      "BDC3C7",
    "success_bg":        "D5F5E3", # Verde claro
    "danger_bg":         "FADBD8", # Rojo claro
}

FONT_TITLE = "Arial"
FONT_BODY = "Arial"

# --- 1. MÉTRICAS CLAVE (Performance & Financials) ---
# Orden estricto de aparición en la sección de MÉTRICAS.
METRICS_ORDER = [
    "SALDO_ACTUAL",
    "ROI_PCT",
    "PROFIT_FACTOR",
    "WINRATE_PCT",
    "TOTAL_TRADES",
    "TRADES_DIA",
    "MAX_DD_PCT",      # Drawdown
    "SHARPE",
    "SQN",
    "ESTABILIDAD",     # Añadido a métricas clave
    "AVG_TRADE",
    "EXPECTATIVA",
    "WIN_STREAK",
    "LOSS_STREAK",
    "NUM_LONGS",
    "NUM_SHORTS"
]

# --- 2. COLUMNAS DE IDENTIFICACIÓN ---
ID_COLS = ["TRIAL", "ESTRATEGIA", "SCORE"]

# --- 3. PARÁMETROS A EXCLUIR (Exclusión Directa) ---
EXCLUDED_PARAMS = {
    "NOMBRE_COMBO", "EXIT_TYPE", "CANTIDAD",
    "PERTURBADO", "SEED", "ACTIVO", "TIMEFRAME", "TF", "ASSET", "SYMBOL",
    "RESULTADO", "METRICS", "COMBO", "ESTATEGIA",
}

# --- 4. FILTRO HEURÍSTICO DE MÉTRICAS BASURA ---
METRIC_KEYWORDS_TO_DROP = [
    # Tipos de resultados financieros
    "PROFIT", "LOSS", "PNL", "NET", "GROSS", "SALDO", "BALANCE", "RETORNO", "RETURN",
    "ROI", "BENEFICIO", "RIESGO", "RISK", "REWARD", "COMISION", "FEES",

    # Estadísticas de trading
    "WIN", "GANADORA", "PERDEDORA", "ACIERTO", "RATE", "PCT", "PORC_", "PERCENT",
    "DRAWDOWN", "DD", "RACHA", "STREAK", "UNDERWATER",

    # Ratios y Estadísticas matemáticas
    "RATIO", "FACTOR", "SHARPE", "SORTINO", "CALMAR", "SQN", "EXPECTATIVA", "KELLY",
    "AVG", "MEAN", "MEDIAN", "STD", "VAR", "MAX", "MIN", "SUM", "TOTAL",
    "ESTABILIDAD", # Si aparece en params por error, se borra (ya está en metrics)

    # Conteos
    "COUNT", "NUM_", "N_", "TRADES", "LONGS", "SHORTS", "CANTIDAD_OP",

    # Otros
    "METRIC", "RESULT", "BEST", "WORST", "DIA_OPERADO", "DURATION", "TIME"
]

# Prefijos a limpiar visualmente
PREFIXES_TO_CLEAN = [
    "ESTRATEGIA_PARAMS_", "STRATEGY_PARAMS_", "PARAM_", "PARAMS_",
    "INDICATOR_", "CONFIG_", "METRICS_"
]


# ==============================================================================
# EXCEL REPORTER - Reporter para integración con OptimizationRunner
# ==============================================================================
import csv
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ReporterProtocol(Protocol):
    """Protocolo base para reporters."""
    def needs_dataframe(self, score: float) -> bool: ...
    def on_trial_end(self, artifacts: Any) -> None: ...
    def on_strategy_end(self, strategy_name: str, study: Any) -> None: ...


@dataclass
class ExcelReporter:
    """
    Excel exporter wrapper - ULTRA OPTIMIZADO (v3.1).
    
    Genera:
    - RESUMEN.csv incremental (cada trial)
    - RESUMEN_DASHBOARD.xlsx al final (con formato profesional)
    - TRADES_TRIAL{n}_SCORE{s}.xlsx para los top N trials
    """

    resumen_path: str = "resultados/excel/resumen.xlsx"
    trades_base_dir: str = "resultados/excel"
    max_archivos: int = 5
    use_fast_mode: bool = True
    
    _csv_resumen_path: Optional[str] = field(default=None, init=False, repr=False)
    _resumen_rows: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _trade_candidates: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _min_candidate_score: float = field(default=float("-inf"), init=False, repr=False)
    _activo: Optional[str] = field(default=None, init=False, repr=False)

    def needs_dataframe(self, score: float) -> bool:
        return False

    @staticmethod
    def _safe_activo_name(activo: str) -> str:
        return str(activo).strip().replace(" ", "_").upper() if activo else "DEFAULT"

    def _excel_dir_for(self, activo: str) -> str:
        return self.trades_base_dir

    def _update_min_score(self):
        if self._trade_candidates:
            self._min_candidate_score = min(c["score"] for c in self._trade_candidates)
        else:
            self._min_candidate_score = float("-inf")

    def on_trial_end(self, artifacts) -> None:
        params_src = getattr(artifacts, "params_reporting", None) or artifacts.params
        activo = None
        if isinstance(params_src, dict):
            activo = params_src.get("__activo") or params_src.get("ACTIVO") or params_src.get("activo")
        
        self._activo = activo
        score = artifacts.score if artifacts.score is not None else 0.0
        
        params = dict(params_src)
        params["NOMBRE_COMBO"] = artifacts.strategy_name
        
        resumen_row = {
            "trial_number": artifacts.trial_number,
            "score": score,
            "metrics": deepcopy(artifacts.metrics) if artifacts.metrics else {},
            "params": {k: v for k, v in params.items() if not str(k).startswith("__")},
            "perturbado": artifacts.perturbado,
            "perturb_seed": artifacts.perturb_seed,
            "strategy_name": artifacts.strategy_name,
        }
        self._resumen_rows.append(resumen_row)

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
            candidate = {
                "score": score,
                "trial_number": artifacts.trial_number,
                "trades": artifacts.trades,
                "params": params,
                "metrics": artifacts.metrics,
                "perturbado": artifacts.perturbado,
                "perturb_seed": artifacts.perturb_seed,
            }
            self._trade_candidates.append(candidate)
            
            if len(self._trade_candidates) > self.max_archivos:
                self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
                removed = self._trade_candidates.pop()
                del removed
            
            self._update_min_score()

    def on_strategy_end(self, strategy_name: str, study) -> None:
        if not self._resumen_rows:
            return
        
        activo = self._activo
        base_dir = self.trades_base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        activo_safe = self._safe_activo_name(str(activo) if activo else "DEFAULT")
        csv_path = os.path.join(base_dir, "RESUMEN.csv")
        
        self._write_resumen_csv(csv_path)
        
        self._trade_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        for candidate in self._trade_candidates[:self.max_archivos]:
            try:
                self._write_trades_excel(base_dir, candidate)
            except Exception as e:
                logger.warning(f"Error guardando trades trial {candidate['trial_number']}: {e}")
        
        try:
            convertir_resumen_csv_a_excel(
                csv_path=csv_path,
                strategy_name=strategy_name,
                activo=activo_safe,
                output_dir=base_dir
            )
        except Exception as e:
            logger.warning(f"Error generando Dashboard Excel: {e}")
        
        self._resumen_rows = []
        self._trade_candidates = []
        self._min_candidate_score = float("-inf")

    def _write_resumen_csv(self, csv_path: str):
        if not self._resumen_rows:
            return
        
        all_keys = set()
        for row in self._resumen_rows:
            all_keys.add("trial")
            all_keys.add("score")
            all_keys.add("strategy")
            if row.get("metrics"):
                all_keys.update(row["metrics"].keys())
            if row.get("params"):
                all_keys.update(f"param_{k}" for k in row["params"].keys())
        
        columns = ["trial", "score", "strategy"]
        metric_cols = sorted([k for k in all_keys if k not in columns and not k.startswith("param_")])
        param_cols = sorted([k for k in all_keys if k.startswith("param_")])
        columns.extend(metric_cols)
        columns.extend(param_cols)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            
            for row in self._resumen_rows:
                csv_row = {
                    "trial": row["trial_number"],
                    "score": row["score"],
                    "strategy": row["strategy_name"],
                }
                if row.get("metrics"):
                    csv_row.update(row["metrics"])
                if row.get("params"):
                    csv_row.update({f"param_{k}": v for k, v in row["params"].items()})
                
                writer.writerow(csv_row)

    def _write_trades_excel(self, trades_dir: str, candidate: Dict[str, Any]):
        trades = candidate["trades"]
        if trades is None or (hasattr(trades, "empty") and trades.empty):
            return
        
        if hasattr(trades, "to_pandas"):
            df_trades = trades.to_pandas()
        else:
            df_trades = trades

        try:
            df_trades = df_trades.copy()
            if isinstance(df_trades.index, pd.DatetimeIndex) and df_trades.index.tz is not None:
                df_trades.index = df_trades.index.tz_localize(None)
            for col in df_trades.columns:
                try:
                    if isinstance(df_trades[col].dtype, pd.DatetimeTZDtype):
                        df_trades[col] = df_trades[col].dt.tz_localize(None)
                except Exception:
                    continue
            for col in df_trades.columns:
                if df_trades[col].dtype != object:
                    continue
                s = df_trades[col]
                try:
                    converted = pd.to_datetime(s, errors="ignore", utc=True)
                    if isinstance(converted.dtype, pd.DatetimeTZDtype):
                        df_trades[col] = converted.dt.tz_localize(None)
                        continue
                except Exception:
                    pass
                try:
                    sample = next((v for v in s.head(50).tolist() if v is not None), None)
                    if sample is None:
                        continue
                    tzinfo = getattr(sample, "tzinfo", None)
                    if tzinfo is None:
                        continue
                    df_trades[col] = s.apply(
                        lambda v: v.replace(tzinfo=None)
                        if hasattr(v, "tzinfo") and v.tzinfo is not None else v
                    )
                except Exception:
                    continue
        except Exception:
            pass
        
        score_str = f"{candidate['score']:.2f}".replace(".", "_")
        filename = f"TRADES_TRIAL{candidate['trial_number']}_SCORE{score_str}.xlsx"
        filepath = os.path.join(trades_dir, filename)
        
        df_trades.to_excel(filepath, index=False, sheet_name="Trades")


# ==============================================================================
# FUNCIÓN PRINCIPAL
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
    """Genera Excel limpio separando Métricas Clave de Parámetros Reales."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    if output_dir is None and excel_path:
        output_dir = os.path.dirname(str(excel_path)) or None

    activo = activo or "UNKNOWN"
    timeframe = timeframe or "UNKNOWN"

    # 1. Cargar
    df = pd.read_csv(csv_path)

    # 2. Normalizar Nombres
    df = _normalizar_nombres(df, strategy_name)

    # 3. Filtrado Inteligente
    df_final, cols_metrics, cols_params = _organizar_y_filtrar_columnas(df)

    # 4. Ordenar filas
    df_final = _ordenar_filas(df_final)

    # 5. Guardar
    final_excel_path = _generar_nombre_archivo(
        csv_path, output_dir, str(activo), strategy_name, str(timeframe)
    )
    os.makedirs(os.path.dirname(final_excel_path) or ".", exist_ok=True)

    with pd.ExcelWriter(final_excel_path, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, startrow=1)

    # 6. Estilos
    _aplicar_estilo_avanzado(
        final_excel_path,
        df_final,
        cols_metrics,
        cols_params,
        saldo_inicial
    )

    return final_excel_path


# ==============================================================================
# LÓGICA DE PROCESAMIENTO
# ==============================================================================

def _normalizar_nombres(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Normaliza nombres a mayúsculas y estandariza métricas clave."""
    df.columns = [str(c).upper().strip() for c in df.columns]

    # Mapeo a nombres estándar de MODELOX
    # Mapeamos variaciones comunes a las claves de METRICS_ORDER
    rename_map = {
        # ID columns
        "STRATEGY": "ESTRATEGIA",
        
        # Sharpe
        "SHARPE_RATIO": "SHARPE",
        
        # Trades por día
        "TRADES_POR_DIA": "TRADES_DIA",
        
        # Total trades
        "N_TRADES": "TOTAL_TRADES",
        "NUM_TRADES": "TOTAL_TRADES",
        "COUNT_TRADES": "TOTAL_TRADES",
        
        # Duración/Avg Trade
        "AVG_TRADE_DURATION": "AVG_TRADE",
        "DURATION_MEAN_MIN": "AVG_TRADE",
        "RETORNO_PROMEDIO": "AVG_TRADE",
        
        # Winrate
        "WIN_RATE_PCT": "WINRATE_PCT",
        "WINRATE": "WINRATE_PCT",
        "PORC_GANADORAS": "WINRATE_PCT",
        
        # Rachas / Streaks
        "RACHA_GANADORA": "WIN_STREAK",
        "RACHA_PERDEDORA": "LOSS_STREAK",

        # Variantes de Drawdown (CRÍTICO: Capturar todas)
        "MAX_DRAWDOWN_PCT": "MAX_DD_PCT",
        "MAX_DRAWDOWN": "MAX_DD_PCT",
        "DRAWDOWN": "MAX_DD_PCT",
        "DD": "MAX_DD_PCT",
        "DD_PCT": "MAX_DD_PCT",
        "MAX_DD": "MAX_DD_PCT",

        # ROI
        "RETURN_PCT": "ROI_PCT",
        "ROI": "ROI_PCT",
        
        # Saldo
        "NET_PROFIT": "SALDO_ACTUAL",
        "PNL_NETO": "SALDO_ACTUAL",
        "NET_PNL": "SALDO_ACTUAL",
        
        # Estandarización de longs/shorts
        "COUNT_LONGS": "NUM_LONGS", "N_LONGS": "NUM_LONGS", "LONGS": "NUM_LONGS",
        "N_TRADES_LONG": "NUM_LONGS",
        "COUNT_SHORTS": "NUM_SHORTS", "N_SHORTS": "NUM_SHORTS", "SHORTS": "NUM_SHORTS",
        "N_TRADES_SHORT": "NUM_SHORTS",
    }

    # Renombrar solo si el destino no existe
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
        elif old in df.columns and new in df.columns:
            # Si ambas existen, borramos la vieja
            df.drop(columns=[old], inplace=True)

    # Asegurar columna ESTRATEGIA
    if "ESTRATEGIA" not in df.columns:
        df.insert(0, "ESTRATEGIA", strategy_name.upper())
    
    # Asegurar columna TRIAL si no existe
    if "TRIAL" not in df.columns and df.index.name != "TRIAL":
        df.insert(0, "TRIAL", range(len(df)))

    # Limpieza visual de prefijos
    new_cols = []
    for col in df.columns:
        if col in METRICS_ORDER or col in ID_COLS:
            new_cols.append(col)
            continue

        clean = col
        for p in PREFIXES_TO_CLEAN:
            if clean.startswith(p):
                clean = clean[len(p):]

        clean = clean.replace("_PCT", "%").replace("PERCENTAGE", "%")
        new_cols.append(clean)

    df.columns = new_cols
    return df


def _organizar_y_filtrar_columnas(df: pd.DataFrame):
    cols = list(df.columns)

    # 1. IDs
    current_ids = [c for c in ID_COLS if c in cols]

    # 2. Métricas Clave (Estricto)
    current_metrics = []
    for m in METRICS_ORDER:
        targets = [m, m.replace("_PCT", "%")]
        found = None
        for t in targets:
            if t in cols:
                found = t
                break
        if found:
            current_metrics.append(found)

    current_metrics = list(dict.fromkeys(current_metrics))

    # 3. Parámetros (Filtro)
    excluded = set(current_ids + current_metrics)
    candidates = [c for c in cols if c not in excluded]

    current_params = []
    for c in candidates:
        # A. Vacíos
        if df[c].astype(str).str.strip().eq("").all():
            continue

        # B. Internos
        if c.startswith("__"):
            continue

        # C. Lista Negra
        if c in EXCLUDED_PARAMS or c.replace("%", "_PCT") in EXCLUDED_PARAMS:
            continue

        # D. FILTRO HEURÍSTICO
        is_garbage = False
        c_upper = c.upper()
        for kw in METRIC_KEYWORDS_TO_DROP:
            if kw in c_upper:
                # Excepciones técnicas
                exceptions = ["STOP", "SL", "TP", "TRAIL", "TIME", "PERIOD", "LEN", "FAST", "SLOW", "SIGNAL", "LIMIT", "THRESHOLD", "SIGMA", "OFFSET", "ATR"]

                has_exception = any(exc in c_upper for exc in exceptions)

                # Palabras que invalidan incluso excepciones
                very_bad_words = ["PROFIT", "WIN", "SALDO", "BALANCE", "DRAWDOWN", "DD", "ROI", "RETORNO", "NUM_", "COUNT", "TRADES", "RESULT", "METRIC"]
                is_very_bad = any(bw in c_upper for bw in very_bad_words)

                if is_very_bad:
                    is_garbage = True
                    break

                if not has_exception:
                    is_garbage = True
                    break

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
    clean_est = re.sub(r'[^A-Z0-9]', '', est.upper())
    clean_tf = re.sub(r'[^a-zA-Z0-9]', '', tf.lower())
    fname = f"RESUMEN_{activo}_{clean_est}_{clean_tf}.xlsx"
    return os.path.join(out_dir or os.path.dirname(csv), fname)


# ==============================================================================
# ESTILOS AVANZADOS
# ==============================================================================

def _aplicar_estilo_avanzado(filepath, df, metrics_cols, params_cols, saldo_ini):
    wb = load_workbook(filepath)
    ws = wb.active
    ws.sheet_view.showGridLines = False

    max_col = ws.max_column
    max_row = ws.max_row

    n_ids = len([c for c in ID_COLS if c in df.columns])
    n_metrics = len(metrics_cols)
    n_params = len(params_cols)

    start_metrics = n_ids + 1
    end_metrics = start_metrics + n_metrics - 1
    start_params = end_metrics + 1
    end_params = start_params + n_params - 1

    # 1. TÍTULOS
    font_group = Font(name=FONT_TITLE, size=12, bold=True, color=COLORS["text_white"])
    align_group = Alignment(horizontal='center', vertical='center')

    for c in range(1, start_metrics):
        ws.cell(row=1, column=c).fill = PatternFill("solid", fgColor=COLORS["header_bg_id"])

    if n_metrics > 0:
        c = ws.cell(row=1, column=start_metrics)
        c.value = "MÉTRICAS CLAVE"
        c.fill = PatternFill("solid", fgColor=COLORS["header_bg_metrics"])
        c.font = font_group
        c.alignment = align_group
        if n_metrics > 1:
            ws.merge_cells(start_row=1, start_column=start_metrics, end_row=1, end_column=end_metrics)

    if n_params > 0:
        c = ws.cell(row=1, column=start_params)
        c.value = "PARÁMETROS ESTRATEGIA"
        c.fill = PatternFill("solid", fgColor=COLORS["header_bg_params"])
        c.font = font_group
        c.alignment = align_group
        if n_params > 1:
            ws.merge_cells(start_row=1, start_column=start_params, end_row=1, end_column=end_params)

    # 2. HEADERS
    font_header = Font(name=FONT_TITLE, size=12, bold=True, color=COLORS["text_white"])
    border_full = Border(
        left=Side(style='thin', color=COLORS["border_color"]),
        right=Side(style='thin', color=COLORS["border_color"]),
        top=Side(style='thin', color=COLORS["border_color"]),
        bottom=Side(style='thin', color=COLORS["border_color"])
    )

    for col in range(1, max_col + 1):
        cell = ws.cell(row=2, column=col)
        cell.font = font_header
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border_full

        if col < start_metrics:
            cell.fill = PatternFill("solid", fgColor=COLORS["header_bg_id"])
        elif col <= end_metrics:
            cell.fill = PatternFill("solid", fgColor=COLORS["header_bg_metrics"])
        else:
            cell.fill = PatternFill("solid", fgColor=COLORS["header_bg_params"])

        val_len = len(str(cell.value)) if cell.value else 0
        ws.column_dimensions[get_column_letter(col)].width = min(max(10, val_len + 2), 22)

    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 30

    # 3. DATOS
    font_body = Font(name=FONT_BODY, size=12, color=COLORS["text_dark"])

    for r in range(3, max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font_body
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_full

            header_val = str(ws.cell(2, c).value).upper()

            if isinstance(cell.value, (int, float)):
                if "TRADES" in header_val and "DIA" in header_val:
                    cell.number_format = "0.00"
                elif "TRADES" in header_val or "NUM_" in header_val:
                    cell.number_format = "0"
                elif "SCORE" in header_val or "%" in header_val or "PCT" in header_val or "RATIO" in header_val or "SHARPE" in header_val or "FACTOR" in header_val or "ESTABILIDAD" in header_val:
                    cell.number_format = "0.00"
                elif "SALDO" in header_val or "PROFIT" in header_val or "PNL" in header_val:
                    cell.number_format = "#,##0.00"

    # 4. CONDITIONAL FORMATTING
    col_map = {str(ws.cell(2, c).value).strip(): get_column_letter(c) for c in range(1, max_col + 1)}

    if "ROI%" in col_map:
        col_roi = col_map["ROI%"]
        ws.conditional_formatting.add(f"{col_roi}3:{col_roi}{max_row}", DataBarRule(
            start_type='min', end_type='max', color="638EC6", showValue=True
        ))

    if "SCORE" in col_map:
        col_score = col_map["SCORE"]
        ws.conditional_formatting.add(f"{col_score}3:{col_score}{max_row}", ColorScaleRule(
            start_type='min', start_color='F8696B',
            mid_type='percentile', mid_value=50, mid_color='FFEB84',
            end_type='max', end_color='63BE7B'
        ))

    if "SALDO_ACTUAL" in col_map:
        l_idx = list(col_map.values()).index(col_map["SALDO_ACTUAL"]) + 1
        for row in range(3, max_row + 1):
            cell = ws.cell(row=row, column=l_idx)
            try:
                val = float(cell.value)
                if val >= saldo_ini * 1.5:
                    cell.fill = PatternFill("solid", fgColor=COLORS["success_bg"])
                    cell.font = Font(name=FONT_BODY, size=12, color="006100", bold=True)
                elif val < saldo_ini:
                    cell.fill = PatternFill("solid", fgColor=COLORS["danger_bg"])
                    cell.font = Font(name=FONT_BODY, size=12, color="9C0006")
            except Exception:
                pass

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
    perturbado: bool = False,
    perturb_seed: int = None,
    skip_trades_file: bool = False,
):
    """Guarda CSV maestro y trades."""
    params = dict(params or {})

    fila = {
        "TRIAL": trial_number,
        "SCORE": score if score is not None else 0,
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

    # Escribir CSV
    if int(trial_number) == 0 and os.path.exists(resumen_csv_path):
        try:
            os.remove(resumen_csv_path)
        except Exception:
            pass

    mode = "w" if not os.path.exists(resumen_csv_path) else "a"
    df_fila.to_csv(resumen_csv_path, index=False, mode=mode, header=(mode == "w"))

    # Guardar Trades
    if not skip_trades_file:
        _gestionar_archivos_trades(df_trades, trades_actual_base, trial_number, score, max_archivos)

def _gestionar_archivos_trades(df, base_path, trial, score, max_files):
    trades_dir = os.path.dirname(base_path) or "."
    os.makedirs(trades_dir, exist_ok=True)

    s_val = score if score is not None else -999
    fname = f"TRADES_TRIAL{trial}_SCORE{s_val:.2f}.xlsx"
    fpath = os.path.join(trades_dir, fname)

    df_export = df.copy()
    for col in df_export.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        if hasattr(df_export[col].dt, "tz") and df_export[col].dt.tz is not None:
            df_export[col] = df_export[col].dt.tz_localize(None)

    with pd.ExcelWriter(fpath, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='Trades', index=False)

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

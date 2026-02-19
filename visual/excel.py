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
    "header_bg_metrics": "262626", # Gris Oscuro
    "header_bg_params":  "595959", # Gris Medio
    "header_bg_id":      "000000", # Negro
    "text_white":        "FFFFFF",
    "text_dark":         "333333", # Gris muy oscuro para texto datos
    "border_color":      "E0E0E0", # Borde muy sutil
    "success_bg":        "E6F4EA", # Verde muy suave pastel
    "danger_bg":         "FCE8E6", # Rojo muy suave pastel
}

FONT_TITLE = "Arial"
FONT_BODY = "Arial"

# --- 1. MÉTRICAS CLAVE (Performance & Financials) ---
# Orden estricto de aparición en la sección de MÉTRICAS.
METRICS_ORDER = [
    "TOTAL_TRADES",
    "LONG",         # Antes NUM_LONGS
    "SHORT",        # Antes NUM_SHORTS
    "PROFIT_FACTOR",
    "WINRATE_PCT",
    "MAX_DD_PCT",
    "SHARPE",
    "SQN",
    "EXPECTATIVA"
]

# --- 2. COLUMNAS DE IDENTIFICACIÓN ---
# Orden estricto de aparición en la sección DATOS.
ID_COLS = ["TRIAL", "ESTRATEGIA", "SCORE"]

# --- 3. PARÁMETROS A EXCLUIR (Exclusión Directa y Dinámica) ---
# Se mantiene lógica de exclusión base, pero se refinará en _organizar_y_filtrar_columnas
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
    _final_excel_path: Optional[str] = field(default=None, init=False, repr=False)

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
            self._final_excel_path = convertir_resumen_csv_a_excel(
                csv_path=csv_path,
                strategy_name=strategy_name,
                activo=activo_safe,
                output_dir=base_dir,
                excel_path=self.resumen_path  # Pass the target path for duplication
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
        
        # Renombrar columnas para visualización profesional
        rename_map = {
            "entry_time": "ENTRY_TIME",
            "exit_time": "EXIT_TIME",
            "type": "SIDE",
            "entry_price": "ENTRY_PRICE",
            "exit_price": "EXIT_PRICE",
            "qty": "CANTIDAD",
            "saldo_usado": "SALDO",
            "pnl_bruto": "GROSS_PNL",
            "comision": "FEES",
            "pnl_neto": "NET_PNL",
            "pnl_pct": "ROI_%",
            "saldo_antes": "BALANCE_PRE",
            "saldo_despues": "BALANCE_POST",
            "reason": "EXIT_REASON",
            "side_int": "SIDE_INT"
        }
        df_export = df_trades.rename(columns=rename_map)
        
        # --- NUEVA LÓGICA DE TRANSFORMACIÓN DE DATOS ---
        
        # 1. Eliminar columnas innecesarias (incluyendo variantes minúsculas/mayúsculas)
        cols_to_drop = ["ENTRY_IDX", "EXIT_IDX", "SIDE_INT", "entry_idx", "exit_idx", "side_int"]
        df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns], inplace=True)
        
        # 2. Renombrar STAKE a SALDO (según petición usuario, ya hecho en map pero por seguridad)
        if "STAKE" in df_export.columns:
            df_export.rename(columns={"STAKE": "SALDO"}, inplace=True)

        # 3. Calcular VOLUMEN y APALANCAMIENTO
        # Volumen = Cantidad * Entry_Price
        # Apalancamiento = Volumen / Saldo
        try:
            qty_col = "CANTIDAD" if "CANTIDAD" in df_export.columns else "QTY"
            if qty_col in df_export.columns and "ENTRY_PRICE" in df_export.columns:
                df_export["VOLUMEN"] = df_export[qty_col] * df_export["ENTRY_PRICE"]
                
                if "SALDO" in df_export.columns:
                    # Evitar division por cero
                    df_export["APALANCAMIENTO"] = df_export.apply(
                        lambda x: x["VOLUMEN"] / x["SALDO"] if x["SALDO"] > 0 else 0, axis=1
                    )
        except Exception as e:
            logger.warning(f"Error calculando Volumen/Apalancamiento: {e}")

        # 4. Mapear EXIT_REASON numérico a String
        # 1=SL, 2=TP, 3=TRAIL, 4=TIME, 0=END
        reason_map = {
            1: "SL",
            2: "TP",
            3: "TRAIL",
            4: "TIME",
            0: "END"
        }
        if "EXIT_REASON" in df_export.columns:
            df_export["EXIT_REASON"] = df_export["EXIT_REASON"].map(reason_map).fillna("UNKNOWN")

        # 5. Corregir ROI_% (Dividir por 100 para formato Excel %)
        if "ROI_%" in df_export.columns:
            df_export["ROI_%"] = df_export["ROI_%"] / 100.0

        # 6. Convertir textos a MAYÚSCULAS
        str_cols = ["SIDE", "EXIT_REASON", "TYPE"]
        for c in str_cols:
            if c in df_export.columns:
                df_export[c] = df_export[c].astype(str).str.upper()

        # Asegurar orden y mayúsculas en columnas
        df_export.columns = [c.upper() for c in df_export.columns]

        # Reordenar columnas para mejor lectura (opcional pero recomendado)
        start_cols = ["ENTRY_TIME", "EXIT_TIME", "SIDE", "EXIT_REASON", "ENTRY_PRICE", "EXIT_PRICE", "CANTIDAD", "VOLUMEN", "SALDO", "APALANCAMIENTO", "GROSS_PNL", "FEES", "NET_PNL", "ROI_%", "BALANCE_PRE", "BALANCE_POST"]
        # Filtrar solo las que existen
        ordered_cols = [c for c in start_cols if c in df_export.columns]
        # Agregar el resto
        remaining = [c for c in df_export.columns if c not in ordered_cols]
        df_export = df_export[ordered_cols + remaining]

        
        
        filename = f"TRIAL {candidate['trial_number']}.xlsx"
        filepath = os.path.join(trades_dir, filename)
        
        # Guardar con engine openpyxl explícito
        df_export.to_excel(filepath, index=False, sheet_name="Trades", engine='openpyxl')
        
        # Aplicar estilos
        try:
            _aplicar_estilo_trades(filepath)
        except Exception as e:
            logger.warning(f"No se pudo aplicar estilo a trades: {e}")


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

    # Continue to duplication logic
    
    # 7. Renombrar a excel_path si se especificó (RESUMEN ID7.xlsx)
    path_to_return = final_excel_path
    if excel_path:
        import shutil
        try:
            target_abs = os.path.abspath(excel_path)
            source_abs = os.path.abspath(final_excel_path)
            if target_abs != source_abs:
                shutil.move(final_excel_path, excel_path)
                path_to_return = excel_path
        except Exception as e:
            pass
            
    # 8. Eliminar CSV original
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    except Exception:
        pass

    return path_to_return

    # 8. Eliminar CSV original
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
    except Exception:
        pass

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
        
        "NUM_TRADES": "TOTAL_TRADES",
        "COUNT_TRADES": "TOTAL_TRADES",
        
        # Winrate
        "WIN_RATE_PCT": "WINRATE_PCT",
        "WINRATE": "WINRATE_PCT",
        "PORC_GANADORAS": "WINRATE_PCT",
        "WIN_RATE": "WINRATE_PCT",
        
        # Variantes de Drawdown
        "MAX_DRAWDOWN_PCT": "MAX_DD_PCT",
        "MAX_DRAWDOWN": "MAX_DD_PCT",
        "DRAWDOWN": "MAX_DD_PCT",
        "DD": "MAX_DD_PCT",
        "DD_PCT": "MAX_DD_PCT",
        "MAX_DD": "MAX_DD_PCT",

        # Estandarización de longs/shorts (A "LONG" y "SHORT")
        "COUNT_LONGS": "LONG", "N_LONGS": "LONG", "LONGS": "LONG", "NUM_LONGS": "LONG",
        "N_TRADES_LONG": "LONG", "TRADES_LONG": "LONG",
        
        "COUNT_SHORTS": "SHORT", "N_SHORTS": "SHORT", "SHORTS": "SHORT", "NUM_SHORTS": "SHORT",
        "N_TRADES_SHORT": "SHORT", "TRADES_SHORT": "SHORT",
        
        # Parámetros de salida (Normalización estricta para el filtrado)
        "EXIT_SL_PCT": "SL", "P_SL": "SL", "SL_PCT": "SL",
        "EXIT_TP_PCT": "TP", "P_TP": "TP", "TP_PCT": "TP",
        "EXIT_TRAIL_ACT_PCT": "ACT", "TRAIL_ACT": "ACT",
        "EXIT_TRAIL_DIST_PCT": "DIST", "TRAIL_DIST": "DIST", "DISTANCE": "DIST",
    }

    # Renombrar solo si el destino no existe
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
        elif old in df.columns and new in df.columns:
            # Si ambas existen, borramos la vieja para evitar duplicados ambiguos
            df.drop(columns=[old], inplace=True)

    # Asegurar columna ESTRATEGIA
    if "ESTRATEGIA" not in df.columns:
        df.insert(0, "ESTRATEGIA", strategy_name.upper())
    
    # Asegurar columna TRIAL si no existe
    if "TRIAL" not in df.columns and df.index.name != "TRIAL":
        df.insert(0, "TRIAL", range(len(df)))

    # Limpieza visual de prefijos de forma genérica
    new_cols = []
    for col in df.columns:
        # Si ya está en las listas oficiales, lo dejamos tal cual
        if col in METRICS_ORDER or col in ID_COLS:
            new_cols.append(col)
            continue
        
        # Check si es uno de los params de salida ya renombrados manualmente arriba
        if col in ["SL", "TP", "ACT", "DIST"]:
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

    # 3. Parámetros (Filtro con Lógica Custom de Salidas)
    excluded = set(current_ids + current_metrics)
    candidates = [c for c in cols if c not in excluded]

    current_params = []
    
    # Detectar tipo de salida predominante para filtrar columnas
    # Buscamos columnas originales o parámetros 'param_exit_type' en el df original si fuera posible,
    # pero aquí ya están renombrados. Asumimos que si hay columnas ACT/DIST con valores no nulos/ceros, es trailing.
    # Pero la regla del usuario es explícita: "Si es FIXED solo SL y TP, si es TRAILING solo SL, TP, ACT y DISTANCE".
    # Como ExcelReporter mezcla trials, si hay CUALQUIER trial con trailing, deberíamos mostrar las columnas.
    # Pero usualmente un reporte es de una ejecución.
    
    exit_cols = {"SL", "TP", "ACT", "DIST"}
    
    # Hacemos una pasada para ver candidatos válidos
    valid_candidates = []
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
        
        valid_candidates.append(c)

    # Filtrado específico de salidas
    # Revisamos si existe ACT o DIST con valores significativos (no todos 0 o vacíos)
    has_trailing_data = False
    for t_col in ["ACT", "DIST"]:
        if t_col in df.columns:
            # Check si hay algun valor > 0 (asumiendo numericos)
            try:
                if (pd.to_numeric(df[t_col], errors='coerce').fillna(0) > 0).any():
                    has_trailing_data = True
                    break
            except:
                pass
    
    for c in valid_candidates:
        if c in exit_cols:
            if c in ["ACT", "DIST"] and not has_trailing_data:
                continue # Ocultar ACT/DIST si no hay trailing activo
            current_params.append(c)
            continue

        # D. FILTRO HEURÍSTICO (Resto de params)
        is_garbage = False
        c_upper = c.upper()
        for kw in METRIC_KEYWORDS_TO_DROP:
            if kw in c_upper:
                # Excepciones técnicas
                exceptions = ["STOP", "SL", "TP", "TRAIL", "TIME", "PERIOD", "LEN", "FAST", "SLOW", "SIGNAL", "LIMIT", "THRESHOLD", "SIGMA", "OFFSET", "ATR", "ACT", "DIST"]
                has_exception = any(exc in c_upper for exc in exceptions)
                
                # Palabras prohibidas fuertes
                very_bad_words = ["PROFIT", "WIN", "SALDO", "BALANCE", "DRAWDOWN", "DD", "ROI", "RETORNO", "NUM_", "COUNT", "TRADES", "RESULT", "METRIC"]
                if any(bw in c_upper for bw in very_bad_words):
                    is_garbage = True
                    break

                if not has_exception:
                    is_garbage = True
                    break

        if not is_garbage:
            current_params.append(c)

    # Ordenar params: Poner SL, TP, ACT, DIST al final o principio? 
    # Mejor orden alfabetico general, pero SL/TP agrupados si es posible.
    # El sort alfabetico los separará, pero es aceptable.
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

    # 1. TÍTULOS DE SECCIONES
    font_group = Font(name=FONT_TITLE, size=12, bold=True, color=COLORS["text_white"])
    align_group = Alignment(horizontal='center', vertical='center')

    # Header 1: DATOS (TRIAL, ESTRATEGIA, SCORE)
    if n_ids > 0:
        c = ws.cell(row=1, column=1)
        c.value = "DATOS"
        c.fill = PatternFill("solid", fgColor=COLORS["header_bg_id"])
        c.font = font_group
        c.alignment = align_group
        if n_ids > 1:
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_ids)

    # Header 2: MÉTRICAS
    if n_metrics > 0:
        c = ws.cell(row=1, column=start_metrics)
        c.value = "METRICAS"
        c.fill = PatternFill("solid", fgColor=COLORS["header_bg_metrics"])
        c.font = font_group
        c.alignment = align_group
        if n_metrics > 1:
            ws.merge_cells(start_row=1, start_column=start_metrics, end_row=1, end_column=end_metrics)
    
    # Header 3: PARÁMETROS
    if n_params > 0:
        c = ws.cell(row=1, column=start_params)
        c.value = "PARAMETROS"  # Usuario pidió "PARAMETROS", sin "ESTRATEGIA"
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
                # Auto-detect percentage values if > 1.0 and header says %
                # Pero mejor confiar en logica previa de dividir / 100
                pass

            if "TRADES" in header_val and "DIA" in header_val:
                cell.number_format = "0.00"
            elif "TRADES" in header_val or "NUM_" in header_val:
                cell.number_format = "0"
            elif "SCORE" in header_val or "%" in header_val or "PCT" in header_val or "RATIO" in header_val or "SHARPE" in header_val or "FACTOR" in header_val or "ESTABILIDAD" in header_val:
                # Si es % y el valor es > 1 (ej 50.0), dividir por 100 para que el 0.00% de excel cuadre
                if "%" in header_val or "PCT" in header_val or "WINRATE" in header_val or "ROI" in header_val:
                    cell.number_format = "0.00%"
                    try:
                        val = float(cell.value)
                        # Si tiene % o PCT en el nombre, lo normalizamos dividiendo por 100 
                        # para que Excel (que trata 1.0 como 100%) lo muestre correctamente.
                        # Ej: 0.34 -> 0.0034 (0.34%) | 3.00 -> 0.03 (3.00%)
                        if "%" in header_val or "PCT" in header_val:
                             cell.value = val / 100.0
                        elif abs(val) > 1.0: # Solo para WINRATE o ROI si vienen en formato entero 
                             cell.value = val / 100.0
                    except:
                        pass
                else:
                    cell.number_format = "0.00"
            elif "SALDO" in header_val or "PROFIT" in header_val or "PNL" in header_val:
                cell.number_format = "#,##0.00"

    # Auto-adjust columns based on content (RESUMEN)
    for col in range(1, max_col + 1):
        max_length = 0
        column = get_column_letter(col)
        # Check header
        header_val = ws.cell(row=2, column=col).value
        if header_val:
            max_length = len(str(header_val))
        
        # Check rows
        for i, row in enumerate(ws.iter_rows(min_row=3, max_row=min(50, max_row), min_col=col, max_col=col)):
            for cell in row:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
        
        adjusted_width = (max_length + 2) * 1.2  # Factor 1.2 para asegurar espacio para negritas
        ws.column_dimensions[column].width = min(adjusted_width, 50)


    # 4. CONDITIONAL FORMATTING
    col_map = {str(ws.cell(2, c).value).strip(): get_column_letter(c) for c in range(1, max_col + 1)}

    if "ROI%" in col_map:
        col_roi = col_map["ROI%"]
        ws.conditional_formatting.add(f"{col_roi}3:{col_roi}{max_row}", DataBarRule(
            start_type='min', end_type='max', color="638EC6", showValue=True
        ))

    if "SCORE" in col_map:
        col_score = col_map["SCORE"]
        # Minimalist Scale: White -> Subtle Grey/Green
        ws.conditional_formatting.add(f"{col_score}3:{col_score}{max_row}", ColorScaleRule(
            start_type='min', start_color='FFFFFF',
            mid_type='percentile', mid_value=50, mid_color='F1F8E9', # Very pale green
            end_type='max', end_color='C8E6C9' # Pale green
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


def _aplicar_estilo_trades(filepath: str):
    """
    Aplica estilos profesionales al Excel de Trades (similar al Dashboard).
    - Headers oscuros
    - Formato de Nímeros
    - Colores condicionales en PnL
    """
    wb = load_workbook(filepath)
    ws = wb.active
    ws.sheet_view.showGridLines = False
    
    max_col = ws.max_column
    max_row = ws.max_row
    
    # 1. HEADERS
    font_header = Font(name=FONT_TITLE, size=11, bold=True, color=COLORS["text_white"])
    border_full = Border(
        left=Side(style='thin', color=COLORS["border_color"]),
        right=Side(style='thin', color=COLORS["border_color"]),
        top=Side(style='thin', color=COLORS["border_color"]),
        bottom=Side(style='thin', color=COLORS["border_color"])
    )
    
    for col in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = font_header
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.fill = PatternFill("solid", fgColor=COLORS["header_bg_metrics"]) 
        cell.border = border_full
        
        # Ajuste ancho
        val_len = len(str(cell.value)) if cell.value else 0
        ws.column_dimensions[get_column_letter(col)].width = max(12, val_len + 4)
        
    ws.row_dimensions[1].height = 25
    
    # 2. DATA
    font_body = Font(name=FONT_BODY, size=10, color=COLORS["text_dark"])
    
    # Identify columns by name
    col_map = {str(ws.cell(1, c).value).upper().strip(): c for c in range(1, max_col + 1)}
    
    for r in range(2, max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font_body
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_full
            
            # Recuperar header para saber formato
            header = str(ws.cell(1, c).value).upper()
            
            # Formatos (2 decimales estricto)
            if "TIME" in header or "DATE" in header:
                 cell.number_format = "YYYY-MM-DD HH:MM:SS"
            elif "%" in header or "PCT" in header or "ROI" in header:
                cell.number_format = "0.00%"
            elif "PRICE" in header or "PNL" in header or "BALANCE" in header or "GROSS" in header or "FEES" in header or "SALDO" in header or "VOLUMEN" in header:
                cell.number_format = "#,##0.00"
            elif "QTY" in header or "CANTIDAD" in header:
                cell.number_format = "0.0000"
            elif "APALANCAMIENTO" in header:
                cell.number_format = "0.00x" # Formato "20.00x" queda bien
            
            # Conditional Formatting
            # PNL: Green/Red - APPLY ONLY TO NET_PNL (Others Neutral)
            if "PNL" in header or "ROI" in header:
                # Remove coloring for GROSS_PNL and ROI as requested
                if "GROSS" in header or "ROI" in header:
                    cell.font = Font(name=FONT_BODY, size=10, color=COLORS["text_dark"])
                    # Ensure no fill is applied (default)
                elif "NET_PNL" in header or "PNL" in header: # Keep coloring only for Net PnL if desired, or remove all if strictly neutral
                     try:
                        val = float(cell.value)
                        if val > 0:
                            cell.fill = PatternFill("solid", fgColor=COLORS["success_bg"])
                            cell.font = Font(name=FONT_BODY, size=10, color="006100")
                        elif val < 0:
                             cell.fill = PatternFill("solid", fgColor=COLORS["danger_bg"])
                             cell.font = Font(name=FONT_BODY, size=10, color="9C0006")
                     except:
                        pass
                    
            # SIDE: Neutral (Minimalist)
            if "SIDE" in header or "TYPE" in header:
                cell.font = Font(name=FONT_BODY, size=10, color=COLORS["text_dark"])

    # Auto-adjust columns based on content
    for col in range(1, max_col + 1):
        max_length = 0
        column = get_column_letter(col)
        # Check header length
        header_val = ws.cell(row=1, column=col).value
        if header_val:
            max_length = len(str(header_val))
        
        # Check first 50 rows for speed
        for i, row in enumerate(ws.iter_rows(min_row=2, max_row=min(52, max_row), min_col=col, max_col=col)):
            for cell in row:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
        
        adjusted_width = (max_length + 2) * 1.2 # Factor 1.2 para asegurar espacio para negritas
        ws.column_dimensions[column].width = min(adjusted_width, 50) # Cap width

    ws.freeze_panes = ws.cell(row=2, column=1)
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

    # Renombrar columnas para consistencia visual
    # Renombrar columnas para consistencia visual
    rename_map = {
        "entry_time": "ENTRY_TIME", "exit_time": "EXIT_TIME",
        "type": "SIDE", "entry_price": "ENTRY_PRICE", "exit_price": "EXIT_PRICE",
        "qty": "CANTIDAD", "saldo_usado": "SALDO",
        "pnl_bruto": "GROSS_PNL", "comision": "FEES",
        "pnl_neto": "NET_PNL", "pnl_pct": "ROI_%",
        "saldo_antes": "BALANCE_PRE", "saldo_despues": "BALANCE_POST",
        "reason": "EXIT_REASON", "side_int": "SIDE_INT"
    }
    # Solo renombra si existen las columnas
    df_export.rename(columns=rename_map, inplace=True)
    
     # --- NUEVA LÓGICA DE TRANSFORMACIÓN DE DATOS (REPLICADA EN FAST MODE) ---
    cols_to_drop = ["ENTRY_IDX", "EXIT_IDX", "SIDE_INT", "entry_idx", "exit_idx", "side_int"]
    df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns], inplace=True)

    qty_col = "CANTIDAD" if "CANTIDAD" in df_export.columns else "QTY"
    if qty_col in df_export.columns and "ENTRY_PRICE" in df_export.columns:
        df_export["VOLUMEN"] = df_export[qty_col] * df_export["ENTRY_PRICE"]
        if "SALDO" in df_export.columns:
             df_export["APALANCAMIENTO"] = df_export.apply(lambda x: x["VOLUMEN"]/x["SALDO"] if x["SALDO"]>0 else 0, axis=1)

    reason_map = {1: "SL", 2: "TP", 3: "TRAIL", 4: "TIME", 0: "END"}
    if "EXIT_REASON" in df_export.columns:
        # Si es numerico aun
        if pd.api.types.is_numeric_dtype(df_export["EXIT_REASON"]):
             df_export["EXIT_REASON"] = df_export["EXIT_REASON"].map(reason_map).fillna("UNKNOWN")

    if "ROI_%" in df_export.columns:
        df_export["ROI_%"] = df_export["ROI_%"] / 100.0

    str_cols = ["SIDE", "EXIT_REASON", "TYPE"]
    for c in str_cols:
        if c in df_export.columns:
            df_export[c] = df_export[c].astype(str).str.upper()

    df_export.columns = [c.upper() for c in df_export.columns] # Asegura mayúsculas en todo caso

    # Reordenar columnas para mejor lectura (opcional pero recomendado)
    start_cols = ["ENTRY_TIME", "EXIT_TIME", "SIDE", "EXIT_REASON", "ENTRY_PRICE", "EXIT_PRICE", "CANTIDAD", "VOLUMEN", "SALDO", "APALANCAMIENTO", "GROSS_PNL", "FEES", "NET_PNL", "ROI_%", "BALANCE_PRE", "BALANCE_POST"]
    ordered_cols = [c for c in start_cols if c in df_export.columns]
    remaining = [c for c in df_export.columns if c not in ordered_cols]
    df_export = df_export[ordered_cols + remaining]


    # Usar openpyxl para permitir edición posterior
    df_export.to_excel(fpath, sheet_name='Trades', index=False, engine='openpyxl')

    # Aplicar estilos
    try:
        _aplicar_estilo_trades(fpath)
    except Exception:
        pass

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

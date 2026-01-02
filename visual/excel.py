import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ====== CONFIGURACIÓN FÁCIL ARRIBA DEL TODO ======
# Guardar SIEMPRE el 100% de trials en un único resumen por estrategia (sin paginado).
# Nota: con muchos trials el Excel puede crecer bastante, pero cumple el requisito.
MAX_TRIALS_POR_RESUMEN = None


def convertir_resumen_csv_a_excel(
    *,
    csv_path: str,
    excel_path: str,
    strategy_name: str,
) -> None:
    """
    Convierte un resumen CSV (rápido durante la optimización) a Excel con formato profesional MEJORADO.

    Recomendación de uso:
    - Durante los trials: guardar resumen en CSV (append)
    - Al final del proceso: llamar a esta función por estrategia para generar el Excel final

    MEJORAS:
    - Elimina columnas innecesarias/internas
    - Orden inteligente de columnas (métricas principales primero)
    - Formato mejorado con colores y ancho adaptativo
    - Headers más claros

    Parámetros:
    - csv_path: ruta al CSV de resumen (idealmente por estrategia)
    - excel_path: ruta de salida del Excel
    - strategy_name: nombre de la estrategia (se añade como columna y se muestra en cabecera)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV de resumen: {csv_path}")

    df = pd.read_csv(csv_path)

    # Asegurar columna explícita de estrategia
    if "ESTRATEGIA" not in df.columns:
        df.insert(
            0,
            "ESTRATEGIA",
            str(strategy_name).replace("+", "_").replace(" ", "_").upper(),
        )

    # ========================================================================
    # ELIMINAR COLUMNAS INNECESARIAS/INTERNAS
    # ========================================================================
    columnas_eliminar = []
    
    # Eliminar claves internas del sistema (__xxx)
    for col in df.columns:
        if str(col).startswith("__"):
            columnas_eliminar.append(col)
    
    # Eliminar columnas redundantes o muy técnicas
    eliminar_exactas = [
        "PERTURBADO", "SEED",  # No relevantes en producción
        "NOMBRE_COMBO",  # Ya está en ESTRATEGIA
        "EXIT_TYPE",  # Ya se sabe por el nombre de carpeta
        "ACTIVO",  # Ya se sabe por el nombre de archivo
        "COMBO",  # Redundante con columnas de params
    ]
    
    for col in eliminar_exactas:
        if col in df.columns:
            columnas_eliminar.append(col)
    
    # Eliminar columnas con todos valores NaN o vacíos
    for col in df.columns:
        if df[col].isna().all() or (df[col].astype(str).str.strip() == "").all():
            columnas_eliminar.append(col)
    
    df = df.drop(columns=columnas_eliminar, errors='ignore')

    # ========================================================================
    # ORDENAR COLUMNAS DE FORMA INTELIGENTE
    # ========================================================================
    
    # Definir orden preferido de columnas principales
    orden_principal = [
        "ESTRATEGIA",
        "TRIAL",
        "SCORE",
        "SALDO_ACTUAL",
        "ROI_PCT",
        "SHARPE_RATIO",
        "SORTINO_RATIO",
        "SQN",
        "MAX_DD_PCT",
        "PROFIT_FACTOR",
        "WINRATE_PCT",
        "EXPECTATIVA",
        "TOTAL_TRADES",
        "TRADES_POR_DIA",
    ]
    
    # Columnas que existen y están en el orden preferido
    cols_principales = [c for c in orden_principal if c in df.columns]
    
    # Resto de columnas (params, etc) - ordenadas alfabéticamente
    cols_resto = sorted([c for c in df.columns if c not in cols_principales])
    
    # Reordenar DataFrame
    df = df[cols_principales + cols_resto]

    # ========================================================================
    # ORDENAR FILAS POR SALDO_ACTUAL (mejor primero)
    # ========================================================================
    if "SALDO_ACTUAL" in df.columns:
        df = df.sort_values(by="SALDO_ACTUAL", ascending=False).reset_index(drop=True)
    elif "SCORE" in df.columns:
        df = df.sort_values(by="SCORE", ascending=False).reset_index(drop=True)

    # ========================================================================
    # FORMATEAR NÚMEROS (redondear a 2 decimales)
    # ========================================================================
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(2)

    # ========================================================================
    # GUARDAR A EXCEL
    # ========================================================================
    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)
    df.to_excel(excel_path, index=False)

    # ========================================================================
    # FORMATO PROFESIONAL CON OPENPYXL
    # ========================================================================
    wb = load_workbook(excel_path)
    ws = wb.active

    # Headers con fondo azul oscuro y texto blanco
    fill_header = PatternFill("solid", fgColor="1F4E78")
    font_header = Font(bold=True, color="FFFFFF", size=11)
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for cell in ws[1]:
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = align_center

    # Alineación centrada para todas las celdas
    for row in ws.iter_rows(
        min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column
    ):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # ========================================================================
    # AUTO ANCHO INTELIGENTE
    # ========================================================================
    for col in range(1, ws.max_column + 1):
        header = ws.cell(row=1, column=col)
        max_length = len(str(header.value)) if header.value is not None else 10
        
        # Calcular ancho basado en contenido
        for row in range(2, min(ws.max_row + 1, 52)):  # Max 50 filas para calcular
            cell = ws.cell(row=row, column=col)
            if cell.value:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)
        
        # Limitar ancho entre 10 y 30 caracteres
        max_length = min(max(max_length + 2, 10), 30)
        ws.column_dimensions[get_column_letter(col)].width = max_length

    # Alto uniforme de filas
    ws.row_dimensions[1].height = 25  # Header más alto
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = 20

    # ========================================================================
    # COLOREAR COLUMNAS CLAVE
    # ========================================================================
    
    # Mapeo de columnas a índices
    col_indices = {}
    for i, cell in enumerate(ws[1], start=1):
        if cell.value:
            col_indices[str(cell.value).upper()] = get_column_letter(i)
    
    # Colorear SALDO_ACTUAL (verde/amarillo/rojo)
    if "SALDO_ACTUAL" in col_indices:
        saldo_col = col_indices["SALDO_ACTUAL"]
        for row in range(2, ws.max_row + 1):
            cell = ws[f"{saldo_col}{row}"]
            try:
                val = float(cell.value)
                if val > 1000:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")  # Verde claro
                    cell.font = Font(bold=True, color="006100")  # Verde oscuro
                elif val >= 400:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")  # Amarillo
                    cell.font = Font(color="9C6500")  # Marrón
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")  # Rojo claro
                    cell.font = Font(color="9C0006")  # Rojo oscuro
            except Exception:
                pass
    
    # Colorear SCORE (verde si positivo, rojo si negativo)
    if "SCORE" in col_indices:
        score_col = col_indices["SCORE"]
        for row in range(2, ws.max_row + 1):
            cell = ws[f"{score_col}{row}"]
            try:
                val = float(cell.value)
                if val >= 0:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                    cell.font = Font(bold=True, color="006100")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
                    cell.font = Font(color="9C0006")
            except Exception:
                pass
    
    # Colorear ROI_PCT (verde/amarillo/rojo)
    if "ROI_PCT" in col_indices:
        roi_col = col_indices["ROI_PCT"]
        for row in range(2, ws.max_row + 1):
            cell = ws[f"{roi_col}{row}"]
            try:
                val = float(cell.value)
                if val > 100:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val >= 0:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            except Exception:
                pass
    
    # Colorear WINRATE_PCT (verde >50%, amarillo 40-50%, rojo <40%)
    if "WINRATE_PCT" in col_indices:
        wr_col = col_indices["WINRATE_PCT"]
        for row in range(2, ws.max_row + 1):
            cell = ws[f"{wr_col}{row}"]
            try:
                val = float(cell.value)
                if val > 50:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val >= 40:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            except Exception:
                pass

    # Colorear MAX_DD_PCT (rojo más intenso cuanto peor)
    if "MAX_DD_PCT" in col_indices:
        dd_col = col_indices["MAX_DD_PCT"]
        for row in range(2, ws.max_row + 1):
            cell = ws[f"{dd_col}{row}"]
            try:
                val = float(cell.value)
                if val > -10:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val > -20:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            except Exception:
                pass

    wb.save(excel_path)
    wb.save(excel_path)


def exportar_trades_excel(
    df_trades: pd.DataFrame,
    resumen_path: str,
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
    """
    Exporta resumen de trials y Excel de trades del trial actual.
    - Resumen: un único archivo por estrategia con el 100% de trials (sin paginado).
    - Trades: se guardan solo los max_archivos mejores según score (top-K).
    """

    # ========= RESUMEN GENERAL =============

    # 1. Detecta nombre de la combinación (solo para contenido, no para el filename)
    nombre_combo = params.get("NOMBRE_COMBO", None)
    if not nombre_combo:
        nombre_combo = "DEFAULT"
    nombre_combo = str(nombre_combo).replace("+", "_").replace(" ", "_").upper()

    resumen_dir = os.path.dirname(resumen_path)
    if resumen_dir and not os.path.exists(resumen_dir):
        os.makedirs(resumen_dir)

    # Resumen único: siempre escribir exactamente en `resumen_path`
    resumen_actual = resumen_path

    if os.path.exists(resumen_actual):
        try:
            df_old = pd.read_excel(resumen_actual)
        except Exception:
            # If the existing resumen file is corrupted (e.g., EOFError), back it up and rebuild.
            try:
                corrupt_path = f"{resumen_actual}.corrupt_{int(time.time())}"
                os.replace(resumen_actual, corrupt_path)
            except Exception:
                pass
            df_old = pd.DataFrame()
    else:
        df_old = pd.DataFrame()

    # ------ Montar fila con métricas y parámetros ------
    fila = {
        "ESTRATEGIA": nombre_combo,
        "TRIAL": trial_number,
        "PERTURBADO": "✅" if perturbado else "❌",
        "SEED": perturb_seed if perturbado else "",
    }
    # Añadir métricas
    for k, v in metrics.items():
        fila[k.upper()] = v

    # Añadir todos los parámetros (plano si anidados)
    def plano(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(plano(v, f"{prefix}{k.upper()}_"))
            else:
                out[f"{prefix}{k.upper()}"] = v
        return out

    fila.update(plano(params))

    # Combo visible
    # Importante: no incluir claves internas/reservadas (p.ej. "__indicators_used")
    # ni metadatos del runner/reporting como si fueran "indicadores".
    combo_keys = [
        k.upper()
        for k, v in params.items()
        if not str(k).startswith("__")
        and k not in {"NOMBRE_COMBO"}
        and (
            (isinstance(v, dict) and v.get("activo", False))
            or (isinstance(v, bool) and v)
        )
    ]
    fila["COMBO"] = " - ".join(combo_keys) if combo_keys else "-"

    # Montar DataFrame resumen
    df_res = pd.DataFrame([fila])
    if not df_old.empty:
        df_res = pd.concat([df_old, df_res], ignore_index=True)
        df_res = df_res.reindex(columns=sorted(set(df_res.columns)), fill_value="")
    if "SALDO_ACTUAL" in df_res.columns:
        df_res = df_res.sort_values(by="SALDO_ACTUAL", ascending=False).reset_index(
            drop=True
        )
    df_res.to_excel(resumen_actual, index=False)

    # Formato visual pro
    try:
        wb = load_workbook(resumen_actual)
    except FileNotFoundError:
        # Si por cualquier motivo el archivo no está disponible inmediatamente
        # (p.ej. creación inicial/carrera de FS), no rompemos el trial.
        return
    ws = wb.active
    fill_header = PatternFill("solid", fgColor="376092")
    font_header = Font(bold=True, color="FFFFFF")
    align_center = Alignment(horizontal="center", vertical="center")
    for cell in ws[1]:
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = align_center
    for row in ws.iter_rows(
        min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column
    ):
        for cell in row:
            cell.alignment = align_center

    # Auto ancho elegante
    for col in range(1, ws.max_column + 1):
        header = ws.cell(row=1, column=col)
        max_length = len(str(header.value)) if header.value is not None else 0
        for row in range(2, ws.max_row + 1):
            cell = ws.cell(row=row, column=col)
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[get_column_letter(col)].width = max_length + 2

    # Alto uniforme
    for row in range(1, ws.max_row + 1):
        ws.row_dimensions[row].height = 18

    # Colorear SALDO ACTUAL y SCORE si existen
    saldo_col = score_col = None
    for i, cell in enumerate(ws[1], start=1):
        if cell.value and "SALDO ACTUAL" in str(cell.value).upper():
            saldo_col = get_column_letter(i)
        if cell.value and "SCORE" in str(cell.value).upper():
            score_col = get_column_letter(i)
    for row in range(2, ws.max_row + 1):
        if saldo_col:
            cell = ws[f"{saldo_col}{row}"]
            try:
                val = float(cell.value)
                if val > 1000:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val >= 400:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            except:
                pass
        if score_col:
            cell = ws[f"{score_col}{row}"]
            try:
                val = float(cell.value)
                if val >= 0:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            except:
                pass
    wb.save(resumen_actual)

    # ========= EXCEL DE TRADES INDIVIDUALES =========
    # Solo guardamos si skip_trades_file es False (optimización: evitar guardar si el score no es mejor)
    if skip_trades_file:
        return  # Saltar guardado de trades individuales, solo se guardó el resumen

    trades_dir = os.path.dirname(trades_actual_base)
    if trades_dir and not os.path.exists(trades_dir):
        os.makedirs(trades_dir)

    # Nombre requerido: TRIAL-<n>_SCORE-<s>.xlsx
    score_str = f"{score:.2f}" if score is not None else "unknown"
    trades_dir = os.path.dirname(trades_actual_base) or os.path.dirname(resumen_actual) or "."
    trades_actual_path = os.path.join(trades_dir, f"TRIAL-{trial_number}_SCORE-{score_str}.xlsx")

    # Copia solo si es necesario modificar (optimización: evitar copia si no hay cambios)
    # Como vamos a modificar columnas y tipos, necesitamos copia
    df_trades = df_trades.copy()
    df_trades.columns = [col.upper() for col in df_trades.columns]
    for col in ["TYPE", "TIPO_SALIDA"]:
        if col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(str).str.upper()
    for col in df_trades.columns:
        vals = df_trades[col].dropna()
        if not vals.empty and np.issubdtype(vals.values[0].__class__, np.datetime64):
            try:
                df_trades[col] = pd.to_datetime(df_trades[col]).dt.tz_localize(None)
            except Exception:
                pass
    float_cols = []
    for col in df_trades.columns:
        if any(
            x in col
            for x in [
                "QTY",
                "PRICE",
                "PNL",
                "COMISION",
                "STAKE",
                "NOTIONAL",
                "SALDO",
                "APALANC",
                "LEVER",
                "VOLUM",
            ]
        ) or col.endswith("_PCT"):
            try:
                df_trades[col] = pd.to_numeric(df_trades[col], errors="coerce").round(2)
                float_cols.append(col)
            except:
                pass
    df_trades.to_excel(trades_actual_path, index=False)
    wb = load_workbook(trades_actual_path)
    ws = wb.active
    ws.insert_rows(1)
    ncols = ws.max_column
    combo_str = fila.get("COMBO", "-")
    ws.cell(row=1, column=1).value = f"INDICADORES: {combo_str}"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(row=1, column=1).font = Font(bold=True)
    ws.cell(row=1, column=1).alignment = Alignment(
        horizontal="center", vertical="center"
    )
    fill_header = PatternFill("solid", fgColor="376092")
    font_header = Font(bold=True, color="FFFFFF")
    align_center = Alignment(horizontal="center", vertical="center")
    for cell in ws[2]:
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = align_center
    for row in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=1, max_col=ncols):
        for col_idx, cell in enumerate(row, start=1):
            cell.alignment = align_center
            col_name = ws.cell(row=2, column=col_idx).value
            if col_name in float_cols:
                cell.number_format = "0.00"
    for col in range(1, ws.max_column + 1):
        header = ws.cell(row=2, column=col)
        max_length = len(str(header.value)) if header.value is not None else 0
        for row in range(3, ws.max_row + 1):
            cell = ws.cell(row=row, column=col)
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        col_letter = get_column_letter(col)
        if str(header.value).strip().upper() == "TYPE":
            ws.column_dimensions[col_letter].width = 20
        else:
            ws.column_dimensions[col_letter].width = max_length + 2
    for row in range(1, ws.max_row + 1):
        ws.row_dimensions[row].height = 18
    pnl_neto_col = None
    for i, cell in enumerate(ws[2], start=1):
        if cell.value and "PNL_NETO" in str(cell.value).upper():
            pnl_neto_col = get_column_letter(i)
            break
    if pnl_neto_col:
        for row in range(3, ws.max_row + 1):
            cell = ws[f"{pnl_neto_col}{row}"]
            try:
                val = float(cell.value)
                if val > 0:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val < 0:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
            except:
                pass
    wb.save(trades_actual_path)

    # Mantener siempre exactamente max_archivos archivos: ordenar por score y eliminar los peores
    if score is not None:
        existing = [
            f
            for f in os.listdir(trades_dir)
            if f.endswith(".xlsx") and f.startswith("TRIAL-")
        ]

        # Extraer scores del nombre del archivo (incluyendo el que acabamos de guardar)
        files_with_scores = []
        for f in existing:
            # Buscar score en el nombre del archivo: formato "TRIAL-{n}_SCORE-{score}.xlsx"
            score_match = re.search(r"TRIAL-\d+_SCORE-([\d.]+)\.xlsx", f)
            if score_match:
                try:
                    score_from_file = float(score_match.group(1))
                    files_with_scores.append((score_from_file, f))
                except (ValueError, TypeError):
                    # Si no se puede extraer el score, usar un valor muy bajo para que se elimine
                    files_with_scores.append((float("-inf"), f))
            else:
                # Archivos antiguos sin score en el nombre: usar valor muy bajo
                files_with_scores.append((float("-inf"), f))

        # Ordenar por score (mejor primero = mayor score)
        files_with_scores.sort(key=lambda x: x[0], reverse=True)

        # Mantener solo los max_archivos mejores y eliminar el resto
        # Así siempre tendremos exactamente max_archivos archivos (o menos si aún no hay suficientes)
        if len(files_with_scores) > max_archivos:
            for _, fname in files_with_scores[max_archivos:]:
                file_path = os.path.join(trades_dir, fname)
                if os.path.exists(file_path):
                    os.remove(file_path)


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
    """
    Versión ULTRA-RÁPIDA de exportar_trades_excel usando CSV append.
    
    Diferencias con exportar_trades_excel:
    - RESUMEN: Escribe a CSV en modo append (100x más rápido que Excel)
    - TRADES: Mantiene formato Excel para archivos individuales (top-K)
    - Al final de la estrategia: convertir CSV→Excel con formato profesional
    
    Ventajas:
    - 100x más rápido durante trials (CSV append vs Excel rewrite)
    - Memoria constante (no carga todo el resumen en memoria)
    - Formato final idéntico (conversión al terminar)
    
    Uso:
    1. Durante trials: llamar a esta función (guarda CSV)
    2. Al final: llamar a convertir_resumen_csv_a_excel()
    """
    
    # ========= RESUMEN EN CSV (ULTRA-RÁPIDO) =============
    
    # 1. Detecta nombre de la combinación
    nombre_combo = params.get("NOMBRE_COMBO", None)
    if not nombre_combo:
        nombre_combo = "DEFAULT"
    nombre_combo = str(nombre_combo).replace("+", "_").replace(" ", "_").upper()
    
    resumen_dir = os.path.dirname(resumen_csv_path)
    if resumen_dir and not os.path.exists(resumen_dir):
        os.makedirs(resumen_dir)
    
    # ------ Montar fila con métricas y parámetros ------
    fila = {
        "ESTRATEGIA": nombre_combo,
        "TRIAL": trial_number,
        "PERTURBADO": "✅" if perturbado else "❌",
        "SEED": perturb_seed if perturbado else "",
    }
    
    # Añadir métricas
    for k, v in metrics.items():
        fila[k.upper()] = v
    
    # Añadir todos los parámetros (plano si anidados)
    def plano(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(plano(v, f"{prefix}{k.upper()}_"))
            else:
                out[f"{prefix}{k.upper()}"] = v
        return out
    
    fila.update(plano(params))
    
    # Combo visible
    combo_keys = [
        k.upper()
        for k, v in params.items()
        if not str(k).startswith("__")
        and k not in {"NOMBRE_COMBO"}
        and (
            (isinstance(v, dict) and v.get("activo", False))
            or (isinstance(v, bool) and v)
        )
    ]
    fila["COMBO"] = " - ".join(combo_keys) if combo_keys else "-"
    
    # APPEND A CSV (100x más rápido que Excel)
    df_fila = pd.DataFrame([fila])
    
    # IMPORTANTE: si el CSV ya existía de una ejecución anterior, puede tener
    # columnas distintas (schema drift) y romper el parseo CSV→Excel.
    # Como Optuna reinicia trial_number en 0 para un run nuevo, reseteamos aquí.
    if int(trial_number) == 0 and os.path.exists(resumen_csv_path):
        try:
            os.remove(resumen_csv_path)
        except Exception:
            pass

    # Si no existe, crear CSV con headers; si existe, append sin headers.
    if not os.path.exists(resumen_csv_path):
        df_fila.to_csv(resumen_csv_path, index=False, mode="w")
    else:
        df_fila.to_csv(resumen_csv_path, index=False, mode="a", header=False)
    
    # ========= EXCEL DE TRADES INDIVIDUALES (IGUAL QUE ANTES) =========
    # Solo guardamos si skip_trades_file es False
    if skip_trades_file:
        return
    
    trades_dir = os.path.dirname(trades_actual_base)
    if trades_dir and not os.path.exists(trades_dir):
        os.makedirs(trades_dir)
    
    # Nombre: TRIAL-<n>_SCORE-<s>.xlsx
    score_str = f"{score:.2f}" if score is not None else "unknown"
    trades_dir = os.path.dirname(trades_actual_base) or trades_dir or "."
    trades_actual_path = os.path.join(trades_dir, f"TRIAL-{trial_number}_SCORE-{score_str}.xlsx")
    
    # Preparar DataFrame de trades
    df_trades = df_trades.copy()
    df_trades.columns = [col.upper() for col in df_trades.columns]
    for col in ["TYPE", "TIPO_SALIDA"]:
        if col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(str).str.upper()
    
    for col in df_trades.columns:
        vals = df_trades[col].dropna()
        if not vals.empty and np.issubdtype(vals.values[0].__class__, np.datetime64):
            try:
                df_trades[col] = pd.to_datetime(df_trades[col]).dt.tz_localize(None)
            except Exception:
                pass
    
    float_cols = []
    for col in df_trades.columns:
        if any(
            x in col
            for x in ["QTY", "PRICE", "PNL", "COMISION", "STAKE", "NOTIONAL", "SALDO"]
        ) or col.endswith("_PCT"):
            try:
                df_trades[col] = pd.to_numeric(df_trades[col], errors="coerce").round(2)
                float_cols.append(col)
            except:
                pass
    
    # Guardar Excel de trades
    df_trades.to_excel(trades_actual_path, index=False)
    
    # Formato profesional (igual que antes)
    wb = load_workbook(trades_actual_path)
    ws = wb.active
    ws.insert_rows(1)
    ncols = ws.max_column
    combo_str = fila.get("COMBO", "-")
    ws.cell(row=1, column=1).value = f"INDICADORES: {combo_str}"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(row=1, column=1).font = Font(bold=True)
    ws.cell(row=1, column=1).alignment = Alignment(horizontal="center", vertical="center")
    
    fill_header = PatternFill("solid", fgColor="376092")
    font_header = Font(bold=True, color="FFFFFF")
    align_center = Alignment(horizontal="center", vertical="center")
    
    for cell in ws[2]:
        cell.fill = fill_header
        cell.font = font_header
        cell.alignment = align_center
    
    for row in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=1, max_col=ncols):
        for col_idx, cell in enumerate(row, start=1):
            cell.alignment = align_center
            col_name = ws.cell(row=2, column=col_idx).value
            if col_name in float_cols:
                cell.number_format = "0.00"
    
    for col in range(1, ws.max_column + 1):
        header = ws.cell(row=2, column=col)
        max_length = len(str(header.value)) if header.value is not None else 0
        for row in range(3, ws.max_row + 1):
            cell = ws.cell(row=row, column=col)
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        col_letter = get_column_letter(col)
        if str(header.value).strip().upper() == "TYPE":
            ws.column_dimensions[col_letter].width = 20
        else:
            ws.column_dimensions[col_letter].width = max_length + 2
    
    for row in range(1, ws.max_row + 1):
        ws.row_dimensions[row].height = 18
    
    # Colorear PNL_NETO
    pnl_neto_col = None
    for i, cell in enumerate(ws[2], start=1):
        if cell.value and "PNL_NETO" in str(cell.value).upper():
            pnl_neto_col = get_column_letter(i)
            break
    
    if pnl_neto_col:
        for row in range(3, ws.max_row + 1):
            cell = ws[f"{pnl_neto_col}{row}"]
            try:
                val = float(cell.value)
                if val > 0:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val < 0:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
            except:
                pass
    
    wb.save(trades_actual_path)
    
    # Mantener solo los max_archivos mejores
    if score is not None:
        existing = [
            f
            for f in os.listdir(trades_dir)
            if f.endswith(".xlsx") and f.startswith("TRIAL-")
        ]
        
        files_with_scores = []
        for f in existing:
            score_match = re.search(r"TRIAL-\d+_SCORE-([\d.]+)\.xlsx", f)
            if score_match:
                try:
                    score_from_file = float(score_match.group(1))
                    files_with_scores.append((score_from_file, f))
                except (ValueError, TypeError):
                    files_with_scores.append((float("-inf"), f))
            else:
                files_with_scores.append((float("-inf"), f))
        
        files_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        if len(files_with_scores) > max_archivos:
            for _, fname in files_with_scores[max_archivos:]:
                file_path = os.path.join(trades_dir, fname)
                if os.path.exists(file_path):
                    os.remove(file_path)

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               DETECTOR DE REGÍMENES DE MERCADO (Polars Engine)             ║
║                                                                            ║
║  Detecta regímenes Alcista / Bajista / Lateral sobre series de precios     ║
║  usando regresión lineal rodante con validación estadística.               ║
║                                                                            ║
║  Autor: Senior Quant Developer                                             ║
║  Engine: Polars (vectorizado, zero-copy, multithreaded)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES & CONFIGURACIÓN POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROLLING_WINDOW: int = 20          # N días para regresión rodante
DEFAULT_R2_THRESHOLD: float = 0.40        # Umbral mínimo de R² (exigente: 40%)
DEFAULT_P_VALUE_THRESHOLD: float = 0.05   # Umbral máximo de p-valor
DEFAULT_NOISE_FILTER_WINDOW: int = 3      # Ventana de mediana para filtro anti-parpadeo
DEFAULT_MIN_PERSISTENCE: int = 3          # Días mínimos de persistencia del régimen

REGIME_MAP: dict[int, str] = {
    1: "Alcista",
    -1: "Bajista",
    0: "Lateral",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos(ruta: str) -> pl.DataFrame:
    """Carga un archivo CSV o Feather y devuelve un DataFrame de Polars.

    Parameters
    ----------
    ruta : str
        Ruta al archivo de datos (.csv, .feather, .arrow, .ipc).

    Returns
    -------
    pl.DataFrame
        DataFrame con columnas OHLCV y un DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        Si la ruta no existe.
    ValueError
        Si la extensión no es soportada o faltan columnas requeridas.
    """
    path = Path(ruta.strip().strip("'\""))

    if not path.exists():
        raise FileNotFoundError(f"❌ Archivo no encontrado: {path}")

    ext = path.suffix.lower()

    print(f"📂 Cargando archivo: {path.name} ({ext})")
    t0 = time.perf_counter()

    if ext == ".csv":
        df = pl.read_csv(
            path,
            try_parse_dates=True,
            infer_schema_length=10_000,
        )
    elif ext in (".feather", ".arrow", ".ipc"):
        df = pl.read_ipc(path)
    elif ext == ".parquet":
        df = pl.read_parquet(path)
    else:
        raise ValueError(
            f"❌ Extensión no soportada: '{ext}'. "
            "Usa .csv, .feather, .arrow, .ipc o .parquet"
        )

    elapsed = time.perf_counter() - t0
    print(f"   ✅ Cargado en {elapsed:.3f}s — {df.shape[0]:,} filas × {df.shape[1]} cols")

    # ── Validar columnas requeridas ──
    cols_upper = {c.upper(): c for c in df.columns}
    required = ["CLOSE"]
    for req in required:
        if req not in cols_upper:
            raise ValueError(
                f"❌ Columna '{req}' no encontrada. "
                f"Columnas disponibles: {df.columns}"
            )

    # ── Normalizar nombres de columnas ──
    rename_map: dict[str, str] = {}
    for standard in ["Open", "High", "Low", "Close", "Volume"]:
        if standard.upper() in cols_upper:
            original = cols_upper[standard.upper()]
            if original != standard:
                rename_map[original] = standard

    if rename_map:
        df = df.rename(rename_map)

    # ── Detectar columna datetime ──
    datetime_col = _detectar_columna_datetime(df)
    if datetime_col and datetime_col != "datetime":
        df = df.rename({datetime_col: "datetime"})
    elif datetime_col is None:
        raise ValueError(
            "❌ No se encontró columna de fecha/hora. "
            "Asegúrate de tener una columna datetime."
        )

    # Asegurar tipo datetime
    if df["datetime"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("datetime").str.to_datetime())
    elif df["datetime"].dtype == pl.Date:
        df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

    return df


def _detectar_columna_datetime(df: pl.DataFrame) -> Optional[str]:
    """Detecta automáticamente la columna datetime del DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame donde buscar.

    Returns
    -------
    Optional[str]
        Nombre de la columna datetime, o None si no se encuentra.
    """
    # Prioridad 1: columnas con tipo datetime/date
    for col_name in df.columns:
        if df[col_name].dtype in (pl.Datetime, pl.Date):
            return col_name

    # Prioridad 2: nombres comunes
    common_names = [
        "datetime", "date", "timestamp", "time", "fecha",
        "Datetime", "Date", "Timestamp", "Time", "Fecha",
        "DATETIME", "DATE", "TIMESTAMP", "TIME", "FECHA",
    ]
    for name in common_names:
        if name in df.columns:
            return name

    # Prioridad 3: primera columna string que parezca fecha
    for col_name in df.columns:
        if df[col_name].dtype == pl.Utf8:
            sample = df[col_name].head(5).to_list()
            try:
                from dateutil.parser import parse as _date_parse
                _date_parse(str(sample[0]))
                return col_name
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESAMPLING A DIARIO
# ─────────────────────────────────────────────────────────────────────────────

def resamplear_a_diario(df: pl.DataFrame) -> pl.DataFrame:
    """Resamplea datos intradiarios a resolución diaria.

    Toma el último Close de cada día. Elimina fines de semana y días
    sin cotización (NaN).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con columnas 'datetime' y 'Close'.

    Returns
    -------
    pl.DataFrame
        DataFrame diario con columnas ['date', 'Close'].
    """
    print("📊 Resampleando a resolución diaria...")
    t0 = time.perf_counter()

    daily = (
        df
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("Close").last())
        .sort("date")
        .filter(
            # Eliminar fines de semana: lunes=1 ... domingo=7
            pl.col("date").dt.weekday().is_in([1, 2, 3, 4, 5])
        )
        .drop_nulls(subset=["Close"])
    )

    elapsed = time.perf_counter() - t0
    print(
        f"   ✅ {daily.shape[0]:,} días hábiles "
        f"({df.shape[0]:,} → {daily.shape[0]:,}) en {elapsed:.3f}s"
    )

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESIÓN LINEAL RODANTE VECTORIZADA
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_linreg_vectorized(
    log_close: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regresión lineal rodante 100% vectorizada con NumPy.

    Calcula pendiente (β), R² y p-valor sobre ventanas deslizantes
    usando la fórmula analítica de mínimos cuadrados.

    Parameters
    ----------
    log_close : np.ndarray
        Array 1D con el logaritmo natural del precio de cierre.
    window : int
        Tamaño de la ventana rodante.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (slopes, r_squared, p_values) — arrays del mismo tamaño que log_close,
        con NaN en las primeras (window-1) posiciones.

    Notes
    -----
    Usa la fórmula analítica para OLS:
        β = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        R² = 1 - SS_res / SS_tot
        t = β / SE(β)  →  p-valor via distribución t con (n-2) gl

    Complejidad: O(n) mediante sumas acumuladas (prefix sums).
    """
    n = len(log_close)
    slopes = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)
    p_values = np.full(n, np.nan)

    if n < window:
        return slopes, r_squared, p_values

    # ── Variables independientes (0, 1, ..., window-1) ──
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    N = float(window)
    denom = N * sum_x2 - sum_x * sum_x  # Denominador constante

    # ── Iterar con stride_tricks para máxima localidad de cache ──
    # Creamos una vista de ventanas deslizantes (zero-copy)
    shape = (n - window + 1, window)
    strides = (log_close.strides[0], log_close.strides[0])
    y_windows = np.lib.stride_tricks.as_strided(log_close, shape=shape, strides=strides)

    # ── Sumas vectorizadas sobre todas las ventanas simultáneamente ──
    sum_y = y_windows.sum(axis=1)         # (M,)
    sum_xy = (x[None, :] * y_windows).sum(axis=1)  # (M,)

    # ── Pendiente β ──
    beta = (N * sum_xy - sum_x * sum_y) / denom  # (M,)

    # ── Intercepto α ──
    alpha = (sum_y - beta * sum_x) / N  # (M,)

    # ── Residuos & R² ──
    y_pred = alpha[:, None] + beta[:, None] * x[None, :]  # (M, window)
    residuals = y_windows - y_pred                          # (M, window)
    ss_res = (residuals ** 2).sum(axis=1)                   # (M,)
    y_mean = sum_y / N                                       # (M,)
    ss_tot = ((y_windows - y_mean[:, None]) ** 2).sum(axis=1)  # (M,)

    # Evitar división por cero
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 1e-15, 1.0 - ss_res / ss_tot, 0.0)
        r2 = np.clip(r2, 0.0, 1.0)

    # ── Error estándar de β y p-valor ──
    df_resid = N - 2.0  # grados de libertad
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = ss_res / df_resid
        se_beta = np.sqrt(mse * N / denom)
        t_stat = np.where(se_beta > 1e-15, beta / se_beta, 0.0)

    # p-valor bilateral usando distribución t
    pval = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=df_resid)

    # ── Colocar resultados en posición correcta ──
    offset = window - 1
    slopes[offset:] = beta
    r_squared[offset:] = r2
    p_values[offset:] = pval

    return slopes, r_squared, p_values


def calcular_metricas_rodantes(
    df: pl.DataFrame,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pl.DataFrame:
    """Calcula métricas de regresión lineal rodante sobre log(Close).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame diario con columnas ['date', 'Close'].
    window : int
        Ventana rodante en días.

    Returns
    -------
    pl.DataFrame
        DataFrame enriquecido con columnas:
        ['log_close', 'slope', 'r_squared', 'p_value'].
    """
    print(f"🔬 Calculando regresión rodante (ventana={window} días)...")
    t0 = time.perf_counter()

    # ── Log del cierre ──
    close_arr = df["Close"].to_numpy().astype(np.float64)
    log_close = np.log(close_arr)

    # ── Regresión vectorizada ──
    slopes, r2, pvals = _rolling_linreg_vectorized(log_close, window)

    # ── Inyectar columnas al DataFrame de Polars ──
    df = df.with_columns(
        pl.Series("log_close", log_close),
        pl.Series("slope", slopes),
        pl.Series("r_squared", r2),
        pl.Series("p_value", pvals),
    )

    elapsed = time.perf_counter() - t0
    print(f"   ✅ Métricas calculadas en {elapsed:.3f}s")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASIFICACIÓN DE REGÍMENES
# ─────────────────────────────────────────────────────────────────────────────

def clasificar_regimenes(
    df: pl.DataFrame,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> pl.DataFrame:
    """Clasifica cada día en un régimen de mercado.

    Lógica:
    - Lateral (0):  p_valor > p_threshold  OR  R² < r2_threshold
    - Alcista (1):  significativo AND R² >= threshold AND β > 0
    - Bajista (-1): significativo AND R² >= threshold AND β < 0

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con columnas ['slope', 'r_squared', 'p_value'].
    r2_threshold : float
        Umbral mínimo de R² para considerar tendencia significativa.
    p_value_threshold : float
        Umbral máximo de p-valor para significancia estadística.

    Returns
    -------
    pl.DataFrame
        DataFrame con nueva columna 'regime_raw' (int: -1, 0, 1).
    """
    print(
        f"🏷️  Clasificando regímenes "
        f"(R²≥{r2_threshold}, p≤{p_value_threshold})..."
    )

    df = df.with_columns(
        pl.when(
            (pl.col("p_value") <= p_value_threshold)
            & (pl.col("r_squared") >= r2_threshold)
            & (pl.col("slope") > 0)
        )
        .then(pl.lit(1))
        .when(
            (pl.col("p_value") <= p_value_threshold)
            & (pl.col("r_squared") >= r2_threshold)
            & (pl.col("slope") < 0)
        )
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("regime_raw")
    )

    # ── Estadísticas rápidas ──
    counts = df.filter(pl.col("regime_raw").is_not_null()).group_by("regime_raw").len()
    for row in counts.iter_rows():
        regime_val, count = row
        label = REGIME_MAP.get(regime_val, "?")
        print(f"   • {label}: {count:,} días")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. FILTRO ANTI-RUIDO (ANTI-FLICKER)
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_filtro_ruido(
    df: pl.DataFrame,
    median_window: int = DEFAULT_NOISE_FILTER_WINDOW,
    min_persistence: int = DEFAULT_MIN_PERSISTENCE,
) -> pl.DataFrame:
    """Filtra el parpadeo de señales usando mediana rodante + persistencia.

    Paso 1: Mediana rodante sobre la señal cruda para suavizar transiciones
             rápidas (ej. 1, -1, 1 → 1, 0, 1 → suavizado).
    Paso 2: Requiere que un régimen persista al menos `min_persistence` días
             consecutivos; si no, se reclasifica como Lateral (0).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con columna 'regime_raw'.
    median_window : int
        Ventana para la mediana rodante (impar recomendado).
    min_persistence : int
        Días mínimos consecutivos para validar un régimen.

    Returns
    -------
    pl.DataFrame
        DataFrame con nueva columna 'regime' (int filtrado).
    """
    print(
        f"🔇 Aplicando filtro anti-ruido "
        f"(mediana={median_window}, persistencia≥{min_persistence})..."
    )
    t0 = time.perf_counter()

    # ── Paso 1: Mediana rodante (implementada con NumPy) ──
    raw = df["regime_raw"].to_numpy().astype(np.float64)
    n = len(raw)

    # Mediana rodante manual vectorizada
    half = median_window // 2
    smoothed = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = raw[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            smoothed[i] = np.median(valid)

    # Redondear al entero más cercano {-1, 0, 1}
    smoothed = np.round(smoothed).astype(np.float64)

    # ── Paso 2: Filtro de persistencia ──
    if min_persistence > 1:
        filtered = smoothed.copy()
        i = 0
        while i < n:
            if np.isnan(filtered[i]):
                i += 1
                continue
            current = filtered[i]
            j = i + 1
            while j < n and not np.isnan(filtered[j]) and filtered[j] == current:
                j += 1
            run_length = j - i
            if run_length < min_persistence and current != 0:
                filtered[i:j] = 0  # Reclasificar como Lateral
            i = j
        smoothed = filtered

    df = df.with_columns(
        pl.Series("regime", smoothed.astype(np.int32))
    )

    elapsed = time.perf_counter() - t0
    print(f"   ✅ Filtro aplicado en {elapsed:.3f}s")

    # ── Estadísticas post-filtro ──
    counts = df.filter(pl.col("regime").is_not_null()).group_by("regime").len()
    print("   📊 Distribución post-filtro:")
    for row in counts.sort("regime").iter_rows():
        regime_val, count = row
        label = REGIME_MAP.get(regime_val, "?")
        print(f"      • {label}: {count:,} días")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGRUPACIÓN DE REGÍMENES CONSECUTIVOS
# ─────────────────────────────────────────────────────────────────────────────

def agrupar_regimenes(df: pl.DataFrame) -> pl.DataFrame:
    """Agrupa días consecutivos con el mismo régimen y reconcilia contradicciones.

    Pipeline de agrupación:
    1. Agrupa días consecutivos con el mismo régimen.
    2. **Reconciliación**: Si un período "Alcista" tiene variación negativa,
       o un "Bajista" tiene variación positiva, se reclasifica como "Lateral"
       (la señal estadística contradice el movimiento real del precio).
    3. **Re-agrupación**: Fusiona períodos "Lateral" consecutivos que hayan
       quedado adyacentes tras la reconciliación.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con columnas ['date', 'Close', 'regime'].

    Returns
    -------
    pl.DataFrame
        DataFrame resumen de regímenes agrupados y reconciliados.
    """
    print("📋 Agrupando regímenes consecutivos...")
    t0 = time.perf_counter()

    # ── Filtrar filas con régimen válido ──
    df_valid = df.filter(pl.col("regime").is_not_null()).sort("date")

    if df_valid.is_empty():
        print("   ⚠️  No hay datos válidos para agrupar.")
        return pl.DataFrame(schema={
            "Fecha_Inicio": pl.Date,
            "Fecha_Fin": pl.Date,
            "Regimen": pl.Utf8,
            "Duracion_Dias": pl.Int64,
            "Variacion_Precio_%": pl.Float64,
        })

    # ── Detectar cambios de régimen para crear grupos ──
    df_valid = df_valid.with_columns(
        (pl.col("regime") != pl.col("regime").shift(1))
        .fill_null(True)
        .cum_sum()
        .alias("group_id")
    )

    # ── Paso 1: Agrupación inicial ──
    summary = (
        df_valid
        .group_by("group_id")
        .agg(
            pl.col("date").first().alias("Fecha_Inicio"),
            pl.col("date").last().alias("Fecha_Fin"),
            pl.col("regime").first().alias("regime_code"),
            pl.col("date").len().alias("Duracion_Dias"),
            pl.col("Close").first().alias("close_inicio"),
            pl.col("Close").last().alias("close_fin"),
        )
        .sort("group_id")
        .with_columns(
            (
                (pl.col("close_fin") - pl.col("close_inicio"))
                / pl.col("close_inicio")
                * 100.0
            ).round(4).alias("Variacion_Precio_%"),
        )
    )

    n_before = summary.shape[0]

    # ── Paso 2: Reconciliación — corregir contradicciones ──
    # Alcista con variación negativa → Lateral
    # Bajista con variación positiva → Lateral
    summary = summary.with_columns(
        pl.when(
            (pl.col("regime_code") == 1) & (pl.col("Variacion_Precio_%") < 0)
        )
        .then(pl.lit(0))
        .when(
            (pl.col("regime_code") == -1) & (pl.col("Variacion_Precio_%") > 0)
        )
        .then(pl.lit(0))
        .otherwise(pl.col("regime_code"))
        .alias("regime_code")
    )

    # Contar correcciones
    n_corrected = (
        summary.filter(
            (pl.col("regime_code") == 0)
        ).shape[0]
        - summary.filter(pl.col("regime_code") == 0).shape[0]
    )

    # ── Paso 3: Re-agrupar Laterales consecutivos tras correcciones ──
    # Convertir a pandas para facilitar la fusión secuencial
    pdf = summary.to_pandas()
    merged_rows = []
    i = 0

    while i < len(pdf):
        row = pdf.iloc[i].copy()
        regime = row["regime_code"]

        # Si es Lateral, intentar fusionar con siguientes Laterales
        if regime == 0:
            j = i + 1
            while j < len(pdf) and pdf.iloc[j]["regime_code"] == 0:
                j += 1
            # Fusionar i..j-1
            if j > i + 1:
                last_row = pdf.iloc[j - 1]
                row["Fecha_Fin"] = last_row["Fecha_Fin"]
                row["close_fin"] = last_row["close_fin"]
                total_dias = int(pdf.iloc[i:j]["Duracion_Dias"].sum())
                row["Duracion_Dias"] = total_dias
                if row["close_inicio"] != 0:
                    row["Variacion_Precio_%"] = round(
                        (row["close_fin"] - row["close_inicio"])
                        / row["close_inicio"] * 100.0,
                        4,
                    )
            merged_rows.append(row)
            i = j
        else:
            merged_rows.append(row)
            i += 1

    import pandas as pd
    merged_pdf = pd.DataFrame(merged_rows)

    # Traducir código a texto
    regime_map_local = {1: "Alcista", -1: "Bajista", 0: "Lateral"}
    merged_pdf["Regimen"] = merged_pdf["regime_code"].map(regime_map_local).fillna("Desconocido")

    # Seleccionar columnas finales
    result = pl.from_pandas(
        merged_pdf[["Fecha_Inicio", "Fecha_Fin", "Regimen", "Duracion_Dias", "Variacion_Precio_%"]]
    )

    n_after = result.shape[0]
    n_reclassified = n_before - n_after  # períodos fusionados

    elapsed = time.perf_counter() - t0
    print(
        f"   ✅ {n_before} períodos → reconciliados → {n_after} períodos "
        f"en {elapsed:.3f}s"
    )

    # ── Resumen rápido ──
    regime_stats = (
        result.group_by("Regimen")
        .agg(
            pl.col("Duracion_Dias").sum().alias("Total_Dias"),
            pl.col("Duracion_Dias").mean().alias("Duracion_Media"),
            pl.len().alias("Num_Periodos"),
        )
        .sort("Total_Dias", descending=True)
    )
    print("   📊 Resumen de regímenes (reconciliados):")
    for row in regime_stats.iter_rows():
        regimen, total, media, periodos = row
        print(
            f"      • {regimen}: {periodos} períodos, "
            f"{total} días totales, media {media:.1f} días"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORTACIÓN A EXCEL — ESTILO ACADÉMICO MINIMALISTA
# ─────────────────────────────────────────────────────────────────────────────

# Raíz del proyecto (para exportar el Excel ahí)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def exportar_a_excel(
    summary: pl.DataFrame,
    detail: pl.DataFrame,
    output_path: str,
    params: Optional[dict] = None,
) -> Path:
    """Exporta el resumen y detalle a Excel con estilo académico minimalista.

    Genera un archivo con tres hojas:
    - 'Resumen': tabla resumen con card de estadísticas y chart.
    - 'Detalle': datos día a día con métricas y régimen.
    - '_ChartData': hoja oculta auxiliar para el gráfico.

    Parameters
    ----------
    summary : pl.DataFrame
        DataFrame resumen de regímenes agrupados.
    detail : pl.DataFrame
        DataFrame diario con métricas y clasificación.
    output_path : str
        Ruta de salida para el archivo .xlsx.
    params : Optional[dict]
        Parámetros del pipeline para incluir en el reporte.

    Returns
    -------
    Path
        Ruta absoluta del archivo exportado.
    """
    import xlsxwriter
    import math

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Exportando a Excel: {out.name}")
    t0 = time.perf_counter()

    workbook = xlsxwriter.Workbook(str(out), {"nan_inf_to_errors": True})

    # ══════════════════════════════════════════════════════════════════════
    # PALETA MINIMALISTA (slate/charcoal — estilo paper académico)
    # ══════════════════════════════════════════════════════════════════════
    C_DARK     = "#2D3436"   # Texto principal oscuro
    C_MID      = "#636E72"   # Texto secundario
    C_LIGHT    = "#718096"   # Texto terciario / labels
    C_SUBTLE   = "#A0AEC0"   # Bordes y líneas sutiles
    C_BORDER   = "#E2E8F0"   # Bordes de celda
    C_ROW_ALT  = "#F7FAFC"   # Fila alternada
    C_HDR_BG   = "#F8F9FA"   # Fondo de header
    C_WHITE    = "#FFFFFF"

    # Colores de régimen — sutiles, no saturados
    C_BULL_BG  = "#F0FFF4"
    C_BULL_TX  = "#276749"
    C_BEAR_BG  = "#FFF5F5"
    C_BEAR_TX  = "#9B2C2C"
    C_SIDE_BG  = "#F7FAFC"
    C_SIDE_TX  = "#4A5568"

    FONT = "Calibri"

    # ── Formato base reutilizable ──
    _BASE = {
        "font_name": FONT, "font_size": 10, "font_color": C_DARK,
        "align": "center", "valign": "vcenter",
        "bottom": 1, "bottom_color": C_BORDER,
        "top": 0, "left": 0, "right": 0,
    }

    def _fmt(bg=C_WHITE, num_format=None, **kw):
        p = {**_BASE, "bg_color": bg}
        if num_format:
            p["num_format"] = num_format
        p.update(kw)
        return workbook.add_format(p)

    # ── Formatos específicos ──
    fmt_hdr = workbook.add_format({
        "bold": True, "font_name": FONT, "font_size": 10,
        "font_color": C_LIGHT, "bg_color": C_HDR_BG,
        "align": "center", "valign": "vcenter",
        "bottom": 2, "bottom_color": C_SUBTLE,
        "top": 0, "left": 0, "right": 0,
        "text_wrap": True,
    })
    fmt_title = workbook.add_format({
        "bold": True, "font_name": FONT, "font_size": 14,
        "font_color": C_DARK,
        "bottom": 2, "bottom_color": C_SUBTLE,
    })
    fmt_subtitle = workbook.add_format({
        "font_name": FONT, "font_size": 10,
        "font_color": C_MID, "italic": True,
    })

    # Celdas normales y alternadas
    fmt_w  = _fmt(C_WHITE)
    fmt_a  = _fmt(C_ROW_ALT)
    fmt_date_w = _fmt(C_WHITE, "yyyy-mm-dd")
    fmt_date_a = _fmt(C_ROW_ALT, "yyyy-mm-dd")
    fmt_pct_w  = _fmt(C_WHITE, "+0.00%;-0.00%")
    fmt_pct_a  = _fmt(C_ROW_ALT, "+0.00%;-0.00%")
    fmt_int_w  = _fmt(C_WHITE, "#,##0")
    fmt_int_a  = _fmt(C_ROW_ALT, "#,##0")
    fmt_num_w  = _fmt(C_WHITE, "0.000000")
    fmt_num_a  = _fmt(C_ROW_ALT, "0.000000")
    fmt_close_w = _fmt(C_WHITE, "#,##0.00")
    fmt_close_a = _fmt(C_ROW_ALT, "#,##0.00")

    # Régimen badges (muy sutiles)
    fmt_bull_w = _fmt(C_BULL_BG, font_color=C_BULL_TX, bold=True)
    fmt_bull_a = _fmt(C_BULL_BG, font_color=C_BULL_TX, bold=True)
    fmt_bear_w = _fmt(C_BEAR_BG, font_color=C_BEAR_TX, bold=True)
    fmt_bear_a = _fmt(C_BEAR_BG, font_color=C_BEAR_TX, bold=True)
    fmt_side_w = _fmt(C_SIDE_BG, font_color=C_SIDE_TX)
    fmt_side_a = _fmt(C_SIDE_BG, font_color=C_SIDE_TX)

    def _regime_fmt(text: str, alt: bool):
        if text == "Alcista":
            return fmt_bull_a if alt else fmt_bull_w
        elif text == "Bajista":
            return fmt_bear_a if alt else fmt_bear_w
        return fmt_side_a if alt else fmt_side_w

    # Card de estadísticas
    fmt_card_label = workbook.add_format({
        "font_name": FONT, "font_size": 9, "font_color": C_LIGHT,
        "align": "left", "valign": "vcenter",
        "bottom": 1, "bottom_color": C_BORDER,
        "top": 0, "left": 0, "right": 0,
        "bg_color": C_WHITE,
    })
    fmt_card_value = workbook.add_format({
        "font_name": FONT, "font_size": 11, "font_color": C_DARK,
        "bold": True, "align": "right", "valign": "vcenter",
        "bottom": 1, "bottom_color": C_BORDER,
        "top": 0, "left": 0, "right": 0,
        "bg_color": C_WHITE,
    })
    fmt_card_title = workbook.add_format({
        "font_name": FONT, "font_size": 10, "font_color": C_MID,
        "bold": True, "align": "left", "valign": "vcenter",
        "bottom": 2, "bottom_color": C_SUBTLE,
        "top": 0, "left": 0, "right": 0,
        "bg_color": C_WHITE,
    })

    # ══════════════════════════════════════════════════════════════════════
    # HOJA AUXILIAR OCULTA PARA CHART
    # ══════════════════════════════════════════════════════════════════════
    ws_aux = workbook.add_worksheet("_ChartData")
    ws_aux.hide()

    # Escribir datos del precio + régimen para el gráfico
    detail_valid = detail.filter(pl.col("regime").is_not_null()).sort("date")
    ws_aux.write_string(0, 0, "Fecha")
    ws_aux.write_string(0, 1, "Close")
    ws_aux.write_string(0, 2, "Regimen")

    dt_fmt_aux = workbook.add_format({"num_format": "yyyy-mm-dd"})
    dates_arr = detail_valid["date"].to_list()
    close_arr = detail_valid["Close"].to_list()
    regime_arr = detail_valid["regime"].to_list()

    for i in range(len(dates_arr)):
        try:
            import datetime as _dt
            d = dates_arr[i]
            if isinstance(d, _dt.date):
                ws_aux.write_datetime(i + 1, 0, _dt.datetime(d.year, d.month, d.day), dt_fmt_aux)
            else:
                ws_aux.write(i + 1, 0, str(d))
        except Exception:
            ws_aux.write(i + 1, 0, str(dates_arr[i]))
        ws_aux.write_number(i + 1, 1, float(close_arr[i]) if close_arr[i] is not None else 0)
        ws_aux.write_number(i + 1, 2, int(regime_arr[i]) if regime_arr[i] is not None else 0)

    n_chart_rows = len(dates_arr)

    # ══════════════════════════════════════════════════════════════════════
    # HOJA 1: RESUMEN
    # ══════════════════════════════════════════════════════════════════════
    ws1 = workbook.add_worksheet("Resumen")
    ws1.hide_gridlines(2)
    ws1.set_tab_color("#4A5568")
    ws1.set_landscape()
    ws1.set_margins(left=0.4, right=0.4, top=0.5, bottom=0.5)

    # ── Título ──
    ws1.set_row(0, 28)
    ws1.merge_range("A1:F1", "Market Regime Detection — Summary Report", fmt_title)
    ws1.set_row(1, 18)
    ws1.merge_range("A2:F2", "Rolling OLS on log(Close) | Statistical classification", fmt_subtitle)

    # ── Card de parámetros (lado derecho, columnas H-I) ──
    ws1.set_column(7, 7, 22)   # H
    ws1.set_column(8, 8, 14)   # I

    card_row = 0
    ws1.write(card_row, 7, "PARAMETERS", fmt_card_title)
    ws1.write(card_row, 8, "", fmt_card_title)
    card_row += 1

    if params:
        card_items = [
            ("Rolling Window (N)", f"{params.get('rolling_window', '—')} days"),
            ("R² Threshold", f"{params.get('r2_threshold', '—')}"),
            ("P-Value Threshold", f"{params.get('p_value_threshold', '—')}"),
            ("Noise Median", f"{params.get('noise_median', '—')}"),
            ("Min Persistence", f"{params.get('noise_persist', '—')} days"),
        ]
        for label, value in card_items:
            ws1.write(card_row, 7, label, fmt_card_label)
            ws1.write(card_row, 8, value, fmt_card_value)
            card_row += 1

    # ── Estadísticas agregadas ──
    card_row += 1
    ws1.write(card_row, 7, "STATISTICS", fmt_card_title)
    ws1.write(card_row, 8, "", fmt_card_title)
    card_row += 1

    summary_pandas = summary.to_pandas()
    total_days = int(summary_pandas["Duracion_Dias"].sum()) if len(summary_pandas) > 0 else 0
    n_regimes = len(summary_pandas)

    for regime_label in ["Alcista", "Bajista", "Lateral"]:
        subset = summary_pandas[summary_pandas["Regimen"] == regime_label]
        n_periods = len(subset)
        total_d = int(subset["Duracion_Dias"].sum()) if n_periods > 0 else 0
        avg_d = f"{subset['Duracion_Dias'].mean():.1f}" if n_periods > 0 else "—"
        pct = f"{total_d / total_days * 100:.1f}%" if total_days > 0 else "—"
        ws1.write(card_row, 7, f"{regime_label}", fmt_card_label)
        ws1.write(card_row, 8, f"{n_periods}p · {total_d}d ({pct})", fmt_card_value)
        card_row += 1

    ws1.write(card_row, 7, "Total Days", fmt_card_label)
    ws1.write(card_row, 8, str(total_days), fmt_card_value)
    card_row += 1
    ws1.write(card_row, 7, "Total Regimes", fmt_card_label)
    ws1.write(card_row, 8, str(n_regimes), fmt_card_value)

    # ── Gráfico de precio con colores de régimen ──
    if n_chart_rows > 0:
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name":       "Close",
            "categories": f"='_ChartData'!$A$2:$A${n_chart_rows + 1}",
            "values":     f"='_ChartData'!$B$2:$B${n_chart_rows + 1}",
            "line":       {"color": "#4A5568", "width": 1.25},
            "marker":     {"type": "none"},
        })
        chart.set_title({
            "name": "Price & Regime Overlay",
            "name_font": {"size": 10, "bold": True, "color": C_DARK, "name": FONT},
        })

        # Eje Y
        close_vals = [float(v) for v in close_arr if v is not None]
        if close_vals:
            y_min = min(close_vals)
            y_max = max(close_vals)
            data_range = y_max - y_min if y_max != y_min else max(abs(y_max) * 0.1, 1.0)

            def _nice_unit(dr, ticks=6):
                if dr <= 0:
                    return 1.0
                raw = dr / ticks
                mag = math.pow(10, math.floor(math.log10(raw)))
                n = raw / mag
                if n <= 1: nice = 1
                elif n <= 2: nice = 2
                elif n <= 5: nice = 5
                else: nice = 10
                return nice * mag

            margin = data_range * 0.03
            chart.set_y_axis({
                "num_format": "#,##0",
                "num_font":   {"size": 8, "color": C_LIGHT, "name": FONT},
                "min": y_min - margin,
                "max": y_max + margin,
                "major_unit": _nice_unit(data_range),
                "major_gridlines": {"visible": True, "line": {"color": "#EDF2F7", "width": 0.5}},
                "minor_gridlines": {"visible": False},
                "line": {"none": True},
            })

        chart.set_x_axis({
            "num_format": "MMM yy",
            "num_font":   {"size": 7, "color": C_SUBTLE, "name": FONT},
            "major_gridlines": {"visible": False},
            "major_tick_mark": "none",
            "minor_tick_mark": "none",
            "line": {"color": C_BORDER, "width": 0.5},
        })
        chart.set_legend({"none": True})
        chart.set_plotarea({"border": {"none": True}, "fill": {"color": C_WHITE}})
        chart.set_chartarea({"border": {"none": True}, "fill": {"color": C_WHITE}})
        chart.set_size({"width": 1000, "height": 380})

        ws1.insert_chart("A4", chart, {"x_offset": 2, "y_offset": 5})

    # ── Tabla de regímenes (debajo del gráfico) ──
    table_start_row = 26  # Después del gráfico (~22 rows)

    ws1.set_row(table_start_row, 4)  # Espaciador fino
    table_start_row += 1

    headers_summary = [
        "Fecha Inicio", "Fecha Fin", "Régimen",
        "Duración", "Var. Precio %",
    ]
    col_widths = [15, 15, 13, 11, 13]

    for col_idx, (header, width) in enumerate(zip(headers_summary, col_widths)):
        ws1.set_column(col_idx, col_idx, width)
        ws1.write(table_start_row, col_idx, header, fmt_hdr)

    for row_idx, (_, row) in enumerate(summary_pandas.iterrows()):
        r = table_start_row + 1 + row_idx
        alt = row_idx % 2 == 1
        regime_text = str(row["Regimen"])

        ws1.write(r, 0, row["Fecha_Inicio"], fmt_date_a if alt else fmt_date_w)
        ws1.write(r, 1, row["Fecha_Fin"], fmt_date_a if alt else fmt_date_w)
        ws1.write(r, 2, regime_text, _regime_fmt(regime_text, alt))
        ws1.write(r, 3, int(row["Duracion_Dias"]), fmt_int_a if alt else fmt_int_w)

        var_pct = row["Variacion_Precio_%"]
        if var_pct is not None and not np.isnan(var_pct):
            ws1.write(r, 4, var_pct / 100.0, fmt_pct_a if alt else fmt_pct_w)
        else:
            ws1.write(r, 4, "", fmt_a if alt else fmt_w)

    # ══════════════════════════════════════════════════════════════════════
    # HOJA 2: DETALLE DIARIO
    # ══════════════════════════════════════════════════════════════════════
    ws2 = workbook.add_worksheet("Detalle Diario")
    ws2.hide_gridlines(2)
    ws2.set_tab_color("#718096")
    ws2.freeze_panes(2, 0)

    detail_cols = ["date", "Close", "log_close", "slope", "r_squared", "p_value", "regime"]
    detail_export = detail.select([c for c in detail_cols if c in detail.columns])

    headers_detail = ["Fecha", "Close", "Log(Close)", "β (Slope)", "R²", "P-Value", "Régimen"]
    col_widths_detail = [14, 14, 13, 15, 11, 11, 12]

    ws2.set_row(0, 22)
    ws2.merge_range("A1:G1", "Daily Metrics — Rolling Linear Regression", fmt_title)
    start_row2 = 1

    for col_idx, (header, width) in enumerate(zip(headers_detail, col_widths_detail)):
        ws2.set_column(col_idx, col_idx, width)
        ws2.write(start_row2, col_idx, header, fmt_hdr)

    detail_pandas = detail_export.to_pandas()
    for row_idx, (_, row) in enumerate(detail_pandas.iterrows()):
        r = start_row2 + 1 + row_idx
        alt = row_idx % 2 == 1
        ws2.set_row(r, 16)

        ws2.write(r, 0, row.get("date"), fmt_date_a if alt else fmt_date_w)
        ws2.write(r, 1, row.get("Close", ""), fmt_close_a if alt else fmt_close_w)
        ws2.write(r, 2, row.get("log_close", ""), fmt_num_a if alt else fmt_num_w)
        ws2.write(r, 3, row.get("slope", ""), fmt_num_a if alt else fmt_num_w)
        ws2.write(r, 4, row.get("r_squared", ""), fmt_num_a if alt else fmt_num_w)
        ws2.write(r, 5, row.get("p_value", ""), fmt_num_a if alt else fmt_num_w)

        regime_val = row.get("regime")
        if regime_val is not None and not np.isnan(regime_val):
            regime_text = REGIME_MAP.get(int(regime_val), "?")
            ws2.write(r, 6, regime_text, _regime_fmt(regime_text, alt))
        else:
            ws2.write(r, 6, "", fmt_a if alt else fmt_w)

    # ── Formato condicional: colorear slope positivo/negativo ──
    n_detail = len(detail_pandas)
    if n_detail > 0:
        fmt_green_cond = workbook.add_format({
            **_BASE, "font_color": C_BULL_TX, "bold": True, "num_format": "0.000000",
        })
        fmt_red_cond = workbook.add_format({
            **_BASE, "font_color": C_BEAR_TX, "bold": True, "num_format": "0.000000",
        })
        ws2.conditional_format(
            start_row2 + 1, 3, start_row2 + n_detail, 3,
            {"type": "cell", "criteria": ">", "value": 0, "format": fmt_green_cond},
        )
        ws2.conditional_format(
            start_row2 + 1, 3, start_row2 + n_detail, 3,
            {"type": "cell", "criteria": "<", "value": 0, "format": fmt_red_cond},
        )

    workbook.close()

    elapsed = time.perf_counter() - t0
    print(f"   ✅ Excel exportado en {elapsed:.3f}s → {out}")

    return out.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_pipeline(
    ruta_archivo: str,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    noise_median_window: int = DEFAULT_NOISE_FILTER_WINDOW,
    noise_min_persistence: int = DEFAULT_MIN_PERSISTENCE,
    output_dir: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Pipeline completo de detección de regímenes de mercado.

    Ejecuta secuencialmente:
    1. Carga de datos (CSV/Feather/Parquet)
    2. Resampling a diario
    3. Cálculo de métricas rodantes (regresión lineal)
    4. Clasificación de regímenes
    5. Filtro anti-ruido
    6. Agrupación de regímenes consecutivos
    7. Exportación a Excel

    Parameters
    ----------
    ruta_archivo : str
        Ruta al archivo de datos.
    rolling_window : int
        Ventana rodante para la regresión (en días).
    r2_threshold : float
        Umbral mínimo de R² para tendencia significativa.
    p_value_threshold : float
        Umbral máximo de p-valor.
    noise_median_window : int
        Ventana de mediana para filtro anti-ruido.
    noise_min_persistence : int
        Días mínimos de persistencia del régimen.
    output_dir : Optional[str]
        Directorio de salida para el Excel. Si None, usa la raíz
        del proyecto (MODELOX/).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, Path]
        (summary, detail, excel_path)
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║    🔍  DETECTOR DE REGÍMENES DE MERCADO — Pipeline Iniciado       ║")
    print("╚" + "═" * 68 + "╝")
    print()

    total_t0 = time.perf_counter()

    # ── Mostrar parámetros ──
    print("⚙️  Parámetros:")
    print(f"   • Ventana rodante (N):     {rolling_window} días")
    print(f"   • Umbral R²:               {r2_threshold}")
    print(f"   • Umbral p-valor:           {p_value_threshold}")
    print(f"   • Mediana anti-ruido:       {noise_median_window}")
    print(f"   • Persistencia mínima:      {noise_min_persistence} días")
    print()

    # ── 1. Carga ──
    df_raw = cargar_datos(ruta_archivo)
    print()

    # ── 2. Resampling ──
    df_daily = resamplear_a_diario(df_raw)
    print()

    # ── 3. Métricas rodantes ──
    df_daily = calcular_metricas_rodantes(df_daily, window=rolling_window)
    print()

    # ── 4. Clasificación ──
    df_daily = clasificar_regimenes(
        df_daily,
        r2_threshold=r2_threshold,
        p_value_threshold=p_value_threshold,
    )
    print()

    # ── 5. Filtro anti-ruido ──
    df_daily = aplicar_filtro_ruido(
        df_daily,
        median_window=noise_median_window,
        min_persistence=noise_min_persistence,
    )
    print()

    # ── 6. Agrupación ──
    summary = agrupar_regimenes(df_daily)
    print()

    # ── 7. Exportación (siempre a la raíz del proyecto) ──
    input_path = Path(ruta_archivo.strip().strip("'\""))
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = _PROJECT_ROOT  # ← RAÍZ del proyecto

    excel_name = f"regimenes_{input_path.stem}.xlsx"

    # Parámetros para incluir en el reporte
    pipeline_params = {
        "rolling_window": rolling_window,
        "r2_threshold": r2_threshold,
        "p_value_threshold": p_value_threshold,
        "noise_median": noise_median_window,
        "noise_persist": noise_min_persistence,
    }

    excel_path = exportar_a_excel(
        summary=summary,
        detail=df_daily,
        output_path=str(out_dir / excel_name),
        params=pipeline_params,
    )

    total_elapsed = time.perf_counter() - total_t0
    print()
    print("╔" + "═" * 68 + "╗")
    print(f"║  ✅  Pipeline completado en {total_elapsed:.2f}s".ljust(69) + "  ║")
    print(f"║  📁  {excel_path}".ljust(69) + "  ║")
    print("╚" + "═" * 68 + "╝")
    print()

    return summary, df_daily, excel_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────

def _solicitar_parametro(
    prompt: str,
    default: float | int,
    tipo: type = float,
) -> float | int:
    """Solicita un parámetro al usuario con valor por defecto.

    Parameters
    ----------
    prompt : str
        Texto a mostrar al usuario.
    default : float | int
        Valor por defecto si el usuario presiona Enter.
    tipo : type
        Tipo al que convertir la entrada.

    Returns
    -------
    float | int
        Valor ingresado o el default.
    """
    raw = input(f"   {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return tipo(raw)
    except ValueError:
        print(f"   ⚠️  Valor inválido, usando default: {default}")
        return default


def main() -> None:
    """Punto de entrada CLI.

    Solicita al usuario arrastrar un archivo y elegir la ventana N.
    Los demás parámetros están fijados con valores exigentes.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║         🔍  DETECTOR DE REGÍMENES DE MERCADO                     ║")
    print("║         Powered by Polars + NumPy Vectorizado                    ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # ── Solicitar archivo ──
    print("📂 Arrastra tu archivo de datos a esta terminal")
    print("   (Formatos soportados: .csv, .feather, .arrow, .parquet)")
    print()
    ruta = input("   → Ruta del archivo: ").strip().strip("'\"")

    if not ruta:
        print("❌ No se proporcionó ninguna ruta. Saliendo.")
        sys.exit(1)

    print()
    rolling_window = int(
        _solicitar_parametro("Ventana rodante N (días)", DEFAULT_ROLLING_WINDOW, int)
    )

    # Parámetros fijos (exigentes pero realistas)
    print()
    print("⚙️  Parámetros fijos:")
    print(f"   • R² ≥ {DEFAULT_R2_THRESHOLD}   (la tendencia debe explicar ≥40% de varianza)")
    print(f"   • p-valor ≤ {DEFAULT_P_VALUE_THRESHOLD}  (significancia estadística estándar)")
    print(f"   • Mediana anti-ruido: {DEFAULT_NOISE_FILTER_WINDOW}")
    print(f"   • Persistencia mínima: {DEFAULT_MIN_PERSISTENCE} días")
    print(f"   • + Reconciliación: alcista con var.<0 → lateral, bajista con var.>0 → lateral")

    # ── Ejecutar pipeline ──
    try:
        summary, detail, excel_path = ejecutar_pipeline(
            ruta_archivo=ruta,
            rolling_window=rolling_window,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

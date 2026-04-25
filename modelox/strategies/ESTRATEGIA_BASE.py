from __future__ import annotations

"""
================================================================================
MODELOX/STRATEGIES/ESTRATEGIA_BASE.PY — BASE MAESTRA PARA ESTRATEGIAS
================================================================================

OBJETIVO:
  Define el contrato oficial entre:
    - RUNNER  (Optuna + Multi-TF)
    - STRATEGY (Generación de señales)
    - ENGINE   (Simulación de trades, Numba)
    - REPORTERS (Rich / Excel / Plots)

FILOSOFÍA:
  1. VELOCIDAD REAL: Polars End-to-End (Zero Pandas en loops).
  2. VECTORIZACIÓN: Nada de iteraciones por fila en Python.
  3. LAZY EVALUATION: Minimizar .collect() (ideal: 1 por trial).
  4. COLUMNAS MÍNIMAS: Devolver solo timestamp y señales.

CONTRATO DE INPUT (df):
  - Polars DataFrame con OHLCV.
  - Multi-Timeframe (MTF) ya aplicado (anti-lookahead garantizado).

CONTRATO DE OUTPUT (signals_df):
  - timestamp + signal_long (bool) + signal_short (bool).
  - Indicadores adicionales opcionales para plot.

================================================================================


PRIORIDAD #1: VELOCIDAD REAL
  - POLARS END-TO-END (ZERO PANDAS)
  - NADA DE LOOPS POR FILA EN PYTHON
  - MINIMIZAR .collect() (IDEAL: 1 POR TRIAL)
  - DEVOLVER SOLO LAS COLUMNAS NECESARIAS (TIMESTAMP + SEÑALES + INDICADORES)

CONTRATO DE INPUT (DF)
  - DF ES POLARS DataFrame
  - COLUMNAS MINIMAS ESPERADAS:
      - timestamp (Datetime)
      - open, high, low, close (Float)
      - volume (Opcional)
  - SI USAS MULTI-TIMEFRAME (MTF), EL BLENDER AÑADE:
      - open_1h, high_1h, low_1h, close_1h, volume_1h, etc.
  - ANTI-LOOKAHEAD MTF YA VIENE APLICADO (shift(1) + join_asof backward)

CONTRATO DE OUTPUT (signals_df)
  - DEBE CONTENER timestamp (PARA JOIN EN ENGINE)
  - DEBE CONTENER:
      - signal_long: Bool
      - signal_short: Bool
  - PUEDE CONTENER INDICADORES PARA PLOTS (RECOMENDADO)
  - RECOMENDACION DE RENDIMIENTO:
      - DEVUELVE SOLO: ["timestamp", "signal_long", "signal_short", ...INDICADORES]

METADATA EN params (IMPORTANTE PARA REPORTING / GRAFICOS)
  - params["__warmup_bars"]: INT
      - NUMERO DE BARRAS INICIALES A ENMASCARAR EN GRAFICO
  - params["__indicators_used"]: list[str]
      - SOLO ESAS COLUMNAS SE INTENTARAN PLOTEAR
  - params["__indicator_bounds"]: dict[str, dict]
      - NIVELES HI/LO/MID PARA OSCILADORES (RSI, ADX, ETC)
  - params["__indicator_specs"]: dict[str, dict]
      - OPCIONAL: PANEL, COLOR, TIPO (HISTOGRAM, OVERLAY)

════════════════════════════════════════════════════════════════════════════════
                    SISTEMA DE SALIDAS - ARQUITECTURA CENTRALIZADA
════════════════════════════════════════════════════════════════════════════════

FUENTE ÚNICA DE VERDAD: modelox/core/exits.py

────────────────────────────────────────────────────────────────────────────────
OPCIÓN 1 — SALIDAS_PERSONALIZADAS = False  (DEFAULT)
────────────────────────────────────────────────────────────────────────────────
  El engine gestiona todo: SL / TP / Trailing según exits.py + configuracion.py.
  Optuna puede optimizar los parámetros de salida automáticamente.
  No necesitas añadir nada extra a generate_signals().

────────────────────────────────────────────────────────────────────────────────
OPCIÓN 2 — SALIDAS_PERSONALIZADAS = True  (SALIDA LÓGICA PROPIA)
────────────────────────────────────────────────────────────────────────────────
  La estrategia define CUÁNDO salir según su propia lógica de indicadores.
  El runner aplica automáticamente (sin que tengas que hacer nada más):

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  REGLA UNIVERSAL — siempre que SALIDAS_PERSONALIZADAS = True:           │
  │                                                                         │
  │  • exit_long / exit_short  (tus columnas) → salida principal           │
  │  • SL fijo de exits.py                    → backstop de emergencia     │
  │  • TP y Trailing                          → desactivados (999 %)       │
  │                                                                         │
  │  exit_type se fuerza a "FIXED" con el sl_pct real del config.          │
  │  Ante crashes, gaps o cualquier movimiento que tu señal no cubra,       │
  │  el SL actúa de última línea de defensa.                                │
  └─────────────────────────────────────────────────────────────────────────┘

  IMPLEMENTACIÓN — sólo necesitas hacer DOS cosas:

    1. Declarar el flag en la clase:
         SALIDAS_PERSONALIZADAS: bool = True

    2. Añadir exit_long / exit_short al final de generate_signals():
         q = q.with_columns([
             self._as_bool(<condicion_salida_long> ).alias("exit_long"),
             self._as_bool(<condicion_salida_short>).alias("exit_short"),
         ])
         return self.finalize_signals(q, keep_cols=[..., "exit_long", "exit_short"])

  Las columnas son booleanas y vectoriales (Polars). El engine las lee barra
  a barra mientras hay una posición abierta y cierra al primer True.

  PRIORIDAD dentro del engine:
    exit_long/short  →  SL (emergencia)  →  (TP/Trailing desactivados)

NOTA CRÍTICA:
  Los % de salida (sl_pct, tp_pct, etc.) son sobre el STAKE (saldo_usado),
  NO sobre el precio de entrada. Ver exits.py para documentación completa.

================================================================================
"""

from typing import Any, Dict, List, Optional, Sequence

import polars as pl


class EstrategiaBase:
    """================================================================================
    CLASE MADRE: ESTRATEGIA BASE
    --------------------------------------------------------------------------------
    ESTA CLASE ES LA BASE DE TODAS LAS ESTRATEGIAS.

    REGLA DE ORO: generate_signals() TIENE QUE SER 100% VECTORIAL (POLARS).
    --------------------------------------------------------------------------------
    """

    # =============================================================================
    # IDENTIDAD
    # =============================================================================
    combinacion_id: int = 0
    name: str = "TEMPLATE_BASE"

    # =============================================================================
    # SALIDAS
    # =============================================================================
    # False → SL/TP/Trailing del engine (exits.py + configuracion.py). DEFAULT.
    # True  → La estrategia aporta exit_long/exit_short en generate_signals().
    #         El runner activa automáticamente el SL fijo como backstop de emergencia.
    #         TP y Trailing quedan desactivados.  Ver docstring del módulo.
    SALIDAS_PERSONALIZADAS: bool = False

    # =============================================================================
    # COLUMNAS DE DATOS REQUERIDAS (OPCIONAL)
    # =============================================================================
    # Declara aquí las columnas extra (más allá de OHLCV) que la estrategia usa.
    # El runner validará que existen en el dataset cargado y mostrará un error claro
    # si el activo no tiene esos datos disponibles (ej: open_interest_5m solo en BTC Vision).
    #
    # Columnas Vision disponibles:
    #   "num_trades", "taker_buy_volume", "taker_sell_volume",
    #   "taker_buy_quote_volume", "taker_sell_quote_volume",
    #   "premium_index_close", "open_interest_5m", "open_interest_value_5m",
    #   "toptrader_count_long_short_ratio_5m", "toptrader_sum_long_short_ratio_5m",
    #   "global_long_short_ratio_5m", "taker_long_short_ratio_5m",
    #   "predicted_funding_rate_1m"
    COLUMNAS_REQUERIDAS: List[str] = []

    # =============================================================================
    # TIMEFRAMES (OPCIONAL)
    # =============================================================================
    # SI UNA ESTRATEGIA DEFINE timeframe_entry / timeframe_exit, EL RUNNER LOS USA
    # PARA SELECCIONAR DF DE SEÑALES Y DF BASE DE BACKTEST.
    timeframe_entry: Optional[str] = None
    timeframe_exit: Optional[str] = None

    # =============================================================================
    # PARAMS OPTUNA (LEGACY / OPCIONAL)
    # =============================================================================
    parametros_optuna: Dict[str, Any] = {}

    # =============================================================================
    # API PUBLICA (CONTRATO)

    # =============================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """DEFINE EL ESPACIO DE BUSQUEDA DE OPTUNA.

        REGLAS:
          - DEVUELVE SOLO PARAMS "HUMANOS" (NO __)
          - LOS __ LOS INYECTA EL RUNNER (CONFIG/EXITS/META)
        """

        return {}

    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        """DEVUELVE TIMEFRAMES EXTRA PARA EL BLENDER (MTF).

        EJEMPLO:
          return ["1h", "4h"]

        NOTA ANTI-LOOKAHEAD:
          LOS DATOS MTF YA VIENEN DESPLAZADOS (shift(1)) Y UNIDOS CON join_asof backward.
          ESO SIGNIFICA QUE close_1h EN UNA VELA 10:15 REPRESENTA LA VELA 1H YA CERRADA.
        """

        return []

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """GENERADOR DE SEÑALES (POLARS FIRST).

        OBLIGATORIO:
          - DEVOLVER timestamp + signal_long + signal_short
          - signal_* DEBEN SER Bool (NULL => False)

        RECOMENDADO:
          - USAR df.lazy() Y HACER 1 SOLO collect() AL FINAL
          - DEVOLVER SOLO COLUMNAS NECESARIAS (MINIMIZA MEMORIA + JOIN)
        """

        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "close"])

        # TEMPLATE "NO TRADING"
        # NOTA: SI CREAS COLUMNAS QUE LUEGO USAS EN SEÑALES, HAZLO EN 2 FASES:
        #   q = q.with_columns([ ...indicadores... ])
        #   q = q.with_columns([ ...señales que referencian indicadores... ])
        # EVITA REFERENCIAR pl.col("X") EN EL MISMO .with_columns() DONDE CREAS "X".
        q = df.lazy()
        q = q.with_columns(
            [
                pl.lit(False).alias("signal_long"),
                pl.lit(False).alias("signal_short"),
            ]
        )
        return self.finalize_signals(q)

    # ── NOTA: decide_exit() fue eliminado ────────────────────────────────────
    # El mecanismo de salidas personalizadas usa columnas vectoriales Polars
    # (exit_long / exit_short) dentro de generate_signals(), NO un hook por trade.
    # Esto mantiene todo 100 % vectorial y compatible con el kernel Numba del engine.
    # Ver docstring del módulo para la guía completa.

    # =============================================================================
    # HELPERS OFICIALES (PARA VELOCIDAD Y CONSISTENCIA)
    # =============================================================================
    @staticmethod
    def _require_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"FALTAN COLUMNAS EN DF: {missing}")

    def _init_params_metadata(self, params: Dict[str, Any]) -> None:
        """INICIALIZA METADATA ESTANDAR EN params.

        NO HACE TRABAJO PESADO; SOLO NORMALIZA CLAVES.
        """

        params.setdefault("__warmup_bars", 1)
        params.setdefault("__indicators_used", [])
        params.setdefault("__indicator_bounds", {})
        params.setdefault("__indicator_specs", {})

        # PARA RICH: INDICA SI LAS SALIDAS SON CUSTOM
        params["__strategy_exit_enabled"] = bool(getattr(self, "SALIDAS_PERSONALIZADAS", False))

    @staticmethod
    def _as_bool(expr: pl.Expr) -> pl.Expr:
        """NORMALIZA EXPRESIONES A BOOL SIN NULLS."""

        return expr.fill_null(False).cast(pl.Boolean)

    # =============================================================================
    # SALIDA ESTANDAR (VALIDACION + SELECT MINIMO)
    # =============================================================================
    @classmethod
    def validate_signals_df(
        cls,
        signals_df: pl.DataFrame,
        *,
        required_cols: Sequence[str] = ("timestamp", "signal_long", "signal_short"),
    ) -> None:
        """VALIDA EL CONTRATO DE SALIDA.

        OBJETIVO:
          - FALLAR RAPIDO CON ERROR CLARO SI UNA ESTRATEGIA DEVUELVE ALGO INVALIDO.

        VALIDACIONES (MINIMAS):
          - EXISTEN required_cols
          - signal_long / signal_short SON BOOL
        """

        missing = [c for c in required_cols if c not in signals_df.columns]
        if missing:
            raise ValueError(
                "signals_df NO CUMPLE EL CONTRATO. FALTAN COLUMNAS: "
                f"{missing}. Columnas actuales: {signals_df.columns}"
            )

        for c in ("signal_long", "signal_short"):
            if c in signals_df.columns:
                dtype = signals_df.schema.get(c)
                if dtype != pl.Boolean:
                    raise ValueError(
                        f"signals_df[{c}] DEBE SER Bool. dtype={dtype}. "
                        "USA _as_bool(...) PARA NORMALIZAR."
                    )

    def finalize_signals(
        self,
        q: pl.LazyFrame,
        *,
        keep_cols: Optional[Sequence[str]] = None,
    ) -> pl.DataFrame:
        """FINALIZA UN LazyFrame A signals_df (SELECT MINIMO + collect + validación).

        USO UNIVERSAL RECOMENDADO:
          - CONSTRUYE TODO EN LAZY
          - LLAMA finalize_signals(...) AL FINAL

        PARAMETROS:
          - keep_cols:
              None  -> DEFAULT MINIMO: [timestamp, signal_long, signal_short]
              list  -> AÑADE INDICADORES/FEATURES PARA REPORTING/PLOTS
        """

        base = ["timestamp", "signal_long", "signal_short"]
        if keep_cols is None:
            cols = base
        else:
            # PRESERVAR ORDEN (timestamp primero), SIN DUPLICADOS
            extras = [c for c in keep_cols if c not in base]
            cols = base + extras

        # AUTO-INCLUDE OHLCV (Case-Insensitive) — EXPLÍCITO Y A PRUEBA DE FALLAS
        # Usamos el schema ya disponible (dict) para evitar collect_schema
        try:
            schema_keys = list(q.schema.keys())
            target_cols = {"open", "high", "low", "close", "volume", "vol"}

            existing_cols_lower = {c.lower(): c for c in schema_keys}
            ohlcv_found = []
            for target in target_cols:
                if target in existing_cols_lower:
                    real_name = existing_cols_lower[target]
                    if real_name not in cols:
                        ohlcv_found.append(real_name)

            if ohlcv_found:
                cols.extend(ohlcv_found)
        except Exception:
            pass  # fallback silencioso

        # Validación defensiva de esquema antes del collect
        try:
            schema_cols = set(q.schema.keys())
            missing = [c for c in cols if c not in schema_cols]
            if missing:
                raise ValueError(
                    f"finalize_signals: faltan columnas antes de collect: {missing}. "
                    f"Schema actual: {sorted(schema_cols)}"
                )
        except Exception:
            # Re-lanzar para dejar rastro claro en el stacktrace
            raise

        try:
            out = q.select(cols).collect()
        except Exception as e:
            # Fallback específico para KeyError:0 (polars ComputeError) → intenta devolver DF vacío con las columnas
            if "KeyError: 0" in str(e):
                try:
                    out = q.select(cols).limit(0).collect()
                except Exception:
                    # Si también falla, re-lanzar con contexto
                    raise RuntimeError(
                        f"finalize_signals collect failed (KeyError 0) y fallback limit(0) también falló: {e}.\n"
                        f"  cols={cols}\n"
                        f"  schema={q.schema}\n"
                        f"  selected_cols_exist={[c for c in cols if c in q.schema]}"
                    ) from e
            else:
                # Propagar contexto detallado para depurar otros errores
                raise RuntimeError(
                    f"finalize_signals collect failed: {e}.\n"
                    f"  cols={cols}\n"
                    f"  schema={q.schema}\n"
                    f"  selected_cols_exist={[c for c in cols if c in q.schema]}"
                ) from e

        self.validate_signals_df(out)
        return out

    # =============================================================================
    # INDICADORES (EXPRESIONES POLARS) - REUSABLES Y SIN LOOPS
    # =============================================================================
    @staticmethod
    def rsi_expr(*, close: pl.Expr, length: int) -> pl.Expr:
        """RSI (WILDER-LIKE) 100% VECTORIAL EN POLARS."""

        length = int(length)
        delta = close.diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        avg_gain = gain.ewm_mean(com=length - 1, min_periods=length)
        avg_loss = loss.ewm_mean(com=length - 1, min_periods=length)
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def ema_expr(*, x: pl.Expr, span: int) -> pl.Expr:
        span = int(span)
        return x.ewm_mean(span=span, adjust=False)

    @staticmethod
    def sma_expr(*, x: pl.Expr, length: int) -> pl.Expr:
        length = int(length)
        return x.rolling_mean(window_size=length)

    @staticmethod
    def rolling_std_expr(*, x: pl.Expr, length: int) -> pl.Expr:
        length = int(length)
        return x.rolling_std(window_size=length)


# =============================================================================
# GUIA UNIVERSAL (SIN EJEMPLOS DE ESTRATEGIAS)
# =============================================================================
#
# OBJETIVO:
#   ESTE ARCHIVO DEBE SER UNA GUIA/CONTRATO UNIVERSAL.
#   LAS ESTRATEGIAS REALES VIVEN EN ARCHIVOS SEPARADOS EN modelox/strategies/.
#
# PATRON RECOMENDADO PARA CUALQUIER INDICADOR (POLARS):
#   1) VALIDAR INPUT:
#        self._init_params_metadata(params)
#        self._require_columns(df, [...])
#
#   2) LAZY FRAME:
#        q = df.lazy()
#
#   3) FASE A (INDICADORES / FEATURES):
#        q = q.with_columns([
#             ...expr.alias("mi_indicador"),
#             ...expr.alias("mi_feature"),
#        ])
#
#   4) FASE B (SEÑALES) - REFERENCIAR SOLO COLUMNAS YA EXISTENTES:
#        sig_long = self._as_bool( ...pl.col("mi_indicador")... )
#        sig_short = self._as_bool( ...pl.col("mi_feature")... )
#        q = q.with_columns([
#             sig_long.alias("signal_long"),
#             sig_short.alias("signal_short"),
#        ])
#
#   5) OUTPUT MINIMO + VALIDACION:
#        return self.finalize_signals(q, keep_cols=[...INDICADORES_QUE_QUIERAS_PLOTEAR...])
#
# ANTI-ERRORES MAS COMUNES:
#   - NO REFERENCIAR COLUMNAS CREADAS EN EL MISMO .with_columns():
#       MAL: q.with_columns([ expr.alias("x"), (pl.col("x")>0).alias("signal_long") ])
#       BIEN: q = q.with_columns([expr.alias("x")]); q = q.with_columns([(pl.col("x")>0)...])
#   - NORMALIZA BOOL:
#       signal_long/short: USA self._as_bool(...) PARA EVITAR NULLs.
#   - 1 SOLO collect() POR TRIAL (IDEAL).
#   - OUTPUT MINIMO: SOLO timestamp + señales + columnas de indicadores necesarias para reporting.
# =============================================================================

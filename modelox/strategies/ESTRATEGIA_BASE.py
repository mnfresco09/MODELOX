from __future__ import annotations

"""================================================================================
ESTRATEGIA BASE (MODELOX) - BASE MAESTRA PARA TODAS LAS ESTRATEGIAS
================================================================================

OBJETIVO
  ESTE ARCHIVO DEFINE EL CONTRATO OFICIAL ENTRE:
    - RUNNER  (OPTUNA + MULTI-TF)
    - STRATEGY (GENERACION DE SEÑALES)
    - ENGINE   (SIMULACION DE TRADES, NUMBA)
    - REPORTERS (RICH / EXCEL / PLOTS)

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

SISTEMA DE SALIDAS
  - POR DEFECTO EL ENGINE CONTROLA SL/TP/TRAILING (exit.py/exits.py)
  - SI NECESITAS SALIDAS PERSONALIZADAS:
      - SALIDAS_PERSONALIZADAS = True
      - IMPLEMENTA decide_exit(...)
  - NOTA CRITICA: LOS % DE SALIDA SON % SOBRE STAKE (SALDO_USADO), NO SOBRE PRECIO

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
    # False = SALIDAS CONTROLADAS POR ENGINE (GLOBAL EXITS)
    # True  = SALIDAS CONTROLADAS POR LA ESTRATEGIA (decide_exit)
    SALIDAS_PERSONALIZADAS: bool = False

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

    def decide_exit(
        self,
        df: pl.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        side: int,
        **kwargs: Any,
    ):
        """SALIDA PERSONALIZADA (SOLO SI SALIDAS_PERSONALIZADAS=True).

        CONTRATO (MINIMO):
          - RETORNA None PARA "NO SALIR"
          - O RETORNA UNA ESTRUCTURA QUE EL ENGINE/EXITS PUEDA INTERPRETAR

        NOTA:
          HOY EL MOTOR PRINCIPAL USA EL SISTEMA GLOBAL stake-based.
          ESTE HOOK QUEDA PARA EXTENSION FUTURA (NO ROMPER COMPAT).
        """

        return None

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

        out = q.select(cols).collect()
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
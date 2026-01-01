"""# =============================================================================
# ESTRATEGIA BASE (TEMPLATE / DOCUMENTACIÓN)
# =============================================================================
#
# ESTE ARCHIVO NO ES UNA ESTRATEGIA EJECUTABLE.
#
# MODELOX auto-descubre estrategias en `modelox/strategies/*.py` vía
# `modelox/strategies/registry.py`.
#
# Una clase se considera "estrategia" SOLO si cumple:
#   - `name` es `str` no vacío
#   - `combinacion_id` es `int` > 0
#   - tiene métodos `suggest_params()` y `generate_signals()`
#
# Para asegurar que esto sea solo una guía, aquí:
#   - `combinacion_id = 0`
#   - `name = ""`
#
# Así queda EXCLUIDA del descubrimiento.
# =============================================================================


"""# =============================================================================
# MAPA MENTAL DEL SISTEMA (CÓMO CONECTA TODO)
# =============================================================================
#
# (1) Runner / Optuna
#     - El runner instancia una estrategia descubierta.
#     - Por trial, llama `strategy.suggest_params(trial)`.
#     - Con esos params, llama `strategy.generate_signals(df, params)`.
#
# (2) Strategy
#     - NO ejecuta órdenes. NO calcula métricas. NO renderiza plots.
#     - Su responsabilidad es:
#         a) calcular sus indicadores (por trial) DENTRO de la estrategia
#         b) producir columnas `signal_long` y `signal_short`
#         c) exponer metadatos para el plot en `params` (opcional)
#
# (3) Indicadores (A PARTIR DE AHORA)
#     - NO existe un módulo central de indicadores.
#     - Cada estrategia es dueña de sus fórmulas y columnas.
#     - Convención recomendada:
#         - Añadir columnas numéricas (Float64) al df (Polars)
#         - Declarar `params["__indicators_used"]` con los nombres EXACTOS de columnas a graficar.
#
# (4) Engine / Backtest
#     - Consume `df_signals` (Pandas) + `signal_long/signal_short`.
#     - Ejecuta la simulación, exits, comisiones, equity, trades, etc.
#
#     EXITS (NOVEDAD)
#     - La lógica de salida GLOBAL vive en `modelox/core/exits.py`.
#     - Modelo global actual:
#         SL/TP por ATR (fijos al inicio) + ejecución intra-vela + TIME EXIT.
#     - Si una estrategia define una lógica distinta, se prioriza la estrategia.
#       Para eso, la estrategia puede implementar opcionalmente:
#
#           decide_exit(df, params, entry_idx, entry_price, side, *, saldo_apertura) -> ExitDecision | None
#
#       Si devuelve `ExitDecision`, el engine usará esa salida.
#
# (5) Reporting
#     - Construye artefactos (trades, equity, métricas, df) por trial.
#     - `PlotReporter` llama al módulo de plot.
#
# (6) Plot (ÚNICO LUGAR)
#     - TODO lo de gráfica vive en `visual/grafico.py`.
#     - El plot recibe `params_reporting` que incluye `__indicators_used` del trial.
#     - Por eso el gráfico puede renderizar EXACTAMENTE los indicadores usados
#       en ese trial.
# =============================================================================


from __future__ import annotations

from typing import Any, Dict

import polars as pl


class EstrategiaBase:
    """# =============================================================================
    # ESTRATEGIA BASE (GUÍA)
    # =============================================================================
    #
    # Usa esta clase como referencia para crear estrategias reales.
    #
    # Reglas obligatorias para estrategias reales:
    #
    #   1) `combinacion_id: int` > 0 y ÚNICO
    #   2) `name: str` no vacío (sirve para reportes/archivos)
    #   3) `suggest_params(self, trial) -> Dict[str, Any]`
    #   4) `generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame`
    #
    # Convenciones clave:
    #
    #   - `params` es mutable (dict) y se usa para:
    #       a) params de la estrategia (los que sugiere Optuna)
    #       b) "params internos" del sistema (prefijo `__`)
    #       c) valores auxiliares para el plot (p.ej. umbrales)
    #
    #   - Claves internas recomendadas:
    #       `__warmup_bars`: int
    #           barras iniciales que NO se deben tradear (indicadores aún inestables)
    #       `__indicators_used`: List[str]
    #           lista EXACTA de columnas de indicadores para graficar en el trial
    #
    #   - Timeframes (NUEVO)
    #       NORMA GENERAL:
    #         - El sistema usa SIEMPRE el timeframe base definido en `general/configuracion.py`:
    #               TIMEFRAME = 5   # minutos
    #         - Si una estrategia NO especifica nada, tanto ENTRADA como SALIDA usan ese timeframe.
    #
    #       SI QUIERES OTRO TIMEFRAME EN LA ESTRATEGIA (OPCIONAL):
    #         - Define atributos en la clase:
    #               timeframe_entry = 15   # o "15m" o 60 / "1h"
    #               timeframe_exit  = 60   # o "1h"
    #
    #       CÓMO FUNCIONA POR DEBAJO:
    #         - `generate_signals()` se ejecuta sobre el dataframe del `timeframe_entry`.
    #         - El backtest (engine) se ejecuta en el timeframe base (CONFIG TIMEFRAME).
    #         - Si `timeframe_entry != TIMEFRAME`, las señales se alinean al timeframe base
    #           sin lookahead (join_asof hacia atrás).
    #         - Si implementas `decide_exit()`, puedes usar el TF de salida leyendo:
    #               params["__df_exit_tf"]
    #           y también:
    #               params["__timeframe_base"], params["__timeframe_entry"], params["__timeframe_exit"]
    #
    #   - Señales:
    #       Debes devolver `df` con estas 2 columnas booleanas:
    #         - `signal_long`
    #         - `signal_short`
    #
    #   - Indicadores:
    #       Se calculan dentro de la estrategia (este archivo es guía).

    #   - `parametros_optuna` (IMPORTANTE):
    #       `ejecutar.py` asume que existe este atributo para listar/mostrar
    #       qué parámetros usa la estrategia. Aunque no optimices nada, define:
    #           parametros_optuna: Dict[str, Any] = {}
    #
    # =============================================================================
    """

    # =====================================================================
    # IMPORTANTE: Para que NO se auto-descubra como estrategia real
    # =====================================================================
    combinacion_id = 0
    name = ""

    # =====================================================================
    # Parámetros de Optuna (RECOMENDADO / compatible con ejecutar.py)
    # - `ejecutar.py` lo usa para listar claves (p.ej. en reporting/UI).
    # - Si tu estrategia no optimiza nada, déjalo como `{}` pero NO lo omitas.
    # =====================================================================
    parametros_optuna: Dict[str, Any] = {
        # "rsi_period": (7, 21, 1),
        # "adx_period": (7, 30, 1),
        # "entry_thr": (0.5, 2.0, 0.05),
    }

    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """# =============================================================================
        # suggest_params(trial)
        # =============================================================================
        #
        # Objetivo
        #   - Definir el espacio de búsqueda de Optuna.
        #   - Retornar un dict "plano" con parámetros numéricos.
        #
        # Reglas
        #   - Los nombres de claves deben ser estables (Optuna los registra).
        #   - Evita condicionales que cambien las claves retornadas.
        #   - Valida coherencia (p.ej. fast < slow) antes de retornarlos.
        #
        # Retorno
        #   Dict[str, Any]
        #     ejemplo: {"rsi_period": 14, "entry_thr": 1.2}
        # =============================================================================
        """

        # Esto es un template; en estrategias reales define rangos aquí.
        return {}

    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """# =============================================================================
        # generate_signals(df, params)
        # =============================================================================
        #
        # Contrato
        #   - input: `df` (Polars) con OHLCV + timestamp
        #   - output: `df` (Polars) con columnas nuevas:
        #       - indicadores (las que tú definas)
        #       - `signal_long`  (bool)
        #       - `signal_short` (bool)
        #
        # Dónde se calculan indicadores
        #   - A partir de ahora: dentro de la estrategia.
        #   - Aquí:
        #       1) calculas y añades columnas al df
        #       2) defines `params["__indicators_used"]`
        #       2.1) (opcional) defines `params["__indicator_bounds"]` / `__indicator_specs`
        #
        # Warmup
        #   - Siempre setea `params["__warmup_bars"]`.
        #   - Debe cubrir: periodos de indicadores + ventanas rolling + márgen.
        #
        # Plot por trial
        #   - `visual/grafico.py` mira `params["__indicators_used"]`.
        #   - Por eso esta lista debe ser EXACTA y derivada de las columnas que realmente añadiste.
        #   - Para líneas/umbrales dinámicos (OB/OS, +/-2, etc):
        #       params["__indicator_bounds"] = {"col": {"hi":..., "lo":..., "mid":...}}
        #     (El runner lo propaga a reporting y la gráfica lo dibuja sin hardcode.)
        # =============================================================================
        """

        # -----------------------------------------------------------------
        # 1) Lee params con defaults (y normaliza)
        # -----------------------------------------------------------------
        # Ejemplo RSI: Optuna puede variar el periodo y los límites
        rsi_period = int(params.get("rsi_period", 14))
        rsi_period = max(2, rsi_period)

        # Rangos dinámicos (p.ej. Optuna sugiere rsi_overbought 60..80 y rsi_oversold 20..40)
        rsi_overbought = float(params.get("rsi_overbought", 70))
        rsi_oversold = float(params.get("rsi_oversold", 30))

        # -----------------------------------------------------------------
        # 2) Define warmup (CRÍTICO)
        # -----------------------------------------------------------------
        params["__warmup_bars"] = rsi_period + 10

        # -----------------------------------------------------------------
        # 2.1) (NUEVO) Bounds/umbrales para la gráfica (por trial)
        # -----------------------------------------------------------------
        # La gráfica dibuja estas líneas dentro del panel del indicador.
        # Se actualizan automáticamente porque vienen en `params` (trial) y el runner los propaga.
        params["__indicator_bounds"] = {
            "rsi": {"hi": rsi_overbought, "lo": rsi_oversold, "mid": 50.0}
        }

        # -----------------------------------------------------------------
        # 2.2) (NUEVO) Specs para forzar cómo se grafica (por trial)
        # -----------------------------------------------------------------
        # Útil cuando:
        # - quieres forzar overlay vs subpanel
        # - quieres renombrar el panel
        # - quieres cambiar precisión o tipo (line/histogram)
        # Nota: `bounds` aquí puede sobrescribir `__indicator_bounds` si lo defines.
        params["__indicator_specs"] = {
            "rsi": {
                "panel": "sub",         # "overlay" o "sub"
                "type": "line",         # "line" o "histogram"
                "name": f"RSI ({rsi_period})",
                "precision": 2,
            }
        }

        # -----------------------------------------------------------------
        # 3) Calcula indicadores (inline)
        # -----------------------------------------------------------------
        # Ejemplo: RSI (SMA) inline
        if "close" in df.columns:
            delta = pl.col("close").diff()
            gain = pl.when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
            avg_gain = gain.rolling_mean(window_size=rsi_period, min_periods=rsi_period)
            avg_loss = loss.rolling_mean(window_size=rsi_period, min_periods=rsi_period)
            rs = avg_gain / avg_loss
            rsi = pl.when(avg_loss == 0).then(100.0).otherwise(100.0 - (100.0 / (1.0 + rs)))
            df = df.with_columns(rsi.cast(pl.Float64).alias("rsi"))
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("rsi"))

        # Lista exacta de columnas a graficar este trial
        params["__indicators_used"] = ["rsi"]

        # -----------------------------------------------------------------
        # 4) Construye señales (vectorizado Polars)
        # -----------------------------------------------------------------
        # Ejemplo didáctico (NO es recomendación de trading):
        #   LONG  si RSI cruza arriba de oversold
        #   SHORT si RSI cruza abajo de overbought
        rsi = pl.col("rsi")
        cross_up = (rsi > rsi_oversold) & (rsi.shift(1) <= rsi_oversold)
        cross_dn = (rsi < rsi_overbought) & (rsi.shift(1) >= rsi_overbought)

        signal_long = cross_up.fill_null(False)
        signal_short = cross_dn.fill_null(False)

        return df.with_columns(
            [
                signal_long.alias("signal_long"),
                signal_short.alias("signal_short"),
            ]
        )


    # =============================================================================
    # OVERRIDE OPCIONAL DE EXITS (SI TU ESTRATEGIA NECESITA SALIDAS PROPIAS)
    # =============================================================================
    #
    # Si implementas este método en una estrategia REAL, el engine llamará aquí
    # en vez de usar la salida global de `modelox/core/exits.py`.
    #
    # Reglas:
    #   - Devuelve `ExitDecision(exit_idx=..., reason=..., exit_price=...)` o `None`.
    #   - `exit_idx` debe ser un índice de barra válido (0..n-1).
    #   - Si `exit_price` es None, el engine usará `close[exit_idx]`.
    #   - `reason` se copiará a `tipo_salida` en el trade.
    #
    # Importante:
    #   - Esta lógica es ESPECÍFICA de la estrategia.
    #   - Los parámetros globales (ATR, SL/TP, TIME) siguen viviendo en exits.py.
    #
    # def decide_exit(
    #     self,
    #     df: pl.DataFrame,
    #     params: Dict[str, Any],
    #     entry_idx: int,
    #     entry_price: float,
    #     side: str,
    #     *,
    #     saldo_apertura: float,
    # ) -> "ExitDecision | None":
    #     ...


# =============================================================================
# CHECKLIST PARA CREAR UNA ESTRATEGIA NUEVA (COPIAR/PEGAR)
# =============================================================================
#
# 1) Crea un archivo nuevo en `modelox/strategies/`:
#       - nombre recomendado: `strategy_<id>_<nombre>.py`
#
# 2) Define la clase (UNA por archivo):
#       class StrategyXXXX:
#           combinacion_id = <int único y >0>
#           name = "<NombreCorto>"
#
# 3) Implementa:
#       - suggest_params(self, trial) -> Dict[str, Any]
#       - generate_signals(self, df, params) -> pl.DataFrame
#       - parametros_optuna: Dict[str, Any]  (para compatibilidad con ejecutar.py)
#
# 4) Dentro de generate_signals:
#       a) normaliza parámetros
#       b) define `params["__warmup_bars"]`
#       c) calcula tus indicadores y añade columnas al `df`
#       d) define `params["__indicators_used"] = ["col1", "col2", ...]` con las columnas EXACTAS a graficar
#       e) (opcional) define `params["__indicator_bounds"]` si quieres niveles en el plot
#       f) crea `signal_long` y `signal_short` (bool)
#       g) retorna df con esas columnas
#
# 5) NUNCA:
#       - importes módulos de plot desde estrategias
#       - uses `visual/*` para cálculos
#
# 6) Si necesitas un indicador nuevo:
#       - impleméntalo dentro de la estrategia donde lo necesites
#       - si lo reutilizas en muchas estrategias, copia/pega el helper (por ahora)
#
# 7) Si quieres que se grafique algo:
#       - asegúrate que sea una columna incluida en `__indicators_used`
#       - la gráfica no calcula nada: solo pinta columnas ya existentes
# =============================================================================

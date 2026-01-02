from __future__ import annotations

"""# =============================================================================
# 🎯 ESTRATEGIA BASE - GUÍA COMPLETA PARA CREAR ESTRATEGIAS EN MODELOX
# =============================================================================
#
# ⚠️ ESTE ARCHIVO NO ES UNA ESTRATEGIA EJECUTABLE (combinacion_id = 0)
#
# 📚 CONTENIDO DE ESTA GUÍA:
#   1. Arquitectura del Sistema (cómo conecta todo)
#   2. Template Mínimo (clase base con ejemplos)
#   3. Ejemplos Completos (6 estrategias diferentes)
#   4. Patrones Comunes (helpers reutilizables)
#   5. Multi-Timeframe (entrada ≠ salida)
#   6. Exits Personalizados (override del sistema global)
#   7. Optimización de Performance
#   8. Checklist Final
#
# =============================================================================


"""# =============================================================================
# 📐 PARTE 1: ARQUITECTURA DEL SISTEMA
# =============================================================================
#
# FLUJO COMPLETO: ejecutar.py → Runner → Optuna → Strategy → Engine → Reporters
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (1) Runner / Optuna                                                     │
# │     - Instancia estrategia descubierta                                  │
# │     - Por cada trial:                                                   │
# │         a) Llama strategy.suggest_params(trial) → params                │
# │         b) Llama strategy.generate_signals(df, params) → df_signals     │
# │         c) Pasa df_signals al engine                                    │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (2) Strategy (TU CÓDIGO)                                                │
# │     - NO ejecuta órdenes                                                │
# │     - NO calcula métricas                                               │
# │     - NO renderiza gráficos                                             │
# │     - SÍ calcula indicadores                                            │
# │     - SÍ genera señales (signal_long/signal_short)                      │
# │     - SÍ puede override exits (opcional)                                │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (3) Indicadores                                                         │
# │     - NO existe módulo central de indicadores                           │
# │     - Cada estrategia implementa sus propias fórmulas inline            │
# │     - Se calculan con Polars (vectorizado, rápido)                      │
# │     - Se añaden como columnas al DataFrame                              │
# │     - Se declaran en params["__indicators_used"]                        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (4) Engine / Backtest                                                   │
# │     - Consume df_signals con signal_long/signal_short                   │
# │     - Genera trades (entry, exit, tipo_salida)                          │
# │     - Simula ejecución (quantity, PnL, comisiones, equity)              │
# │     - Usa exits.py GLOBAL (salvo que estrategia override)               │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (5) Exits (exits.py)                                                    │
# │     - Sistema GLOBAL (SL/TP por ATR + TIME EXIT)                        │
# │     - 2 modos:                                                          │
# │         a) atr_fixed: SL/TP fijos al inicio del trade                   │
# │         b) trailing: SL ajustable + emergency SL fijo                   │
# │     - Estrategia puede override con decide_exit()                       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (6) Reporting                                                           │
# │     - Construye artefactos por trial (trades, equity, métricas, df)     │
# │     - RichReporter: Consola Bloomberg-style                             │
# │     - ExcelReporter: CSV append rápido → Excel al final                 │
# │     - PlotReporter: HTML interactivo (solo top-5)                       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ (7) Plot (visual/grafico.py)                                            │
# │     - Único lugar donde se grafican resultados                          │
# │     - Recibe params["__indicators_used"] del trial                      │
# │     - Dibuja indicadores dinámicamente (sin hardcode)                   │
# │     - Respeta bounds dinámicos (OB/OS, +/-2σ, etc.)                     │
# └─────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================


"""# =============================================================================
# 📋 PARTE 2: CONTRATO DE UNA ESTRATEGIA (Reglas Obligatorias)
# =============================================================================
#
# Para que MODELOX auto-descubra tu estrategia, la clase DEBE cumplir:
#
#   ✅ combinacion_id: int > 0  (único, identifica la estrategia)
#   ✅ name: str (no vacío, para reportes/archivos)
#   ✅ suggest_params(self, trial) -> Dict[str, Any]
#   ✅ generate_signals(self, df: pl.DataFrame, params: Dict) -> pl.DataFrame
#   ✅ parametros_optuna: Dict[str, Any] (para compatibilidad con ejecutar.py)
#
# OPCIONAL:
#   ⭐ timeframe_entry: int | str | None  (None = usa CONFIG.TIMEFRAME)
#   ⭐ timeframe_exit: int | str | None   (None = usa CONFIG.TIMEFRAME)
#   ⭐ decide_exit(...)  (override del sistema global de salidas)
#
# =============================================================================


"""# =============================================================================
# 🔑 PARTE 3: CLAVES INTERNAS DEL SISTEMA (params["__xxx"])
# =============================================================================
#
# El diccionario `params` tiene 2 tipos de valores:
#
# A) PARÁMETROS DE ESTRATEGIA (sugeridos por Optuna)
#    - "rsi_period", "ma_fast", "threshold", etc.
#    - Vienen de suggest_params()
#
# B) METADATOS DEL SISTEMA (prefijo "__")
#    - Los crea la estrategia en generate_signals()
#    - Los consume el runner/engine/reporters
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CLAVES INTERNAS OBLIGATORIAS:                                           │
# │                                                                         │
# │ params["__warmup_bars"]: int                                            │
# │   - Barras iniciales donde NO se debe tradear                           │
# │   - Debe cubrir: períodos de indicadores + ventanas rolling + margen    │
# │   - Ejemplo: RSI(14) + Rolling(50) → warmup = 64 mínimo                 │
# │                                                                         │
# │ params["__indicators_used"]: List[str]                                  │
# │   - Lista EXACTA de columnas de indicadores a graficar                  │
# │   - Debe coincidir con columnas añadidas al DataFrame                   │
# │   - Ejemplo: ["rsi", "bb_upper", "bb_lower"]                            │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CLAVES INTERNAS OPCIONALES:                                             │
# │                                                                         │
# │ params["__indicator_bounds"]: Dict[str, Dict[str, float]]               │
# │   - Niveles dinámicos para graficar (por trial)                         │
# │   - Formato:                                                            │
# │       {                                                                 │
# │         "rsi": {"hi": 70, "lo": 30, "mid": 50},                         │
# │         "zscore": {"hi": 2.0, "lo": -2.0, "mid": 0.0}                   │
# │       }                                                                 │
# │   - El plot dibuja estas líneas automáticamente                         │
# │                                                                         │
# │ params["__indicator_specs"]: Dict[str, Dict[str, Any]]                  │
# │   - Configuración avanzada de cómo graficar cada indicador              │
# │   - Formato:                                                            │
# │       {                                                                 │
# │         "rsi": {                                                        │
# │           "panel": "sub",          # "overlay" o "sub"                  │
# │           "type": "line",          # "line" o "histogram"               │
# │           "name": "RSI (14)",      # Nombre en leyenda                  │
# │           "precision": 2,          # Decimales en tooltip               │
# │           "bounds": {...}          # Sobrescribe __indicator_bounds     │
# │         }                                                               │
# │       }                                                                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CLAVES INYECTADAS POR EL RUNNER (solo lectura):                         │
# │                                                                         │
# │ params["__activo"]: str                   # "BTC", "GOLD", etc.         │
# │ params["ACTIVO"]: str                     # Alias de __activo           │
# │ params["__exit_settings"]: ExitSettings   # Config de SL/TP/TIME        │
# │ params["__timeframe_base"]: str           # "5m", "1h", etc.            │
# │ params["__timeframe_entry"]: str          # TF usado en generate_signals│
# │ params["__timeframe_exit"]: str           # TF usado en decide_exit     │
# │ params["__df_exit_tf"]: pl.DataFrame      # DataFrame del TF de salida  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================

from typing import Any, Dict, List

import polars as pl


class EstrategiaBase:
    """# ==========================================================================
    # 🎓 TEMPLATE MÍNIMO - Estructura Base de una Estrategia
    # ==========================================================================
    #
    # Esta clase NO es ejecutable (combinacion_id = 0).
    # Úsala como referencia para crear estrategias reales.
    #
    # Más abajo encontrarás 6 EJEMPLOS COMPLETOS de estrategias diferentes.
    # ==========================================================================
    """

    # ======================================================================
    # IDENTIFICACIÓN (Obligatorio)
    # ======================================================================
    combinacion_id = 0  # > 0 para estrategias reales (único)
    name = ""  # No vacío para estrategias reales

    # ======================================================================
    # TIMEFRAMES (Opcional)
    # ======================================================================
    # Si NO defines esto, se usa CONFIG.TIMEFRAME para entrada y salida
    # timeframe_entry = None  # None = usa CONFIG.TIMEFRAME
    # timeframe_exit = None   # None = usa CONFIG.TIMEFRAME

    # ======================================================================
    # PARÁMETROS DE OPTUNA (Obligatorio para compatibilidad)
    # ======================================================================
    # Formato: {"param_name": (min, max, step)} para ints/floats
    #          {"param_name": ["value1", "value2"]} para categoricals
    parametros_optuna: Dict[str, Any] = {}
    # Ejemplo:
    # parametros_optuna: Dict[str, Any] = {
    #     "rsi_period": (7, 21, 1),
    #     "rsi_overbought": (65, 80, 1),
    #     "rsi_oversold": (20, 35, 1),
    # }

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

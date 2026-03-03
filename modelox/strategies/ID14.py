from __future__ import annotations
"""
================================================================================
MODELOX/STRATEGIES/ID14.PY
================================================================================
ID        : 14
NOMBRE    : ADXDI — ADX + Cruce DI+/DI- + Confirmación Z-score ADX
MERCADO   : Crypto
TIMEFRAME : El que el sistema pase (sin timeframe fijo)
--------------------------------------------------------------------------------
LÓGICA DE ENTRADA:
  ADX + DI+/DI- (Wilder, period=14 fijo):
    - True Range (TR) = max(H-L, |H-Cprev|, |L-Cprev|)
    - +DM = max(H - H_prev, 0) si H-H_prev > L_prev-L, else 0
    - -DM = max(L_prev - L, 0) si L_prev-L > H-H_prev, else 0
    - ATR14  = EWM(TR,  com=13)
    - +DI14  = EWM(+DM, com=13) / ATR14 * 100
    - -DI14  = EWM(-DM, com=13) / ATR14 * 100
    - DX     = |+DI - -DI| / (+DI + -DI) * 100
    - ADX    = EWM(DX, com=13)

  CAPA 1 — FILTRO TENDENCIA:
    - ADX > adx_threshold (Optuna [20,40] step=5)
    - Confirma que hay tendencia activa, no rango

  CAPA 2 — DIRECCIÓN POR CRUCE DI+/DI-:
    - Cruce alcista : DI+ cruza DI- hacia arriba (DI+>DI- y antes DI+<=DI-)
    - Cruce bajista : DI- cruza DI+ hacia arriba (DI->DI+ y antes DI-<=DI+)

  CAPA 3 — CONFIRMACIÓN Z-score ADX:
    - z_adx = (ADX - mean(ADX, 14)) / std(ADX, 14)  [ventana=14, fijo]
    - z_adx >= 1.5 fijo → la fuerza de tendencia es estadísticamente relevante
    - Filtra señales donde ADX supera el umbral pero no es extremo históricamente

  SEÑAL LONG  : ADX > adx_threshold AND cruce DI+↑DI- AND z_adx >= 1.5
                Solo en FLANCO. Mutuamente excluyente con SHORT.

  SEÑAL SHORT : ADX > adx_threshold AND cruce DI-↑DI+ AND z_adx >= 1.5
                Solo en FLANCO. Mutuamente excluyente con LONG.

SALIDAS:
  SALIDAS_PERSONALIZADAS = False
  → SL / TP controlados 100% por exits.py (engine global)

RANGOS OPTUNA (1 parámetro — máximo anti-overfitting):
  adx_threshold : int [20, 40] step=5

PARÁMETROS FIJOS:
  adx_period  : 14   (Wilder clásico)
  z_window    : 14   (misma que adx_period)
  z_threshold : 1.5  (umbral Z-score ADX)

TIMEFRAME:
  Sin timeframe fijo — usa el que el sistema pase.

IMPLEMENTACIÓN:
  - 100% Polars vectorial (Zero Pandas, Zero loops por fila)
  - ADX/DI via ewm_mean encadenado (Wilder-like, com=period-1)
  - Z-score ADX via rolling_mean + rolling_std nativos Polars
  - 1 solo .collect() al final en finalize_signals()
================================================================================
"""

from typing import Any, Dict, List
import polars as pl

from modelox.strategies.ESTRATEGIA_BASE import EstrategiaBase


class EstrategiaID14ADXDI(EstrategiaBase):
    """
    ADXDI — ADX + Cruce DI+/DI- con confirmación Z-score.
    Entra en tendencias estadísticamente fuertes cuando el momentum
    direccional cambia de manos con fuerza real detrás.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    combinacion_id: int = 14
    name: str           = "ADXDI"

    # ── Salidas: engine global vía exits.py ────────────────────────────────────
    SALIDAS_PERSONALIZADAS: bool = False

    # ── Timeframe: None — usa el que el sistema pase ───────────────────────────

    # ==========================================================================
    # ESPACIO DE BÚSQUEDA OPTUNA — 1 parámetro
    # ==========================================================================
    def suggest_params(self, trial: Any) -> Dict[str, Any]:
        """
        1 parámetro Optuna (máximo anti-overfitting).
        Fijos: adx_period=14, z_window=14, z_threshold=1.5
        Total combinaciones: 5 — espacio mínimo absoluto.
        """
        return {
            "adx_threshold": trial.suggest_int("adx_threshold", 5, 30, step=1),
        }

    # ==========================================================================
    # TIMEFRAMES EXTRA (MTF) — no requeridos
    # ==========================================================================
    def get_required_timeframes(self, params: Dict[str, Any]) -> List[str]:
        return []

    # ==========================================================================
    # GENERATE SIGNALS — 100% POLARS VECTORIAL, 1 COLLECT
    # ==========================================================================
    def generate_signals(self, df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        FASES:
          A) True Range + DM+ + DM- (componentes Wilder)
          B) ATR14, +DI14, -DI14 via ewm_mean (com=13)
          C) DX + ADX via ewm_mean (com=13)
          D) Z-score del ADX (ventana=14)
          E) Cruce DI+/DI- (cambio de signo del diferencial)
          F) Condiciones base Long/Short con triple filtro
          G) Flancos: señal solo en cambio False→True
          H) finalize_signals (1 collect)
        """

        # ── Init ───────────────────────────────────────────────────────────────
        self._init_params_metadata(params)
        self._require_columns(df, ["timestamp", "high", "low", "close"])

        # ── Parámetros ────────────────────────────────────────────────────────
        adx_threshold = int  (params.get("adx_threshold", 25))
        adx_period    = 14    # FIJO: Wilder clásico
        z_window      = 14    # FIJO: misma ventana
        z_threshold   = 1.5   # FIJO: umbral Z-score

        # ── Metadata para reporter / plots ────────────────────────────────────
        params["__warmup_bars"]     = adx_period * 3 + z_window + 2
        params["__indicators_used"] = ["adx", "di_plus", "di_minus", "z_adx"]
        params["__indicator_specs"] = {
            "adx":      {"panel": "sub1", "color": "#FFD600", "tipo": "line"},
            "di_plus":  {"panel": "sub1", "color": "#00E676", "tipo": "line"},
            "di_minus": {"panel": "sub1", "color": "#FF1744", "tipo": "line"},
            "z_adx":    {"panel": "sub2", "color": "#AB47BC", "tipo": "line"},
        }
        params["__indicator_bounds"] = {
            "adx":   {"lo": 0.0,         "hi": 100.0, "mid": float(adx_threshold)},
            "z_adx": {"lo": -z_threshold, "hi": None,  "mid": z_threshold},
        }

        q = df.lazy()

        # ══════════════════════════════════════════════════════════════════════
        # FASE A — True Range + DM+ + DM-
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("high") .shift(1).alias("_h_prev"),
            pl.col("low")  .shift(1).alias("_l_prev"),
            pl.col("close").shift(1).alias("_c_prev"),
        ])

        q = q.with_columns([
            # True Range
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("_c_prev")).abs(),
                (pl.col("low")  - pl.col("_c_prev")).abs(),
            ).alias("_tr"),

            # +DM: movimiento alcista neto
            pl.when(
                (pl.col("high") - pl.col("_h_prev")) >
                (pl.col("_l_prev") - pl.col("low"))
            )
            .then(pl.max_horizontal(pl.col("high") - pl.col("_h_prev"), pl.lit(0.0)))
            .otherwise(0.0)
            .alias("_dm_plus"),

            # -DM: movimiento bajista neto
            pl.when(
                (pl.col("_l_prev") - pl.col("low")) >
                (pl.col("high") - pl.col("_h_prev"))
            )
            .then(pl.max_horizontal(pl.col("_l_prev") - pl.col("low"), pl.lit(0.0)))
            .otherwise(0.0)
            .alias("_dm_minus"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE B — ATR14, +DI14, -DI14 via ewm_mean (Wilder, com=period-1)
        # ══════════════════════════════════════════════════════════════════════
        com = adx_period - 1  # com=13 para period=14

        q = q.with_columns([
            pl.col("_tr")      .ewm_mean(com=com, min_periods=adx_period).alias("_atr14"),
            pl.col("_dm_plus") .ewm_mean(com=com, min_periods=adx_period).alias("_sdm_plus"),
            pl.col("_dm_minus").ewm_mean(com=com, min_periods=adx_period).alias("_sdm_minus"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_atr14").abs() > 1e-12)
            .then(pl.col("_sdm_plus")  / pl.col("_atr14") * 100.0)
            .otherwise(0.0)
            .alias("di_plus"),

            pl.when(pl.col("_atr14").abs() > 1e-12)
            .then(pl.col("_sdm_minus") / pl.col("_atr14") * 100.0)
            .otherwise(0.0)
            .alias("di_minus"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE C — DX + ADX
        # DX  = |DI+ - DI-| / (DI+ + DI-) * 100
        # ADX = EWM(DX, com=13)
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.when((pl.col("di_plus") + pl.col("di_minus")).abs() > 1e-12)
            .then(
                (pl.col("di_plus") - pl.col("di_minus")).abs() /
                (pl.col("di_plus") + pl.col("di_minus")) * 100.0
            )
            .otherwise(0.0)
            .alias("_dx")
        ])

        q = q.with_columns([
            pl.col("_dx")
            .ewm_mean(com=com, min_periods=adx_period)
            .alias("adx")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE D — Z-score del ADX (ventana=14, fijo)
        # z_adx >= 1.5 → ADX estadísticamente elevado vs su historia reciente
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            pl.col("adx").rolling_mean(window_size=z_window).alias("_adx_mean"),
            pl.col("adx").rolling_std (window_size=z_window).alias("_adx_std"),
        ])

        q = q.with_columns([
            pl.when(pl.col("_adx_std").abs() > 1e-12)
            .then((pl.col("adx") - pl.col("_adx_mean")) / pl.col("_adx_std"))
            .otherwise(0.0)
            .alias("z_adx")
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE E — Cruce DI+/DI- (cambio de signo del diferencial)
        # Diferencial = DI+ - DI-
        # Cruce alcista: diferencial cruza de negativo a positivo
        # Cruce bajista: diferencial cruza de positivo a negativo
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (pl.col("di_plus") - pl.col("di_minus")).alias("_di_diff")
        ])

        q = q.with_columns([
            pl.col("_di_diff").shift(1).fill_null(0.0).alias("_di_diff_prev")
        ])

        q = q.with_columns([
            (
                (pl.col("_di_diff_prev") <= 0.0) &
                (pl.col("_di_diff")      >  0.0)
            ).fill_null(False).alias("_cruce_long"),

            (
                (pl.col("_di_diff_prev") >= 0.0) &
                (pl.col("_di_diff")      <  0.0)
            ).fill_null(False).alias("_cruce_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE F — Condiciones base con triple filtro
        # Long:  cruce DI+↑DI- AND ADX > threshold AND z_adx >= 1.5
        # Short: cruce DI-↑DI+ AND ADX > threshold AND z_adx >= 1.5
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            (
                pl.col("_cruce_long") &
                (pl.col("adx")   > float(adx_threshold)) &
                (pl.col("z_adx") >= z_threshold)
            ).fill_null(False).alias("_cond_long"),

            (
                pl.col("_cruce_short") &
                (pl.col("adx")   > float(adx_threshold)) &
                (pl.col("z_adx") >= z_threshold)
            ).fill_null(False).alias("_cond_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE G — Flancos: señal solo en cambio False→True
        # ══════════════════════════════════════════════════════════════════════
        q = q.with_columns([
            self._as_bool(
                pl.col("_cond_long") &
                ~pl.col("_cond_long").shift(1).fill_null(False)
            ).alias("signal_long"),

            self._as_bool(
                pl.col("_cond_short") &
                ~pl.col("_cond_short").shift(1).fill_null(False) &
                ~pl.col("_cond_long")
            ).alias("signal_short"),
        ])

        # ══════════════════════════════════════════════════════════════════════
        # FASE H — finalize_signals (1 collect + validación contrato)
        # ══════════════════════════════════════════════════════════════════════
        return self.finalize_signals(
            q,
            keep_cols=[
                "adx",
                "di_plus",
                "di_minus",
                "z_adx",
            ],
        )
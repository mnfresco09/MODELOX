"""modelox/optimizers/qmc.py

═══════════════════════════════════════════════════════════════════════════════
    ██████╗ ███╗   ███╗ ██████╗
   ██╔═══██╗████╗ ████║██╔════╝
   ██║   ██║██╔████╔██║██║     
   ██║▄▄ ██║██║╚██╔╝██║██║     
   ╚██████╔╝██║ ╚═╝ ██║╚██████╗
    ╚══▀▀═╝ ╚═╝     ╚═╝ ╚═════╝
    
    QUASI-MONTE CARLO SAMPLER — SECUENCIAS DE BAJA DISCREPANCIA
═══════════════════════════════════════════════════════════════════════════════

DESCRIPCIÓN:
============
QMC utiliza secuencias de baja discrepancia (Sobol) para cubrir el espacio
de parámetros de forma EQUIDISTANTE y DETERMINISTA. No necesita scoring:
cada punto está matemáticamente posicionado para maximizar la distancia
entre todos los puntos muestreados.

FILOSOFÍA:
==========
Si tienes un espacio de 1 millón de combinaciones y solo 6.000 trials,
QMC se asegura de que esos 6.000 puntos estén PERFECTAMENTE distribuidos
por todo el mapa, sin dejar huecos ni acumular puntos en zonas.

VENTAJAS:
=========
  ✓ COBERTURA MÁXIMA del espacio de parámetros
  ✓ DETERMINISTA y REPRODUCIBLE (misma seed → mismos puntos)
  ✓ NO NECESITA SCORING — pura geometría matemática
  ✓ IDEAL PARA EXPLORACIÓN EXHAUSTIVA de todo el espacio
  ✓ ANTI-OVERFITTING por diseño (no persigue picos)
  ✓ COMPLEMENTO PERFECTO para luego refinar con CMA/TPE/GT

DIFERENCIAS CON OTROS SAMPLERS:
================================
  - CMA/TPE/GT: Usan el score para GUIAR la búsqueda → sesgo hacia picos
  - QMC: IGNORA el score completamente → cobertura uniforme pura
  - El score se retorna SOLO para que Optuna lo registre, pero NO influye
    en qué punto se muestrea a continuación

CUÁNDO USAR QMC:
================
  1. Exploración inicial: Antes de CMA/TPE para mapear el terreno
  2. Validación: Verificar que no hay regiones inexploradas
  3. Estudios de sensibilidad: Cobertura uniforme para ver qué importa
  4. Anti-overfitting extremo: Cero sesgo hacia resultados pasados

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import gc
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import optuna
import polars as pl
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import QMCSampler
from .storage import resolve_storage_for_strategy

if TYPE_CHECKING:
    pass

# =============================================================================
# IMPORTS INTERNOS
# =============================================================================
from modelox.core.engine import BacktestParams, calculate_performance_vectorized_numba
from modelox.core.metrics import resumen_metricas
from modelox.core.types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from modelox.core.exits import resolve_exit_settings_for_trial

# =============================================================================
# SILENCIAR WARNINGS
# =============================================================================
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
#  ██████╗ ███╗   ███╗ ██████╗    ███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
# ██╔═══██╗████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝ 
# ██║   ██║██╔████╔██║██║         ███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
# ██║▄▄ ██║██║╚██╔╝██║██║         ╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
# ╚██████╔╝██║ ╚═╝ ██║╚██████╗    ███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
#  ╚══▀▀═╝ ╚═╝     ╚═╝ ╚═════╝    ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
#                                                                                          
# SISTEMA DE SCORING QMC — PASIVO (NO INFLUYE EN LA BÚSQUEDA)
# =============================================================================


# =============================================================================
# [SECCIÓN 1] CONFIGURACIÓN DEL SCORING QMC
# =============================================================================

@dataclass
class QMCScoringConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │              CONFIGURACIÓN DEL SCORING QMC v1.0                         │
    │                                                                         │
    │  ARQUITECTURA: Score PASIVO — no influye en el muestreo               │
    │  FILOSOFÍA: QMC es puramente matemático (secuencias Sobol)             │
    │  El score se calcula SOLO para registro y comparación posterior        │
    │  RANGO SALIDA: [1, 1000]                                               │
    └────────────────────────────────────────────────────────────────────────┘
    
    NOTA IMPORTANTE:
    ================
    A diferencia de CMA/TPE/GT, el QMC Sampler NO usa el score para decidir
    qué punto muestrear a continuación. La secuencia Sobol está predeterminada
    matemáticamente. El score aquí es INFORMATIVO: se calcula para que al
    final puedas ver qué combinaciones fueron las mejores, pero NO altera
    la exploración en absoluto.
    """
    
    # =========================================================================
    # 1.1 RANGO DE SALIDA DEL SCORE (INFORMATIVO)
    # =========================================================================
    SCORE_MIN: float = 1.0               # MÍNIMO ABSOLUTO (NUNCA 0)
    SCORE_MAX: float = 1000.0            # MÁXIMO ABSOLUTO
    
    # =========================================================================
    # 1.2 PESOS PARA SCORE INFORMATIVO
    # =========================================================================
    # El score se calcula con promedio ponderado simple para REGISTRO.
    # Estos pesos NO afectan la búsqueda — solo el valor almacenado.
    WEIGHT_SHARPE: float = 0.30
    WEIGHT_SQN: float = 0.20
    WEIGHT_ROI: float = 0.20
    WEIGHT_DRAWDOWN: float = 0.15
    WEIGHT_TRADES: float = 0.15
    
    # =========================================================================
    # 1.3 UMBRALES MÍNIMOS (SOLO PARA REGISTRO)
    # =========================================================================
    MIN_TRADES_FOR_VALID: int = 5        # Muy permisivo — queremos registrar todo
    MAX_DRAWDOWN_LIMIT: float = 100.0    # Sin límite real — registrar todo
    MIN_TRADES_PER_DAY: float = 0.15     # Mínimo trades/día → debajo aplica penalización
    LOW_TRADES_PENALTY_DIVISOR: float = 5.0  # Penalización: score_final / 5


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
QMC_SCORING_CONFIG = QMCScoringConfig()


# =============================================================================
# [SECCIÓN 2] CLASE SCORER QMC
# =============================================================================

class QMCScorer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                     SCORER PASIVO QMC v1.0                              │
    │                                                                         │
    │  FILOSOFÍA: Score INFORMATIVO — no guía la búsqueda                    │
    │  ARQUITECTURA: Promedio ponderado simple de métricas normalizadas      │
    │  RANGO: [1, 1000]                                                      │
    │                                                                         │
    │  ⚠ ESTE SCORE NO INFLUYE EN EL MUESTREO QMC                           │
    │  La secuencia Sobol está predeterminada matemáticamente.               │
    │  El score solo sirve para identificar las mejores combinaciones        │
    │  DESPUÉS de que la exploración haya terminado.                         │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        study: Optional[optuna.Study] = None,
        config: Optional[QMCScoringConfig] = None,
    ):
        """
        INICIALIZA EL SCORER QMC.
        
        ARGS:
            study: OBJETO OPTUNA.STUDY (NO SE USA PARA GUIAR, SOLO REGISTRO)
            config: CONFIGURACIÓN PERSONALIZADA (USA DEFAULT SI NONE)
        """
        self.study = study
        self.config = config or QMC_SCORING_CONFIG
    
    # =========================================================================
    # [2.1] FUNCIONES AUXILIARES
    # =========================================================================
    
    @staticmethod
    def _safe_get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        """EXTRAE VALOR NUMÉRICO DE FORMA SEGURA."""
        try:
            val = metrics.get(key, default)
            if val is None:
                return default
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                return default
            return f_val
        except Exception:
            return default
    
    # =========================================================================
    # [2.2] NORMALIZACIÓN DE MÉTRICAS
    # =========================================================================
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normaliza Sharpe a [0, 1] con sigmoide suave."""
        try:
            return 1.0 / (1.0 + math.exp(-1.5 * (sharpe - 1.0)))
        except (OverflowError, ValueError):
            return 0.5
    
    def _normalize_sqn(self, sqn: float) -> float:
        """Normaliza SQN a [0, 1]. SQN > 4 = excelente (Van Tharp)."""
        return min(1.0, max(0.0, sqn / 4.0))
    
    def _normalize_roi(self, roi: float) -> float:
        """Normaliza ROI a [0, 1] con escala logarítmica."""
        if roi <= 0:
            return max(0.0, 0.5 + roi / 200.0)
        return min(1.0, 0.5 + 0.5 * math.log1p(roi) / math.log1p(100))
    
    def _penalize_drawdown(self, drawdown: float) -> float:
        """Penalización por drawdown [0, 1] (1 = sin DD, 0 = catastrófico)."""
        return max(0.0, 1.0 - drawdown / 100.0)
    
    def _normalize_trades(self, n_trades: int) -> float:
        """Normaliza cantidad de trades a [0, 1]."""
        if n_trades <= 0:
            return 0.0
        return min(1.0, math.log1p(n_trades) / math.log1p(100))
    
    # =========================================================================
    # [2.3] FUNCIÓN PRINCIPAL: compute_score
    # =========================================================================
    
    def compute_score(
        self,
        trial: Optional[optuna.Trial],
        metrics: Mapping[str, Any],
        returns: Optional[np.ndarray] = None,
        equity_curve: Optional[np.ndarray] = None,
        trades_pnl: Optional[np.ndarray] = None,
        total_days: Optional[float] = None,
        saldo_inicial: Optional[float] = None,
    ) -> float:
        """
        SCORING QMC (INFORMATIVO):
            - Penalización de frecuencia: si trades/día < 0.15 => score / 5
            - Componentes principales:
                * Profit Factor (más alto = mejor)
                * Win Rate (más alto = mejor)
                * Drawdown (más bajo = mejor)

        Fórmula:
            pf_norm = PF / (PF + 1)
            wr_norm = WR / 100
            dd_norm = 1 - DD/100
            score   = 1000 * pf_norm * wr_norm * dd_norm

        Notas:
            - dd_norm invierte la lógica del DD para que menor DD puntúe más.
            - El score es pasivo (QMC no lo usa para muestrear).
        """
        # SCORE = ROI de la estrategia (sin B&H)
        final_score = self._safe_get(metrics, "roi", 0.0)

        if not math.isfinite(final_score):
            final_score = 0.0

        if trial is not None:
            try:
                trial.set_user_attr('final_score', float(final_score))
            except Exception:
                pass

        return float(final_score)


# =============================================================================
#  ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗██████╗ 
# ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝██╔══██╗
# ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗  ██████╔╝
# ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝  ██╔══██╗
# ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗██║  ██║
#  ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
#
# OPTIMIZADOR QMC — QUASI-MONTE CARLO CON SECUENCIAS SOBOL
# =============================================================================


# =============================================================================
# [SECCIÓN 3] CONFIGURACIÓN DEL OPTIMIZADOR QMC
# =============================================================================

@dataclass
class QMCOptimizerConfig:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                  CONFIGURACIÓN DEL OPTIMIZADOR QMC                      │
    │                                                                         │
    │  QMC usa secuencias Sobol para distribución equidistante.              │
    │  No necesita n_startup_trials ni fase de calentamiento porque          │
    │  TODOS los puntos están predeterminados desde el inicio.               │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 3.1 CONFIGURACIÓN GENERAL OPTUNA
    # =========================================================================
    SEED: Optional[int] = None           # SEMILLA PARA REPRODUCIBILIDAD
                                          # (None = secuencia diferente cada vez)
    N_JOBS: int = 1                       # WORKERS PARALELOS
    CREATE_DATABASE: bool = True          # True = crear SQLite por estrategia (IDx.db)
    STUDY_NAME_PREFIX: str = "MODELOX"    # PREFIJO PARA NOMBRES DE ESTUDIO
    
    # =========================================================================
    # 3.2 CONFIGURACIÓN ESPECÍFICA QMC (SOBOL)
    # =========================================================================
    # QMCSampler usa un sampler base como fallback para parámetros
    # categóricos o condicionales que Sobol no puede manejar.
    # Por defecto se usa RandomSampler como independiente.
    # True: avisa si usas suggest_categorical (no compatible con QMC); usa solo suggest_float/suggest_int.
    WARN_INDEPENDENT_SAMPLING: bool = True
    WARN_ASYNCHRONOUS_SEEDING: bool = False
    
    # QMC tipo de secuencia: "sobol" (por defecto en Optuna)
    # Scramble: True mezcla la secuencia para mayor diversidad
    QMC_TYPE: str = "sobol"              # Tipo de secuencia QMC
    SCRAMBLE: bool = True                # Scramble de secuencia Sobol


# =============================================================================
# INSTANCIA DE CONFIGURACIÓN POR DEFECTO
# =============================================================================
QMC_OPTIMIZER_CONFIG = QMCOptimizerConfig()


# =============================================================================
# [SECCIÓN 4] CLASE OPTIMIZER QMC
# =============================================================================

class QMCOptimizer:
    """
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       OPTIMIZADOR QMC                                   │
    │                                                                         │
    │  QUASI-MONTE CARLO CON SECUENCIAS SOBOL                                │
    │                                                                         │
    │  CARACTERÍSTICAS:                                                       │
    │    ✓ COBERTURA EQUIDISTANTE del espacio de parámetros                  │
    │    ✓ DETERMINISTA (misma seed → mismos puntos)                         │
    │    ✓ NO USA EL SCORE para guiar la búsqueda                           │
    │    ✓ IDEAL para exploración exhaustiva sin sesgo                       │
    │    ✓ ANTI-OVERFITTING por diseño                                       │
    │                                                                         │
    │  FLUJO:                                                                 │
    │    1. Se genera la secuencia Sobol completa a priori                   │
    │    2. Cada trial mapea un punto Sobol al espacio de parámetros         │
    │    3. Se ejecuta backtest y se registra el score (INFORMATIVO)         │
    │    4. El score NO altera el siguiente punto (ya está decidido)         │
    └────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        n_trials: int,
        reporters: Sequence[Reporter] = (),
        optimizer_config: Optional[QMCOptimizerConfig] = None,
        scoring_config: Optional[QMCScoringConfig] = None,
        activo: Optional[str] = None,
    ):
        """
        INICIALIZA EL OPTIMIZADOR QMC.
        
        ARGS:
            config: CONFIGURACIÓN DE BACKTEST
            n_trials: NÚMERO DE TRIALS (PUNTOS SOBOL A MUESTREAR)
            reporters: LISTA DE REPORTERS PARA RESULTADOS
            optimizer_config: CONFIGURACIÓN DEL OPTIMIZADOR
            scoring_config: CONFIGURACIÓN DEL SCORING (INFORMATIVO)
            activo: NOMBRE DEL ACTIVO (OPCIONAL)
        """
        self.config = config
        self.n_trials = n_trials
        self.reporters = list(reporters)
        self.optimizer_config = optimizer_config or QMC_OPTIMIZER_CONFIG
        self.scoring_config = scoring_config or QMC_SCORING_CONFIG
        self.activo = activo
        
        # ESTADO INTERNO
        self._last_study: Optional[optuna.Study] = None
        self._scorer: Optional[QMCScorer] = None
    
    # =========================================================================
    # [4.1] CREAR ESTUDIO OPTUNA
    # =========================================================================
    
    def _create_study(self, strategy_name: str, strategy_id: Optional[int] = None) -> optuna.Study:
        """
        CREA UN ESTUDIO OPTUNA CON SAMPLER QMC (SOBOL).
        
        La secuencia Sobol garantiza que los puntos muestreados estén
        equidistantemente distribuidos en el hipercubo unitario, y
        Optuna los mapea al espacio de parámetros de la estrategia.
        
        RETURNS:
            OBJETO OPTUNA.STUDY CONFIGURADO CON QMC
        """
        cfg = self.optimizer_config
        
        # CONSTRUIR NOMBRE DEL ESTUDIO
        parts = [cfg.STUDY_NAME_PREFIX, str(strategy_name), "QMC"]
        if self.activo:
            parts.append(str(self.activo))
        study_name = self._slug("_".join(parts))
        
        # CREAR SAMPLER QMC (SOBOL)
        sampler = QMCSampler(
            seed=cfg.SEED,
            scramble=cfg.SCRAMBLE,
            warn_independent_sampling=cfg.WARN_INDEPENDENT_SAMPLING,
            warn_asynchronous_seeding=cfg.WARN_ASYNCHRONOUS_SEEDING,
        )
        
        storage = resolve_storage_for_strategy(
            create_database=bool(cfg.CREATE_DATABASE),
            strategy_id=strategy_id,
        )

        # CREAR ESTUDIO
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=False,
        )
        
        # INICIALIZAR SCORER (INFORMATIVO)
        self._scorer = QMCScorer(study=study, config=self.scoring_config)
        
        return study
    
    @staticmethod
    def _slug(s: str) -> str:
        """GENERA UN SLUG VÁLIDO PARA NOMBRES DE ESTUDIO."""
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')[:50]
    
    # =========================================================================
    # [4.2] PREPARAR PARÁMETROS
    # =========================================================================
    
    def _prepare_params(
        self,
        trial: optuna.Trial,
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """PREPARA PARÁMETROS PARA UN TRIAL."""
        params_puros = strategy.suggest_params(trial)
        params_rt = dict(params_puros)
        
        # INYECTAR VALORES DE CONFIGURACIÓN
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)

        
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        # RESOLVER CONFIGURACIÓN DE SALIDA
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # ALIASES PARA COMPATIBILIDAD
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # TIMEFRAMES
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        # FORZAR salidas SIEMPRE en 1m para máxima precisión (SL/TP/Trailing)
        exit_tf = "1m"
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    # =========================================================================
    # [4.3] FUNCIÓN OBJETIVO
    # =========================================================================
    
    def _create_objective(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> Callable[[optuna.Trial], float]:
        """
        CREA LA FUNCIÓN OBJETIVO PARA QMC.
        
        NOTA: El score retornado es INFORMATIVO. QMC no lo usa para
        decidir el siguiente punto — la secuencia Sobol ya lo decide.
        """
        
        # IMPORTAR COMPONENTES NECESARIOS
        from modelox.core.runner import SignalGenerator, BacktestEngine, periodic_cleanup
        
        def objective(trial: optuna.Trial) -> float:
            t0_total = time.perf_counter()
            
            # LIMPIEZA PERIÓDICA
            periodic_cleanup(trial.number)
            
            params_rt = self._prepare_params(trial, strategy, base_tf)
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            # GENERAR SEÑALES
            signals_df = SignalGenerator.generate_signals(df_entry, strategy, params_rt, df_map)
            
            # EJECUTAR BACKTEST
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_entry, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                return 0.0

            trial.set_user_attr("metricas", metrics)

            # ── SCORE = ROI de la estrategia (sin contar B&H) ───────────────
            if isinstance(trades_df, pl.DataFrame) and "is_buy_hold" in trades_df.columns:
                _strat_only = trades_df.filter(~pl.col("is_buy_hold"))
                if not _strat_only.is_empty():
                    _metrics_strat = resumen_metricas(
                        _strat_only,
                        saldo_inicial=float(self.config.saldo_inicial),
                    )
                    score = float(_metrics_strat.get("roi", 0.0))
                else:
                    score = 0.0
            else:
                # Sin B&H: ROI total = ROI de estrategia
                score = float(metrics.get("roi", 0.0))
            
            # CREAR ARTIFACTS
            artifacts = TrialArtifacts(
                strategy_name=strategy.name,
                trial_number=trial.number,
                params=params_rt,
                params_reporting=params_rt,
                score=score,
                metrics=metrics,
                df_signals=None,
                trades=trades_df.to_pandas(),
                equity_curve=equity_curve,
                indicators_used=params_rt.get("__indicators_used", []),
            )
            
            for reporter in self.reporters:
                reporter.on_trial_end(artifacts)
            
            return score
        
        return objective
    
    # =========================================================================
    # [4.4] OPTIMIZAR
    # =========================================================================
    
    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> optuna.Study:
        """
        ┌────────────────────────────────────────────────────────────────────┐
        │                  EJECUTAR EXPLORACIÓN QMC (SOBOL)                   │
        │                                                                     │
        │  Cada trial corresponde a un punto de la secuencia Sobol.          │
        │  Los puntos están EQUIDISTANTEMENTE distribuidos en el             │
        │  espacio de parámetros. No hay convergencia — hay COBERTURA.       │
        └────────────────────────────────────────────────────────────────────┘
        
        ARGS:
            df: DATAFRAME CON DATOS OHLCV
            strategy: ESTRATEGIA A EXPLORAR
            df_by_timeframe: DICT CON DATAFRAMES POR TIMEFRAME
            base_timeframe: TIMEFRAME BASE
        
        RETURNS:
            OBJETO OPTUNA.STUDY CON RESULTADOS
        """
        base_tf = base_timeframe or "1m"
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # CREAR ESTUDIO CON SAMPLER QMC
        study = self._create_study(
            strategy.name,
            strategy_id=getattr(strategy, "combinacion_id", None),
        )
        
        # CREAR OBJETIVO
        objective = self._create_objective(df_base, df_map, strategy, base_tf)
        
        # EJECUTAR EXPLORACIÓN
        study.optimize(
            objective,
            n_trials=int(self.n_trials),
            n_jobs=int(self.optimizer_config.N_JOBS),
            gc_after_trial=True,
            catch=(Exception,),
        )
        
        self._last_study = study
        return study
    
    # =========================================================================
    # [4.5] PROPIEDADES
    # =========================================================================
    
    @property
    def last_study(self) -> Optional[optuna.Study]:
        """RETORNA EL ÚLTIMO ESTUDIO EJECUTADO."""
        return self._last_study
    
    @property
    def scorer(self) -> Optional[QMCScorer]:
        """RETORNA EL SCORER UTILIZADO."""
        return self._scorer


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def _slug(s: str) -> str:
    """Genera un slug válido para nombres de estudio."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "study"


def create_qmc_study(
    strategy_name: str,
    strategy_id: Optional[int] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
    study_name_prefix: str = "MODELOX",
    create_database: bool = True,
    storage: Optional[str] = None,
    scramble: bool = True,
) -> optuna.Study:
    """
    Crea un estudio Optuna con sampler QMC (Quasi-Monte Carlo / Sobol).
    
    QMC (Quasi-Monte Carlo):
    - Usa secuencias Sobol de baja discrepancia
    - Cobertura EQUIDISTANTE del espacio de parámetros
    - NO usa el score para guiar — pura matemática
    - Ideal para exploración exhaustiva sin sesgo
    
    Args:
        strategy_name: Nombre de la estrategia
        activo: Nombre del activo (opcional)
        seed: Semilla aleatoria (None = diferente cada vez)
        study_name_prefix: Prefijo para el nombre del estudio
        create_database: True para usar SQLite por estrategia (IDx.db)
        storage: URI de almacenamiento (si se pasa, tiene prioridad)
        scramble: Scramble de secuencia Sobol (recomendado True)
    
    Returns:
        optuna.Study configurado con QMC Sampler
    """
    # Construir nombre del estudio
    parts = [study_name_prefix, str(strategy_name), "QMC"]
    if activo:
        parts.append(str(activo))
    study_name = _slug("_".join(parts))
    
    # Crear sampler QMC (Sobol)
    sampler = QMCSampler(
        seed=seed,
        scramble=scramble,
        warn_independent_sampling=False,
        warn_asynchronous_seeding=False,
    )
    
    if storage is None:
        storage = resolve_storage_for_strategy(
            create_database=create_database,
            strategy_id=strategy_id,
        )

    # Crear estudio
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
    )
    
    return study


def score_qmc(
    metrics: Mapping[str, Any],
    trial: Optional[optuna.Trial] = None,
    equity_curve: Optional[List[float]] = None,
    trades_pnl: Optional[np.ndarray] = None,
    total_days: Optional[float] = None,
    saldo_inicial: Optional[float] = None,
    trades_df: Optional[Any] = None,
) -> float:
    """
    SCORING QMC — Score = ROI de estrategia (sin B&H).

    Reglas:
      · Score = sum(pnl_neto trades estrategia) / saldo_inicial * 100
      · Si trades/día < 0.15: score / 5 (positivo) o score * 5 (negativo)
      · Límite inferior: 0 (nunca negativo)
    """
    import polars as _pl

    _saldo_ini = float(saldo_inicial) if saldo_inicial and saldo_inicial > 0 else 10_000.0
    _total_days = float(total_days) if total_days and total_days > 0 else 1.0

    # ── Filtrar trades de estrategia (excluir B&H) ───────────────────────
    if trades_df is not None and isinstance(trades_df, _pl.DataFrame) and "is_buy_hold" in trades_df.columns:
        _strat = trades_df.filter(~_pl.col("is_buy_hold"))
        _pnl_strat = float(_strat["pnl_neto"].sum()) if not _strat.is_empty() else 0.0
        _n_strat   = len(_strat)
    else:
        # Sin columna B&H: todos los trades son de estrategia
        _pnl_strat = float(np.sum(trades_pnl)) if trades_pnl is not None and len(trades_pnl) > 0 else 0.0
        _n_strat   = len(trades_pnl) if trades_pnl is not None else 0

    # ── Score base = ROI estrategia ──────────────────────────────────────
    score = (_pnl_strat / _saldo_ini * 100.0)

    # ── Límite inferior: nunca negativo ──────────────────────────────────
    score = max(0.0, score)

    if not math.isfinite(score):
        score = 0.0

    if trial is not None:
        try:
            trial.set_user_attr("score_qmc_roi",   round(_pnl_strat / _saldo_ini * 100.0, 4))
            trial.set_user_attr("score_qmc_final",  round(score, 4))
        except Exception:
            pass

    return float(score)


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    "QMCOptimizer",
    "QMCOptimizerConfig",
    "QMCScorer",
    "QMCScoringConfig",
    "QMC_SCORING_CONFIG",
    "QMC_OPTIMIZER_CONFIG",
    "create_qmc_study",
    "score_qmc",
]

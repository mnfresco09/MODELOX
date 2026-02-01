"""
# =============================================================================
#
#      ██████╗██╗   ██╗ ██████╗██╗     ██╗ ██████╗
#     ██╔════╝╚██╗ ██╔╝██╔════╝██║     ██║██╔════╝
#     ██║      ╚████╔╝ ██║     ██║     ██║██║     
#     ██║       ╚██╔╝  ██║     ██║     ██║██║     
#     ╚██████╗   ██║   ╚██████╗███████╗██║╚██████╗
#      ╚═════╝   ╚═╝    ╚═════╝╚══════╝╚═╝ ╚═════╝
#
#     CYCLIC_OPTIMIZER.PY - DESCENSO DE COORDENADAS CÍCLICO
#
# =============================================================================
#
#     ALGORITMO: Cyclic Coordinate Descent (CCD)
#     
#     CONCEPTO:
#     En lugar de optimizar todos los parámetros simultáneamente,
#     optimiza UN SOLO parámetro a la vez mientras mantiene los
#     demás fijos en sus mejores valores actuales.
#
#     FLUJO:
#     1. Ciclo N: Para cada parámetro P_i en {P1, P2, ..., Pk}:
#        - Fija todos los P_j (j ≠ i) en best_global
#        - Optimiza P_i libremente
#        - Actualiza best_global[P_i] con el mejor valor encontrado
#     2. Repite ciclos hasta CONVERGENCIA:
#        - Los parámetros ya no cambian significativamente
#        - O se alcanza max_cycles
#
#     IMPLEMENTACIÓN:
#     - PartialFixedSampler: Bloquea parámetros en valores fijos
#     - Ask-and-Tell Interface: Control granular trial por trial
#     - Criterio de convergencia automático
#
#     VENTAJAS:
#     - Encuentra interacciones entre parámetros
#     - Más robusto que optimización global en espacios grandes
#     - Interpretable: sabes qué parámetro está siendo optimizado
#
# =============================================================================
"""

from __future__ import annotations

import time
import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

try:
    import optuna
    from optuna.samplers import BaseSampler, TPESampler, RandomSampler, GridSampler
    from optuna.distributions import (
        BaseDistribution,
        FloatDistribution,
        IntDistribution,
        CategoricalDistribution,
    )
    from optuna.trial import TrialState
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from .scoring import score_optuna, set_study_for_scorer
from .metrics import resetear_estadisticas_globales
from .types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from .exits import (
    resolve_exit_settings_for_trial,
    DEFAULT_EXIT_SL_PCT_RANGE,
    DEFAULT_EXIT_TP_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE,
    DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Máximo de trials por ciclo para el grupo de exits (SL + TP o SL + Trailing)
EXIT_GROUP_MAX_TRIALS = 350

# Mínimo de trades por día para considerar un trial válido (con 100% de datos)
# Se ajusta proporcionalmente cuando se usa fracción de datos
MIN_TRADES_POR_DIA_BASE = 0.22

# Fracción de datos usada por ciclo (anti-overfitting)
DATA_FRACTION_PER_CYCLE = 0.25

@dataclass
class CyclicOptimizerConfig:
    """
    Configuración del Optimizador de Descenso de Coordenadas Cíclico.
    
    PARÁMETROS DE CICLO
    -------------------
    max_cycles: int
        Número máximo de ciclos completos (pasadas por todos los parámetros).
        Si convergence_check=True, puede terminar antes.
        Default: 15
    
    min_cycles: int
        Mínimo de ciclos garantizados antes de verificar convergencia.
        Default: 3
    
    CONVERGENCIA ADAPTATIVA POR PARÁMETRO
    -------------------------------------
    param_min_trials: int
        Mínimo de trials antes de evaluar convergencia del parámetro.
        Default: 20
    
    param_max_trials: int
        Máximo de trials por parámetro (seguridad).
        Default: 200
    
    param_patience: int
        Trials consecutivos sin mejora para considerar convergido.
        Default: 15
    
    param_min_improvement: float
        Mejora mínima para considerar progreso (evita ruido).
        Default: 0.001 (0.1%)
    
    CRITERIO DE CONVERGENCIA ENTRE CICLOS
    -------------------------------------
    convergence_check: bool
        Si True, termina cuando los parámetros no cambian entre ciclos.
        Default: True
    
    convergence_threshold: float
        Umbral de cambio relativo para considerar convergencia.
        Si todos los parámetros cambian menos que este %, se considera convergido.
        Default: 0.01 (1%)
    
    min_cycles: int
        Mínimo de ciclos antes de verificar convergencia.
        Garantiza al menos N pasadas completas.
        Default: 2
    
    SAMPLER INTERNO
    ---------------
    param_sampler: str
        Sampler para optimizar cada parámetro individual.
        - "tpe": TPESampler (recomendado, aprende de trials anteriores)
        - "random": RandomSampler (exploración pura)
        Default: "tpe"
    
    VISUALIZACIÓN
    -------------
    verbose: bool
        Mostrar progreso detallado con Rich.
        Default: True
    
    show_cycle_summary: bool
        Mostrar resumen al final de cada ciclo.
        Default: True
    
    PARÁMETROS DE SALIDA
    --------------------
    include_exit_params: bool
        Si True, incluye SL/TP/Trailing en la optimización cíclica.
        Solo aplica si optimize_exits=True en BacktestConfig.
        Default: True
    """
    # MODO DE OPERACIÓN
    use_n_trials: bool = False  # True = usa N_TRIALS, False = convergencia adaptativa
    n_trials_total: int = 10000  # Total de trials a usar (solo si use_n_trials=True)
    trials_per_param_fixed: Optional[int] = None  # None = auto, o forzar valor
    
    # Ciclos
    max_cycles: int = 15  # Suficientes para convergencia
    min_cycles: int = 3   # MÍNIMO 3 vueltas garantizadas
    
    # Convergencia entre ciclos (solo modo convergencia)
    convergence_check: bool = True
    convergence_threshold: float = 0.02  # 2% cambio entre ciclos
    
    # Convergencia ADAPTATIVA por parámetro (solo modo convergencia)
    param_min_trials: int = 20      # Mínimo antes de evaluar convergencia
    param_max_trials: int = 200     # Máximo por parámetro (seguridad)
    param_patience: int = 15        # Trials sin mejora = convergió
    param_min_improvement: float = 0.001  # 0.1% mejora mínima
    
    # MESETAS: Usar centroide de meseta en lugar de mejor valor exacto
    use_plateau_centroid: bool = True  # True = fija con centroide, False = valor exacto
    plateau_tolerance: float = 0.02    # 2% tolerancia para definir meseta (score similar)
    plateau_min_points: int = 5        # Mínimo puntos para considerar meseta válida
    
    # AGRUPAR EXITS: Optimizar SL/TP juntos (o SL/Trail juntos)
    group_exit_params: bool = True  # True = exits como bloque, False = uno a uno
    
    # Sampler
    param_sampler: str = "tpe"  # "tpe" o "random"
    
    # Visualización
    verbose: bool = True
    show_cycle_summary: bool = True
    
    # Parámetros de salida (SL/TP/Trailing)
    include_exit_params: bool = True  # Incluir exits en optimización cíclica
    
    # Semilla
    seed: Optional[int] = None


@dataclass
class PlateauInfo:
    """Información de una meseta detectada para un parámetro."""
    param_name: str
    centroid: float           # Centro de la meseta
    min_value: float          # Límite inferior
    max_value: float          # Límite superior
    n_points: int             # Puntos en la meseta
    mean_score: float         # Score promedio en la meseta
    best_score: float         # Mejor score en la meseta
    best_value: Any           # Valor del mejor score


@dataclass
class ParameterOptimizationResult:
    """Resultado de optimizar un solo parámetro."""
    param_name: str
    old_value: Any
    new_value: Any
    old_score: float
    new_score: float
    n_trials: int
    elapsed_time: float
    improved: bool
    relative_change: float
    plateau: Optional[PlateauInfo] = None  # Info de meseta si se detectó


@dataclass
class CycleResult:
    """Resultado de un ciclo completo."""
    cycle_number: int
    param_results: List[ParameterOptimizationResult]
    best_params_before: Dict[str, Any]
    best_params_after: Dict[str, Any]
    best_score_before: float
    best_score_after: float
    total_trials: int
    elapsed_time: float
    converged: bool


@dataclass
class CyclicOptimizationResult:
    """Resultado final de la optimización cíclica."""
    # Mejor resultado
    best_params: Dict[str, Any]
    best_score: float
    
    # Historial de ciclos
    cycles: List[CycleResult]
    total_cycles: int
    converged: bool
    convergence_cycle: Optional[int]
    
    # Estadísticas
    total_trials: int
    total_time: float
    params_trajectory: Dict[str, List[Any]]  # Historial de cada parámetro
    score_trajectory: List[float]
    
    # Estudios Optuna (para inspección avanzada)
    studies: List["optuna.Study"] = field(default_factory=list)


# =============================================================================
# PARTIAL FIXED SAMPLER
# =============================================================================

class PartialFixedSampler(BaseSampler):
    """
    Sampler que fija algunos parámetros y optimiza solo uno.
    
    Este sampler es el corazón del Descenso de Coordenadas:
    - Parámetros fijos: Siempre devuelven el valor especificado
    - Parámetro libre: Se optimiza con el sampler interno (TPE/Random)
    
    Uso:
    ```python
    fixed_params = {"rsi_period": 14, "ema_length": 200}
    free_param = "zlema_fast_len"
    
    sampler = PartialFixedSampler(
        fixed_params=fixed_params,
        free_param=free_param,
        internal_sampler=TPESampler(seed=42),
    )
    ```
    """
    
    def __init__(
        self,
        fixed_params: Dict[str, Any],
        free_param: str,
        internal_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            fixed_params: Dict con parámetros a mantener fijos y sus valores.
            free_param: Nombre del parámetro a optimizar.
            internal_sampler: Sampler para el parámetro libre (default: TPESampler).
            seed: Semilla para reproducibilidad.
        """
        self._fixed_params = fixed_params.copy()
        self._free_param = free_param
        self._seed = seed
        
        if internal_sampler is None:
            self._internal_sampler = TPESampler(
                seed=seed,
                multivariate=False,  # Optimizamos 1 param a la vez
            )
        else:
            self._internal_sampler = internal_sampler
    
    def infer_relative_search_space(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> Dict[str, BaseDistribution]:
        """Infiere el espacio de búsqueda relativo."""
        # Solo el parámetro libre necesita espacio de búsqueda
        return self._internal_sampler.infer_relative_search_space(study, trial)
    
    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Muestrea parámetros relativos."""
        # Delegar al sampler interno para el parámetro libre
        return self._internal_sampler.sample_relative(study, trial, search_space)
    
    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """
        Muestrea un parámetro independiente.
        
        - Si es un parámetro fijo: devuelve el valor fijo
        - Si es el parámetro libre: usa el sampler interno
        """
        # ¿Es un parámetro fijo?
        if param_name in self._fixed_params:
            fixed_value = self._fixed_params[param_name]
            
            # Validar que el valor fijo es válido para la distribución
            if isinstance(param_distribution, CategoricalDistribution):
                if fixed_value in param_distribution.choices:
                    return fixed_value
                else:
                    # Si el valor fijo no está en las opciones, usar el primero
                    warnings.warn(
                        f"Valor fijo {fixed_value} no válido para {param_name}. "
                        f"Usando {param_distribution.choices[0]}"
                    )
                    return param_distribution.choices[0]
            else:
                # Para Float/Int, asegurar que está en el rango
                low = getattr(param_distribution, 'low', None)
                high = getattr(param_distribution, 'high', None)
                
                if low is not None and high is not None:
                    # Clampear al rango válido
                    clamped = max(low, min(high, fixed_value))
                    if clamped != fixed_value:
                        warnings.warn(
                            f"Valor fijo {fixed_value} fuera de rango [{low}, {high}] "
                            f"para {param_name}. Usando {clamped}"
                        )
                    return clamped
                return fixed_value
        
        # Es el parámetro libre o uno desconocido: usar sampler interno
        return self._internal_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
    
    def reseed_rng(self) -> None:
        """Reinicializa el generador de números aleatorios."""
        self._internal_sampler.reseed_rng()
    
    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """Callback después de cada trial."""
        self._internal_sampler.after_trial(study, trial, state, values)


# =============================================================================
# CYCLIC COORDINATE DESCENT OPTIMIZER
# =============================================================================

@dataclass
class CyclicCoordinateOptimizer:
    """
    Optimizador de Descenso de Coordenadas Cíclico.
    
    ALGORITMO:
    1. Inicializa best_params con valores por defecto o aleatorios
    2. Para cada ciclo:
       - Para cada parámetro P:
         a) Fija todos los demás en best_params
         b) Optimiza solo P hasta convergencia (adaptativo)
         c) Actualiza best_params[P] si mejora
    3. Repite hasta convergencia o max_cycles
    
    EJEMPLO DE USO:
    ```python
    optimizer = CyclicCoordinateOptimizer(
        config=backtest_config,
        reporters=[rich_reporter],
        cyclic_config=CyclicOptimizerConfig(
            max_cycles=10,
            min_cycles=3,
            param_patience=15,  # Convergencia adaptativa
        ),
    )
    
    result = optimizer.optimize(
        df=df_ohlcv,
        strategy=my_strategy,
        df_by_timeframe={"1m": df_1m, "1h": df_1h},
        base_timeframe="1m",
    )
    
    print(f"Mejor score: {result.best_score}")
    print(f"Mejores params: {result.best_params}")
    print(f"Convergió en ciclo: {result.convergence_cycle}")
    ```
    """
    
    config: BacktestConfig
    reporters: Sequence[Reporter]
    cyclic_config: CyclicOptimizerConfig = field(default_factory=CyclicOptimizerConfig)
    activo: Optional[str] = None
    
    # Configuración de perturbación (opcional)
    perturbation_config: Optional[Any] = None
    
    # Estado interno
    _best_params: Dict[str, Any] = field(default_factory=dict)
    _best_score: float = field(default=-np.inf)
    _param_names: List[str] = field(default_factory=list)
    _param_distributions: Dict[str, BaseDistribution] = field(default_factory=dict)
    _studies: List["optuna.Study"] = field(default_factory=list)
    _total_trials: int = 0
    _exit_param_names: List[str] = field(default_factory=list)  # Parámetros de salida
    _strategy_param_names: List[str] = field(default_factory=list)  # Parámetros de estrategia
    _current_cycle_number: int = 0  # Número de ciclo actual para reportar
    
    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
    ) -> CyclicOptimizationResult:
        """
        Ejecuta la optimización de Descenso de Coordenadas Cíclico.
        
        Args:
            df: DataFrame con datos OHLCV del timeframe base.
            strategy: Estrategia a optimizar (debe tener suggest_params).
            df_by_timeframe: Dict de DataFrames por timeframe (para MTF).
            base_timeframe: Timeframe base (ej: "1m", "1h").
        
        Returns:
            CyclicOptimizationResult con todo el análisis.
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Optuna es requerido para CyclicCoordinateOptimizer")
        
        total_start = time.perf_counter()
        cfg = self.cyclic_config
        
        # Normalizar timeframe
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1h")
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        # 1. DESCUBRIR PARÁMETROS Y SUS DISTRIBUCIONES
        self._discover_parameters(strategy)
        
        if cfg.verbose:
            self._print_header(strategy.name)
        
        # 2. INICIALIZAR BEST_PARAMS (primera pasada exploratoria)
        self._initialize_best_params(df_base, df_map, strategy, base_tf)
        
        # 3. EJECUTAR CICLOS
        cycles: List[CycleResult] = []
        params_trajectory: Dict[str, List[Any]] = {p: [] for p in self._param_names}
        score_trajectory: List[float] = []
        converged = False
        convergence_cycle = None
        
        for cycle_num in range(1, cfg.max_cycles + 1):
            if cfg.verbose:
                self._print_cycle_header(cycle_num, cfg.max_cycles)
            
            cycle_result = self._run_cycle(
                cycle_number=cycle_num,
                df_base=df_base,
                df_map=df_map,
                strategy=strategy,
                base_tf=base_tf,
            )
            
            cycles.append(cycle_result)
            
            # Actualizar trayectorias
            for param in self._param_names:
                params_trajectory[param].append(self._best_params.get(param))
            score_trajectory.append(self._best_score)
            
            if cfg.show_cycle_summary:
                self._print_cycle_summary(cycle_result)
            
            # Verificar si debemos parar
            if cfg.use_n_trials:
                # MODO N_TRIALS: Parar solo cuando se acaben los trials
                if self._total_trials >= cfg.n_trials_total:
                    if cfg.verbose:
                        print(f"\n   ✅ Trials completados: {self._total_trials}/{cfg.n_trials_total}")
                    break
                # No parar por convergencia en modo N_TRIALS, seguir hasta acabar
            else:
                # MODO CONVERGENCIA: Parar cuando converge (después de min_cycles)
                if cfg.convergence_check and cycle_num >= cfg.min_cycles:
                    if cycle_result.converged:
                        converged = True
                        convergence_cycle = cycle_num
                        if cfg.verbose:
                            self._print_convergence_message(cycle_num)
                        break
        
        total_time = time.perf_counter() - total_start
        
        # 4. CONSTRUIR RESULTADO FINAL
        result = CyclicOptimizationResult(
            best_params=self._best_params.copy(),
            best_score=self._best_score,
            cycles=cycles,
            total_cycles=len(cycles),
            converged=converged,
            convergence_cycle=convergence_cycle,
            total_trials=self._total_trials,
            total_time=total_time,
            params_trajectory=params_trajectory,
            score_trajectory=score_trajectory,
            studies=self._studies,
        )
        
        if cfg.verbose:
            self._print_final_summary(result)
        
        return result
    
    def _discover_parameters(self, strategy: Strategy) -> None:
        """
        Descubre los parámetros de la estrategia usando un trial dummy.
        
        Incluye:
        1. Parámetros de la estrategia (suggest_params)
        2. Parámetros de salida (SL/TP/Trailing) si optimize_exits=True
        
        Es genérico para cualquier estrategia que implemente suggest_params.
        """
        # Clase para capturar las llamadas a suggest_*
        class ParamCaptureTrial:
            def __init__(self):
                self.params = {}
                self.distributions = {}
            
            def suggest_int(self, name, low, high, step=1, log=False):
                self.params[name] = (low + high) // 2  # Valor medio
                self.distributions[name] = IntDistribution(low, high, step=step, log=log)
                return self.params[name]
            
            def suggest_float(self, name, low, high, step=None, log=False):
                self.params[name] = (low + high) / 2  # Valor medio
                self.distributions[name] = FloatDistribution(low, high, step=step, log=log)
                return self.params[name]
            
            def suggest_categorical(self, name, choices):
                self.params[name] = choices[0]  # Primera opción
                self.distributions[name] = CategoricalDistribution(choices)
                return self.params[name]
        
        capture_trial = ParamCaptureTrial()
        
        # 1. DESCUBRIR PARÁMETROS DE LA ESTRATEGIA
        try:
            strategy.suggest_params(capture_trial)
        except Exception as e:
            # Algunos parámetros pueden depender de otros, ignorar errores
            pass
        
        self._param_names = list(capture_trial.params.keys())
        self._param_distributions = capture_trial.distributions.copy()
        self._best_params = capture_trial.params.copy()
        
        # 2. AÑADIR PARÁMETROS DE SALIDA (SL/TP/TRAILING) SI ESTÁ HABILITADO
        optimize_exits = bool(getattr(self.config, "optimize_exits", False))
        include_exits = self.cyclic_config.include_exit_params
        
        if optimize_exits and include_exits:
            exit_type = str(getattr(self.config, "exit_type", "pnl_fixed")).lower()
            
            # SL siempre se optimiza
            sl_rng = tuple(getattr(self.config, "exit_sl_pct_range", DEFAULT_EXIT_SL_PCT_RANGE))
            self._param_names.append("exit_sl_pct")
            self._param_distributions["exit_sl_pct"] = FloatDistribution(
                sl_rng[0], sl_rng[1], step=sl_rng[2]
            )
            self._best_params["exit_sl_pct"] = (sl_rng[0] + sl_rng[1]) / 2
            
            # TP solo si es pnl_fixed o all
            if exit_type in {"pnl_fixed", "all"}:
                tp_rng = tuple(getattr(self.config, "exit_tp_pct_range", DEFAULT_EXIT_TP_PCT_RANGE))
                self._param_names.append("exit_tp_pct")
                self._param_distributions["exit_tp_pct"] = FloatDistribution(
                    tp_rng[0], tp_rng[1], step=tp_rng[2]
                )
                self._best_params["exit_tp_pct"] = (tp_rng[0] + tp_rng[1]) / 2
            
            # Trailing solo si es pnl_trailing, percent_trailing o all
            if exit_type in {"pnl_trailing", "percent_trailing", "all"}:
                act_rng = tuple(getattr(self.config, "exit_trail_act_pct_range", DEFAULT_EXIT_TRAIL_ACT_PCT_RANGE))
                dist_rng = tuple(getattr(self.config, "exit_trail_dist_pct_range", DEFAULT_EXIT_TRAIL_DIST_PCT_RANGE))
                
                self._param_names.append("exit_trail_act_pct")
                self._param_distributions["exit_trail_act_pct"] = FloatDistribution(
                    act_rng[0], act_rng[1], step=act_rng[2]
                )
                self._best_params["exit_trail_act_pct"] = (act_rng[0] + act_rng[1]) / 2
                
                self._param_names.append("exit_trail_dist_pct")
                self._param_distributions["exit_trail_dist_pct"] = FloatDistribution(
                    dist_rng[0], dist_rng[1], step=dist_rng[2]
                )
                self._best_params["exit_trail_dist_pct"] = (dist_rng[0] + dist_rng[1]) / 2
        
        # Guardar qué parámetros son de exit para tratarlos especialmente
        self._exit_param_names = [p for p in self._param_names if p.startswith("exit_")]
        self._strategy_param_names = [p for p in self._param_names if not p.startswith("exit_")]
        
        if self.cyclic_config.verbose:
            print(f"\n   📋 Parámetros descubiertos: {len(self._param_names)}")
            
            if self._strategy_param_names:
                print(f"\n      🎯 Estrategia ({len(self._strategy_param_names)}):") 
                for name in self._strategy_param_names:
                    dist = self._param_distributions.get(name)
                    if isinstance(dist, IntDistribution):
                        print(f"         • {name}: int[{dist.low}, {dist.high}] step={dist.step}")
                    elif isinstance(dist, FloatDistribution):
                        print(f"         • {name}: float[{dist.low:.4f}, {dist.high:.4f}] step={dist.step}")
                    elif isinstance(dist, CategoricalDistribution):
                        print(f"         • {name}: categorical{list(dist.choices)}")
            
            if self._exit_param_names:
                print(f"\n      🚪 Salidas ({len(self._exit_param_names)}):")
                for name in self._exit_param_names:
                    dist = self._param_distributions.get(name)
                    if isinstance(dist, FloatDistribution):
                        print(f"         • {name}: float[{dist.low:.2f}%, {dist.high:.2f}%] step={dist.step}%")
    
    def _initialize_best_params(
        self,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> None:
        """
        Inicializa best_params con una pequeña búsqueda exploratoria.
        
        Esto da un punto de partida razonable antes de empezar
        la optimización coordenada.
        """
        cfg = self.cyclic_config
        
        if cfg.verbose:
            print("\n   🎯 Inicializando con búsqueda exploratoria...")
        
        # Crear estudio para exploración inicial
        sampler = RandomSampler(seed=cfg.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        set_study_for_scorer(study)
        
        # Trials iniciales para exploración
        n_init_trials = 300
        
        objective = self._create_objective(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            fixed_params={},  # Sin parámetros fijos
            free_param=None,  # Todos libres
            cycle_number=0,  # Exploración inicial = ciclo 0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=n_init_trials,
                n_jobs=1,
                show_progress_bar=False,
                catch=(Exception,),
            )
        
        self._total_trials += n_init_trials
        
        # ════════════════════════════════════════════════════════════════
        # FILTRO DE CALIDAD: trades_por_dia >= umbral
        # Si el mejor trial no cumple, buscar el mejor que sí cumpla
        # En fase exploratoria usamos 100% de datos, así que umbral completo
        # ════════════════════════════════════════════════════════════════
        data_fraction = 1.0  # Exploración usa 100% de datos
        MIN_TRADES_POR_DIA = MIN_TRADES_POR_DIA_BASE * data_fraction
        
        # Buscar el mejor trial que cumpla el filtro de trades/día
        valid_trials = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if trial.value is None:
                continue
            
            metricas = trial.user_attrs.get("metricas", {})
            trades_por_dia = metricas.get("trades_por_dia", 0.0) or 0.0
            
            if trades_por_dia >= MIN_TRADES_POR_DIA:
                valid_trials.append(trial)
        
        # Ordenar por score descendente
        valid_trials.sort(key=lambda t: t.value or 0, reverse=True)
        
        best_trial = None
        if valid_trials:
            best_trial = valid_trials[0]
            if cfg.verbose and study.best_trial and study.best_trial != best_trial:
                orig_tpd = study.best_trial.user_attrs.get("metricas", {}).get("trades_por_dia", 0)
                print(f"   ⚠️ Mejor trial original tenía trades/día={orig_tpd:.3f} < {MIN_TRADES_POR_DIA}")
                print(f"      Usando siguiente mejor con trades/día >= {MIN_TRADES_POR_DIA}")
        elif study.best_trial:
            # Si ninguno cumple, usar el mejor original con advertencia
            best_trial = study.best_trial
            if cfg.verbose:
                tpd = study.best_trial.user_attrs.get("metricas", {}).get("trades_por_dia", 0)
                print(f"   ⚠️ Ningún trial cumple trades/día >= {MIN_TRADES_POR_DIA}")
                print(f"      Usando mejor disponible (trades/día={tpd:.3f})")
        
        if best_trial:
            # GUARDAR TODOS los params del mejor trial como valores base
            # Los params incluyen tanto estrategia como exits sugeridos
            best_params_from_trial = best_trial.params.copy()
            
            # Asegurar que todos los params descubiertos tienen valor
            # (en caso de que alguno no se haya sugerido en el mejor trial)
            for pname in self._param_names:
                if pname in best_params_from_trial:
                    self._best_params[pname] = best_params_from_trial[pname]
                # Si no está en el trial, mantener el valor default de _best_params
            
            self._best_score = best_trial.value
            
            if cfg.verbose:
                metricas = best_trial.user_attrs.get("metricas", {})
                tpd = metricas.get("trades_por_dia", 0.0) or 0.0
                print(f"   ✅ Score inicial: {self._best_score:.4f} (trades/día: {tpd:.3f})")
                # Mostrar params iniciales
                for pname in self._param_names:
                    pval = self._best_params.get(pname, "???")
                    print(f"      • {pname}: {pval}")
        
        self._studies.append(study)
    
    def _run_cycle(
        self,
        cycle_number: int,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> CycleResult:
        """
        Ejecuta un ciclo completo de optimización coordenada.
        
        Itera sobre cada parámetro (o grupo de parámetros si group_exit_params=True),
        optimizándolo mientras mantiene los demás fijos.
        
        ANTI-OVERFITTING: Cada ciclo usa un 25% contiguo aleatorio de los datos.
        """
        cfg = self.cyclic_config
        cycle_start = time.perf_counter()
        
        # Guardar número de ciclo actual para que _create_objective lo use
        self._current_cycle_number = cycle_number
        
        # ════════════════════════════════════════════════════════════════
        # SELECCIÓN ALEATORIA DE DATOS PARA ESTE CICLO (anti-overfitting)
        # ════════════════════════════════════════════════════════════════
        data_fraction = DATA_FRACTION_PER_CYCLE
        total_rows = len(df_base)
        window_size = int(total_rows * data_fraction)
        
        # Elegir inicio aleatorio (puede variar en cada ciclo)
        rng = np.random.default_rng(seed=(cfg.seed or 42) + cycle_number * 1000)
        max_start = total_rows - window_size
        start_idx = rng.integers(0, max_start + 1)
        end_idx = start_idx + window_size
        
        # Crear subconjunto de datos para este ciclo
        df_base_cycle = df_base.slice(start_idx, window_size)
        df_map_cycle = {}
        for tf, df_tf in df_map.items():
            # Proporcionar el mismo rango temporal para cada timeframe
            tf_rows = len(df_tf)
            tf_ratio = tf_rows / total_rows
            tf_start = int(start_idx * tf_ratio)
            tf_window = int(window_size * tf_ratio)
            df_map_cycle[tf] = df_tf.slice(tf_start, tf_window)
        
        if cfg.verbose:
            pct_start = (start_idx / total_rows) * 100
            pct_end = (end_idx / total_rows) * 100
            print(f"\n      📊 Datos ciclo {cycle_number}: {pct_start:.0f}%-{pct_end:.0f}% ({window_size:,} filas)")
        
        best_before = self._best_params.copy()
        score_before = self._best_score
        
        param_results: List[ParameterOptimizationResult] = []
        cycle_trials = 0
        
        # Obtener grupos de parámetros a optimizar
        param_groups = self._get_param_groups()
        
        for group in param_groups:
            if len(group) == 1:
                # Parámetro individual
                param_name = group[0]
                if cfg.verbose:
                    print(f"\n      🔧 Optimizando: {param_name}")
                
                result = self._optimize_single_param(
                    param_name=param_name,
                    df_base=df_base_cycle,  # Usar datos del ciclo
                    df_map=df_map_cycle,    # Usar datos del ciclo
                    strategy=strategy,
                    base_tf=base_tf,
                    data_fraction=data_fraction,  # Para ajustar filtros
                )
                
                param_results.append(result)
                cycle_trials += result.n_trials
                
                if cfg.verbose:
                    change_str = f"{result.relative_change*100:+.2f}%" if result.relative_change != 0 else "sin cambio"
                    status = "✅" if result.improved else "➖"
                    print(f"         {status} {result.old_value} → {result.new_value} ({change_str})")
                    print(f"         Score: {result.old_score:.4f} → {result.new_score:.4f}")
            else:
                # Grupo de parámetros (exits agrupados)
                if cfg.verbose:
                    print(f"\n      🔧 Optimizando GRUPO: {', '.join(group)}")
                
                result = self._optimize_param_group(
                    param_names=group,
                    df_base=df_base_cycle,  # Usar datos del ciclo
                    df_map=df_map_cycle,    # Usar datos del ciclo
                    strategy=strategy,
                    base_tf=base_tf,
                )
                
                param_results.append(result)
                cycle_trials += result.n_trials
                
                if cfg.verbose:
                    status = "✅" if result.improved else "➖"
                    print(f"         {status} Score: {result.old_score:.4f} → {result.new_score:.4f}")
                    for pname in group:
                        old_v = best_before.get(pname)
                        new_v = self._best_params.get(pname)
                        print(f"            • {pname}: {old_v} → {new_v}")
        
        # Verificar convergencia
        converged = self._check_convergence(best_before, self._best_params)
        
        cycle_time = time.perf_counter() - cycle_start
        
        return CycleResult(
            cycle_number=cycle_number,
            param_results=param_results,
            best_params_before=best_before,
            best_params_after=self._best_params.copy(),
            best_score_before=score_before,
            best_score_after=self._best_score,
            total_trials=cycle_trials,
            elapsed_time=cycle_time,
            converged=converged,
        )
    
    def _get_param_groups(self) -> List[List[str]]:
        """
        Agrupa parámetros para optimización.
        
        ORDEN: Exits PRIMERO, luego estrategia
        
        Si group_exit_params=True, los exits van juntos:
        - [exit_sl_pct, exit_tp_pct] para SL+TP
        - [exit_sl_pct, exit_trail_act_pct, exit_trail_dist_pct] para trailing
        
        Los parámetros de estrategia van uno a uno.
        """
        cfg = self.cyclic_config
        
        # Nombres de parámetros de exit
        exit_params = {"exit_sl_pct", "exit_tp_pct", "exit_trail_act_pct", "exit_trail_dist_pct"}
        
        # Separar estrategia y exits
        strategy_params = [p for p in self._param_names if p not in exit_params]
        current_exit_params = [p for p in self._param_names if p in exit_params]
        
        groups: List[List[str]] = []
        
        # ═══ EXITS PRIMERO ═══
        # Parámetros de exit: agrupados o uno a uno
        if current_exit_params:
            if cfg.group_exit_params:
                # Agrupar todos los exits juntos
                groups.append(current_exit_params)
            else:
                # Uno a uno
                for p in current_exit_params:
                    groups.append([p])
        
        # ═══ ESTRATEGIA DESPUÉS ═══
        # Parámetros de estrategia: uno a uno
        for p in strategy_params:
            groups.append([p])
        
        return groups
    
    def _get_trials_per_param(self, n_params: int = 1) -> int:
        """
        Calcula los trials por parámetro (o grupo) en modo N_TRIALS.
        
        Si trials_per_param_fixed está definido, lo usa.
        Si no, calcula automáticamente: n_trials_total / (num_params * ciclos_estimados)
        
        Args:
            n_params: Número de parámetros en el grupo (para dar más trials a grupos)
        """
        cfg = self.cyclic_config
        
        # Si hay valor fijo, usarlo (multiplicado por n_params si es grupo)
        if cfg.trials_per_param_fixed is not None:
            return cfg.trials_per_param_fixed * n_params
        
        # Calcular automáticamente
        num_groups = len(self._get_param_groups()) if self._param_names else 1
        ciclos_estimados = (cfg.min_cycles + cfg.max_cycles) // 2  # Promedio
        
        trials_per_group = cfg.n_trials_total // (num_groups * ciclos_estimados)
        
        # Dar más trials a grupos con más parámetros
        trials = trials_per_group * max(1, n_params // 2 + 1)
        
        # Mínimo 10 trials por grupo
        return max(10, trials)
    
    def _optimize_param_group(
        self,
        param_names: List[str],
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> ParameterOptimizationResult:
        """
        Optimiza un GRUPO de parámetros juntos (ej: SL + TP).
        
        Todos los parámetros del grupo varían libremente mientras
        los demás (estrategia) están fijos.
        """
        cfg = self.cyclic_config
        start_time = time.perf_counter()
        
        old_values = {p: self._best_params.get(p) for p in param_names}
        
        # ════════════════════════════════════════════════════════════════
        # SCORE INTERNO: Recalcular con datos actuales
        # SCORE RICH: Empezará de 0 (estudio nuevo)
        # ════════════════════════════════════════════════════════════════
        old_score = self._evaluate_params(
            params=self._best_params,
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
        )
        # NO actualizar self._best_score aquí - se actualiza solo si mejora
        
        # Crear sampler: parámetros fuera del grupo están fijos
        fixed_params = {k: v for k, v in self._best_params.items() if k not in param_names}
        
        if cfg.param_sampler.lower() == "random":
            internal_sampler = RandomSampler(seed=cfg.seed)
        else:
            # Para grupos, usar TPE multivariado para capturar correlaciones
            internal_sampler = TPESampler(seed=cfg.seed, multivariate=True)
        
        # PartialFixedSampler con múltiples parámetros libres
        sampler = PartialFixedSampler(
            fixed_params=fixed_params,
            free_param=None,  # Múltiples libres
            internal_sampler=internal_sampler,
            seed=cfg.seed,
        )
        
        # Crear estudio NUEVO para este grupo
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"cyclic_group_{'_'.join(param_names)}",
        )
        # RESETEAR ESTADÍSTICAS GLOBALES para que TRIAL=0 y BEST=0 en rich
        resetear_estadisticas_globales()
        set_study_for_scorer(study)
        
        # Crear objetivo - PASAR EL GRUPO COMPLETO
        objective = self._create_objective(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            fixed_params=fixed_params,
            free_param=None,  # No es un solo param
            free_params_group=param_names,  # GRUPO de params libres
            cycle_number=self._current_cycle_number,  # Ciclo actual
        )
        
        # Calcular trials (más para grupos)
        n_trials = self._get_trials_per_param(n_params=len(param_names))
        
        # LIMITAR TRIALS PARA GRUPO DE EXITS
        exit_param_names = {"exit_sl_pct", "exit_tp_pct", "exit_trail_act_pct", "exit_trail_dist_pct"}
        is_exit_group = all(p in exit_param_names for p in param_names)
        if is_exit_group:
            n_trials = min(n_trials, EXIT_GROUP_MAX_TRIALS)
        
        actual_trials = 0
        
        if cfg.use_n_trials:
            # MODO N_TRIALS
            if cfg.verbose:
                print(f"         [DEBUG] Grupo exits: ejecutando {n_trials} trials...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=1,
                    show_progress_bar=False,
                    # catch=(Exception,),  # Desactivado para ver errores
                )
            actual_trials = n_trials
        else:
            # MODO CONVERGENCIA
            best_score_so_far = old_score
            trials_without_improvement = 0
            
            # Limitar también en modo convergencia para exits
            max_trials_group = cfg.param_max_trials * len(param_names)
            if is_exit_group:
                max_trials_group = min(max_trials_group, EXIT_GROUP_MAX_TRIALS)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                while actual_trials < max_trials_group:
                    study.optimize(
                        objective,
                        n_trials=1,
                        n_jobs=1,
                        show_progress_bar=False,
                        catch=(Exception,),
                    )
                    actual_trials += 1
                    
                    current_best = study.best_value if study.best_trial else old_score
                    
                    if current_best > best_score_so_far + cfg.param_min_improvement:
                        best_score_so_far = current_best
                        trials_without_improvement = 0
                    else:
                        trials_without_improvement += 1
                    
                    # Más paciencia para grupos
                    patience = cfg.param_patience * len(param_names)
                    min_trials = cfg.param_min_trials * len(param_names)
                    
                    if actual_trials >= min_trials:
                        if trials_without_improvement >= patience:
                            break
        
        self._total_trials += actual_trials
        self._studies.append(study)
        
        elapsed = time.perf_counter() - start_time
        
        # Analizar resultado
        new_values = old_values.copy()
        new_score = old_score
        improved = False
        
        if study.best_trial and study.best_trial.value > old_score:
            for pname in param_names:
                new_values[pname] = study.best_trial.params.get(pname, old_values[pname])
                self._best_params[pname] = new_values[pname]
            new_score = study.best_trial.value
            self._best_score = new_score
            improved = True
        
        # Calcular cambio relativo promedio
        relative_change = 0.0
        count = 0
        for pname in param_names:
            old_v = old_values.get(pname)
            new_v = new_values.get(pname)
            if old_v is not None and new_v is not None:
                if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
                    if old_v != 0:
                        relative_change += abs(new_v - old_v) / abs(old_v)
                        count += 1
        if count > 0:
            relative_change /= count
        
        return ParameterOptimizationResult(
            param_name=f"GROUP:{'+'.join(param_names)}",  # Nombre compuesto
            old_value=old_values,
            new_value=new_values,
            old_score=old_score,
            new_score=new_score,
            n_trials=actual_trials,
            elapsed_time=elapsed,
            improved=improved,
            relative_change=relative_change,
            plateau=None,  # No detectamos mesetas para grupos (más complejo)
        )
    
    def _optimize_single_param(
        self,
        param_name: str,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
        data_fraction: float = 1.0,
    ) -> ParameterOptimizationResult:
        """
        Optimiza un solo parámetro.
        
        DOS MODOS:
        - use_n_trials=True: Usa número fijo de trials (trials_per_param calculado)
        - use_n_trials=False: Convergencia adaptativa (para cuando no mejora)
        
        Usa PartialFixedSampler para bloquear los parámetros fijos.
        
        Args:
            data_fraction: Fracción de datos usada (para ajustar filtros de calidad)
        """
        cfg = self.cyclic_config
        start_time = time.perf_counter()
        
        old_value = self._best_params.get(param_name)
        
        # ════════════════════════════════════════════════════════════════
        # SCORE INTERNO: Recalcular con datos actuales
        # SCORE RICH: Empezará de 0 (estudio nuevo)
        # ════════════════════════════════════════════════════════════════
        old_score = self._evaluate_params(
            params=self._best_params,
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
        )
        # NO actualizar self._best_score aquí - se actualiza solo si mejora
        
        # Parámetros fijos (todos menos el que optimizamos)
        fixed_params = {k: v for k, v in self._best_params.items() if k != param_name}
        
        # ========================================
        # GENERAR GRID DE VALORES PARA 100% COBERTURA
        # ========================================
        dist = self._param_distributions.get(param_name)
        grid_values = []
        
        if dist is not None:
            if isinstance(dist, IntDistribution):
                step = dist.step if dist.step else 1
                grid_values = list(range(dist.low, dist.high + 1, step))
            elif isinstance(dist, FloatDistribution):
                step = dist.step if dist.step else (dist.high - dist.low) / 100
                val = dist.low
                while val <= dist.high + 1e-9:  # Tolerancia para floats
                    grid_values.append(round(val, 6))
                    val += step
        
        if not grid_values:
            # Fallback: 50 valores uniformes
            grid_values = list(np.linspace(dist.low if dist else 0, dist.high if dist else 100, 50))
        
        # Crear estudio NUEVO para este parámetro
        study = optuna.create_study(
            direction="maximize",
            study_name=f"cyclic_param_{param_name}",
        )
        # RESETEAR ESTADÍSTICAS GLOBALES para que TRIAL=0 y BEST=0 en rich
        resetear_estadisticas_globales()
        set_study_for_scorer(study)
        
        # ENCOLAR TODOS LOS VALORES EN ORDEN SECUENCIAL
        # Esto garantiza: 50, 51, 52, 53, 54...
        for val in grid_values:
            study.enqueue_trial({param_name: val})
        
        # Crear objetivo
        objective = self._create_objective(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            fixed_params=fixed_params,
            free_param=param_name,
            cycle_number=self._current_cycle_number,  # Ciclo actual
        )
        
        actual_trials = 0
        
        # Número de trials = tamaño del grid (100% cobertura garantizada)
        trials_for_param = len(grid_values)
        
        # ========================================
        # EJECUTAR EN ORDEN SECUENCIAL
        # ========================================
        if cfg.verbose:
            print(f"         [DEBUG] Secuencial {param_name}: {grid_values[0]} → {grid_values[-1]} ({trials_for_param} valores)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=trials_for_param,
                n_jobs=1,
                show_progress_bar=False,
                # catch=(Exception,),  # Desactivado para ver errores
            )
        actual_trials = trials_for_param
        
        self._total_trials += actual_trials
        self._studies.append(study)
        
        elapsed = time.perf_counter() - start_time
        
        # ════════════════════════════════════════════════════════════════
        # DETECTAR PARÁMETRO ÓPTIMO: PERCENTIL 80 DE SQN
        # ════════════════════════════════════════════════════════════════
        # Filtrar por ROI > 0 y trades/día >= umbral
        # Tomar top 20% por SQN y calcular MEDIANA
        # Si no hay suficientes trials válidos, usar mejor trial
        # ════════════════════════════════════════════════════════════════
        
        plateau_info = None
        new_value = old_value
        new_score = old_score
        improved = False
        
        # Detectar parámetro óptimo via percentil 80 (siempre intentamos)
        if cfg.use_plateau_centroid:
            plateau_info = self._detect_plateau(study, param_name, data_fraction)
        
        if plateau_info and plateau_info.n_points >= cfg.plateau_min_points:
            # ✅ PERCENTIL 80 VÁLIDO - usar mediana del top 20%
            new_value = plateau_info.centroid
            new_score = plateau_info.mean_score  # Score promedio del top
            improved = (new_score > old_score)
            
            # Actualizar best global
            self._best_params[param_name] = new_value
            if new_score > self._best_score:
                self._best_score = new_score
            
            if cfg.verbose:
                print(f"      � Top 20% SQN: {plateau_info.n_points} trials en rango [{plateau_info.min_value:.4f} - {plateau_info.max_value:.4f}]")
                print(f"         MEDIANA: {plateau_info.centroid:.4f} (valor óptimo robusto)")
                print(f"         SQN promedio top: {plateau_info.mean_score:.4f}")
        
        elif not cfg.use_plateau_centroid and study.best_trial and study.best_trial.value > old_score:
            # Percentil 80 deshabilitado - usar mejor valor directo
            new_value = study.best_trial.params.get(param_name, old_value)
            new_score = study.best_trial.value
            improved = True
            self._best_params[param_name] = new_value
            self._best_score = new_score
            
            if cfg.verbose:
                print(f"      ⚠️ Percentil 80 deshabilitado - usando mejor valor: {new_value}")
        
        else:
            # ❌ NO hay suficientes trials válidos - usar mejor trial como fallback
            if study.best_trial and study.best_trial.value > old_score:
                new_value = study.best_trial.params.get(param_name, old_value)
                new_score = study.best_trial.value
                improved = True
                self._best_params[param_name] = new_value
                self._best_score = new_score
                
                if cfg.verbose:
                    print(f"      ⚠️ Pocos trials válidos para percentil 80 - usando mejor trial: {new_value}")
                    print(f"         (score: {new_score:.4f})")
            else:
                # Mantener valor anterior
                if cfg.verbose:
                    best_trial_val = study.best_trial.params.get(param_name) if study.best_trial else None
                    best_trial_score = study.best_trial.value if study.best_trial else None
                    print(f"      ⚠️ Sin mejora - manteniendo valor: {old_value}")
                    if best_trial_val is not None:
                        print(f"         (mejor trial: {best_trial_val} con score {best_trial_score:.4f})")
        
        # Calcular cambio relativo
        relative_change = 0.0
        if old_value is not None and new_value is not None:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if old_value != 0:
                    relative_change = abs(new_value - old_value) / abs(old_value)
                elif new_value != 0:
                    relative_change = 1.0
        
        return ParameterOptimizationResult(
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            old_score=old_score,
            new_score=new_score,
            n_trials=actual_trials,
            elapsed_time=elapsed,
            improved=improved,
            relative_change=relative_change,
            plateau=plateau_info,
        )
    
    def _detect_plateau(
        self,
        study: "optuna.Study",
        param_name: str,
        data_fraction: float = 1.0,
    ) -> Optional[PlateauInfo]:
        """
        Detecta el valor óptimo del parámetro usando percentil 80 de SQN.
        
        ALGORITMO ROBUSTO ANTI-OVERFITTING:
        1. Filtrar trials con ROI > 0 (solo rentables)
        2. Filtrar trials con trades_por_dia >= umbral (ajustado por data_fraction)
        3. Ordenar por SQN y tomar percentil 80 (top 20% mejor)
        4. Calcular MEDIANA del parámetro de ese top
        
        Args:
            study: Estudio de Optuna con los trials
            param_name: Nombre del parámetro a analizar
            data_fraction: Fracción de datos usada (para ajustar umbral trades/día)
        
        Returns:
            PlateauInfo con la mediana y estadísticas, o None si no hay suficientes datos.
        """
        cfg = self.cyclic_config
        
        # Umbral de trades/día ajustado por fracción de datos
        # Con 25% de datos, el umbral se reduce proporcionalmente
        min_trades_por_dia = MIN_TRADES_POR_DIA_BASE * data_fraction
        
        # ════════════════════════════════════════════════════════════════
        # 1. RECOPILAR DATOS CON FILTROS DE CALIDAD
        # ════════════════════════════════════════════════════════════════
        trials_data = []  # (param_val, sqn, roi, trades_por_dia, score)
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if trial.value is None:
                continue
            
            param_val = trial.params.get(param_name)
            if param_val is None or not isinstance(param_val, (int, float)):
                continue
            
            # Obtener métricas del trial
            metricas = trial.user_attrs.get("metricas", {})
            sqn = metricas.get("sqn", 0.0) or 0.0
            roi = metricas.get("roi", 0.0) or 0.0
            trades_por_dia = metricas.get("trades_por_dia", 0.0) or 0.0
            
            # ════════════════════════════════════════════════════════════
            # FILTRO 1: ROI > 0 (solo rentables)
            # ════════════════════════════════════════════════════════════
            if roi <= 0:
                continue
            
            # ════════════════════════════════════════════════════════════
            # FILTRO 2: trades_por_dia >= umbral (ajustado por data_fraction)
            # Con 25% de datos: 0.25 * 0.25 = 0.0625 trades/día mínimo
            # ════════════════════════════════════════════════════════════
            if trades_por_dia < min_trades_por_dia:
                continue
            
            trials_data.append((param_val, sqn, roi, trades_por_dia, trial.value))
        
        # Necesitamos mínimo de puntos para calcular percentil
        min_points_for_percentile = max(cfg.plateau_min_points, 5)
        if len(trials_data) < min_points_for_percentile:
            return None
        
        # ════════════════════════════════════════════════════════════════
        # 2. ORDENAR POR SQN Y TOMAR PERCENTIL 80 (top 20%)
        # ════════════════════════════════════════════════════════════════
        # Ordenar por SQN descendente (mejores primero)
        trials_data.sort(key=lambda x: x[1], reverse=True)
        
        # Calcular cuántos trials tomar (percentil 80 = top 20%)
        n_top = max(1, int(len(trials_data) * 0.20))
        top_trials = trials_data[:n_top]
        
        # ════════════════════════════════════════════════════════════════
        # 3. CALCULAR MEDIANA DEL PARÁMETRO DEL TOP 20%
        # ════════════════════════════════════════════════════════════════
        top_params = [t[0] for t in top_trials]
        top_sqns = [t[1] for t in top_trials]
        top_scores = [t[4] for t in top_trials]
        
        # Mediana (más robusta que media)
        sorted_params = sorted(top_params)
        n = len(sorted_params)
        if n % 2 == 0:
            centroid = (sorted_params[n // 2 - 1] + sorted_params[n // 2]) / 2
        else:
            centroid = sorted_params[n // 2]
        
        min_val = min(top_params)
        max_val = max(top_params)
        mean_score = sum(top_scores) / len(top_scores)
        mean_sqn = sum(top_sqns) / len(top_sqns)
        
        # Mejor punto del top (por SQN - ya está ordenado)
        best_value = top_trials[0][0]
        best_plateau_score = top_trials[0][4]
        
        return PlateauInfo(
            param_name=param_name,
            centroid=centroid,
            min_value=min_val,
            max_value=max_val,
            n_points=len(top_trials),
            mean_score=mean_score,
            best_score=best_plateau_score,
            best_value=best_value,
        )
    
    def _check_convergence(
        self,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
    ) -> bool:
        """
        Verifica si los parámetros han convergido.
        
        Convergencia = todos los parámetros numéricos cambiaron
        menos que convergence_threshold.
        """
        threshold = self.cyclic_config.convergence_threshold
        
        for param_name in self._param_names:
            old_val = old_params.get(param_name)
            new_val = new_params.get(param_name)
            
            if old_val is None or new_val is None:
                continue
            
            # Para categóricos, verificar igualdad
            if not isinstance(old_val, (int, float)):
                if old_val != new_val:
                    return False
                continue
            
            # Para numéricos, verificar cambio relativo
            if old_val == 0 and new_val == 0:
                continue
            elif old_val == 0:
                relative_change = 1.0
            else:
                relative_change = abs(new_val - old_val) / abs(old_val)
            
            if relative_change > threshold:
                return False
        
        return True
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
    ) -> float:
        """
        Evalúa un conjunto de parámetros y devuelve el score.
        
        Se usa para recalcular el score base antes de optimizar cada parámetro.
        """
        from .runner import SignalGenerator, BacktestEngine, apply_perturbation
        from .metrics import resumen_metricas
        
        try:
            # Parámetros de exit
            exit_param_names = {"exit_sl_pct", "exit_tp_pct", "exit_trail_act_pct", "exit_trail_dist_pct"}
            
            # Preparar params runtime
            params_rt = dict(params)
            params_rt["__activo"] = self.activo
            params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
            params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
            params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
            params_rt["__comision_pct"] = float(self.config.comision_pct)
            params_rt["__comision_sides"] = int(self.config.comision_sides)
            params_rt["__saldo_usado"] = float(self.config.saldo_usado)
            params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
            params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
            
            # Timeframes
            entry_tf = getattr(strategy, "timeframe_entry", None) or base_tf
            exit_tf = getattr(strategy, "timeframe_exit", None) or base_tf
            params_rt["__timeframe_base"] = base_tf
            params_rt["__timeframe_entry"] = normalize_timeframe_to_suffix(entry_tf)
            params_rt["__timeframe_exit"] = normalize_timeframe_to_suffix(exit_tf)
            
            # Configuración de salidas
            exit_type = str(getattr(self.config, "exit_type", "pnl_fixed")).lower()
            sl_pct = float(params.get("exit_sl_pct", getattr(self.config, "exit_sl_pct", 2.0)))
            tp_pct = float(params.get("exit_tp_pct", getattr(self.config, "exit_tp_pct", 4.0)))
            trail_act_pct = float(params.get("exit_trail_act_pct", getattr(self.config, "exit_trail_act_pct", 1.5)))
            trail_dist_pct = float(params.get("exit_trail_dist_pct", getattr(self.config, "exit_trail_dist_pct", 0.5)))
            
            params_rt["__exit_type"] = exit_type
            params_rt["__exit_sl_pct"] = sl_pct
            params_rt["__exit_tp_pct"] = tp_pct
            params_rt["__exit_trail_act_pct"] = trail_act_pct
            params_rt["__exit_trail_dist_pct"] = trail_dist_pct
            
            # Generar señales
            df_entry = df_map.get(params_rt["__timeframe_entry"], df_base)
            df_entry_perturbed = apply_perturbation(df_entry, self.config, is_backtest=True)
            generator = SignalGenerator(strategy)
            df_signals = generator.generate(df_entry_perturbed, params_rt)
            
            if df_signals is None or df_signals.is_empty():
                return -999.0
            
            # Backtest
            df_exit = df_map.get(params_rt["__timeframe_exit"], df_base)
            df_exit_perturbed = apply_perturbation(df_exit, self.config, is_backtest=True)
            engine = BacktestEngine(self.config)
            trades_df, equity, stats = engine.run(
                signals_df=df_signals,
                ohlcv_exit_df=df_exit_perturbed,
                params=params_rt,
            )
            
            if trades_df is None or trades_df.is_empty():
                return -999.0
            
            # Calcular score
            metricas = resumen_metricas(trades_df, equity, self.config)
            return float(score_optuna(metricas, params_rt, df_base))
            
        except Exception:
            return -999.0
    
    def _create_objective(
        self,
        *,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
        fixed_params: Dict[str, Any],
        free_param: Optional[str],
        free_params_group: Optional[List[str]] = None,  # Para grupos de parámetros
        cycle_number: int = 0,  # Número de ciclo actual (0 = exploración inicial)
    ) -> Callable[["optuna.Trial"], float]:
        """
        Crea función objetivo para Optuna.
        
        Maneja tanto parámetros de estrategia como de salida (SL/TP/Trailing).
        
        Args:
            fixed_params: Parámetros que deben mantenerse fijos
            free_param: Parámetro único que varía (para optimización individual)
            free_params_group: Lista de parámetros que varían (para optimización de grupo)
        """
        from .runner import SignalGenerator, BacktestEngine, apply_perturbation
        from .metrics import resumen_metricas
        
        # Determinar qué parámetros son libres
        # CASO ESPECIAL: Si no se especifica nada, TODOS son libres (exploración inicial)
        all_params_free = (free_param is None and free_params_group is None)
        
        if free_params_group:
            free_params_set = set(free_params_group)
        elif free_param:
            free_params_set = {free_param}
        elif all_params_free:
            # Todos los parámetros son libres (exploración inicial)
            free_params_set = set(self._param_names)
        else:
            free_params_set = set()  # Todos fijos
        
        # Parámetros de exit
        exit_param_names = {"exit_sl_pct", "exit_tp_pct", "exit_trail_act_pct", "exit_trail_dist_pct"}
        
        # Distribuciones de exit
        exit_distributions = {
            name: dist for name, dist in self._param_distributions.items()
            if name.startswith("exit_")
        }
        
        def objective(trial: "optuna.trial.Trial") -> float:
            try:
                # 1. OBTENER PARÁMETROS DE ESTRATEGIA
                # Determinar si estamos optimizando exits o estrategia
                strategy_free_params = [p for p in free_params_set if p not in exit_param_names]
                exit_free_params = [p for p in free_params_set if p in exit_param_names]
                
                params_puros = {}
                
                # CASO 0: TODOS los parámetros son libres (exploración inicial)
                if all_params_free:
                    # Usar suggest_params para todos los de estrategia
                    params_puros = strategy.suggest_params(trial)
                    # Los exits se manejan abajo
                
                # CASO 1: Solo EXITS varían (estrategia 100% fija)
                elif not strategy_free_params:
                    # Copiar TODOS los parámetros de estrategia desde fixed_params o _best_params
                    for pname in self._param_names:
                        if pname not in exit_param_names:
                            if pname in fixed_params:
                                params_puros[pname] = fixed_params[pname]
                            elif pname in self._best_params:
                                params_puros[pname] = self._best_params[pname]
                
                # CASO 2: Un parámetro de estrategia varía (GridSampler)
                elif len(strategy_free_params) == 1:
                    free_strategy_param = strategy_free_params[0]
                    dist = self._param_distributions.get(free_strategy_param)
                    
                    # Copiar params fijos de estrategia
                    for pname in self._param_names:
                        if pname not in exit_param_names and pname != free_strategy_param:
                            if pname in fixed_params:
                                params_puros[pname] = fixed_params[pname]
                            elif pname in self._best_params:
                                params_puros[pname] = self._best_params[pname]
                    
                    # Obtener el valor del GridSampler para el parámetro libre
                    if dist is not None:
                        if isinstance(dist, IntDistribution):
                            params_puros[free_strategy_param] = trial.suggest_int(
                                free_strategy_param, dist.low, dist.high, step=dist.step
                            )
                        elif isinstance(dist, FloatDistribution):
                            params_puros[free_strategy_param] = trial.suggest_float(
                                free_strategy_param, dist.low, dist.high, step=dist.step
                            )
                
                # CASO 3: Múltiples params de estrategia (grupo - usa suggest_params)
                else:
                    params_puros = strategy.suggest_params(trial)
                    # Forzar valores fijos
                    for pname, pval in fixed_params.items():
                        if pname not in exit_param_names:
                            params_puros[pname] = pval
                
                # 2. PARÁMETROS DE EXIT
                for exit_param, dist in exit_distributions.items():
                    if exit_param in free_params_set:
                        # Este exit VARÍA - sugerir valor
                        if isinstance(dist, FloatDistribution):
                            params_puros[exit_param] = trial.suggest_float(
                                exit_param,
                                dist.low,
                                dist.high,
                                step=dist.step,
                            )
                        elif isinstance(dist, IntDistribution):
                            params_puros[exit_param] = trial.suggest_int(
                                exit_param,
                                dist.low,
                                dist.high,
                                step=dist.step,
                            )
                    else:
                        # Este exit está FIJO - usar valor guardado
                        if exit_param in fixed_params:
                            params_puros[exit_param] = fixed_params[exit_param]
                        elif exit_param in self._best_params:
                            params_puros[exit_param] = self._best_params[exit_param]
                        else:
                            # Default: punto medio
                            params_puros[exit_param] = (dist.low + dist.high) / 2
                
                # 3. PREPARAR PARÁMETROS RUNTIME
                params_rt = dict(params_puros)
                params_rt["__activo"] = self.activo
                params_rt["__cycle_number"] = cycle_number  # Número de ciclo para Excel/CSV
                params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
                params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
                params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
                params_rt["__comision_pct"] = float(self.config.comision_pct)
                params_rt["__comision_sides"] = int(self.config.comision_sides)
                params_rt["__saldo_usado"] = float(self.config.saldo_usado)
                params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
                params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
                
                # Timeframes
                entry_tf = getattr(strategy, "timeframe_entry", None) or base_tf
                exit_tf = getattr(strategy, "timeframe_exit", None) or base_tf
                params_rt["__timeframe_base"] = base_tf
                params_rt["__timeframe_entry"] = normalize_timeframe_to_suffix(entry_tf)
                params_rt["__timeframe_exit"] = normalize_timeframe_to_suffix(exit_tf)
                
                # 4. CONFIGURACIÓN DE SALIDAS
                exit_type = str(getattr(self.config, "exit_type", "pnl_fixed")).lower()
                
                # Usar valores sugeridos (que vienen del PartialFixedSampler)
                sl_pct = float(params_puros.get("exit_sl_pct", getattr(self.config, "exit_sl_pct", 2.0)))
                tp_pct = float(params_puros.get("exit_tp_pct", getattr(self.config, "exit_tp_pct", 4.0)))
                trail_act_pct = float(params_puros.get("exit_trail_act_pct", getattr(self.config, "exit_trail_act_pct", 1.5)))
                trail_dist_pct = float(params_puros.get("exit_trail_dist_pct", getattr(self.config, "exit_trail_dist_pct", 0.5)))
                
                params_rt["__exit_type"] = exit_type
                params_rt["__exit_sl_pct"] = sl_pct
                params_rt["__exit_tp_pct"] = tp_pct
                params_rt["__exit_trail_act_pct"] = trail_act_pct
                params_rt["__exit_trail_dist_pct"] = trail_dist_pct
                
                # Aliases para compatibilidad
                params_rt["exit_type"] = exit_type
                params_rt["exit_sl_pct"] = sl_pct
                params_rt["exit_tp_pct"] = tp_pct
                params_rt["exit_trail_act_pct"] = trail_act_pct
                params_rt["exit_trail_dist_pct"] = trail_dist_pct
                
                # Seleccionar DataFrame
                df_entry = df_map.get(params_rt["__timeframe_entry"], df_base)
                
                # Perturbación (si está configurada)
                df_trial = df_entry
                if self.perturbation_config and getattr(self.perturbation_config, 'enabled', False):
                    df_trial, _, _ = apply_perturbation(
                        df_entry, self.perturbation_config, trial.number
                    )
                
                # Generar señales
                signals_df = SignalGenerator.generate_signals(df_trial, strategy, params_rt, df_map)
                
                # Backtest - usa la firma correcta
                trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                    df_trial, signals_df, self.config, params_rt, strategy,
                )
                
                if trades_df.is_empty():
                    return 0.0
                
                # ══════════════════════════════════════════════════════════════
                # GUARDAR MÉTRICAS EN EL TRIAL (para filtros de calidad)
                # ══════════════════════════════════════════════════════════════
                trial.set_user_attr("metricas", metrics)
                
                # Calcular score
                score = float(score_optuna(metrics))
                
                # Crear artifacts para reporters
                artifacts = TrialArtifacts(
                    strategy_name=strategy.name,
                    trial_number=trial.number,
                    params=params_rt,
                    params_reporting=params_rt,
                    score=score,
                    metrics=metrics,
                    df_signals=None,
                    trades=trades_df.to_pandas() if not trades_df.is_empty() else None,
                    equity_curve=equity_curve,
                    indicators_used=params_rt.get("__indicators_used", []),
                    perturbado=self.perturbation_config.enabled if self.perturbation_config else False,
                    perturb_seed=None,
                    neighborhood_result=None,
                )
                
                for reporter in self.reporters:
                    try:
                        reporter.on_trial_end(artifacts)
                    except Exception:
                        pass
                
                return score
                
            except Exception as e:
                # Trial fallido
                return float('-inf')
        
        return objective
    
    # =========================================================================
    # VISUALIZACIÓN
    # =========================================================================
    
    def _print_header(self, strategy_name: str) -> None:
        """Imprime header del optimizador."""
        cfg = self.cyclic_config
        print("\n" + "=" * 70)
        print("   🔄 CYCLIC COORDINATE DESCENT OPTIMIZER")
        print("=" * 70)
        print(f"\n   📈 Estrategia: {strategy_name}")
        print(f"   🔁 Ciclos: {cfg.min_cycles} min → {cfg.max_cycles} max")
        
        if cfg.use_n_trials:
            # MODO N_TRIALS
            trials_per = self._get_trials_per_param()
            print(f"   ⚡ Modo: N_TRIALS ({cfg.n_trials_total} trials total)")
            print(f"   🎯 Trials por parámetro: ~{trials_per}")
            print("   ⚠️  NO para por convergencia, usa TODOS los trials")
        else:
            # MODO CONVERGENCIA
            print("   ⚡ Modo: CONVERGENCIA ADAPTATIVA")
            print(f"   🎯 Convergencia por param: {cfg.param_patience} trials sin mejora")
            print(f"   📊 Convergencia entre ciclos: {cfg.convergence_threshold*100:.1f}%")
    
    def _print_cycle_header(self, cycle_num: int, max_cycles: int) -> None:
        """Imprime header de un ciclo."""
        print("\n" + "-" * 50)
        print(f"   🔄 CICLO {cycle_num}/{max_cycles}")
        print("-" * 50)
    
    def _print_cycle_summary(self, result: CycleResult) -> None:
        """Imprime resumen de un ciclo."""
        improved_count = sum(1 for r in result.param_results if r.improved)
        print(f"\n   📊 Resumen Ciclo {result.cycle_number}:")
        print(f"      • Parámetros mejorados: {improved_count}/{len(result.param_results)}")
        print(f"      • Score: {result.best_score_before:.4f} → {result.best_score_after:.4f}")
        print(f"      • Tiempo: {result.elapsed_time:.1f}s")
        print(f"      • Trials: {result.total_trials}")
    
    def _print_convergence_message(self, cycle_num: int) -> None:
        """Imprime mensaje de convergencia."""
        print("\n" + "=" * 50)
        print(f"   ✅ ¡CONVERGENCIA ALCANZADA EN CICLO {cycle_num}!")
        print("=" * 50)
    
    def _print_final_summary(self, result: CyclicOptimizationResult) -> None:
        """Imprime resumen final."""
        print("\n" + "=" * 70)
        print("   📊 RESUMEN FINAL - CYCLIC COORDINATE DESCENT")
        print("=" * 70)
        print(f"\n   🏆 Mejor Score: {result.best_score:.4f}")
        print(f"   🔁 Ciclos completados: {result.total_cycles}")
        print(f"   ✅ Convergió: {'Sí' if result.converged else 'No'}")
        if result.convergence_cycle:
            print(f"   📍 Ciclo de convergencia: {result.convergence_cycle}")
        print(f"   🎯 Total trials: {result.total_trials}")
        print(f"   ⏱️  Tiempo total: {result.total_time:.1f}s")
        
        print("\n   📋 Mejores parámetros:")
        for param, value in result.best_params.items():
            if isinstance(value, float):
                print(f"      • {param}: {value:.4f}")
            else:
                print(f"      • {param}: {value}")
        
        # Mostrar evolución del score
        if result.score_trajectory:
            print("\n   📈 Evolución del Score:")
            for i, score in enumerate(result.score_trajectory, 1):
                bar_len = int((score / max(result.score_trajectory)) * 30)
                bar = "█" * bar_len
                print(f"      Ciclo {i}: {score:.4f} {bar}")


# =============================================================================
# FUNCIÓN DE CONVENIENCIA
# =============================================================================

def run_cyclic_optimization(
    *,
    df: pl.DataFrame,
    strategy: Strategy,
    backtest_config: BacktestConfig,
    reporters: Sequence[Reporter],
    cyclic_config: Optional[CyclicOptimizerConfig] = None,
    df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
    base_timeframe: Optional[str] = None,
    activo: Optional[str] = None,
    perturbation_config: Optional[Any] = None,
) -> CyclicOptimizationResult:
    """
    Ejecuta optimización de Descenso de Coordenadas Cíclico.
    
    Función de conveniencia que crea y ejecuta el optimizador.
    
    Args:
        df: DataFrame OHLCV del timeframe base.
        strategy: Estrategia a optimizar.
        backtest_config: Configuración de backtest.
        reporters: Lista de reporters para visualización.
        cyclic_config: Configuración del optimizador cíclico.
        df_by_timeframe: Dict de DataFrames por timeframe.
        base_timeframe: Timeframe base.
        activo: Nombre del activo (BTC, GOLD, etc.).
        perturbation_config: Configuración de perturbación.
    
    Returns:
        CyclicOptimizationResult con el resultado completo.
    """
    if cyclic_config is None:
        cyclic_config = CyclicOptimizerConfig()
    
    optimizer = CyclicCoordinateOptimizer(
        config=backtest_config,
        reporters=reporters,
        cyclic_config=cyclic_config,
        activo=activo,
        perturbation_config=perturbation_config,
    )
    
    return optimizer.optimize(
        df=df,
        strategy=strategy,
        df_by_timeframe=df_by_timeframe,
        base_timeframe=base_timeframe,
    )


# =============================================================================
# INTEGRACIÓN CON CONFIGURACION.PY
# =============================================================================

def get_cyclic_config_from_settings() -> CyclicOptimizerConfig:
    """
    Crea configuración de CyclicOptimizer desde configuracion.py.
    
    Lee las variables CYCLIC_* de configuracion.py si existen.
    Soporta dos modos:
    - use_n_trials=True: Usa N_TRIALS hasta acabarlos
    - use_n_trials=False: Convergencia adaptativa
    """
    try:
        from general.configuracion import (
            N_TRIALS,
            CYCLIC_USE_N_TRIALS,
            CYCLIC_MAX_CYCLES,
            CYCLIC_MIN_CYCLES,
            CYCLIC_CONVERGENCE_THRESHOLD,
            CYCLIC_PARAM_MIN_TRIALS,
            CYCLIC_PARAM_MAX_TRIALS,
            CYCLIC_PARAM_PATIENCE,
            CYCLIC_PARAM_MIN_IMPROVEMENT,
            CYCLIC_TRIALS_PER_PARAM_FIXED,
            CYCLIC_USE_PLATEAU,
            CYCLIC_PLATEAU_TOLERANCE,
            CYCLIC_PLATEAU_MIN_POINTS,
            CYCLIC_GROUP_EXITS,
            CYCLIC_PARAM_SAMPLER,
            CYCLIC_VERBOSE,
            CYCLIC_INCLUDE_EXITS,
        )
        
        return CyclicOptimizerConfig(
            use_n_trials=CYCLIC_USE_N_TRIALS,
            n_trials_total=N_TRIALS,
            trials_per_param_fixed=CYCLIC_TRIALS_PER_PARAM_FIXED,
            max_cycles=CYCLIC_MAX_CYCLES,
            min_cycles=CYCLIC_MIN_CYCLES,
            convergence_threshold=CYCLIC_CONVERGENCE_THRESHOLD,
            param_min_trials=CYCLIC_PARAM_MIN_TRIALS,
            param_max_trials=CYCLIC_PARAM_MAX_TRIALS,
            param_patience=CYCLIC_PARAM_PATIENCE,
            param_min_improvement=CYCLIC_PARAM_MIN_IMPROVEMENT,
            use_plateau_centroid=CYCLIC_USE_PLATEAU,
            plateau_tolerance=CYCLIC_PLATEAU_TOLERANCE,
            plateau_min_points=CYCLIC_PLATEAU_MIN_POINTS,
            group_exit_params=CYCLIC_GROUP_EXITS,
            param_sampler=CYCLIC_PARAM_SAMPLER,
            verbose=CYCLIC_VERBOSE,
            include_exit_params=CYCLIC_INCLUDE_EXITS,
        )
    except ImportError:
        # Si no existen las variables, usar defaults
        return CyclicOptimizerConfig()

"""
# =============================================================================
#
#     ██████╗ ██╗      █████╗ ████████╗███████╗ █████╗ ██╗   ██╗
#     ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔══██╗██║   ██║
#     ██████╔╝██║     ███████║   ██║   █████╗  ███████║██║   ██║
#     ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██╔══██║██║   ██║
#     ██║     ███████╗██║  ██║   ██║   ███████╗██║  ██║╚██████╔╝
#     ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝
#
#     PLATEAU_OPTIMIZER.PY - OPTIMIZADOR EN 3 FASES
#
# =============================================================================
#
#     FASE 1: EXPLORACIÓN (40% de trials)
#     - RandomSampler o QMC para llenar el espacio de parámetros
#     - Genera materia prima para clustering
#
#     FASE 2: DETECCIÓN DE MESETAS
#     - HDBSCAN sobre los trials de Fase 1
#     - Encuentra clusters (mesetas) de parámetros buenos
#     - Descarta picos aislados (overfitting)
#
#     FASE 3: REFINAMIENTO LOCAL (CMA-ES)
#     - Para cada meseta, CMA-ES dentro de sus límites
#     - Afina la solución en la zona segura
#
#     RESULTADO: "Mejor centroide refinado" (solución robusta)
#
# =============================================================================
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import polars as pl

try:
    import optuna
    from optuna.samplers import RandomSampler, TPESampler, CmaEsSampler, QMCSampler
    from optuna.exceptions import ExperimentalWarning
    _OPTUNA_AVAILABLE = True
    _QMC_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    _QMC_AVAILABLE = False

try:
    from optuna.samplers import QMCSampler
    _QMC_AVAILABLE = True
except ImportError:
    _QMC_AVAILABLE = False

from .topology import (
    PlateauConfig,
    PlateauResult,
    TopologyAnalysis,
    analyze_topology,
    print_topology_report,
)
from .scoring import (
    score_optuna,
    set_study_for_scorer,
)
from .types import (
    BacktestConfig,
    Reporter,
    Strategy,
    TrialArtifacts,
    normalize_timeframe_to_suffix,
)
from .exits import resolve_exit_settings_for_trial

# Silenciar warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)


# =============================================================================
# CONFIGURACIÓN DEL SISTEMA DE MESETAS
# =============================================================================

@dataclass
class PlateauOptimizerConfig:
    """
    Configuración del optimizador de mesetas de 3 fases.
    
    FASE 1: EXPLORACIÓN
    -------------------
    exploration_ratio: Porcentaje de trials para exploración (0.0 - 1.0)
                       RECOMENDADO: 0.40 (40% exploración, 60% para refinamiento)
    
    exploration_sampler: Sampler para fase de exploración
                         - "random": RandomSampler (aleatorio puro)
                         - "qmc": QMCSampler (RECOMENDADO) - Secuencia Sobol
                                  Cobertura sistemática del espacio
                                  Más trials = mayor resolución
                         - "tpe": TPESampler (algo de guía, pero greedy)
    
    FASE 2: CLUSTERING / FILTRADO
    -----------------------------
    plateau_config: Configuración de PlateauConfig para DBSCAN/HDBSCAN
    
    auto_tune_dbscan: Si True, ajusta eps automáticamente según los datos
    
    FILTRADO ANTES DEL CLUSTERING (AUTOMÁTICO):
        Se aplican DOS filtros secuenciales:
        
        1. FILTRO ROI: Descarta trials con ROI < 0%
           → Elimina todas las estrategias perdedoras
        
        2. FILTRO MEDIA (μ): Descarta trials con score < media
           → Elimina la mitad inferior de la distribución
           
        Ejemplo: 2500 trials → Filtro ROI → 1800 → Filtro Media → ~900 para clustering
    
    FASE 3: REFINAMIENTO
    --------------------
    min_trials_per_plateau: Mínimo de trials por meseta (si hay muchas mesetas)
                            RECOMENDADO: 50-100
    
    max_plateaus_to_refine: Límite máximo de mesetas (0 = sin límite)
                            RECOMENDADO: 0 (refinar todas)
    
    GENERAL
    -------
    verbose: Mostrar progreso detallado
    
    save_intermediate: Guardar resultados intermedios (para debugging)
    """
    # Fase 1
    exploration_ratio: float = 0.50  # 50% exploración, 50% refinamiento
    exploration_sampler: str = "qmc"  # "qmc" (recomendado), "random" o "tpe"
    
    # Fase 2 - Filtrado por MEDIA por defecto
    plateau_config: PlateauConfig = field(default_factory=PlateauConfig)
    auto_tune_dbscan: bool = True
    
    # Fase 3 - Distribución proporcional
    min_trials_per_plateau: int = 50   # Mínimo por meseta
    max_plateaus_to_refine: int = 0    # 0 = sin límite (refinar todas)
    
    # General
    verbose: bool = True
    save_intermediate: bool = False
    
    def __post_init__(self):
        # Los filtros de Fase 2 son automáticos (ROI >= 0 y Score >= media)
        pass


@dataclass
class PhaseResult:
    """Resultado de una fase de optimización."""
    phase_name: str
    n_trials: int
    best_score: float
    best_params: Dict[str, Any]
    best_trial_number: int
    elapsed_time: float
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlateauOptimizationResult:
    """
    Resultado final de la optimización por mesetas.
    """
    # Resultados por fase
    phase1_exploration: PhaseResult
    phase2_clustering: Optional[TopologyAnalysis]
    phase3_refinements: List[PhaseResult]
    
    # Mejor resultado global
    best_plateau: Optional[PlateauResult]
    best_refined_params: Dict[str, Any]
    best_refined_score: float
    best_refined_trial: int
    
    # Comparación con enfoque tradicional
    best_exploration_score: float
    improvement_over_exploration: float
    
    # Estadísticas
    total_trials: int
    total_time: float
    
    # Estudios Optuna (para inspección)
    exploration_study: Optional["optuna.Study"] = None
    refinement_studies: List["optuna.Study"] = field(default_factory=list)


# =============================================================================
# SAMPLER HÍBRIDO
# =============================================================================

class HybridPhaseSampler(optuna.samplers.BaseSampler):
    """
    Sampler híbrido que cambia de estrategia según la fase.
    
    Primeros X% de trials: RandomSampler (exploración)
    Resto: TPESampler (explotación)
    
    Este sampler es para uso DENTRO de una fase si se quiere
    combinar exploración y explotación en un solo estudio.
    """
    
    def __init__(
        self,
        exploration_ratio: float = 0.40,
        n_total_trials: int = 1000,
        seed: Optional[int] = None,
    ):
        self._exploration_ratio = exploration_ratio
        self._n_exploration = int(n_total_trials * exploration_ratio)
        self._random_sampler = RandomSampler(seed=seed)
        self._tpe_sampler = TPESampler(seed=seed, multivariate=True, group=True)
        self._seed = seed
    
    def infer_relative_search_space(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> Dict[str, "optuna.distributions.BaseDistribution"]:
        if trial.number < self._n_exploration:
            return self._random_sampler.infer_relative_search_space(study, trial)
        return self._tpe_sampler.infer_relative_search_space(study, trial)
    
    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, "optuna.distributions.BaseDistribution"],
    ) -> Dict[str, Any]:
        if trial.number < self._n_exploration:
            return self._random_sampler.sample_relative(study, trial, search_space)
        return self._tpe_sampler.sample_relative(study, trial, search_space)
    
    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: "optuna.distributions.BaseDistribution",
    ) -> Any:
        if trial.number < self._n_exploration:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        return self._tpe_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )


# =============================================================================
# PLATEAU OPTIMIZER
# =============================================================================

@dataclass
class PlateauOptimizer:
    """
    Optimizador de 3 Fases para encontrar mesetas de parámetros.
    
    USO:
    ```python
    optimizer = PlateauOptimizer(
        config=backtest_config,
        n_trials=5000,
        reporters=[...],
        plateau_config=PlateauOptimizerConfig(
            exploration_ratio=0.40,
            refine_top_n_plateaus=3,
        ),
    )
    
    result = optimizer.optimize(
        df=data,
        strategy=my_strategy,
        ...
    )
    
    # Mejor resultado robusto (de meseta)
    print(result.best_refined_params)
    print(result.best_refined_score)
    ```
    """
    
    config: BacktestConfig
    n_trials: int
    reporters: Sequence[Reporter]
    plateau_config: PlateauOptimizerConfig = field(default_factory=PlateauOptimizerConfig)
    activo: Optional[str] = None
    seed: Optional[int] = None
    
    # Estado interno
    _exploration_study: Optional["optuna.Study"] = None
    _refinement_studies: List["optuna.Study"] = field(default_factory=list)
    _topology_analysis: Optional[TopologyAnalysis] = None
    
    def optimize(
        self,
        *,
        df: pl.DataFrame,
        strategy: Strategy,
        df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
        base_timeframe: Optional[str] = None,
        perturbation_config: Optional[Any] = None,
    ) -> PlateauOptimizationResult:
        """
        Ejecuta la optimización de 3 fases.
        
        Returns:
            PlateauOptimizationResult con todo el análisis
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("Optuna es requerido")
        
        total_start = time.perf_counter()
        base_tf = normalize_timeframe_to_suffix(base_timeframe or "1h")
        df_map = df_by_timeframe or {base_tf: df}
        df_base = df_map.get(base_tf, df)
        
        cfg = self.plateau_config
        
        # Calcular distribución de trials
        n_exploration = int(self.n_trials * cfg.exploration_ratio)
        n_refinement_total = self.n_trials - n_exploration
        
        if cfg.verbose:
            self._print_phase_header("INICIO OPTIMIZACIÓN POR MESETAS", {
                "Total trials": self.n_trials,
                "Exploración (Fase 1)": n_exploration,
                "Refinamiento (Fase 3)": f"{n_refinement_total} (proporcional entre mesetas)",
                "Límite mesetas": "Sin límite" if cfg.max_plateaus_to_refine == 0 else cfg.max_plateaus_to_refine,
            })
        
        # =====================================================================
        # FASE 1: EXPLORACIÓN MASIVA
        # =====================================================================
        phase1_result = self._run_exploration_phase(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            n_trials=n_exploration,
            perturbation_config=perturbation_config,
        )
        
        # =====================================================================
        # FASE 2: DETECCIÓN DE MESETAS
        # =====================================================================
        if cfg.verbose:
            self._print_phase_header("FASE 2: DETECCIÓN DE MESETAS", {})
        
        phase2_start = time.perf_counter()
        
        self._topology_analysis = analyze_topology(
            study=self._exploration_study,
            config=cfg.plateau_config,
            verbose=cfg.verbose,
        )
        
        phase2_time = time.perf_counter() - phase2_start
        
        if cfg.verbose:
            print_topology_report(self._topology_analysis)
        
        # =====================================================================
        # FASE 3: REFINAMIENTO CMA-ES
        # =====================================================================
        phase3_results = []
        
        if self._topology_analysis.plateaus:
            # Determinar cuántas mesetas refinar
            all_plateaus = self._topology_analysis.plateaus
            if cfg.max_plateaus_to_refine > 0:
                plateaus_to_refine = all_plateaus[:cfg.max_plateaus_to_refine]
            else:
                plateaus_to_refine = all_plateaus  # Refinar TODAS
            
            n_plateaus = len(plateaus_to_refine)
            
            # Distribuir trials proporcionalmente entre mesetas
            n_per_plateau = max(cfg.min_trials_per_plateau, n_refinement_total // n_plateaus)
            
            if cfg.verbose:
                print(f"\n📊 Distribución: {n_refinement_total} trials ÷ {n_plateaus} mesetas = {n_per_plateau} trials/meseta")
            
            for i, plateau in enumerate(plateaus_to_refine):
                if cfg.verbose:
                    self._print_phase_header(
                        f"FASE 3.{i+1}: REFINAMIENTO CMA-ES (Meseta {plateau.cluster_id})",
                        {
                            "Score medio meseta": f"{plateau.mean_score:.2f}",
                            "Trials en meseta": plateau.n_trials,
                            "Trials refinamiento": n_per_plateau,
                        }
                    )
                
                refinement_result = self._run_refinement_phase(
                    plateau=plateau,
                    df_base=df_base,
                    df_map=df_map,
                    strategy=strategy,
                    base_tf=base_tf,
                    n_trials=n_per_plateau,
                    perturbation_config=perturbation_config,
                )
                
                phase3_results.append(refinement_result)
        else:
            if cfg.verbose:
                print("\n⚠️ No se encontraron mesetas. Usando resultado de exploración.")
        
        # =====================================================================
        # RESULTADO FINAL
        # =====================================================================
        total_time = time.perf_counter() - total_start
        
        # Encontrar el mejor resultado refinado
        best_refined_score = phase1_result.best_score
        best_refined_params = phase1_result.best_params
        best_refined_trial = phase1_result.best_trial_number
        best_plateau = None
        
        for i, (plateau, result) in enumerate(
            zip(self._topology_analysis.plateaus[:len(phase3_results)], phase3_results)
        ):
            if result.best_score > best_refined_score:
                best_refined_score = result.best_score
                best_refined_params = result.best_params
                best_refined_trial = result.best_trial_number
                best_plateau = plateau
        
        improvement = best_refined_score - phase1_result.best_score
        
        result = PlateauOptimizationResult(
            phase1_exploration=phase1_result,
            phase2_clustering=self._topology_analysis,
            phase3_refinements=phase3_results,
            best_plateau=best_plateau,
            best_refined_params=best_refined_params,
            best_refined_score=best_refined_score,
            best_refined_trial=best_refined_trial,
            best_exploration_score=phase1_result.best_score,
            improvement_over_exploration=improvement,
            total_trials=self.n_trials,
            total_time=total_time,
            exploration_study=self._exploration_study,
            refinement_studies=self._refinement_studies,
        )
        
        if cfg.verbose:
            self._print_final_report(result)
        
        return result
    
    def _run_exploration_phase(
        self,
        *,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
        n_trials: int,
        perturbation_config: Optional[Any],
    ) -> PhaseResult:
        """
        FASE 1: Exploración sistemática del espacio de parámetros.
        
        Samplers disponibles:
        - "qmc": Quasi-Monte Carlo (Sobol) - RECOMENDADO
                 Cobertura sistemática y uniforme de TODO el espacio
                 Más trials = mayor resolución, sin concentrarse en ningún punto
        - "random": RandomSampler - Aleatorio puro
        - "tpe": TPESampler - Con aprendizaje (tiende a concentrarse)
        """
        cfg = self.plateau_config
        
        if cfg.verbose:
            sampler_desc = {
                "qmc": "QMC (Sobol) - Cobertura sistemática uniforme",
                "random": "RANDOM - Aleatorio puro",
                "tpe": "TPE - Con aprendizaje bayesiano",
            }.get(cfg.exploration_sampler.lower(), cfg.exploration_sampler.upper())
            
            self._print_phase_header("FASE 1: EXPLORACIÓN SISTEMÁTICA", {
                "Trials": n_trials,
                "Sampler": sampler_desc,
            })
        
        start_time = time.perf_counter()
        
        # Crear sampler según configuración
        sampler_type = cfg.exploration_sampler.lower()
        
        if sampler_type == "qmc" and _QMC_AVAILABLE:
            # QMC con secuencia Sobol: cobertura uniforme y sistemática
            # - scramble=True: añade algo de variación manteniendo uniformidad
            # - warn_independent_sampling=False: evita warnings
            # - seed: QMC necesita seed explícito (usamos 42 si no hay)
            qmc_seed = self.seed if self.seed is not None else 42
            sampler = QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=qmc_seed,
                warn_independent_sampling=False,
            )
            if cfg.verbose:
                print(f"   📐 Usando secuencia Sobol: {n_trials} puntos cubrirán el espacio uniformemente")
        elif sampler_type == "qmc" and not _QMC_AVAILABLE:
            # Fallback a Random si QMC no está disponible
            if cfg.verbose:
                print("   ⚠️ QMCSampler no disponible, usando RandomSampler")
            sampler = RandomSampler(seed=self.seed)
        elif sampler_type == "random":
            sampler = RandomSampler(seed=self.seed)
        else:  # tpe
            sampler = TPESampler(seed=self.seed, multivariate=True, group=True)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"plateau_exploration_{strategy.name}",
        )
        
        set_study_for_scorer(study)
        self._exploration_study = study
        
        # Crear objetivo
        objective = self._create_objective(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            perturbation_config=perturbation_config,
            phase_name="exploration",
        )
        
        # Notificar a reporters del cambio de fase
        self._notify_phase_change("EXPLORACIÓN MASIVA", n_trials)
        
        # Optimizar (sin barra de progreso de Optuna - usamos nuestro panel)
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            catch=(Exception,),
            show_progress_bar=False,
        )
        
        elapsed = time.perf_counter() - start_time
        
        best_trial = study.best_trial
        
        return PhaseResult(
            phase_name="exploration",
            n_trials=len(study.trials),
            best_score=best_trial.value if best_trial else 0.0,
            best_params=best_trial.params if best_trial else {},
            best_trial_number=best_trial.number if best_trial else 0,
            elapsed_time=elapsed,
            extra_info={
                "sampler": cfg.exploration_sampler,
                "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            },
        )
    
    def _run_refinement_phase(
        self,
        *,
        plateau: PlateauResult,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
        n_trials: int,
        perturbation_config: Optional[Any],
    ) -> PhaseResult:
        """
        FASE 3: Refinamiento CMA-ES restringido a una meseta.
        """
        start_time = time.perf_counter()
        
        # Crear sampler CMA-ES
        # Usamos el centroide como punto inicial
        x0 = plateau.centroid_params
        sigma0 = 0.2  # Desviación inicial conservadora
        
        sampler = CmaEsSampler(
            seed=self.seed,
            n_startup_trials=5,
            warn_independent_sampling=False,
            x0=x0,
            sigma0=sigma0,
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"plateau_refine_{strategy.name}_cluster{plateau.cluster_id}",
        )
        
        set_study_for_scorer(study)
        self._refinement_studies.append(study)
        
        # Crear objetivo CON BOUNDS RESTRINGIDOS
        objective = self._create_objective(
            df_base=df_base,
            df_map=df_map,
            strategy=strategy,
            base_tf=base_tf,
            perturbation_config=perturbation_config,
            phase_name=f"refine_cluster{plateau.cluster_id}",
            param_bounds=plateau.param_bounds,
            centroid_params=plateau.centroid_params,
        )
        
        # Notificar a reporters del cambio de fase
        self._notify_phase_change(f"REFINAMIENTO CMA-ES (Meseta {plateau.cluster_id})", n_trials)
        
        # Optimizar (sin barra de progreso de Optuna - usamos nuestro panel)
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            catch=(Exception,),
            show_progress_bar=False,
        )
        
        elapsed = time.perf_counter() - start_time
        
        best_trial = study.best_trial
        
        return PhaseResult(
            phase_name=f"refinement_cluster_{plateau.cluster_id}",
            n_trials=len(study.trials),
            best_score=best_trial.value if best_trial else plateau.mean_score,
            best_params=best_trial.params if best_trial else plateau.centroid_params,
            best_trial_number=best_trial.number if best_trial else 0,
            elapsed_time=elapsed,
            extra_info={
                "cluster_id": plateau.cluster_id,
                "cluster_mean_score": plateau.mean_score,
                "improvement": (best_trial.value - plateau.mean_score) if best_trial else 0.0,
            },
        )
    
    def _create_objective(
        self,
        *,
        df_base: pl.DataFrame,
        df_map: Dict[str, pl.DataFrame],
        strategy: Strategy,
        base_tf: str,
        perturbation_config: Optional[Any],
        phase_name: str,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        centroid_params: Optional[Dict[str, float]] = None,
    ) -> Callable[["optuna.Trial"], float]:
        """
        Crea función objetivo para Optuna.
        
        Si param_bounds está definido, restringe la búsqueda a esos límites.
        Si centroid_params está definido, usa esos valores para parámetros no en bounds.
        """
        from .runner import SignalGenerator, BacktestEngine, apply_perturbation, PerturbationConfig
        
        def objective(trial: "optuna.trial.Trial") -> float:
            # Sugerir parámetros
            if param_bounds:
                # Modo refinamiento: usar bounds de la meseta + centroid para parámetros fijos
                params_puros = self._suggest_params_bounded(trial, strategy, param_bounds, centroid_params)
            else:
                # Modo exploración: usar suggest_params normal
                params_puros = strategy.suggest_params(trial)
            
            params_rt = self._prepare_params(params_puros, trial, strategy, base_tf)
            
            entry_tf = params_rt["__timeframe_entry"]
            df_entry = df_map.get(entry_tf, df_base)
            
            # Perturbación (si está habilitada)
            df_trial = df_entry
            perturb_info = {"perturbation_applied": False}
            
            if perturbation_config and getattr(perturbation_config, 'enabled', False):
                df_trial, _, perturb_info = apply_perturbation(
                    df_entry, perturbation_config, trial.number
                )
            
            # Generar señales
            signals_df = SignalGenerator.generate_signals(df_trial, strategy, params_rt, df_map)
            
            # Backtest
            trades_df, equity_curve, metrics = BacktestEngine.run_backtest(
                df_trial, signals_df, self.config, params_rt, strategy,
            )
            
            if trades_df.is_empty():
                return 0.0
            
            trial.set_user_attr("metricas", metrics)
            
            # Score basado en calidad (sin test de vecindario)
            score = float(score_optuna(metrics))
            
            # Artifacts para reporters
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
                perturbado=perturb_info.get("perturbation_applied", False),
                perturb_seed=None,
                neighborhood_result=None,
            )
            
            for reporter in self.reporters:
                try:
                    reporter.on_trial_end(artifacts)
                except Exception as e:
                    import traceback
                    print(f"[ERROR en reporter] {type(reporter).__name__}: {e}")
                    traceback.print_exc()
            
            return score
        
        return objective
    
    def _suggest_params_bounded(
        self,
        trial: "optuna.trial.Trial",
        strategy: Strategy,
        bounds: Dict[str, Tuple[float, float]],
        centroid_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Sugiere parámetros restringidos a los bounds de la meseta.
        
        Para parámetros no en bounds, usa los valores del centroide si están disponibles.
        Esto asegura que durante el refinamiento, los parámetros de estrategia se mantienen
        fijos mientras solo se optimizan los parámetros de salida (exit params).
        """
        # Obtener la definición original de parámetros
        parametros_optuna = getattr(strategy, "parametros_optuna", {})
        params = {}
        
        # Si parametros_optuna está vacío, usar centroid_params directamente para parámetros fijos
        if not parametros_optuna and centroid_params:
            # Copiar todos los parámetros del centroide que no están en bounds
            for name, value in centroid_params.items():
                if name not in bounds:
                    params[name] = int(round(value)) if isinstance(value, float) and value == int(value) else value
            
            # Sugerir solo los parámetros en bounds (exit params)
            for name, (bound_min, bound_max) in bounds.items():
                # Inferir tipo del centroid o usar float por defecto
                if centroid_params and name in centroid_params:
                    ref_value = centroid_params[name]
                    is_int = isinstance(ref_value, int) or (isinstance(ref_value, float) and ref_value == int(ref_value))
                else:
                    is_int = False
                
                if is_int:
                    low, high = int(bound_min), int(bound_max)
                    if low > high:
                        low, high = high, low
                    params[name] = trial.suggest_int(name, low, high) if low != high else low
                else:
                    if bound_min > bound_max:
                        bound_min, bound_max = bound_max, bound_min
                    params[name] = trial.suggest_float(name, bound_min, bound_max) if abs(bound_max - bound_min) > 1e-10 else bound_min
            
            return params
        
        for name, spec in parametros_optuna.items():
            if name in bounds:
                # Restringir al rango de la meseta
                bound_min, bound_max = bounds[name]
                
                if spec["type"] == "int":
                    # Asegurar que los límites son enteros válidos
                    low = max(spec.get("low", bound_min), int(bound_min))
                    high = min(spec.get("high", bound_max), int(bound_max))
                    if low > high:
                        low, high = high, low
                    if low == high:
                        params[name] = low
                    else:
                        step = spec.get("step", 1)
                        params[name] = trial.suggest_int(name, low, high, step=step)
                        
                elif spec["type"] == "float":
                    low = max(spec.get("low", bound_min), bound_min)
                    high = min(spec.get("high", bound_max), bound_max)
                    if low > high:
                        low, high = high, low
                    if abs(low - high) < 1e-10:
                        params[name] = low
                    else:
                        step = spec.get("step")
                        log = spec.get("log", False)
                        if log and low <= 0:
                            low = 1e-6
                        params[name] = trial.suggest_float(name, low, high, step=step, log=log)
                else:
                    # Categórico u otro: usar suggest normal
                    params[name] = strategy.suggest_params(trial).get(name)
            elif centroid_params and name in centroid_params:
                # Usar valor del centroide (parámetro fijo durante refinamiento)
                value = centroid_params[name]
                if spec["type"] == "int":
                    params[name] = int(round(value))
                else:
                    params[name] = value
            else:
                # Fallback: usar suggest normal (no debería ocurrir en refinamiento)
                normal_params = strategy.suggest_params(trial)
                if name in normal_params:
                    params[name] = normal_params[name]
        
        return params
    
    def _prepare_params(
        self,
        params_puros: Dict[str, Any],
        trial: "optuna.trial.Trial",
        strategy: Strategy,
        base_tf: str,
    ) -> Dict[str, Any]:
        """
        Prepara parámetros inyectando configuración del sistema.
        """
        params_rt = dict(params_puros)
        
        params_rt["__activo"] = self.activo
        params_rt["__saldo_inicial"] = float(self.config.saldo_inicial)
        params_rt["__saldo_operativo_max"] = float(self.config.saldo_operativo_max)
        params_rt["__qty_max_activo"] = float(self.config.qty_max_activo)
        params_rt["__comision_pct"] = float(self.config.comision_pct)
        params_rt["__comision_sides"] = int(self.config.comision_sides)
        params_rt["__saldo_usado"] = float(self.config.saldo_usado)
        params_rt["__apalancamiento_max"] = float(self.config.apalancamiento_max)
        params_rt["__strategy_exit_enabled"] = bool(getattr(strategy, "SALIDAS_PERSONALIZADAS", False))
        
        # Resolver configuración de salida
        exit_settings = resolve_exit_settings_for_trial(trial=trial, config=self.config)
        params_rt["__exit_type"] = exit_settings.exit_type
        params_rt["__exit_sl_pct"] = exit_settings.sl_pct
        params_rt["__exit_tp_pct"] = exit_settings.tp_pct
        params_rt["__exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["__exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # Aliases
        params_rt["exit_type"] = exit_settings.exit_type
        params_rt["exit_sl_pct"] = exit_settings.sl_pct
        params_rt["exit_tp_pct"] = exit_settings.tp_pct
        params_rt["exit_trail_act_pct"] = exit_settings.trail_act_pct
        params_rt["exit_trail_dist_pct"] = exit_settings.trail_dist_pct
        
        # Timeframes
        entry_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_entry", None) or base_tf)
        exit_tf = normalize_timeframe_to_suffix(getattr(strategy, "timeframe_exit", None) or base_tf)
        
        params_rt["__timeframe_base"] = base_tf
        params_rt["__timeframe_entry"] = entry_tf
        params_rt["__timeframe_exit"] = exit_tf
        
        return params_rt
    
    def _print_phase_header(self, title: str, info: Dict[str, Any]) -> None:
        """Imprime cabecera de fase."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            
            console = Console()
            
            content = f"[bold cyan]{title}[/bold cyan]"
            if info:
                content += "\n\n" + "\n".join(f"[white]{k}:[/white] {v}" for k, v in info.items())
            
            console.print()
            console.print(Panel(content, border_style="blue"))
            
        except ImportError:
            print(f"\n{'='*60}")
            print(f" {title}")
            for k, v in info.items():
                print(f"   {k}: {v}")
            print('='*60)
    
    def _notify_phase_change(self, phase_name: str, n_trials: int) -> None:
        """
        Notifica a los reporters del cambio de fase.
        
        Esto permite que el panel elegante muestre la fase actual
        y el progreso correcto (ej: "42/2000").
        """
        for reporter in self.reporters:
            # Si el reporter tiene método set_phase, usarlo
            if hasattr(reporter, 'set_phase'):
                try:
                    reporter.set_phase(phase_name, n_trials)
                except Exception:
                    pass
    
    def _print_final_report(self, result: PlateauOptimizationResult) -> None:
        """Imprime reporte final."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            
            console = Console()
            
            # Panel principal
            improvement_str = f"+{result.improvement_over_exploration:.2f}" if result.improvement_over_exploration > 0 else f"{result.improvement_over_exploration:.2f}"
            
            content = (
                f"[bold green]✅ OPTIMIZACIÓN POR MESETAS COMPLETADA[/bold green]\n\n"
                f"[white]Tiempo total:[/white] {result.total_time:.1f}s\n"
                f"[white]Trials totales:[/white] {result.total_trials}\n\n"
                f"[white]Score Exploración:[/white] {result.best_exploration_score:.2f}\n"
                f"[white]Score Refinado:[/white] [bold]{result.best_refined_score:.2f}[/bold]\n"
                f"[white]Mejora:[/white] {improvement_str}\n"
            )
            
            if result.best_plateau:
                content += f"\n[white]Meseta Ganadora:[/white] Cluster {result.best_plateau.cluster_id}"
            
            console.print()
            console.print(Panel(content, title="📊 RESULTADO FINAL", border_style="green"))
            
            # Tabla de parámetros
            if result.best_refined_params:
                table = Table(title="🎯 Parámetros Robustos", show_header=True)
                table.add_column("Parámetro", style="cyan")
                table.add_column("Valor", justify="right", style="green")
                
                for k, v in result.best_refined_params.items():
                    if not k.startswith("__"):
                        if isinstance(v, float):
                            table.add_row(k, f"{v:.4f}")
                        else:
                            table.add_row(k, str(v))
                
                console.print(table)
            
        except ImportError:
            print(f"\n{'='*60}")
            print("RESULTADO FINAL")
            print(f"Score Exploración: {result.best_exploration_score:.2f}")
            print(f"Score Refinado: {result.best_refined_score:.2f}")
            print(f"Mejora: {result.improvement_over_exploration:.2f}")
            print('='*60)


# =============================================================================
# FUNCIÓN DE CONVENIENCIA
# =============================================================================

def run_plateau_optimization(
    *,
    df: pl.DataFrame,
    strategy: Strategy,
    backtest_config: BacktestConfig,
    n_trials: int,
    reporters: Sequence[Reporter],
    plateau_config: Optional[PlateauOptimizerConfig] = None,
    df_by_timeframe: Optional[Dict[str, pl.DataFrame]] = None,
    base_timeframe: Optional[str] = None,
    perturbation_config: Optional[Any] = None,
    activo: Optional[str] = None,
    seed: Optional[int] = None,
) -> PlateauOptimizationResult:
    """
    Función de conveniencia para ejecutar optimización por mesetas.
    
    Esta es la forma más simple de usar el sistema de mesetas.
    """
    if plateau_config is None:
        plateau_config = PlateauOptimizerConfig()
    
    optimizer = PlateauOptimizer(
        config=backtest_config,
        n_trials=n_trials,
        reporters=reporters,
        plateau_config=plateau_config,
        activo=activo,
        seed=seed,
    )
    
    return optimizer.optimize(
        df=df,
        strategy=strategy,
        df_by_timeframe=df_by_timeframe,
        base_timeframe=base_timeframe,
        perturbation_config=perturbation_config,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuración
    "PlateauOptimizerConfig",
    "PlateauOptimizationResult",
    "PhaseResult",
    
    # Clases principales
    "PlateauOptimizer",
    "HybridPhaseSampler",
    
    # Funciones
    "run_plateau_optimization",
]

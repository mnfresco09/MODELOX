"""modelox/core/topology.py

═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS TOPOLÓGICO DE MESETAS - "El Ojo del Sistema"
═══════════════════════════════════════════════════════════════════════════════

Este módulo implementa la detección de "mesetas de parámetros" usando clustering
HDBSCAN (Hierarchical DBSCAN). Las mesetas son regiones del espacio de parámetros 
donde múltiples combinaciones similares producen buenos resultados consistentes.

¿POR QUÉ HDBSCAN EN LUGAR DE DBSCAN?
------------------------------------
DBSCAN tiene un problema fatal: usa un radio fijo (eps).
- Si eps es pequeño: solo detecta mesetas muy densas, pierde las amplias
- Si eps es grande: fusiona mesetas distintas, pierde precisión

HDBSCAN resuelve esto:
- No usa radio fijo - construye un árbol jerárquico de densidades
- Detecta clusters de DENSIDAD VARIABLE simultáneamente
- Proporciona PROBABILIDADES de pertenencia (soft clustering)
  → Podemos filtrar puntos dudosos de los bordes

FILOSOFÍA:
    - Un "pico" es un punto aislado con buen score pero sin vecinos buenos
    - Una "meseta" es una región densa donde MUCHOS puntos tienen buen score
    - Las mesetas indican ROBUSTEZ: pequeños cambios en parámetros no arruinan el resultado

PIPELINE:
    1. Extraer trials con score > percentil (auto-adaptable)
    2. Normalizar parámetros a [0, 1]
    3. Aplicar HDBSCAN para encontrar clusters
    4. Filtrar por probabilidad de pertenencia (>75% = centro puro)
    5. Calcular centroide de cada cluster
    6. Seleccionar el trial más cercano al centroide como "representante robusto"

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    # Fallback a DBSCAN si HDBSCAN no está disponible
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        pass

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# =============================================================================
# CONFIGURACIÓN DEL ANÁLISIS TOPOLÓGICO
# =============================================================================

@dataclass
class PlateauConfig:
    """
    Configuración para detección de mesetas con HDBSCAN.
    
    HDBSCAN PARAMETERS (Recomendado - Sin eps fijo)
    -----------------------------------------------
    min_cluster_size: Tamaño mínimo de un cluster para ser considerado meseta.
                      RECOMENDADO: 10-30 para 1000+ trials.
                      Más pequeño = más mesetas pequeñas detectadas.
                      Más grande = solo mesetas muy pobladas.
    
    min_samples: Controla la "conservaduridad" del clustering.
                 Valores altos = clusters más densos, menos ruido incluido.
                 RECOMENDADO: 5-15 (usualmente igual o menor a min_cluster_size).
    
    cluster_selection_epsilon: (Opcional) Fusiona clusters muy cercanos.
                               0.0 = sin fusión (más clusters pequeños).
                               0.1-0.3 = fusiona clusters adyacentes.
    
    SOFT CLUSTERING - PROBABILIDADES
    ---------------------------------
    min_membership_probability: Probabilidad mínima para considerar un punto "puro".
                                HDBSCAN asigna prob. de pertenencia a cada punto:
                                - 0.9+ = Centro del cluster (muy confiable)
                                - 0.5-0.9 = Borde del cluster (aceptable)
                                - <0.5 = Punto dudoso (descartar)
                                RECOMENDADO: 0.5-0.75
    
    FILTERING (FASE 2)
    -------------------
    El filtrado elimina trials de baja calidad ANTES del clustering.
    Se aplican DOS filtros secuenciales:
    
    1. FILTRO ROI: Descarta trials con ROI < 0%
       → Elimina todas las estrategias perdedoras
       → Solo pasan trials rentables al clustering
    
    2. FILTRO MEDIA (μ): Descarta trials con score < media
       → Elimina la mitad inferior de la distribución
       → Se adapta automáticamente a cualquier escala de scores
       
       Ejemplo: Si la media es 112, descarta todo score < 112
    
    min_trials_for_plateau: Mínimo de trials (después de filtrar)
                            para considerar una meseta válida.
    
    SELECTION
    ---------
    centroid_selection: Método para elegir el representante:
                        - "centroid": Trial más cercano al centro geométrico (RECOMENDADO)
                        - "best": Trial con mejor score en el cluster
                        - "median": Trial con score mediano en el cluster
                        - "highest_prob": Trial con mayor probabilidad de pertenencia
    
    FALLBACK
    --------
    use_hdbscan: Si True (default), usa HDBSCAN. Si False, usa DBSCAN legacy.
    eps: Solo usado si use_hdbscan=False (DBSCAN legacy).
    """
    # HDBSCAN - Parámetros principales
    min_cluster_size: int = 15       # Tamaño mínimo de meseta
    min_samples: int = 8             # Conservaduridad del clustering
    cluster_selection_epsilon: float = 0.0  # Fusión de clusters (0 = sin fusión)
    
    # Soft Clustering - Filtrado por probabilidad
    min_membership_probability: float = 0.5  # Solo puntos con >50% de confianza
    
    # Filtering Fase 2 - Filtros secuenciales
    min_trials_for_plateau: int = 10 # Mínimo tras filtrar
    
    # NOTA: Los filtros aplicados son:
    # 1. ROI >= 0% (elimina perdedores)
    # 2. Score >= media (μ) (elimina mitad inferior)
    
    # Selection
    centroid_selection: str = "centroid"  # "centroid", "best", "median", "highest_prob"
    
    # Fallback a DBSCAN
    use_hdbscan: bool = True
    eps: float = 0.15  # Solo para DBSCAN legacy
    
    # Legacy (no usado)
    n_plateaus_to_refine: int = 3
    
    # Parámetros a incluir en el análisis (None = todos los numéricos)
    params_to_analyze: Optional[List[str]] = None
    
    # Excluir parámetros que empiezan con estos prefijos
    exclude_prefixes: Tuple[str, ...] = ("__", "param_", "NOMBRE")
    params_to_analyze: Optional[List[str]] = None
    
    # Excluir parámetros que empiezan con estos prefijos
    exclude_prefixes: Tuple[str, ...] = ("__", "param_", "NOMBRE")


@dataclass
class PlateauResult:
    """
    Resultado de una meseta detectada.
    
    Contiene información sobre el cluster encontrado y su representante robusto.
    """
    cluster_id: int
    n_trials: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    
    # Centroide (valores medios de parámetros)
    centroid_params: Dict[str, float]
    
    # Trial representante (más cercano al centroide o mejor según config)
    representative_trial_number: int
    representative_params: Dict[str, Any]
    representative_score: float
    
    # Límites del cluster para refinamiento CMA-ES
    param_bounds: Dict[str, Tuple[float, float]]
    
    # Trials en este cluster
    trial_numbers: List[int]
    
    # Densidad del cluster (trials / volumen)
    density_score: float = 0.0


@dataclass
class TopologyAnalysis:
    """
    Resultado completo del análisis topológico.
    """
    # Mesetas encontradas (ordenadas por score medio descendente)
    plateaus: List[PlateauResult]
    
    # Puntos de ruido (picos aislados)
    noise_trials: List[int]
    
    # Estadísticas globales
    total_trials_analyzed: int
    n_plateaus_found: int
    n_noise_points: int
    
    # Parámetros usados
    config: PlateauConfig
    
    # Espacio de parámetros analizado
    param_names: List[str]
    param_ranges: Dict[str, Tuple[float, float]]


# =============================================================================
# EXTRACCIÓN DE TRIALS
# =============================================================================

def extract_trials_data(
    study: "optuna.Study",
    config: PlateauConfig,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extrae datos de trials completados del estudio Optuna.
    
    Returns:
        (lista de dicts con params, score y roi, lista de nombres de parámetros)
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está disponible")
    
    trials_data = []
    all_param_names = set()
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        
        if trial.value is None:
            continue
        
        # Extraer ROI de user_attrs (métricas)
        metricas = trial.user_attrs.get("metricas", {})
        roi = metricas.get("roi_pct", metricas.get("roi", 0.0))
        if roi is None:
            roi = 0.0
        
        # Filtrar parámetros
        params = {}
        for k, v in trial.params.items():
            # Excluir por prefijos
            if any(k.startswith(p) for p in config.exclude_prefixes):
                continue
            
            # Solo incluir parámetros numéricos
            if not isinstance(v, (int, float)):
                continue
            
            # Filtrar por lista específica si se proporciona
            if config.params_to_analyze is not None:
                if k not in config.params_to_analyze:
                    continue
            
            params[k] = float(v)
            all_param_names.add(k)
        
        if not params:
            continue
        
        trials_data.append({
            "trial_number": trial.number,
            "score": trial.value,
            "roi": float(roi),
            "params": params,
        })
    
    # Ordenar por score descendente
    trials_data.sort(key=lambda x: x["score"], reverse=True)
    
    return trials_data, sorted(all_param_names)


def filter_top_trials(
    trials_data: List[Dict[str, Any]],
    config: PlateauConfig,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filtra trials aplicando DOS filtros secuenciales:
    
    1. FILTRO ROI: Descarta trials con ROI < 0%
       → Elimina estrategias perdedoras
    
    2. FILTRO MEDIA: Descarta trials con score < media (μ) GLOBAL
       → La media se calcula sobre TODOS los trials de Fase 1
       → NO sobre los que pasaron el filtro ROI
    
    Args:
        trials_data: Lista de trials con scores y ROI
        config: Configuración (no usada, filtros son fijos)
        verbose: Si True, imprime información del filtrado
    
    Returns:
        Lista filtrada con trials de calidad
    """
    if not trials_data:
        return []
    
    n_initial = len(trials_data)
    
    # =========================================================================
    # CALCULAR MEDIA GLOBAL (sobre TODOS los trials, antes de filtrar)
    # Esta es la μ que muestra el panel Rich durante la Fase 1
    # =========================================================================
    all_scores = np.array([t["score"] for t in trials_data])
    global_mean = float(np.mean(all_scores))
    global_std = float(np.std(all_scores))
    global_min = float(np.min(all_scores))
    global_max = float(np.max(all_scores))
    
    # =========================================================================
    # FILTRO 1: ROI >= 0% (eliminar perdedores)
    # =========================================================================
    filtered_roi = [t for t in trials_data if t.get("roi", 0.0) >= 0.0]
    n_after_roi = len(filtered_roi)
    n_roi_rejected = n_initial - n_after_roi
    
    if not filtered_roi:
        if verbose:
            print(f"\n⚠️ Filtro ROI: Todos los {n_initial} trials tienen ROI < 0%")
        return []
    
    # =========================================================================
    # FILTRO 2: Score >= Media GLOBAL (μ de TODOS los trials)
    # =========================================================================
    # Usar la media GLOBAL calculada antes del filtro ROI
    threshold = global_mean
    
    # Aplicar filtro usando la media global
    filtered_final = [t for t in filtered_roi if t["score"] >= threshold]
    n_final = len(filtered_final)
    n_mean_rejected = n_after_roi - n_final
    
    if verbose:
        print(f"\n📊 Filtrado Fase 2 - Dos Filtros Secuenciales")
        print(f"   ─────────────────────────────────────")
        print(f"   Media GLOBAL (μ) de Fase 1: {global_mean:.1f}")
        print(f"   Rango scores: [{global_min:.1f}, {global_max:.1f}]")
        print(f"   Desviación (σ): {global_std:.1f}")
        print(f"   ─────────────────────────────────────")
        print(f"   FILTRO 1 - ROI >= 0%:")
        print(f"     Descartados (ROI < 0%): {n_roi_rejected} trials")
        print(f"     Pasan: {n_after_roi}/{n_initial}")
        print(f"   ─────────────────────────────────────")
        print(f"   FILTRO 2 - Score >= μ ({global_mean:.1f}):")
        print(f"     Descartados (score < {threshold:.1f}): {n_mean_rejected} trials")
        print(f"     Pasan: {n_final}/{n_after_roi}")
        print(f"   ─────────────────────────────────────")
        print(f"   ✅ RESULTADO FINAL: {n_final}/{n_initial} trials para clustering ({100*n_final/n_initial:.1f}%)")
    
    return filtered_final


# =============================================================================
# NORMALIZACIÓN DE PARÁMETROS
# =============================================================================

def normalize_params(
    trials_data: List[Dict[str, Any]],
    param_names: List[str],
) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """
    Normaliza parámetros al rango [0, 1] para que DBSCAN funcione correctamente.
    
    Returns:
        (matriz normalizada NxP, diccionario de rangos originales)
    """
    n_trials = len(trials_data)
    n_params = len(param_names)
    
    # Extraer matriz de parámetros
    X = np.zeros((n_trials, n_params))
    for i, trial in enumerate(trials_data):
        for j, name in enumerate(param_names):
            X[i, j] = trial["params"].get(name, 0.0)
    
    # Calcular rangos
    param_ranges = {}
    for j, name in enumerate(param_names):
        col = X[:, j]
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        param_ranges[name] = (min_val, max_val)
    
    # Normalizar con MinMax
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, param_ranges


def denormalize_params(
    normalized_values: np.ndarray,
    param_names: List[str],
    param_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """
    Convierte valores normalizados de vuelta al espacio original.
    """
    result = {}
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[name]
        range_val = max_val - min_val
        if range_val > 0:
            result[name] = min_val + normalized_values[i] * range_val
        else:
            result[name] = min_val
    return result


# =============================================================================
# CLUSTERING DBSCAN
# =============================================================================

def find_optimal_eps(
    X: np.ndarray,
    min_samples: int,
    k_multiplier: float = 1.5,
) -> float:
    """
    Estima eps óptimo usando el método del codo (k-distance graph).
    
    Args:
        X: Datos normalizados
        min_samples: Parámetro min_samples de DBSCAN
        k_multiplier: Factor para calcular k (k = min_samples * k_multiplier)
    
    Returns:
        eps estimado
    """
    if not _SKLEARN_AVAILABLE:
        return 0.15  # Default
    
    k = max(2, int(min_samples * k_multiplier))
    k = min(k, len(X) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # k-distance para cada punto
    k_distances = np.sort(distances[:, -1])
    
    # Encontrar el "codo" usando segunda derivada
    if len(k_distances) < 10:
        return float(np.median(k_distances))
    
    # Suavizar
    window = min(5, len(k_distances) // 10)
    if window > 1:
        k_distances_smooth = np.convolve(k_distances, np.ones(window)/window, mode='valid')
    else:
        k_distances_smooth = k_distances
    
    # Calcular segunda derivada
    if len(k_distances_smooth) > 2:
        d2 = np.diff(np.diff(k_distances_smooth))
        elbow_idx = np.argmax(d2) + 1
        elbow_idx = min(elbow_idx, len(k_distances) - 1)
        eps = float(k_distances[elbow_idx])
    else:
        eps = float(np.median(k_distances))
    
    # Limitar a rango razonable
    return max(0.05, min(0.5, eps))


def apply_clustering(
    X: np.ndarray,
    config: PlateauConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica HDBSCAN (o DBSCAN como fallback) para encontrar clusters (mesetas).
    
    HDBSCAN ventajas sobre DBSCAN:
    - No necesita eps (radio fijo)
    - Detecta clusters de densidad variable
    - Proporciona probabilidades de pertenencia (soft clustering)
    
    Args:
        X: Datos normalizados
        config: Configuración del análisis
        verbose: Si True, imprime información del clustering
    
    Returns:
        Tuple de (labels, probabilities)
        - labels: Array de labels (-1 = ruido, 0+ = cluster ID)
        - probabilities: Array de probabilidades de pertenencia [0, 1]
    """
    n_samples = len(X)
    
    if config.use_hdbscan and _HDBSCAN_AVAILABLE:
        # =====================================================
        # HDBSCAN - Clustering jerárquico adaptativo
        # =====================================================
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass (mejor para mesetas)
            prediction_data=True,  # Necesario para probabilidades
        )
        
        clusterer.fit(X)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        
        if verbose:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            print(f"\n🔬 HDBSCAN Clustering:")
            print(f"   Método: Jerárquico (sin eps fijo)")
            print(f"   min_cluster_size: {config.min_cluster_size}")
            print(f"   min_samples: {config.min_samples}")
            print(f"   Clusters encontrados: {n_clusters}")
            print(f"   Puntos de ruido: {n_noise} ({100*n_noise/n_samples:.1f}%)")
            if n_clusters > 0:
                for cid in range(n_clusters):
                    mask = labels == cid
                    avg_prob = np.mean(probabilities[mask])
                    print(f"   Cluster {cid}: {np.sum(mask)} puntos, prob. media: {avg_prob:.2f}")
    
    else:
        # =====================================================
        # DBSCAN - Fallback (requiere eps)
        # =====================================================
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn no está disponible")
        
        from sklearn.cluster import DBSCAN
        
        # Auto-calcular eps si es necesario
        eps = find_optimal_eps(X, config.min_samples) if config.eps <= 0 else config.eps
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=config.min_samples,
            metric='euclidean',
            n_jobs=-1,
        )
        
        labels = dbscan.fit_predict(X)
        # DBSCAN no da probabilidades - asignar 1.0 a todo
        probabilities = np.where(labels >= 0, 1.0, 0.0)
        
        if verbose:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            print(f"\n🔬 DBSCAN Clustering (fallback):")
            print(f"   eps: {eps:.3f}")
            print(f"   min_samples: {config.min_samples}")
            print(f"   Clusters encontrados: {n_clusters}")
            print(f"   Puntos de ruido: {n_noise} ({100*n_noise/n_samples:.1f}%)")
    
    return labels, probabilities


# Legacy alias para compatibilidad
def apply_dbscan(
    X: np.ndarray,
    config: PlateauConfig,
    auto_eps: bool = False,
) -> np.ndarray:
    """Legacy wrapper - usa apply_clustering internamente."""
    labels, _ = apply_clustering(X, config, verbose=False)
    return labels


# =============================================================================
# SELECCIÓN DE REPRESENTANTES
# =============================================================================

def calculate_centroid(
    cluster_data: List[Dict[str, Any]],
    param_names: List[str],
) -> Dict[str, float]:
    """
    Calcula el centroide (media) de un cluster.
    """
    if not cluster_data:
        return {}
    
    centroid = {}
    for name in param_names:
        values = [t["params"].get(name, 0.0) for t in cluster_data]
        centroid[name] = float(np.mean(values))
    
    return centroid


def find_nearest_to_centroid(
    cluster_data: List[Dict[str, Any]],
    centroid: Dict[str, float],
    param_names: List[str],
) -> Dict[str, Any]:
    """
    Encuentra el trial más cercano al centroide.
    """
    if not cluster_data:
        return {}
    
    if len(cluster_data) == 1:
        return cluster_data[0]
    
    centroid_arr = np.array([centroid.get(n, 0.0) for n in param_names])
    
    min_dist = float('inf')
    nearest = cluster_data[0]
    
    for trial in cluster_data:
        trial_arr = np.array([trial["params"].get(n, 0.0) for n in param_names])
        dist = np.linalg.norm(trial_arr - centroid_arr)
        if dist < min_dist:
            min_dist = dist
            nearest = trial
    
    return nearest


def select_representative(
    cluster_data: List[Dict[str, Any]],
    param_names: List[str],
    method: str = "centroid",
    probabilities: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Selecciona el trial representante de un cluster.
    
    Args:
        cluster_data: Trials del cluster
        param_names: Nombres de parámetros
        method: "centroid", "best", "median", o "highest_prob"
        probabilities: Array de probabilidades de pertenencia (para highest_prob)
    
    Returns:
        (trial representante, centroide del cluster)
    """
    if not cluster_data:
        return {}, {}
    
    centroid = calculate_centroid(cluster_data, param_names)
    
    if method == "best":
        # El mejor score del cluster
        representative = max(cluster_data, key=lambda x: x["score"])
    elif method == "median":
        # Score mediano
        sorted_cluster = sorted(cluster_data, key=lambda x: x["score"])
        mid_idx = len(sorted_cluster) // 2
        representative = sorted_cluster[mid_idx]
    elif method == "highest_prob" and probabilities is not None and len(probabilities) > 0:
        # El punto con mayor probabilidad de pertenencia (soft clustering)
        max_prob_idx = int(np.argmax(probabilities))
        representative = cluster_data[max_prob_idx]
    else:
        # centroid (default): más cercano al centro
        representative = find_nearest_to_centroid(cluster_data, centroid, param_names)
    
    return representative, centroid


def calculate_cluster_bounds(
    cluster_data: List[Dict[str, Any]],
    param_names: List[str],
    margin: float = 0.1,
) -> Dict[str, Tuple[float, float]]:
    """
    Calcula los límites de un cluster para refinamiento CMA-ES.
    
    Args:
        cluster_data: Trials del cluster
        param_names: Nombres de parámetros
        margin: Margen a añadir a los límites (10% por defecto)
    
    Returns:
        Diccionario {param_name: (min, max)}
    """
    bounds = {}
    
    for name in param_names:
        values = [t["params"].get(name, 0.0) for t in cluster_data]
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        # Añadir margen
        range_val = max_val - min_val
        min_val -= range_val * margin
        max_val += range_val * margin
        
        bounds[name] = (min_val, max_val)
    
    return bounds


def calculate_density(
    cluster_data: List[Dict[str, Any]],
    param_names: List[str],
    param_ranges: Dict[str, Tuple[float, float]],
) -> float:
    """
    Calcula la densidad del cluster (trials / volumen normalizado).
    """
    if len(cluster_data) < 2:
        return 0.0
    
    # Calcular "volumen" del cluster en espacio normalizado
    volume = 1.0
    for name in param_names:
        values = [t["params"].get(name, 0.0) for t in cluster_data]
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        # Normalizar a rango global
        global_min, global_max = param_ranges.get(name, (min_val, max_val))
        global_range = global_max - global_min
        if global_range > 0:
            normalized_range = (max_val - min_val) / global_range
        else:
            normalized_range = 1.0
        
        volume *= max(normalized_range, 0.001)
    
    # Densidad = trials / volumen
    density = len(cluster_data) / max(volume, 1e-10)
    
    return float(density)


# =============================================================================
# ANÁLISIS PRINCIPAL
# =============================================================================

def analyze_topology(
    study: "optuna.Study",
    config: Optional[PlateauConfig] = None,
    verbose: bool = True,
) -> TopologyAnalysis:
    """
    Ejecuta el análisis topológico completo con HDBSCAN.
    
    Este es el punto de entrada principal para detectar mesetas.
    HDBSCAN detecta automáticamente clusters de densidad variable
    y proporciona soft clustering con probabilidades de pertenencia.
    
    Args:
        study: Estudio Optuna con trials completados
        config: Configuración (usa defaults si None)
        verbose: Si True, muestra información del proceso
    
    Returns:
        TopologyAnalysis con todas las mesetas detectadas
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn es requerido para análisis topológico. Instalar: pip install scikit-learn")
    
    if config is None:
        config = PlateauConfig()
    
    # 1. Extraer datos de trials
    trials_data, param_names = extract_trials_data(study, config)
    
    if not trials_data:
        return TopologyAnalysis(
            plateaus=[],
            noise_trials=[],
            total_trials_analyzed=0,
            n_plateaus_found=0,
            n_noise_points=0,
            config=config,
            param_names=param_names,
            param_ranges={},
        )
    
    # 2. Filtrar top trials (AUTO-ADAPTABLE por percentil)
    top_trials = filter_top_trials(trials_data, config, verbose=verbose)
    
    if len(top_trials) < config.min_samples:
        # No hay suficientes trials para clustering
        return TopologyAnalysis(
            plateaus=[],
            noise_trials=[t["trial_number"] for t in top_trials],
            total_trials_analyzed=len(top_trials),
            n_plateaus_found=0,
            n_noise_points=len(top_trials),
            config=config,
            param_names=param_names,
            param_ranges={},
        )
    
    # 3. Normalizar parámetros
    X_normalized, param_ranges = normalize_params(top_trials, param_names)
    
    # 4. Aplicar HDBSCAN (o DBSCAN fallback)
    labels, probabilities = apply_clustering(X_normalized, config, verbose=verbose)
    
    # 5. Filtrar puntos de baja probabilidad de pertenencia (soft clustering)
    # HDBSCAN proporciona probabilidades; DBSCAN retorna 1.0 para todos
    min_prob = config.min_membership_probability
    
    # 6. Agrupar trials por cluster
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    cluster_probs: Dict[int, List[float]] = {}  # Probabilidades por cluster
    noise_trials = []
    
    for i, trial in enumerate(top_trials):
        label = int(labels[i])
        prob = float(probabilities[i])
        
        # Filtrar por probabilidad mínima de pertenencia
        if label == -1 or prob < min_prob:
            noise_trials.append(trial["trial_number"])
        else:
            if label not in clusters:
                clusters[label] = []
                cluster_probs[label] = []
            clusters[label].append(trial)
            cluster_probs[label].append(prob)
    
    # 7. Construir resultados de mesetas
    plateaus = []
    
    for cluster_id, cluster_data in clusters.items():
        if len(cluster_data) < config.min_trials_for_plateau:
            # Cluster muy pequeño → tratarlo como ruido
            noise_trials.extend([t["trial_number"] for t in cluster_data])
            continue
        
        scores = [t["score"] for t in cluster_data]
        probs = np.array(cluster_probs[cluster_id])
        
        # Seleccionar representante (ahora puede usar highest_prob)
        representative, centroid = select_representative(
            cluster_data, param_names, config.centroid_selection, probs
        )
        
        # Calcular límites para CMA-ES
        bounds = calculate_cluster_bounds(cluster_data, param_names)
        
        # Calcular densidad
        density = calculate_density(cluster_data, param_names, param_ranges)
        
        # Score de confianza basado en probabilidades medias
        avg_membership_prob = float(np.mean(probs))
        
        plateau = PlateauResult(
            cluster_id=cluster_id,
            n_trials=len(cluster_data),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            centroid_params=centroid,
            representative_trial_number=representative["trial_number"],
            representative_params=representative["params"],
            representative_score=representative["score"],
            param_bounds=bounds,
            trial_numbers=[t["trial_number"] for t in cluster_data],
            density_score=density * avg_membership_prob,  # Ponderar por confianza
        )
        
        plateaus.append(plateau)
    
    # 8. Ordenar mesetas por score medio (descendente)
    plateaus.sort(key=lambda p: p.mean_score, reverse=True)
    
    # 8. Retornar TODAS las mesetas (el límite se aplica en plateau_optimizer)
    # Ya no limitamos aquí: plateaus = plateaus[:config.n_plateaus_to_refine]
    
    return TopologyAnalysis(
        plateaus=plateaus,
        noise_trials=noise_trials,
        total_trials_analyzed=len(top_trials),
        n_plateaus_found=len(plateaus),
        n_noise_points=len(noise_trials),
        config=config,
        param_names=param_names,
        param_ranges=param_ranges,
    )


# =============================================================================
# VISUALIZACIÓN Y REPORTING
# =============================================================================

def print_topology_report(analysis: TopologyAnalysis) -> None:
    """
    Imprime un reporte del análisis topológico.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Header
        console.print()
        console.print(Panel(
            f"[bold cyan]📊 ANÁLISIS TOPOLÓGICO DE MESETAS[/bold cyan]\n\n"
            f"Trials analizados: {analysis.total_trials_analyzed}\n"
            f"Mesetas encontradas: {analysis.n_plateaus_found}\n"
            f"Puntos de ruido (picos): {analysis.n_noise_points}",
            border_style="cyan"
        ))
        
        if not analysis.plateaus:
            console.print("[yellow]⚠️ No se encontraron mesetas. Considera ajustar eps o min_samples.[/yellow]")
            return
        
        # Tabla de mesetas
        table = Table(title="🏔️ Mesetas Detectadas", show_header=True, header_style="bold magenta")
        table.add_column("Cluster", justify="center", style="cyan")
        table.add_column("Trials", justify="right", style="green")
        table.add_column("Score Medio", justify="right", style="yellow")
        table.add_column("Score Std", justify="right", style="dim")
        table.add_column("Rango", justify="center")
        table.add_column("Densidad", justify="right", style="blue")
        
        for p in analysis.plateaus:
            table.add_row(
                str(p.cluster_id),
                str(p.n_trials),
                f"{p.mean_score:.2f}",
                f"±{p.std_score:.2f}",
                f"[{p.min_score:.1f}, {p.max_score:.1f}]",
                f"{p.density_score:.2f}",
            )
        
        console.print(table)
        
        # Detalles de la mejor meseta
        best = analysis.plateaus[0]
        console.print()
        console.print(Panel(
            f"[bold green]🎯 MEJOR MESETA (Cluster {best.cluster_id})[/bold green]\n\n"
            f"[white]Trial Representante:[/white] #{best.representative_trial_number}\n"
            f"[white]Score Representante:[/white] {best.representative_score:.2f}\n\n"
            f"[white]Parámetros del Centroide:[/white]\n" +
            "\n".join(f"  • {k}: {v:.4f}" for k, v in best.centroid_params.items()),
            border_style="green"
        ))
        
    except ImportError:
        # Fallback sin Rich
        print("\n" + "="*60)
        print("ANÁLISIS TOPOLÓGICO DE MESETAS")
        print("="*60)
        print(f"Trials analizados: {analysis.total_trials_analyzed}")
        print(f"Mesetas encontradas: {analysis.n_plateaus_found}")
        print(f"Puntos de ruido: {analysis.n_noise_points}")
        
        for p in analysis.plateaus:
            print(f"\nCluster {p.cluster_id}: {p.n_trials} trials, score medio {p.mean_score:.2f}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuración
    "PlateauConfig",
    "PlateauResult",
    "TopologyAnalysis",
    
    # Funciones principales
    "analyze_topology",
    "extract_trials_data",
    "filter_top_trials",
    "apply_dbscan",
    "find_optimal_eps",
    
    # Utilidades
    "normalize_params",
    "denormalize_params",
    "calculate_centroid",
    "select_representative",
    "calculate_cluster_bounds",
    
    # Reporting
    "print_topology_report",
]

"""
================================================================================
MODELOX/PERTURBACIONES — DATA AUGMENTATION OHLCV
================================================================================

Módulo de perturbación de datos OHLCV para entrenamiento robusto.
Cada trial de Optuna recibe datos ligeramente mutados para evitar overfitting.

Uso:
    from modelox.perturbaciones import HighFrequencyOHLCVAugmenter, assert_ohlcv_coherence

    augmenter = HighFrequencyOHLCVAugmenter(price_scale=0.05, volume_scale=0.1)
    df_mutado = augmenter.augment(df.lazy())
    assert_ohlcv_coherence(df_mutado)
================================================================================
"""

from .augmenter import HighFrequencyOHLCVAugmenter, assert_ohlcv_coherence

__all__ = ["HighFrequencyOHLCVAugmenter", "assert_ohlcv_coherence"]

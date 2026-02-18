"""
================================================================================
IA/MODEL.PY — ARQUITECTURA GRU Y FUNCIÓN DE PÉRDIDA ASIMÉTRICA
================================================================================
Arquitectura:
  Input  → GRU(64, 2 capas, tanh, dropout=0.4)
         → Dense(32, tanh) → Dropout(0.4)
         → Dense(1, sigmoid)

Pérdida asimétrica:
  AsymmetricDirectionalLoss:
    BCE + penalización exponencial cuando signo predicho ≠ signo real
    Prioriza exactitud de dirección sobre magnitud
================================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from IA.config import (
    N_FEATURES, GRU_UNITS, N_GRU_LAYERS, DROPOUT,
    FC_UNITS, ALPHA_LOSS,
)

# ─── Dispositivo automático ───────────────────────────────────────────────────
def get_device() -> torch.device:
    """Detecta MPS (Apple Silicon), CUDA, o CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# MODELO GRU
# =============================================================================

class GRUTradingModel(nn.Module):
    """
    Red neuronal GRU de 2 capas para clasificación binaria de señales de trading.

    Entrada : (batch, lookback, n_features)
    Salida  : (batch,) — probabilidad de LONG [0, 1]

    Arquitectura:
      GRU Layer 1: input_size → gru_units (tanh, dropout inter-capas)
      GRU Layer 2: gru_units  → gru_units (tanh)
      FC Layer 1 : gru_units  → fc_units  (tanh + dropout)
      FC Layer 2 : fc_units   → 1         (sigmoid)
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        gru_units:   int   = GRU_UNITS,
        n_layers:    int   = N_GRU_LAYERS,
        dropout:     float = DROPOUT,
        fc_units:    int   = FC_UNITS,
    ) -> None:
        super().__init__()

        self.gru_units = gru_units
        self.n_layers  = n_layers

        # Capas GRU apiladas — PyTorch aplica dropout ENTRE capas (no en la última)
        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = gru_units,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        # Capa fully-connected
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(gru_units, fc_units)
        self.fc2     = nn.Linear(fc_units,  1)

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform para FC, orthogonal para GRU."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, n_features)
        returns: (batch,) probabilidades [0, 1]
        """
        # GRU — tomamos el último hidden state de la última capa
        gru_out, _ = self.gru(x)          # (batch, lookback, gru_units)
        last        = gru_out[:, -1, :]   # (batch, gru_units) — último timestep

        # Fully connected con tanh
        out = self.dropout(last)
        out = torch.tanh(self.fc1(out))   # activación tanh (como especificado)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out)).squeeze(-1)  # (batch,)
        return out

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inferencia sin gradientes."""
        self.eval()
        return self.forward(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# FUNCIÓN DE PÉRDIDA ASIMÉTRICA DIRECCIONAL
# =============================================================================

class AsymmetricDirectionalLoss(nn.Module):
    """
    Pérdida asimétrica que prioriza la exactitud DIRECCIONAL.

    Fórmula:
        L = BCE(pred, target) * exp(α * error_dirección * confianza)

    Donde:
        error_dirección = 1 si signo(pred) ≠ signo(target), 0 si igual
        confianza       = |pred - 0.5| * 2  ∈ [0, 1]
        α               = factor de penalización (> 1)

    Cuando el modelo está equivocado Y muy seguro de sí mismo,
    la penalización es máxima (e^α ≈ 7.4 para α=2).

    Cuando el modelo está equivocado pero inseguro (prob ≈ 0.5),
    la penalización es mínima (e^0 = 1 → solo BCE normal).
    """

    def __init__(self, alpha: float = ALPHA_LOSS, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha      = alpha
        self.pos_weight = pos_weight  # class weight para label=1 (LONG)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (batch,) — probabilidades sigmoid [0, 1]
        target : (batch,) — etiquetas {0.0, 1.0}
        """
        # ── BCE con class weight ──────────────────────────────────────
        pw     = torch.tensor(self.pos_weight, device=pred.device, dtype=pred.dtype)
        bce    = F.binary_cross_entropy(
            pred, target,
            weight=None,
            reduction="none",
        )
        # Aplicar class weight manualmente
        weight = torch.where(target == 1.0, pw, torch.ones_like(target))
        bce    = bce * weight

        # ── Error de dirección ────────────────────────────────────────
        pred_dir  = (pred > 0.5).float()
        dir_error = (pred_dir != target).float()     # 1 si equivocado

        # ── Confianza del modelo ──────────────────────────────────────
        confidence = (pred - 0.5).abs() * 2.0        # 0 = inseguro, 1 = muy seguro

        # ── Penalización exponencial ──────────────────────────────────
        penalty = torch.exp(self.alpha * dir_error * confidence)

        return (bce * penalty).mean()


# =============================================================================
# UTILIDADES
# =============================================================================

def build_model(
    n_features: int   = N_FEATURES,
    gru_units:  int   = GRU_UNITS,
    n_layers:   int   = N_GRU_LAYERS,
    dropout:    float = DROPOUT,
    fc_units:   int   = FC_UNITS,
) -> GRUTradingModel:
    """Construye y mueve el modelo al dispositivo correcto."""
    model = GRUTradingModel(n_features, gru_units, n_layers, dropout, fc_units)
    return model.to(DEVICE)


def model_summary(model: GRUTradingModel) -> dict:
    """Resumen del modelo para mostrar en consola."""
    return {
        "arquitectura"   : "GRU",
        "n_features"     : model.gru.input_size,
        "gru_units"      : model.gru_units,
        "n_layers"       : model.n_layers,
        "fc_units"       : model.fc1.out_features,
        "dropout"        : model.dropout.p,
        "parametros"     : model.count_parameters(),
        "dispositivo"    : str(DEVICE),
    }

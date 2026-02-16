"""
Representation analysis: collect hidden activations on fixed probe set for A and B;
compute CKA or RSA between A and B representations per layer.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn


def collect_hidden_activations(
    model: nn.Module,
    probe_inputs: torch.Tensor,
    layer_ids: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """
    Run model on probe_inputs and collect hidden activations at each linear layer.
    Assumes model has .net (Sequential) with linear and Tanh. Returns list of (batch, hidden_dim).
    """
    activations = []
    x = probe_inputs
    for m in model.net:
        if hasattr(m, "weight") and m.weight.dim() == 2:
            x = m(x)
            activations.append(x.detach())
        else:
            x = m(x)
    return activations


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA (Centered Kernel Alignment) between X and Y.
    X: (n, d1), Y: (n, d2). Returns scalar in [0, 1].
    """
    X = X.double()
    Y = Y.double()
    n = X.size(0)
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    xx = (Xc.t() @ Xc).norm().item() ** 2
    yy = (Yc.t() @ Yc).norm().item() ** 2
    xy = (Xc.t() @ Yc).norm().item() ** 2
    if xx * yy == 0:
        raise ValueError(
            "linear_CKA: degenerate activations (zero or constant), cannot compute similarity"
        )
    return float(xy / (xx * yy) ** 0.5)


def representation_similarity(
    acts_A: List[torch.Tensor],
    acts_B: List[torch.Tensor],
) -> List[float]:
    """Per-layer CKA between acts_A[i] and acts_B[i]."""
    return [linear_CKA(a, b) for a, b in zip(acts_A, acts_B)]

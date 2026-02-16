"""
Online statistics for rewiring: per-parameter EMA of |grad|, grad^2, optionally |w|.
Updated after backward, before optimizer.step(). Efficient tensor ops only.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from src.models.masked_mlp import MaskedMLP


class RewiringSignals:
    """Per-weight EMA tensors: grad_abs, grad_sq. Updated after backward, before step."""

    def __init__(self, model: MaskedMLP, decay: float = 0.99):
        self.decay = decay
        self._emas: Dict[int, dict] = {}
        self._layers: List[nn.Module] = model._weighted_layers()
        for i, layer in enumerate(self._layers):
            w = layer.weight
            device = w.device
            self._emas[i] = {
                "grad_abs": torch.zeros_like(w, device=device),
                "grad_sq": torch.zeros_like(w, device=device),
            }

    def update(self, model: MaskedMLP) -> None:
        """Call after loss.backward(); update EMAs from current gradients."""
        layers = model._weighted_layers()
        for idx, layer in enumerate(layers):
            if idx not in self._emas:
                continue
            w = layer.weight
            g = w.grad
            if g is None:
                raise RuntimeError(
                    "RewiringSignals.update: layer {} has no gradients "
                    "(ensure loss.backward() was called and parameters are not frozen)".format(idx)
                )
            ema = self._emas[idx]
            new_abs = g.detach().abs()
            new_sq = g.detach().square()
            ema["grad_abs"].mul_(self.decay).add_(new_abs, alpha=1 - self.decay)
            ema["grad_sq"].mul_(self.decay).add_(new_sq, alpha=1 - self.decay)

    def get_ema_grad_abs(self, layer_id: int) -> torch.Tensor:
        return self._emas[layer_id]["grad_abs"]

    def get_ema_grad_sq(self, layer_id: int) -> torch.Tensor:
        return self._emas[layer_id]["grad_sq"]

    def num_layers(self) -> int:
        return len(self._emas)

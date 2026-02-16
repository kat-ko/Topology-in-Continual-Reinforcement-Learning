"""
Masked MLP: sparse connectivity via dense weights and binary masks (W_eff = W * M).
Masks are non-trainable. Supports edge budget, export/import masks, and active/inactive edge listing for rewiring.
"""
from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def _ensure_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if x.device != device else x


class MaskedLinear(nn.Module):
    """Linear layer with binary mask: output = (weight * mask) @ input + bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mask = torch.ones(out_features, in_features)
        self.register_buffer("_mask_buf", self.mask.clone())
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self._mask_buf: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use buffer for mask so it's on same device and not trained
        w = self.weight * self._mask_buf
        out = nn.functional.linear(x, w, self.bias)
        return out

    def set_mask(self, mask: torch.Tensor) -> None:
        self._mask_buf.copy_(mask.to(dtype=self._mask_buf.dtype, device=self._mask_buf.device))

    def get_mask(self) -> torch.Tensor:
        return self._mask_buf.clone()


class MaskedMLP(nn.Module):
    """
    MLP with masked weights: input_dim -> hidden -> [hidden2 ->] output_dim.
    Minimal nonlinearity (tanh). Masks are non-trainable.
    """

    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 4,
        hidden_dims: Union[int, List[int]] = 50,
        seed: int = 0,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self._layer_dims = dims
        self._masked_layers: List[MaskedLinear] = []
        layers = []
        for i in range(len(dims) - 1):
            lin = MaskedLinear(dims[i], dims[i + 1], bias=True)
            self._masked_layers.append(lin)
            layers.append(lin)
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights(seed)

    def _init_weights(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        for m in self.modules():
            if isinstance(m, (nn.Linear, MaskedLinear)):
                if hasattr(m, "weight"):
                    nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _weighted_layers(self) -> List[MaskedLinear]:
        return [m for m in self.modules() if isinstance(m, MaskedLinear)]

    def count_edges(self) -> Tuple[List[int], int]:
        """Per-layer edge counts and total."""
        per_layer = []
        for layer in self._weighted_layers():
            n = int(layer.get_mask().sum().item())
            per_layer.append(n)
        return per_layer, sum(per_layer)

    def total_capacity(self) -> int:
        """Total number of weight parameters (max edges)."""
        return sum(l.weight.numel() for l in self._weighted_layers())

    def set_edge_budget(self, E_target: int, rng: Union[int, np.random.Generator, None] = None) -> None:
        """Set masks so total active edges equals E_target. Distribution: proportional to layer size. If rng is None, seed 0 is used; pass explicit rng for reproducibility."""
        if rng is None:
            rng = np.random.default_rng(0)
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        layers = self._weighted_layers()
        total_params = sum(l.weight.numel() for l in layers)
        if E_target > total_params:
            E_target = total_params
        # Proportional allocation per layer (floor), then assign remainder to first layer
        allocations = []
        remainder = E_target
        for l in layers:
            n = l.weight.numel()
            p = n / total_params
            alloc = int(E_target * p)
            allocations.append(min(alloc, n))
            remainder -= allocations[-1]
        # Add remainder to first layer that has room
        for i in range(len(allocations)):
            cap = layers[i].weight.numel()
            add = min(max(0, remainder), cap - allocations[i])
            allocations[i] += add
            remainder -= add
        for layer, k in zip(layers, allocations):
            mask = torch.zeros_like(layer.weight)
            flat = mask.view(-1)
            idx = rng.choice(flat.numel(), size=min(k, flat.numel()), replace=False)
            flat[idx] = 1
            layer.set_mask(mask)

    def export_masks(self) -> List[torch.Tensor]:
        """Return list of mask tensors (cpu, numpy or torch)."""
        return [l.get_mask().detach().cpu() for l in self._weighted_layers()]

    def import_masks(self, masks: List[torch.Tensor]) -> None:
        """Set masks from list of tensors (same order as layers)."""
        layers = self._weighted_layers()
        assert len(masks) == len(layers)
        for l, m in zip(layers, masks):
            l.set_mask(m.to(l._mask_buf.device))

    def list_active_edges(self, layer_id: int) -> List[Tuple[int, int]]:
        """Return list of (row, col) indices of active edges for layer layer_id."""
        layer = self._weighted_layers()[layer_id]
        m = layer.get_mask()
        rows, cols = (m > 0.5).nonzero(as_tuple=True)
        return [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]

    def list_inactive_edges(self, layer_id: int) -> List[Tuple[int, int]]:
        """Return list of (row, col) indices of inactive edges for layer layer_id."""
        layer = self._weighted_layers()[layer_id]
        m = layer.get_mask()
        rows, cols = (m < 0.5).nonzero(as_tuple=True)
        return [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]

    def get_layer_mask_tensor(self, layer_id: int) -> torch.Tensor:
        """Return the mask tensor for a layer (for in-place rewiring)."""
        return self._weighted_layers()[layer_id]._mask_buf

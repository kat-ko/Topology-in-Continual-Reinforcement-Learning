"""
Prune-and-grow operations on masked MLP: prune k edges by score, grow k edges (random or by strategy).
Enforces E_target after each full step.
"""
from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import torch

from src.models.masked_mlp import MaskedMLP


def prune_k_edges(
    model: MaskedMLP,
    layer_id: int,
    scores: torch.Tensor,
    k: int,
) -> List[Tuple[int, int]]:
    """
    Prune k active edges with lowest score. scores shape = weight shape.
    Returns list of (row, col) pruned.
    """
    layer = model._weighted_layers()[layer_id]
    mask = layer.get_mask()
    active = (mask > 0.5).nonzero(as_tuple=False)
    if len(active) <= k:
        pruned = active
    else:
        # score per active edge (lower = prune first)
        s = scores[mask > 0.5]
        idx = torch.topk(s, k, largest=False).indices
        pruned = active[idx]
    pruned_list = [(int(r), int(c)) for r, c in pruned.tolist()]
    m = model.get_layer_mask_tensor(layer_id)
    for r, c in pruned_list:
        m[r, c] = 0.0
    return pruned_list


def grow_k_edges(
    model: MaskedMLP,
    layer_id: int,
    k: int,
    rng: Union[int, np.random.Generator, None] = None,
) -> List[Tuple[int, int]]:
    """
    Grow k edges among currently inactive (uniform random).
    Returns list of (row, col) grown.
    If rng is None, a default generator is used (seed 0), so reproducibility requires passing an explicit rng.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    layer = model._weighted_layers()[layer_id]
    inactive = model.list_inactive_edges(layer_id)
    if len(inactive) < k:
        k = len(inactive)
    if k == 0:
        return []
    chosen = rng.choice(len(inactive), size=k, replace=False)
    grown = [inactive[i] for i in chosen]
    m = model.get_layer_mask_tensor(layer_id)
    for r, c in grown:
        m[r, c] = 1.0
    return grown


def apply_prune_and_grow(
    model: MaskedMLP,
    layer_id: int,
    prune_scores: torch.Tensor,
    k: int,
    rng: Union[int, np.random.Generator, None] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Prune k and grow k; returns (pruned_list, grown_list). If rng is None, seed 0 is used; pass explicit rng for reproducibility."""
    pruned = prune_k_edges(model, layer_id, prune_scores, k)
    grown = grow_k_edges(model, layer_id, k, rng)
    return pruned, grown

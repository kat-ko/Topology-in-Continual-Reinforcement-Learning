"""
Functional modularity via lesions: lesion edge subsets and measure task-specific performance drop.
Supports lesioning edges added during B, or a fraction of edges by usage.
"""
from __future__ import annotations
from typing import List, Set, Tuple
import copy
import numpy as np
import torch

from src.models.masked_mlp import MaskedMLP


def lesion_edges(
    model: MaskedMLP,
    layer_id: int,
    edges_to_remove: List[Tuple[int, int]],
) -> None:
    """Zero out mask for given (row, col) edges in layer. Modifies model in place."""
    m = model.get_layer_mask_tensor(layer_id)
    for r, c in edges_to_remove:
        m[r, c] = 0.0


def get_edges_added_during_B(
    mask_before_B: List[torch.Tensor],
    mask_after_B: List[torch.Tensor],
) -> List[List[Tuple[int, int]]]:
    """
    Compare masks: edges that are 0 before B and 1 after B = added during B.
    Returns per-layer list of (row, col).
    """
    added = []
    for ma, mb in zip(mask_before_B, mask_after_B):
        diff = (mb > 0.5) & (ma < 0.5)
        rows, cols = diff.nonzero(as_tuple=True)
        added.append([(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())])
    return added


def lesion_subset_and_evaluate(
    model: MaskedMLP,
    layer_id: int,
    edges: List[Tuple[int, int]],
    eval_fn,
    device: torch.device,
) -> float:
    """Temporarily lesion edges, run eval_fn(model), restore mask; return metric."""
    m = model.get_layer_mask_tensor(layer_id)
    backup = m.clone()
    for r, c in edges:
        m[r, c] = 0.0
    try:
        score = eval_fn(model, device)
    finally:
        m.copy_(backup)
    return score

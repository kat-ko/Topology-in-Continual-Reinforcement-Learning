"""
Prune-and-grow rewiring rule: every R steps, prune k=floor(p*active) lowest-score edges, grow k random inactive.
Prune score = alpha_w*|w| + alpha_gabs*ema(|grad|) + alpha_gsq*ema(grad^2).
Genotype: E_target, R, p, alpha_w, alpha_gabs, alpha_gsq.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from src.models.masked_mlp import MaskedMLP
from src.rewiring.signals import RewiringSignals
from src.rewiring.mask_ops import apply_prune_and_grow


@dataclass
class RewiringGenotype:
    E_target: int
    R: int
    p: float
    alpha_w: float
    alpha_gabs: float
    alpha_gsq: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.E_target),
            float(self.R),
            self.p,
            self.alpha_w,
            self.alpha_gabs,
            self.alpha_gsq,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "RewiringGenotype":
        return cls(
            E_target=int(round(v[0])),
            R=max(1, int(round(v[1]))),
            p=float(np.clip(v[2], 1e-4, 1.0)),
            alpha_w=float(max(0, v[3])),
            alpha_gabs=float(max(0, v[4])),
            alpha_gsq=float(max(0, v[5])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_rewiring_step(
    model: MaskedMLP,
    signals: RewiringSignals,
    genotype: RewiringGenotype,
    step: int,
    rng: Union[int, np.random.Generator, None],
    log: Optional[Dict[str, List]] = None,
) -> int:
    """
    If step % R == 0, perform prune-and-grow for each layer (proportional k per layer).
    Returns total turnover count.
    If rng is None, a default generator is used (step-based seed); pass explicit rng for reproducibility.
    """
    if step <= 0 or step % genotype.R != 0:
        return 0
    if rng is None:
        rng = np.random.default_rng(step)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    total_turnover = 0
    per_layer = model._weighted_layers()
    _, total_edges = model.count_edges()
    if total_edges == 0:
        return 0
    k_total = max(1, int(genotype.p * total_edges))
    # Distribute k across layers proportionally
    n_per_layer = []
    remainder = k_total
    for layer in per_layer:
        n_active = int(layer.get_mask().sum().item())
        p = n_active / total_edges
        ki = int(k_total * p)
        ki = min(ki, n_active)
        n_per_layer.append(ki)
        remainder -= ki
    for i in range(min(remainder, len(per_layer))):
        n_per_layer[i] += 1
    for layer_id, k in enumerate(n_per_layer):
        if k <= 0:
            continue
        layer = per_layer[layer_id]
        w = layer.weight.detach().abs()
        ga = signals.get_ema_grad_abs(layer_id)
        gs = signals.get_ema_grad_sq(layer_id)
        prune_score = (
            genotype.alpha_w * w
            + genotype.alpha_gabs * ga
            + genotype.alpha_gsq * gs
        )
        pruned, grown = apply_prune_and_grow(model, layer_id, prune_score, k, rng)
        total_turnover += len(pruned) + len(grown)
        if log is not None:
            log.setdefault("pruned", []).append((layer_id, pruned))
            log.setdefault("grown", []).append((layer_id, grown))
    if log is not None:
        log.setdefault("turnover", []).append(total_turnover)
    return total_turnover


def apply_random_rewiring_step(
    model: MaskedMLP,
    genotype: RewiringGenotype,
    step: int,
    rng: Union[int, np.random.Generator, None],
    log: Optional[Dict[str, List]] = None,
) -> int:
    """Same schedule as apply_rewiring_step but prune/grow by random selection (no scores). If rng is None, step-based seed is used; pass explicit rng for reproducibility."""
    if step <= 0 or step % genotype.R != 0:
        return 0
    if rng is None:
        rng = np.random.default_rng(step)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    from src.rewiring.mask_ops import prune_k_edges, grow_k_edges
    total_turnover = 0
    per_layer = model._weighted_layers()
    _, total_edges = model.count_edges()
    if total_edges == 0:
        return 0
    k_total = max(1, int(genotype.p * total_edges))
    n_per_layer = []
    remainder = k_total
    for layer in per_layer:
        n_active = int(layer.get_mask().sum().item())
        p = n_active / total_edges
        ki = min(int(k_total * p), n_active)
        n_per_layer.append(ki)
        remainder -= ki
    for i in range(min(remainder, len(per_layer))):
        n_per_layer[i] += 1
    for layer_id, k in enumerate(n_per_layer):
        if k <= 0:
            continue
        layer = per_layer[layer_id]
        # Random scores: shuffle so we prune "lowest" randomly
        scores = rng.random(layer.weight.shape)
        pruned = prune_k_edges(model, layer_id, torch.from_numpy(scores).to(layer.weight.device), k)
        grown = grow_k_edges(model, layer_id, k, rng)
        total_turnover += len(pruned) + len(grown)
    if log is not None:
        log.setdefault("turnover", []).append(total_turnover)
    return total_turnover


def make_rewiring_hook(
    model: MaskedMLP,
    signals: RewiringSignals,
    genotype: RewiringGenotype,
    rng: Union[int, np.random.Generator, None],
    log: Optional[Dict[str, List]] = None,
):
    """Return a callable rewiring_hook(model, step, phase) for use in inner_loop."""
    def hook(m, step: int, phase: str) -> int:
        return apply_rewiring_step(m, signals, genotype, step, rng, log)
    return hook

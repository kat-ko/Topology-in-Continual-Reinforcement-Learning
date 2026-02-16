"""
Evaluate one individual: run inner_loop n_seeds x n_task_samples times; aggregate obj1=performance, obj2=E_target.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch

from src.tasks.season_task import generate_stream
from src.models.masked_mlp import MaskedMLP
from src.rewiring.signals import RewiringSignals
from src.rewiring.rewiring_rule import apply_rewiring_step
from src.training.inner_loop import run_inner_loop
from src.analysis.metrics import continual_learning_score
from src.evolution.individual import Individual
from src.utils.seeding import set_seed


def evaluate_individual(
    ind: Individual,
    shift_angle_rad: float,
    n_steps_a1: int,
    n_steps_b: int,
    n_steps_a2: int,
    hidden: int,
    lr: float,
    n_seeds: int,
    n_task_samples: int,
    E_min: int,
    E_max: int,
    device: str,
    base_seed: int = 0,
    wB: float = 1.0,
    wA: float = 1.0,
) -> tuple:
    """
    Run inner loop n_seeds * n_task_samples times; return (mean_performance, mean_cost).
    Cost = E_target (same for all runs of this individual).
    """
    perf_list = []
    g = ind.genotype
    E = int(np.clip(g.E_target, E_min, E_max))
    for run_idx in range(n_seeds * n_task_samples):
        seed = base_seed + run_idx * 1000
        set_seed(seed)
        trials_a1, trials_b, trials_a2, _ = generate_stream(
            seed, shift_angle_rad, n_steps_a1, n_steps_b, n_steps_a2, include_test_trials=False
        )
        model = MaskedMLP(input_dim=12, output_dim=4, hidden_dims=hidden, seed=seed + 1).to(device)
        model.set_edge_budget(E, rng=np.random.default_rng(seed + 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        signals = RewiringSignals(model, decay=0.99)
        rng = np.random.default_rng(seed + 3)

        def hook(m, step, phase):
            return apply_rewiring_step(m, signals, g, step, rng, None)

        def sig_update(m):
            signals.update(m)

        results = run_inner_loop(
            model,
            trials_a1,
            trials_b,
            trials_a2,
            optimizer,
            torch.device(device),
            rewiring_hook=hook,
            signals_update=sig_update,
            seed=seed,
        )
        score = continual_learning_score(results, wB=wB, wA=wA)
        perf_list.append(score)
    perf_arr = np.array(perf_list)
    if not np.all(np.isfinite(perf_arr)):
        bad_idx = np.where(~np.isfinite(perf_arr))[0]
        raise ValueError(
            "evaluate_individual: non-finite performance score at run index(s) {} "
            "(check inner_loop and trial generation)".format(bad_idx.tolist())
        )
    obj1 = float(np.mean(perf_list))
    obj2 = float(E)
    return obj1, obj2

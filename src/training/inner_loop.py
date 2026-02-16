"""
Inner-loop training: A1 -> B -> A2 with optional rewiring hook.
Returns metrics (A_before_B, B_final, A_after_B, retention) and structural logs.
"""
from __future__ import annotations
import warnings
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from src.tasks.season_task import (
    Trial,
    accuracy,
    response_angle_from_cos_sin,
    N_STIM,
    INPUT_DIM,
    OUTPUT_DIM,
)
from src.models.masked_mlp import MaskedMLP


def validate_inner_loop_results(results: Dict[str, Any]) -> None:
    """
    Raise ValueError if key metrics are not finite or are outside [0, 1].
    Call after run_inner_loop to catch silent bad numbers.
    """
    for key in ("A1_final", "B_final", "A_after_B"):
        val = results.get(key)
        if val is None:
            raise ValueError("validate_inner_loop_results: missing key '{}'".format(key))
        if not np.isfinite(val):
            raise ValueError(
                "validate_inner_loop_results: '{}' is not finite (got {})".format(key, val)
            )
        if not (0 <= val <= 1):
            raise ValueError(
                "validate_inner_loop_results: '{}' should be in [0, 1], got {}".format(key, val)
            )


def trials_to_tensors(trials: List[Trial], device: torch.device):
    inputs = np.stack([t.input_vec for t in trials], axis=0)
    targets = np.stack([t.target for t in trials], axis=0)
    probes = np.array([t.feature_probe for t in trials], dtype=np.int64)
    return (
        torch.from_numpy(inputs).float().to(device),
        torch.from_numpy(targets).float().to(device),
        torch.from_numpy(probes).long().to(device),
    )


def compute_phase_accuracy(
    model: nn.Module,
    trials: List[Trial],
    device: torch.device,
    probe_feature: int = 1,
) -> float:
    if not trials:
        raise ValueError("compute_phase_accuracy: trials list is empty")
    inputs, targets, probes = trials_to_tensors(trials, device)
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    idx = 2 * probe_feature
    pred_angle = response_angle_from_cos_sin(
        out[:, idx].cpu().numpy(), out[:, idx + 1].cpu().numpy()
    )
    tgt_angle = response_angle_from_cos_sin(
        targets[:, idx].cpu().numpy(), targets[:, idx + 1].cpu().numpy()
    )
    mask = probes.cpu().numpy() == probe_feature
    if mask.sum() == 0:
        raise ValueError(
            "compute_phase_accuracy: no trials with probe_feature={}".format(probe_feature)
        )
    acc = accuracy(pred_angle[mask], tgt_angle[mask])
    return float(np.mean(acc))


def run_inner_loop(
    model: MaskedMLP,
    trials_a1: List[Trial],
    trials_b: List[Trial],
    trials_a2: List[Trial],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rewiring_hook: Optional[Callable[[MaskedMLP, int, str], int]] = None,
    signals_update: Optional[Callable[[MaskedMLP], None]] = None,
    skip_test_updates: bool = True,
    eval_last_k: int = 6 * N_STIM,
    seed: int = 0,
    capture_masks_at_phase_end: bool = False,
) -> Dict[str, Any]:
    """
    Train A1, then B, then evaluate A2 (no A2 training in v1).
    Step count per phase = len(trials_a1), len(trials_b), len(trials_a2) respectively.
    If rewiring_hook is provided, call it each step after signals_update and before optimizer.step().
    eval_last_k: number of trials used for A1_final and B_final; if a phase has fewer trials, all are used.
    """
    torch.manual_seed(seed)
    criterion = nn.MSELoss()
    results = {
        "A1_final": None,
        "B_final": None,
        "A_after_B": None,
        "retention": None,
        "learning_curve_a1": [],
        "learning_curve_b": [],
        "edge_count_over_time": [],
        "turnover_events": [],
        "masks_at_phase_end": {},
    }
    step = 0

    def train_phase(trials: List[Trial], phase: str) -> float:
        nonlocal step
        model.train()
        inputs, targets, probes = trials_to_tensors(trials, device)
        perm = torch.randperm(inputs.size(0), device=device)
        inputs = inputs[perm]
        targets = targets[perm]
        probes = probes[perm]
        losses = []
        for i in range(inputs.size(0)):
            inp = inputs[i : i + 1]
            tgt = targets[i : i + 1]
            prb = probes[i].item()
            optimizer.zero_grad()
            out = model(inp)
            idx = 2 * prb
            loss = criterion(out[:, idx : idx + 2], tgt[:, idx : idx + 2])
            loss.backward()
            if signals_update is not None:
                signals_update(model)
            if rewiring_hook is not None:
                turnover = rewiring_hook(model, step, phase)
                if turnover > 0:
                    results["turnover_events"].append({"step": step, "phase": phase, "turnover": turnover})
            optimizer.step()
            step += 1
            losses.append(loss.item())
            if rewiring_hook is not None or step % 100 == 0:
                _, total_edges = model.count_edges()
                results["edge_count_over_time"].append({"step": step, "edges": total_edges})
        if phase == "A1":
            results["learning_curve_a1"].extend(losses)
        elif phase == "B":
            results["learning_curve_b"].extend(losses)
        return np.mean(losses) if losses else 0.0

    # A1
    train_phase(trials_a1, "A1")
    if capture_masks_at_phase_end:
        results["masks_at_phase_end"]["A1"] = [m.get_mask().clone().cpu() for m in model._weighted_layers()]
    if len(trials_a1) < eval_last_k:
        warnings.warn(
            "Phase A1 has fewer than eval_last_k trials; using all {} for final accuracy.".format(
                len(trials_a1)
            )
        )
    results["A1_final"] = compute_phase_accuracy(
        model, trials_a1[-eval_last_k:] if len(trials_a1) >= eval_last_k else trials_a1, device
    )
    # B
    train_phase(trials_b, "B")
    if capture_masks_at_phase_end:
        results["masks_at_phase_end"]["B"] = [m.get_mask().clone().cpu() for m in model._weighted_layers()]
    if len(trials_b) < eval_last_k:
        warnings.warn(
            "Phase B has fewer than eval_last_k trials; using all {} for final accuracy.".format(
                len(trials_b)
            )
        )
    results["B_final"] = compute_phase_accuracy(
        model, trials_b[-eval_last_k:] if len(trials_b) >= eval_last_k else trials_b, device
    )
    # A2 eval only
    results["A_after_B"] = compute_phase_accuracy(model, trials_a2, device)
    results["retention"] = results["A_after_B"] - results["A1_final"]
    validate_inner_loop_results(results)
    return results

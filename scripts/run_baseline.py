"""
Phase 1 baseline: train a fixed fully-connected MLP on A1->B->A2 for same/near/far.
Reproduces qualitative same/near/far differences (Holton direction: far worse retention).
"""
import os
import sys
import argparse
import json
from typing import Optional

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.season_task import (
    Similarity,
    generate_stream,
    accuracy,
    response_angle_from_cos_sin,
    N_STIM,
    INPUT_DIM,
    OUTPUT_DIM,
)
from src.utils.seeding import set_seed
from src.utils.config import make_run_dir, save_config
from src.utils.logging import save_metrics
from src.models.masked_mlp import MaskedMLP


class SimpleMLP(nn.Module):
    """Fixed fully-connected MLP: 12 -> 50 -> 4 with tanh (minimal nonlinearity)."""
    def __init__(self, hidden: int = 50, seed: int = 0):
        super().__init__()
        set_seed(seed)
        self.fc1 = nn.Linear(INPUT_DIM, hidden)
        self.fc2 = nn.Linear(hidden, OUTPUT_DIM)
        self.act = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


def trials_to_tensors(trials, device):
    """Convert list of Trial to (inputs, targets, feature_probes)."""
    inputs = np.stack([t.input_vec for t in trials], axis=0)
    targets = np.stack([t.target for t in trials], axis=0)
    probes = np.array([t.feature_probe for t in trials], dtype=np.int64)
    return (
        torch.from_numpy(inputs).float().to(device),
        torch.from_numpy(targets).float().to(device),
        torch.from_numpy(probes).long().to(device),
    )


def compute_phase_accuracy(model, trials, device, probe_feature: int = 1):
    """Mean accuracy over trials where feature_probe == probe_feature (winter=1)."""
    if not trials:
        return float("nan")
    inputs, targets, probes = trials_to_tensors(trials, device)
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    # probed feature: 0 -> out[:, 0:2], 1 -> out[:, 2:4]
    idx = 2 * probe_feature
    pred_angle = response_angle_from_cos_sin(
        out[:, idx].cpu().numpy(),
        out[:, idx + 1].cpu().numpy(),
    )
    tgt_angle = response_angle_from_cos_sin(
        targets[:, idx].cpu().numpy(),
        targets[:, idx + 1].cpu().numpy(),
    )
    mask = (probes.cpu().numpy() == probe_feature)
    if mask.sum() == 0:
        return float("nan")
    acc = accuracy(pred_angle[mask], tgt_angle[mask])
    return float(np.mean(acc))


def train_phase(model, optimizer, trials, device):
    """One phase training. Loss = MSE on probed feature cos/sin."""
    model.train()
    inputs, targets, probes = trials_to_tensors(trials, device)
    criterion = nn.MSELoss()
    perm = torch.randperm(inputs.size(0), device=device)
    inputs = inputs[perm]
    targets = targets[perm]
    probes = probes[perm]
    total_loss = 0.0
    for i in range(inputs.size(0)):
        inp = inputs[i : i + 1]
        tgt = targets[i : i + 1]
        prb = probes[i].item()
        optimizer.zero_grad()
        out = model(inp)
        idx = 2 * prb
        loss = criterion(out[:, idx : idx + 2], tgt[:, idx : idx + 2])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(inputs.size(0), 1)


def run_baseline_one_similarity(
    similarity: Similarity,
    seed: int,
    n_steps_a1: int,
    n_steps_b: int,
    n_steps_a2: int,
    hidden: int,
    lr: float,
    device: str,
    use_masked: bool = False,
    E_target: Optional[int] = None,
) -> dict:
    shift = similarity.value
    trials_a1, trials_b, trials_a2, _ = generate_stream(
        seed, shift, n_steps_a1, n_steps_b, n_steps_a2, include_test_trials=False
    )
    set_seed(seed + 100)
    if use_masked:
        model = MaskedMLP(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=hidden, seed=seed + 100).to(device)
        E = model.total_capacity() if E_target is None else min(E_target, model.total_capacity())
        model.set_edge_budget(E, rng=seed + 101)
    else:
        model = SimpleMLP(hidden=hidden, seed=seed + 100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_phase(model, optimizer, trials_a1, device)
    a1_final = compute_phase_accuracy(model, trials_a1[-N_STIM * 6 :], device)

    train_phase(model, optimizer, trials_b, device)
    b_final = compute_phase_accuracy(model, trials_b[-N_STIM * 6 :], device)

    a2_acc = compute_phase_accuracy(model, trials_a2, device)
    retention = a2_acc - a1_final

    return {
        "similarity": similarity.name.lower(),
        "shift_rad": float(shift),
        "A1_final": a1_final,
        "B_final": b_final,
        "A_after_B": a2_acc,
        "retention": retention,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps-a1", type=int, default=50)
    parser.add_argument("--n-steps-b", type=int, default=50)
    parser.add_argument("--n-steps-a2", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--use-masked", action="store_true", help="Use MaskedMLP (fixed topology or static sparse)")
    parser.add_argument("--E-target", type=int, default=None, help="Edge budget for masked MLP (static sparse); default=full capacity")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(args.results_dir)
    config = {
        "script": "run_baseline",
        "seed": args.seed,
        "n_steps_a1": args.n_steps_a1,
        "n_steps_b": args.n_steps_b,
        "n_steps_a2": args.n_steps_a2,
        "hidden": args.hidden,
        "lr": args.lr,
        "use_masked": args.use_masked,
        "E_target": args.E_target,
    }
    save_config(config, os.path.join(run_dir, "config.json"))

    metrics_by_similarity = []
    for sim in [Similarity.SAME, Similarity.NEAR, Similarity.FAR]:
        m = run_baseline_one_similarity(
            sim, args.seed, args.n_steps_a1, args.n_steps_b, args.n_steps_a2,
            args.hidden, args.lr, device, use_masked=args.use_masked, E_target=args.E_target,
        )
        metrics_by_similarity.append(m)
        print(f"{sim.name}: A1_final={m['A1_final']:.4f}, B_final={m['B_final']:.4f}, A_after_B={m['A_after_B']:.4f}, retention={m['retention']:.4f}")

    metrics = {"by_similarity": metrics_by_similarity}
    save_metrics(metrics, os.path.join(run_dir, "metrics.json"))
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()

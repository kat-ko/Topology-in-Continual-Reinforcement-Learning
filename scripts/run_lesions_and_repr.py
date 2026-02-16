"""
Run lesion and representation analysis on selected solutions (e.g. from Pareto).
Optional: freeze rewiring after A1; ablate edges added during B.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.tasks.season_task import generate_stream, Similarity
from src.models.masked_mlp import MaskedMLP
from src.rewiring.signals import RewiringSignals
from src.rewiring.rewiring_rule import RewiringGenotype, apply_rewiring_step
from src.training.inner_loop import run_inner_loop, compute_phase_accuracy, trials_to_tensors
from src.analysis.lesions import get_edges_added_during_B, lesion_subset_and_evaluate
from src.analysis.repr import collect_hidden_activations, representation_similarity
from src.utils.seeding import set_seed
from src.utils.config import make_run_dir, save_config
from src.utils.logging import save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity", choices=["same", "near", "far"], default="near")
    parser.add_argument("--E-target", type=int, default=400)
    parser.add_argument("--freeze-rewiring-after-a1", action="store_true")
    parser.add_argument("--ablate-B-edges", action="store_true")
    parser.add_argument("--n-steps-a1", type=int, default=20)
    parser.add_argument("--n-steps-b", type=int, default=20)
    parser.add_argument("--n-steps-a2", type=int, default=6)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shift = {"same": 0.0, "near": np.pi / 6, "far": np.pi}[args.similarity]

    trials_a1, trials_b, trials_a2, _ = generate_stream(
        args.seed, shift, args.n_steps_a1, args.n_steps_b, args.n_steps_a2, include_test_trials=False
    )
    model = MaskedMLP(input_dim=12, output_dim=4, hidden_dims=50, seed=args.seed).to(device)
    model.set_edge_budget(args.E_target, rng=np.random.default_rng(args.seed + 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    signals = RewiringSignals(model, decay=0.99)
    genotype = RewiringGenotype(E_target=args.E_target, R=20, p=0.1, alpha_w=0.1, alpha_gabs=0.5, alpha_gsq=0.5)
    rng = np.random.default_rng(args.seed + 2)

    if args.freeze_rewiring_after_a1:
        def hook(m, step, phase):
            if phase == "B" or phase == "A2":
                return 0
            return apply_rewiring_step(m, signals, genotype, step, rng, None)
    else:
        def hook(m, step, phase):
            return apply_rewiring_step(m, signals, genotype, step, rng, None)

    results = run_inner_loop(
        model, trials_a1, trials_b, trials_a2, optimizer, device,
        rewiring_hook=hook, signals_update=signals.update, seed=args.seed,
        capture_masks_at_phase_end=True,
    )
    masks_at_end = results.get("masks_at_phase_end", {})
    masks_before_B = masks_at_end.get("A1", [m.get_mask().clone() for m in model._weighted_layers()])
    masks_after_B = masks_at_end.get("B", [m.get_mask().clone() for m in model._weighted_layers()])

    # Representation similarity: probe on A vs B stimuli
    probe_A = trials_to_tensors(trials_a1[-12:], device)[0]
    probe_B = trials_to_tensors(trials_b[-12:], device)[0]
    with torch.no_grad():
        acts_A = collect_hidden_activations(model, probe_A)
        acts_B = collect_hidden_activations(model, probe_B)
    cka_per_layer = representation_similarity(acts_A[:-1], acts_B[:-1])
    metrics = {
        "A1_final": results["A1_final"],
        "B_final": results["B_final"],
        "A_after_B": results["A_after_B"],
        "retention": results["retention"],
        "CKA_per_layer": cka_per_layer,
    }

    if args.ablate_B_edges:
        added = get_edges_added_during_B(masks_before_B, masks_after_B)
        def eval_fn(m, dev):
            return compute_phase_accuracy(m, trials_a2, dev)
        baseline_score = eval_fn(model, device)
        metrics["ablate_B_edges_baseline"] = baseline_score
        lesioned_scores = []
        for layer_id, edges in enumerate(added):
            if not edges:
                lesioned_scores.append(baseline_score)
                continue
            s = lesion_subset_and_evaluate(model, layer_id, edges, eval_fn, device)
            lesioned_scores.append(s)
        metrics["ablate_B_edges_per_layer_drop"] = [baseline_score - s for s in lesioned_scores]

    run_dir = make_run_dir(args.results_dir)
    save_config(vars(args), os.path.join(run_dir, "config.json"))
    save_metrics(metrics, os.path.join(run_dir, "metrics.json"))
    print("Results:", run_dir)
    print("CKA per layer:", cka_per_layer)


if __name__ == "__main__":
    main()

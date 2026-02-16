"""
Phase 3 debug: run inner loop with hand-coded rewiring rule.
Sanity: edge count == E_target after rewiring; turnover > 0 when rewiring enabled.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.tasks.season_task import Similarity, generate_stream
from src.models.masked_mlp import MaskedMLP
from src.rewiring.signals import RewiringSignals
from src.rewiring.rewiring_rule import RewiringGenotype, apply_rewiring_step, apply_random_rewiring_step
from src.training.inner_loop import run_inner_loop
from src.utils.seeding import set_seed
from src.utils.config import make_run_dir, save_config
from src.utils.logging import save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity", choices=["same", "near", "far"], default="near")
    parser.add_argument("--n-steps-a1", type=int, default=30)
    parser.add_argument("--n-steps-b", type=int, default=30)
    parser.add_argument("--n-steps-a2", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--E-target", type=int, default=400)
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--no-rewiring", action="store_true")
    parser.add_argument("--random-rewiring", action="store_true", help="Same E,R,p but random prune/grow")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shift = {"same": 0.0, "near": np.pi / 6, "far": np.pi}[args.similarity]

    trials_a1, trials_b, trials_a2, _ = generate_stream(
        args.seed, shift, args.n_steps_a1, args.n_steps_b, args.n_steps_a2, include_test_trials=False
    )
    rng = np.random.default_rng(args.seed + 200)
    model = MaskedMLP(input_dim=12, output_dim=4, hidden_dims=args.hidden, seed=args.seed + 100).to(device)
    model.set_edge_budget(args.E_target, rng=rng)
    _, total_edges = model.count_edges()
    assert total_edges == args.E_target, f"Initial edges {total_edges} != E_target {args.E_target}"

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    rewiring_hook = None
    signals_update = None
    log = {"turnover": [], "pruned": [], "grown": []}

    if not args.no_rewiring:
        signals = RewiringSignals(model, decay=0.99)
        genotype = RewiringGenotype(
            E_target=args.E_target,
            R=args.R,
            p=args.p,
            alpha_w=0.1,
            alpha_gabs=0.5,
            alpha_gsq=0.5,
        )
        if args.random_rewiring:
            def hook(m, step, phase):
                return apply_random_rewiring_step(m, genotype, step, rng, log)
            rewiring_hook = hook
            signals_update = None
        else:
            def hook(m, step, phase):
                return apply_rewiring_step(m, signals, genotype, step, rng, log)
            rewiring_hook = hook
            signals_update = signals.update

    results = run_inner_loop(
        model,
        trials_a1,
        trials_b,
        trials_a2,
        optimizer,
        device,
        rewiring_hook=rewiring_hook,
        signals_update=signals_update,
        seed=args.seed,
    )

    # Sanity: after run, edges still E_target
    _, total_edges_after = model.count_edges()
    assert total_edges_after == args.E_target, f"Final edges {total_edges_after} != E_target {args.E_target}"
    print(f"Edge count sanity: {total_edges_after} == {args.E_target} OK")

    if not args.no_rewiring:
        total_turnover = sum(log["turnover"])
        assert total_turnover > 0, "Expected turnover > 0 when rewiring enabled"
        print(f"Turnover sanity: total turnover {total_turnover} > 0 OK")

    print(f"A1_final={results['A1_final']:.4f}, B_final={results['B_final']:.4f}, A_after_B={results['A_after_B']:.4f}, retention={results['retention']:.4f}")

    run_dir = make_run_dir(args.results_dir)
    save_config(
        {
            "script": "run_inner_rewiring_debug",
            "seed": args.seed,
            "similarity": args.similarity,
            "E_target": args.E_target,
            "R": args.R,
            "p": args.p,
            "no_rewiring": args.no_rewiring,
        },
        os.path.join(run_dir, "config.json"),
    )
    save_metrics(
        {k: v for k, v in results.items() if k != "edge_count_over_time" and k != "turnover_events"},
        os.path.join(run_dir, "metrics.json"),
    )
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()

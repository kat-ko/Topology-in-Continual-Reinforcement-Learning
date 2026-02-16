"""
Run NSGA-II evolution for one similarity condition. Saves population.csv and pareto.csv.
"""
import os
import sys
import argparse
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.tasks.season_task import Similarity
from src.evolution.individual import Individual, mutate
from src.evolution.evaluator import evaluate_individual
from src.evolution.nsga2 import non_dominated_sort, evolve_population
from src.rewiring.rewiring_rule import RewiringGenotype
from src.utils.seeding import set_seed
from src.utils.config import make_run_dir, save_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity", choices=["same", "near", "far"], default="near")
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--E-min", type=int, default=200)
    parser.add_argument("--E-max", type=int, default=800)
    parser.add_argument("--n-steps-a1", type=int, default=20)
    parser.add_argument("--n-steps-b", type=int, default=20)
    parser.add_argument("--n-steps-a2", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-task-samples", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    shift = {"same": 0.0, "near": np.pi / 6, "far": np.pi}[args.similarity]
    rng = np.random.default_rng(args.seed)

    run_dir = make_run_dir(args.results_dir)
    save_config(vars(args), os.path.join(run_dir, "config.json"))

    # Initial population: random genotypes
    population = []
    for _ in range(args.pop_size):
        E = rng.integers(args.E_min, args.E_max + 1)
        g = RewiringGenotype(
            E_target=E,
            R=max(1, int(rng.integers(10, 100))),
            p=float(rng.uniform(0.05, 0.3)),
            alpha_w=float(rng.uniform(0.01, 0.5)),
            alpha_gabs=float(rng.uniform(0.01, 0.5)),
            alpha_gsq=float(rng.uniform(0.01, 0.5)),
        )
        population.append(Individual(genotype=g))

    for gen in range(args.generations):
        # Evaluate
        for ind in population:
            if ind.objectives[0] == 0.0 and ind.objectives[1] == 0.0:
                o1, o2 = evaluate_individual(
                    ind,
                    shift,
                    args.n_steps_a1,
                    args.n_steps_b,
                    args.n_steps_a2,
                    args.hidden,
                    args.lr,
                    args.n_seeds,
                    args.n_task_samples,
                    args.E_min,
                    args.E_max,
                    device,
                    base_seed=args.seed + gen * 10000,
                )
                ind.objectives = (o1, o2)
        # Offspring: mutate each
        offspring = [mutate(p, args.E_min, args.E_max, sigma=0.2, rng=rng) for p in population]
        for ind in offspring:
            o1, o2 = evaluate_individual(
                ind,
                shift,
                args.n_steps_a1,
                args.n_steps_b,
                args.n_steps_a2,
                args.hidden,
                args.lr,
                args.n_seeds,
                args.n_task_samples,
                args.E_min,
                args.E_max,
                device,
                base_seed=args.seed + gen * 10000 + 5000,
            )
            ind.objectives = (o1, o2)
        population = evolve_population(population, offspring, args.E_min, args.E_max, 0.2, rng)
        print(f"Gen {gen}: pop size {len(population)}, fronts computed")

    # Save population
    rows = []
    for i, ind in enumerate(population):
        g = ind.genotype
        rows.append({
            "id": i,
            "E_target": g.E_target,
            "R": g.R,
            "p": g.p,
            "alpha_w": g.alpha_w,
            "alpha_gabs": g.alpha_gabs,
            "alpha_gsq": g.alpha_gsq,
            "obj1_performance": ind.objectives[0],
            "obj2_cost": ind.objectives[1],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "population.csv"), index=False)

    # Pareto front (front 0)
    fronts = non_dominated_sort(population)
    pareto = fronts[0] if fronts else []
    pareto_rows = []
    for i, ind in enumerate(pareto):
        g = ind.genotype
        pareto_rows.append({
            "E_target": g.E_target,
            "R": g.R,
            "p": g.p,
            "alpha_w": g.alpha_w,
            "alpha_gabs": g.alpha_gabs,
            "alpha_gsq": g.alpha_gsq,
            "obj1_performance": ind.objectives[0],
            "obj2_cost": ind.objectives[1],
        })
    pd.DataFrame(pareto_rows).to_csv(os.path.join(run_dir, "pareto.csv"), index=False)
    print(f"Pareto front: {len(pareto)} solutions. Results in {run_dir}")


if __name__ == "__main__":
    main()

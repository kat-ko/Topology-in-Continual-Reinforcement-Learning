"""
Run evolution separately for same, near, far; save Pareto fronts per condition for comparison.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.evolution.individual import Individual, mutate
from src.evolution.evaluator import evaluate_individual
from src.evolution.nsga2 import non_dominated_sort, evolve_population
from src.rewiring.rewiring_rule import RewiringGenotype
from src.utils.seeding import set_seed
from src.utils.config import make_run_dir, save_config


def run_one_similarity(
    similarity: str,
    shift_angle: float,
    pop_size: int,
    generations: int,
    E_min: int,
    E_max: int,
    n_steps_a1: int,
    n_steps_b: int,
    n_steps_a2: int,
    n_seeds: int,
    n_task_samples: int,
    hidden: int,
    lr: float,
    base_seed: int,
    results_dir: str,
    device: str,
) -> str:
    """Run evolution for one similarity; return run_dir."""
    rng = np.random.default_rng(base_seed)
    run_dir = make_run_dir(results_dir)
    save_config(
        {"script": "run_similarity_sweep", "similarity": similarity, "shift_angle": shift_angle, "pop_size": pop_size, "generations": generations},
        os.path.join(run_dir, "config.json"),
    )
    population = []
    for _ in range(pop_size):
        E = rng.integers(E_min, E_max + 1)
        g = RewiringGenotype(
            E_target=E,
            R=max(1, int(rng.integers(10, 100))),
            p=float(rng.uniform(0.05, 0.3)),
            alpha_w=float(rng.uniform(0.01, 0.5)),
            alpha_gabs=float(rng.uniform(0.01, 0.5)),
            alpha_gsq=float(rng.uniform(0.01, 0.5)),
        )
        population.append(Individual(genotype=g))
    for gen in range(generations):
        for ind in population:
            if ind.objectives[0] == 0.0 and ind.objectives[1] == 0.0:
                o1, o2 = evaluate_individual(
                    ind, shift_angle, n_steps_a1, n_steps_b, n_steps_a2,
                    hidden, lr, n_seeds, n_task_samples, E_min, E_max, device,
                    base_seed=base_seed + gen * 10000,
                )
                ind.objectives = (o1, o2)
        offspring = [mutate(p, E_min, E_max, sigma=0.2, rng=rng) for p in population]
        for ind in offspring:
            o1, o2 = evaluate_individual(
                ind, shift_angle, n_steps_a1, n_steps_b, n_steps_a2,
                hidden, lr, n_seeds, n_task_samples, E_min, E_max, device,
                base_seed=base_seed + gen * 10000 + 5000,
            )
            ind.objectives = (o1, o2)
        population = evolve_population(population, offspring, E_min, E_max, 0.2, rng)
    fronts = non_dominated_sort(population)
    pareto = fronts[0] if fronts else []
    rows = [{"E_target": ind.genotype.E_target, "R": ind.genotype.R, "p": ind.genotype.p,
             "obj1_performance": ind.objectives[0], "obj2_cost": ind.objectives[1]} for ind in pareto]
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "pareto.csv"), index=False)
    pd.DataFrame([{"id": i, "E_target": ind.genotype.E_target, "obj1": ind.objectives[0], "obj2": ind.objectives[1]}
                  for i, ind in enumerate(population)]).to_csv(os.path.join(run_dir, "population.csv"), index=False)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--E-min", type=int, default=200)
    parser.add_argument("--E-max", type=int, default=600)
    parser.add_argument("--n-steps-a1", type=int, default=15)
    parser.add_argument("--n-steps-b", type=int, default=15)
    parser.add_argument("--n-steps-a2", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-task-samples", type=int, default=2)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    conditions = [("same", 0.0), ("near", np.pi / 6), ("far", np.pi)]
    dirs = []
    for sim, shift in conditions:
        d = run_one_similarity(
            sim, shift, args.pop_size, args.generations, args.E_min, args.E_max,
            args.n_steps_a1, args.n_steps_b, args.n_steps_a2, args.n_seeds, args.n_task_samples,
            50, 0.01, args.seed, args.results_dir, device,
        )
        dirs.append((sim, d))
        print(f"{sim}: {d}")
    print("Sweep done. Pareto fronts:", [d for _, d in dirs])


if __name__ == "__main__":
    main()

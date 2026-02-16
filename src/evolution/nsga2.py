"""
NSGA-II: non-dominated sort, crowding distance, selection, mutation.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from src.evolution.individual import Individual, mutate, crossover


def dominates(a: Individual, b: Individual) -> bool:
    """a dominates b iff a is no worse on all objectives and strictly better on at least one."""
    oa, ob = a.objectives, b.objectives
    # Maximize obj1 (performance), minimize obj2 (cost)
    better1 = oa[0] >= ob[0]
    better2 = oa[1] <= ob[1]
    strict1 = oa[0] > ob[0]
    strict2 = oa[1] < ob[1]
    return (better1 and better2) and (strict1 or strict2)


def non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Return list of fronts (front 0 = nondominated)."""
    fronts = []
    remaining = list(population)
    while remaining:
        front = []
        for p in remaining:
            if not any(dominates(q, p) for q in remaining if q is not p):
                front.append(p)
        fronts.append(front)
        for p in front:
            remaining.remove(p)
    return fronts


def crowding_distance(front: List[Individual], obj_indices: Tuple[int, int] = (0, 1)) -> None:
    """Assign crowding distance in-place. obj_indices: (maximize, minimize)."""
    if len(front) <= 2:
        for p in front:
            p.crowding = float("inf")
        return
    for p in front:
        p.crowding = 0.0
    for idx in obj_indices:
        front_sorted = sorted(front, key=lambda x: x.objectives[idx])
        obj_min = front_sorted[0].objectives[idx]
        obj_max = front_sorted[-1].objectives[idx]
        span = obj_max - obj_min
        if span <= 0:
            continue
        front_sorted[0].crowding = float("inf")
        front_sorted[-1].crowding = float("inf")
        for i in range(1, len(front_sorted) - 1):
            front_sorted[i].crowding += (
                front_sorted[i + 1].objectives[idx] - front_sorted[i - 1].objectives[idx]
            ) / span
    # For obj2 (cost) we minimize, so lower is better - distance still positive
    # We use same formula; for minimize objectives the neighbor diff is correct.


def select(
    population: List[Individual],
    fronts: List[List[Individual]],
    size: int,
    E_min: int,
    E_max: int,
) -> List[Individual]:
    """Select size individuals using rank and crowding."""
    selected = []
    for rank, front in enumerate(fronts):
        if len(selected) + len(front) <= size:
            for p in front:
                p.rank = rank
                selected.append(p)
        else:
            crowding_distance(front)
            front_sorted = sorted(front, key=lambda x: -x.crowding)
            need = size - len(selected)
            for p in front_sorted[:need]:
                p.rank = rank
                selected.append(p)
            break
    return selected


def evolve_population(
    population: List[Individual],
    offspring: List[Individual],
    E_min: int,
    E_max: int,
    sigma: float,
    rng: np.random.Generator,
) -> List[Individual]:
    """Combine population + offspring, sort, select to population size."""
    combined = population + offspring
    fronts = non_dominated_sort(combined)
    return select(combined, fronts, len(population), E_min, E_max)

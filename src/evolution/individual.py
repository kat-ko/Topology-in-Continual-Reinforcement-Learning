"""
Evolution individual: genotype vector (E_target, R, p, alpha_w, alpha_gabs, alpha_gsq).
Encode/decode for NSGA-II.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.rewiring.rewiring_rule import RewiringGenotype


@dataclass
class Individual:
    genotype: RewiringGenotype
    objectives: Tuple[float, float] = (0.0, 0.0)  # (performance, cost=E_target)
    rank: int = -1
    crowding: float = 0.0

    def to_vector(self) -> np.ndarray:
        return self.genotype.to_vector()

    @classmethod
    def from_vector(
        cls,
        v: np.ndarray,
        E_min: int,
        E_max: int,
    ) -> "Individual":
        g = RewiringGenotype.from_vector(v)
        g.E_target = int(np.clip(g.E_target, E_min, E_max))
        return cls(genotype=g)

    def copy(self) -> "Individual":
        return Individual(
            genotype=RewiringGenotype(
                E_target=self.genotype.E_target,
                R=self.genotype.R,
                p=self.genotype.p,
                alpha_w=self.genotype.alpha_w,
                alpha_gabs=self.genotype.alpha_gabs,
                alpha_gsq=self.genotype.alpha_gsq,
            ),
            objectives=self.objectives,
            rank=self.rank,
            crowding=self.crowding,
        )


def mutate(
    ind: Individual,
    E_min: int,
    E_max: int,
    sigma: float = 0.2,
    rng: np.random.Generator = None,
) -> Individual:
    """Gaussian mutation on alpha_*; bounded perturb on p, R; discrete step on E_target. If rng is None, a default generator is used; pass explicit rng for reproducibility."""
    if rng is None:
        rng = np.random.default_rng()
    v = ind.to_vector()
    # v[0]=E_target, v[1]=R, v[2]=p, v[3:6]=alphas
    v[0] = int(np.clip(v[0] + rng.integers(-max(1, (E_max - E_min) // 10), (E_max - E_min) // 10 + 1), E_min, E_max))
    v[1] = max(1, int(v[1]) + rng.integers(-5, 6))
    v[2] = float(np.clip(v[2] + rng.normal(0, sigma), 0.01, 1.0))
    v[3] = float(max(0, v[3] + rng.normal(0, sigma)))
    v[4] = float(max(0, v[4] + rng.normal(0, sigma)))
    v[5] = float(max(0, v[5] + rng.normal(0, sigma)))
    return Individual.from_vector(v, E_min, E_max)


def crossover(
    p1: Individual,
    p2: Individual,
    E_min: int,
    E_max: int,
    rng: np.random.Generator = None,
) -> Individual:
    """Uniform crossover. If rng is None, a default generator is used; pass explicit rng for reproducibility."""
    if rng is None:
        rng = np.random.default_rng()
    v1 = p1.to_vector()
    v2 = p2.to_vector()
    mask = rng.random(len(v1)) < 0.5
    v = np.where(mask, v1, v2)
    return Individual.from_vector(v, E_min=E_min, E_max=E_max)

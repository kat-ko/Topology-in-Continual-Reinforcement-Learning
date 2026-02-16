"""Aggregate metrics: performance score from inner loop results."""
from __future__ import annotations
from typing import Dict, Any


def continual_learning_score(
    results: Dict[str, Any],
    wB: float = 1.0,
    wA: float = 1.0,
) -> float:
    """Performance score = wB * B_final + wA * A_after_B."""
    return wB * results.get("B_final", 0.0) + wA * results.get("A_after_B", 0.0)

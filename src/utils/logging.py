"""Run logging: write metrics and optional dumps to run directory."""
import json
import os
from typing import Any

def save_metrics(metrics: dict[str, Any], path: str) -> None:
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def load_metrics(path: str) -> dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)

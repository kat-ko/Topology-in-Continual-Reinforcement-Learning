"""Config utilities: save/load run config as JSON."""
import json
import os
from datetime import datetime
from typing import Any

def make_run_dir(results_root: str = "results") -> str:
    """Create a timestamped run directory under results/. Returns path."""
    os.makedirs(results_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config(config: dict[str, Any], path: str) -> None:
    """Save config dict to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def load_config(path: str) -> dict[str, Any]:
    """Load config from JSON file."""
    with open(path) as f:
        return json.load(f)

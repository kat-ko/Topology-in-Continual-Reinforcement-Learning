"""Strict seeding for reproducibility. All randomness must be controllable."""
import os
import random
import numpy as np

def set_seed(seed: int):
    """Set random seeds for numpy, random, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_rng(seed: int):
    """Return a numpy Generator for reproducible streams."""
    return np.random.default_rng(seed)

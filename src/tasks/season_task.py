"""
Season task: A1 -> B -> A2 continual learning protocol.
Reimplements Holton et al. task logic (external/transfer-interference).
Reference: basic_funcs.py (one-hot, labels), preprocessing.py (rule = winter - summer),
neural_network.py (accuracy = 1 - |wrapped_error|/pi).
"""
from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
import numpy as np

N_STIM = 6
INPUT_DIM = 12   # n_stim * 2, Holton convention
OUTPUT_DIM = 4   # [cos_f0, sin_f0, cos_f1, sin_f1]


class Similarity(enum.Enum):
    SAME = 0.0
    NEAR = np.pi / 6
    FAR = np.pi


def wrap_to_pi(values: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi]. (Holton: basic_funcs.wrap_to_pi.)"""
    return (np.asarray(values) + np.pi) % (2 * np.pi) - np.pi


def response_angle_from_cos_sin(cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Convert cos/sin to angle in radians (atan2(sin, cos) -> angle)."""
    return np.arctan2(sin, cos)


def wrapped_error(pred_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    """Signed error wrapped to [-pi, pi]."""
    return wrap_to_pi(pred_angle - target_angle)


def accuracy(pred_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    """Accuracy = 1 - |wrapped_error|/pi. (Holton: neural_network.compute_accuracy.)"""
    err = np.abs(wrapped_error(pred_angle, target_angle))
    return 1.0 - (err / np.pi)


@dataclass
class Trial:
    stimulus_id: int
    feature_probe: int
    input_vec: np.ndarray
    target: np.ndarray
    test_trial: bool = False


def make_one_hot(stimulus_id: int, dim: int = INPUT_DIM) -> np.ndarray:
    """12-dim one-hot at index stimulus_id (0..5). Holton uses indices 0-5."""
    x = np.zeros(dim, dtype=np.float32)
    x[stimulus_id] = 1.0
    return x


def create_task_definition(
    seed: int,
    shift_angle_rad: Optional[float] = None,
    similarity: Optional[Similarity] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Task A and Task B angle definitions for 6 stimuli.
    Returns (summer_A, winter_A, summer_B, winter_B) each shape (6,).
    Task A: rule_A = winter_A - summer_A (wrapped).
    Task B: same summer; winter_B = winter_A + shift (wrapped). So rule_B = rule_A + shift.
    """
    rng = np.random.default_rng(seed)
    if shift_angle_rad is None:
        shift_angle_rad = similarity.value if similarity is not None else 0.0

    summer_A = wrap_to_pi(rng.uniform(-np.pi, np.pi, N_STIM))
    rule_A = wrap_to_pi(rng.uniform(-np.pi, np.pi, N_STIM))
    winter_A = wrap_to_pi(summer_A + rule_A)

    rule_B = wrap_to_pi(rule_A + shift_angle_rad)
    winter_B = wrap_to_pi(summer_A + rule_B)

    return summer_A, winter_A, summer_A.copy(), winter_B


def angles_to_targets(summer: np.ndarray, winter: np.ndarray) -> np.ndarray:
    """(6,) summer, (6,) winter -> (6, 4) targets [cos_f0, sin_f0, cos_f1, sin_f1]."""
    targets = np.zeros((N_STIM, OUTPUT_DIM), dtype=np.float32)
    targets[:, 0] = np.cos(summer)
    targets[:, 1] = np.sin(summer)
    targets[:, 2] = np.cos(winter)
    targets[:, 3] = np.sin(winter)
    return targets


def generate_phase_trials(
    summer: np.ndarray,
    winter: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    include_test_trials: bool = False,
    test_stim_id: Optional[int] = None,
) -> List[Trial]:
    """
    Generate a list of trials for one phase (A1, B, or A2).
    Each step we present all 6 stimuli; for each we probe feature 0 or 1 (alternating or random).
    """
    trials = []
    targets = angles_to_targets(summer, winter)
    for step in range(n_steps):
        for stim_id in range(N_STIM):
            feature_probe = rng.integers(0, 2)
            inp = make_one_hot(stim_id)
            target = targets[stim_id]
            is_test = include_test_trials and test_stim_id is not None and stim_id == test_stim_id and feature_probe == 1
            trials.append(Trial(
                stimulus_id=stim_id,
                feature_probe=feature_probe,
                input_vec=inp,
                target=target,
                test_trial=is_test,
            ))
    return trials


def generate_stream(
    seed: int,
    shift_angle_rad: float,
    n_steps_a1: int,
    n_steps_b: int,
    n_steps_a2: int,
    include_test_trials: bool = False,
) -> Tuple[List[Trial], List[Trial], List[Trial], np.ndarray]:
    """
    Generate full A1, B, A2 trial streams and target arrays for the run.
    Returns (trials_a1, trials_b, trials_a2, (summer_A, winter_A, summer_B, winter_B)).
    """
    summer_A, winter_A, summer_B, winter_B = create_task_definition(seed, shift_angle_rad=shift_angle_rad)
    rng = np.random.default_rng(seed + 1)
    test_stim = rng.integers(0, N_STIM) if include_test_trials else None

    trials_a1 = generate_phase_trials(summer_A, winter_A, n_steps_a1, rng, include_test_trials, test_stim)
    trials_b = generate_phase_trials(summer_B, winter_B, n_steps_b, rng, include_test_trials, test_stim)
    trials_a2 = generate_phase_trials(summer_A, winter_A, n_steps_a2, rng, include_test_trials, test_stim)

    angles = (summer_A, winter_A, summer_B, winter_B)
    return trials_a1, trials_b, trials_a2, angles

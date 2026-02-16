# Evolutionary Structural Plasticity on Continual Learning (Season Task)

This codebase implements the Holton et al. A1→B→A2 continual "season" task and extends it with an evolutionary outer loop (NSGA-II) that evolves online structural plasticity (prune-and-grow rewiring) under multi-objective optimization (continual learning performance vs connection cost).

**Reference (read-only):** Task protocol and metrics follow [external/transfer-interference](external/transfer-interference) (Holton et al.).

## Setup

```bash
pip install -r requirements.txt
```

No package install (no `pip install -e .`); run scripts from repo root with `python scripts/<script>.py`.

## Project layout

- **src/** — Main package
  - **tasks/** — `season_task.py`: task generator (same/near/far similarity, trial streams)
  - **models/** — `masked_mlp.py`: MLP with binary masks (W_eff = W * M)
  - **training/** — `inner_loop.py`: A1→B→A2 training with optional rewiring hook
  - **rewiring/** — `signals.py`, `mask_ops.py`, `rewiring_rule.py`: EMA gradients, prune/grow, genotype
  - **evolution/** — `individual.py`, `evaluator.py`, `nsga2.py`: NSGA-II
  - **analysis/** — `metrics.py`, `lesions.py`, `repr.py`: score, lesion selectivity, CKA
  - **utils/** — `seeding.py`, `config.py`, `logging.py`
- **scripts/** — Entry points
- **results/** — Created at runtime; each run gets a timestamped subfolder with `config.json`, `metrics.json`, and (for evolution) `population.csv`, `pareto.csv`.

## How to run

### 1. Baseline (fixed topology)

Train a fully-connected MLP on same/near/far; reproduces qualitative Holton-style differences (e.g. far worse retention than same).

```bash
python scripts/run_baseline.py [--seed 42] [--n-steps-a1 50] [--n-steps-b 50] [--n-steps-a2 10]
```

**Static sparse baseline:** Same network with fixed edge budget, no rewiring:

```bash
python scripts/run_baseline.py --use-masked --E-target 400
```

**Fixed topology with MaskedMLP (full mask):**

```bash
python scripts/run_baseline.py --use-masked
```

### 2. Inner loop + rewiring (debug)

Run A1→B→A2 with hand-coded rewiring; sanity checks: edge count == E_target after rewiring, turnover > 0.

```bash
python scripts/run_inner_rewiring_debug.py [--similarity near] [--E-target 400] [--R 50] [--p 0.1]
```

**No rewiring:**

```bash
python scripts/run_inner_rewiring_debug.py --no-rewiring --E-target 400
```

**Random rewiring control** (same E_target, R, p; prune/grow by random selection):

```bash
python scripts/run_inner_rewiring_debug.py --random-rewiring --E-target 400 --R 50 --p 0.1
```

### 3. Evolution (one similarity)

NSGA-II for one condition; writes `results/<run_id>/population.csv` and `pareto.csv`.

```bash
python scripts/run_evolution.py --similarity near [--pop-size 8] [--generations 3] [--E-min 200] [--E-max 800]
```

### 4. Similarity sweep

Evolve separate populations for same, near, far; save Pareto fronts per condition.

```bash
python scripts/run_similarity_sweep.py [--pop-size 6] [--generations 2]
```

### 5. Pareto analysis

Load population/pareto CSVs and compute nondominated set:

```bash
python scripts/analyze_pareto.py results/<run_id>/population.csv [results/<run_id2>/population.csv ...] [--output pareto_merged.csv]
```

### 6. Lesions and representation

Freeze rewiring after A1; ablate edges added during B; CKA between A and B representations.

```bash
python scripts/run_lesions_and_repr.py [--freeze-rewiring-after-a1] [--ablate-B-edges]
```

## Baselines (summary)

| Baseline | How to run |
|----------|------------|
| **Fixed topology** | `run_baseline.py` (no --use-masked) or `run_baseline.py --use-masked` |
| **Static sparse** | `run_baseline.py --use-masked --E-target <N>` |
| **Random rewiring** | `run_inner_rewiring_debug.py --random-rewiring --E-target <N> --R <R> --p <p>` |
| **Performance-only evolution** | Use `run_evolution.py` with single objective (modify evaluator to only return obj1; or compare Pareto cost-extreme to a performance-only run). |

## Results location

All runs write under **results/** with a timestamped subfolder:

- `config.json` — Full run config
- `metrics.json` — Aggregate metrics (A1_final, B_final, A_after_B, retention, etc.)
- **Evolution runs:** `population.csv`, `pareto.csv`

## Sanity checks

1. **Edge count:** After each rewiring step, total edges should equal E_target (asserted in `run_inner_rewiring_debug.py`).
2. **Turnover:** With rewiring enabled, total turnover over a short run should be > 0 (asserted in debug script).
3. **Fixed baseline:** Same/near/far should show directional pattern (e.g. retention: same > near > far).
4. **Random vs evolved:** Random rewiring should not consistently outperform evolved rewiring at matched E_target, R, p (run both and compare metrics).

## License / reference

External reference: Holton, E., Braun, L., Thompson, J. A. F., Grohn, J., & Summerfield, C. (2025). *Humans and neural networks show similar patterns of transfer and interference during continual learning.* Code in `external/transfer-interference` is not modified; this project reimplements task logic and adds evolution/rewiring.

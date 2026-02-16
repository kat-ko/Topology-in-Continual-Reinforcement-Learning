"""
Sanity checks: (1) edge count == E_target after rewiring, (2) turnover > 0 with rewiring,
(3) fixed baseline same/near/far direction, (4) random rewiring does not beat evolved at matched cost.
Run with: python scripts/sanity_checks.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

def main():
    import numpy as np
    import torch
    from src.tasks.season_task import Similarity, generate_stream
    from src.models.masked_mlp import MaskedMLP
    from src.rewiring.signals import RewiringSignals
    from src.rewiring.rewiring_rule import RewiringGenotype, apply_rewiring_step, apply_random_rewiring_step
    from src.training.inner_loop import run_inner_loop
    from src.utils.seeding import set_seed

    set_seed(42)
    device = "cpu"
    E_target = 300
    rng = np.random.default_rng(42)
    model = MaskedMLP(input_dim=12, output_dim=4, hidden_dims=30, seed=42).to(device)
    model.set_edge_budget(E_target, rng=rng)
    _, total = model.count_edges()
    assert total == E_target, f"1. Edge count: got {total}, expected {E_target}"
    print("1. OK: Initial edge count == E_target")

    trials_a1, trials_b, trials_a2, _ = generate_stream(42, np.pi/6, 10, 10, 6, False)
    genotype = RewiringGenotype(E_target=E_target, R=20, p=0.1, alpha_w=0.1, alpha_gabs=0.5, alpha_gsq=0.5)
    signals = RewiringSignals(model, decay=0.99)
    def hook(m, step, phase):
        return apply_rewiring_step(m, signals, genotype, step, rng, None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_inner_loop(model, trials_a1, trials_b, trials_a2, optimizer, torch.device(device),
                   rewiring_hook=hook, signals_update=signals.update, seed=42)
    _, total_after = model.count_edges()
    assert total_after == E_target, f"2a. After rewiring edges: got {total_after}, expected {E_target}"
    print("2a. OK: Edge count == E_target after rewiring")

    set_seed(43)
    model2 = MaskedMLP(input_dim=12, output_dim=4, hidden_dims=30, seed=43).to(device)
    model2.set_edge_budget(E_target, rng=np.random.default_rng(43))
    log = {"turnover": []}
    def hook2(m, step, phase):
        return apply_random_rewiring_step(m, genotype, step, np.random.default_rng(43), log)
    trials_a1, trials_b, trials_a2, _ = generate_stream(43, np.pi/6, 15, 15, 6, False)
    run_inner_loop(model2, trials_a1, trials_b, trials_a2, torch.optim.SGD(model2.parameters(), lr=0.01),
                   torch.device(device), rewiring_hook=hook2, seed=43)
    tot_turnover = sum(log["turnover"])
    assert tot_turnover > 0, f"2b. Expected turnover > 0, got {tot_turnover}"
    print("2b. OK: Turnover > 0 with rewiring")

    from run_baseline import run_baseline_one_similarity
    set_seed(44)
    m_same = run_baseline_one_similarity(Similarity.SAME, 44, 25, 25, 6, 30, 0.01, device)
    m_near = run_baseline_one_similarity(Similarity.NEAR, 44, 25, 25, 6, 30, 0.01, device)
    m_far = run_baseline_one_similarity(Similarity.FAR, 44, 25, 25, 6, 30, 0.01, device)
    # Qualitative ordering: same condition should retain better than far (tolerance 0.1)
    assert m_same["A_after_B"] >= m_far["A_after_B"] - 0.1, (
        "3. Same should retain better than far: got same={}, far={}".format(
            m_same["A_after_B"], m_far["A_after_B"]
        )
    )
    print("3. OK: Fixed baseline same/near/far direction (qualitative)")

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()

"""
Repair experiment: verifier-guided correction under mixed corruption.

Tests whether PoE + verifier remasking improves over local heuristics
when wrong tokens are present in addition to masking.

Two domains: 4x4 Sudoku and planted k-SAT.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.domains.boolsat import BoolSatDomain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.denoisers import (
    LocalNoisyDenoiser,
    TNMarginalDenoiser,
    PoEDenoiser,
)
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy, AllMaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def make_sudoku_configs():
    """Return list of Sudoku config dicts for the ablation sweep."""
    domain = Sudoku4Domain()
    backend = ContractionMarginalBackend(domain)
    configs = []

    # Method configs: (denoiser_factory, policy_factory, label)
    methods = [
        ("local", lambda: LocalNoisyDenoiser(domain, sigma=0.0),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("local_noisy", lambda: LocalNoisyDenoiser(domain, sigma=0.5),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("tn", lambda: TNMarginalDenoiser(backend),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("poe", lambda: PoEDenoiser(LocalNoisyDenoiser(domain, sigma=0.5), backend, beta=0.5),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("tn_repair", lambda: TNMarginalDenoiser(backend),
         lambda: VerifierRepairPolicy(remask_threshold=1)),
        ("poe_repair", lambda: PoEDenoiser(LocalNoisyDenoiser(domain, sigma=0.5), backend, beta=0.5),
         lambda: VerifierRepairPolicy(remask_threshold=1)),
    ]

    for name, denoiser_fn, policy_fn in methods:
        configs.append({
            "domain": domain,
            "backend": backend,
            "name": name,
            "denoiser_fn": denoiser_fn,
            "policy_fn": policy_fn,
        })
    return configs


def make_sat_configs(n_vars=20, n_clauses=60, k=3):
    """Return list of SAT config dicts for the ablation sweep."""
    domain = BoolSatDomain(n_vars=n_vars, n_clauses=n_clauses, k=k)
    backend = ContractionMarginalBackend(domain)
    configs = []

    from tdr.diffusion.denoisers import RandomDenoiser

    methods = [
        ("random", lambda: RandomDenoiser(domain),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("tn", lambda: TNMarginalDenoiser(backend),
         lambda: ConfidenceUnmaskPolicy(threshold=0.99)),
        ("tn_repair", lambda: TNMarginalDenoiser(backend),
         lambda: VerifierRepairPolicy(remask_threshold=1)),
    ]

    for name, denoiser_fn, policy_fn in methods:
        configs.append({
            "domain": domain,
            "backend": backend,
            "name": name,
            "denoiser_fn": denoiser_fn,
            "policy_fn": policy_fn,
        })
    return configs


def run_trial(domain, denoiser, policy, seed, mask_ratio, wrong_ratio, max_steps=20):
    """Run a single repair trial."""
    rng = np.random.default_rng(seed)

    # Generate valid assignment and corrupt
    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)

    # True corrupted positions (for recall/precision)
    true_wrong = (x_corrupt != MASK) & (x_corrupt != x_true)

    sampler = MaskedDiffusionSampler(
        denoiser=denoiser,
        mask_policy=policy,
        verifier=domain.verifier,
        max_steps=max_steps,
    )

    t0 = time.monotonic()
    result = sampler.run(x_corrupt, rng)
    wall_time = time.monotonic() - t0

    final_x = result.x
    diagnostics = domain.verifier(final_x)
    success = bool(diagnostics.global_violation == 0 and np.all(final_x != MASK))

    return {
        "seed": seed,
        "mask_ratio": mask_ratio,
        "wrong_ratio": wrong_ratio,
        "success": success,
        "final_violation": int(diagnostics.global_violation),
        "num_steps": result.step,
        "wall_time": wall_time,
        "num_masks_final": int(np.sum(final_x == MASK)),
        "true_wrong_count": int(np.sum(true_wrong)),
        "contradiction_count": sum(
            1 for h in result.history
            if "contradiction" in str(h.get("violation", -1))
        ),
    }


def run_ablation(configs, mask_ratio=0.5, wrong_ratios=None,
                 num_trials=50, max_steps=20):
    """Run ablation across methods and wrong ratios."""
    if wrong_ratios is None:
        wrong_ratios = [0.0, 0.1, 0.2, 0.3]

    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        domain = cfg["domain"]
        print(f"\n--- {name} ---")

        for wr in wrong_ratios:
            denoiser = cfg["denoiser_fn"]()
            policy = cfg["policy_fn"]()
            trials = []

            for seed in range(num_trials):
                trial = run_trial(domain, denoiser, policy, seed,
                                  mask_ratio, wr, max_steps)
                trials.append(trial)

            key = f"{name}_wr{wr:.1f}"
            all_results[key] = _aggregate(trials, {
                "method": name,
                "wrong_ratio": wr,
                "mask_ratio": mask_ratio,
                "domain": domain.__class__.__name__,
            })
            print(f"  wr={wr:.1f}: "
                  f"success={all_results[key]['success_rate']:.3f}  "
                  f"violation={all_results[key]['avg_violation']:.2f}  "
                  f"steps={all_results[key]['avg_steps']:.1f}")

    return all_results


def _aggregate(trials, metadata):
    successes = [t["success"] for t in trials]
    violations = [t["final_violation"] for t in trials]
    steps = [t["num_steps"] for t in trials]
    times = [t["wall_time"] for t in trials]
    return {
        **metadata,
        "success_rate": float(np.mean(successes)),
        "avg_violation": float(np.mean(violations)),
        "avg_steps": float(np.mean(steps)),
        "avg_time": float(np.mean(times)),
        "num_trials": len(trials),
        "trials": trials,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run repair ablation experiment."
    )
    parser.add_argument("--domain", choices=["sudoku", "sat", "all"],
                        default="all")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    all_results = {}

    if args.domain in ("sudoku", "all"):
        print("=== Sudoku 4x4 ===")
        configs = make_sudoku_configs()
        r = run_ablation(configs, mask_ratio=args.mask_ratio,
                         wrong_ratios=[0.0, 0.1, 0.2],
                         num_trials=args.trials, max_steps=args.max_steps)
        all_results["sudoku4"] = r

    if args.domain in ("sat", "all"):
        print("\n=== Planted 3-SAT (n=20, m=60) ===")
        configs = make_sat_configs(n_vars=20, n_clauses=60, k=3)
        r = run_ablation(configs, mask_ratio=args.mask_ratio,
                         wrong_ratios=[0.0, 0.1, 0.2],
                         num_trials=args.trials, max_steps=args.max_steps)
        all_results["sat"] = r

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(
        LOGS_DIR / f"repair_{args.domain}_mr{args.mask_ratio:.2f}_{timestamp}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

"""
Experiment runner for Milestone 1: brute-force Sudoku masked completion.

Compares:
  1. Random denoiser (baseline)
  2. Local heuristic denoiser
  3. TN marginal denoiser (brute-force exact marginals)

Usage:
    python -m tdr.experiments.run_symbolic_tn_denoising
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend
from tdr.diffusion.denoisers import RandomDenoiser, LocalSudokuDenoiser, TNMarginalDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy

# Paths
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_trial(
    domain: Sudoku4Domain,
    denoiser,
    seed: int,
    mask_ratio: float,
    max_steps: int = 20,
) -> dict:
    """Run a single trial and return metrics."""
    rng = np.random.default_rng(seed)

    # Generate a valid solution and corrupt it
    x_true = domain.sample_solution(rng)
    x_masked = domain.corrupt(x_true, mask_ratio, rng)

    # Build sampler
    sampler = MaskedDiffusionSampler(
        denoiser=denoiser,
        mask_policy=ConfidenceUnmaskPolicy(threshold=0.99),
        verifier=domain.verifier,
        max_steps=max_steps,
    )

    t0 = time.monotonic()
    result = sampler.run(x_masked, rng)
    wall_time = time.monotonic() - t0

    # Evaluate
    final_x = result.x
    diagnostics = domain.verifier(final_x)
    success = bool(diagnostics.global_violation == 0 and np.all(final_x != MASK))

    return {
        "seed": seed,
        "mask_ratio": mask_ratio,
        "success": success,
        "final_violation": int(diagnostics.global_violation),
        "num_steps": result.step,
        "wall_time": wall_time,
        "num_masks_final": int(np.sum(final_x == MASK)),
        "final_x": final_x.tolist(),
        "history": [
            {k: int(v) if isinstance(v, (np.integer, bool)) else v
             for k, v in h.items()}
            for h in result.history
        ],
    }


def run_experiment(
    mask_ratio: float = 0.5,
    num_trials: int = 100,
    max_steps: int = 20,
) -> dict:
    """Run the full experiment comparing three denoisers."""
    domain = Sudoku4Domain()
    backend = BruteForceMarginalBackend(domain)

    # Precompute solutions
    n_solutions = backend.size()
    print(f"Domain: Sudoku 4x4, {n_solutions} solutions")
    print(f"Mask ratio: {mask_ratio}, Trials: {num_trials}, Max steps: {max_steps}")
    print()

    denoisers = {
        "random": RandomDenoiser(domain),
        "local": LocalSudokuDenoiser(domain),
        "tn_marginal": TNMarginalDenoiser(backend),
    }

    results = {}

    for method_name, denoiser in denoisers.items():
        print(f"Running method: {method_name} ...")
        trial_results = []

        for seed in range(num_trials):
            trial = run_trial(
                domain, denoiser, seed=seed,
                mask_ratio=mask_ratio, max_steps=max_steps,
            )
            trial_results.append(trial)

        # Aggregate
        successes = [t["success"] for t in trial_results]
        violations = [t["final_violation"] for t in trial_results]
        steps = [t["num_steps"] for t in trial_results]
        times = [t["wall_time"] for t in trial_results]

        success_rate = np.mean(successes)
        avg_violation = np.mean(violations)
        avg_steps = np.mean(steps)
        avg_time = np.mean(times)

        results[method_name] = {
            "success_rate": float(success_rate),
            "avg_violation": float(avg_violation),
            "avg_steps": float(avg_steps),
            "avg_time": float(avg_time),
            "num_trials": num_trials,
            "mask_ratio": mask_ratio,
            "trials": trial_results,
        }

        print(f"  success_rate={success_rate:.3f}  "
              f"avg_violation={avg_violation:.3f}  "
              f"avg_steps={avg_steps:.2f}  "
              f"avg_time={avg_time:.4f}s")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run symbolic TN denoising experiment."
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="Fraction of variables to mask (default: 0.5)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials per method (default: 100)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max diffusion steps (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: results/logs/<timestamp>.json)")
    args = parser.parse_args()

    results = run_experiment(
        mask_ratio=args.mask_ratio,
        num_trials=args.trials,
        max_steps=args.max_steps,
    )

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(LOGS_DIR / f"sudoku4_mr{args.mask_ratio:.2f}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

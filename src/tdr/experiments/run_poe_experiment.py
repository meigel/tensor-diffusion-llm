"""
Experiment runner for Milestone 3: Product-of-Experts denoising.

Compares noisy denoiser vs TN-only vs product-of-experts across
beta and sigma sweeps.

Usage:
    python -m tdr.experiments.run_poe_experiment
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.denoisers import (
    NoisyDenoiser,
    PoEDenoiser,
    TNMarginalDenoiser,
)
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

BETA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SIGMA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0]

DOMAIN = Sudoku4Domain()
BACKEND = ContractionMarginalBackend(DOMAIN)
DENOISER_TN = TNMarginalDenoiser(BACKEND)


def run_trial(denoiser, seed: int, mask_ratio: float, max_steps: int = 20) -> dict:
    """Run a single completion trial."""
    rng = np.random.default_rng(seed)
    x_true = DOMAIN.sample_solution(rng)
    x_masked = DOMAIN.corrupt(x_true, mask_ratio, rng)

    sampler = MaskedDiffusionSampler(
        denoiser=denoiser,
        mask_policy=ConfidenceUnmaskPolicy(threshold=0.99),
        verifier=DOMAIN.verifier,
        max_steps=max_steps,
    )

    t0 = time.monotonic()
    result = sampler.run(x_masked, rng)
    wall_time = time.monotonic() - t0

    final_x = result.x
    diagnostics = DOMAIN.verifier(final_x)
    success = bool(diagnostics.global_violation == 0 and np.all(final_x != MASK))

    return {
        "seed": seed,
        "mask_ratio": mask_ratio,
        "success": success,
        "final_violation": int(diagnostics.global_violation),
        "num_steps": result.step,
        "wall_time": wall_time,
        "num_masks_final": int(np.sum(final_x == MASK)),
    }


def run_sweep(mask_ratio: float = 0.5, num_trials: int = 50, max_steps: int = 20) -> dict:
    """Run beta/sigma sweep experiment.

    For each (sigma, beta) pair, runs num_trials and records metrics.
    Also runs TN-only and noisy-only baselines for comparison.
    """
    results = {}

    # TN-only baseline
    print("=== TN-only baseline ===")
    trials_tn = [run_trial(DENOISER_TN, seed=s, mask_ratio=mask_ratio, max_steps=max_steps)
                 for s in range(num_trials)]
    results["TN"] = _aggregate(trials_tn, {"method": "TN", "sigma": None, "beta": None})
    print(f"  success={results['TN']['success_rate']:.3f}  steps={results['TN']['avg_steps']:.2f}")

    # Sweep
    for sigma in SIGMA_VALUES:
        print(f"\n=== sigma={sigma} ===")
        # Noisy-only denoiser (beta=1.0 equivalent, via PoEDenoiser with beta=1)
        noisy_denoiser = NoisyDenoiser(BACKEND, sigma=sigma)

        for beta in BETA_VALUES:
            if beta == 1.0:
                # Pure noisy denoiser (no TN correction)
                denoiser = noisy_denoiser
                label = f"noisy_s{sigma}"
            elif beta == 0.0:
                # Pure TN (no noisy)
                denoiser = DENOISER_TN
                label = f"TN"
            else:
                # PoE combination
                denoiser = PoEDenoiser(noisy_denoiser, BACKEND, beta=beta)
                label = f"poe_s{sigma}_b{beta}"

            trials = [run_trial(denoiser, seed=s, mask_ratio=mask_ratio, max_steps=max_steps)
                      for s in range(num_trials)]

            key = label
            results[key] = _aggregate(trials, {
                "method": "noisy" if beta == 1.0 else "TN" if beta == 0.0 else "poe",
                "sigma": sigma,
                "beta": beta,
            })

            print(f"  beta={beta:.2f}: success={results[key]['success_rate']:.3f}  "
                  f"steps={results[key]['avg_steps']:.2f}  "
                  f"violation={results[key]['avg_violation']:.3f}")

    return results


def _aggregate(trials: list[dict], metadata: dict) -> dict:
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
        description="Run PoE beta/sigma sweep experiment."
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_sweep(
        mask_ratio=args.mask_ratio,
        num_trials=args.trials,
        max_steps=args.max_steps,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(LOGS_DIR / f"poe_sweep_mr{args.mask_ratio:.2f}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

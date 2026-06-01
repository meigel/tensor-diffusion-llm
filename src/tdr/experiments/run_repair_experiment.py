"""
Repair experiment: verifier-guided correction under mixed corruption.

Tests whether verifier-guided remasking improves robustness to wrong
tokens across two domains: 4x4 Sudoku and planted k-SAT.

Two critical baselines isolate the mechanism:
  - repair + random denoiser: proves it's not "just remask until lucky"
  - repair + local heuristic: shows TN marginals help beyond local info

SAT uses fresh random formulas per trial (formula_seed = trial seed),
so results generalize across formula instances.
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
    RandomDenoiser,
    LocalNoisyDenoiser,
    TNMarginalDenoiser,
    PoEDenoiser,
)
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain factories — create fresh domains per trial for proper generalization
# ---------------------------------------------------------------------------

def run_trial_sudoku(denoiser_factory, policy_factory, seed,
                     mask_ratio, wrong_ratio, max_steps=20):
    """Sudoku: domain is fixed (same 288 solutions), only corruption varies."""
    domain = Sudoku4Domain()
    backend = ContractionMarginalBackend(domain)
    rng = np.random.default_rng(seed)

    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)

    denoiser = denoiser_factory(domain, backend)
    policy = policy_factory()
    sampler = MaskedDiffusionSampler(
        denoiser=denoiser, mask_policy=policy,
        verifier=domain.verifier, max_steps=max_steps,
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
    }


def run_trial_sat(denoiser_factory, policy_factory, seed,
                  mask_ratio, wrong_ratio, max_steps=20,
                  n_vars=20, n_clauses=60, k=3):
    """SAT: fresh random formula per trial (formula_seed = seed).

    This ensures results generalize across formula instances, not
    just corruption seeds of a single formula.
    """
    domain = BoolSatDomain(n_vars=n_vars, n_clauses=n_clauses, k=k,
                           formula_seed=seed)
    backend = ContractionMarginalBackend(domain)
    rng = np.random.default_rng(seed + 10000)  # different seed for corruption

    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)

    denoiser = denoiser_factory(domain, backend)
    policy = policy_factory()
    sampler = MaskedDiffusionSampler(
        denoiser=denoiser, mask_policy=policy,
        verifier=domain.verifier, max_steps=max_steps,
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
    }


# ---------------------------------------------------------------------------
# Denoiser factories (domain, backend) -> denoiser
# ---------------------------------------------------------------------------

def make_sudoku_methods():
    """Return list of (name, denoiser_factory, policy_factory) for Sudoku."""
    methods = []

    # No-repair baselines
    methods.append(("local",
        lambda d, b: LocalNoisyDenoiser(d, sigma=0.0),
        lambda: ConfidenceUnmaskPolicy(threshold=0.99)))
    methods.append(("tn",
        lambda d, b: TNMarginalDenoiser(b),
        lambda: ConfidenceUnmaskPolicy(threshold=0.99)))
    methods.append(("poe",
        lambda d, b: PoEDenoiser(LocalNoisyDenoiser(d, sigma=0.5), b, beta=0.5),
        lambda: ConfidenceUnmaskPolicy(threshold=0.99)))

    # Repair baselines (mechanism-isolating)
    methods.append(("repair_random",
        lambda d, b: RandomDenoiser(d),
        lambda: VerifierRepairPolicy(remask_threshold=1)))
    methods.append(("repair_local",
        lambda d, b: LocalNoisyDenoiser(d, sigma=0.0),
        lambda: VerifierRepairPolicy(remask_threshold=1)))

    # TN + repair
    methods.append(("tn_repair",
        lambda d, b: TNMarginalDenoiser(b),
        lambda: VerifierRepairPolicy(remask_threshold=1)))
    methods.append(("poe_repair",
        lambda d, b: PoEDenoiser(LocalNoisyDenoiser(d, sigma=0.5), b, beta=0.5),
        lambda: VerifierRepairPolicy(remask_threshold=1)))

    return methods


def make_sat_methods():
    """Return list of (name, denoiser_factory, policy_factory) for SAT."""
    methods = []

    # No-repair baselines
    methods.append(("random",
        lambda d, b: RandomDenoiser(d),
        lambda: ConfidenceUnmaskPolicy(threshold=0.99)))
    methods.append(("tn",
        lambda d, b: TNMarginalDenoiser(b),
        lambda: ConfidenceUnmaskPolicy(threshold=0.99)))

    # Repair baselines (mechanism-isolating)
    methods.append(("repair_random",
        lambda d, b: RandomDenoiser(d),
        lambda: VerifierRepairPolicy(remask_threshold=1)))

    # TN + repair
    methods.append(("tn_repair",
        lambda d, b: TNMarginalDenoiser(b),
        lambda: VerifierRepairPolicy(remask_threshold=1)))

    return methods


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_ablation(domain_name, method_list, trial_fn, num_trials=100,
                 mask_ratio=0.5, wrong_ratios=None, max_steps=20, **kwargs):
    """Run ablation across methods and wrong ratios.

    Args:
        domain_name: "sudoku" or "sat" (for display).
        method_list: list of (name, denoiser_factory, policy_factory).
        trial_fn: callable(denoiser_factory, policy_factory, seed, ...) -> dict.
        num_trials: number of trials per (method, wrong_ratio) cell.
        wrong_ratios: list of wrong-token probabilities.
        max_steps: max diffusion steps.

    Returns:
        results dict keyed by method_wrX.X.
    """
    if wrong_ratios is None:
        wrong_ratios = [0.0, 0.1, 0.2, 0.3]

    all_results = {}

    for name, denoiser_fn, policy_fn in method_list:
        print(f"\n--- {name} ---")

        for wr in wrong_ratios:
            trials = []
            for seed in range(num_trials):
                trial = trial_fn(
                    denoiser_fn, policy_fn, seed,
                    mask_ratio=mask_ratio, wrong_ratio=wr,
                    max_steps=max_steps, **kwargs,
                )
                trials.append(trial)

            key = f"{name}_wr{wr:.2f}"
            stats = _compute_stats(trials)
            all_results[key] = {
                "method": name,
                "domain": domain_name,
                "wrong_ratio": wr,
                "mask_ratio": mask_ratio,
                "num_trials": len(trials),
                **stats,
                "trials": trials,
            }

            print(f"  wr={wr:.2f}: "
                  f"success={stats['success_rate']:.3f}±{stats['success_se']:.3f}  "
                  f"steps={stats['avg_steps']:.1f}  "
                  f"violation={stats['avg_violation']:.2f}")

    return all_results


def _compute_stats(trials):
    """Return aggregated stats with standard error."""
    successes = np.array([t["success"] for t in trials])
    violations = np.array([t["final_violation"] for t in trials])
    steps = np.array([t["num_steps"] for t in trials])
    times = np.array([t["wall_time"] for t in trials])

    n = len(trials)
    success_rate = float(np.mean(successes))
    success_se = float(np.sqrt(success_rate * (1 - success_rate) / max(n - 1, 1)))

    return {
        "success_rate": success_rate,
        "success_se": success_se,
        "avg_violation": float(np.mean(violations)),
        "std_violation": float(np.std(violations)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "avg_time": float(np.mean(times)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results, domain_name, output_dir=None):
    """Generate paper plots: success rate and steps vs wrong_ratio."""
    if output_dir is None:
        output_dir = str(PLOTS_DIR)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Collect unique methods and wrong ratios
    methods = sorted(set(r["method"] for r in all_results.values()))
    wrong_ratios = sorted(set(r["wrong_ratio"] for r in all_results.values()))

    # ---- Plot 1: success rate vs wrong_ratio ----
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        xs = []
        ys = []
        yerrs = []
        for wr in wrong_ratios:
            key = f"{method}_wr{wr:.2f}"
            if key in all_results:
                xs.append(wr)
                ys.append(all_results[key]["success_rate"])
                yerrs.append(all_results[key]["success_se"])
        ax.errorbar(xs, ys, yerr=yerrs, label=method, marker="o",
                    color=color, capsize=3, linewidth=1.5)

    ax.set_xlabel("Wrong-token ratio")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{domain_name}: Success vs wrong-token ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{domain_name}_success_vs_wrong.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ---- Plot 2: steps vs wrong_ratio ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in zip(methods, colors):
        xs = []
        ys = []
        for wr in wrong_ratios:
            key = f"{method}_wr{wr:.2f}"
            if key in all_results:
                xs.append(wr)
                ys.append(all_results[key]["avg_steps"])
        ax.plot(xs, ys, label=method, marker="o",
                color=color, linewidth=1.5)

    ax.set_xlabel("Wrong-token ratio")
    ax.set_ylabel("Average steps")
    ax.set_title(f"{domain_name}: Steps vs wrong-token ratio")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{domain_name}_steps_vs_wrong.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run definitive repair ablation experiment."
    )
    parser.add_argument("--domain", choices=["sudoku", "sat", "all"],
                        default="all")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PDF generation")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    wrong_ratios = [0.0, 0.05, 0.1, 0.2, 0.3]
    all_results = {}

    if args.domain in ("sudoku", "all"):
        print("=" * 60)
        print("Sudoku 4x4")
        print("=" * 60)
        methods = make_sudoku_methods()
        r = run_ablation("sudoku", methods,
                         run_trial_sudoku,
                         num_trials=args.trials,
                         mask_ratio=args.mask_ratio,
                         wrong_ratios=wrong_ratios,
                         max_steps=args.max_steps)
        all_results["sudoku4"] = r
        if not args.no_plots:
            plot_results(r, "sudoku4")
        print()

    if args.domain in ("sat", "all"):
        print("=" * 60)
        print("Planted 3-SAT (n=20, m=60)")
        print("=" * 60)
        methods = make_sat_methods()
        r = run_ablation("sat", methods,
                         run_trial_sat,
                         num_trials=args.trials,
                         mask_ratio=args.mask_ratio,
                         wrong_ratios=wrong_ratios,
                         max_steps=args.max_steps,
                         n_vars=20, n_clauses=60, k=3)
        all_results["sat"] = r
        if not args.no_plots:
            plot_results(r, "sat")
        print()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(
        LOGS_DIR / f"repair_definitive_{args.domain}_mr{args.mask_ratio:.2f}_{timestamp}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

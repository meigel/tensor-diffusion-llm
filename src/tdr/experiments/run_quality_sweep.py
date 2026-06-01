"""\
Denoiser-quality sweep: repair benefit across a continuous spectrum.

Shows that verifier-guided remasking is most valuable when the denoiser
is imperfect — repair benefit is inversely related to denoiser quality.

Uses a mixture denoiser p = (1-α) · q_TN + α · uniform that interpolates
between exact TN marginals (α=0, high accuracy) and random guessing (α=1,
low accuracy), providing a controlled accuracy spectrum.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.run_quality_sweep --domain sudoku --trials 200
    python -m tdr.experiments.run_quality_sweep --domain sat --trials 200
    python -m tdr.experiments.run_quality_sweep --domain both --trials 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.domains.boolsat import BoolSatDomain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.denoisers import TNMarginalDenoiser, RandomDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Mixture denoiser
# ---------------------------------------------------------------------------

class MixtureDenoiser:
    """Interpolates between TN marginals and uniform random.

    p_i(v) = (1 - α) · q_i(v) + α · (1 / d)

    where q_i(v) are exact TN marginals and α ∈ [0, 1] is the mixture
    weight. α = 0 → exact TN (high accuracy), α = 1 → random (low).

    Attributes:
        tn_denoiser: TNMarginalDenoiser instance.
        alpha: Mixture weight α ∈ [0, 1].
        n: Number of variables.
        d: Max domain size.
    """

    def __init__(self, tn_denoiser: TNMarginalDenoiser, alpha: float):
        self.tn = tn_denoiser
        self.alpha = alpha
        self.n = tn_denoiser.n
        self.d = tn_denoiser.d

    def predict(self, x_masked: np.ndarray,
                rng: np.random.Generator | None = None) -> np.ndarray:
        q = self.tn.predict(x_masked, rng=rng)
        if self.alpha == 0.0:
            return q
        uniform = np.full_like(q, 1.0 / self.d)
        return (1.0 - self.alpha) * q + self.alpha * uniform


# ---------------------------------------------------------------------------
# Denoiser accuracy calibration
# ---------------------------------------------------------------------------

def measure_denoiser_accuracy(
    domain, denoiser, num_samples: int = 500, mask_ratio: float = 0.5,
    rng_seed: int = 42,
) -> float:
    """Measure one-step masked-position accuracy on clean data (no wrong tokens).

    Returns fraction of masked positions where denoiser argmax matches
    the true value.
    """
    rng = np.random.default_rng(rng_seed)
    correct = 0
    total = 0
    for _ in range(num_samples):
        x_true = domain.sample_solution(rng)
        x_masked = domain.corrupt(x_true, mask_ratio, rng)
        dist = denoiser.predict(x_masked, rng=rng)
        preds = dist.argmax(axis=1)
        for i in range(len(x_masked)):
            if x_masked[i] == MASK:
                if preds[i] == x_true[i]:
                    correct += 1
                total += 1
    return correct / max(total, 1)


def calibrate_alphas(
    domain, backend, alpha_vals: list[float], num_samples: int = 500,
) -> list[dict]:
    """Measure per-alpha accuracy and return calibration table."""
    name = domain.__class__.__name__
    print(f"\nCalibrating {name} ({len(alpha_vals)} alphas)...")
    tn = TNMarginalDenoiser(backend)
    calib = []
    for alpha in alpha_vals:
        den = MixtureDenoiser(tn, alpha)
        acc = measure_denoiser_accuracy(domain, den, num_samples)
        calib.append({"alpha": alpha, "accuracy": round(acc, 4)})
        print(f"  alpha={alpha:.2f} -> accuracy={acc:.4f}")
    # Random baseline
    random_den = RandomDenoiser(domain)
    acc = measure_denoiser_accuracy(domain, random_den, num_samples)
    calib.append({"alpha": 1.0, "accuracy": round(acc, 4), "method": "random"})
    print(f"  random        -> accuracy={acc:.4f}")
    return calib


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    domain, backend, alpha, mask_ratio, wrong_ratio, use_repair, seed,
    is_sat=False,
) -> dict:
    """Run a single repair trial with the mixture denoiser."""
    if is_sat:
        # SAT: fresh formula per trial
        n_vars = domain._n
        n_clauses = domain._n_clauses
        k = domain._k
        dom = BoolSatDomain(n_vars=n_vars, n_clauses=n_clauses, k=k,
                            formula_seed=seed)
        bk = ContractionMarginalBackend(dom)
    else:
        dom = domain
        bk = backend

    tn = TNMarginalDenoiser(bk)
    den = MixtureDenoiser(tn, alpha) if alpha < 1.0 else RandomDenoiser(dom)

    rng = np.random.default_rng(seed + 10000)
    x_true = dom.sample_solution(rng)
    x_corrupt = dom.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)

    policy = (
        VerifierRepairPolicy(remask_threshold=1)
        if use_repair
        else ConfidenceUnmaskPolicy(threshold=0.99)
    )

    sampler = MaskedDiffusionSampler(
        denoiser=den,
        mask_policy=policy,
        verifier=dom.verifier,
        max_steps=20,
    )

    result = sampler.run(x_corrupt, rng)
    final_x = result.x
    diag = dom.verifier(final_x)
    success = bool(diag.global_violation == 0 and np.all(final_x != MASK))

    return {
        "seed": seed,
        "success": success,
        "final_violation": int(diag.global_violation),
        "num_steps": result.step,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    domain_name: str, alpha_vals: list[float], wrong_ratios: list[float],
    num_trials: int, mask_ratio: float,
) -> dict:
    """Run denoiser-quality sweep for one domain.

    Returns dict keyed by '{label}_wr{wrong_ratio}' with per-alpha,
    per-policy, per-wr success rates and accuracy.
    """
    print(f"\n{'='*60}")
    print(f"SWEEP: {domain_name}")
    print(f"{'='*60}")

    # Create domain and backend
    if domain_name == "sudoku":
        domain = Sudoku4Domain()
        backend = ContractionMarginalBackend(domain)
        is_sat = False
    else:
        domain = BoolSatDomain(n_vars=20, n_clauses=60, k=3, formula_seed=0)
        backend = ContractionMarginalBackend(domain)
        is_sat = True

    # Calibrate accuracy for each alpha
    calib = calibrate_alphas(domain, backend, alpha_vals, num_samples=500)
    alpha_to_acc = {c["alpha"]: c["accuracy"] for c in calib if "alpha" in c}

    all_results = {}

    for alpha in alpha_vals:
        accuracy = alpha_to_acc.get(alpha, 0.0)
        print(f"\n--- alpha={alpha:.2f} (accuracy={accuracy:.4f}) ---")

        for use_repair in [False, True]:
            label = f"a{alpha:.2f}" + ("_repair" if use_repair else "")
            policy_name = "repair" if use_repair else "no_repair"

            for wr in wrong_ratios:
                successes = []
                details = []

                for seed in range(num_trials):
                    tr = run_trial(
                        domain, backend, alpha, mask_ratio, wr,
                        use_repair, seed, is_sat=is_sat,
                    )
                    successes.append(tr["success"])
                    details.append(tr)

                sr = float(np.mean(successes))
                se = float(np.sqrt(sr * (1 - sr) / max(num_trials - 1, 1)))

                key = f"{label}_wr{wr:.2f}"
                all_results[key] = {
                    "domain": domain_name,
                    "alpha": alpha,
                    "accuracy": accuracy,
                    "repair": use_repair,
                    "wrong_ratio": wr,
                    "success_rate": sr,
                    "success_se": se,
                    "num_trials": num_trials,
                    "trials": details,
                }

                print(f"  {policy_name:10s} wr={wr:.2f}: "
                      f"success={sr:.4f}±{se:.4f}")

    all_results["_calibration"] = calib
    all_results["_metadata"] = {
        "domain": domain_name,
        "alpha_vals": alpha_vals,
        "wrong_ratios": wrong_ratios,
        "num_trials": num_trials,
        "mask_ratio": mask_ratio,
    }
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quality_sweep(sudoku_results=None, sat_results=None):
    """Generate 2-panel figure: repair lift vs accuracy, success vs accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    domain_data = {}
    if sudoku_results:
        domain_data["Sudoku 4×4"] = sudoku_results
    if sat_results:
        domain_data["3-SAT (n=20)"] = sat_results

    markers = {"Sudoku 4×4": "o", "3-SAT (n=20)": "s"}

    # Panel A: Repair lift vs denoiser accuracy
    ax = axes[0]
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    for domain_label, data in domain_data.items():
        for wr in [0.0, 0.1, 0.2, 0.3]:
            x_vals, y_vals = [], []
            for alpha in sorted(set(
                e["alpha"] for e in data.values()
                if isinstance(e, dict) and "alpha" in e
            )):
                base_key = f"a{alpha:.2f}"
                repair_key = f"{base_key}_repair_wr{wr:.2f}"
                no_repair_key = f"{base_key}_wr{wr:.2f}"
                if repair_key in data and no_repair_key in data:
                    lift = (data[repair_key]["success_rate"]
                            - data[no_repair_key]["success_rate"])
                    acc = data[no_repair_key]["accuracy"]
                    x_vals.append(acc)
                    y_vals.append(lift)
            if x_vals:
                ax.plot(x_vals, y_vals, marker=markers[domain_label],
                        label=f"{domain_label} wr={wr:.1f}", linewidth=1.5)

    ax.set_xlabel("One-step denoiser accuracy")
    ax.set_ylabel("Repair lift (Δ success rate)")
    ax.set_title("Verifier repair benefit vs denoiser quality")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel B: Success rate vs denoiser accuracy (at wr=0.2)
    ax = axes[1]
    wr_target = 0.2

    for domain_label, data in domain_data.items():
        for repair_flag, ls, label_prefix in [
            (False, "-", "No repair"),
            (True, "--", "Verifier repair"),
        ]:
            x_vals, y_vals, y_errs = [], [], []
            for alpha in sorted(set(
                e["alpha"] for e in data.values()
                if isinstance(e, dict) and "alpha" in e
            )):
                base_key = f"a{alpha:.2f}"
                key = f"{base_key}{'_repair' if repair_flag else ''}_wr{wr_target:.2f}"
                if key in data:
                    x_vals.append(data[key]["accuracy"])
                    y_vals.append(data[key]["success_rate"])
                    y_errs.append(data[key]["success_se"])
            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                           marker=markers[domain_label],
                           linestyle=ls, capsize=3, linewidth=1.5,
                           label=f"{domain_label} {label_prefix}")

    ax.set_xlabel("One-step denoiser accuracy")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Success rate at wrong-token ratio=0.2")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / "quality_sweep.pdf"
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)

    png_path = PLOTS_DIR / "quality_sweep.png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["sudoku", "sat", "both"],
                        default="both")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # Mixture weights: 0=pure TN, 1=pure random
    alpha_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    wrong_ratios = [0.0, 0.1, 0.2]

    sudoku_results = None
    sat_results = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.domain in ("sudoku", "both"):
        sudoku_results = run_sweep(
            "sudoku", alpha_vals, wrong_ratios, args.trials, args.mask_ratio,
        )
        path = LOGS_DIR / f"quality_sweep_sudoku_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(sudoku_results, f, indent=2, default=str)
        print(f"\nSaved: {path}")

    if args.domain in ("sat", "both"):
        sat_results = run_sweep(
            "sat", alpha_vals, wrong_ratios, args.trials, args.mask_ratio,
        )
        path = LOGS_DIR / f"quality_sweep_sat_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(sat_results, f, indent=2, default=str)
        print(f"\nSaved: {path}")

    if not args.no_plots:
        plot_quality_sweep(sudoku_results, sat_results)

    print("\nDone.")


if __name__ == "__main__":
    main()

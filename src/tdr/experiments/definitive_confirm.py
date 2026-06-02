"""\
Definitive confirmation: coupling levels 0, 1, 7 with n=200, all baselines.

Per Codex's recommendation: one clean experiment to lock in the numbers
before paper rewrite. Focus on the sweet spot (level 1) with a wrong-ratio
curve, and levels 0, 7 as bridges.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.definitive_confirm
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdr import MASK
from tdr.domains.json_schema_v2 import FIELD_TO_IDX
from tdr.domains.base import VerifierDiagnostics
from tdr.domains.json_schema_v3 import check_constraints_v3, decode_array
from tdr.experiments.coupling_sweep import (
    CoupledJsonDomain, adversarial_corrupt, measure_residual_precision,
    COUPLED_CONSTRAINTS,
)
from tdr.training.datasets import DenoisingDataset
from tdr.training.train_denoiser import MLPDenoiser, train_epoch
from tdr.diffusion.denoisers import LearnedDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy, BaseMaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy, RandomRemaskPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

N_TRIALS = 200
WR_MAIN = 0.4
MAX_STEPS = 10


# -----------------------------------------------------------------------
# Baseline policies
# -----------------------------------------------------------------------

class GlobalRetryPolicy(BaseMaskPolicy):
    """Remask all observed positions — no localization."""
    def select_remask(self, x, diagnostics, rng=None):
        return x != MASK


class PassFailRepairPolicy(BaseMaskPolicy):
    """If any violation exists, remask all; else do nothing.
    Simulates a global pass/fail verifier with no localization."""
    def select_remask(self, x, diagnostics, rng=None):
        if diagnostics.global_violation > 0:
            return x != MASK
        return np.zeros(len(x), dtype=bool)

    def select_fill(self, x, dist, diagnostics, rng=None):
        return x == MASK


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(domain, denoiser, wrong_ratio, n_trials=N_TRIALS):
    """Run all policies and return results dict."""
    results = {}
    rng_global = np.random.default_rng(42)

    configs = [
        ("no_repair", ConfidenceUnmaskPolicy(0.99), domain.verifier),
        ("verifier_repair", VerifierRepairPolicy(1), domain.verifier),
        ("random_remask", RandomRemaskPolicy(0.25), domain.verifier),
        ("global_retry", GlobalRetryPolicy(), domain.verifier),
        ("pass_fail", PassFailRepairPolicy(), domain.verifier),
    ]

    for pname, policy, verifier in configs:
        successes = []
        precisions = []
        recalls = []
        for seed in range(n_trials):
            rng = np.random.default_rng(seed + 10000)
            x_true = domain.sample_solution(rng)
            x_corrupt = adversarial_corrupt(domain, x_true, 0.3, wrong_ratio, rng)

            # Measure residual precision on the corrupted state
            prec, rec = measure_residual_precision(domain, x_corrupt, x_true, domain.verifier)
            precisions.append(prec)
            recalls.append(rec)

            sampler = MaskedDiffusionSampler(
                denoiser=denoiser, mask_policy=policy,
                verifier=verifier, max_steps=MAX_STEPS,
            )
            result = sampler.run(x_corrupt, rng)
            diag = domain.verifier(result.x)
            successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))

        sr = float(np.mean(successes))
        se = float(np.sqrt(sr * (1 - sr) / max(n_trials - 1, 1)))
        results[pname] = {
            "success_rate": sr,
            "success_se": se,
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
        }

    return results


def main():
    print("=" * 60)
    print("DEFINITIVE CONFIRMATION")
    print("=" * 60)

    device = torch.device("cpu")

    # Train MLP on base v2 domain
    print("\n--- Training MLP on base domain (v2) ---")
    base_domain = CoupledJsonDomain(num_coupled=0)
    N, D = base_domain.num_variables(), base_domain.max_domain_size()
    input_dim = N * (D + 1)
    output_dim = N * D

    mlp = MLPDenoiser(input_dim, output_dim, [256, 256])
    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_data = DenoisingDataset(base_domain, 20000, 0.3, rng_seed=0)
    loader = train_data.get_dataloader(64, shuffle=True)

    for epoch in range(30):
        train_epoch(mlp, loader, opt, crit, device, D)
    mlp.eval()
    mlp_wrap = LearnedDenoiser(mlp, N, D)
    print("  Done.")

    # Evaluate
    all_results = {}

    # Main levels at wr=0.4
    print(f"\n{'='*60}")
    print(f"MAIN EVALUATION (wr={WR_MAIN}, n={N_TRIALS}, steps={MAX_STEPS})")
    print(f"{'='*60}")

    for nc in [0, 1, 7]:
        domain = CoupledJsonDomain(num_coupled=nc)
        cname = COUPLED_CONSTRAINTS[nc-1][0] if nc > 0 else "base_v2"
        print(f"\n--- Coupling level {nc} ({cname}) ---")

        results = evaluate(domain, mlp_wrap, WR_MAIN, N_TRIALS)

        for pname, r in results.items():
            lift = r["success_rate"] - results.get("no_repair", {}).get("success_rate", 0)
            print(f"  {pname:20s}: sr={r['success_rate']:.4f}+-{r['success_se']:.4f}  "
                  f"lift={lift:+.3f}  prec={r['precision']:.3f}  rec={r['recall']:.3f}")

        all_results[f"level{nc}_wr{WR_MAIN:.2f}"] = {
            "coupling_level": nc,
            "wrong_ratio": WR_MAIN,
            **results,
        }

    # Level 1 wrong-ratio sweep
    print(f"\n{'='*60}")
    print(f"LEVEL 1 WRONG-RATIO SWEEP (n={N_TRIALS})")
    print(f"{'='*60}")

    domain_l1 = CoupledJsonDomain(num_coupled=1)

    for wr in [0.2, 0.4, 0.6]:
        print(f"\n--- Level 1, wr={wr:.1f} ---")
        results = evaluate(domain_l1, mlp_wrap, wr, N_TRIALS)

        for pname, r in results.items():
            lift = r["success_rate"] - results.get("no_repair", {}).get("success_rate", 0)
            print(f"  {pname:20s}: sr={r['success_rate']:.4f}+-{r['success_se']:.4f}  "
                  f"lift={lift:+.3f}")

        all_results[f"level1_wr{wr:.2f}"] = {
            "coupling_level": 1,
            "wrong_ratio": wr,
            **results,
        }

    # Save
    all_results["_config"] = {
        "n_trials": N_TRIALS,
        "max_steps": MAX_STEPS,
        "mask_ratio": 0.3,
        "corruption": "adversarial",
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"definitive_confirm_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for nc in [0, 1, 7]:
        key = f"level{nc}_wr{WR_MAIN:.2f}"
        if key in all_results:
            r = all_results[key]
            nr = r.get("no_repair", {}).get("success_rate", 0)
            vr = r.get("verifier_repair", {}).get("success_rate", 0)
            pf = r.get("pass_fail", {}).get("success_rate", 0)
            pr = r.get("verifier_repair", {}).get("precision", 0)
            rc = r.get("verifier_repair", {}).get("recall", 0)
            print(f"  Level {nc}: no_rep={nr:.3f}  repair={vr:.3f}  pass_fail={pf:.3f}  "
                  f"prec={pr:.3f}  rec={rc:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Stress test: find where verifier repair breaks and compare baselines.

Tests JSON-v2 at higher wrong ratios and with degraded verifiers
to find regimes where methods are distinguishable.

Also implements direct baseline comparisons:
- Verifier repair (our method)
- Confidence-only remasking (ReMDM-style)
- Random remasking (control)
- Global retry (pass/fail verifier, no localization)

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.stress_test
"""

import os, sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdr import MASK
from tdr.domains.json_schema_v2 import (
    JsonSchemaV2Domain, FIELD_NAMES, FIELD_TO_IDX, check_constraints,
    decode_array,
)
from tdr.domains.base import VerifierDiagnostics
from tdr.training.train_denoiser import MLPDenoiser
from tdr.diffusion.denoisers import LearnedDenoiser, RandomDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy, BaseMaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy, RandomRemaskPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(LOGS_DIR, exist_ok=True)

# Load trained MLP
d = JsonSchemaV2Domain()
N = d.num_variables()
D = d.max_domain_size()
DOMAIN_SIZES = [d.domain_size(i) for i in range(N)]

INPUT_DIM = N * (D + 1)
OUTPUT_DIM = N * D

model = MLPDenoiser(INPUT_DIM, OUTPUT_DIM, [256, 256])
ckpt = CKPT_DIR / "denoiser_mlp_jsonv2.pt"
if ckpt.exists():
    import torch
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    print(f"Loaded: {ckpt}")
else:
    print(f"No checkpoint at {ckpt}")

mlp_denoiser = LearnedDenoiser(model, N, D)
random_denoiser = RandomDenoiser(d)

MASK_RATIO = 0.3
WRONG_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
NUM_TRIALS = 100


# -----------------------------------------------------------------------
# Degraded verifiers
# -----------------------------------------------------------------------

def make_degraded_verifier(domain, mode="exact"):
    """Create a verifier with controlled degradation.

    Modes:
      exact:       full constraint checking (normal)
      partial_50:  randomly hides 50% of constraint groups
      group_only:  returns group-level residuals (not variable-level)
      pass_fail:   returns global violation count, zero residuals
    """
    rng = np.random.default_rng(42)

    if mode == "exact":
        return domain.verifier

    elif mode == "partial_50":
        # Randomly select half of constraints to ignore
        all_constraint_names = [
            "C1_provider_region", "C2_encryption", "C3_production",
            "C4_database", "C5_compliance", "C6_free_tier",
            "C7_storage_size", "C8_serverless",
        ]
        active = set(rng.choice(all_constraint_names, size=4, replace=False))

        def partial_verifier(x):
            instance = decode_array(x)
            all_violations = check_constraints(instance)
            # Filter to only active constraints
            violations = []
            for desc, fields in all_violations:
                # Match violation to constraint by field pattern
                if _classify_violation(desc, fields) in active:
                    violations.append((desc, fields))
            return _build_diagnostics(domain, violations)

        return partial_verifier

    elif mode == "group_only":
        # Constraints checked, but residuals are uniform across all
        # variables when any violation exists (no localization signal).
        def group_verifier(x):
            instance = decode_array(x)
            violations = check_constraints(instance)
            n = domain.num_variables()
            gv = len(violations)
            local_residuals = np.ones(n, dtype=np.int64) if gv > 0 else np.zeros(n, dtype=np.int64)
            return VerifierDiagnostics(
                global_violation=gv,
                local_residuals=local_residuals,
            )
        return group_verifier

    elif mode == "pass_fail":
        # Only global pass/fail — no residual information.
        def pf_verifier(x):
            instance = decode_array(x)
            violations = check_constraints(instance)
            return VerifierDiagnostics(
                global_violation=len(violations),
                local_residuals=np.zeros(domain.num_variables(), dtype=np.int64),
            )
        return pf_verifier

    raise ValueError(f"Unknown mode: {mode}")


def _classify_violation(desc, fields):
    """Map a violation to a constraint group name."""
    if "region" in desc:
        return "C1_provider_region"
    if "encryption" in desc and "multi" in desc:
        return "C2_encryption"
    if "encryption" in desc and "backup" in desc:
        return "C2_encryption"
    if "production" in desc:
        return "C3_production"
    if "database" in desc:
        return "C4_database"
    if "hipaa" in desc or "pci_dss" in desc:
        return "C5_compliance"
    if "free" in desc:
        return "C6_free_tier"
    if "storage" in desc and "memory" in desc:
        return "C7_storage_size"
    if "storage" in desc and "instance" in desc:
        return "C7_storage_size"
    if "serverless" in desc:
        return "C8_serverless"
    return "other"


def _build_diagnostics(domain, violations):
    """Build VerifierDiagnostics from constraint violation list."""
    n = domain.num_variables()
    field_counts = {}
    for desc, fields in violations:
        for f in fields:
            idx = FIELD_TO_IDX[f]
            field_counts[idx] = field_counts.get(idx, 0) + 1
    local_residuals = np.zeros(n, dtype=np.int64)
    for idx, count in field_counts.items():
        local_residuals[idx] = count
    return VerifierDiagnostics(
        global_violation=len(violations),
        local_residuals=local_residuals,
    )


# -----------------------------------------------------------------------
# Global retry policy (ReMDM-like: no localization, just retry all)
# -----------------------------------------------------------------------

class GlobalRetryPolicy(BaseMaskPolicy):
    """Retry all observed positions — no localization.

    Remasks ALL observed positions at each step. This simulates a
    pass/fail-only verifier with global retry (the weakest baseline).
    """
    def select_remask(self, x, diagnostics, rng=None):
        return x != MASK  # remask all observed positions


# -----------------------------------------------------------------------
# Trial runner
# -----------------------------------------------------------------------

def run_trial(domain, denoiser, policy, verifier, mask_ratio, wrong_ratio, seed):
    """Run a single repair trial."""
    rng = np.random.default_rng(seed)
    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)
    sampler = MaskedDiffusionSampler(
        denoiser=denoiser, mask_policy=policy,
        verifier=verifier, max_steps=20,
    )
    result = sampler.run(x_corrupt, rng)
    diag = domain.verifier(result.x)
    return int(diag.global_violation == 0 and np.all(result.x != -1))


def evaluate_config(domain, denoiser, policy, verifier, label):
    """Evaluate one configuration across wrong ratios."""
    print(f"\n  --- {label} ---")
    results = {}
    for wr in WRONG_RATIOS:
        successes = []
        for seed in range(NUM_TRIALS):
            successes.append(
                run_trial(domain, denoiser, policy, verifier, MASK_RATIO, wr, seed)
            )
        sr = float(np.mean(successes))
        se = float(np.sqrt(sr * (1 - sr) / max(NUM_TRIALS - 1, 1)))
        results[wr] = {"success_rate": sr, "success_se": se}
        print(f"    wr={wr:.1f}: success={sr:.4f}±{se:.4f}")
    return results


def main():
    print("=" * 60)
    print("STRESS TEST — JSON-v2")
    print("=" * 60)

    all_results = {}

    # ---- Experiment 1: High wrong ratios ----
    print("\n" + "=" * 60)
    print("EXP 1: HIGH WRONG RATIOS")
    print("=" * 60)

    exact_verifier = d.verifier

    for denoiser, dname in [(mlp_denoiser, "mlp"), (random_denoiser, "random")]:
        for policy, pname in [
            (VerifierRepairPolicy(remask_threshold=1), "verifier_repair"),
            (ConfidenceUnmaskPolicy(threshold=0.99), "no_repair"),
            (RandomRemaskPolicy(remask_fraction=0.25), "random_remask"),
            (GlobalRetryPolicy(), "global_retry"),
        ]:
            results = evaluate_config(d, denoiser, policy, exact_verifier,
                                      f"{dname}_{pname}")
            for wr, r in results.items():
                key = f"high_wr_{dname}_{pname}_wr{wr:.2f}"
                all_results[key] = {
                    "denoiser": dname, "policy": pname, "wrong_ratio": wr,
                    "verifier_mode": "exact", **r,
                }

    # ---- Experiment 2: Verifier degradation ----
    print("\n" + "=" * 60)
    print("EXP 2: VERIFIER DEGRADATION at wr=0.4")
    print("=" * 60)

    wr_fixed = 0.4
    verifier_modes = ["exact", "partial_50", "group_only", "pass_fail"]

    for vmode in verifier_modes:
        verifier = make_degraded_verifier(d, vmode)
        for denoiser, dname in [(mlp_denoiser, "mlp"), (random_denoiser, "random")]:
            for policy, pname in [
                (VerifierRepairPolicy(remask_threshold=1), "verifier_repair"),
                (ConfidenceUnmaskPolicy(threshold=0.99), "no_repair"),
            ]:
                results = evaluate_config(d, denoiser, policy, verifier,
                                          f"{dname}_{pname}_verifier_{vmode}")
                # Only record wr_fixed for this experiment
                r = results.get(wr_fixed, {})
                key = f"verifier_deg_{dname}_{pname}_{vmode}"
                all_results[key] = {
                    "denoiser": dname, "policy": pname, "verifier_mode": vmode,
                    "wrong_ratio": wr_fixed,
                    "success_rate": r.get("success_rate", -1),
                    "success_se": r.get("success_se", 0),
                }
                print(f"    -> {dname} {pname} with {vmode}: "
                      f"sr={r.get('success_rate', -1):.4f}")

    # ---- Save ----
    all_results["_config"] = {
        "domain": "json_v2",
        "wrong_ratios": WRONG_RATIOS,
        "mask_ratio": MASK_RATIO,
        "num_trials": NUM_TRIALS,
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"stress_test_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

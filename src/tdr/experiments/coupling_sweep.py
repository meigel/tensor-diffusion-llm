"""\
Coupling-density sweep: find the sweet spot where verifier repair
gives 50-80% vs 20-40% no-repair.

Incrementally adds coupled constraints from JSON-v3 on top of JSON-v2
base, measuring repair lift and residual informativeness at each level.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.coupling_sweep
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdr import MASK
from tdr.domains.json_schema_v2 import (
    FIELD_NAMES, FIELD_DOMAINS, FIELD_TO_IDX, N_FIELDS, MAX_DOMAIN,
    encode_instance, decode_array, PROVIDER_REGIONS,
)
from tdr.domains.base import FiniteReasoningDomain, VerifierDiagnostics, Factor
from tdr.domains.json_schema_v3 import check_constraints_v2, check_constraints_v3
from tdr.training.datasets import DenoisingDataset
from tdr.training.train_denoiser import MLPDenoiser, train_epoch
from tdr.diffusion.denoisers import LearnedDenoiser
from tdr.diffusion.transformer_mdlm import TransformerDenoiserModel, MDLMTransformerDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Configurable coupled domain
# -----------------------------------------------------------------------

# The 7 coupled constraints from v3, each as (name, check_fn_fields)
COUPLED_CONSTRAINTS = [
    ("C9_free_compliance", ["cost_tier", "compliance"]),
    ("C10_database_free", ["service_type", "cost_tier"]),
    ("C11_prod_encryption_retention", ["environment", "encryption", "backup_retention_days"]),
    ("C12_multi_az_free", ["multi_az", "cost_tier"]),
    ("C13_serverless_compliance", ["service_type", "compliance"]),
    ("C14_large_storage_tier", ["storage_gb", "cost_tier"]),
    ("C15_max_instances_tier", ["max_instances", "cost_tier"]),
]


def check_constraints_coupled(instance, num_coupled: int):
    """Check v2 constraints + first N coupled constraints."""
    violations = check_constraints_v2(instance)
    if num_coupled <= 0:
        return violations

    comp = instance.get("compliance")
    cost = instance.get("cost_tier")
    env = instance.get("environment")
    st = instance.get("service_type")
    encryption = instance.get("encryption")
    multi_az = instance.get("multi_az")
    storage = instance.get("storage_gb")
    mx = instance.get("max_instances")

    # Always check all 7 but only count first num_coupled
    coupled_violations = []

    # C9
    if cost == "free" and comp in ("hipaa", "pci_dss"):
        coupled_violations.append(("free incompatible with compliance", ["cost_tier", "compliance"]))

    # C10
    if st == "database" and cost == "free":
        coupled_violations.append(("database incompatible with free tier", ["service_type", "cost_tier"]))

    # C11
    if env == "production" and encryption == "enabled":
        ret = instance.get("backup_retention_days")
        if ret is not None and ret < 90:
            coupled_violations.append(("production+encryption requires retention >= 90",
                                       ["environment", "encryption", "backup_retention_days"]))

    # C12
    if multi_az is True and cost == "free":
        coupled_violations.append(("multi-AZ incompatible with free tier", ["multi_az", "cost_tier"]))

    # C13
    if st == "serverless" and comp in ("hipaa", "pci_dss"):
        coupled_violations.append(("serverless incompatible with compliance", ["service_type", "compliance"]))

    # C14
    if storage is not None and storage > 2000 and cost not in ("enterprise", "premium", None):
        coupled_violations.append(("storage > 2000GB requires enterprise/premium", ["storage_gb", "cost_tier"]))

    # C15
    if mx is not None and mx > 20 and cost != "enterprise":
        coupled_violations.append(("max_instances > 20 requires enterprise", ["max_instances", "cost_tier"]))

    violations.extend(coupled_violations[:num_coupled])
    return violations


class CoupledJsonDomain(FiniteReasoningDomain):
    """JSON domain with configurable coupling density."""
    def __init__(self, num_coupled: int = 0):
        self.num_coupled = num_coupled

    def num_variables(self): return N_FIELDS
    def domain_size(self, i): return len(FIELD_DOMAINS[FIELD_NAMES[i]])
    def max_domain_size(self): return MAX_DOMAIN

    def sample_solution(self, rng):
        """Generate valid config satisfying v2 + first N coupled constraints."""
        for attempt in range(200):
            provider = rng.choice(FIELD_DOMAINS["provider"])
            region = rng.choice(PROVIDER_REGIONS[provider])
            st = rng.choice(FIELD_DOMAINS["service_type"])
            env = rng.choice(FIELD_DOMAINS["environment"])
            cost_tier = rng.choice(FIELD_DOMAINS["cost_tier"])
            comp = rng.choice(FIELD_DOMAINS["compliance"])

            # Resolve contradictions for active constraints
            if self.num_coupled >= 1 and cost_tier == "free" and comp in ("hipaa", "pci_dss"):
                if rng.random() < 0.5: cost_tier = "standard"
                else: comp = "none"

            if self.num_coupled >= 2 and st == "database" and cost_tier == "free":
                cost_tier = "standard"

            multi_az = bool(rng.choice([True, False]))
            if self.num_coupled >= 4 and cost_tier == "free":
                multi_az = False

            if self.num_coupled >= 3:
                pass  # handled below

            if self.num_coupled >= 5 and st == "serverless" and comp in ("hipaa", "pci_dss"):
                if rng.random() < 0.5: st = "compute"
                else: comp = "none"

            encryption = rng.choice(FIELD_DOMAINS["encryption"])
            backup_enabled = bool(rng.choice([True, False]))
            if encryption == "enabled" or st == "database":
                multi_az = True; backup_enabled = True
            if comp in ("hipaa", "pci_dss"):
                encryption = "enabled"; backup_enabled = True

            max_instances = int(rng.choice(FIELD_DOMAINS["max_instances"]))
            if self.num_coupled >= 7 and max_instances > 20:
                cost_tier = "enterprise"

            instance_size = rng.choice(FIELD_DOMAINS["instance_size"])
            auto_scaling = bool(rng.choice([True, False]))
            if cost_tier == "free":
                instance_size = rng.choice(["small", "medium"])
                auto_scaling = False; max_instances = min(max_instances, 2)

            if st == "serverless":
                auto_scaling = True; max_instances = max(max_instances, 5); instance_size = "small"

            storage_gb = int(rng.choice(FIELD_DOMAINS["storage_gb"]))
            if self.num_coupled >= 6 and storage_gb > 2000:
                if cost_tier not in ("enterprise", "premium"):
                    cost_tier = rng.choice(["enterprise", "premium"])

            if storage_gb > 1000:
                instance_size = rng.choice(["large", "xlarge"])
                memory_gb = int(rng.choice([m for m in FIELD_DOMAINS["memory_gb"] if m >= 16]))
            else:
                memory_gb = int(rng.choice(FIELD_DOMAINS["memory_gb"]))

            if st == "database": storage_gb = max(storage_gb, 100)

            backup_retention_days = int(rng.choice(FIELD_DOMAINS["backup_retention_days"]))
            logging_level = rng.choice(FIELD_DOMAINS["logging_level"])
            monitoring = rng.choice(FIELD_DOMAINS["monitoring"])

            if env == "production":
                backup_enabled = True
                backup_retention_days = max(backup_retention_days, 30)
                if monitoring == "disabled": monitoring = "basic"
                if logging_level == "debug": logging_level = "info"

            if self.num_coupled >= 3 and env == "production" and encryption == "enabled":
                backup_retention_days = max(backup_retention_days, 90)

            if comp in ("hipaa", "pci_dss"):
                if monitoring == "disabled": monitoring = "basic"
                if logging_level == "debug": logging_level = "info"

            cpu_cores = int(rng.choice(FIELD_DOMAINS["cpu_cores"]))

            instance = {
                "provider": provider, "region": region,
                "service_type": st, "instance_size": instance_size,
                "cpu_cores": cpu_cores, "memory_gb": memory_gb,
                "storage_gb": storage_gb, "encryption": encryption,
                "backup_enabled": backup_enabled,
                "backup_retention_days": backup_retention_days,
                "multi_az": multi_az, "environment": env,
                "auto_scaling": auto_scaling, "max_instances": max_instances,
                "logging_level": logging_level, "monitoring": monitoring,
                "cost_tier": cost_tier, "compliance": comp,
            }

            violations = check_constraints_coupled(instance, self.num_coupled)
            if len(violations) == 0:
                return encode_instance(instance)

        raise RuntimeError(f"Failed to generate after 200 attempts (coupled={self.num_coupled})")

    def verifier(self, x):
        n = self.num_variables()
        instance = decode_array(x)
        violations = check_constraints_coupled(instance, self.num_coupled)
        field_counts = {}
        for desc, fields in violations:
            for f in fields:
                idx = FIELD_TO_IDX[f]
                field_counts[idx] = field_counts.get(idx, 0) + 1
        local_residuals = np.zeros(n, dtype=np.int64)
        for idx, count in field_counts.items():
            local_residuals[idx] = count
        return VerifierDiagnostics(len(violations), local_residuals)

    def build_factors(self): return []
    def enumerate_solutions(self): raise NotImplementedError


# -----------------------------------------------------------------------
# Evaluation + residual analysis
# -----------------------------------------------------------------------

def adversarial_corrupt(domain, x_true, mask_ratio, wrong_ratio, rng):
    """Corrupt with bias toward high-conflict fields."""
    HIGH_CONFLICT = [
        FIELD_TO_IDX["cost_tier"], FIELD_TO_IDX["compliance"],
        FIELD_TO_IDX["service_type"], FIELD_TO_IDX["environment"],
        FIELD_TO_IDX["encryption"], FIELD_TO_IDX["multi_az"],
    ]
    x = domain.corrupt(x_true, mask_ratio, rng)
    observed = np.where(x != MASK)[0]
    n_corrupt = max(1, int(len(observed) * wrong_ratio))
    if n_corrupt > 0 and wrong_ratio > 0:
        conflict = [i for i in observed if i in HIGH_CONFLICT]
        others = [i for i in observed if i not in HIGH_CONFLICT]
        rng.shuffle(conflict); rng.shuffle(others)
        chosen = conflict[:n_corrupt]
        if len(chosen) < n_corrupt:
            chosen.extend(others[:n_corrupt - len(chosen)])
        for idx in chosen:
            ds = domain.domain_size(idx)
            wrong = [v for v in range(ds) if v != x_true[idx]]
            if wrong: x[idx] = rng.choice(wrong)
    return x


def measure_residual_precision(domain, x_corrupt, x_true, verifier):
    """Measure how precisely residuals identify corrupted positions.

    Precision: fraction of residual-positive positions that are actually corrupted.
    Recall: fraction of corrupted positions that have positive residual.
    """
    diag = verifier(x_corrupt)
    residuals = diag.local_residuals
    corrupted = (x_corrupt != MASK) & (x_corrupt != x_true)
    residual_pos = residuals > 0

    tp = np.sum(corrupted & residual_pos)
    fp = np.sum(~corrupted & residual_pos)
    fn = np.sum(corrupted & ~residual_pos)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(precision), float(recall)


def run_eval(domain, denoiser, num_trials=100, mask_ratio=0.3, wrong_ratio=0.4, max_steps=10):
    """Evaluate repair and measure residual precision."""
    policy_repair = VerifierRepairPolicy(remask_threshold=1)
    policy_no_repair = ConfidenceUnmaskPolicy(threshold=0.99)

    results = {"repair": [], "no_repair": [], "precision": [], "recall": []}

    for seed in range(num_trials):
        rng = np.random.default_rng(seed)
        x_true = domain.sample_solution(rng)
        x_corrupt = adversarial_corrupt(domain, x_true, mask_ratio, wrong_ratio, rng)

        # Measure residual precision on the corrupted state
        prec, rec = measure_residual_precision(domain, x_corrupt, x_true, domain.verifier)
        results["precision"].append(prec)
        results["recall"].append(rec)

        # Repair
        sampler = MaskedDiffusionSampler(
            denoiser=denoiser, mask_policy=policy_repair,
            verifier=domain.verifier, max_steps=max_steps,
        )
        result = sampler.run(x_corrupt, rng)
        diag = domain.verifier(result.x)
        results["repair"].append(int(diag.global_violation == 0 and np.all(result.x != MASK)))

        # No repair
        sampler2 = MaskedDiffusionSampler(
            denoiser=denoiser, mask_policy=policy_no_repair,
            verifier=domain.verifier, max_steps=max_steps,
        )
        result2 = sampler2.run(x_corrupt, rng)
        diag2 = domain.verifier(result2.x)
        results["no_repair"].append(int(diag2.global_violation == 0 and np.all(result2.x != -1)))

    return results


def main():
    print("=" * 60)
    print("COUPLING-DENSITY SWEEP")
    print("=" * 60)

    device = torch.device("cpu")
    all_results = {}

    # Train a single MLP on the BASE domain (v2, no coupling)
    # This denoiser is used for all coupling levels
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
    print("  Training complete.")

    # Sweep coupling levels
    coupling_levels = list(range(0, 8))  # 0 = v2, 7 = v3

    for nc in coupling_levels:
        domain = CoupledJsonDomain(num_coupled=nc)
        cname = COUPLED_CONSTRAINTS[nc-1][0] if 1 <= nc <= len(COUPLED_CONSTRAINTS) else 'full_v3'
        print(f"\n--- Coupling level {nc}/7 ({cname}) ---")

        results = run_eval(domain, mlp_wrap, num_trials=100, wrong_ratio=0.4)

        sr_repair = float(np.mean(results["repair"]))
        sr_no_repair = float(np.mean(results["no_repair"]))
        se_repair = float(np.sqrt(sr_repair * (1 - sr_repair) / 99))
        se_no_repair = float(np.sqrt(sr_no_repair * (1 - sr_no_repair) / 99))
        avg_prec = float(np.mean(results["precision"]))
        avg_recall = float(np.mean(results["recall"]))
        lift = sr_repair - sr_no_repair

        entry = {
            "num_coupled": nc,
            "constraint_name": COUPLED_CONSTRAINTS[nc-1][0] if nc > 0 else "base_v2",
            "constraint_fields": COUPLED_CONSTRAINTS[nc-1][1] if nc > 0 else [],
            "repair_success": sr_repair,
            "repair_se": se_repair,
            "no_repair_success": sr_no_repair,
            "no_repair_se": se_no_repair,
            "lift": lift,
            "residual_precision": avg_prec,
            "residual_recall": avg_recall,
        }
        all_results[f"coupled_{nc}"] = entry

        print(f"  No repair: {sr_no_repair:.3f} +/- {se_no_repair:.3f}")
        print(f"  Repair:    {sr_repair:.3f} +/- {se_repair:.3f}")
        print(f"  Lift:      {lift:.3f}")
        print(f"  Residual precision: {avg_prec:.3f}, recall: {avg_recall:.3f}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"coupling_sweep_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")

    # Summary
    print("\n" + "=" * 60)
    print("SWEET SPOT ANALYSIS")
    print("=" * 60)
    for nc in coupling_levels:
        e = all_results[f"coupled_{nc}"]
        print(f"  Coupled={nc}: no_repair={e['no_repair_success']:.3f}, "
              f"repair={e['repair_success']:.3f}, lift={e['lift']:.3f}, "
              f"prec={e['residual_precision']:.3f}, rec={e['residual_recall']:.3f}")


if __name__ == "__main__":
    main()

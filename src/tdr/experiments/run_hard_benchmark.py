"""\
Hard benchmark: JSON-v3 with edit budget + adversarial corruptions.

Per Codex's design, this creates a regime where verifier repair
achieves ~60-80% instead of 100%, enabling method comparison.

Key design elements:
1. Adversarial corruptions: target high-conflict fields (cost_tier,
   compliance, service_type, environment)
2. Edit-distance success: require valid AND within k edits of input
3. Limited repair budget: max 2-4 remasks per step, 3-5 steps max
4. Compare verifier repair vs group-only vs pass/fail vs no repair

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.run_hard_benchmark
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdr import MASK
from tdr.domains.json_schema_v3 import (
    JsonSchemaV3Domain, FIELD_NAMES, FIELD_TO_IDX, FIELD_DOMAINS,
    check_constraints_v3, decode_array,
)
from tdr.domains.base import VerifierDiagnostics
from tdr.training.datasets import DenoisingDataset, compute_denoising_accuracy
from tdr.training.train_denoiser import MLPDenoiser, train_epoch
from tdr.diffusion.denoisers import LearnedDenoiser, RandomDenoiser
from tdr.diffusion.transformer_mdlm import TransformerDenoiserModel, MDLMTransformerDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy, BaseMaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

d = JsonSchemaV3Domain()
N = d.num_variables()
D = d.max_domain_size()
DOMAIN_SIZES = [d.domain_size(i) for i in range(N)]
INPUT_DIM = N * (D + 1)
OUTPUT_DIM = N * D

# Fields that cause cascading violations when corrupted
HIGH_CONFLICT_FIELDS = [
    FIELD_TO_IDX["cost_tier"],
    FIELD_TO_IDX["compliance"],
    FIELD_TO_IDX["service_type"],
    FIELD_TO_IDX["environment"],
    FIELD_TO_IDX["encryption"],
    FIELD_TO_IDX["multi_az"],
]

MASK_RATIO = 0.3
WRONG_RATIOS = [0.0, 0.4, 0.6, 0.8]
NUM_TRIALS = 100
MAX_STEPS = 5         # tight budget
MAX_REMASK = 3        # max positions to remask per step
MAX_EDIT_DISTANCE = 4 # success requires valid AND close to input


def hamming(x, y):
    """Count positions where two arrays differ (ignoring MASK)."""
    return int(np.sum((x != MASK) & (y != MASK) & (x != y)))


def adversarial_corrupt(x_true, mask_ratio, wrong_ratio, rng):
    """Corrupt with bias toward high-conflict fields."""
    x = d.corrupt(x_true, mask_ratio, rng)

    # Identify observed positions
    observed = np.where(x != MASK)[0]
    n_corrupt = max(1, int(len(observed) * wrong_ratio))

    if n_corrupt > 0 and wrong_ratio > 0:
        # Prioritize high-conflict fields
        conflict_candidates = [i for i in observed if i in HIGH_CONFLICT_FIELDS]
        other_candidates = [i for i in observed if i not in HIGH_CONFLICT_FIELDS]
        rng.shuffle(conflict_candidates)
        rng.shuffle(other_candidates)

        # Fill corruption slots: first from conflict fields, then others
        chosen = []
        chosen.extend(conflict_candidates[:n_corrupt])
        if len(chosen) < n_corrupt:
            remaining = n_corrupt - len(chosen)
            chosen.extend(other_candidates[:remaining])

        # Apply corruptions
        for idx in chosen:
            ds = d.domain_size(idx)
            wrong_vals = [v for v in range(ds) if v != x_true[idx]]
            if wrong_vals:
                x[idx] = rng.choice(wrong_vals)

    return x


# -----------------------------------------------------------------------
# Restricted remask policy (budgeted)
# -----------------------------------------------------------------------

class BudgetedVerifierRepair(VerifierRepairPolicy):
    """Verifier repair with per-step remask budget."""
    def __init__(self, max_remask=3, **kwargs):
        super().__init__(**kwargs)
        self.max_remask = max_remask
        self.top_k = max_remask


# -----------------------------------------------------------------------
# Degraded verifiers (same as stress_test)
# -----------------------------------------------------------------------

def make_degraded_verifier(domain, mode):
    if mode == "exact":
        return domain.verifier
    if mode == "group_only":
        def v(x):
            inst = decode_array(x); vios = check_constraints_v3(inst); n = domain.num_variables()
            gv = len(vios)
            return VerifierDiagnostics(gv, np.ones(n, dtype=np.int64) if gv>0 else np.zeros(n, dtype=np.int64))
        return v
    if mode == "pass_fail":
        def v(x):
            inst = decode_array(x); vios = check_constraints_v3(inst)
            return VerifierDiagnostics(len(vios), np.zeros(domain.num_variables(), dtype=np.int64))
        return v
    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------
# Training + experiment
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HARD BENCHMARK — JSON-v3")
    print("=" * 60)
    print(f"\nN={N}, D={D}, domain_sizes={DOMAIN_SIZES}")
    print(f"High-conflict fields: {[FIELD_NAMES[i] for i in HIGH_CONFLICT_FIELDS]}")
    print(f"Max steps={MAX_STEPS}, max_remask={MAX_REMASK}, edit_dist<={MAX_EDIT_DISTANCE}")

    device = torch.device("cpu")

    # ---- Train MLP ----
    print("\n--- Training MLP ---")
    mlp = MLPDenoiser(INPUT_DIM, OUTPUT_DIM, [256, 256])
    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_data = DenoisingDataset(d, 20000, MASK_RATIO, rng_seed=0)
    loader = train_data.get_dataloader(64, shuffle=True)

    for epoch in range(30):
        train_epoch(mlp, loader, opt, crit, device, D)
        if (epoch+1) % 10 == 0:
            acc = compute_denoising_accuracy(d, mlp, 500, MASK_RATIO, device)
            print(f"  Epoch {epoch+1:2d}/30: acc={acc:.4f}")

    ckpt_path = CKPT_DIR / "denoiser_mlp_jsonv3.pt"
    torch.save(mlp.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # ---- Train MDLM ----
    print("\n--- Training MDLM ---")
    mdlm = TransformerDenoiserModel(n=N, d=D, embed_dim=128, nhead=4, num_layers=3)
    opt2 = optim.Adam(mdlm.parameters(), lr=1e-3)

    for epoch in range(30):
        train_epoch(mdlm, loader, opt2, crit, device, D)
        if (epoch+1) % 10 == 0:
            acc = compute_denoising_accuracy(d, mdlm, 500, MASK_RATIO, device)
            print(f"  Epoch {epoch+1:2d}/30: acc={acc:.4f}")

    ckpt_path2 = CKPT_DIR / "denoiser_mdlm_jsonv3.pt"
    torch.save(mdlm.state_dict(), ckpt_path2)
    print(f"Saved: {ckpt_path2}")

    mlp.eval(); mdlm.eval()
    mlp_wrap = LearnedDenoiser(mlp, N, D)
    mdlm_wrap = MDLMTransformerDenoiser(mdlm, N, D, domain_sizes=DOMAIN_SIZES)
    rand_wrap = RandomDenoiser(d)

    # ---- Evaluation ----
    all_results = {}

    configs = [
        ("mlp", mlp_wrap, "exact"),
        ("mlp", mlp_wrap, "group_only"),
        ("mlp", mlp_wrap, "pass_fail"),
        ("mdlm", mdlm_wrap, "exact"),
        ("mdlm", mdlm_wrap, "group_only"),
        ("mdlm", mdlm_wrap, "pass_fail"),
        ("random", rand_wrap, "exact"),
        ("random", rand_wrap, "pass_fail"),
    ]

    print("\n" + "=" * 60)
    print("EVALUATION (adversarial corruptions, edit-distance success)")
    print("=" * 60)

    for denoiser_name, denoiser, vmode in configs:
        print(f"\n  --- {denoiser_name} + {vmode} ---")
        verifier = make_degraded_verifier(d, vmode)
        policy = BudgetedVerifierRepair(max_remask=MAX_REMASK) if vmode != "pass_fail" else ConfidenceUnmaskPolicy(0.99)
        if vmode == "pass_fail":
            # Pass/fail: no localization, use default fill
            no_repair_policy = ConfidenceUnmaskPolicy(0.99)

        for wr in WRONG_RATIOS:
            for use_repair in [True, False]:
                # For pass_fail, repair mode = same as no_repair (no localization signal)
                if vmode == "pass_fail" and use_repair:
                    p = ConfidenceUnmaskPolicy(0.99)  # no improvement possible
                elif use_repair:
                    p = BudgetedVerifierRepair(max_remask=MAX_REMASK)
                else:
                    p = ConfidenceUnmaskPolicy(0.99)

                successes = []
                edit_dists = []
                for seed in range(NUM_TRIALS):
                    rng = np.random.default_rng(seed)
                    x_true = d.sample_solution(rng)
                    x_corrupt = adversarial_corrupt(x_true, MASK_RATIO, wr, rng)

                    sampler = MaskedDiffusionSampler(
                        denoiser=denoiser, mask_policy=p,
                        verifier=verifier, max_steps=MAX_STEPS,
                    )
                    result = sampler.run(x_corrupt, rng)
                    final_x = result.x
                    diag = d.verifier(final_x)

                    valid = bool(diag.global_violation == 0 and np.all(final_x != MASK))
                    dist = hamming(x_corrupt, final_x)
                    edit_dists.append(dist)
                    successes.append(int(valid and dist <= MAX_EDIT_DISTANCE))

                sr = float(np.mean(successes))
                se = float(np.sqrt(sr * (1 - sr) / max(NUM_TRIALS - 1, 1)))
                avg_edit = float(np.mean(edit_dists))

                label = f"{denoiser_name}_{vmode}{'_repair' if use_repair else '_norepair'}"
                key = f"{label}_wr{wr:.2f}"
                all_results[key] = {
                    "denoiser": denoiser_name, "verifier_mode": vmode,
                    "repair": use_repair, "wrong_ratio": wr,
                    "success_rate": sr, "success_se": se,
                    "avg_edit_distance": avg_edit,
                    "num_trials": NUM_TRIALS,
                }
                print(f"    wr={wr:.1f} {'repair' if use_repair else 'no_rep':8s}: "
                      f"sr={sr:.4f}±{se:.4f}  avg_edit={avg_edit:.1f}")

    # ---- Save ----
    all_results["_config"] = {
        "domain": "json_v3",
        "max_steps": MAX_STEPS,
        "max_remask": MAX_REMASK,
        "max_edit_distance": MAX_EDIT_DISTANCE,
        "wrong_ratios": WRONG_RATIOS,
        "num_trials": NUM_TRIALS,
        "high_conflict_fields": [FIELD_NAMES[i] for i in HIGH_CONFLICT_FIELDS],
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"hard_benchmark_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

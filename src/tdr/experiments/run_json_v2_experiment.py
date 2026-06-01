"""\
Joint training + repair experiment for JSON-v2 domain.

Trains MLP and MDLM denoisers on JSON-v2 data, then evaluates
all denoiser × policy combinations.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.run_json_v2_experiment
"""

import os, sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tdr import MASK
from tdr.domains.json_schema_v2 import JsonSchemaV2Domain, FIELD_NAMES
from tdr.training.datasets import DenoisingDataset, compute_denoising_accuracy
from tdr.training.train_denoiser import MLPDenoiser, train_epoch
from tdr.diffusion.denoisers import LearnedDenoiser, RandomDenoiser
from tdr.diffusion.transformer_mdlm import TransformerDenoiserModel, MDLMTransformerDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy, RandomRemaskPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

d = JsonSchemaV2Domain()
N = d.num_variables()
D = d.max_domain_size()
DOMAIN_SIZES = [d.domain_size(i) for i in range(N)]
WRONG_RATIOS = [0.0, 0.2, 0.4]
MASK_RATIO = 0.3
NUM_TRIALS = 100


def eval_denoiser(domain, denoiser, label):
    """Evaluate denoiser with multiple policies."""
    print(f"\n  --- {label} ---")
    results = {}

    policies = {
        "no_repair": ConfidenceUnmaskPolicy(threshold=0.99),
        "verifier_repair": VerifierRepairPolicy(remask_threshold=1),
        "confidence_remask": RandomRemaskPolicy(remask_fraction=0.25),
    }

    for pname, policy in policies.items():
        for wr in WRONG_RATIOS:
            successes = []
            for seed in range(NUM_TRIALS):
                rng = np.random.default_rng(seed)
                x_true = domain.sample_solution(rng)
                x_corrupt = domain.mixed_corrupt(x_true, MASK_RATIO, wr, rng)
                sampler = MaskedDiffusionSampler(
                    denoiser=denoiser, mask_policy=policy,
                    verifier=domain.verifier, max_steps=20,
                )
                result = sampler.run(x_corrupt, rng)
                diag = domain.verifier(result.x)
                successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))

            sr = float(np.mean(successes))
            se = float(np.sqrt(sr * (1 - sr) / max(NUM_TRIALS - 1, 1)))
            key = f"{label}_{pname}_wr{wr:.2f}"
            results[key] = {
                "method": label, "policy": pname, "wrong_ratio": wr,
                "success_rate": sr, "success_se": se, "num_trials": NUM_TRIALS,
            }
            print(f"    {pname:20s} wr={wr:.1f}: success={sr:.4f}±{se:.4f}")

    return results


def main():
    print("=" * 60)
    print("JSON-v2 EXPERIMENT")
    print("=" * 60)
    print(f"\nDomain: {N} fields, max_d={D}, domain_sizes={DOMAIN_SIZES}")

    device = torch.device("cpu")

    # ---- Train MLP ----
    print("\n--- Training MLP ---")
    input_dim = N * (D + 1)
    output_dim = N * D
    mlp = MLPDenoiser(input_dim, output_dim, [256, 256])
    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    train_data = DenoisingDataset(d, 20000, MASK_RATIO, rng_seed=0)
    loader = train_data.get_dataloader(64, shuffle=True)

    for epoch in range(30):
        train_epoch(mlp, loader, opt, crit, device, D)
        if (epoch + 1) % 10 == 0:
            acc = compute_denoising_accuracy(d, mlp, 500, MASK_RATIO, device)
            print(f"  Epoch {epoch+1:2d}/30: acc={acc:.4f}")

    ckpt_path = CKPT_DIR / "denoiser_mlp_jsonv2.pt"
    torch.save(mlp.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # ---- Train MDLM ----
    print("\n--- Training MDLM ---")
    mdlm = TransformerDenoiserModel(n=N, d=D, embed_dim=128, nhead=4, num_layers=3)
    opt2 = optim.Adam(mdlm.parameters(), lr=1e-3)

    for epoch in range(30):
        train_epoch(mdlm, loader, opt2, crit, device, D)
        if (epoch + 1) % 10 == 0:
            acc = compute_denoising_accuracy(d, mdlm, 500, MASK_RATIO, device)
            print(f"  Epoch {epoch+1:2d}/30: acc={acc:.4f}")

    ckpt_path2 = CKPT_DIR / "denoiser_mdlm_jsonv2.pt"
    torch.save(mdlm.state_dict(), ckpt_path2)
    print(f"Saved: {ckpt_path2}")

    # ---- Evaluate ----
    all_results = {}

    # Learned denoisers
    mlp.eval()
    mlp_wrapper = LearnedDenoiser(mlp, N, D)
    all_results.update(eval_denoiser(d, mlp_wrapper, "mlp"))

    mdlm.eval()
    mdlm_wrapper = MDLMTransformerDenoiser(mdlm, N, D, domain_sizes=DOMAIN_SIZES)
    all_results.update(eval_denoiser(d, mdlm_wrapper, "mdlm"))

    # Random baseline
    rand = RandomDenoiser(d)
    all_results.update(eval_denoiser(d, rand, "random"))

    # ---- Save ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"jsonv2_experiment_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")
    print("Done.")


if __name__ == "__main__":
    main()

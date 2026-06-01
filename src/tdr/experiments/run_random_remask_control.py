"""Random-remasking control: isolates whether localization matters."""
import json, numpy as np, time
from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.denoisers import TNMarginalDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy, RandomRemaskPolicy
from pathlib import Path

domain = Sudoku4Domain()
backend = ContractionMarginalBackend(domain)
tn = TNMarginalDenoiser(backend)
wrong_ratios = [0.0, 0.1, 0.2, 0.3]
num_trials = 100

results = {}
for label, policy in [
    ("tn_no_repair", ConfidenceUnmaskPolicy(threshold=0.99)),
    ("tn_repair", VerifierRepairPolicy(remask_threshold=1)),
    ("tn_random_remask", RandomRemaskPolicy(remask_fraction=0.25)),
]:
    print(f"\n--- {label} ---")
    for wr in wrong_ratios:
        successes = []
        for seed in range(num_trials):
            rng = np.random.default_rng(seed)
            x_true = domain.sample_solution(rng)
            x_corrupt = domain.mixed_corrupt(x_true, 0.5, wr, rng)
            sampler = MaskedDiffusionSampler(denoiser=tn, mask_policy=policy, verifier=domain.verifier, max_steps=20)
            result = sampler.run(x_corrupt, rng)
            diag = domain.verifier(result.x)
            successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))
        sr = np.mean(successes)
        se = np.sqrt(sr*(1-sr)/max(num_trials-1,1))
        print(f"  wr={wr:.1f}: success={sr:.3f}+-{se:.3f}")

# Save
logs_dir = Path(__file__).resolve().parents[2] / "results" / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
json.dump(results, open(logs_dir/"random_remask_control.json","w"), indent=2)
print("\nDone.")

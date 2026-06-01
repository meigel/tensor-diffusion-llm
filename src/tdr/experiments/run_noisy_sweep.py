"""Denoiser-quality sweep: repair vs non-repair across noise levels."""
import json, numpy as np
from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.denoisers import NoisyDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

domain = Sudoku4Domain()
backend = ContractionMarginalBackend(domain)
sigma_vals = [0.0, 0.25, 0.5, 1.0, 2.0]
wrong_ratios = [0.0, 0.1, 0.2]
num_trials = 50
mask_ratio = 0.5

results = {}
for sigma in sigma_vals:
    noisy = NoisyDenoiser(backend, sigma=sigma)
    for use_repair in [False, True]:
        label = f"noisy_s{sigma}" + ("_repair" if use_repair else "")
        policy = VerifierRepairPolicy(remask_threshold=1) if use_repair else ConfidenceUnmaskPolicy(threshold=0.99)
        print(f"\n--- {label} ---")
        
        for wr in wrong_ratios:
            successes = []
            for seed in range(num_trials):
                rng = np.random.default_rng(seed)
                x_true = domain.sample_solution(rng)
                x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wr, rng)
                sampler = MaskedDiffusionSampler(denoiser=noisy, mask_policy=policy, verifier=domain.verifier, max_steps=20)
                result = sampler.run(x_corrupt, rng)
                diag = domain.verifier(result.x)
                successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))
            sr = np.mean(successes)
            print(f"  wr={wr:.1f}: success={sr:.3f}")

json.dump(results, open("/tmp/noisy_sweep.json", "w"), indent=2)
print("\nDone. Results saved to /tmp/noisy_sweep.json")

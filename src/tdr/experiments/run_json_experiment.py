"""Quick test + train JSON denoiser + run experiment."""
import numpy as np, time
from tdr import MASK
from tdr.domains.json_schema import JsonSchemaDomain, FIELD_NAMES, FIELD_TO_IDX, FIELD_DOMAINS, decode_array
from tdr.diffusion.denoisers import LearnedDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.verifier_policy import VerifierRepairPolicy, RandomRemaskPolicy, ConfidenceFillThenRemask
from tdr.policies.entropy_policy import BaseMaskPolicy
from pathlib import Path
import torch, torch.nn as nn

d = JsonSchemaDomain()
n, max_d = d.num_variables(), d.max_domain_size()
print(f"Domain: {n} vars, max_d={max_d}")
print(f"Field sizes: {[d.domain_size(i) for i in range(n)]}")

# Verify cross-field constraint: admin needs high clearance
x_admin = np.array([
    FIELD_DOMAINS["username"].index("alice"), 18,
    FIELD_DOMAINS["role"].index("admin"),
    FIELD_DOMAINS["clearance"].index("low"),  # WRONG — should be high
    1,  # True
    FIELD_DOMAINS["tier"].index("pro"),
    FIELD_DOMAINS["region"].index("us"),
], dtype=np.int64)
diag = d.verifier(x_admin)
print(f"Admin+low clearance: violations={diag.global_violation}, residuals={diag.local_residuals}")
assert diag.global_violation >= 1, "Cross-field constraint should fire"

# Train MLP denoiser
input_dim = n * (max_d + 1)
output_dim = n * max_d
print(f"\nTraining MLP: {input_dim} -> [256,256] -> {output_dim}")
model = nn.Sequential(
    nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, output_dim),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

from tdr.training.datasets import DenoisingDataset
train_data = DenoisingDataset(d, 10000, 0.5, rng_seed=0)
loader = train_data.get_dataloader(64, shuffle=True)

for epoch in range(20):
    model.train()
    total_loss = 0.0
    for state, target, mask in loader:
        logits = model(state).reshape(-1, max_d)
        targets_flat = target.reshape(-1).long()
        mask_flat = mask.reshape(-1)
        masked_idx = mask_flat.nonzero(as_tuple=True)[0]
        if len(masked_idx) > 0:
            loss = criterion(logits[masked_idx], targets_flat[masked_idx])
        else:
            loss = torch.tensor(0.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

# Evaluate repair vs no repair vs confidence remask
learned = LearnedDenoiser(model, n, max_d)

class NoRemaskFillAll(BaseMaskPolicy):
    def select_fill(self, x, dist, diagnostics, rng=None):
        return x == MASK

print("\n=== JSON repair experiment ===")
methods = [
    ("no_repair", NoRemaskFillAll()),
    ("verifier_repair", VerifierRepairPolicy(remask_threshold=1)),
]
for wr in [0.0, 0.2, 0.4]:
    for name, policy in methods:
        successes = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            x_true = d.sample_solution(rng)
            x_corrupt = d.mixed_corrupt(x_true, 0.3, wr, rng)
            sampler = MaskedDiffusionSampler(denoiser=learned, mask_policy=policy, verifier=d.verifier, max_steps=20)
            result = sampler.run(x_corrupt, rng)
            diag = d.verifier(result.x)
            successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))
        sr = np.mean(successes)
        print(f"  {name} wr={wr:.1f}: success={sr:.3f}")

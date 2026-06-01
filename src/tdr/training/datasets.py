"""
Dataset generation for training learned denoisers.

Generates pairs of (corrupted_state, target_values) where the denoiser
must predict the correct value at masked positions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain


class DenoisingDataset(Dataset):
    """Dataset of (corrupted_state, clean_state) pairs for denoiser training.

    Each sample is a corrupted state (with MASK entries) and the
    corresponding clean valid assignment. The loss is computed only
    over masked positions.

    Args:
        domain: A FiniteReasoningDomain.
        num_samples: Number of training samples.
        mask_ratio: Probability of masking each position.
        rng_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        domain: FiniteReasoningDomain,
        num_samples: int = 10000,
        mask_ratio: float = 0.5,
        rng_seed: int = 0,
    ):
        self.domain = domain
        self.n = domain.num_variables()
        self.d = domain.max_domain_size()
        self.num_samples = num_samples
        self.mask_ratio = mask_ratio

        # Pre-generate all data
        rng = np.random.default_rng(rng_seed)
        self.states = np.zeros((num_samples, self.n), dtype=np.int64)
        self.targets = np.zeros((num_samples, self.n), dtype=np.int64)
        self.mask_flags = np.zeros((num_samples, self.n), dtype=bool)

        for i in range(num_samples):
            clean = domain.sample_solution(rng)
            corrupted = domain.corrupt(clean, mask_ratio, rng)
            self.states[i] = corrupted
            self.targets[i] = clean
            self.mask_flags[i] = (corrupted == MASK)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._to_tensor(self.states[idx]), \
               torch.from_numpy(self.targets[idx]), \
               torch.from_numpy(self.mask_flags[idx])

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert integer-coded state to one-hot encoding.

        Encoding: d domain values + 1 MASK channel = d+1 channels.
        """
        n, d = self.n, self.d
        one_hot = torch.zeros(n, d + 1, dtype=torch.float32)
        for i in range(n):
            if x[i] == MASK:
                one_hot[i, d] = 1.0  # MASK channel
            else:
                one_hot[i, x[i]] = 1.0  # value channel
        return one_hot.flatten()  # shape (n * (d+1),)

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def compute_denoising_accuracy(
    domain: FiniteReasoningDomain,
    model: torch.nn.Module,
    num_samples: int = 1000,
    mask_ratio: float = 0.5,
    device: str = "cpu",
) -> float:
    """Compute masked-position accuracy of a denoiser on clean (no wrong tokens) data.

    Returns:
        Fraction of masked positions where argmax prediction matches target.
    """
    dataset = DenoisingDataset(domain, num_samples, mask_ratio, rng_seed=999)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for state, target, mask in dataset:
            # state: (n*(d+1),), target: (n,), mask: (n,)
            logits = model(state.unsqueeze(0).to(device))  # (1, n*d)
            logits = logits.reshape(1, dataset.n, dataset.d)

            # Domain-mask invalid classes for heterogeneous domains
            # (e.g. JSON where age has 63 values but role has 4)
            for i in range(dataset.n):
                ds = dataset.domain.domain_size(i)
                if ds < dataset.d:
                    logits[0, i, ds:] = float("-inf")

            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (n,)
            masked_positions = mask.numpy()
            target_np = target.numpy()
            for i in np.where(masked_positions)[0]:
                if preds[i] == target_np[i]:
                    correct += 1
                total += 1
    return correct / max(total, 1)

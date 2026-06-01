"""\
Train a small denoiser on masked completion data.

Supports MLP and transformer MDLM architectures.

Usage:
    python -m tdr.training.train_denoiser --domain sudoku --epochs 50
    python -m tdr.training.train_denoiser --domain sat --n-vars 20 --n-clauses 60 --epochs 50
    python -m tdr.training.train_denoiser --domain json --model mdlm --epochs 30
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.domains.boolsat import BoolSatDomain
from tdr.domains.json_schema import JsonSchemaDomain
from tdr.training.datasets import DenoisingDataset, compute_denoising_accuracy
from tdr.diffusion.denoisers import LearnedDenoiser
from tdr.diffusion.transformer_mdlm import (
    TransformerDenoiserModel,
    MDLMTransformerDenoiser,
)
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class MLPDenoiser(nn.Module):
    """Small MLP for masked denoising.

    Input:  one-hot encoded state, shape (n * (d+1),)
    Output: logits for each position, shape (n * d,)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, dataloader, optimizer, criterion, device, d):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for state, target, mask in dataloader:
        state, target, mask = state.to(device), target.to(device), mask.to(device)

        # Forward
        logits = model(state)  # (batch, n*d)
        batch_size = state.shape[0]
        logits = logits.reshape(-1, d)  # (batch*n, d)
        targets_flat = target.reshape(-1)  # (batch*n,)
        mask_flat = mask.reshape(-1)  # (batch*n,)

        # Loss only over masked positions
        masked_indices = mask_flat.nonzero(as_tuple=True)[0]
        if len(masked_indices) > 0:
            loss = criterion(logits[masked_indices], targets_flat[masked_indices].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        # else: all-observed batch — skip backward/step to avoid
        # backward() on a no-grad tensor

    return total_loss / max(num_batches, 1)


def evaluate(model, domain, num_samples=500, mask_ratio=0.5, device="cpu"):
    """Evaluate masked-position accuracy."""
    return compute_denoising_accuracy(domain, model, num_samples, mask_ratio, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["sudoku", "sat", "json"], default="sudoku")
    parser.add_argument("--model", choices=["mlp", "mdlm"], default="mlp",
                        help="Denoiser architecture: mlp (flat) or mdlm (transformer)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="MLP hidden dims (only used with --model mlp)")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="Transformer embedding dim (only used with --model mdlm)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Transformer attention heads (only used with --model mdlm)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Transformer encoder layers (only used with --model mdlm)")
    parser.add_argument("--n-vars", type=int, default=20)
    parser.add_argument("--n-clauses", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create domain
    if args.domain == "sudoku":
        domain = Sudoku4Domain()
        n, d = 16, 4
    elif args.domain == "json":
        domain = JsonSchemaDomain()
        n, d = domain.num_variables(), domain.max_domain_size()
    else:
        domain = BoolSatDomain(n_vars=args.n_vars, n_clauses=args.n_clauses, k=3, formula_seed=0)
        n, d = args.n_vars, 2

    input_dim = n * (d + 1)  # one-hot with MASK
    output_dim = n * d

    print(f"Domain: {args.domain}, n={n}, d={d}")
    print(f"Model: {args.model}", end="")

    # Model
    if args.model == "mdlm":
        model = TransformerDenoiserModel(
            n=n, d=d,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.embed_dim * 4,
        ).to(device)
        print(f" Transformer(embed_dim={args.embed_dim}, nhead={args.nhead}, "
              f"layers={args.num_layers})")
        denoiser_cls = MDLMTransformerDenoiser
        checkpoint_suffix = "mdlm"
    else:
        model = MLPDenoiser(input_dim, output_dim, args.hidden).to(device)
        print(f" MLP {input_dim} -> {args.hidden} -> {output_dim}")
        denoiser_cls = LearnedDenoiser
        checkpoint_suffix = args.domain

    # Data
    train_dataset = DenoisingDataset(domain, args.train_samples, args.mask_ratio, rng_seed=0)
    train_loader = train_dataset.get_dataloader(args.batch_size, shuffle=True)
    val_dataset = DenoisingDataset(domain, 2000, args.mask_ratio, rng_seed=999)
    val_loader = val_dataset.get_dataloader(args.batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.monotonic()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, d)
        val_acc = evaluate(model, domain, num_samples=500, mask_ratio=args.mask_ratio, device=device)
        elapsed = time.monotonic() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc

    # Save checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"denoiser_{checkpoint_suffix}_{args.domain}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nBest val_acc: {best_acc:.4f}")
    print(f"Saved: {checkpoint_path}")

    # Quick repair evaluation
    print("\n=== Repair evaluation ===")
    # Pass per-position domain sizes for heterogeneous domains (mdlm only)
    if args.model == "mdlm":
        domain_sizes = [domain.domain_size(i) for i in range(n)]
        learned = MDLMTransformerDenoiser(model, n, d, domain_sizes=domain_sizes)
    else:
        learned = LearnedDenoiser(model, n, d)
    for wr in [0.0, 0.1, 0.2]:
        for use_repair in [False, True]:
            policy = VerifierRepairPolicy(remask_threshold=1) if use_repair else ConfidenceUnmaskPolicy(threshold=0.99)
            label = "learned_repair" if use_repair else "learned"
            successes = []
            for seed in range(50):
                rng = np.random.default_rng(seed)
                x_true = domain.sample_solution(rng)
                x_corrupt = domain.mixed_corrupt(x_true, args.mask_ratio, wr, rng)
                sampler = MaskedDiffusionSampler(
                    denoiser=learned, mask_policy=policy,
                    verifier=domain.verifier, max_steps=20,
                )
                result = sampler.run(x_corrupt, rng)
                diag = domain.verifier(result.x)
                successes.append(int(diag.global_violation == 0 and np.all(result.x != MASK)))
            sr = np.mean(successes)
            se = np.sqrt(sr * (1 - sr) / 49)
            print(f"  {label} wr={wr:.1f}: success={sr:.3f}±{se:.3f}")


if __name__ == "__main__":
    main()

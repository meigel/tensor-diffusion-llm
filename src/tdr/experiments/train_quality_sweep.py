"""\
Train MLP denoiser with epoch checkpoints for quality sweep.

Trains a standard MLP denoiser but saves checkpoints at specified epochs
so we can evaluate repair benefit at varying denoiser quality levels.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python -m tdr.experiments.train_quality_sweep
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.training.datasets import DenoisingDataset
from tdr.training.train_denoiser import MLPDenoiser, train_epoch
from tdr.diffusion.denoisers import LearnedDenoiser, RandomDenoiser, LocalSudokuDenoiser, TNMarginalDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy
from tdr.tn.marginals import ContractionMarginalBackend

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DOMAIN = Sudoku4Domain()
N, D = 16, 4
INPUT_DIM = N * (D + 1)
OUTPUT_DIM = N * D
MASK_RATIO = 0.5
WRONG_RATIOS = [0.0, 0.1, 0.2]
NUM_TRIALS = 100


def measure_accuracy(domain, model, num_samples=500):
    """Measure one-step masked-position accuracy."""
    rng = np.random.default_rng(42)
    correct = total = 0
    for _ in range(num_samples):
        x_true = domain.sample_solution(rng)
        x_masked = domain.corrupt(x_true, MASK_RATIO, rng)
        # One-hot encode
        oh = torch.zeros(1, N * (D + 1))
        for i in range(N):
            if x_masked[i] == -1:
                oh[0, i * (D + 1) + D] = 1.0
            else:
                oh[0, i * (D + 1) + x_masked[i]] = 1.0
        with torch.no_grad():
            logits = model(oh).reshape(1, N, D)
            preds = logits.argmax(dim=-1).squeeze(0).numpy()
        for i in range(N):
            if x_masked[i] == -1:
                correct += preds[i] == x_true[i]
                total += 1
    return correct / max(total, 1)


def run_repair_trial(domain, denoiser, wrong_ratio, use_repair, seed):
    """Run a single repair trial."""
    rng = np.random.default_rng(seed)
    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, MASK_RATIO, wrong_ratio, rng)
    policy = (
        VerifierRepairPolicy(remask_threshold=1)
        if use_repair
        else ConfidenceUnmaskPolicy(threshold=0.99)
    )
    sampler = MaskedDiffusionSampler(
        denoiser=denoiser, mask_policy=policy,
        verifier=domain.verifier, max_steps=20,
    )
    result = sampler.run(x_corrupt, rng)
    diag = domain.verifier(result.x)
    return int(diag.global_violation == 0 and np.all(result.x != -1))


def evaluate_denoiser(domain, denoiser, label):
    """Evaluate denoiser with and without repair at multiple wrong ratios."""
    print(f"\n  Evaluating {label}...")
    acc = None
    if hasattr(denoiser, 'model') or isinstance(denoiser, (LearnedDenoiser,)):
        m = denoiser.model if hasattr(denoiser, 'model') else denoiser
        acc = measure_accuracy(domain, m)
        print(f"    accuracy={acc:.4f}")

    results = {}
    for wr in WRONG_RATIOS:
        for use_repair in [False, True]:
            successes = []
            for seed in range(NUM_TRIALS):
                successes.append(
                    run_repair_trial(domain, denoiser, wr, use_repair, seed)
                )
            sr = float(np.mean(successes))
            se = float(np.sqrt(sr * (1 - sr) / max(NUM_TRIALS - 1, 1)))
            key = f"{label}{'_repair' if use_repair else ''}_wr{wr:.2f}"
            results[key] = {
                "label": label,
                "accuracy": acc,
                "repair": use_repair,
                "wrong_ratio": wr,
                "success_rate": sr,
                "success_se": se,
                "num_trials": NUM_TRIALS,
            }
            print(f"    {key}: success={sr:.4f}±{se:.4f}")

    return results, acc


def main():
    print("=" * 60)
    print("DENOISER QUALITY SWEEP — Sudoku 4x4")
    print("=" * 60)

    domain = DOMAIN
    backend = ContractionMarginalBackend(domain)

    all_results = {}
    metadata = []

    # ---- Training data ----
    train_dataset = DenoisingDataset(domain, 20000, MASK_RATIO, rng_seed=0)
    train_loader = train_dataset.get_dataloader(64, shuffle=True)

    # ---- Train MLP with epoch checkpoints ----
    model = MLPDenoiser(INPUT_DIM, OUTPUT_DIM, [256, 256])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    checkpoint_epochs = [1, 2, 5, 10, 25, 50]
    print(f"\nTraining MLP (checkpoints at epochs {checkpoint_epochs})...")

    for epoch in range(1, max(checkpoint_epochs) + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, D)

        if epoch in checkpoint_epochs:
            ckpt_path = CHECKPOINT_DIR / f"mlp_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Epoch {epoch}: saved {ckpt_path.name}")

    # ---- Evaluate each checkpoint ----
    print("\n" + "=" * 60)
    print("EVALUATING CHECKPOINTS")
    print("=" * 60)

    for epoch in checkpoint_epochs:
        model = MLPDenoiser(INPUT_DIM, OUTPUT_DIM, [256, 256])
        ckpt_path = CHECKPOINT_DIR / f"mlp_epoch{epoch}.pt"
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        denoiser = LearnedDenoiser(model, N, D)
        label = f"mlp_epoch{epoch}"
        results_row, acc = evaluate_denoiser(domain, denoiser, label)
        all_results.update(results_row)
        metadata.append({"label": label, "epoch": epoch, "accuracy": acc})

    # ---- Evaluate TN ----
    tn = TNMarginalDenoiser(backend)
    results_row, tn_acc = evaluate_denoiser(domain, tn, "tn")
    all_results.update(results_row)
    metadata.append({"label": "tn", "epoch": -1, "accuracy": tn_acc})

    # ---- Evaluate Local ----
    local = LocalSudokuDenoiser(domain)
    results_row, local_acc = evaluate_denoiser(domain, local, "local")
    all_results.update(results_row)
    metadata.append({"label": "local", "epoch": -1, "accuracy": local_acc})

    # ---- Evaluate Random ----
    rand = RandomDenoiser(domain)
    results_row, rand_acc = evaluate_denoiser(domain, rand, "random")
    all_results.update(results_row)
    metadata.append({"label": "random", "epoch": -1, "accuracy": rand_acc})

    # ---- Save results ----
    all_results["_metadata"] = metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"quality_sweep_checkpoints_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {path}")

    # ---- Generate figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Collect unique accuracy levels and their data
        points = {}  # (accuracy, wrong_ratio, is_repair) → success_rate
        for key, entry in all_results.items():
            if key.startswith("_"):
                continue
            acc = entry.get("accuracy")
            wr = entry.get("wrong_ratio")
            repair = entry.get("repair")
            sr = entry.get("success_rate")
            se = entry.get("success_se", 0)
            if acc is not None and wr is not None and repair is not None:
                points.setdefault((acc, wr, repair), (sr, se))

        # Panel A: Repair lift vs accuracy
        ax = axes[0]
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for wr in WRONG_RATIOS:
            x_vals, y_vals = [], []
            accs = sorted(set(a for (a, w, r) in points if w == wr and r is False))
            for acc in accs:
                no_rep = points.get((acc, wr, False))
                rep = points.get((acc, wr, True))
                if no_rep and rep:
                    x_vals.append(acc)
                    y_vals.append(rep[0] - no_rep[0])
            if x_vals:
                ax.plot(x_vals, y_vals, marker="o", label=f"wr={wr:.1f}", linewidth=2, markersize=8)
        ax.set_xlabel("One-step denoiser accuracy")
        ax.set_ylabel("Repair lift (Δ success rate)")
        ax.set_title("Verifier repair benefit vs denoiser quality (Sudoku)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel B: Success rate vs accuracy at wr=0.2
        ax = axes[1]
        wr_target = 0.2
        for is_repair, ls, lbl in [(False, "-", "No repair"), (True, "--", "Verifier repair")]:
            x_vals, y_vals, y_errs = [], [], []
            accs = sorted(set(a for (a, w, r) in points if abs(w - wr_target) < 0.01 and r == is_repair))
            for acc in accs:
                pt = points.get((acc, wr_target, is_repair))
                if pt:
                    x_vals.append(acc)
                    y_vals.append(pt[0])
                    y_errs.append(pt[1])
            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                           marker="o", linestyle=ls, capsize=4,
                           linewidth=2, markersize=8, label=lbl)
        ax.set_xlabel("One-step denoiser accuracy")
        ax.set_ylabel("Success rate")
        ax.set_title(f"Success rate at wrong-token ratio = {wr_target}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = PLOTS_DIR / "quality_sweep_checkpoints.pdf"
        fig.savefig(plot_path)
        print(f"Plot: {plot_path}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available, skipping plot")

    print("\nDone.")


if __name__ == "__main__":
    main()

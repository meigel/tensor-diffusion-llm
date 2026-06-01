"""\
Cross-formula SAT learned denoiser evaluation.

The paper currently claims "Learned + repair = 100% on SAT" but this is
based on a single formula (formula_seed=0). This script evaluates the
trained denoiser across 50 different random formulas to test whether
the result generalizes.

Usage:
    source ~/work/venv/python-ml/bin/activate
    python src/tdr/experiments/eval_sat_learned.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tdr import MASK
from tdr.domains.boolsat import BoolSatDomain
from tdr.training.train_denoiser import MLPDenoiser
from tdr.diffusion.denoisers import LearnedDenoiser
from tdr.diffusion.sampler import MaskedDiffusionSampler
from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy
from tdr.policies.verifier_policy import VerifierRepairPolicy

# Paths
CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "results" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "denoiser_sat.pt"

# Domain parameters
N_VARS = 20
N_CLAUSES = 60
K = 3
N = N_VARS
D = 2  # Boolean
INPUT_DIM = N * (D + 1)  # 60
OUTPUT_DIM = N * D  # 40


def run_trial(domain, denoiser, mask_ratio, wrong_ratio, use_repair, seed, rng):
    """Run a single repair trial on a given domain."""
    x_true = domain.sample_solution(rng)
    x_corrupt = domain.mixed_corrupt(x_true, mask_ratio, wrong_ratio, rng)

    policy = (
        VerifierRepairPolicy(remask_threshold=1)
        if use_repair
        else ConfidenceUnmaskPolicy(threshold=0.99)
    )

    sampler = MaskedDiffusionSampler(
        denoiser=denoiser,
        mask_policy=policy,
        verifier=domain.verifier,
        max_steps=20,
    )

    result = sampler.run(x_corrupt, rng)
    final_x = result.x
    diag = domain.verifier(final_x)
    success = bool(diag.global_violation == 0 and np.all(final_x != MASK))

    return {
        "seed": seed,
        "formula_seed": domain._formula_seed,
        "mask_ratio": mask_ratio,
        "wrong_ratio": wrong_ratio,
        "use_repair": use_repair,
        "success": success,
        "final_violation": int(diag.global_violation),
        "num_steps": result.step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--num-formulas", type=int, default=50,
                        help="Number of different SAT formulas to test")
    parser.add_argument("--trials-per-formula", type=int, default=20,
                        help="Corruption trials per formula per condition")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--wrong-ratios", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = MLPDenoiser(INPUT_DIM, OUTPUT_DIM, hidden_dims=[256, 256])
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    denoiser = LearnedDenoiser(model, N, D)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model: MLP {INPUT_DIM} -> [256,256] -> {OUTPUT_DIM}")
    print(f"Device: {device}")
    print()

    # Sweep across formulas
    formula_seeds = list(range(args.num_formulas))
    wrong_ratios = args.wrong_ratios

    print(f"Formulas: {len(formula_seeds)}, "
          f"trials per formula: {args.trials_per_formula}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Wrong ratios: {wrong_ratios}")
    print()

    results = []

    t_start = time.monotonic()
    for fi, formula_seed in enumerate(formula_seeds):
        domain = BoolSatDomain(
            n_vars=N_VARS, n_clauses=N_CLAUSES, k=K,
            formula_seed=formula_seed,
        )

        for wr in wrong_ratios:
            for use_repair in [False, True]:
                n_success = 0
                trial_results = []
                for trial_seed in range(args.trials_per_formula):
                    rng = np.random.default_rng(trial_seed + 10000 * formula_seed)
                    tr = run_trial(
                        domain, denoiser,
                        args.mask_ratio, wr, use_repair,
                        trial_seed, rng,
                    )
                    trial_results.append(tr)
                    if tr["success"]:
                        n_success += 1

                sr = n_success / args.trials_per_formula
                se = np.sqrt(sr * (1 - sr) / max(args.trials_per_formula - 1, 1))
                results.extend(trial_results)

                label = "repair" if use_repair else "learned"
                print(f"  formula={formula_seed:3d} {label} wr={wr:.1f}: "
                      f"success={sr:.3f}±{se:.3f}  "
                      f"({n_success}/{args.trials_per_formula})")

        elapsed = time.monotonic() - t_start
        eta = (elapsed / (fi + 1)) * (len(formula_seeds) - fi - 1)
        print(f"  [{fi+1}/{len(formula_seeds)}] elapsed={elapsed:.0f}s "
              f"eta={eta:.0f}s")
        print()

    # Aggregate per (wr, use_repair)
    print("=" * 60)
    print("AGGREGATE RESULTS (across all formulas)")
    print("=" * 60)
    for wr in wrong_ratios:
        for use_repair in [False, True]:
            relevant = [r for r in results
                        if r["wrong_ratio"] == wr
                        and r["use_repair"] == use_repair]
            successes = [r["success"] for r in relevant]
            n = len(successes)
            agg_sr = np.mean(successes)
            agg_se = np.sqrt(agg_sr * (1 - agg_sr) / max(n - 1, 1))
            label = "repair" if use_repair else "learned"
            print(f"  {label} wr={wr:.1f}: "
                  f"success={agg_sr:.4f}±{agg_se:.4f}  "
                  f"(n={n}, formulas={args.num_formulas}, "
                  f"trials/formula={args.trials_per_formula})")

    elapsed_total = time.monotonic() - t_start
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    main()

"""Save denoiser-quality sweep results to results/logs with plots."""
import json, os, numpy as np
from pathlib import Path

# Load the existing sweep results from /tmp
try:
    with open("/tmp/noisy_sweep.json") as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Results weren't saved — re-run
    print("Running denoiser-quality sweep...")
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
    
    data = {}
    for sigma in sigma_vals:
        noisy = NoisyDenoiser(backend, sigma=sigma)
        for use_repair in [False, True]:
            label = f"noisy_s{sigma}" + ("_repair" if use_repair else "")
            policy = VerifierRepairPolicy(remask_threshold=1) if use_repair else ConfidenceUnmaskPolicy(threshold=0.99)
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
                key = f"{label}_wr{wr:.2f}"
                data[key] = {
                    "method": label,
                    "sigma": sigma,
                    "repair": use_repair,
                    "wrong_ratio": wr,
                    "success_rate": float(np.mean(successes)),
                    "num_trials": num_trials,
                }

# Save to results/logs
logs_dir = Path(__file__).resolve().parents[3] / "results" / "logs"
os.makedirs(logs_dir, exist_ok=True)
path = logs_dir / "noisy_sweep.json"
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved: {path}")

# Generate plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sigma_vals = sorted(set(d["sigma"] for d in data.values()))
    wrong_ratios = sorted(set(d["wrong_ratio"] for d in data.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sigma_vals)))
    
    # Plot 1: With repair
    ax = axes[0]
    for sigma, color in zip(sigma_vals, colors):
        xs, ys = [], []
        for wr in wrong_ratios:
            key = f"noisy_s{sigma}_repair_wr{wr:.2f}"
            if key in data:
                xs.append(wr)
                ys.append(data[key]["success_rate"])
        ax.plot(xs, ys, marker="o", color=color, label=f"σ={sigma}", linewidth=1.5)
    ax.set_xlabel("Wrong-token ratio")
    ax.set_ylabel("Success rate")
    ax.set_title("Noisy TN + repair")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Without repair
    ax = axes[1]
    for sigma, color in zip(sigma_vals, colors):
        xs, ys = [], []
        for wr in wrong_ratios:
            key = f"noisy_s{sigma}_wr{wr:.2f}"
            if key in data:
                xs.append(wr)
                ys.append(data[key]["success_rate"])
        ax.plot(xs, ys, marker="o", color=color, label=f"σ={sigma}", linewidth=1.5)
    ax.set_xlabel("Wrong-token ratio")
    ax.set_ylabel("Success rate")
    ax.set_title("Noisy TN without repair")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plots_dir = Path(__file__).resolve().parents[3] / "results" / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = plots_dir / "noisy_sweep.pdf"
    fig.savefig(plot_path)
    print(f"Plot: {plot_path}")
    plt.close(fig)
except ImportError:
    print("matplotlib not available, skipping plot")

print("Done.")

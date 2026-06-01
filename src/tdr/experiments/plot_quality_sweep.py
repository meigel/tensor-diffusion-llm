"""\
Generate quality-sweep figure from existing experiment results.
Maps denoiser types to measured one-step accuracies and plots
repair-lift vs denoiser quality.
"""

import json, os, sys
from pathlib import Path

import numpy as np

# Paths
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Measured denoiser accuracies (50% mask ratio, no wrong tokens)
DENOISER_ACCURACY = {
    "sudoku": {
        "random": 0.249,
        "local": 0.734,
        "tn": 0.917,
    },
    "sat": {
        "random": 0.297,
        "tn": 0.828,
    },
}

# Denoiser name mapping in experiment logs → internal names
NAME_MAP = {
    "sudoku": {
        "repair_random": "random",
        "random": "random",
        "repair_local": "local",
        "local": "local",
        "tn_repair": "tn",
        "tn": "tn",
        "poe_repair": None,  # skip
        "poe": None,
        "repair": None,  # skip — not a denoiser
    },
    "sat": {
        "repair_random": "random",
        "random": "random",
        "tn_repair": "tn",
        "tn": "tn",
    },
}

# Method categorisation
def method_is_repair(method_name):
    return method_name.endswith("_repair")

def method_has_verifier_repair(method_name):
    return method_name.startswith("repair_") or method_name.endswith("_repair")

def get_accuracy(domain, method_name):
    m = method_name
    if m == "repair_random": m = "random"
    elif m == "repair_local": m = "local"
    elif m == "tn_repair": m = "tn"
    elif m == "poe_repair": m = "poe"
    return DENOISER_ACCURACY[domain].get(m, None)


def load_results(path):
    """Load experiment results from JSON log."""
    with open(path) as f:
        return json.load(f)


def extract_quality_data(results, domain):
    """Extract (accuracy, wrong_ratio, success_rate, is_repair) tuples."""
    points = []
    denoiser_accuracy = DENOISER_ACCURACY[domain]
    name_map = NAME_MAP[domain]

    for key, entry in results.items():
        if key.startswith("_"):
            continue
        method = entry.get("method", "")
        wr = entry.get("wrong_ratio")
        sr = entry.get("success_rate")

        if method is None or wr is None or sr is None:
            continue

        # Map to base denoiser name
        base = name_map.get(method)
        if base is None or base not in denoiser_accuracy:
            continue

        acc = denoiser_accuracy[base]
        is_repair = method_has_verifier_repair(method)

        points.append({
            "method": method,
            "base_denoiser": base,
            "accuracy": acc,
            "wrong_ratio": wr,
            "success_rate": sr,
            "success_se": entry.get("success_se", 0),
            "is_repair": is_repair,
        })

    return points


def plot_quality_sweep(sudoku_points, sat_points):
    """Generate 2-panel figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    domain_data = {}
    if sudoku_points:
        domain_data["Sudoku 4×4"] = sudoku_points
    if sat_points:
        domain_data["3-SAT (n=20)"] = sat_points

    markers = {"Sudoku 4×4": "o", "3-SAT (n=20)": "s"}

    # ---- Panel A: Repair lift vs denoiser accuracy ----
    ax = axes[0]
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    for domain_label, points in domain_data.items():
        for wr in sorted(set(p["wrong_ratio"] for p in points)):
            # Get accuracy → repair lift pairs
            acc_to_lift = {}
            for p in points:
                if p["wrong_ratio"] != wr:
                    continue
                base = p["base_denoiser"]
                acc = p["accuracy"]
                is_repair = p["is_repair"]

                key = (acc, base)
                if key not in acc_to_lift:
                    acc_to_lift[key] = {"no_repair": None, "repair": None}
                if is_repair:
                    acc_to_lift[key]["repair"] = p["success_rate"]
                else:
                    acc_to_lift[key]["no_repair"] = p["success_rate"]

            x_vals, y_vals = [], []
            for (acc, base), vals in sorted(acc_to_lift.items()):
                if vals["no_repair"] is not None and vals["repair"] is not None:
                    x_vals.append(acc)
                    y_vals.append(vals["repair"] - vals["no_repair"])

            if x_vals:
                ax.plot(x_vals, y_vals, marker=markers[domain_label],
                        label=f"{domain_label} wr={wr:.1f}", linewidth=2, markersize=8)

    ax.set_xlabel("One-step denoiser accuracy")
    ax.set_ylabel("Repair lift (Δ success rate)")
    ax.set_title("Verifier repair benefit vs denoiser quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 1.0)

    # ---- Panel B: Success rate vs denoiser accuracy (wr=0.2) ----
    ax = axes[1]
    wr_target = 0.2

    for domain_label, points in domain_data.items():
        for is_repair, ls, lbl in [
            (False, "-", "No repair"),
            (True, "--", "Verifier repair"),
        ]:
            x_vals, y_vals, y_errs = [], [], []
            for p in points:
                if abs(p["wrong_ratio"] - wr_target) > 0.01:
                    continue
                if p["is_repair"] != is_repair:
                    continue
                # Deduplicate by accuracy
                acc = p["accuracy"]
                if acc not in x_vals:
                    x_vals.append(acc)
                    y_vals.append(p["success_rate"])
                    y_errs.append(p.get("success_se", 0))

            if x_vals:
                # Sort by accuracy
                order = np.argsort(x_vals)
                x_vals = [x_vals[i] for i in order]
                y_vals = [y_vals[i] for i in order]
                y_errs = [y_errs[i] for i in order]
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                           marker=markers[domain_label],
                           linestyle=ls, capsize=4, linewidth=2, markersize=8,
                           label=f"{domain_label} {lbl}")

    ax.set_xlabel("One-step denoiser accuracy")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Success rate at wrong-token ratio = {wr_target}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 1.0)

    fig.tight_layout()
    path = PLOTS_DIR / "quality_sweep.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
    png_path = PLOTS_DIR / "quality_sweep.png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    plt.close(fig)


def main():
    # Load the most recent definitive experiment results
    log_dir = LOGS_DIR
    sudoku_path = None
    sat_path = None

    # Look for the definitive experiment JSON
    for f in sorted(os.listdir(log_dir)):
        if f.startswith("repair_definitive_all") and f.endswith(".json"):
            # This file contains both domains
            path = log_dir / f
            print(f"Loading: {path}")
            results = load_results(path)
            sudoku_pts = extract_quality_data(results.get("sudoku4", {}), "sudoku")
            sat_pts = extract_quality_data(results.get("sat", {}), "sat")
            print(f"  Sudoku: {len(sudoku_pts)} data points")
            print(f"  SAT: {len(sat_pts)} data points")

            # Also load quality sweep results for calibration data
            qs_path = LOGS_DIR / "quality_sweep_sudoku_20260601_225755.json"
            if qs_path.exists():
                print(f"\nCalibration from quality sweep run:")
                qs = load_results(qs_path)
                for c in qs.get("_calibration", []):
                    print(f"  {c}")

            plot_quality_sweep(sudoku_pts, sat_pts)
            return

    print("No definitive experiment results found.")
    print(f"Found files: {[f for f in os.listdir(log_dir) if f.endswith('.json')]}")


if __name__ == "__main__":
    main()

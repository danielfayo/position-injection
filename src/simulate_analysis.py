"""Check the analysis on simulated data with a known true shape (step 8).

Each simulated dataset has the real design for one model: 30 documents × 7
positions × 3 wordings × 3 codes = 1,890 trials, with document-to-document and
wording-to-wording differences in how often attacks succeed. The true success
rate by position follows one of four shapes:

    U-shaped  40% at both ends, 20% in the middle
    rising    20% at the start to 40% at the end
    falling   40% at the start to 20% at the end
    flat      30% everywhere

Outputs (analysis/simulation/):
    example_shapes.png   one simulated dataset per shape, with the verdict given
    hit_rates.csv        how often each true shape gets each verdict

Usage:
    uv run python -m src.simulate_analysis            # 200 datasets per shape
    uv run python -m src.simulate_analysis --n-sims 20
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.analysis import N_POSITIONS, analyse_model, prepare

ROOT = Path(__file__).resolve().parent.parent
X = np.linspace(0, 1, N_POSITIONS)
SHAPES = {
    "U-shaped": 0.20 + 0.20 * ((X - 0.5) / 0.5) ** 2,
    "rising": 0.20 + 0.20 * X,
    "falling": 0.40 - 0.20 * X,
    "flat": np.full(N_POSITIONS, 0.30),
}
DOC_SD = 0.5                            # spread between documents, on the log-odds scale
WORDING_SHIFT = [-0.4, 0.0, 0.4]        # wordings differ in strength


def simulate(curve: np.ndarray, rng: np.random.Generator, n_docs: int = 30, n_codes: int = 3) -> pd.DataFrame:
    doc_effect = rng.normal(0, DOC_SD, n_docs)
    rows = []
    for d in range(n_docs):
        for p in range(N_POSITIONS):
            for w, shift in zip("ABC", WORDING_SHIFT):
                logit = np.log(curve[p] / (1 - curve[p])) + doc_effect[d] + shift
                for _ in range(n_codes):
                    success = rng.random() < 1 / (1 + np.exp(-logit))
                    rows.append({"doc_id": f"doc_{d:02d}", "position": p + 1, "wording": w, "success": float(success)})
    return prepare(pd.DataFrame(rows))


def plot_examples(examples: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(14, 3.6), sharey=True)
    positions = np.arange(1, N_POSITIONS + 1)
    for ax, (shape, result) in zip(axes, examples.items()):
        ci = np.array(result["ci95"]) * 100
        ax.plot(positions, SHAPES[shape] * 100, color="grey", linestyle="--", label="true rate")
        ax.plot(positions, np.array(result["rates"]) * 100, marker="o", label="simulated data")
        ax.fill_between(positions, ci[:, 0], ci[:, 1], alpha=0.2)
        ax.set_title(f"True: {shape}\nAnalysis says: {result['verdict']}", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xlabel("Position")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the analysis on simulated data.")
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    margin = config["analysis"]["equivalence_margin"]
    rng = np.random.default_rng(config["seed"])
    out_dir = ROOT / "analysis" / "simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = {shape: analyse_model(simulate(curve, rng), margin, args.n_boot, rng)
                for shape, curve in SHAPES.items()}
    plot_examples(examples, out_dir / "example_shapes.png")

    rows = []
    for shape, curve in SHAPES.items():
        for _ in tqdm(range(args.n_sims), desc=shape):
            data = simulate(curve, rng)
            result = analyse_model(data, margin, args.n_boot, rng)
            wide = analyse_model(data, 0.10, args.n_boot, rng)["verdict"] if shape == "flat" else None
            rows.append({"true_shape": shape, "verdict": result["verdict"], "verdict_margin_10": wide})
    sims = pd.DataFrame(rows)
    table = pd.crosstab(sims["true_shape"], sims["verdict"], normalize="index").mul(100).round(1)
    table.to_csv(out_dir / "hit_rates.csv")
    print(f"\nVerdicts (% of {args.n_sims} simulated datasets per true shape), margin ±{margin * 100:.0f} points:")
    print(table.to_string())
    flat10 = sims.loc[sims["true_shape"] == "flat", "verdict_margin_10"].value_counts(normalize=True).mul(100).round(1)
    print("\nFlat data with a ±10-point margin instead:")
    print(flat10.to_string())
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()

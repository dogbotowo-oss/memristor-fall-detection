"""Re-plot the sigma noise-robustness figure from the saved CSV (annotation fix)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAPSHOT = Path(__file__).resolve().parent.parent
IN_CSV = SNAPSHOT / "reports" / "noise_robustness_sigma_c1_s34_data1_s11.csv"
OUT_PNG = SNAPSHOT / "image" / "noise_robustness_sigma_c1_s34_data1_s11.png"


def main() -> None:
    with IN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = [float(r["sigma_pct"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for key, label, color, marker in [
        ("accuracy", "Accuracy", "#1f77b4", "o"),
        ("balanced_accuracy", "Balanced acc", "#2ca02c", "s"),
        ("adl_specificity", "ADL specificity", "#ff7f0e", "^"),
        ("fall_recall", "Fall recall", "#d62728", "v"),
    ]:
        ax.plot(x, [float(r[key]) for r in rows], marker=marker, color=color,
                linewidth=1.8, markersize=5, label=label)

    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.annotate("deployed sigma = 0.1%", xy=(0.1, 100), xytext=(0.8, 88),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    ax.axvline(3.5, color="purple", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.annotate("training noise 3.5%", xy=(3.5, 100), xytext=(5.2, 96),
                fontsize=8, color="purple",
                arrowprops=dict(arrowstyle="-", color="purple", lw=0.8))

    ax.set_xlabel("Readout noise sigma (% full scale)")
    ax.set_ylabel("Subject4 metric (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()

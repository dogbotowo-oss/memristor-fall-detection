"""4x4 confusion matrix: Ideal vs Non-Ideal (Full), matched protocol.

Both models: data1, seed11, thresholds frozen from each model's own val scan
(Ideal thr=0.45, Full thr=0.65; Ideal predictions at 0.65 are identical, so a
common-threshold presentation at 0.65 is also valid).

Ideal (c1_s34_data1_ideal_s11): TP12 TN17 FP3 FN5
Full  (v999 c1_s34_data1_s11):  TP15 TN15 FP5 FN2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SNAPSHOT = Path(__file__).resolve().parent.parent
OUT_PNG = SNAPSHOT / "image" / "confusion_ideal_vs_full_s4.png"

LABELS = ["I-ADL", "I-Fall", "NI-ADL", "NI-Fall"]
MAT = np.array([
    [17, 3, 0, 0],   # Ideal actual ADL: TN=17, FP=3
    [5, 12, 0, 0],   # Ideal actual Fall: FN=5, TP=12
    [0, 0, 15, 5],   # Full actual ADL: TN=15, FP=5
    [0, 0, 2, 15],   # Full actual Fall: FN=2, TP=15
])


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=160)
    im = ax.imshow(MAT, cmap="Reds", vmin=0, vmax=20)
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Predict Class")
    ax.set_ylabel("Actual Class")
    for i in range(4):
        for j in range(4):
            v = MAT[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="white" if v >= 12 else "black")
    # block separator between Ideal and Non-Ideal
    for pos in (1.5,):
        ax.axhline(pos - 0.0, color="gray", lw=1.2)
        ax.axvline(pos - 0.0, color="gray", lw=1.2)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()

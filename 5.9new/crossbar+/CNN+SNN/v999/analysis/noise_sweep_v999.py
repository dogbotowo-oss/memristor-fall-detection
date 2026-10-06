"""Inference-only readout-noise robustness sweep for the v999 model.

Protocol: threshold frozen at the val-selected 0.65 (no re-selection — this
simulates post-deployment device noise increase). For each noise level,
Subject4 is evaluated once (deterministic per-window noise seeds).

Noise levels are given in percent of relative readout noise:
0/15/30/45/60/75/90% -> crossbar_readout_noise_scale 0.0..0.90.
The deployed/training level is 0.25 (25%), so the sweep brackets 0-3.6x.

Outputs:
  reports/noise_robustness_c1_s34_data1_s11.csv
  image/noise_robustness_c1_s34_data1_s11.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strict_eval_s30 as sev

SNAPSHOT = HERE.parent
MODEL_PATH = SNAPSHOT / "models" / "automatic_fall_event_detector_c1_s34_data1_s11.pth"
THRESHOLD = 0.65
# sigma_eff sweep: read_sigma = 0.004 * scale * dyn (dyn ~= 1).
# Points: sigma = 0..30% FS in 5% steps -> scale = sigma/0.004.
SIGMA_PCTS = [0, 5, 10, 15, 20, 25, 30]
SCALES = [s / 100.0 / 0.004 for s in SIGMA_PCTS]
OUT_CSV = SNAPSHOT / "reports" / "noise_robustness_sigma_c1_s34_data1_s11.csv"
OUT_PNG = SNAPSHOT / "image" / "noise_robustness_sigma_c1_s34_data1_s11.png"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(MODEL_PATH, device)
    videos = sorted(mod.list_video_files(sev.SUBJECT4_DIR))
    print(f"{len(videos)} Subject4 videos, threshold frozen at {THRESHOLD}")

    rows = []
    for sigma_pct, scale in zip(SIGMA_PCTS, SCALES):
        cfg = dict(config)
        cfg["crossbar_readout_noise_scale"] = scale
        _, summary = sev.evaluate_at_threshold(mod, model, cfg, videos, THRESHOLD, device)
        row = {
            "sigma_pct": sigma_pct,
            "noise_scale": scale,
            "accuracy": float(summary["accuracy"]) * 100,
            "balanced_accuracy": float(summary["balanced_accuracy"]) * 100,
            "adl_specificity": float(summary["adl_specificity"]) * 100,
            "fall_recall": float(summary["fall_recall"]) * 100,
            "precision": float(summary["precision"]) * 100,
            "tp": summary["tp"], "tn": summary["tn"],
            "fp": summary["fp"], "fn": summary["fn"],
        }
        rows.append(row)
        print(f"sigma {sigma_pct:3d}% (scale {scale:5.1f}) | acc {row['accuracy']:5.2f} "
              f"bal {row['balanced_accuracy']:5.2f} spec {row['adl_specificity']:5.1f} "
              f"rec {row['fall_recall']:5.2f}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", OUT_CSV)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    x = [r["sigma_pct"] for r in rows]
    for key, label, color, marker in [
        ("accuracy", "Accuracy", "#1f77b4", "o"),
        ("balanced_accuracy", "Balanced acc", "#2ca02c", "s"),
        ("adl_specificity", "ADL specificity", "#ff7f0e", "^"),
        ("fall_recall", "Fall recall", "#d62728", "v"),
    ]:
        ax.plot(x, [r[key] for r in rows], marker=marker, color=color, linewidth=1.8,
                markersize=5, label=label)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.6, 5, "deployed sigma=0.1% (scale 0.25)", fontsize=8, color="gray")
    ax.axvline(3.5, color="purple", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(3.8, 20, "training noise 3.5%", fontsize=8, color="purple")
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

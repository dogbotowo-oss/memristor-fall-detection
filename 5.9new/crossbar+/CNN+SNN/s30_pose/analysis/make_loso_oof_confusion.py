# Build LOSO out-of-fold pooled confusion matrices (I/NI block format) + PNGs.
# Pools per-video predictions from the 4 LOSO folds (rescan outputs) for
# Full (non-ideal) and Ideal models, under both threshold rules:
#   gtest  = bal rule (bal-acc optimal)
#   gtestR = rec rule (recall-prioritized)
# Outputs land in C:\Users\czs\Desktop\rr.
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\s30_pose\reports")
OUT = Path(r"C:\Users\czs\Desktop\rr")
NOISE_SRC = Path(
    r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v999\reports"
    r"\noise_robustness_sigma_c1_s34_data1_s11.csv"
)

RULES = {"rec": "gtestR", "bal": "gtest"}
VARIANTS = {
    "I": [f"c1_s34_data1_loso_ts{k}_ideal_s11" for k in range(1, 5)],
    "NI": [f"c1_s34_data1_loso_ts{k}_s11" for k in range(1, 5)],
}


def pool_confusion(tags, suffix):
    # Return (tn, fp, fn, tp, n) pooled over folds; label 0=ADL, 1=Fall.
    tn = fp = fn = tp = 0
    n = 0
    for tag in tags:
        path = REPORTS / f"{tag}_{suffix}_video_metrics.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t, p = int(row["true_label"]), int(row["pred_label"])
                n += 1
                if t == 0 and p == 0:
                    tn += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 1 and p == 0:
                    fn += 1
                else:
                    tp += 1
    return tn, fp, fn, tp, n


def block_csv(rule, conf_i, conf_ni):
    i_tn, i_fp, i_fn, i_tp, _ = conf_i
    n_tn, n_fp, n_fn, n_tp, _ = conf_ni
    rows = [
        ["Actual-Class", "I-ADL", "I-Fall", "NI-ADL", "NI-Fall"],
        ["I-ADL", i_tn, i_fp, 0, 0],
        ["I-Fall", i_fn, i_tp, 0, 0],
        ["NI-ADL", 0, 0, n_tn, n_fp],
        ["NI-Fall", 0, 0, n_fn, n_tp],
    ]
    path = OUT / f"LOSO_OOF_confusion_Ideal_vs_NonIdeal_{rule}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def metrics_line(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = 100.0 * (tn + tp) / total
    spec = 100.0 * tn / (tn + fp)
    rec = 100.0 * tp / (tp + fn)
    return acc, spec, rec


def plot_matrix(tn, fp, fn, tp, title, path):
    cm = np.array([[tn, fp], [fn, tp]])
    acc, spec, rec = metrics_line(tn, fp, fn, tp)
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1], labels=["ADL", "Fall"])
    ax.set_yticks([0, 1], labels=["ADL", "Fall"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\nacc={acc:.2f}  spec={spec:.2f}  recall={rec:.2f}",
                 fontsize=10)
    for r in range(2):
        for c in range(2):
            color = "white" if cm[r, c] > cm.max() * 0.6 else "black"
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    fontsize=16, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for rule, suffix in RULES.items():
        confs = {}
        for var, tags in VARIANTS.items():
            confs[var] = pool_confusion(tags, suffix)
        csv_path = block_csv(rule, confs["I"], confs["NI"])
        for var, name in (("I", "Ideal"), ("NI", "NonIdeal")):
            tn, fp, fn, tp, n = confs[var]
            acc, spec, rec = metrics_line(tn, fp, fn, tp)
            png = OUT / f"LOSO_OOF_confusion_{name}_{rule}.png"
            plot_matrix(tn, fp, fn, tp,
                        f"LOSO OOF ({name}, {rule} rule, N={n})", png)
            summary.append(
                f"[{rule}] {name}: N={n} tn={tn} fp={fp} fn={fn} tp={tp} "
                f"acc={acc:.2f} spec={spec:.2f} rec={rec:.2f}"
            )
        print(f"wrote {csv_path.name}")
    shutil.copy2(NOISE_SRC, OUT / NOISE_SRC.name)
    print(f"copied {NOISE_SRC.name}")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

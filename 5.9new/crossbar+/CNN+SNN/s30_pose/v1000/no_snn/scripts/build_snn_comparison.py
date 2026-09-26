"""Build fold-level and pooled v1000 SNN versus no-SNN comparisons."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\rray\12.18-2")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
SNN_REPORTS = S30 / "reports"
NO_SNN_ROOT = S30 / "v1000" / "no_snn"
NO_SNN_REPORTS = NO_SNN_ROOT / "reports"
OUT = NO_SNN_ROOT / "comparison"
SEED = 11
SCHEMES = {"zenodo_only": "test", "balanced": "gtest", "recall_priority": "gtestR"}
METRICS = ("accuracy", "balanced_accuracy", "adl_specificity", "fall_recall")


def snn_tag(subject: int) -> str:
    return f"c1_s34_data1_loso_ts{subject}_s{SEED}"


def no_snn_tag(subject: int) -> str:
    return f"c1_s34_data1_v1000_loso_ts{subject}_nosnn_s{SEED}"


def load_metric(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    scheme: str, subject: int, configuration: str, data: dict[str, object]
) -> dict[str, object]:
    return {
        "scheme": scheme,
        "test_subject": subject,
        "configuration": configuration,
        "threshold": data["threshold"],
        "n": sum(int(data[key]) for key in ("tn", "fp", "fn", "tp")),
        **{metric: 100.0 * float(data[metric]) for metric in METRICS},
        **{key: int(data[key]) for key in ("tn", "fp", "fn", "tp")},
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> list[dict[str, object]]:
    rows = []
    for scheme, suffix in SCHEMES.items():
        for subject in (1, 2, 3, 4):
            inputs = {
                "SNN": SNN_REPORTS / f"{snn_tag(subject)}_{suffix}_metrics.json",
                "no-SNN": NO_SNN_REPORTS / f"{no_snn_tag(subject)}_{suffix}_metrics.json",
            }
            for configuration, path in inputs.items():
                rows.append(metric_row(scheme, subject, configuration, load_metric(path)))
    return rows


def build_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for scheme in SCHEMES:
        for configuration in ("SNN", "no-SNN"):
            group = [
                row
                for row in rows
                if row["scheme"] == scheme and row["configuration"] == configuration
            ]
            totals = {key: sum(int(row[key]) for row in group) for key in ("tn", "fp", "fn", "tp")}
            total_n = sum(totals.values())
            pooled_accuracy = 100.0 * (totals["tn"] + totals["tp"]) / total_n
            pooled_specificity = 100.0 * totals["tn"] / (totals["tn"] + totals["fp"])
            pooled_recall = 100.0 * totals["tp"] / (totals["tp"] + totals["fn"])
            summary = {
                "scheme": scheme,
                "configuration": configuration,
                "folds": len(group),
                "pooled_n": total_n,
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                summary[f"macro_{metric}_mean"] = statistics.mean(values)
                summary[f"macro_{metric}_sd"] = statistics.stdev(values)
            summary.update(
                {
                    "pooled_accuracy": pooled_accuracy,
                    "pooled_adl_specificity": pooled_specificity,
                    "pooled_fall_recall": pooled_recall,
                    "pooled_balanced_accuracy": (pooled_specificity + pooled_recall) / 2.0,
                    **totals,
                }
            )
            summaries.append(summary)
    return summaries


def build_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deltas = []
    for scheme in SCHEMES:
        for subject in (1, 2, 3, 4):
            pair = {
                row["configuration"]: row
                for row in rows
                if row["scheme"] == scheme and row["test_subject"] == subject
            }
            deltas.append(
                {
                    "scheme": scheme,
                    "test_subject": subject,
                    **{
                        f"delta_{metric}_snn_minus_no_snn": float(pair["SNN"][metric])
                        - float(pair["no-SNN"][metric])
                        for metric in METRICS
                    },
                }
            )
    return deltas


def plot_paired(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), dpi=220, sharey=True)
    colors = {"no-SNN": "#9aa0a6", "SNN": "#1f77b4"}
    label_offsets = {1: 8, 2: 3, 3: -4, 4: -10}
    for axis, scheme in zip(axes, SCHEMES, strict=True):
        configuration_values = {"no-SNN": [], "SNN": []}
        for subject in (1, 2, 3, 4):
            pair = {
                row["configuration"]: row
                for row in rows
                if row["scheme"] == scheme and row["test_subject"] == subject
            }
            values = [pair["no-SNN"]["balanced_accuracy"], pair["SNN"]["balanced_accuracy"]]
            configuration_values["no-SNN"].append(float(values[0]))
            configuration_values["SNN"].append(float(values[1]))
            axis.plot([0, 1], values, color="#c9cdd2", linewidth=1.2, zorder=1)
            axis.scatter([0, 1], values, color=[colors["no-SNN"], colors["SNN"]], s=35, zorder=2)
            axis.annotate(
                f"S{subject}",
                (1, values[1]),
                xytext=(7, label_offsets[subject]),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )
        means = [statistics.mean(configuration_values[name]) for name in ("no-SNN", "SNN")]
        deviations = [statistics.stdev(configuration_values[name]) for name in ("no-SNN", "SNN")]
        axis.errorbar(
            [-0.08, 0.92],
            means,
            yerr=deviations,
            fmt="D",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            capsize=3,
            linewidth=1.1,
            zorder=3,
        )
        axis.set_xticks([0, 1], ["no-SNN", "SNN"])
        axis.set_xlim(-0.25, 1.28)
        axis.set_title(scheme.replace("_", " ").title(), fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy (%)")
    fig.suptitle("v1000 four-fold LOSO: matched SNN ablation", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "v1000_loso_snn_vs_no_snn_paired.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(rows: list[dict[str, object]]) -> None:
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "adl_specificity": "ADL specificity",
        "fall_recall": "Fall recall",
    }
    primary = [row for row in rows if row["scheme"] == "balanced"]
    configurations = ("no-SNN", "SNN")
    colors = {"no-SNN": "#9aa0a6", "SNN": "#1f77b4"}
    centers = np.arange(len(METRICS), dtype=float)
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.2, 4.5), dpi=220)
    for index, configuration in enumerate(configurations):
        configuration_rows = [
            row for row in primary if row["configuration"] == configuration
        ]
        means = [
            statistics.mean(float(row[metric]) for row in configuration_rows)
            for metric in METRICS
        ]
        deviations = [
            statistics.stdev(float(row[metric]) for row in configuration_rows)
            for metric in METRICS
        ]
        positions = centers + (index - 0.5) * width
        axis.bar(
            positions,
            means,
            width=width,
            yerr=deviations,
            capsize=3,
            color=colors[configuration],
            alpha=0.82,
            edgecolor="white",
            label=f"{configuration} mean±SD",
            zorder=2,
        )
        for subject_index, row in enumerate(configuration_rows):
            jitter = (subject_index - 1.5) * 0.025
            axis.scatter(
                positions + jitter,
                [float(row[metric]) for metric in METRICS],
                s=18,
                color="black",
                alpha=0.72,
                zorder=3,
            )
    axis.set_xticks(centers, [metric_labels[metric] for metric in METRICS])
    axis.set_ylabel("Performance (%)")
    axis.set_ylim(0, 110)
    axis.grid(axis="y", alpha=0.25, zorder=1)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    axis.set_title("v1000 four-fold LOSO: SNN vs no-SNN (balanced rule)")
    fig.tight_layout()
    fig.savefig(OUT / "v1000_loso_snn_vs_no_snn_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusions(summary_rows: list[dict[str, object]]) -> None:
    primary = {
        row["configuration"]: row
        for row in summary_rows
        if row["scheme"] == "balanced"
    }
    matrices = {
        configuration: np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        for configuration, row in primary.items()
    }
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=220)
    image = None
    for axis, configuration in zip(axes, ("no-SNN", "SNN"), strict=True):
        matrix = matrices[configuration]
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks([0, 1], ["ADL", "Fall"])
        axis.set_yticks([0, 1], ["ADL", "Fall"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"{configuration} (pooled OOF, N={matrix.sum()})", fontsize=9)
        for row_index in range(2):
            for column_index in range(2):
                value = int(matrix[row_index, column_index])
                color = "white" if value > maximum * 0.55 else "black"
                axis.text(column_index, row_index, str(value), ha="center", va="center", color=color, fontsize=14)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle("v1000 LOSO SNN ablation (balanced threshold rule)", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.16, top=0.82, wspace=0.35)
    fig.savefig(OUT / "v1000_loso_snn_vs_no_snn_oof_confusion.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    summaries = build_summaries(rows)
    deltas = build_deltas(rows)
    write_csv(OUT / "v1000_loso_snn_vs_no_snn_fold_metrics.csv", rows)
    write_csv(OUT / "v1000_loso_snn_vs_no_snn_summary.csv", summaries)
    write_csv(OUT / "v1000_loso_snn_vs_no_snn_paired_deltas.csv", deltas)
    plot_paired(rows)
    plot_metric_summary(rows)
    plot_confusions(summaries)
    print(f"comparison outputs written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

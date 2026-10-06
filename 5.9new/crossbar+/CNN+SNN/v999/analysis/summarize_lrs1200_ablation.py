"""Create per-seed and aggregate reports for the valid-LRS ablation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


SEEDS = (1, 3, 5, 11, 42)
METHODS = {
    "SNN": "c1_s34_lrs1200_s{seed}",
    "no-SNN": "c1_s34_lrs1200_nosnn_s{seed}",
}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "adl_specificity",
    "fall_recall",
    "precision",
)


def load_metrics(reports_dir: Path, tag: str) -> tuple[dict[str, object], dict[str, object]]:
    val_path = reports_dir / f"{tag}_val_summary.json"
    subject4_path = reports_dir / f"{tag}_subject4_metrics.json"
    with val_path.open(encoding="utf-8") as file_handle:
        val_metrics = json.load(file_handle)
    with subject4_path.open(encoding="utf-8") as file_handle:
        subject4_metrics = json.load(file_handle)
    return val_metrics, subject4_metrics


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    rows: list[dict[str, object]] = []
    for method_name, tag_template in METHODS.items():
        for seed in SEEDS:
            tag = tag_template.format(seed=seed)
            val_metrics, subject4_metrics = load_metrics(reports_dir, tag)
            row: dict[str, object] = {
                "method": method_name,
                "seed": seed,
                "tag": tag,
                "val_threshold": val_metrics["best_threshold"],
            }
            for metric_name in METRICS:
                row[f"val_{metric_name}"] = val_metrics[metric_name]
                row[f"subject4_{metric_name}"] = subject4_metrics[metric_name]
            rows.append(row)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for method_name in METHODS:
        method_rows = [row for row in rows if row["method"] == method_name]
        aggregate[method_name] = {}
        for split_name in ("val", "subject4"):
            for metric_name in METRICS:
                column_name = f"{split_name}_{metric_name}"
                aggregate[method_name][column_name] = mean_and_std(
                    [float(row[column_name]) for row in method_rows]
                )

    csv_path = reports_dir / "lrs1200_ablation_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = reports_dir / "lrs1200_ablation_summary.json"
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "lrs_retention_max_time_seconds": 1200,
                "lrs_selection": "Measurement time strictly earlier than 1200 s",
                "seeds": list(SEEDS),
                "per_seed_csv": str(csv_path),
                "aggregate": aggregate,
            },
            file_handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()

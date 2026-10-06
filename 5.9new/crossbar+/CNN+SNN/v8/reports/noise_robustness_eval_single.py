
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"F:\12.18-2")
V8_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8"
BASE_SCRIPT = V8_DIR / "reports" / "noise_robustness_eval.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("noise_robustness_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v8 final noise robustness condition at a time.")
    parser.add_argument("--noise-type", choices=["none", "gaosi", "possion", "salt"], required=True)
    args = parser.parse_args()

    base = load_base_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = base.load_module()
    model, config = mod.load_model_checkpoint(base.MODEL_PATH, device)
    model.eval()

    use_crossbar = bool(config.get("use_crossbar", True))
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", 0.0))
    if use_crossbar:
        hrs_values, lrs_values = mod.load_conductance_pair(base.DATA_DIR)
    else:
        hrs_values = np.array([0.1], dtype=np.float32)
        lrs_values = np.array([0.9], dtype=np.float32)
    device_dynamics = (
        mod.load_device_dynamics(base.DATA_DIR)
        if use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
        else None
    )
    resources = {
        "hrs_values": hrs_values,
        "lrs_values": lrs_values,
        "device_dynamics": device_dynamics,
        "crossbar_readout_noise_scale": crossbar_readout_noise_scale,
    }

    videos = mod.list_video_files(base.SUBJECT4_DIR)
    if not videos:
        raise RuntimeError(f"No Subject4 videos found: {base.SUBJECT4_DIR}")

    noise_type = args.noise_type
    out_dir = base.OUTPUT_DIRS[noise_type]
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [(0, 0.0)] if noise_type == "none" else list(enumerate(base.LEVELS))
    summary_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []

    summary_fields = [
        "noise_type",
        "noise_percent",
        "noise_level",
        "level_index",
        "threshold",
        "gaussian_sigma",
        "salt_probability",
        "poisson_peak",
        "total",
        "accuracy",
        "accuracy_percent",
        "balanced_accuracy",
        "balanced_accuracy_percent",
        "adl_specificity",
        "adl_specificity_percent",
        "fall_recall",
        "fall_recall_percent",
        "precision",
        "precision_percent",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    video_fields = ["noise_type", "noise_percent", "level_index", "threshold", "video_path", "true_label", "pred_label"]

    for level_index, level in levels:
        rows, summary = base.evaluate_level(
            mod,
            model,
            config,
            videos,
            device,
            resources,
            noise_type=noise_type,
            level=level,
            level_index=level_index,
        )
        summary_rows.append(summary)
        video_rows.extend(rows)
        write_csv(out_dir / "shuju.csv", summary_rows, summary_fields)
        write_csv(out_dir / "video_shuju.csv", video_rows, video_fields)
        (out_dir / "shuju.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        base.write_readme(out_dir.parent / "README.md", noise_type, summary_rows)
        print(
            f"{noise_type} {level * 100:.1f}% | "
            f"acc={summary['accuracy_percent']:.2f} "
            f"bal={summary['balanced_accuracy_percent']:.2f} "
            f"adl={summary['adl_specificity_percent']:.2f} "
            f"fall={summary['fall_recall_percent']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

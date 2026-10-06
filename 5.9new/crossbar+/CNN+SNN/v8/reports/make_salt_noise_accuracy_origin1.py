from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"F:\12.18-2")
V8_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8"
BASE_EVAL_PATH = V8_DIR / "reports" / "noise_robustness_eval.py"
DEFAULT_OUTPUT_PATH = (
    V8_DIR
    / "reports"
    / "noise_robustness_subject4_shuju"
    / "origin_tables"
    / "Salt_noise_accuracy_origin1.csv"
)
DEFAULT_SALT_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Salt noise accuracy Origin table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument(
        "--levels",
        type=str,
        default=",".join(str(level) for level in DEFAULT_SALT_LEVELS),
        help="Comma-separated salt noise levels in 0-1 range.",
    )
    return parser.parse_args()


def parse_levels(text: str) -> list[float]:
    levels: list[float] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Salt level must be within [0, 1]: {value}")
        levels.append(value)
    if not levels:
        raise ValueError("At least one salt level is required.")
    return levels


def load_base_module():
    spec = importlib.util.spec_from_file_location("noise_robustness_eval_base", BASE_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base script: {BASE_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> None:
    args = parse_args()
    salt_levels = parse_levels(args.levels)
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

    output_rows: list[dict[str, object]] = []
    for level_index, level in enumerate(salt_levels):
        _, summary = base.evaluate_level(
            mod,
            model,
            config,
            videos,
            device,
            resources,
            noise_type="salt",
            level=level,
            level_index=level_index,
        )
        output_rows.append(
            {
                "NoiseLevel_percent": f"{summary['noise_percent']:.1f}",
                "Accuracy_percent": f"{summary['accuracy_percent']:.6f}",
                "BalancedAccuracy_percent": f"{summary['balanced_accuracy_percent']:.6f}",
                "ADLSpecificity_percent": f"{summary['adl_specificity_percent']:.6f}",
                "FallRecall_percent": f"{summary['fall_recall_percent']:.6f}",
                "Precision_percent": f"{summary['precision_percent']:.6f}",
                "TP": int(summary["tp"]),
                "TN": int(summary["tn"]),
                "FP": int(summary["fp"]),
                "FN": int(summary["fn"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "NoiseLevel_percent",
                "Accuracy_percent",
                "BalancedAccuracy_percent",
                "ADLSpecificity_percent",
                "FallRecall_percent",
                "Precision_percent",
                "TP",
                "TN",
                "FP",
                "FN",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPORTS_DIR = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports")
BASE_SCRIPT = REPORTS_DIR / "noise_robustness_eval.py"
LEVELS = [i / 16 for i in range(17)]

NOISE_CONFIGS = {
    "gaosi": {
        "folder": REPORTS_DIR.parent / "v8-final-gaosi" / "shuju",
        "origin_name": "Gaussian_noise_accuracy_origin17.csv",
        "display_name": "Gaussian",
    },
    "possion": {
        "folder": REPORTS_DIR.parent / "v8-final-possion" / "shuju",
        "origin_name": "Poisson_noise_accuracy_origin17.csv",
        "display_name": "Poisson",
    },
    "salt": {
        "folder": REPORTS_DIR.parent / "v8-final-salt" / "shuju",
        "origin_name": "Salt_noise_accuracy_origin17.csv",
        "display_name": "Salt",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("noise_robustness_eval_base17", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_origin_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "NoiseLevel_percent": row["noise_percent"],
                "Accuracy_percent": row["accuracy_percent"],
                "BalancedAccuracy_percent": row["balanced_accuracy_percent"],
                "ADLSpecificity_percent": row["adl_specificity_percent"],
                "FallRecall_percent": row["fall_recall_percent"],
                "Precision_percent": row["precision_percent"],
                "TP": row["tp"],
                "TN": row["tn"],
                "FP": row["fp"],
                "FN": row["fn"],
            }
        )
    return out


def build_combined_origin_rows(by_noise: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    keys = ["gaosi", "possion", "salt"]
    row_count = len(by_noise[keys[0]])
    combined: list[dict[str, object]] = []
    for idx in range(row_count):
        g = by_noise["gaosi"][idx]
        p = by_noise["possion"][idx]
        s = by_noise["salt"][idx]
        combined.append(
            {
                "NoiseLevel_percent": g["noise_percent"],
                "Gaussian_Accuracy_percent": g["accuracy_percent"],
                "Poisson_Accuracy_percent": p["accuracy_percent"],
                "Salt_Accuracy_percent": s["accuracy_percent"],
                "Gaussian_FallRecall_percent": g["fall_recall_percent"],
                "Poisson_FallRecall_percent": p["fall_recall_percent"],
                "Salt_FallRecall_percent": s["fall_recall_percent"],
            }
        )
    return combined


def main() -> None:
    base = load_base_module()
    device = base.torch.device("cuda" if base.torch.cuda.is_available() else "cpu")
    mod = base.load_module()
    model, config = mod.load_model_checkpoint(base.MODEL_PATH, device)
    model.eval()

    use_crossbar = bool(config.get("use_crossbar", True))
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", 0.0))
    if use_crossbar:
        hrs_values, lrs_values = mod.load_conductance_pair(base.DATA_DIR)
    else:
        hrs_values = base.np.array([0.1], dtype=base.np.float32)
        lrs_values = base.np.array([0.9], dtype=base.np.float32)
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

    combined_rows: list[dict[str, object]] = []
    by_noise: dict[str, list[dict[str, object]]] = {}

    for noise_type, cfg in NOISE_CONFIGS.items():
        out_dir = cfg["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, object]] = []
        video_rows: list[dict[str, object]] = []

        for level_index, level in enumerate(LEVELS):
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
            combined_rows.append(summary)
            base.write_csv(out_dir / "shuju17.csv", summary_rows, summary_fields)
            base.write_csv(out_dir / "video_shuju17.csv", video_rows, video_fields)
            (out_dir / "shuju17.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"{noise_type} {level * 100:.2f}% | "
                f"acc={summary['accuracy_percent']:.2f} "
                f"bal={summary['balanced_accuracy_percent']:.2f} "
                f"adl={summary['adl_specificity_percent']:.2f} "
                f"fall={summary['fall_recall_percent']:.2f}"
            )

        by_noise[noise_type] = summary_rows

    combined_dir = REPORTS_DIR / "noise_robustness_subject4_shuju"
    combined_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(combined_dir / "all_noise_shuju17.csv", combined_rows, list(combined_rows[0].keys()))
    (combined_dir / "all_noise_shuju17.json").write_text(json.dumps(combined_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    origin_dir = combined_dir / "origin_tables"
    origin_dir.mkdir(parents=True, exist_ok=True)
    for noise_type, cfg in NOISE_CONFIGS.items():
        origin_rows = build_origin_rows(by_noise[noise_type])
        origin_fields = list(origin_rows[0].keys())
        base.write_csv(origin_dir / cfg["origin_name"], origin_rows, origin_fields)

    combined_origin_rows = build_combined_origin_rows(by_noise)
    base.write_csv(
        origin_dir / "All_noise_accuracy_origin17.csv",
        combined_origin_rows,
        list(combined_origin_rows[0].keys()),
    )

    print(origin_dir / "Gaussian_noise_accuracy_origin17.csv")
    print(origin_dir / "Poisson_noise_accuracy_origin17.csv")
    print(origin_dir / "Salt_noise_accuracy_origin17.csv")
    print(origin_dir / "All_noise_accuracy_origin17.csv")


if __name__ == "__main__":
    main()

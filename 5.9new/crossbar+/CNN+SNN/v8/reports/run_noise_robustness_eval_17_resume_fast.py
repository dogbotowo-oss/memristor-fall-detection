from __future__ import annotations

import csv
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
    },
    "possion": {
        "folder": REPORTS_DIR.parent / "v8-final-possion" / "shuju",
        "origin_name": "Poisson_noise_accuracy_origin17.csv",
    },
    "salt": {
        "folder": REPORTS_DIR.parent / "v8-final-salt" / "shuju",
        "origin_name": "Salt_noise_accuracy_origin17.csv",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("noise_robustness_eval_base17_fast", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_existing_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cleaned: list[dict[str, object]] = []
    for row in rows:
        cleaned.append({str(key).lstrip("\ufeff"): value for key, value in row.items()})
    return cleaned


def sort_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(row: dict[str, object]) -> tuple[float, int]:
        return (float(row["noise_percent"]), int(row["level_index"]))

    return sorted(rows, key=key)


def sort_video_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(row: dict[str, object]) -> tuple[float, str]:
        return (float(row["noise_percent"]), str(row["video_path"]))

    return sorted(rows, key=key)


def build_origin_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in sort_summary_rows(rows):
        out.append(
            {
                "NoiseLevel_percent": float(row["noise_percent"]),
                "Accuracy_percent": float(row["accuracy_percent"]),
                "BalancedAccuracy_percent": float(row["balanced_accuracy_percent"]),
                "ADLSpecificity_percent": float(row["adl_specificity_percent"]),
                "FallRecall_percent": float(row["fall_recall_percent"]),
                "Precision_percent": float(row["precision_percent"]),
                "TP": int(row["tp"]),
                "TN": int(row["tn"]),
                "FP": int(row["fp"]),
                "FN": int(row["fn"]),
            }
        )
    return out


def build_combined_origin_rows(by_noise: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    gaosi_rows = sort_summary_rows(by_noise["gaosi"])
    possion_rows = sort_summary_rows(by_noise["possion"])
    salt_rows = sort_summary_rows(by_noise["salt"])
    row_count = min(len(gaosi_rows), len(possion_rows), len(salt_rows))
    combined: list[dict[str, object]] = []
    for idx in range(row_count):
        g = gaosi_rows[idx]
        p = possion_rows[idx]
        s = salt_rows[idx]
        combined.append(
            {
                "NoiseLevel_percent": float(g["noise_percent"]),
                "Gaussian_Accuracy_percent": float(g["accuracy_percent"]),
                "Poisson_Accuracy_percent": float(p["accuracy_percent"]),
                "Salt_Accuracy_percent": float(s["accuracy_percent"]),
                "Gaussian_FallRecall_percent": float(g["fall_recall_percent"]),
                "Poisson_FallRecall_percent": float(p["fall_recall_percent"]),
                "Salt_FallRecall_percent": float(s["fall_recall_percent"]),
            }
        )
    return combined


def cache_videos(base, mod, config: dict[str, object], videos: list[Path]) -> list[dict[str, object]]:
    image_size = int(config.get("image_size", 64))
    use_pixel_human = bool(config.get("use_pixel_human", False))
    use_silhouette_human = bool(config.get("use_silhouette_human", False))
    pixel_grid_size = int(config.get("pixel_grid_size", 20))

    cache: list[dict[str, object]] = []
    for video_path in videos:
        frames, fps = base.read_video_frames_memory_safe(
            mod,
            video_path,
            image_size=image_size,
            use_pixel_human=use_pixel_human,
            use_silhouette_human=use_silhouette_human,
            pixel_grid_size=pixel_grid_size,
        )
        cache.append(
            {
                "video_path": video_path,
                "frames": frames,
                "fps": fps,
                "true_label": base.video_true_label(mod, video_path),
            }
        )
        print(f"cached {video_path.name} frames={len(frames)} fps={fps:.3f}", flush=True)
    return cache


def predict_cached_video(base, mod, model, config, item, device, resources, noise_type: str, level: float, level_index: int):
    window_size = int(config.get("window_size", 16))
    infer_stride = int(config.get("infer_stride", 4))
    use_crossbar = bool(config.get("use_crossbar", True))
    frames = base.apply_image_noise(
        item["frames"],
        noise_type=noise_type,
        level=level,
        level_index=level_index,
        seed=base.stable_seed(noise_type, level_index, item["video_path"]),
    )
    inference = mod.predict_video_frames(
        model=model,
        frames=frames,
        fps=float(item["fps"]),
        window_size=window_size,
        stride=infer_stride,
        use_crossbar=use_crossbar,
        hrs_values=base.np.asarray(resources["hrs_values"], dtype=base.np.float32),
        lrs_values=base.np.asarray(resources["lrs_values"], dtype=base.np.float32),
        device_dynamics=resources["device_dynamics"],
        crossbar_readout_noise_scale=float(resources["crossbar_readout_noise_scale"]),
        device=device,
        batch_size=max(1, min(int(config.get("batch_size", 8)), 8)),
        fall_prob_threshold=0.0,
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    return {
        "video_path": str(item["video_path"]),
        "fps": float(item["fps"]),
        "num_frames": int(inference["num_frames"]),
        "frame_probabilities": inference["frame_probabilities"],
        "true_label": int(item["true_label"]),
    }


def evaluate_level_cached(base, mod, model, config, cache, device, resources, noise_type: str, level: float, level_index: int):
    rows: list[dict[str, object]] = []
    for item in cache:
        result = predict_cached_video(base, mod, model, config, item, device, resources, noise_type, level, level_index)
        pred_label = base.classify_video(mod, result, base.THRESHOLD)
        rows.append(
            {
                "noise_type": noise_type,
                "noise_percent": level * 100.0,
                "level_index": level_index,
                "threshold": base.THRESHOLD,
                "video_path": str(item["video_path"]),
                "true_label": int(result["true_label"]),
                "pred_label": int(pred_label),
            }
        )
    metrics = base.compute_metrics(rows)
    summary = {
        "noise_type": noise_type,
        "noise_percent": level * 100.0,
        "noise_level": level,
        "level_index": level_index,
        "threshold": base.THRESHOLD,
        "gaussian_sigma": 0.30 * level if noise_type == "gaosi" and level_index > 0 else 0.0,
        "salt_probability": level if noise_type == "salt" and level_index > 0 else 0.0,
        "poisson_peak": base.poisson_peak(level_index) if noise_type == "possion" else None,
        **metrics,
    }
    return rows, summary


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
    cache = cache_videos(base, mod, config, videos)

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

    by_noise: dict[str, list[dict[str, object]]] = {}
    combined_rows: list[dict[str, object]] = []

    for noise_type, cfg in NOISE_CONFIGS.items():
        out_dir = cfg["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "shuju17.csv"
        video_path = out_dir / "video_shuju17.csv"
        json_path = out_dir / "shuju17.json"

        summary_rows = sort_summary_rows(load_existing_rows(summary_path))
        video_rows = sort_video_rows(load_existing_rows(video_path))
        completed = {round(float(row["noise_percent"]), 6) for row in summary_rows}

        for level_index, level in enumerate(LEVELS):
            noise_percent = round(level * 100.0, 6)
            if noise_percent in completed:
                print(f"skip {noise_type} {noise_percent:.2f}% already done", flush=True)
                continue
            rows, summary = evaluate_level_cached(
                base,
                mod,
                model,
                config,
                cache,
                device,
                resources,
                noise_type=noise_type,
                level=level,
                level_index=level_index,
            )
            summary_rows.append(summary)
            video_rows.extend(rows)
            summary_rows = sort_summary_rows(summary_rows)
            video_rows = sort_video_rows(video_rows)
            base.write_csv(summary_path, summary_rows, summary_fields)
            base.write_csv(video_path, video_rows, video_fields)
            json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"{noise_type} {level * 100:.2f}% | "
                f"acc={summary['accuracy_percent']:.2f} "
                f"bal={summary['balanced_accuracy_percent']:.2f} "
                f"adl={summary['adl_specificity_percent']:.2f} "
                f"fall={summary['fall_recall_percent']:.2f}",
                flush=True,
            )

        by_noise[noise_type] = sort_summary_rows(summary_rows)
        combined_rows.extend(by_noise[noise_type])

    combined_dir = REPORTS_DIR / "noise_robustness_subject4_shuju"
    combined_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(combined_dir / "all_noise_shuju17.csv", combined_rows, list(combined_rows[0].keys()))
    (combined_dir / "all_noise_shuju17.json").write_text(json.dumps(combined_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    origin_dir = combined_dir / "origin_tables"
    origin_dir.mkdir(parents=True, exist_ok=True)
    for noise_type, cfg in NOISE_CONFIGS.items():
        rows = build_origin_rows(by_noise[noise_type])
        base.write_csv(origin_dir / cfg["origin_name"], rows, list(rows[0].keys()))

    combined_origin_rows = build_combined_origin_rows(by_noise)
    base.write_csv(origin_dir / "All_noise_accuracy_origin17.csv", combined_origin_rows, list(combined_origin_rows[0].keys()))

    print(origin_dir / "Gaussian_noise_accuracy_origin17.csv", flush=True)
    print(origin_dir / "Poisson_noise_accuracy_origin17.csv", flush=True)
    print(origin_dir / "Salt_noise_accuracy_origin17.csv", flush=True)
    print(origin_dir / "All_noise_accuracy_origin17.csv", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path


REPORTS_DIR = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports")
BASE_SCRIPT = REPORTS_DIR / "noise_robustness_eval.py"
LEVELS = [i / 15 for i in range(16)]
REPEAT_COUNT = 5

NOISE_CONFIGS = {
    "gaosi": {
        "folder": REPORTS_DIR.parent / "v8-final-gaosi" / "shuju",
        "origin_name": "Gaussian_noise_fallrecall_origin16_repeat5.csv",
    },
    "possion": {
        "folder": REPORTS_DIR.parent / "v8-final-possion" / "shuju",
        "origin_name": "Poisson_noise_fallrecall_origin16_repeat5.csv",
    },
    "salt": {
        "folder": REPORTS_DIR.parent / "v8-final-salt" / "shuju",
        "origin_name": "Salt_noise_fallrecall_origin16_repeat5.csv",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("noise_robustness_eval_base16_fast", BASE_SCRIPT)
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
    return sorted(
        rows,
        key=lambda row: (
            float(row["noise_percent"]),
            int(row["level_index"]),
            int(row.get("repeat_id", 0)),
        ),
    )


def sort_video_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["noise_percent"]),
            int(row["level_index"]),
            int(row.get("repeat_id", 0)),
            str(row["video_path"]),
        ),
    )


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def summarize_repeat_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in sort_summary_rows(rows):
        key = (int(row["level_index"]), round(float(row["noise_percent"]), 6))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (level_index, noise_percent), group_rows in sorted(grouped.items()):
        accuracy_values = [float(row["accuracy_percent"]) for row in group_rows]
        balanced_values = [float(row["balanced_accuracy_percent"]) for row in group_rows]
        adl_values = [float(row["adl_specificity_percent"]) for row in group_rows]
        fall_values = [float(row["fall_recall_percent"]) for row in group_rows]
        precision_values = [float(row["precision_percent"]) for row in group_rows]
        out.append(
            {
                "noise_type": str(group_rows[0]["noise_type"]),
                "noise_percent": float(noise_percent),
                "noise_level": float(group_rows[0]["noise_level"]),
                "level_index": int(level_index),
                "repeat_count": len(group_rows),
                "threshold": float(group_rows[0]["threshold"]),
                "gaussian_sigma": float(group_rows[0]["gaussian_sigma"] or 0.0),
                "salt_probability": float(group_rows[0]["salt_probability"] or 0.0),
                "poisson_peak": (
                    float(group_rows[0]["poisson_peak"])
                    if group_rows[0]["poisson_peak"] not in (None, "", "None")
                    else ""
                ),
                "accuracy_mean_percent": statistics.mean(accuracy_values),
                "accuracy_std_percent": sample_std(accuracy_values),
                "balanced_accuracy_mean_percent": statistics.mean(balanced_values),
                "balanced_accuracy_std_percent": sample_std(balanced_values),
                "adl_specificity_mean_percent": statistics.mean(adl_values),
                "adl_specificity_std_percent": sample_std(adl_values),
                "fall_recall_mean_percent": statistics.mean(fall_values),
                "fall_recall_std_percent": sample_std(fall_values),
                "precision_mean_percent": statistics.mean(precision_values),
                "precision_std_percent": sample_std(precision_values),
            }
        )
    return out


def build_origin_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in summarize_repeat_rows(rows):
        out.append(
            {
                "NoiseLevel_percent": float(row["noise_percent"]),
                "RepeatCount": int(row["repeat_count"]),
                "Accuracy_mean_percent": float(row["accuracy_mean_percent"]),
                "Accuracy_std_percent": float(row["accuracy_std_percent"]),
                "BalancedAccuracy_mean_percent": float(row["balanced_accuracy_mean_percent"]),
                "BalancedAccuracy_std_percent": float(row["balanced_accuracy_std_percent"]),
                "ADLSpecificity_mean_percent": float(row["adl_specificity_mean_percent"]),
                "ADLSpecificity_std_percent": float(row["adl_specificity_std_percent"]),
                "FallRecall_mean_percent": float(row["fall_recall_mean_percent"]),
                "FallRecall_std_percent": float(row["fall_recall_std_percent"]),
                "Precision_mean_percent": float(row["precision_mean_percent"]),
                "Precision_std_percent": float(row["precision_std_percent"]),
            }
        )
    return out


def build_combined_origin_rows(by_noise: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    gaosi_rows = summarize_repeat_rows(by_noise["gaosi"])
    possion_rows = summarize_repeat_rows(by_noise["possion"])
    salt_rows = summarize_repeat_rows(by_noise["salt"])
    row_count = min(len(gaosi_rows), len(possion_rows), len(salt_rows))
    combined: list[dict[str, object]] = []
    for idx in range(row_count):
        g = gaosi_rows[idx]
        p = possion_rows[idx]
        s = salt_rows[idx]
        combined.append(
            {
                "NoiseLevel_percent": float(g["noise_percent"]),
                "Gaussian_FallRecall_mean_percent": float(g["fall_recall_mean_percent"]),
                "Gaussian_FallRecall_std_percent": float(g["fall_recall_std_percent"]),
                "Poisson_FallRecall_mean_percent": float(p["fall_recall_mean_percent"]),
                "Poisson_FallRecall_std_percent": float(p["fall_recall_std_percent"]),
                "Salt_FallRecall_mean_percent": float(s["fall_recall_mean_percent"]),
                "Salt_FallRecall_std_percent": float(s["fall_recall_std_percent"]),
                "Gaussian_BalancedAccuracy_mean_percent": float(g["balanced_accuracy_mean_percent"]),
                "Gaussian_BalancedAccuracy_std_percent": float(g["balanced_accuracy_std_percent"]),
                "Poisson_BalancedAccuracy_mean_percent": float(p["balanced_accuracy_mean_percent"]),
                "Poisson_BalancedAccuracy_std_percent": float(p["balanced_accuracy_std_percent"]),
                "Salt_BalancedAccuracy_mean_percent": float(s["balanced_accuracy_mean_percent"]),
                "Salt_BalancedAccuracy_std_percent": float(s["balanced_accuracy_std_percent"]),
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


def predict_cached_video(
    base,
    mod,
    model,
    config,
    item,
    device,
    resources,
    noise_type: str,
    level: float,
    level_index: int,
    repeat_id: int,
):
    window_size = int(config.get("window_size", 16))
    infer_stride = int(config.get("infer_stride", 4))
    use_crossbar = bool(config.get("use_crossbar", True))
    frames = base.apply_image_noise(
        item["frames"],
        noise_type=noise_type,
        level=level,
        level_index=level_index,
        seed=base.stable_seed(noise_type, level_index, repeat_id, item["video_path"]),
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


def evaluate_level_cached(
    base,
    mod,
    model,
    config,
    cache,
    device,
    resources,
    noise_type: str,
    level: float,
    level_index: int,
    repeat_id: int,
):
    rows: list[dict[str, object]] = []
    for item in cache:
        result = predict_cached_video(
            base,
            mod,
            model,
            config,
            item,
            device,
            resources,
            noise_type,
            level,
            level_index,
            repeat_id,
        )
        pred_label = base.classify_video(mod, result, base.THRESHOLD)
        rows.append(
            {
                "noise_type": noise_type,
                "noise_percent": level * 100.0,
                "level_index": level_index,
                "repeat_id": repeat_id,
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
        "repeat_id": repeat_id,
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
        "repeat_id",
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
    video_fields = [
        "noise_type",
        "noise_percent",
        "level_index",
        "repeat_id",
        "threshold",
        "video_path",
        "true_label",
        "pred_label",
    ]

    by_noise: dict[str, list[dict[str, object]]] = {}
    combined_rows: list[dict[str, object]] = []
    combined_meanstd_rows: list[dict[str, object]] = []

    for noise_type, cfg in NOISE_CONFIGS.items():
        out_dir = cfg["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "shuju16_repeat5.csv"
        video_path = out_dir / "video_shuju16_repeat5.csv"
        json_path = out_dir / "shuju16_repeat5.json"
        meanstd_path = out_dir / "shuju16_repeat5_meanstd.csv"

        summary_rows = sort_summary_rows(load_existing_rows(summary_path))
        video_rows = sort_video_rows(load_existing_rows(video_path))
        completed = {
            (round(float(row["noise_percent"]), 6), int(row.get("repeat_id", 0)))
            for row in summary_rows
        }

        for repeat_id in range(1, REPEAT_COUNT + 1):
            for level_index, level in enumerate(LEVELS):
                noise_percent = round(level * 100.0, 6)
                complete_key = (noise_percent, repeat_id)
                if complete_key in completed:
                    print(f"skip {noise_type} {noise_percent:.2f}% repeat={repeat_id} already done", flush=True)
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
                    repeat_id=repeat_id,
                )
                summary_rows.append(summary)
                video_rows.extend(rows)
                summary_rows = sort_summary_rows(summary_rows)
                video_rows = sort_video_rows(video_rows)
                completed.add(complete_key)
                meanstd_rows = summarize_repeat_rows(summary_rows)
                base.write_csv(summary_path, summary_rows, summary_fields)
                base.write_csv(video_path, video_rows, video_fields)
                json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
                if meanstd_rows:
                    base.write_csv(meanstd_path, meanstd_rows, list(meanstd_rows[0].keys()))
                print(
                    f"{noise_type} {level * 100:.2f}% repeat={repeat_id} | "
                    f"acc={summary['accuracy_percent']:.2f} "
                    f"bal={summary['balanced_accuracy_percent']:.2f} "
                    f"adl={summary['adl_specificity_percent']:.2f} "
                    f"fall={summary['fall_recall_percent']:.2f}",
                    flush=True,
                )

        by_noise[noise_type] = sort_summary_rows(summary_rows)
        combined_rows.extend(by_noise[noise_type])
        combined_meanstd_rows.extend(summarize_repeat_rows(summary_rows))

    combined_dir = REPORTS_DIR / "noise_robustness_subject4_shuju"
    combined_dir.mkdir(parents=True, exist_ok=True)
    if combined_rows:
        base.write_csv(combined_dir / "all_noise_shuju16_repeat5.csv", combined_rows, list(combined_rows[0].keys()))
        (combined_dir / "all_noise_shuju16_repeat5.json").write_text(
            json.dumps(combined_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if combined_meanstd_rows:
        base.write_csv(
            combined_dir / "all_noise_shuju16_repeat5_meanstd.csv",
            combined_meanstd_rows,
            list(combined_meanstd_rows[0].keys()),
        )

    origin_dir = combined_dir / "origin_tables"
    origin_dir.mkdir(parents=True, exist_ok=True)
    for noise_type, cfg in NOISE_CONFIGS.items():
        rows = build_origin_rows(by_noise[noise_type])
        if rows:
            base.write_csv(origin_dir / cfg["origin_name"], rows, list(rows[0].keys()))

    combined_origin_rows = build_combined_origin_rows(by_noise)
    if combined_origin_rows:
        base.write_csv(
            origin_dir / "All_noise_fallrecall_origin16_repeat5.csv",
            combined_origin_rows,
            list(combined_origin_rows[0].keys()),
        )

    print(origin_dir / "Gaussian_noise_fallrecall_origin16_repeat5.csv", flush=True)
    print(origin_dir / "Poisson_noise_fallrecall_origin16_repeat5.csv", flush=True)
    print(origin_dir / "Salt_noise_fallrecall_origin16_repeat5.csv", flush=True)
    print(origin_dir / "All_noise_fallrecall_origin16_repeat5.csv", flush=True)


if __name__ == "__main__":
    main()

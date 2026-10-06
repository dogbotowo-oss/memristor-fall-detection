from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"F:\12.18-2")
V8_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8"
CODE_PATH = V8_DIR / "code" / "automatic_fall_detector.py"
MODEL_PATH = V8_DIR / "models" / "automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth"
DATA_DIR = ROOT / "data"
GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"

THRESHOLD = 0.70
LEVELS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
OUTPUT_DIRS = {
    "none": V8_DIR / "v8-final-none" / "shuju",
    "gaosi": V8_DIR / "v8-final-gaosi" / "shuju",
    "possion": V8_DIR / "v8-final-possion" / "shuju",
    "salt": V8_DIR / "v8-final-salt" / "shuju",
}


def load_module():
    spec = importlib.util.spec_from_file_location("automatic_fall_detector_v8_noise", CODE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {CODE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def poisson_peak(level_index: int) -> float | None:
    if level_index == 0:
        return None
    t = level_index / (len(LEVELS) - 1)
    max_peak = 220.0
    min_peak = 6.0
    return float(max_peak * ((min_peak / max_peak) ** t))


def apply_image_noise(frames: np.ndarray, noise_type: str, level: float, level_index: int, seed: int) -> np.ndarray:
    if noise_type == "none" or level_index == 0:
        return frames.astype(np.float32, copy=True)

    rng = np.random.default_rng(seed)
    x = frames.astype(np.float32, copy=True)

    if noise_type == "gaosi":
        sigma = 0.30 * float(level)
        return np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)

    if noise_type == "possion":
        peak = float(poisson_peak(level_index) or 220.0)
        return np.clip(rng.poisson(np.clip(x, 0.0, 1.0) * peak).astype(np.float32) / peak, 0.0, 1.0)

    if noise_type == "salt":
        out = x.copy()
        mask = rng.random(x.shape) < float(level)
        salt_mask = rng.random(x.shape) < 0.5
        out[mask & salt_mask] = 1.0
        out[mask & ~salt_mask] = 0.0
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unsupported noise_type: {noise_type}")


def read_video_frames_memory_safe(mod, video_path: Path, image_size: int, use_pixel_human: bool = False, use_silhouette_human: bool = False, pixel_grid_size: int = 20) -> tuple[np.ndarray, float]:
    if use_pixel_human or use_silhouette_human:
        return mod.read_video_frames(
            video_path,
            image_size=image_size,
            use_pixel_human=use_pixel_human,
            use_silhouette_human=use_silhouette_human,
            pixel_grid_size=pixel_grid_size,
        )

    cv2 = mod.cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append((small.astype(np.float32) / 255.0).astype(np.float32))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if fps <= 0:
        raise ValueError(f"Unable to read FPS from {video_path}")
    return np.stack(frames, axis=0).astype(np.float32), fps


def video_true_label(mod, video_path: Path) -> int:
    label_id = int(mod.infer_video_level_label_id(video_path, "fall"))
    return 1 if label_id in {mod.LABEL_NAME_TO_ID["pre_fall"], mod.LABEL_NAME_TO_ID["fall"]} else 0


def classify_video(mod, result: dict[str, object], threshold: float) -> int:
    frame_probs = np.asarray(result["frame_probabilities"], dtype=np.float32)
    fps = float(result["fps"])
    frame_preds = mod.calibrate_frame_predictions(
        frame_probs,
        fps=fps,
        fall_prob_threshold=float(threshold),
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    process_segment = mod.select_process_segment(
        frame_probs,
        frame_preds,
        fps,
        fall_prob_threshold=float(threshold),
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    return 1 if process_segment is not None else 0


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    tp = sum(1 for row in rows if row["true_label"] == 1 and row["pred_label"] == 1)
    tn = sum(1 for row in rows if row["true_label"] == 0 and row["pred_label"] == 0)
    fp = sum(1 for row in rows if row["true_label"] == 0 and row["pred_label"] == 1)
    fn = sum(1 for row in rows if row["true_label"] == 1 and row["pred_label"] == 0)
    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    adl_specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fall_recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "total": total,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "balanced_accuracy": 0.5 * (adl_specificity + fall_recall),
        "balanced_accuracy_percent": 50.0 * (adl_specificity + fall_recall),
        "adl_specificity": adl_specificity,
        "adl_specificity_percent": adl_specificity * 100.0,
        "fall_recall": fall_recall,
        "fall_recall_percent": fall_recall * 100.0,
        "precision": precision,
        "precision_percent": precision * 100.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def predict_video(
    mod,
    model,
    config: dict[str, object],
    video_path: Path,
    device: torch.device,
    resources: dict[str, object],
    noise_type: str,
    level: float,
    level_index: int,
) -> dict[str, object]:
    window_size = int(config.get("window_size", 16))
    image_size = int(config.get("image_size", 64))
    infer_stride = int(config.get("infer_stride", 4))
    use_crossbar = bool(config.get("use_crossbar", True))
    use_pixel_human = bool(config.get("use_pixel_human", False))
    use_silhouette_human = bool(config.get("use_silhouette_human", False))
    pixel_grid_size = int(config.get("pixel_grid_size", 20))

    frames, fps = read_video_frames_memory_safe(
        mod,
        video_path,
        image_size=image_size,
        use_pixel_human=use_pixel_human,
        use_silhouette_human=use_silhouette_human,
        pixel_grid_size=pixel_grid_size,
    )
    frames = apply_image_noise(
        frames,
        noise_type=noise_type,
        level=level,
        level_index=level_index,
        seed=stable_seed(noise_type, level_index, video_path),
    )
    inference = mod.predict_video_frames(
        model=model,
        frames=frames,
        fps=fps,
        window_size=window_size,
        stride=infer_stride,
        use_crossbar=use_crossbar,
        hrs_values=np.asarray(resources["hrs_values"], dtype=np.float32),
        lrs_values=np.asarray(resources["lrs_values"], dtype=np.float32),
        device_dynamics=resources["device_dynamics"],
        crossbar_readout_noise_scale=float(resources["crossbar_readout_noise_scale"]),
        device=device,
        batch_size=max(1, min(int(config.get("batch_size", 8)), 4)),
        fall_prob_threshold=0.0,
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    return {
        "video_path": str(video_path),
        "fps": float(fps),
        "num_frames": int(inference["num_frames"]),
        "frame_probabilities": inference["frame_probabilities"],
        "true_label": video_true_label(mod, video_path),
    }


def evaluate_level(
    mod,
    model,
    config: dict[str, object],
    videos: list[Path],
    device: torch.device,
    resources: dict[str, object],
    noise_type: str,
    level: float,
    level_index: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for video_path in videos:
        item = predict_video(
            mod,
            model,
            config,
            video_path,
            device,
            resources,
            noise_type=noise_type,
            level=level,
            level_index=level_index,
        )
        pred_label = classify_video(mod, item, THRESHOLD)
        rows.append(
            {
                "noise_type": noise_type,
                "noise_percent": level * 100.0,
                "level_index": level_index,
                "threshold": THRESHOLD,
                "video_path": str(video_path),
                "true_label": int(item["true_label"]),
                "pred_label": int(pred_label),
            }
        )
    metrics = compute_metrics(rows)
    summary = {
        "noise_type": noise_type,
        "noise_percent": level * 100.0,
        "noise_level": level,
        "level_index": level_index,
        "threshold": THRESHOLD,
        "gaussian_sigma": 0.30 * level if noise_type == "gaosi" and level_index > 0 else 0.0,
        "salt_probability": level if noise_type == "salt" and level_index > 0 else 0.0,
        "poisson_peak": poisson_peak(level_index) if noise_type == "possion" else None,
        **metrics,
    }
    return rows, summary


def write_readme(path: Path, noise_type: str, summaries: list[dict[str, object]]) -> None:
    best = summaries[0] if summaries else {}
    text = f"""# v8 final 图像噪声鲁棒性数据

本目录保存 `v8 final p006` 固定模型在 Subject4 上的测试阶段图像噪声扫描数据。

- 模型：`{MODEL_PATH}`
- 阈值：`{THRESHOLD}`
- 数据：GMDCSA Subject4 独立最终测试集
- 说明：这里不是重新训练曲线，横轴应使用 `noise_percent`；如果要画严格的 `Training epoch - Accuracy`，需要重新训练并保存每个 epoch 的模型。
- 噪声类型：`{noise_type}`
- 主数据：`shuju.csv`
- 视频级数据：`video_shuju.csv`
- 基线第一行 Accuracy：`{float(best.get("accuracy_percent", 0.0)):.2f}%`

Origin 建议：

1. 导入 `shuju.csv`。
2. X 轴选择 `noise_percent`。
3. Y 轴选择 `accuracy_percent`。
4. 如果需要补充曲线，可同时画 `balanced_accuracy_percent`、`adl_specificity_percent`、`fall_recall_percent`。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_module()
    model, config = mod.load_model_checkpoint(MODEL_PATH, device)
    model.eval()

    use_crossbar = bool(config.get("use_crossbar", True))
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", 0.0))
    if use_crossbar:
        hrs_values, lrs_values = mod.load_conductance_pair(DATA_DIR)
    else:
        hrs_values = np.array([0.1], dtype=np.float32)
        lrs_values = np.array([0.9], dtype=np.float32)
    device_dynamics = (
        mod.load_device_dynamics(DATA_DIR)
        if use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
        else None
    )
    resources = {
        "hrs_values": hrs_values,
        "lrs_values": lrs_values,
        "device_dynamics": device_dynamics,
        "crossbar_readout_noise_scale": crossbar_readout_noise_scale,
    }

    videos = mod.list_video_files(SUBJECT4_DIR)
    if not videos:
        raise RuntimeError(f"No Subject4 videos found: {SUBJECT4_DIR}")

    combined_rows: list[dict[str, object]] = []
    for noise_type, out_dir in OUTPUT_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        levels = [(0, 0.0)] if noise_type == "none" else list(enumerate(LEVELS))
        summary_rows: list[dict[str, object]] = []
        video_rows: list[dict[str, object]] = []
        for level_index, level in levels:
            rows, summary = evaluate_level(
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
            print(
                f"{noise_type} {level * 100:.1f}% | "
                f"acc={summary['accuracy_percent']:.2f} "
                f"bal={summary['balanced_accuracy_percent']:.2f} "
                f"adl={summary['adl_specificity_percent']:.2f} "
                f"fall={summary['fall_recall_percent']:.2f}"
            )

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
        write_csv(out_dir / "shuju.csv", summary_rows, summary_fields)
        write_csv(out_dir / "video_shuju.csv", video_rows, video_fields)
        (out_dir / "shuju.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        write_readme(out_dir.parent / "README.md", noise_type, summary_rows)

    combined_dir = V8_DIR / "reports" / "noise_robustness_subject4_shuju"
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_csv(combined_dir / "all_noise_shuju.csv", combined_rows, list(combined_rows[0].keys()))
    (combined_dir / "all_noise_shuju.json").write_text(json.dumps(combined_rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

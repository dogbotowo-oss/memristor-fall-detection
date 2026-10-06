from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SOURCE = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\code\automatic_fall_detector.py"
)
DEFAULT_CHECKPOINT = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\models"
    r"\snn_group_channel_gate_p000\automatic_fall_event_detector_snn_group_channel_gate_p000.pth"
)
DEFAULT_SUBJECT_ROOT = Path(
    r"F:\12.18-2\测试集\13354453\ekramalam"
    r"\GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    r"\ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0713-snn-channel-diagnostic\reports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SNN gate/channel contributions without retraining.")
    parser.add_argument("--source-code", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--subject-root", type=Path, default=DEFAULT_SUBJECT_ROOT)
    parser.add_argument("--zenodo-train", type=Path, default=Path(r"F:\12.18-2\zenodo_falldb_video_split\train"))
    parser.add_argument("--zenodo-val", type=Path, default=Path(r"F:\12.18-2\zenodo_falldb_video_split\val"))
    parser.add_argument("--data-dir", type=Path, default=Path(r"F:\12.18-2\data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--hard-percentile", type=float, default=0.88)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def load_detector_module(source_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("v8_detector_channel_diagnostic", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import detector source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_split_windows(
    detector: Any,
    roots: list[Path],
    config: dict[str, Any],
    stride: int,
    include_train_adl_context: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if include_train_adl_context:
        kwargs = {
            "adl_context_path_keyword": str(config.get("adl_context_path_keyword", "")),
            "adl_context_min_percentile": float(config.get("adl_context_min_percentile", 0.95)),
            "adl_context_radius_windows": int(config.get("adl_context_radius_windows", 0)),
            "adl_context_extra_copies": int(config.get("adl_context_extra_copies", 0)),
        }
    return detector.load_external_labeled_windows(
        roots,
        image_size=int(config["image_size"]),
        window_size=int(config["window_size"]),
        stride=max(1, stride),
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=bool(config.get("use_pixel_human", False)),
        use_silhouette_human=bool(config.get("use_silhouette_human", False)),
        pixel_grid_size=int(config.get("pixel_grid_size", 20)),
        **kwargs,
    )


def build_dataset(
    detector: Any,
    windows: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    data_dir: Path,
    base_seed: int,
) -> Any:
    use_crossbar = bool(config.get("use_crossbar", True))
    if use_crossbar:
        hrs_values, lrs_values = detector.load_conductance_pair(data_dir)
    else:
        hrs_values = np.array([0.1], dtype=np.float32)
        lrs_values = np.array([0.9], dtype=np.float32)
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    readout_noise = float(config.get("crossbar_readout_noise_scale", 0.0))
    device_dynamics = detector.load_device_dynamics(data_dir) if (use_device_dynamics or readout_noise > 0) else None
    return detector.CombinedFallDataset(
        video_windows=windows,
        video_labels=labels,
        auxiliary_records=[],
        window_size=int(config["window_size"]),
        image_size=int(config["image_size"]),
        use_crossbar=use_crossbar,
        use_pixel_human=bool(config.get("use_pixel_human", False)),
        use_silhouette_human=bool(config.get("use_silhouette_human", False)),
        pixel_grid_size=int(config.get("pixel_grid_size", 20)),
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=readout_noise,
        augment=False,
        base_seed=base_seed,
    )


def collect_split(
    detector: Any,
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = detector.make_loader(dataset, max(1, batch_size), False, None, device)
    collected: dict[str, list[np.ndarray]] = {
        "pooled": [],
        "raw_snn": [],
        "gate": [],
        "labels": [],
        "hard_scores": [],
    }
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_hard_score in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_size_now, time_steps, channels, height, width = batch_x.shape
            encoded = batch_x.view(batch_size_now * time_steps, channels, height, width)
            encoded = model.encoder(encoded).view(batch_size_now * time_steps, -1)
            encoded = model.proj(encoded).view(batch_size_now, time_steps, -1)
            temporal_out, _ = model.temporal(encoded)
            pooled = torch.cat(
                [
                    temporal_out.mean(dim=1),
                    temporal_out.max(dim=1).values,
                    temporal_out[:, -1, :],
                ],
                dim=1,
            )
            event_features = model.extract_temporal_event_features(batch_x)
            raw_snn = model.run_lif_temporal_branch(batch_x, event_features)
            event_score = model.extract_fall_event_score(event_features)
            _gated_snn, gate = model.apply_snn_fusion_gate(pooled, raw_snn, event_score)
            collected["pooled"].append(pooled.cpu().numpy())
            collected["raw_snn"].append(raw_snn.cpu().numpy())
            collected["gate"].append(gate.cpu().numpy())
            collected["labels"].append(batch_y.numpy())
            collected["hard_scores"].append(batch_hard_score.numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def selection_masks(
    labels: np.ndarray,
    hard_scores: np.ndarray,
    normal_id: int,
    fall_id: int,
    hard_percentile: float,
) -> dict[str, np.ndarray]:
    normal_mask = labels == normal_id
    fall_mask = labels == fall_id
    normal_scores = hard_scores[normal_mask]
    if normal_scores.size:
        threshold = float(np.quantile(normal_scores, np.clip(hard_percentile, 0.0, 0.99)))
        hard_adl_mask = normal_mask & (hard_scores >= threshold)
    else:
        threshold = float("nan")
        hard_adl_mask = normal_mask
    return {
        "adl": normal_mask,
        "hard_adl": hard_adl_mask,
        "fall": fall_mask,
        "hard_threshold": np.asarray([threshold], dtype=np.float32),
    }


def safe_column_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not bool(np.any(mask)):
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    return values[mask].mean(axis=0).astype(np.float32)


def split_channel_stats(
    payload: dict[str, np.ndarray],
    normal_id: int,
    fall_id: int,
    hard_percentile: float,
) -> dict[str, Any]:
    masks = selection_masks(
        payload["labels"], payload["hard_scores"], normal_id, fall_id, hard_percentile
    )
    contribution = payload["raw_snn"] * payload["gate"]
    stats: dict[str, Any] = {
        "hard_threshold": float(masks["hard_threshold"][0]),
        "counts": {name: int(mask.sum()) for name, mask in masks.items() if name != "hard_threshold"},
    }
    for value_name, values in (
        ("raw", payload["raw_snn"]),
        ("gate", payload["gate"]),
        ("contribution", contribution),
    ):
        for class_name in ("adl", "hard_adl", "fall"):
            stats[f"{value_name}_{class_name}"] = safe_column_mean(values, masks[class_name])
    overall_std = contribution.std(axis=0).astype(np.float32)
    stats["discrimination"] = stats["contribution_fall"] - stats["contribution_hard_adl"]
    stats["normalized_discrimination"] = stats["discrimination"] / np.maximum(overall_std, 1e-6)
    return stats


def classification_metrics(predictions: np.ndarray, labels: np.ndarray, normal_id: int, fall_id: int) -> dict[str, float]:
    accuracy = float(np.mean(predictions == labels))
    normal_mask = labels == normal_id
    fall_mask = labels == fall_id
    normal_accuracy = float(np.mean(predictions[normal_mask] == labels[normal_mask]))
    fall_accuracy = float(np.mean(predictions[fall_mask] == labels[fall_mask]))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float((normal_accuracy + fall_accuracy) / 2.0),
        "adl_specificity": normal_accuracy,
        "fall_recall": fall_accuracy,
    }


def evaluate_channel_mask(
    model: torch.nn.Module,
    payload: dict[str, np.ndarray],
    channel_mask: np.ndarray,
    normal_id: int,
    fall_id: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    mask_tensor = torch.from_numpy(channel_mask.astype(np.float32)).to(device).view(1, -1)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(payload["labels"]), max(1, batch_size)):
            end = min(start + max(1, batch_size), len(payload["labels"]))
            pooled = torch.from_numpy(payload["pooled"][start:end]).to(device)
            raw_snn = torch.from_numpy(payload["raw_snn"][start:end]).to(device)
            gate = torch.from_numpy(payload["gate"][start:end]).to(device)
            gated_snn = raw_snn * gate * mask_tensor
            logits = model.head(torch.cat([pooled, gated_snn], dim=1))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return classification_metrics(
        np.concatenate(predictions), payload["labels"], normal_id, fall_id
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    detector = load_detector_module(args.source_code)
    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    model, config = detector.load_model_checkpoint(args.checkpoint, device)
    if str(config.get("snn_fusion_gate_mode")) != "group_channel":
        raise ValueError("Channel diagnostics require a group_channel checkpoint.")

    train_roots = [args.subject_root / "Subject 1", args.subject_root / "Subject 2", args.zenodo_train]
    val_roots = [args.subject_root / "Subject 3", args.zenodo_val]
    train_windows, train_labels, train_summary = load_split_windows(
        detector, train_roots, config, args.stride, include_train_adl_context=True
    )
    train_dataset = build_dataset(detector, train_windows, train_labels, config, args.data_dir, 42)
    train_payload = collect_split(detector, model, train_dataset, args.batch_size, device)
    del train_dataset, train_windows

    val_windows, val_labels, val_summary = load_split_windows(
        detector, val_roots, config, args.stride, include_train_adl_context=False
    )
    val_dataset = build_dataset(detector, val_windows, val_labels, config, args.data_dir, 10041)
    val_payload = collect_split(detector, model, val_dataset, args.batch_size, device)

    normal_id = int(detector.LABEL_NAME_TO_ID["normal"])
    fall_id = int(detector.LABEL_NAME_TO_ID["fall"])
    train_stats = split_channel_stats(train_payload, normal_id, fall_id, args.hard_percentile)
    val_stats = split_channel_stats(val_payload, normal_id, fall_id, args.hard_percentile)
    channel_count = int(train_payload["raw_snn"].shape[1])
    group_dim = channel_count // 3
    group_names = ("mean_spike", "max_spike", "last_spike")

    order = np.argsort(-train_stats["normalized_discrimination"])
    ranks = np.empty(channel_count, dtype=np.int64)
    ranks[order] = np.arange(1, channel_count + 1)
    channel_rows: list[dict[str, Any]] = []
    for channel_index in range(channel_count):
        group_index = channel_index // group_dim
        channel_rows.append(
            {
                "channel_index": channel_index,
                "group": group_names[group_index],
                "channel_in_group": channel_index % group_dim,
                "train_rank": int(ranks[channel_index]),
                "train_gate_adl": float(train_stats["gate_adl"][channel_index]),
                "train_gate_hard_adl": float(train_stats["gate_hard_adl"][channel_index]),
                "train_gate_fall": float(train_stats["gate_fall"][channel_index]),
                "train_contribution_adl": float(train_stats["contribution_adl"][channel_index]),
                "train_contribution_hard_adl": float(train_stats["contribution_hard_adl"][channel_index]),
                "train_contribution_fall": float(train_stats["contribution_fall"][channel_index]),
                "train_discrimination": float(train_stats["discrimination"][channel_index]),
                "train_normalized_discrimination": float(train_stats["normalized_discrimination"][channel_index]),
                "val_gate_adl": float(val_stats["gate_adl"][channel_index]),
                "val_gate_hard_adl": float(val_stats["gate_hard_adl"][channel_index]),
                "val_gate_fall": float(val_stats["gate_fall"][channel_index]),
                "val_contribution_adl": float(val_stats["contribution_adl"][channel_index]),
                "val_contribution_hard_adl": float(val_stats["contribution_hard_adl"][channel_index]),
                "val_contribution_fall": float(val_stats["contribution_fall"][channel_index]),
                "val_discrimination": float(val_stats["discrimination"][channel_index]),
                "val_normalized_discrimination": float(val_stats["normalized_discrimination"][channel_index]),
            }
        )
    channel_rows.sort(key=lambda row: int(row["train_rank"]))
    write_csv(args.output_dir / "channel_importance.csv", channel_rows)

    group_rows: list[dict[str, Any]] = []
    for group_index, group_name in enumerate(group_names):
        start = group_index * group_dim
        end = start + group_dim
        group_rows.append(
            {
                "group": group_name,
                "channel_start": start,
                "channel_end": end - 1,
                "train_discrimination_mean": float(np.mean(train_stats["discrimination"][start:end])),
                "train_normalized_discrimination_mean": float(np.mean(train_stats["normalized_discrimination"][start:end])),
                "val_discrimination_mean": float(np.mean(val_stats["discrimination"][start:end])),
                "val_normalized_discrimination_mean": float(np.mean(val_stats["normalized_discrimination"][start:end])),
                "val_gate_adl_mean": float(np.mean(val_stats["gate_adl"][start:end])),
                "val_gate_hard_adl_mean": float(np.mean(val_stats["gate_hard_adl"][start:end])),
                "val_gate_fall_mean": float(np.mean(val_stats["gate_fall"][start:end])),
            }
        )
    write_csv(args.output_dir / "group_summary.csv", group_rows)

    mask_rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {"all_96": np.ones(channel_count, dtype=np.float32)}
    for keep_count in (24, 48, 72):
        mask = np.zeros(channel_count, dtype=np.float32)
        mask[order[:keep_count]] = 1.0
        masks[f"top_{keep_count}"] = mask
    for group_index, group_name in enumerate(group_names):
        mask = np.zeros(channel_count, dtype=np.float32)
        mask[group_index * group_dim : (group_index + 1) * group_dim] = 1.0
        masks[f"group_{group_name}"] = mask
    for left, right in ((0, 1), (0, 2), (1, 2)):
        mask = np.zeros(channel_count, dtype=np.float32)
        mask[left * group_dim : (left + 1) * group_dim] = 1.0
        mask[right * group_dim : (right + 1) * group_dim] = 1.0
        masks[f"groups_{group_names[left]}_{group_names[right]}"] = mask

    for mask_name, channel_mask in masks.items():
        metrics = evaluate_channel_mask(
            model, val_payload, channel_mask, normal_id, fall_id, device, args.batch_size
        )
        mask_rows.append(
            {
                "mask": mask_name,
                "kept_channels": int(channel_mask.sum()),
                **metrics,
            }
        )
    mask_rows.sort(key=lambda row: (-float(row["balanced_accuracy"]), -float(row["fall_recall"])))
    write_csv(args.output_dir / "mask_validation_scan.csv", mask_rows)

    recommended = mask_rows[0]
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "data_rule": {
            "ranking": "Subject1-2 + Zenodo train",
            "validation": "Subject3 + Zenodo val",
            "subject4_used": False,
        },
        "train_windows": int(len(train_payload["labels"])),
        "val_windows": int(len(val_payload["labels"])),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "train_hard_adl_threshold": float(train_stats["hard_threshold"]),
        "val_hard_adl_threshold": float(val_stats["hard_threshold"]),
        "train_counts": train_stats["counts"],
        "val_counts": val_stats["counts"],
        "negative_train_channels": int(np.sum(train_stats["discrimination"] <= 0)),
        "negative_val_channels": int(np.sum(val_stats["discrimination"] <= 0)),
        "recommended_inference_mask": recommended,
        "note": "Mask scan is diagnostic only; the selected structure must be retrained before final comparison.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "recommended_masks.json").write_text(
        json.dumps(
            {
                name: np.flatnonzero(mask).astype(int).tolist()
                for name, mask in masks.items()
                if name.startswith("top_")
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

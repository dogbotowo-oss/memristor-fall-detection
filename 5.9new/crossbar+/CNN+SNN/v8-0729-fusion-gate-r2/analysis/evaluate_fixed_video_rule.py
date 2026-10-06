"""Evaluate one checkpoint with fixed window and video decision rules only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
TAG = sys.argv[1]
SOURCE_NAME = sys.argv[2] if len(sys.argv) > 2 else "all"
THRESHOLD = 0.90
ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and (path / "13354453").exists())
GMDCSA_ROOT = next(path for path in (TESTSET_DIR / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76"))
S3_DIR = GMDCSA_ROOT / "Subject 3"
ZEN_DIR = ROOT / "zenodo_falldb_video_split" / "val"
CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}_ema.pth"


def metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float | int]:
    true_labels = true_labels.astype(np.int64)
    predicted_labels = predicted_labels.astype(np.int64)
    tp = int(((true_labels == 1) & (predicted_labels == 1)).sum())
    tn = int(((true_labels == 0) & (predicted_labels == 0)).sum())
    fp = int(((true_labels == 0) & (predicted_labels == 1)).sum())
    fn = int(((true_labels == 1) & (predicted_labels == 0)).sum())
    specificity = tn / max(tn + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(true_labels), 1),
        "balanced_accuracy": 0.5 * (specificity + recall),
        "adl_specificity": specificity,
        "fall_recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    source_dirs = {
        "subject3": [S3_DIR],
        "zenodo": [ZEN_DIR],
        "all": [S3_DIR, ZEN_DIR],
    }
    if SOURCE_NAME not in source_dirs:
        raise ValueError(f"Unknown source {SOURCE_NAME!r}; use subject3, zenodo, or all")
    sys.argv = ["run_refined_gate_r2.py"]
    spec = importlib.util.spec_from_file_location("fixed_rule_runner", RUNNER_PATH)
    r2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r2
    spec.loader.exec_module(r2)
    base = r2.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = dict(checkpoint["config"])
    r2.REFINED_MODE = str(cfg.get("snn_fusion_gate_mode", "group"))
    r2.REFINED_FLOOR = float(cfg.get("snn_fusion_gate_floor", 0.2))
    r2.REFINED_GROUP_BIAS = float(cfg.get("refined_group_bias", 0.0))
    r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
    r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.0))
    model = r2.RefinedFallEventDetectorR2(
        hidden_dim=int(cfg.get("hidden_dim", 64)), num_classes=len(base.LABEL_NAME_TO_ID),
        use_snn_temporal_branch=True, snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
        snn_decay=float(cfg.get("snn_decay", 0.82)), snn_threshold=float(cfg.get("snn_threshold", 0.30)),
        snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
        learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
        use_snn_fusion_gate=True, snn_fusion_gate_mode="group",
        snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
        snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
        snn_input_mode=str(cfg.get("snn_input_mode", "scalar7")),
        snn_pool_mode=str(cfg.get("snn_pool_mode", "global3")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
    device_dynamics = base.load_device_dynamics(DATA_DIR)
    fall_id = base.LABEL_NAME_TO_ID["fall"]

    def score(stride: int):
        windows, labels, summary = base.load_external_labeled_windows(
            source_dirs[SOURCE_NAME], image_size=int(cfg.get("image_size", 64)),
            window_size=int(cfg.get("window_size", 16)), stride=stride, max_videos=0,
            fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
        )
        dataset = base.CombinedFallDataset(
            video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels, auxiliary_records=[],
            window_size=int(cfg.get("window_size", 16)), image_size=int(cfg.get("image_size", 64)),
            use_crossbar=True, use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
            hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
            crossbar_readout_noise_scale=float(cfg.get("crossbar_readout_noise_scale", 0.25)),
            augment=False, base_seed=10041,
        )
        print(f"Scoring stride={stride} windows={len(dataset)} batch=128", flush=True)
        probabilities, predictions, aux_probabilities = [], [], []
        with torch.inference_mode():
            for offset in range(0, len(dataset), 128):
                batch = torch.stack([dataset[index][0] for index in range(offset, min(offset + 128, len(dataset)))])
                logits, gate_info = model.forward_with_gate(batch.to(device))
                probabilities.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
                predictions.append((torch.argmax(logits, dim=1) == fall_id).cpu())
                auxiliary_logits = gate_info.get("snn_class_logits")
                if auxiliary_logits is not None:
                    aux_probabilities.append(torch.softmax(auxiliary_logits, dim=1)[:, 1].cpu())
        return np.concatenate([part.numpy() for part in probabilities]), np.concatenate(
            [part.numpy() for part in predictions]
        ).astype(np.int64), labels, summary, (
            np.concatenate([part.numpy() for part in aux_probabilities]) if aux_probabilities else None
        )

    probabilities16, predictions16, labels16, _summary16, aux16 = score(16)
    true16 = (labels16 == fall_id).astype(np.int64)
    window_metrics = metrics(true16, predictions16)
    probabilities4, _predictions4, _labels4, summary4, _aux4 = score(4)
    video_ground_truth = json.loads((EXP_ROOT / "analysis" / "val_video_gt.json").read_text(encoding="utf-8"))
    true_videos, predicted_videos, offset = [], [], 0
    for row in summary4["videos"]:
        count = int(row["num_windows"])
        peak = float(probabilities4[offset:offset + count].max())
        offset += count
        video_path = Path(row["video_path"])
        try:
            relative_path = video_path.relative_to(S3_DIR)
        except ValueError:
            relative_path = video_path.relative_to(ZEN_DIR)
        true_videos.append(int(video_ground_truth[str(relative_path).replace("\\", "/")]["true_fall"]))
        predicted_videos.append(int(peak >= THRESHOLD))
    result = {
        "tag": TAG,
        "source": SOURCE_NAME,
        "ema_source_epoch": cfg.get("weight_ema_source_epoch"),
        "fixed_video_rule": "stride=4, video peak fall probability >= 0.90",
        "window_stride16": window_metrics,
        "video_stride4": metrics(np.asarray(true_videos), np.asarray(predicted_videos)),
    }
    if aux16 is not None:
        order = np.argsort(aux16)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(aux16) + 1)
        positives = true16 == 1
        result["snn_aux_auc"] = float(
            (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
            / max(positives.sum() * (~positives).sum(), 1)
        )
    output_path = EXP_ROOT / "analysis" / f"{TAG}_fixed_gt_{SOURCE_NAME}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate S13 with the frozen validation protocol and no threshold search."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_s13_semantic_residual.py"
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_ema0999_s13_semres"
THRESHOLD = 0.90
ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and (path / "13354453").exists())
GMDCSA_ROOT = next(
    path
    for path in (TESTSET_DIR / "13354453" / "ekramalam").glob(
        "GMDCSA24*/ekramalam-GMDCSA24*5abac76"
    )
)
S3_DIR = GMDCSA_ROOT / "Subject 3"
ZEN_DIR = ROOT / "zenodo_falldb_video_split" / "val"
MODEL_DIR = EXP_ROOT / "models" / TAG


def load_runner() -> Any:
    sys.argv = ["run_s13_semantic_residual.py"]
    spec = importlib.util.spec_from_file_location("s13_fixed_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import S13 runner: {RUNNER_PATH}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.base.LABEL_NAME_TO_ID = {"normal": 0, "running": 0, "pre_fall": 1, "fall": 1}
    runner.base.LABEL_ID_TO_NAME = {0: "normal", 1: "fall"}
    return runner


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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / max(positive_count * negative_count, 1)
    )


def source_for_video(video_path: Path) -> str:
    try:
        video_path.relative_to(S3_DIR)
        return "subject3"
    except ValueError:
        video_path.relative_to(ZEN_DIR)
        return "zenodo"


def source_vector(summary: dict[str, Any]) -> np.ndarray:
    sources: list[str] = []
    for row in summary["videos"]:
        source = source_for_video(Path(row["video_path"]))
        sources.extend([source] * int(row["num_windows"]))
    return np.asarray(sources)


def build_dataset(base: Any, windows: np.ndarray, labels: np.ndarray, cfg: dict[str, Any], seed: int):
    hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
    device_dynamics = base.load_device_dynamics(DATA_DIR)
    return base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32),
        video_labels=labels,
        auxiliary_records=[],
        window_size=int(cfg.get("window_size", 16)),
        image_size=int(cfg.get("image_size", 64)),
        use_crossbar=True,
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=float(cfg.get("crossbar_readout_noise_scale", 0.25)),
        augment=False,
        base_seed=seed,
    )


def load_model(runner: Any, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["config"])
    class_names = checkpoint.get("class_names")
    if class_names != {0: "normal", 1: "fall"}:
        raise ValueError(f"S13 checkpoint is not binary: class_names={class_names!r}")
    runner.SEMANTIC_DIM = int(cfg.get("snn_semantic_dim_per_group", 48))
    runner.RESIDUAL_MAX_SCALE = float(cfg.get("snn_residual_max_scale", 0.25))
    runner.DYNAMIC_ENCODER_MODE = str(cfg.get("snn_dynamic_encoder_mode", "legacy"))
    runner.DYNAMICS_MODE = str(cfg.get("snn_dynamics_mode", "uniform"))
    model = runner.S13SemanticResidualDetector(
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_classes=2,
        use_snn_temporal_branch=True,
        snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
        snn_decay=float(cfg.get("snn_decay", 0.82)),
        snn_threshold=float(cfg.get("snn_threshold", 0.30)),
        snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
        learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
        use_snn_fusion_gate=True,
        snn_fusion_gate_mode="group_channel",
        snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
        snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
        snn_input_mode="scalar7_flow",
        snn_pool_mode="seg4x3",
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, cfg


def score(model: torch.nn.Module, dataset: Any, device: torch.device, diagnostics: bool):
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    aux_probabilities: list[np.ndarray] = []
    semantic_gates: list[np.ndarray] = []
    residual_scales: list[np.ndarray] = []
    residual_norm_ratios: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(dataset), 128):
            end = min(offset + 128, len(dataset))
            batch = torch.stack([dataset[index][0] for index in range(offset, end)]).to(device)
            logits, gate_info = model.forward_with_gate(batch)
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            if diagnostics:
                aux_probabilities.append(
                    torch.softmax(gate_info["snn_class_logits"], dim=1)[:, 1].cpu().numpy()
                )
                semantic_gates.append(gate_info["snn_semantic_gate"].cpu().numpy())
                residual_scales.append(gate_info["snn_residual_scale"].cpu().numpy())
                residual_norm_ratios.append(gate_info["snn_residual_norm_ratio"].cpu().numpy())
    result = {
        "probabilities": np.concatenate(probabilities),
        "predictions": np.concatenate(predictions).astype(np.int64),
    }
    if diagnostics:
        result.update(
            {
                "aux_probabilities": np.concatenate(aux_probabilities),
                "semantic_gates": np.concatenate(semantic_gates),
                "residual_scales": np.concatenate(residual_scales),
                "residual_norm_ratios": np.concatenate(residual_norm_ratios),
            }
        )
    return result


def evaluate_checkpoint(
    runner: Any,
    checkpoint_path: Path,
    dataset16: Any,
    labels16: np.ndarray,
    summary16: dict[str, Any],
    dataset4: Any,
    summary4: dict[str, Any],
    video_ground_truth: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    print(f"Evaluating {checkpoint_path.name}", flush=True)
    model, cfg = load_model(runner, checkpoint_path, device)
    scored16 = score(model, dataset16, device, diagnostics=True)
    true16 = labels16.astype(np.int64)
    sources16 = source_vector(summary16)
    window_results = {
        "all": metrics(true16, scored16["predictions"]),
        "subject3": metrics(true16[sources16 == "subject3"], scored16["predictions"][sources16 == "subject3"]),
        "zenodo": metrics(true16[sources16 == "zenodo"], scored16["predictions"][sources16 == "zenodo"]),
    }
    gates = scored16["semantic_gates"]
    diagnostic = {
        "snn_aux_auc": auc(scored16["aux_probabilities"], true16),
        "main_snn_correlation": float(
            np.corrcoef(scored16["probabilities"], scored16["aux_probabilities"])[0, 1]
        ),
        "dynamic_gate_mean": float(gates[:, 0].mean()),
        "state_gate_mean": float(gates[:, 1].mean()),
        "dynamic_gate_adl_mean": float(gates[true16 == 0, 0].mean()),
        "dynamic_gate_fall_mean": float(gates[true16 == 1, 0].mean()),
        "state_gate_adl_mean": float(gates[true16 == 0, 1].mean()),
        "state_gate_fall_mean": float(gates[true16 == 1, 1].mean()),
        "residual_scale": float(scored16["residual_scales"].mean()),
        "residual_to_main_norm_ratio": float(scored16["residual_norm_ratios"].mean()),
    }
    if model.snn_decay_logit is not None and model.snn_threshold_logit is not None:
        learned_decay = model.snn_decay_min + (model.snn_decay_max - model.snn_decay_min) * torch.sigmoid(
            model.snn_decay_logit.detach()
        )
        learned_threshold = model.snn_threshold_min + (
            model.snn_threshold_max - model.snn_threshold_min
        ) * torch.sigmoid(model.snn_threshold_logit.detach())
        if getattr(model, "snn_dynamic_encoder_mode", "legacy") == "multistream":
            transient_end = int(model.snn_transient_hidden)
            flow_end = transient_end + int(model.snn_flow_hidden)
            diagnostic["lif_dynamics"] = {
                "transient": {
                    "decay": float(learned_decay[:transient_end].mean()),
                    "threshold": float(learned_threshold[:transient_end].mean()),
                },
                "flow": {
                    "decay": float(learned_decay[transient_end:flow_end].mean()),
                    "threshold": float(learned_threshold[transient_end:flow_end].mean()),
                },
                "state": {
                    "decay": float(learned_decay[flow_end:].mean()),
                    "threshold": float(learned_threshold[flow_end:].mean()),
                },
            }

    scored4 = score(model, dataset4, device, diagnostics=False)
    original_residual_scale_raw = model.snn_residual_scale_raw.detach().clone()
    with torch.no_grad():
        model.snn_residual_scale_raw.fill_(-30.0)
    ablated16 = score(model, dataset16, device, diagnostics=False)
    ablated4 = score(model, dataset4, device, diagnostics=False)
    with torch.no_grad():
        model.snn_residual_scale_raw.copy_(original_residual_scale_raw)
    del model
    torch.cuda.empty_cache()
    true_videos: list[int] = []
    predicted_videos: list[int] = []
    video_sources: list[str] = []
    offset = 0
    for row in summary4["videos"]:
        count = int(row["num_windows"])
        peak = float(scored4["probabilities"][offset:offset + count].max())
        offset += count
        video_path = Path(row["video_path"])
        source = source_for_video(video_path)
        root = S3_DIR if source == "subject3" else ZEN_DIR
        relative_path = str(video_path.relative_to(root)).replace("\\", "/")
        true_videos.append(int(video_ground_truth[relative_path]["true_fall"]))
        predicted_videos.append(int(peak >= THRESHOLD))
        video_sources.append(source)
    true_video_array = np.asarray(true_videos)
    predicted_video_array = np.asarray(predicted_videos)
    video_source_array = np.asarray(video_sources)
    video_results = {
        "all": metrics(true_video_array, predicted_video_array),
        "subject3": metrics(
            true_video_array[video_source_array == "subject3"],
            predicted_video_array[video_source_array == "subject3"],
        ),
        "zenodo": metrics(
            true_video_array[video_source_array == "zenodo"],
            predicted_video_array[video_source_array == "zenodo"],
        ),
    }
    ablated_video_predictions: list[int] = []
    offset = 0
    for row in summary4["videos"]:
        count = int(row["num_windows"])
        peak = float(ablated4["probabilities"][offset:offset + count].max())
        offset += count
        ablated_video_predictions.append(int(peak >= THRESHOLD))
    ablated_video_array = np.asarray(ablated_video_predictions)
    ablated_window_metrics = metrics(true16, ablated16["predictions"])
    ablated_video_metrics = metrics(true_video_array, ablated_video_array)
    snn_ablation = {
        "window_stride16_without_snn_residual": ablated_window_metrics,
        "video_stride4_peak_090_without_snn_residual": ablated_video_metrics,
        "window_balanced_accuracy_delta_full_minus_ablation": (
            window_results["all"]["balanced_accuracy"] - ablated_window_metrics["balanced_accuracy"]
        ),
        "video_balanced_accuracy_delta_full_minus_ablation": (
            video_results["all"]["balanced_accuracy"] - ablated_video_metrics["balanced_accuracy"]
        ),
        "mean_absolute_window_probability_delta": float(
            np.abs(scored16["probabilities"] - ablated16["probabilities"]).mean()
        ),
        "window_decision_changes": int((scored16["predictions"] != ablated16["predictions"]).sum()),
        "video_decision_changes": int((predicted_video_array != ablated_video_array).sum()),
    }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_kind": "ema" if checkpoint_path.stem.endswith("_ema") else "raw",
        "binary_labels": {"0": "normal", "1": "fall"},
        "config": {
            "model_selection_mode": cfg.get("model_selection_mode"),
            "weight_ema_decay": cfg.get("weight_ema_decay"),
            "weight_ema_source_epoch": cfg.get("weight_ema_source_epoch"),
            "snn_residual_max_scale": cfg.get("snn_residual_max_scale", 0.25),
            "snn_residual_init_scale": cfg.get("snn_residual_init_scale"),
            "snn_semantic_dim_per_group": cfg.get("snn_semantic_dim_per_group", 48),
            "snn_dynamic_encoder_mode": cfg.get("snn_dynamic_encoder_mode", "legacy"),
            "snn_dynamics_mode": cfg.get("snn_dynamics_mode", "uniform"),
            "snn_semantic_gate_margin_weight": cfg.get("snn_semantic_gate_margin_weight", 0.0),
            "snn_semantic_gate_margin": cfg.get("snn_semantic_gate_margin", 0.0),
        },
        "window_stride16": window_results,
        "video_stride4_peak_090": video_results,
        "snn_diagnostics": diagnostic,
        "snn_residual_ablation": snn_ablation,
    }


def main() -> None:
    runner = load_runner()
    base = runner.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = [
        MODEL_DIR / f"automatic_fall_event_detector_{TAG}.pth",
        MODEL_DIR / f"automatic_fall_event_detector_{TAG}_ema.pth",
    ]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
    reference_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    cfg = dict(reference_checkpoint["config"])
    windows16, labels16, summary16 = base.load_external_labeled_windows(
        [S3_DIR, ZEN_DIR],
        image_size=int(cfg.get("image_size", 64)),
        window_size=int(cfg.get("window_size", 16)),
        stride=16,
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
    )
    if len(windows16) != 1076:
        raise ValueError(f"Validation protocol changed: expected 1076 windows, got {len(windows16)}")
    dataset16 = build_dataset(base, windows16, labels16, cfg, seed=10041)
    windows4, labels4, summary4 = base.load_external_labeled_windows(
        [S3_DIR, ZEN_DIR],
        image_size=int(cfg.get("image_size", 64)),
        window_size=int(cfg.get("window_size", 16)),
        stride=4,
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
    )
    dataset4 = build_dataset(base, windows4, labels4, cfg, seed=10041)
    print(
        f"Protocol confirmed: stride16={len(dataset16)} windows, "
        f"stride4={len(dataset4)} windows, threshold={THRESHOLD:.2f}",
        flush=True,
    )
    video_ground_truth = json.loads(
        (EXP_ROOT / "analysis" / "val_video_gt.json").read_text(encoding="utf-8")
    )
    results = [
        evaluate_checkpoint(
            runner,
            checkpoint_path,
            dataset16,
            labels16,
            summary16,
            dataset4,
            summary4,
            video_ground_truth,
            device,
        )
        for checkpoint_path in checkpoint_paths
    ]
    output = {
        "tag": TAG,
        "protocol": {
            "labels": {"0": "normal", "1": "fall"},
            "validation_windows": 1076,
            "window_rule": "argmax over two logits",
            "video_rule": "stride=4, peak fall probability >= 0.90",
            "threshold_scan": False,
            "subject4_used": False,
        },
        "results": results,
    }
    output_path = EXP_ROOT / "analysis" / f"{TAG}_fixed_s13.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

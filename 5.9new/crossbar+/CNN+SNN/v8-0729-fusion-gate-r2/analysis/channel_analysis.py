"""SNN channel separability + mask ablation analysis for the C1 checkpoint.

Step 1: per-channel ADL/Fall separability on the 1076-window validation set.
Step 2: post-gate channel/group mask ablation (head-only evaluation, exact).

Read-only with respect to model weights. Validation data only (Subject3 +
Zenodo val); no Subject4. Val inputs replicate the training-time pipeline
deterministically (same base_seed = 42 + 9999, augment off).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
C1_CKPT = (
    Path(sys.argv[1]) if len(sys.argv) > 1
    else EXP_ROOT / "models" / "group_bias00_grustage_aux010" / "automatic_fall_event_detector_group_bias00_grustage_aux010.pth"
)
OUT_SUFFIX = str(sys.argv[2]) if len(sys.argv) > 2 else ""
OUT_DIR = EXP_ROOT / "analysis"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT3_DIR = GMDCSA_ROOT / "Subject 3"
ZENODO_VAL_DIR = ROOT / "zenodo_falldb_video_split" / "val"


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2"] = r2
spec.loader.exec_module(r2)
base = r2.base

r2.REFINED_MODE = "group"
r2.REFINED_FLOOR = 0.2
r2.REFINED_GROUP_BIAS = 0.0
r2.REFINED_CHANNEL_BIAS = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(C1_CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", False))
model = r2.RefinedFallEventDetectorR2(
    hidden_dim=int(cfg.get("hidden_dim", 64)),
    num_classes=len(base.LABEL_ID_TO_NAME),
    use_snn_temporal_branch=True,
    snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
    snn_decay=float(cfg.get("snn_decay", 0.82)),
    snn_threshold=float(cfg.get("snn_threshold", 0.55)),
    snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
    learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
    use_snn_fusion_gate=True,
    snn_fusion_gate_mode="group",
    snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
    snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
    snn_input_mode=str(cfg.get("snn_input_mode", "scalar7")),
    snn_pool_mode=str(cfg.get("snn_pool_mode", "global3")),
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
log(f"loaded C1 checkpoint, gate mode={model.snn_fusion_gate_mode}")

hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

windows, labels, summary = base.load_external_labeled_windows(
    [SUBJECT3_DIR, ZENODO_VAL_DIR],
    image_size=64,
    window_size=16,
    stride=16,
    max_videos=0,
    fall_name_pattern="fall",
    use_pixel_human=False,
    use_silhouette_human=False,
    pixel_grid_size=20,
)
assert len(windows) == 1076, f"protocol violation: {len(windows)}"
log(f"external_val_labeled_windows={len(windows)}")

val_dataset = base.CombinedFallDataset(
    video_windows=np.asarray(windows, dtype=np.float32),
    video_labels=labels,
    auxiliary_records=[],
    window_size=16,
    image_size=64,
    use_crossbar=True,
    use_pixel_human=False,
    use_silhouette_human=False,
    pixel_grid_size=20,
    hrs_values=hrs_values,
    lrs_values=lrs_values,
    device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25,
    augment=False,
    base_seed=42 + 9999,
)

true_binary = (labels == base.LABEL_NAME_TO_ID["fall"]).astype(np.int64)
hard_scores = np.asarray([float(s["hard_negative_score"]) for s in val_dataset.samples], dtype=np.float32)

# -------- cached forward pass --------
pooled_list, snn_list, gate_list, fes_list = [], [], [], []
logits_list = []
BATCH = 32
with torch.no_grad():
    for offset in range(0, len(val_dataset), BATCH):
        batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
        batch_x = batch_x.to(device)
        bs, ts, ch, hh, ww = batch_x.shape
        encoded = model.encoder(batch_x.view(bs * ts, ch, hh, ww)).view(bs * ts, -1)
        encoded = model.proj(encoded).view(bs, ts, -1)
        temporal_out, _ = model.temporal(encoded)
        pooled = torch.cat(
            [temporal_out.mean(dim=1), temporal_out.max(dim=1).values, temporal_out[:, -1, :]], dim=1
        )
        features = model.extract_temporal_event_features(
            batch_x, mode=str(cfg.get("snn_input_mode", "scalar7"))
        )
        snn_pooled = model.run_lif_temporal_branch(batch_x, features)
        fes = model.extract_fall_event_score(features)
        gated, gate = model.apply_snn_fusion_gate(pooled, snn_pooled, fes)
        logits = model.head(torch.cat([pooled, gated], dim=1))
        pooled_list.append(pooled.cpu())
        snn_list.append(snn_pooled.cpu())
        gate_list.append(gate.cpu())
        fes_list.append(fes.cpu())
        logits_list.append(logits.cpu())

pooled = torch.cat(pooled_list)
snn_pooled = torch.cat(snn_list).numpy()
gate = torch.cat(gate_list).numpy()
fes = torch.cat(fes_list).numpy().reshape(-1)
base_logits = torch.cat(logits_list)
gated_base = (torch.cat(snn_list) * torch.cat(gate_list))
log(f"forward pass cached: pooled={tuple(pooled.shape)} snn={snn_pooled.shape}")

GROUP_NAMES = ["mean", "max", "last"]
GROUP_DIM = snn_pooled.shape[1] // 3


def metrics_from_logits(logits: torch.Tensor) -> dict:
    pred = (torch.argmax(logits, dim=1).numpy() == base.LABEL_NAME_TO_ID["fall"]).astype(np.int64)
    tp = int(np.sum((true_binary == 1) & (pred == 1)))
    tn = int(np.sum((true_binary == 0) & (pred == 0)))
    fp = int(np.sum((true_binary == 0) & (pred == 1)))
    fn = int(np.sum((true_binary == 1) & (pred == 0)))
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": (tp + tn) / len(true_binary),
        "balanced_accuracy": 0.5 * (spec + rec),
        "adl_specificity": spec,
        "fall_recall": rec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


baseline = metrics_from_logits(base_logits)
log(f"baseline: bal={baseline['balanced_accuracy']:.4f} spec={baseline['adl_specificity']:.4f} rec={baseline['fall_recall']:.4f}")

# -------- Step 1: channel separability --------
OUT_DIR.mkdir(parents=True, exist_ok=True)
fall_mask = true_binary == 1
rows = []
for k in range(snn_pooled.shape[1]):
    values = snn_pooled[:, k]
    vf, va = values[fall_mask], values[~fall_mask]
    # AUC via rank statistic
    order = np.argsort(np.concatenate([vf, va]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[: len(vf)].sum() - len(vf) * (len(vf) + 1) / 2) / (len(vf) * len(va))
    dprime = (vf.mean() - va.mean()) / np.sqrt(0.5 * (vf.var() + va.var()) + 1e-12)
    rows.append(
        {
            "channel": k,
            "group": GROUP_NAMES[k // GROUP_DIM],
            "channel_in_group": k % GROUP_DIM,
            "auc": round(float(auc), 4),
            "d_prime": round(float(dprime), 4),
            "mean_fall": round(float(vf.mean()), 4),
            "mean_adl": round(float(va.mean()), 4),
        }
    )
with (OUT_DIR / (f"channel_separability{OUT_SUFFIX}.csv")).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
log(f"saved channel_separability.csv rows={len(rows)}")

# gate behavior by class and by hard-negative level
group_gate = gate.reshape(len(gate), 3, GROUP_DIM)[:, :, 0]
gate_rows = []
for g, name in enumerate(GROUP_NAMES):
    values = group_gate[:, g]
    gate_rows.append({"group": name, "mean_fall": round(float(values[fall_mask].mean()), 4),
                      "mean_adl": round(float(values[~fall_mask].mean()), 4),
                      "mean_adl_hardtop10": round(float(values[~fall_mask][np.argsort(hard_scores[~fall_mask])[-len(values[~fall_mask]) // 10:]].mean()), 4)})
with (OUT_DIR / (f"group_gate_behavior{OUT_SUFFIX}.csv")).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0].keys()))
    writer.writeheader()
    writer.writerows(gate_rows)
log(f"saved group_gate_behavior.csv: {gate_rows}")

# -------- Step 2: mask ablation (head-only) --------
def logits_with_mask(mask: np.ndarray) -> torch.Tensor:
    masked = gated_base * torch.from_numpy(mask.astype(np.float32))
    with torch.no_grad():
        return model.head(torch.cat([pooled.to(device), masked.to(device)], dim=1)).cpu()


ablation_rows = []
base_line = baseline
for target in ["none"] + [f"group_{n}" for n in GROUP_NAMES] + [f"ch_{k}" for k in range(snn_pooled.shape[1])]:
    mask = np.ones(snn_pooled.shape[1], dtype=np.float32)
    if target.startswith("group_"):
        g = GROUP_NAMES.index(target.split("_")[1])
        mask[g * GROUP_DIM : (g + 1) * GROUP_DIM] = 0.0
    elif target.startswith("ch_"):
        mask[int(target.split("_")[1])] = 0.0
    if target == "none":
        metrics = base_line
    else:
        metrics = metrics_from_logits(logits_with_mask(mask))
    ablation_rows.append(
        {
            "mask_target": target,
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "adl_specificity": round(metrics["adl_specificity"], 4),
            "fall_recall": round(metrics["fall_recall"], 4),
            "delta_bal": round(metrics["balanced_accuracy"] - base_line["balanced_accuracy"], 4),
            "delta_spec": round(metrics["adl_specificity"] - base_line["adl_specificity"], 4),
            "delta_recall": round(metrics["fall_recall"] - base_line["fall_recall"], 4),
        }
    )
with (OUT_DIR / (f"channel_mask_ablation{OUT_SUFFIX}.csv")).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ablation_rows)
log(f"saved channel_mask_ablation.csv rows={len(ablation_rows)}")

# quick conclusion printout
suppress_candidates = [r for r in ablation_rows if r["mask_target"].startswith("ch_") and r["delta_spec"] >= 0.01 and r["delta_recall"] >= -0.01]
keep_channels = [r for r in ablation_rows if r["mask_target"].startswith("ch_") and r["delta_recall"] <= -0.02]
log(f"suppress candidates (mask improves spec without hurting recall): {[r['mask_target'] for r in suppress_candidates]}")
log(f"fall-critical channels (mask hurts recall): {[r['mask_target'] for r in keep_channels]}")
log("analysis complete")

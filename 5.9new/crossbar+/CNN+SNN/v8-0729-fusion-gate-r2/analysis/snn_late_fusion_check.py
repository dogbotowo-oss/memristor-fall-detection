"""Post-hoc check: does the SNN class head carry usable signal for late fusion?

Uses the S3 EMA checkpoint (snn_class_head was trained with aux weight 0.3).
No training, no threshold scan: lambda grid fixed a priori at {0.1, 0.2, 0.3}.
Final logits = main logits + lambda * snn_class_head logits (binary head
expanded to 4-class logit space on {normal, fall} entries only).

Protocol: 1076-window validation set only, deterministic, read-only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
CKPT = (
    Path(sys.argv[1]) if len(sys.argv) > 1
    else EXP_ROOT / "models" / "c1_ema0999_s3" / "automatic_fall_event_detector_c1_ema0999_s3_ema.pth"
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", False))
model = r2.RefinedFallEventDetectorR2(
    hidden_dim=int(cfg.get("hidden_dim", 64)),
    num_classes=len(base.LABEL_NAME_TO_ID),
    use_snn_temporal_branch=True,
    snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
    snn_decay=float(cfg.get("snn_decay", 0.82)),
    snn_threshold=float(cfg.get("snn_threshold", 0.30)),
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
log("loaded S3 EMA checkpoint")

hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

windows, labels, _ = base.load_external_labeled_windows(
    [GMDCSA_ROOT / "Subject 3", ROOT / "zenodo_falldb_video_split" / "val"],
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
normal_id = base.LABEL_NAME_TO_ID["normal"]
fall_id = base.LABEL_NAME_TO_ID["fall"]

main_logits_list, snn_logits_list = [], []
BATCH = 32
with torch.no_grad():
    for offset in range(0, len(val_dataset), BATCH):
        batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
        batch_x = batch_x.to(device)
        logits, gate_info = model.forward_with_gate(batch_x)
        snn_aux = model.snn_class_head(gate_info["snn_pooled_pregate"])
        main_logits_list.append(logits.cpu())
        snn_logits_list.append(snn_aux.cpu())

main_logits = torch.cat(main_logits_list).numpy()
snn_logits = torch.cat(snn_logits_list).numpy()


def metrics(logits: np.ndarray) -> tuple[float, float, float]:
    pred = (np.argmax(logits, axis=1) == fall_id).astype(np.int64)
    tp = int(np.sum((true_binary == 1) & (pred == 1)))
    tn = int(np.sum((true_binary == 0) & (pred == 0)))
    fp = int(np.sum((true_binary == 0) & (pred == 1)))
    fn = int(np.sum((true_binary == 1) & (pred == 0)))
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 0.5 * (spec + rec), spec, rec


# snn head standalone: fall margin as a score
snn_margin = snn_logits[:, 1] - snn_logits[:, 0]
main_margin = main_logits[:, fall_id] - main_logits[:, normal_id]


def auc(scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos = true_binary == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


bal, spec_v, rec_v = metrics(main_logits)
log(f"main head alone      : bal={bal:.4f} spec={spec_v:.4f} rec={rec_v:.4f} marginAUC={auc(main_margin):.3f}")

snn_pred = (np.argmax(snn_logits, axis=1) == 1).astype(np.int64)
tp = int(np.sum((true_binary == 1) & (snn_pred == 1)))
tn = int(np.sum((true_binary == 0) & (snn_pred == 0)))
fp = int(np.sum((true_binary == 0) & (snn_pred == 1)))
fn = int(np.sum((true_binary == 1) & (snn_pred == 0)))
snn_spec = tn / max(tn + fp, 1)
snn_rec = tp / max(tp + fn, 1)
log(f"snn head alone       : bal={0.5 * (snn_spec + snn_rec):.4f} spec={snn_spec:.4f} rec={snn_rec:.4f} marginAUC={auc(snn_margin):.3f}")
log(f"corr(main_margin, snn_margin) = {float(np.corrcoef(main_margin, snn_margin)[0, 1]):.3f}")

for lam in (0.1, 0.2, 0.3):
    fused = main_logits.copy()
    fused[:, normal_id] += lam * snn_logits[:, 0]
    fused[:, fall_id] += lam * snn_logits[:, 1]
    bal, spec_v, rec_v = metrics(fused)
    fused_margin = fused[:, fall_id] - fused[:, normal_id]
    log(f"late fusion lam={lam:.1f} : bal={bal:.4f} spec={spec_v:.4f} rec={rec_v:.4f} marginAUC={auc(fused_margin):.3f}")

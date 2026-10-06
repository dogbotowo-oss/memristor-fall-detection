"""End-to-end verification of the v89 release model.

v89 = v88 config + temporal slow-down augmentation (slow_flip_half, aug
weight ~0.5). Loads v89\models\automatic_fall_event_detector_v89.pth through
the v89 runner (built-in late fusion lambda=0.2) and evaluates on the
protocol validation set (Subject3 + Zenodo val, 1076 windows, deterministic).
Expected result (argmax rule, same as v88): acc=0.7491 bal=0.7519
spec=0.7086 rec=0.7952, TP/TN/FP/FN=400/406/167/103.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

V89_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v89")
RUNNER_PATH = V89_ROOT / "code" / "run_refined_gate_r2.py"
CKPT = V89_ROOT / "models" / "automatic_fall_event_detector_v89.pth"

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
spec = importlib.util.spec_from_file_location("r2v89", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2v89"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
r2.REFINED_MODE = str(cfg.get("snn_fusion_gate_mode", "group"))
r2.REFINED_FLOOR = float(cfg.get("snn_fusion_gate_floor", 0.2))
r2.REFINED_GROUP_BIAS = float(cfg.get("refined_group_bias", 0.0))
r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.2))
log(f"config: mode={r2.REFINED_MODE} bias={r2.REFINED_GROUP_BIAS} "
    f"drop_fes={r2.REFINED_DROP_FES_CONTEXT} lambda={r2.REFINED_LATE_FUSION_LAMBDA} "
    f"ema_epoch={cfg.get('weight_ema_source_epoch')}")

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
fall_id = base.LABEL_NAME_TO_ID["fall"]

logits_list = []
BATCH = 32
with torch.no_grad():
    for offset in range(0, len(val_dataset), BATCH):
        batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
        logits_list.append(model(batch_x.to(device)).cpu())

logits = torch.cat(logits_list).numpy()
pred = (np.argmax(logits, axis=1) == fall_id).astype(np.int64)
tp = int(np.sum((true_binary == 1) & (pred == 1)))
tn = int(np.sum((true_binary == 0) & (pred == 0)))
fp = int(np.sum((true_binary == 0) & (pred == 1)))
fn = int(np.sum((true_binary == 1) & (pred == 0)))
spec_v = tn / max(tn + fp, 1)
rec_v = tp / max(tp + fn, 1)
acc_v = (tp + tn) / max(tp + tn + fp + fn, 1)
log(f"v89 built-in fusion  : acc={acc_v:.4f} bal={0.5 * (spec_v + rec_v):.4f} spec={spec_v:.4f} rec={rec_v:.4f}")
log(f"confusion TP/TN/FP/FN = {tp}/{tn}/{fp}/{fn}")
log("expected             : acc=0.7491 bal=0.7519 spec=0.7086 rec=0.7952 TP/TN/FP/FN=400/406/167/103")

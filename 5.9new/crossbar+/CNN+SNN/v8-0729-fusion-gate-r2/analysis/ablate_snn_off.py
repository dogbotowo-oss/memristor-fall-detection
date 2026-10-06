"""SNN-off ablation: does the SNN branch help recall (fall) or spec (ADL)?

Loads a checkpoint, scores val windows twice:
  1. normal forward
  2. SNN branch zeroed (run_lif_temporal_branch returns zeros)
Then reports window-level five metrics for both, so the contribution of the
SNN branch decomposes into delta_spec (ADL side) vs delta_recall (fall side).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"

TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_s25_relu"
VAL_SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "Subject 3"

CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}.pth"
if not CKPT.exists():
    CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}_ema.pth"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(d for d in ROOT.iterdir() if d.is_dir() and (d / "13354453").exists())
GMDCSA_ROOT = next(
    p for p in (TESTSET_DIR / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
VAL_DIRS = [GMDCSA_ROOT / VAL_SUBJECT, ROOT / "zenodo_falldb_video_split" / "val"]
OUT_PATH = EXP_ROOT / "analysis" / f"snn_off_ablation_{TAG}.json"


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2_abl", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_abl"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.0))
r2.REFINED_ACTION_NUM_CLASSES = int(cfg.get("refined_action_num_classes", 0))
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
    snn_event_encoding=bool(cfg.get("snn_event_encoding", False)),
    snn_event_enc_threshold=float(cfg.get("snn_event_enc_threshold", 0.05)),
    snn_event_enc_sharpness=float(cfg.get("snn_event_enc_sharpness", 10.0)),
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
log(f"loaded {TAG}")

windows, labels, summary_w = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
fall_id = base.LABEL_NAME_TO_ID["fall"]
y = (labels == fall_id).astype(np.int64)
log(f"val windows={len(windows)} fall={int(y.sum())} adl={int((y == 0).sum())}")

dataset = base.CombinedFallDataset(
    video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
    auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=42 + 9999,
    pose_windows=base.load_pose_windows_for_summary(summary_w, 16, 16),
)

orig_lif = model.run_lif_temporal_branch


def zero_lif(x, features=None):
    out = orig_lif(x, features)
    return torch.zeros_like(out)


def score(snn_off: bool) -> np.ndarray:
    if snn_off:
        model.run_lif_temporal_branch = zero_lif
    probs = []
    bs = 64
    with torch.no_grad():
        for off in range(0, len(dataset), bs):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + bs, len(dataset)))]).to(device)
            ps = torch.stack([dataset[i][5] for i in range(off, min(off + bs, len(dataset)))]).to(device)
            logits, _ = model.forward_with_gate(xs, pose=ps)
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu().numpy())
    if snn_off:
        model.run_lif_temporal_branch = orig_lif
    return np.concatenate(probs)


def metrics(probs: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = (probs >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "acc": round((tp + tn) / len(y), 4),
        "spec": round(tn / max(tn + fp, 1), 4),
        "rec": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "mAP": round(float(average_precision_score(y, probs)), 4),
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "fall_med": round(float(np.median(probs[y == 1])), 4),
        "adl_med": round(float(np.median(probs[y == 0])), 4),
    }


p_on = score(snn_off=False)
p_off = score(snn_off=True)
m_on, m_off = metrics(p_on), metrics(p_off)
delta = {k: round(m_off[k] - m_on[k], 4) for k in ["acc", "spec", "rec", "f1", "mAP", "auc"]}
log(f"SNN on : {m_on}")
log(f"SNN off: {m_off}")
log(f"delta(off-on): {delta}")
OUT_PATH.write_text(json.dumps({"tag": TAG, "snn_on": m_on, "snn_off": m_off,
                                "delta_off_minus_on": delta}, indent=2))
log(f"saved {OUT_PATH}")

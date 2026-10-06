"""Membrane potential trajectory analysis: ADL vs fall patterns.

Loads S22 checkpoint, scores val windows, captures membrane potential
trajectories from the LIF branch, and analyzes:
  1. Mean membrane trajectory per class (ADL vs fall)
  2. Which ADL windows have trajectories most similar to fall
  3. Per-timestep membrane statistics (peak, mean, final, rise speed)
  4. Channel-level: which LIF channels discriminate best/worst
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
OUT_DIR = EXP_ROOT / "analysis"
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_s22_lif"
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


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2_memb", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_memb"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

# --- load model ---
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
log(f"loaded {TAG}: event_enc={cfg.get('snn_event_encoding', False)} "
    f"input={cfg.get('snn_input_mode')} pool={cfg.get('snn_pool_mode')}")

# --- load val data (stride 16 for speed) ---
windows, labels, summary = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
fall_id = base.LABEL_NAME_TO_ID["fall"]
true_labels = (labels == fall_id).astype(np.int64)
log(f"val windows={len(windows)} fall={int(true_labels.sum())} adl={int((true_labels == 0).sum())}")

dataset = base.CombinedFallDataset(
    video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
    auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=42 + 9999,
)

# --- collect membrane trajectories ---
all_membranes = []  # list of (T, C) per window
all_fall_probs = []
batch_size = 32
with torch.no_grad():
    for off in range(0, len(dataset), batch_size):
        xs = torch.stack([dataset[i][0] for i in range(off, min(off + batch_size, len(dataset)))])
        logits, gate_info = model.forward_with_gate(xs.to(device))
        probs = torch.softmax(logits, dim=1)[:, fall_id].cpu().numpy()
        all_fall_probs.append(probs)
        memb = model._last_membrane_train  # (B, T, C)
        if memb is not None:
            all_membranes.append(memb.cpu().numpy())
all_fall_probs = np.concatenate(all_fall_probs)
all_membranes = np.concatenate(all_membranes, axis=0)  # (N, T, C)
log(f"membrane shape: {all_membranes.shape} (N windows, T timesteps, C channels)")

del model
torch.cuda.empty_cache()

# --- analysis ---
adl_mask = true_labels == 0
fall_mask = true_labels == 1

# 1. Mean trajectory per class
adl_mean_traj = all_membranes[adl_mask].mean(axis=0)  # (T, C)
fall_mean_traj = all_membranes[fall_mask].mean(axis=0)

# 2. Per-timestep class difference (mean across channels)
adl_ts = adl_mean_traj.mean(axis=1)  # (T,)
fall_ts = fall_mean_traj.mean(axis=1)
log("\n=== Per-timestep membrane potential (mean across channels) ===")
log(f"{'t':>3s} {'ADL':>8s} {'Fall':>8s} {'diff':>8s}")
for t in range(all_membranes.shape[1]):
    log(f"{t:3d} {adl_ts[t]:8.4f} {fall_ts[t]:8.4f} {fall_ts[t] - adl_ts[t]:+8.4f}")

# 3. Channel discriminability: for each channel, compute separation
#    between ADL and fall membrane distributions
log("\n=== Channel discriminability (top 10 best / bottom 5 worst) ===")
T = all_membranes.shape[1]
C = all_membranes.shape[2]
channel_sep = []
for c in range(C):
    # use peak membrane per window for this channel
    adl_peak = all_membranes[adl_mask, :, c].max(axis=1)
    fall_peak = all_membranes[fall_mask, :, c].max(axis=1)
    sep = abs(fall_peak.mean() - adl_peak.mean()) / (adl_peak.std() + fall_peak.std() + 1e-8)
    channel_sep.append((c, sep, adl_peak.mean(), fall_peak.mean()))
channel_sep.sort(key=lambda x: x[1], reverse=True)
log(f"{'ch':>4s} {'sep':>8s} {'adl_pk':>8s} {'fall_pk':>8s}")
for c, sep, ap, fp in channel_sep[:10]:
    log(f"{c:4d} {sep:8.4f} {ap:8.4f} {fp:8.4f}")
log("  ... worst 5:")
for c, sep, ap, fp in channel_sep[-5:]:
    log(f"{c:4d} {sep:8.4f} {ap:8.4f} {fp:8.4f}")

# 4. Confusable ADL: find ADL windows whose membrane trajectories
#    are most similar to the fall mean trajectory
log("\n=== Most fall-like ADL windows (top 15) ===")
fall_template = fall_mean_traj  # (T, C)
adl_membranes = all_membranes[adl_mask]  # (N_adl, T, C)
adl_indices = np.where(adl_mask)[0]
adl_probs = all_fall_probs[adl_mask]
# cosine similarity to fall template
adl_flat = adl_membranes.reshape(len(adl_membranes), -1)
fall_flat = fall_template.reshape(-1)
norms = np.linalg.norm(adl_flat, axis=1) * np.linalg.norm(fall_flat) + 1e-8
similarities = (adl_flat @ fall_flat) / norms
top_confusable = np.argsort(-similarities)[:15]
log(f"{'idx':>6s} {'sim':>8s} {'prob':>8s} {'peak_m':>8s} {'final_m':>8s}")
for rank, idx in enumerate(top_confusable):
    w_idx = adl_indices[idx]
    peak_m = float(adl_membranes[idx].max())
    final_m = float(adl_membranes[idx, -1].mean())
    log(f"{w_idx:6d} {similarities[idx]:8.4f} {adl_probs[idx]:8.4f} {peak_m:8.4f} {final_m:8.4f}")

# 5. Trajectory shape features: rise speed, peak time, decay ratio
log("\n=== Trajectory shape comparison ===")
for name, mask in [("ADL", adl_mask), ("Fall", fall_mask)]:
    memb = all_membranes[mask]  # (N, T, C)
    mean_traj = memb.mean(axis=(0, 2))  # (T,) avg across windows and channels
    peak_t = int(np.argmax(mean_traj))
    peak_v = float(mean_traj[peak_t])
    final_v = float(mean_traj[-1])
    rise_speed = peak_v / max(peak_t + 1, 1)
    decay_ratio = final_v / max(peak_v, 1e-8)
    log(f"{name:5s}: peak_t={peak_t:2d} peak_v={peak_v:.4f} final={final_v:.4f} "
        f"rise={rise_speed:.4f} decay_ratio={decay_ratio:.4f}")

# 6. Summary statistics
log("\n=== Summary ===")
adl_peak_all = all_membranes[adl_mask].max(axis=(1, 2))
fall_peak_all = all_membranes[fall_mask].max(axis=(1, 2))
log(f"ADL peak membrane: mean={adl_peak_all.mean():.4f} med={np.median(adl_peak_all):.4f} "
    f"p90={np.percentile(adl_peak_all, 90):.4f}")
log(f"Fall peak membrane: mean={fall_peak_all.mean():.4f} med={np.median(fall_peak_all):.4f} "
    f"p10={np.percentile(fall_peak_all, 10):.4f}")
log(f"Overlap: ADL p90={np.percentile(adl_peak_all, 90):.4f} vs Fall p10={np.percentile(fall_peak_all, 10):.4f} "
    f"-> {'OVERLAP' if np.percentile(adl_peak_all, 90) > np.percentile(fall_peak_all, 10) else 'SEPARATED'}")

# save
result = {
    "tag": TAG, "val_subject": VAL_SUBJECT,
    "n_windows": int(len(true_labels)),
    "n_fall": int(fall_mask.sum()), "n_adl": int(adl_mask.sum()),
    "membrane_shape": list(all_membranes.shape),
    "adl_peak_mean": float(adl_peak_all.mean()),
    "fall_peak_mean": float(fall_peak_all.mean()),
    "channel_top5": [{"ch": int(c), "sep": float(s)} for c, s, _, _ in channel_sep[:5]],
}
(OUT_DIR / f"membrane_analysis_{TAG}.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
log(f"\nsaved: {OUT_DIR / f'membrane_analysis_{TAG}.json'}")

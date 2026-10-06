"""S23: video-level aggregation comparison on S22 stride-4 window scores.

No retraining. Reuses S22 checkpoint, scores stride-4 windows once, then
compares aggregation strategies:
  1. max            (legacy, S8-S21)
  2. top-3 mean     (S22)
  3. fraction above window-thr (S23 candidate)
  4. mean of all windows
  5. median of all windows

For fraction-based: sweep window_thr x ratio_thr grid.
Report five metrics (acc/spec/rec/f1/bal) for each strategy at its
Youden-J constrained operating point.
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

EMA_CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}_ema.pth"
RAW_CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}.pth"
CKPT = EMA_CKPT if EMA_CKPT.exists() else RAW_CKPT

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
spec = importlib.util.spec_from_file_location("r2_s23", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_s23"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log(f"loaded {TAG}: {ckpt_path.name}")
    return model


def build_dataset(windows, labels):
    return base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32),
        video_labels=labels,
        auxiliary_records=[],
        window_size=16, image_size=64, use_crossbar=True,
        use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
        hrs_values=hrs_values, lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=0.25, augment=False, base_seed=42 + 9999,
    )


def score_windows(model, dataset):
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    probs = []
    with torch.no_grad():
        for off in range(0, len(dataset), 32):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + 32, len(dataset)))])
            logits, _ = model.forward_with_gate(xs.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    return torch.cat(probs).numpy()


def five_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    bal = (spec + rec) / 2
    return {"acc": acc, "bal": bal, "spec": spec, "rec": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def youden_j(scores, y_true, lo, hi, step=0.005):
    best_j, best_thr = -1.0, lo
    for thr in np.arange(lo, hi + step / 2, step):
        pred = (scores >= thr).astype(int)
        m = five_metrics(y_true, pred)
        j = m["spec"] + m["rec"] - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr, best_j


# --- score stride-4 windows ---
model = load_model(CKPT)
windows4, labels4, summary4 = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=4, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
ds4 = build_dataset(windows4, labels4)
p4 = score_windows(model, ds4)
del model
torch.cuda.empty_cache()

# --- build per-video window arrays ---
fall_id = base.LABEL_NAME_TO_ID["fall"]
vid_windows = []  # list of (np.ndarray, int_label, str_name)
offset = 0
for item in summary4["videos"]:
    count = int(item["num_windows"])
    vid_probs = p4[offset:offset + count]
    is_fall = int(base.infer_video_level_label_id(Path(item["video_path"]), "fall") == fall_id)
    vid_windows.append((vid_probs, is_fall, Path(item["video_path"]).name))
    offset += count
y_true = np.array([v[1] for v in vid_windows])

log(f"\n{len(vid_windows)} videos: fall={int(y_true.sum())} adl={int((y_true == 0).sum())}")

# --- strategy 1: max ---
scores_max = np.array([v[0].max() for v in vid_windows])
thr, j = youden_j(scores_max, y_true, 0.05, 0.95)
m = five_metrics(y_true, (scores_max >= thr).astype(int))
log(f"\n[1] max            thr={thr:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
    f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

# --- strategy 2: top-3 mean ---
scores_top3 = np.array([
    float(np.mean(np.sort(v[0])[-3:])) if len(v[0]) >= 3 else float(np.mean(v[0]))
    for v in vid_windows
])
thr, j = youden_j(scores_top3, y_true, 0.05, 0.95)
m = five_metrics(y_true, (scores_top3 >= thr).astype(int))
log(f"[2] top-3 mean     thr={thr:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
    f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

# --- strategy 3: fraction above window threshold ---
log("\n[3] fraction-above-threshold grid:")
log(f"    {'win_thr':>8s} {'ratio':>6s} {'acc':>7s} {'bal':>7s} {'spec':>7s} {'rec':>7s} {'f1':>7s}  TP/TN/FP/FN")
best_frac = None
for win_thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fracs = np.array([
        float((v[0] >= win_thr).sum()) / max(len(v[0]), 1)
        for v in vid_windows
    ])
    thr_r, j_r = youden_j(fracs, y_true, 0.01, 0.99, step=0.01)
    m_r = five_metrics(y_true, (fracs >= thr_r).astype(int))
    log(f"    {win_thr:8.2f} {thr_r:6.2f} {m_r['acc']:7.4f} {m_r['bal']:7.4f} "
        f"{m_r['spec']:7.4f} {m_r['rec']:7.4f} {m_r['f1']:7.4f}  {m_r['tp']}/{m_r['tn']}/{m_r['fp']}/{m_r['fn']}")
    if best_frac is None or m_r["bal"] > best_frac[1]["bal"]:
        best_frac = (win_thr, m_r, thr_r)

# --- strategy 4: mean ---
scores_mean = np.array([float(np.mean(v[0])) for v in vid_windows])
thr, j = youden_j(scores_mean, y_true, 0.01, 0.95)
m = five_metrics(y_true, (scores_mean >= thr).astype(int))
log(f"\n[4] mean           thr={thr:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
    f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

# --- strategy 5: median ---
scores_med = np.array([float(np.median(v[0])) for v in vid_windows])
thr, j = youden_j(scores_med, y_true, 0.01, 0.95)
m = five_metrics(y_true, (scores_med >= thr).astype(int))
log(f"[5] median         thr={thr:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
    f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

# --- summary ---
log(f"\n=== Best fraction: win_thr={best_frac[0]:.2f} ratio_thr={best_frac[2]:.2f} "
    f"bal={best_frac[1]['bal']:.4f} spec={best_frac[1]['spec']:.4f} rec={best_frac[1]['rec']:.4f} ===")

# save
result = {"tag": TAG, "val_subject": VAL_SUBJECT,
          "best_fraction": {"win_thr": best_frac[0], "ratio_thr": best_frac[2],
                            **best_frac[1]}}
(OUT_DIR / f"s23_agg_{TAG}.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
log(f"saved: {OUT_DIR / f's23_agg_{TAG}.json'}")

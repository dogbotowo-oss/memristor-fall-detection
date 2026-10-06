"""S22 P1 evaluator: five metrics with constrained threshold, no ROC-AUC.

Metrics: sensitivity, specificity, accuracy, F1, mAP (PR-AUC, recorded only).
Video aggregation: top-3 mean (not max).
Threshold: Youden-J constrained to [0.40, 0.75]; unconstrained reported as diagnostic.
Usage: eval_s22.py <tag> [val_subject]   (val_subject: "Subject 3" or "Subject 2")
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
spec = importlib.util.spec_from_file_location("r2_s22", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_s22"] = r2
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
        use_snn_temporal_branch=bool(cfg.get("use_snn_temporal_branch", True)),
        snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
        snn_decay=float(cfg.get("snn_decay", 0.82)),
        snn_threshold=float(cfg.get("snn_threshold", 0.30)),
        snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
        learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
        use_snn_fusion_gate=bool(cfg.get("use_snn_fusion_gate", True)),
        snn_fusion_gate_mode="group",
        snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
        snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
        snn_input_mode=str(cfg.get("snn_input_mode", "scalar7")),
        snn_pool_mode=str(cfg.get("snn_pool_mode", "global3")),
        snn_silent=bool(cfg.get("snn_silent", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log(f"loaded {TAG}: {ckpt_path.name} action_classes={r2.REFINED_ACTION_NUM_CLASSES} "
        f"input={cfg.get('snn_input_mode')} pool={cfg.get('snn_pool_mode')}")
    return model, cfg


def build_dataset(windows, labels, summary=None, stride=16):
    pose = None
    if summary is not None and str(cfg.get("snn_input_mode", "")) == "posekin":
        pose = base.load_pose_windows_for_summary(summary, 16, stride)
    return base.CombinedFallDataset(
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
        pose_windows=pose,
    )


def score_windows(model, dataset):
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    probs = []
    batch = 32
    with torch.no_grad():
        for off in range(0, len(dataset), batch):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + batch, len(dataset)))])
            ps = torch.stack([dataset[i][5] for i in range(off, min(off + batch, len(dataset)))])
            logits, _ = model.forward_with_gate(xs.to(device), pose=ps.to(device))
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
    return {"acc": acc, "spec": spec, "rec": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def average_precision(y_true, scores):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1).astype(np.float64)
    precision = tp_cum / np.arange(1, len(y_sorted) + 1)
    recall = tp_cum / max(float((y_true == 1).sum()), 1.0)
    return float(np.trapz(precision, recall))


def youden_j(scores, y_true, lo, hi):
    best_j, best_thr = -1.0, lo
    for thr in np.arange(lo, hi + 0.001, 0.005):
        pred = (scores >= thr).astype(int)
        m = five_metrics(y_true, pred)
        j = m["spec"] + m["rec"] - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr, best_j


model, cfg = load_model(CKPT)

# --- window-level (stride 16) ---
windows16, labels16, summary16 = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
fall_id = base.LABEL_NAME_TO_ID["fall"]
true16 = (labels16 == fall_id).astype(np.int64)
ds16 = build_dataset(windows16, labels16, summary16, 16)
p16 = score_windows(model, ds16)
log(f"\n=== Window-level (stride16, {len(windows16)} windows, {VAL_SUBJECT}) ===")
log(f"fall={int(true16.sum())} adl={int((true16 == 0).sum())}")
m16 = five_metrics(true16, (p16 >= 0.5).astype(int))
log(f"thr=0.50 | acc={m16['acc']:.4f} spec={m16['spec']:.4f} rec={m16['rec']:.4f} "
    f"f1={m16['f1']:.4f} | {m16['tp']}/{m16['tn']}/{m16['fp']}/{m16['fn']}")
ap16 = average_precision(true16, p16)
log(f"mAP={ap16:.4f} (recorded)")
fall_med = float(np.median(p16[true16 == 1]))
adl_med = float(np.median(p16[true16 == 0]))
log(f"gap: fall_med={fall_med:.4f} adl_med={adl_med:.4f} gap={fall_med - adl_med:.4f}")

# --- video-level (stride 4, top-3 mean) ---
windows4, labels4, summary4 = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=4, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
ds4 = build_dataset(windows4, labels4, summary4, 4)
p4 = score_windows(model, ds4)
del model
torch.cuda.empty_cache()

video_scores, video_labels = [], []
offset = 0
for item in summary4["videos"]:
    count = int(item["num_windows"])
    vid_probs = p4[offset:offset + count]
    is_fall = int(base.infer_video_level_label_id(Path(item["video_path"]), "fall") == fall_id)
    if count >= 3:
        vid_score = float(np.mean(np.sort(vid_probs)[-3:]))
    else:
        vid_score = float(np.mean(vid_probs))
    video_scores.append(vid_score)
    video_labels.append(is_fall)
    offset += count
video_scores = np.array(video_scores)
video_labels = np.array(video_labels)

log(f"\n=== Video-level (stride4, top-3 mean, {len(video_scores)} videos) ===")
log(f"fall={int(video_labels.sum())} adl={int((video_labels == 0).sum())}")
ap_vid = average_precision(video_labels, video_scores)
log(f"mAP={ap_vid:.4f} (recorded)")

thr_con, j_con = youden_j(video_scores, video_labels, 0.40, 0.75)
thr_unc, j_unc = youden_j(video_scores, video_labels, 0.05, 0.95)
m_con = five_metrics(video_labels, (video_scores >= thr_con).astype(int))
m_unc = five_metrics(video_labels, (video_scores >= thr_unc).astype(int))
log(f"\nYouden-J constrained [0.40,0.75]: thr={thr_con:.3f} J={j_con:.4f}")
log(f"  acc={m_con['acc']:.4f} spec={m_con['spec']:.4f} rec={m_con['rec']:.4f} "
    f"f1={m_con['f1']:.4f} | {m_con['tp']}/{m_con['tn']}/{m_con['fp']}/{m_con['fn']}")
log(f"Youden-J unconstrained:           thr={thr_unc:.3f} J={j_unc:.4f}")
log(f"  acc={m_unc['acc']:.4f} spec={m_unc['spec']:.4f} rec={m_unc['rec']:.4f} "
    f"f1={m_unc['f1']:.4f} | {m_unc['tp']}/{m_unc['tn']}/{m_unc['fp']}/{m_unc['fn']}")

passes = (m_con["acc"] >= 0.75 and m_con["spec"] >= 0.70 and m_con["rec"] >= 0.70
          and thr_con <= 0.75 and (fall_med - adl_med) >= 0.15)
log(f"\n=== VERDICT: {'PASS' if passes else 'FAIL'} ===")
log(f"  acc>=0.75: {'Y' if m_con['acc'] >= 0.75 else 'N'} ({m_con['acc']:.4f})")
log(f"  spec>=0.70: {'Y' if m_con['spec'] >= 0.70 else 'N'} ({m_con['spec']:.4f})")
log(f"  rec>=0.70: {'Y' if m_con['rec'] >= 0.70 else 'N'} ({m_con['rec']:.4f})")
log(f"  thr<=0.75: {'Y' if thr_con <= 0.75 else 'N'} ({thr_con:.3f})")
log(f"  gap>=0.15: {'Y' if (fall_med - adl_med) >= 0.15 else 'N'} ({fall_med - adl_med:.4f})")

result = {
    "tag": TAG, "val_subject": VAL_SUBJECT, "ckpt": CKPT.name,
    "window": {**m16, "mAP": ap16, "fall_med": fall_med, "adl_med": adl_med,
               "gap": fall_med - adl_med},
    "video_constrained": {**m_con, "mAP": ap_vid, "thr": thr_con, "j": j_con},
    "video_unconstrained": {**m_unc, "thr": thr_unc, "j": j_unc},
    "passes": passes,
}
out_path = OUT_DIR / f"s22_eval_{TAG}.json"
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
log(f"saved: {out_path}")

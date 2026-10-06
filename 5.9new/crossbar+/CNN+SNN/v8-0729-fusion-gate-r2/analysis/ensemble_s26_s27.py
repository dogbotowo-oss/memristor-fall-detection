"""Decision-layer ensemble of S26 (cnnfeat, fall-amplifier SNN) and S27
(cnnfeat_delta, ADL-suppressor SNN). Equal-weight probability average: no new
parameters, nothing fitted on val. Tests whether the two SNN roles are
complementary at the decision layer (option B).
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
OUT_PATH = EXP_ROOT / "analysis" / "ensemble_s26_s27.json"
TAGS = ["c1_s26_cnnfeat", "c1_s27_cnnfeat_delta"]
VAL_SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "Subject 3"

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
spec = importlib.util.spec_from_file_location("r2_ens", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_ens"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)


def load_model(tag: str):
    ckpt = EXP_ROOT / "models" / tag / f"automatic_fall_event_detector_{tag}.pth"
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
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
        snn_cnnfeat_detach=bool(cfg.get("snn_cnnfeat_detach", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log(f"loaded {tag}: input={cfg.get('snn_input_mode')}")
    return model


def build_dataset(windows, labels):
    return base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
        auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
        use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
        hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
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
    return {"acc": round(acc, 4), "spec": round(spec, 4), "rec": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def youden_j(scores, y_true, lo, hi):
    best_j, best_thr = -1.0, lo
    for thr in np.arange(lo, hi + 0.001, 0.005):
        m = five_metrics(y_true, (scores >= thr).astype(int))
        j = m["spec"] + m["rec"] - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr, best_j


models = [load_model(t) for t in TAGS]
fall_id = base.LABEL_NAME_TO_ID["fall"]
results = {"tags": TAGS, "val_subject": VAL_SUBJECT}

# --- window level ---
windows16, labels16, _ = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
y16 = (labels16 == fall_id).astype(np.int64)
ds16 = build_dataset(windows16, labels16)
per_model16 = [score_windows(m, ds16) for m in models]
ens16 = np.mean(per_model16, axis=0)
m = five_metrics(y16, (ens16 >= 0.5).astype(int))
fall_med = float(np.median(ens16[y16 == 1]))
adl_med = float(np.median(ens16[y16 == 0]))
results["window"] = {**m, "fall_med": round(fall_med, 4), "adl_med": round(adl_med, 4),
                     "gap": round(fall_med - adl_med, 4)}
log(f"window ensemble: {results['window']}")

# --- video level ---
windows4, labels4, summary4 = base.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=4, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
ds4 = build_dataset(windows4, labels4)
per_model4 = [score_windows(m, ds4) for m in models]
ens4 = np.mean(per_model4, axis=0)
for m in models:
    del m
torch.cuda.empty_cache()

video_scores, video_labels, offset = [], [], 0
for item in summary4["videos"]:
    count = int(item["num_windows"])
    vid_probs = ens4[offset:offset + count]
    is_fall = int(base.infer_video_level_label_id(Path(item["video_path"]), "fall") == fall_id)
    video_scores.append(float(np.mean(np.sort(vid_probs)[-3:])) if count >= 3 else float(np.mean(vid_probs)))
    video_labels.append(is_fall)
    offset += count
video_scores = np.array(video_scores)
video_labels = np.array(video_labels)

thr_con, j_con = youden_j(video_scores, video_labels, 0.40, 0.75)
thr_unc, j_unc = youden_j(video_scores, video_labels, 0.05, 0.95)
m_con = five_metrics(video_labels, (video_scores >= thr_con).astype(int))
m_unc = five_metrics(video_labels, (video_scores >= thr_unc).astype(int))
results["video_constrained"] = {**m_con, "thr": round(thr_con, 3), "j": round(j_con, 4)}
results["video_unconstrained"] = {**m_unc, "thr": round(thr_unc, 3), "j": round(j_unc, 4)}
log(f"video constrained: {results['video_constrained']}")
log(f"video unconstrained: {results['video_unconstrained']}")

OUT_PATH.write_text(json.dumps(results, indent=2))
log(f"saved {OUT_PATH}")

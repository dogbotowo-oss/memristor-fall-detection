"""Subject4 one-shot test for S22 and S23b with frozen val-derived thresholds.

Aggregation: max (best from S23 aggregation comparison).
Thresholds: each model's own val-fold Youden-J (from eval_s22.py results).
Protocol disclosure: Subject4 was previously observed during S10 selection
and failure diagnosis. These are post-leakage measurements, not pristine
one-shot tests. Models never trained on Subject4.
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
OUT_DIR = EXP_ROOT / "reports" / "s22_s23_subject4"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
STRIDE = 4

# Frozen thresholds from val fold (Subject3+Zenodo val) Youden-J, max aggregation.
# From eval_s23_aggregation.py output.
MODELS = {
    "s22": {
        "ckpt": EXP_ROOT / "models" / "c1_s22_lif" / "automatic_fall_event_detector_c1_s22_lif.pth",
        "thr": 0.740,  # constrained Youden-J [0.40,0.75] on val fold1
        "thr_uncon": 0.870,
    },
    "s23b": {
        "ckpt": EXP_ROOT / "models" / "c1_s23b_tgt02" / "automatic_fall_event_detector_c1_s23b_tgt02.pth",
        "thr": 0.695,  # constrained Youden-J [0.40,0.75] on val fold1
        "thr_uncon": 0.935,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2_s4", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_s4"] = r2
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
    return model


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


# --- load Subject4 data ---
windows4, labels4, summary4 = base.load_external_labeled_windows(
    [SUBJECT4_DIR], image_size=64, window_size=16, stride=STRIDE, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
dataset = base.CombinedFallDataset(
    video_windows=np.asarray(windows4, dtype=np.float32),
    video_labels=labels4,
    auxiliary_records=[],
    window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values,
    device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=10041,
)
fall_id = base.LABEL_NAME_TO_ID["fall"]

# --- per-video ground truth ---
video_labels = []
offset = 0
for item in summary4["videos"]:
    count = int(item["num_windows"])
    is_fall = int(base.infer_video_level_label_id(Path(item["video_path"]), "fall") == fall_id)
    video_labels.append((is_fall, count, Path(item["video_path"]).name))
    offset += count
y_true = np.array([v[0] for v in video_labels])
log(f"Subject4: {len(video_labels)} videos, fall={int(y_true.sum())} adl={int((y_true == 0).sum())}")

# --- score each model ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}
for name, info in MODELS.items():
    model = load_model(info["ckpt"])
    probs = []
    with torch.no_grad():
        for off in range(0, len(dataset), 32):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + 32, len(dataset)))])
            logits, _ = model.forward_with_gate(xs.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    p4 = torch.cat(probs).numpy()
    del model
    torch.cuda.empty_cache()

    # max aggregation per video
    peaks = []
    off = 0
    for _, count, _ in video_labels:
        peaks.append(float(p4[off:off + count].max()))
        off += count
    peaks = np.array(peaks)

    for label, thr in [("constrained", info["thr"]), ("unconstrained", info["thr_uncon"])]:
        pred = (peaks >= thr).astype(int)
        m = five_metrics(y_true, pred)
        key = f"{name}_{label}"
        results[key] = {**m, "thr": thr}
        log(f"{key:25s} thr={thr:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
            f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | "
            f"TP/TN/FP/FN={m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

# --- save ---
out = {"protocol": "post-leakage measurement; models never trained on Subject4; "
                    "thresholds frozen from val fold1 Youden-J",
       "aggregation": "max", "results": results}
(OUT_DIR / "subject4_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
log(f"\nsaved: {OUT_DIR / 'subject4_results.json'}")

"""Subject4 test for S28 (CNN+GRU+SNN) vs S28b (CNN+GRU-only baseline).

Thresholds frozen from each model's own val fold1 eval (top-3 mean,
constrained Youden-J): S28=0.735, S28b=0.400. Aggregation on Subject4 uses
the same top-3 mean (consistency with the threshold derivation); max
aggregation reported as secondary reference. Window-level at thr=0.50 also
reported (canonical fixed point, no tuning).

Protocol disclosure: Subject4 was previously observed during S10 selection
and S22/S23 diagnostics. These are post-leakage measurements, not pristine
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
OUT_DIR = EXP_ROOT / "reports" / "s28_subject4"
CLI_TAGS = set(sys.argv[1:])

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"

MODELS = {
    "s28_cnn_gru_snn": {
        "ckpt": EXP_ROOT / "models" / "c1_s28_cnnfeat_noise" / "automatic_fall_event_detector_c1_s28_cnnfeat_noise.pth",
        "thr": 0.735,
    },
    "s28b_cnn_gru_only": {
        "ckpt": EXP_ROOT / "models" / "c1_s28b_cnngru" / "automatic_fall_event_detector_c1_s28b_cnngru.pth",
        "thr": 0.400,
    },
    "s28c_cnn_gru_fairinit": {
        "ckpt": EXP_ROOT / "models" / "c1_s28c_cnngru_fair" / "automatic_fall_event_detector_c1_s28c_cnngru_fair.pth",
        "thr": 0.415,
    },
    "s28d_snn_silent": {
        "ckpt": EXP_ROOT / "models" / "c1_s28d_snn_silent" / "automatic_fall_event_detector_c1_s28d_snn_silent.pth",
        "thr": 0.590,
    },
    "s30b_fallpreserve": {
        "ckpt": EXP_ROOT / "models" / "c1_s30b_fallpreserve" / "automatic_fall_event_detector_c1_s30b_fallpreserve.pth",
        "thr": 0.740,
    },
    "s31_posekin": {
        "ckpt": EXP_ROOT / "models" / "c1_s31_posekin" / "automatic_fall_event_detector_c1_s31_posekin.pth",
        "thr": 0.740,
    },
}
if CLI_TAGS:
    MODELS = {k: v for k, v in MODELS.items() if k in CLI_TAGS}


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2_s4s28", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_s4s28"] = r2
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
        snn_event_encoding=bool(cfg.get("snn_event_encoding", False)),
        snn_cnnfeat_detach=bool(cfg.get("snn_cnnfeat_detach", True)),
        snn_silent=bool(cfg.get("snn_silent", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log(f"loaded {ckpt_path.parent.name}: snn={cfg.get('use_snn_temporal_branch')} "
        f"input={cfg.get('snn_input_mode')}")
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
    return {"acc": round(acc, 4), "bal": round((spec + rec) / 2, 4), "spec": round(spec, 4),
            "rec": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def score(model, dataset):
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    probs = []
    with torch.no_grad():
        for off in range(0, len(dataset), 32):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + 32, len(dataset)))])
            ps = torch.stack([dataset[i][5] for i in range(off, min(off + 32, len(dataset)))])
            logits, _ = model.forward_with_gate(xs.to(device), pose=ps.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    return torch.cat(probs).numpy()


windows16, labels16, summary16 = base.load_external_labeled_windows(
    [SUBJECT4_DIR], image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
fall_id = base.LABEL_NAME_TO_ID["fall"]
y16 = (labels16 == fall_id).astype(np.int64)
ds16 = base.CombinedFallDataset(
    video_windows=np.asarray(windows16, dtype=np.float32), video_labels=labels16,
    auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=10041,
    pose_windows=base.load_pose_windows_for_summary(summary16, 16, 16),
)
log(f"Subject4 windows: {len(y16)} fall={int(y16.sum())} adl={int((y16 == 0).sum())}")

windows4, labels4, summary4 = base.load_external_labeled_windows(
    [SUBJECT4_DIR], image_size=64, window_size=16, stride=4, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
ds4 = base.CombinedFallDataset(
    video_windows=np.asarray(windows4, dtype=np.float32), video_labels=labels4,
    auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=10041,
    pose_windows=base.load_pose_windows_for_summary(summary4, 16, 4),
)
video_meta, offset = [], 0
for item in summary4["videos"]:
    count = int(item["num_windows"])
    is_fall = int(base.infer_video_level_label_id(Path(item["video_path"]), "fall") == fall_id)
    video_meta.append((is_fall, count))
    offset += count
yv = np.array([v[0] for v in video_meta])
log(f"Subject4 videos: {len(yv)} fall={int(yv.sum())} adl={int((yv == 0).sum())}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}
for name, info in MODELS.items():
    model = load_model(info["ckpt"])
    p16 = score(model, ds16)
    p4 = score(model, ds4)
    del model
    torch.cuda.empty_cache()

    m16 = five_metrics(y16, (p16 >= 0.5).astype(int))
    results[f"{name}_window_thr0.5"] = m16
    log(f"{name} window@0.50 | acc={m16['acc']:.4f} bal={m16['bal']:.4f} "
        f"spec={m16['spec']:.4f} rec={m16['rec']:.4f} f1={m16['f1']:.4f}")

    top3, peak, off = [], [], 0
    for _, count in video_meta:
        seg = p4[off:off + count]
        top3.append(float(np.mean(np.sort(seg)[-3:])) if count >= 3 else float(np.mean(seg)))
        peak.append(float(seg.max()))
        off += count
    top3, peak = np.array(top3), np.array(peak)
    for agg_name, scores in [("top3mean", top3), ("max", peak)]:
        m = five_metrics(yv, (scores >= info["thr"]).astype(int))
        results[f"{name}_video_{agg_name}"] = {**m, "thr": info["thr"]}
        log(f"{name} video-{agg_name}@{info['thr']:.3f} | acc={m['acc']:.4f} bal={m['bal']:.4f} "
            f"spec={m['spec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} | "
            f"TP/TN/FP/FN={m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

out = {"protocol": "post-leakage measurement; models never trained on Subject4; "
                  "thresholds frozen from val fold1 (top-3 mean constrained Youden-J)",
       "results": results}
(OUT_DIR / "subject4_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
log(f"saved: {OUT_DIR / 'subject4_results.json'}")

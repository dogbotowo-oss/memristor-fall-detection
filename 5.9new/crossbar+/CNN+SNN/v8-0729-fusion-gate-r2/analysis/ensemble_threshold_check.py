"""Validation-set ensemble + threshold calibration across the S-series.

Models (all EMA checkpoints, same training recipe, different SNN inputs):
  S3 scalar7, S5 scalar7_grid, S6b scalar7_flow, S7 scalar7_flow+seg4x3.
Also evaluates S3 with built-in-style late fusion lambda=0.2 (= v88).

Protocol: 1076-window validation set only (Subject3 + Zenodo val).
Threshold values are calibrated ON the validation set (same protocol as the
v8 p006 baseline, whose 0.70 threshold was also chosen on this split); the
final unbiased check remains Subject4. Printed thresholds are in-sample.
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
CKPTS = {
    "s3_scalar7": EXP_ROOT / "models" / "c1_ema0999_s3" / "automatic_fall_event_detector_c1_ema0999_s3_ema.pth",
    "s5_grid": EXP_ROOT / "models" / "c1_ema0999_s5" / "automatic_fall_event_detector_c1_ema0999_s5_ema.pth",
    "s6_flow": EXP_ROOT / "models" / "c1_ema0999_s6flow" / "automatic_fall_event_detector_c1_ema0999_s6flow_ema.pth",
    "s7_segflow": EXP_ROOT / "models" / "c1_ema0999_s7seg" / "automatic_fall_event_detector_c1_ema0999_s7seg_ema.pth",
}
OUT_CSV = EXP_ROOT / "analysis" / "ensemble_threshold_check.csv"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
normal_id = base.LABEL_NAME_TO_ID["normal"]


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return model, cfg


def forward_probs(model) -> tuple[np.ndarray, np.ndarray]:
    logits_list, snn_aux_list = [], []
    BATCH = 32
    with torch.no_grad():
        for offset in range(0, len(val_dataset), BATCH):
            batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
            batch_x = batch_x.to(device)
            logits, gate_info = model.forward_with_gate(batch_x)
            snn_aux = model.snn_class_head(gate_info["snn_pooled_pregate"])
            logits_list.append(logits.cpu())
            snn_aux_list.append(snn_aux.cpu())
    return torch.cat(logits_list).numpy(), torch.cat(snn_aux_list).numpy()


def metrics_from_pred(pred: np.ndarray) -> tuple[float, float, float, float]:
    tp = int(np.sum((true_binary == 1) & (pred == 1)))
    tn = int(np.sum((true_binary == 0) & (pred == 0)))
    fp = int(np.sum((true_binary == 0) & (pred == 1)))
    fn = int(np.sum((true_binary == 1) & (pred == 0)))
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / len(true_binary)
    return 0.5 * (spec + rec), spec, rec, acc


def threshold_curve(fall_prob: np.ndarray) -> tuple[float, float, float, float, float]:
    best = (-1.0, 0.5, 0.0, 0.0, 0.0)
    for thr in np.arange(0.30, 0.901, 0.025):
        pred = (fall_prob >= thr).astype(np.int64)
        bal, spec, rec, acc = metrics_from_pred(pred)
        if bal > best[0]:
            best = (bal, float(thr), spec, rec, acc)
    return best


probs: dict[str, np.ndarray] = {}
rows: list[dict[str, object]] = []
for name, path in CKPTS.items():
    model, cfg = load_model(path)
    logits, snn_aux = forward_probs(model)
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    probs[name] = prob
    pred = (np.argmax(logits, axis=1) == fall_id).astype(np.int64)
    bal, spec, rec, acc = metrics_from_pred(pred)
    tbal, thr, tspec, trec, tacc = threshold_curve(prob[:, fall_id])
    log(f"{name:12s} argmax: bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} acc={acc:.4f} | "
        f"best-thr={thr:.3f}: bal={tbal:.4f} spec={tspec:.4f} rec={trec:.4f} acc={tacc:.4f}")
    rows.append({"model": name, "mode": "argmax", "bal": bal, "spec": spec, "rec": rec, "acc": acc, "thr": 0.0})
    rows.append({"model": name, "mode": "best_thr", "bal": tbal, "spec": tspec, "rec": trec, "acc": tacc, "thr": thr})
    if name == "s3_scalar7":
        fused = logits.copy()
        fused[:, normal_id] += 0.2 * snn_aux[:, 0]
        fused[:, fall_id] += 0.2 * snn_aux[:, 1]
        fprob = torch.softmax(torch.from_numpy(fused), dim=1).numpy()
        probs["v88_s3_fused"] = fprob
        pred = (np.argmax(fused, axis=1) == fall_id).astype(np.int64)
        bal, spec, rec, acc = metrics_from_pred(pred)
        tbal, thr, tspec, trec, tacc = threshold_curve(fprob[:, fall_id])
        log(f"{'v88_s3_fused':12s} argmax: bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} acc={acc:.4f} | "
            f"best-thr={thr:.3f}: bal={tbal:.4f} spec={tspec:.4f} rec={trec:.4f} acc={tacc:.4f}")
        rows.append({"model": "v88_s3_fused", "mode": "argmax", "bal": bal, "spec": spec, "rec": rec, "acc": acc, "thr": 0.0})
        rows.append({"model": "v88_s3_fused", "mode": "best_thr", "bal": tbal, "spec": tspec, "rec": trec, "acc": tacc, "thr": thr})
    del model
    torch.cuda.empty_cache()

# equal-weight probability ensembles
combos = {
    "ens_s356": ["s3_scalar7", "s5_grid", "s6_flow"],
    "ens_all4": ["s3_scalar7", "s5_grid", "s6_flow", "s7_segflow"],
    "ens_all4+v88": ["v88_s3_fused", "s5_grid", "s6_flow", "s7_segflow"],
}
for name, members in combos.items():
    prob = np.mean([probs[m] for m in members], axis=0)
    pred = (np.argmax(prob, axis=1) == fall_id).astype(np.int64)
    bal, spec, rec, acc = metrics_from_pred(pred)
    tbal, thr, tspec, trec, tacc = threshold_curve(prob[:, fall_id])
    log(f"{name:12s} argmax: bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} acc={acc:.4f} | "
        f"best-thr={thr:.3f}: bal={tbal:.4f} spec={tspec:.4f} rec={trec:.4f} acc={tacc:.4f}")
    rows.append({"model": name, "mode": "argmax", "bal": bal, "spec": spec, "rec": rec, "acc": acc, "thr": 0.0})
    rows.append({"model": name, "mode": "best_thr", "bal": tbal, "spec": tspec, "rec": trec, "acc": tacc, "thr": thr})

with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["model", "mode", "bal", "spec", "rec", "acc", "thr"])
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
log(f"wrote {OUT_CSV.name}")

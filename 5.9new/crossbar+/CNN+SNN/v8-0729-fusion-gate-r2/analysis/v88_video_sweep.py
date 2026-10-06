"""Offline video-level sweep for v88 (+ S5/S6/S7) on the protocol val set.

Runs each checkpoint once over the 1076 protocol val windows (Subject3 +
Zenodo val, stride 16, 64x64), stores per-window fused fall probabilities,
then aggregates per video with several rules (peak / mean / top-k mean)
and sweeps decision thresholds offline.

Ground truth: Subject3 video is fall iff any of its labeled windows is fall;
Zenodo val from the ADL/Fall subfolder name. Threshold tuning on val is
in-sample, same as the p006 protocol.
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
OUT_DIR = EXP_ROOT / "analysis"

CKPTS = {
    "v88": Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v88\models\automatic_fall_event_detector_v88.pth"),
    "s5_grid": EXP_ROOT / "models" / "c1_ema0999_s5" / "automatic_fall_event_detector_c1_ema0999_s5_ema.pth",
    "s6_flow": EXP_ROOT / "models" / "c1_ema0999_s6flow" / "automatic_fall_event_detector_c1_ema0999_s6flow_ema.pth",
    "s7_segflow": EXP_ROOT / "models" / "c1_ema0999_s7seg" / "automatic_fall_event_detector_c1_ema0999_s7seg_ema.pth",
}

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(d for d in ROOT.iterdir() if d.is_dir() and (d / "13354453").exists())
GMDCSA_ROOT = next(
    p for p in (TESTSET_DIR / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
S3_DIR = GMDCSA_ROOT / "Subject 3"
ZEN_DIR = ROOT / "zenodo_falldb_video_split" / "val"


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
windows, labels, summary = base.load_external_labeled_windows(
    [S3_DIR, ZEN_DIR],
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

fall_id = base.LABEL_NAME_TO_ID["fall"]

bounds = []  # (name, split, true_fall, start, end)
offset = 0
for row in summary["videos"]:
    n = int(row["num_windows"])
    vpath = Path(row["video_path"])
    try:
        vpath.relative_to(S3_DIR)
        split = "subject3"
        true_fall = bool((labels[offset : offset + n] == fall_id).any())
    except ValueError:
        rel = vpath.relative_to(ZEN_DIR)
        split = "zenodo_val"
        true_fall = str(rel).startswith("Fall")
    bounds.append((vpath.name, split, true_fall, offset, offset + n))
    offset += n
assert offset == len(windows)
log(f"videos={len(bounds)} fall={sum(b[2] for b in bounds)} adl={sum(1 for b in bounds if not b[2])}")

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


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint.get("config", {}))
    r2.REFINED_MODE = str(cfg.get("snn_fusion_gate_mode", "group"))
    r2.REFINED_FLOOR = float(cfg.get("snn_fusion_gate_floor", 0.2))
    r2.REFINED_GROUP_BIAS = float(cfg.get("refined_group_bias", 0.0))
    r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
    r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.2))
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
    log(f"loaded {ckpt_path.parent.name}: lambda={r2.REFINED_LATE_FUSION_LAMBDA} "
        f"input={cfg.get('snn_input_mode', 'scalar7')} pool={cfg.get('snn_pool_mode', 'global3')}")
    return model


def window_fall_probs(model) -> np.ndarray:
    out = []
    batch = 32
    with torch.no_grad():
        for off in range(0, len(val_dataset), batch):
            xs = torch.stack([val_dataset[i][0] for i in range(off, min(off + batch, len(val_dataset)))])
            logits = model(xs.to(device))
            out.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    return torch.cat(out).numpy()


probs: dict[str, np.ndarray] = {}
for name, path in CKPTS.items():
    if not path.exists():
        log(f"skip {name}: missing {path}")
        continue
    model = load_model(path)
    probs[name] = window_fall_probs(model)
    del model
    torch.cuda.empty_cache()

np.save(OUT_DIR / "v88_val_window_probs.npy", np.stack(list(probs.values())))


def agg_features(win_prob: np.ndarray) -> dict[str, np.ndarray]:
    feats = {"peak": [], "mean": [], "top2": [], "top3": [], "nwin": []}
    for _, _, _, s, e in bounds:
        p = np.sort(win_prob[s:e])[::-1]
        feats["peak"].append(p[0])
        feats["mean"].append(p.mean())
        feats["top2"].append(p[:2].mean())
        feats["top3"].append(p[:3].mean())
        feats["nwin"].append(e - s)
    return {k: np.asarray(v) for k, v in feats.items()}


video_split = [b[1] for b in bounds]
video_true = np.asarray([b[2] for b in bounds], dtype=np.int64)
s3_mask = np.asarray([s == "subject3" for s in video_split])

combos = {name: [name] for name in probs}
if all(m in probs for m in ("v88", "s5_grid", "s6_flow")):
    combos["ens_v88_s5_s6"] = ["v88", "s5_grid", "s6_flow"]
if len(probs) == 4:
    combos["ens_all4"] = list(probs)

feat_store: dict[str, dict[str, np.ndarray]] = {}
rows = []
for cname, members in combos.items():
    win_prob = np.mean([probs[m] for m in members], axis=0)
    feats = agg_features(win_prob)
    feat_store[cname] = feats
    for i, (vname, split, tf, _, _) in enumerate(bounds):
        rows.append({
            "combo": cname, "video": vname, "split": split, "true_fall": int(tf),
            "peak": round(float(feats["peak"][i]), 6), "mean": round(float(feats["mean"][i]), 6),
            "top2": round(float(feats["top2"][i]), 6), "top3": round(float(feats["top3"][i]), 6),
            "nwin": int(feats["nwin"][i]),
        })

csv_path = OUT_DIR / "v88_video_scores.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
log(f"saved {csv_path}")


def metrics(pred: np.ndarray, mask: np.ndarray | None = None):
    y = video_true if mask is None else video_true[mask]
    p = pred if mask is None else pred[mask]
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(len(y), 1)
    return acc, 0.5 * (spec + rec), spec, rec, (tp, tn, fp, fn)


log("")
log("=== video-level sweep (combined = subject3 + zenodo_val) ===")
best_rows = []
for cname, feats in feat_store.items():
    for rule in ("peak", "top2", "top3", "mean"):
        score = feats[rule]
        for thr in np.arange(0.30, 0.951, 0.05):
            pred = (score >= thr).astype(np.int64)
            acc, bal, spec, rec, conf = metrics(pred)
            _, s3bal, _, _, _ = metrics(pred, s3_mask)
            _, zbal, _, _, _ = metrics(pred, ~s3_mask)
            flag = " <<<" if (acc >= 0.80 and spec >= 0.75 and rec >= 0.75) else ""
            log(f"{cname:14s} {rule:5s} thr={thr:.2f} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} "
                f"rec={rec:.4f} | TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]} | "
                f"s3_bal={s3bal:.4f} zen_bal={zbal:.4f}{flag}")
            best_rows.append((bal, acc, spec, rec, cname, rule, float(thr), conf))

log("")
log("=== top-10 by combined balanced accuracy ===")
for bal, acc, spec, rec, cname, rule, thr, conf in sorted(best_rows, reverse=True)[:10]:
    log(f"{cname:14s} {rule:5s} thr={thr:.2f} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} "
        f"| TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]}")

"""Dense-stride inference sweep: does finer temporal sampling lift video-level scores?

Same protocol val videos as v88_video_sweep.py, but windows extracted with a
smaller stride (default 4 instead of 16) so per-video peaks/counts are
estimated more densely. Ground truth per video is frozen from the stride-16
protocol CSV (v88_video_scores.csv, combo=v88 rows) to keep the label rule
identical across strides.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4

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


# frozen GT from the stride-16 protocol run, keyed by dataset-relative path
# (bare file names collide: Subject3 ADL/01.mp4 vs Fall/01.mp4)
gt_map = json.loads((OUT_DIR / "val_video_gt.json").read_text(encoding="utf-8"))

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
    stride=STRIDE,
    max_videos=0,
    fall_name_pattern="fall",
    use_pixel_human=False,
    use_silhouette_human=False,
    pixel_grid_size=20,
)
log(f"stride={STRIDE} windows={len(windows)}")

bounds = []  # (name, split, true_fall, start, end)
offset = 0
for row in summary["videos"]:
    n = int(row["num_windows"])
    vpath = Path(row["video_path"])
    try:
        rel = str(vpath.relative_to(S3_DIR))
    except ValueError:
        rel = str(vpath.relative_to(ZEN_DIR))
    entry = gt_map[rel.replace("\\", "/")]
    bounds.append((vpath.name, entry["split"], int(entry["true_fall"]), offset, offset + n))
    offset += n
assert offset == len(windows)

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
    return model


def window_fall_probs(model) -> np.ndarray:
    fall_id = base.LABEL_NAME_TO_ID["fall"]
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
        continue
    model = load_model(path)
    probs[name] = window_fall_probs(model)
    del model
    torch.cuda.empty_cache()
    log(f"scored {name}")

np.save(OUT_DIR / f"v88_val_window_probs_stride{STRIDE}.npy", np.stack(list(probs.values())))
(OUT_DIR / f"v88_val_bounds_stride{STRIDE}.json").write_text(
    json.dumps([{"video": b[0], "split": b[1], "true_fall": b[2], "start": b[3], "end": b[4]}
                for b in bounds], indent=1), encoding="utf-8")
log(f"saved stride-{STRIDE} probs + bounds")

video_true = np.asarray([b[2] for b in bounds], dtype=np.int64)


def metrics(pred: np.ndarray):
    y, p = video_true, pred
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / len(y)
    return acc, 0.5 * (spec + rec), spec, rec, (tp, tn, fp, fn)


combos = {n: probs[n] for n in probs}
if all(m in probs for m in ("v88", "s5_grid", "s6_flow")):
    combos["ens_v88_s5_s6"] = (probs["v88"] + probs["s5_grid"] + probs["s6_flow"]) / 3
if len(probs) == 4:
    combos["ens_all4"] = np.mean([probs[m] for m in ("v88", "s5_grid", "s6_flow", "s7_segflow")], axis=0)

results = []
for cname, wp in combos.items():
    peaks = np.asarray([wp[s:e].max() for _, _, _, s, e in bounds])
    for thr in np.arange(0.30, 0.951, 0.025):
        pred = (peaks >= thr).astype(np.int64)
        acc, bal, spec, rec, conf = metrics(pred)
        results.append((bal, acc, spec, rec, cname, "peak", round(float(thr), 3), 1, conf))
    for thr in np.arange(0.30, 0.951, 0.05):
        above = np.asarray([(wp[s:e] >= thr).sum() for _, _, _, s, e in bounds])
        for k in (2, 3, 4, 6, 8):
            pred = (above >= k).astype(np.int64)
            acc, bal, spec, rec, conf = metrics(pred)
            results.append((bal, acc, spec, rec, cname, "count", round(float(thr), 3), k, conf))

results.sort(reverse=True)
log("")
log(f"=== stride={STRIDE} top-15 video-level ===")
for bal, acc, spec, rec, cname, rule, thr, k, conf in results[:15]:
    log(f"{cname:14s} {rule:5s} thr={thr:.3f} k={k} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} "
        f"rec={rec:.4f} | TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]}")

goal = [r for r in results if r[1] >= 0.80 and r[2] >= 0.75 and r[3] >= 0.75]
log(f"rows meeting acc>=0.80 & spec>=0.75 & rec>=0.75: {len(goal)}")
for bal, acc, spec, rec, cname, rule, thr, k, conf in goal[:15]:
    log(f"{cname:14s} {rule:5s} thr={thr:.3f} k={k} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} "
        f"rec={rec:.4f} | TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]}")

"""S8 (train-time late fusion) evaluation: window-level + video-level.

Computes S8 probabilities on the protocol val windows at stride 16 (window-
level argmax metrics + SNN aux head AUC) and stride 4 (video-level peak
sweep), saves the stride-4 probs alongside the cached S-series probs, then
compares S8 vs v88 on the 9 v88-FN / 3 v88-FP videos and ensemble combos.
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_ema0999_s8fuse"
S8_CKPT = EXP_ROOT / "models" / TAG / f"automatic_fall_event_detector_{TAG}_ema.pth"

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


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint.get("config", {}))
    r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
    r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.0))
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
    log(f"loaded S8: lambda={r2.REFINED_LATE_FUSION_LAMBDA} "
        f"input={cfg.get('snn_input_mode')} pool={cfg.get('snn_pool_mode')}")
    return model


def build_dataset(windows, labels):
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
    )


def score(model, dataset):
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    probs, aux_probs = [], []
    batch = 32
    with torch.no_grad():
        for off in range(0, len(dataset), batch):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + batch, len(dataset)))])
            logits, gate_info = model.forward_with_gate(xs.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
            aux = gate_info.get("snn_class_logits")
            if aux is not None:
                aux_probs.append(torch.softmax(aux, dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy()
    a = torch.cat(aux_probs).numpy() if aux_probs else None
    return p, a


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


model = load_model(S8_CKPT)

# --- stride 16: protocol window-level ---
windows, labels, _ = base.load_external_labeled_windows(
    [S3_DIR, ZEN_DIR], image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
assert len(windows) == 1076
ds = build_dataset(windows, labels)
p16, a16 = score(model, ds)
fall_id = base.LABEL_NAME_TO_ID["fall"]
true16 = (labels == fall_id).astype(np.int64)
pred = (p16 >= 0.5).astype(np.int64)  # argmax-equivalent for the fused two-class mass
pred_argmax = pred  # fused fall prob vs 0.5 approximates argmax on {normal, fall}
tp = int(((true16 == 1) & (pred_argmax == 1)).sum()); tn = int(((true16 == 0) & (pred_argmax == 0)).sum())
fp = int(((true16 == 0) & (pred_argmax == 1)).sum()); fn = int(((true16 == 1) & (pred_argmax == 0)).sum())
spec = tn / max(tn + fp, 1); rec = tp / max(tp + fn, 1)
log(f"s8 window-level(stride16): bal={0.5*(spec+rec):.4f} spec={spec:.4f} rec={rec:.4f} TP/TN/FP/FN={tp}/{tn}/{fp}/{fn}")
if a16 is not None:
    main16 = p16
    log(f"s8 snn-aux AUC={auc(a16, true16):.4f} corr(main,snn)={np.corrcoef(main16, a16)[0,1]:.4f}")

# --- stride 4: video-level ---
windows4, labels4, summary4 = base.load_external_labeled_windows(
    [S3_DIR, ZEN_DIR], image_size=64, window_size=16, stride=4, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
log(f"stride=4 windows={len(windows4)}")
ds4 = build_dataset(windows4, labels4)
p4, _ = score(model, ds4)
del model
torch.cuda.empty_cache()

gt_map = json.loads((OUT_DIR / "val_video_gt.json").read_text(encoding="utf-8"))
bounds_raw = json.loads((OUT_DIR / "v88_val_bounds_stride4.json").read_text(encoding="utf-8"))
b4 = []  # (video, split, true_fall, start, end) in summary order
off = 0
name_to_cached = {b["video"]: b for b in bounds_raw}
for row in summary4["videos"]:
    n = int(row["num_windows"])
    vpath = Path(row["video_path"])
    try:
        rel = str(vpath.relative_to(S3_DIR))
    except ValueError:
        rel = str(vpath.relative_to(ZEN_DIR))
    entry = gt_map[rel.replace("\\", "/")]
    b4.append({"video": vpath.name, "split": entry["split"], "true_fall": entry["true_fall"],
               "start": off, "end": off + n})
    off += n
assert off == len(windows4)

np.save(OUT_DIR / f"{TAG}_val_window_probs_stride4.npy", p4)
(OUT_DIR / f"{TAG}_val_bounds_stride4.json").write_text(json.dumps(b4, indent=1), encoding="utf-8")
log(f"saved {TAG} stride-4 probs + bounds")

# --- video-level sweep: s8 alone + ensembles with cached models ---
cached = np.load(OUT_DIR / "v88_val_window_probs_stride4.npy")  # [v88, s5, s6, s7]
cached_bounds = [(b["start"], b["end"]) for b in bounds_raw]
cached_true = np.array([b["true_fall"] for b in bounds_raw])

# align s8 bounds to cached order by video name (both come from the same loader order;
# verify sizes match per position)
aligned = all(b4[i]["video"] == bounds_raw[i]["video"] and b4[i]["end"] - b4[i]["start"] == cached_bounds[i][1] - cached_bounds[i][0]
              for i in range(len(b4)))
assert aligned and len(windows4) == cached.shape[1], "stride-4 window layout changed; re-score all models"
log(f"bounds alignment with cached S-series: OK ({len(b4)} videos)")

combos = {
    TAG.replace("c1_ema0999_", ""): p4,
    f"ens_{TAG.replace('c1_ema0999_', '')}_v88": (p4 + cached[0]) / 2,
    f"ens_{TAG.replace('c1_ema0999_', '')}_s5": (p4 + cached[1]) / 2,
    f"ens_{TAG.replace('c1_ema0999_', '')}_v88_s5_s6": (p4 + cached[0] + cached[1] + cached[2]) / 4,
}


def metrics(pred):
    y = cached_true
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    spec = tn / max(tn + fp, 1); rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / len(y)
    return acc, 0.5 * (spec + rec), spec, rec, (tp, tn, fp, fn)


results = []
for cname, wp in combos.items():
    peaks = np.array([wp[s:e].max() for s, e in cached_bounds])
    for thr in np.arange(0.50, 0.951, 0.025):
        acc, bal, spec, rec, conf = metrics((peaks >= thr).astype(int))
        results.append((bal, acc, spec, rec, cname, round(float(thr), 3), conf))
results.sort(key=lambda r: r[0], reverse=True)
log("")
log("=== stride-4 video-level top-12 (s8 + ensembles) ===")
for bal, acc, spec, rec, cname, thr, conf in results[:12]:
    log(f"{cname:16s} peak>={thr:.3f} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} | "
        f"TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]}")
goal = [r for r in results if r[1] >= 0.80 and r[2] >= 0.75 and r[3] >= 0.75]
log(f"rows meeting acc>=0.80 & spec>=0.75 & rec>=0.75: {len(goal)}")
for bal, acc, spec, rec, cname, thr, conf in goal[:10]:
    log(f"{cname:16s} peak>={thr:.3f} | acc={acc:.4f} bal={bal:.4f} spec={spec:.4f} rec={rec:.4f} | "
        f"TP/TN/FP/FN={conf[0]}/{conf[1]}/{conf[2]}/{conf[3]}")

# --- blind-spot movement: the 9 v88-FN and 3 v88-FP videos ---
v88_peaks = np.array([cached[0][s:e].max() for s, e in cached_bounds])
s8_peaks = np.array([p4[s:e].max() for s, e in cached_bounds])
log("")
log("=== v88-FN videos (9): peak v88 vs s8 ===")
for i in range(len(cached_true)):
    if cached_true[i] == 1 and v88_peaks[i] < 0.85:
        log(f"{bounds_raw[i]['video']:26s} {bounds_raw[i]['split']:10s} v88={v88_peaks[i]:.4f} s8={s8_peaks[i]:.4f}")
log("=== v88-FP videos (3): peak v88 vs s8 ===")
for i in range(len(cached_true)):
    if cached_true[i] == 0 and v88_peaks[i] >= 0.85:
        log(f"{bounds_raw[i]['video']:26s} {bounds_raw[i]['split']:10s} v88={v88_peaks[i]:.4f} s8={s8_peaks[i]:.4f}")

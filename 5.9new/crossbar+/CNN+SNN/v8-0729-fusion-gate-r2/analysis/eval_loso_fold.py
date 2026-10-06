"""LOSO fold evaluation: window-level + video-level on a chosen held-out subject.

Usage: eval_loso_fold.py <tag> [val_subject]   (default val subject: "Subject 2")

Same protocol as s8_eval.py (stride16 window metrics, stride4 video peak sweep)
but evaluated on <val_subject> + Zenodo val, without the cached v88 ensemble
comparison (those caches only exist for the Subject-3 fold).
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
TAG = sys.argv[1]
VAL_SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "Subject 2"
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
spec = importlib.util.spec_from_file_location("r2_loso", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_loso"] = r2
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
    with torch.no_grad():
        for off in range(0, len(dataset), 32):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + 32, len(dataset)))])
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
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def confusion(scores, truths, thr):
    tp = sum(1 for s, y in zip(scores, truths) if y == 1 and s >= thr)
    fn = sum(1 for s, y in zip(scores, truths) if y == 1 and s < thr)
    tn = sum(1 for s, y in zip(scores, truths) if y == 0 and s < thr)
    fp = sum(1 for s, y in zip(scores, truths) if y == 0 and s >= thr)
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"acc": (tp + tn) / max(len(scores), 1), "bal": (spec + rec) / 2,
            "spec": spec, "rec": rec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    log(f"LOSO eval | tag={TAG} val_subject={VAL_SUBJECT}")
    model = load_model(CKPT)

    windows, labels, _ = base.load_external_labeled_windows(
        VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
        fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    )
    log(f"stride16 windows={len(windows)}")
    p16, a16 = score(model, build_dataset(windows, labels))
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    true16 = (labels == fall_id).astype(np.int64)
    m = confusion(p16.tolist(), true16.tolist(), 0.5)
    log(f"window(stride16): bal={m['bal']:.4f} spec={m['spec']:.4f} rec={m['rec']:.4f} "
        f"TP/TN/FP/FN={m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")
    if a16 is not None:
        log(f"snn-aux AUC={auc(a16, true16):.4f} corr(main,snn)={np.corrcoef(p16, a16)[0, 1]:.4f}")

    windows4, labels4, summary4 = base.load_external_labeled_windows(
        VAL_DIRS, image_size=64, window_size=16, stride=4, max_videos=0,
        fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    )
    log(f"stride4 windows={len(windows4)}")
    p4, _ = score(model, build_dataset(windows4, labels4))

    videos = []
    offset = 0
    for item in summary4["videos"]:
        count = int(item["num_windows"])
        video_path = Path(item["video_path"])
        split = VAL_SUBJECT.replace(" ", "").lower() if "subject" in str(video_path).lower() else "zenodo_val"
        peak = float(p4[offset:offset + count].max())
        videos.append({"video": video_path.name, "split": split, "true_fall": int(
            base.infer_video_level_label_id(video_path, "fall") == fall_id), "peak": peak})
        offset += count
    assert offset == len(p4)

    ths = np.linspace(0.30, 0.95, 14)
    rows = [(t, confusion([v["peak"] for v in videos], [v["true_fall"] for v in videos], float(t))) for t in ths]
    best_t, best_m = max(rows, key=lambda r: r[1]["bal"])
    log(f"video best@{best_t:.2f}: acc={best_m['acc']:.4f} bal={best_m['bal']:.4f} "
        f"spec={best_m['spec']:.4f} rec={best_m['rec']:.4f} "
        f"TP/TN/FP/FN={best_m['tp']}/{best_m['tn']}/{best_m['fp']}/{best_m['fn']}")
    for split in sorted({v["split"] for v in videos}):
        sub = [v for v in videos if v["split"] == split]
        m = confusion([v["peak"] for v in sub], [v["true_fall"] for v in sub], float(best_t))
        log(f"  {split}: acc={m['acc']:.3f} spec={m['spec']:.3f} rec={m['rec']:.3f} "
            f"TP/TN/FP/FN={m['tp']}/{m['tn']}/{m['fp']}/{m['fn']} (n={len(sub)})")
    fall_peaks = np.array([v["peak"] for v in videos if v["true_fall"] == 1])
    adl_peaks = np.array([v["peak"] for v in videos if v["true_fall"] == 0])
    log(f"peak stats: fall mean={fall_peaks.mean():.3f} median={np.median(fall_peaks):.3f} | "
        f"adl mean={adl_peaks.mean():.3f} median={np.median(adl_peaks):.3f} | video AUC={auc(np.concatenate([fall_peaks, adl_peaks]), np.concatenate([np.ones(len(fall_peaks)), np.zeros(len(adl_peaks))]).astype(np.int64)):.4f}")

    np.save(OUT_DIR / f"{TAG}_loso_val_window_probs_stride4.npy", p4)
    (OUT_DIR / f"{TAG}_loso_val_bounds_stride4.json").write_text(
        json.dumps(videos, indent=1, ensure_ascii=False), encoding="utf-8")
    log("saved loso stride-4 probs + bounds")


if __name__ == "__main__":
    main()

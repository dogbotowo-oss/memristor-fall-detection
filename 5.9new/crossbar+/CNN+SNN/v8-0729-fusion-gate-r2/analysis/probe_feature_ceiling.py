"""Feature-ceiling probe: how discriminative are the raw SNN input features?

Bypasses the LIF entirely. Extracts the exact scalar7_flow feature tensors the
SNN branch consumes (extract_temporal_event_features, applied to the same
crossbar-transformed windows the model sees), then fits simple classifiers
(logistic regression / MLP) on train windows and reports val AUC.

If even these probes stay near 0.5-0.6 AUC, the bottleneck is the input
features themselves, and no LIF/dynamics change can fix the SNN branch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
OUT_PATH = EXP_ROOT / "analysis" / "feature_ceiling_probe.json"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(d for d in ROOT.iterdir() if d.is_dir() and (d / "13354453").exists())
GMDCSA_ROOT = next(
    p for p in (TESTSET_DIR / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
ZENODO = ROOT / "zenodo_falldb_video_split"
TRAIN_DIRS = [GMDCSA_ROOT / "Subject 1", GMDCSA_ROOT / "Subject 2", ZENODO / "train"]
VAL_DIRS = [GMDCSA_ROOT / "Subject 3", ZENODO / "val"]

MODE = "scalar7_flow"


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2_probe", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_probe"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

extract = base.FallEventDetector.extract_temporal_event_features


def load_split(dirs, tag):
    t0 = time.time()
    windows, labels, summary = base.load_external_labeled_windows(
        dirs, image_size=64, window_size=16, stride=16, max_videos=0,
        fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False,
        pixel_grid_size=20,
    )
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    y = (labels == fall_id).astype(np.int64)
    log(f"{tag}: windows={len(windows)} fall={int(y.sum())} adl={int((y == 0).sum())} "
        f"load={time.time() - t0:.0f}s")
    ds = base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
        auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
        use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
        hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=0.25, augment=False, base_seed=42,
    )
    feats = []
    bs = 64
    with torch.no_grad():
        for off in range(0, len(ds), bs):
            xs = torch.stack([ds[i][0] for i in range(off, min(off + bs, len(ds)))]).to(device)
            feats.append(extract(xs, mode=MODE).cpu().numpy())
    X = np.concatenate(feats, axis=0)  # (N, T, F)
    log(f"{tag}: features={X.shape} extract_done")
    return X, y


def reps(X):
    return {
        "flat": X.reshape(len(X), -1),
        "stats": np.concatenate([X.mean(1), X.max(1), X.std(1)], axis=1),
        "delta": np.concatenate(
            [np.diff(X, axis=1).max(1), np.diff(X, axis=1).mean(1), X[:, -1] - X[:, 0]], axis=1
        ),
    }


def main():
    Xtr, ytr = load_split(TRAIN_DIRS, "train")
    Xva, yva = load_split(VAL_DIRS, "val")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    tr_reps = reps(Xtr)
    va_reps = reps(Xva)
    results = {}
    for name in ["flat", "stats", "delta"]:
        sc = StandardScaler().fit(tr_reps[name])
        Ztr, Zva = sc.transform(tr_reps[name]), sc.transform(va_reps[name])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        lr.fit(Ztr, ytr)
        auc_lr = roc_auc_score(yva, lr.predict_proba(Zva)[:, 1])
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True,
                            random_state=0)
        mlp.fit(Ztr, ytr)
        auc_mlp = roc_auc_score(yva, mlp.predict_proba(Zva)[:, 1])
        results[name] = {"lr_auc": round(float(auc_lr), 4), "mlp_auc": round(float(auc_mlp), 4)}
        log(f"rep={name}: LR AUC={auc_lr:.4f}  MLP AUC={auc_mlp:.4f}")

    # reference: in-sample train AUC on flat (upper sanity bound)
    sc = StandardScaler().fit(tr_reps["flat"])
    Ztr = sc.transform(tr_reps["flat"])
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Ztr, ytr)
    results["train_insample_flat_lr_auc"] = round(
        float(roc_auc_score(ytr, lr.predict_proba(Ztr)[:, 1])), 4
    )
    log(f"train in-sample flat LR AUC={results['train_insample_flat_lr_auc']:.4f}")

    OUT_PATH.write_text(json.dumps({"mode": MODE, "results": results}, indent=2))
    log(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

"""SNN-off ablation for the v8 p006 baseline (uses the ORIGINAL v8 module so
the LIF behaves exactly as p006 was trained - no S25 output ReLU, no R2
wrapper). Quantifies how much p006's SNN branch contributes to fall recall vs
ADL specificity on the Subject3 val split.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

V8_CODE = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8\code\automatic_fall_detector.py")
P006_CKPT = Path(
    r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models"
    r"\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth"
)
OUT_PATH = Path(
    r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\analysis\snn_off_ablation_p006.json"
)

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
TESTSET_DIR = next(d for d in ROOT.iterdir() if d.is_dir() and (d / "13354453").exists())
GMDCSA_ROOT = next(
    p for p in (TESTSET_DIR / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
VAL_DIRS = [GMDCSA_ROOT / "Subject 3", ROOT / "zenodo_falldb_video_split" / "val"]


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["automatic_fall_detector.py"]
spec = importlib.util.spec_from_file_location("v8_base", V8_CODE)
v8 = importlib.util.module_from_spec(spec)
sys.modules["v8_base"] = v8
spec.loader.exec_module(v8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrs_values, lrs_values = v8.load_conductance_pair(DATA_DIR)
device_dynamics = v8.load_device_dynamics(DATA_DIR)

checkpoint = torch.load(P006_CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
model = v8.FallEventDetector(
    hidden_dim=int(cfg.get("hidden_dim", 64)),
    num_classes=len(v8.LABEL_NAME_TO_ID),
    use_snn_temporal_branch=True,
    snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
    snn_decay=float(cfg.get("snn_decay", 0.82)),
    snn_threshold=float(cfg.get("snn_threshold", 0.55)),
    snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
    learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
    use_snn_fusion_gate=bool(cfg.get("use_snn_fusion_gate", True)),
    snn_fusion_gate_mode=str(cfg.get("snn_fusion_gate_mode", "group_channel")),
    snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
    snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
log(f"loaded p006: gate={cfg.get('snn_fusion_gate_mode')} thr={cfg.get('snn_threshold')}")

windows, labels, _ = v8.load_external_labeled_windows(
    VAL_DIRS, image_size=64, window_size=16, stride=16, max_videos=0,
    fall_name_pattern="fall", use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
)
fall_id = v8.LABEL_NAME_TO_ID["fall"]
y = (labels == fall_id).astype(np.int64)
log(f"val windows={len(windows)} fall={int(y.sum())} adl={int((y == 0).sum())}")

dataset = v8.CombinedFallDataset(
    video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
    auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
    use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
    hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25, augment=False, base_seed=42 + 9999,
)

orig_lif = model.run_lif_temporal_branch


def zero_lif(x, features=None):
    return torch.zeros_like(orig_lif(x, features))


def score(snn_off: bool) -> np.ndarray:
    if snn_off:
        model.run_lif_temporal_branch = zero_lif
    probs = []
    with torch.no_grad():
        for off in range(0, len(dataset), 64):
            xs = torch.stack([dataset[i][0] for i in range(off, min(off + 64, len(dataset)))]).to(device)
            logits, _ = model.forward_with_gate(xs)
            probs.append(torch.softmax(logits, dim=1)[:, fall_id].cpu().numpy())
    if snn_off:
        model.run_lif_temporal_branch = orig_lif
    return np.concatenate(probs)


def metrics(probs: np.ndarray) -> dict:
    pred = (probs >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    order = np.argsort(-probs)
    ys = y[order]
    tp_c = np.cumsum(ys == 1).astype(np.float64)
    ap = float(np.trapz(tp_c / np.arange(1, len(y) + 1), tp_c / max(float((y == 1).sum()), 1.0)))
    return {
        "acc": round((tp + tn) / len(y), 4),
        "spec": round(tn / max(tn + fp, 1), 4),
        "rec": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "mAP": round(ap, 4),
        "fall_med": round(float(np.median(probs[y == 1])), 4),
        "adl_med": round(float(np.median(probs[y == 0])), 4),
    }


p_on = score(snn_off=False)
p_off = score(snn_off=True)
m_on, m_off = metrics(p_on), metrics(p_off)
delta = {k: round(m_off[k] - m_on[k], 4) for k in ["acc", "spec", "rec", "f1", "mAP", "fall_med", "adl_med"]}
log(f"p006 SNN on : {m_on}")
log(f"p006 SNN off: {m_off}")
log(f"delta(off-on): {delta}")
OUT_PATH.write_text(json.dumps({"tag": "p006", "snn_on": m_on, "snn_off": m_off,
                                "delta_off_minus_on": delta}, indent=2))
log(f"saved {OUT_PATH}")

"""Build a fair-init checkpoint for the CNN+GRU-only baseline.

The no-SNN model's head is (128, 384) but the v7 pretrained head is
(128, 480): the last 96 input columns are the SNN slots. Slicing them off
lets the baseline inherit the v7-pretrained head instead of starting from
random noise (the S28b handicap). SNN/gate keys are dropped entirely.
"""

from pathlib import Path

import torch

V7 = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v7\models\automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth")
OUT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2\models\init\v7_cnngru_head384_init.pth")

sd = torch.load(V7, map_location="cpu", weights_only=False)["model_state"]
new_sd = {}
for k, v in sd.items():
    if k.startswith("snn_"):
        continue
    if k == "head.0.weight":
        v = v[:, :384].contiguous()
    new_sd[k] = v

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state": new_sd, "config": {"note": "v7 with SNN/gate keys dropped and head.0 sliced to 384 (CNN+GRU-only fair init)"}}, OUT)
print(f"saved {OUT} keys={len(new_sd)} head.0={tuple(new_sd['head.0.weight'].shape)}")

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
SELECTIVE_CHANNELS = (56, 46, 62, 45, 44, 47, 33, 53, 41, 38, 61, 15)
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_selective_gate_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class SelectiveGateFallEventDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = "group_channel" if requested_mode == "group_selective12" else requested_mode
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode == "group_selective12":
            self.snn_fusion_gate_mode = "group_selective12"
            self.snn_channel_gate = None
            self.selective_gate_logits = nn.Parameter(torch.zeros(len(SELECTIVE_CHANNELS)))
            self.register_buffer(
                "selective_channel_indices",
                torch.tensor(SELECTIVE_CHANNELS, dtype=torch.long),
            )
        else:
            self.register_parameter("selective_gate_logits", None)
            self.register_buffer("selective_channel_indices", torch.empty(0, dtype=torch.long))

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode != "group_selective12":
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.snn_group_gate is None or self.selective_gate_logits is None:
            raise RuntimeError("Group and selective gates are required.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        group_gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        channel_scale = torch.ones_like(snn_pooled)
        selected_scale = 0.5 + 0.5 * torch.sigmoid(self.selective_gate_logits)
        channel_scale[:, self.selective_channel_indices] = selected_scale.view(1, -1)
        gate = group_gate * channel_scale
        return snn_pooled * gate, gate


original_parse_args = base.parse_args


def selective_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = "group_selective12"
    LAST_ARGS = parsed
    return parsed


def update_checkpoint_metadata() -> None:
    if LAST_ARGS is None or not LAST_ARGS.model_path.exists():
        return
    try:
        checkpoint = torch.load(LAST_ARGS.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(LAST_ARGS.model_path, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config["snn_fusion_gate_mode"] = "group_selective12"
    config["selective_channels"] = list(SELECTIVE_CHANNELS)
    config["selective_gate_min_scale"] = 0.5
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def main() -> None:
    base.FallEventDetector = SelectiveGateFallEventDetector
    base.parse_args = selective_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

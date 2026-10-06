from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
HIGH_CHANNELS = (68, 95, 80, 36, 63, 27, 23, 48, 31, 4, 35, 59, 55, 16, 39, 54, 58, 57, 50, 52, 3, 7, 2, 22)
LOW_CHANNELS = (56, 46, 62, 45, 44, 47, 33, 53, 41, 38, 61, 15)
CHANNEL_SCALE_RANGE = 0.5
GATE_GRADIENT_MULTIPLIER = 5.0
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_centered_gate_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class CenteredGateFallEventDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = "group_channel" if requested_mode == "group_centered_prior" else requested_mode
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode == "group_centered_prior":
            self.snn_fusion_gate_mode = "group_centered_prior"
            self.snn_channel_gate = None
            prior_logits = torch.zeros(96, dtype=torch.float32)
            prior_value = math.atanh(0.5)
            prior_logits[list(HIGH_CHANNELS)] = prior_value
            prior_logits[list(LOW_CHANNELS)] = -prior_value
            self.channel_prior_logits = nn.Parameter(prior_logits)
            self.channel_prior_logits.register_hook(
                lambda gradient: gradient * GATE_GRADIENT_MULTIPLIER
            )
            self.register_buffer("high_channel_indices", torch.tensor(HIGH_CHANNELS, dtype=torch.long))
            self.register_buffer("low_channel_indices", torch.tensor(LOW_CHANNELS, dtype=torch.long))
        else:
            self.register_parameter("channel_prior_logits", None)
            self.register_buffer("high_channel_indices", torch.empty(0, dtype=torch.long))
            self.register_buffer("low_channel_indices", torch.empty(0, dtype=torch.long))

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode != "group_centered_prior":
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.snn_group_gate is None or self.channel_prior_logits is None:
            raise RuntimeError("Group gate and centered channel prior are required.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        group_gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        channel_scale = 1.0 + CHANNEL_SCALE_RANGE * torch.tanh(self.channel_prior_logits)
        gate = group_gate * channel_scale.view(1, -1)
        return snn_pooled * gate, gate


original_parse_args = base.parse_args


def centered_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = "group_centered_prior"
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
    config["snn_fusion_gate_mode"] = "group_centered_prior"
    config["high_weight_channels"] = list(HIGH_CHANNELS)
    config["low_weight_channels"] = list(LOW_CHANNELS)
    config["channel_scale_range"] = CHANNEL_SCALE_RANGE
    config["gate_gradient_multiplier"] = GATE_GRADIENT_MULTIPLIER
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def main() -> None:
    base.FallEventDetector = CenteredGateFallEventDetector
    base.parse_args = centered_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

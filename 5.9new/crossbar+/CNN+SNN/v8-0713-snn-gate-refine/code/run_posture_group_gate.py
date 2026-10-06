from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
CONTEXT_NAMES = (
    "impact",
    "peak_motion",
    "center_drop",
    "posture_change",
    "ground_hold",
    "confirmed_fall",
)
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_posture_group_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class PostureGroupFallEventDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = "group_channel" if requested_mode == "group_posture_guided" else requested_mode
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        snn_hidden_dim = int(kwargs.get("snn_hidden_dim", 32))
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode == "group_posture_guided":
            self.snn_fusion_gate_mode = "group_posture_guided"
            self.snn_group_gate = None
            self.snn_channel_gate = None
            gate_input_dim = hidden_dim * 6 + snn_hidden_dim * 3 + len(CONTEXT_NAMES)
            self.posture_group_gate = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.posture_group_gate[2].bias, 0.0)
        else:
            self.posture_group_gate = None

    @staticmethod
    def _peak_score(series: torch.Tensor) -> torch.Tensor:
        peak = series.amax(dim=1, keepdim=True)
        baseline = series.mean(dim=1, keepdim=True)
        spread = series.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return torch.sigmoid(((peak - baseline) / spread - 1.0) * 1.5)

    def extract_fall_event_score(self, features: torch.Tensor) -> torch.Tensor:
        event_series = features[:, :, 6]
        motion = features[:, :, 1]
        center_y = features[:, :, 2]
        soft_height = features[:, :, 4]
        aspect = features[:, :, 5]

        impact = self._peak_score(event_series)
        peak_motion = self._peak_score(motion)
        tail_size = max(1, features.size(1) // 4)

        head_center = center_y[:, :tail_size].mean(dim=1, keepdim=True)
        tail_center = center_y[:, -tail_size:].mean(dim=1, keepdim=True)
        center_drop = torch.relu(tail_center - head_center)
        center_drop_score = 1.0 - torch.exp(-8.0 * center_drop)

        head_height = soft_height[:, :tail_size].mean(dim=1, keepdim=True)
        tail_height = soft_height[:, -tail_size:].mean(dim=1, keepdim=True)
        height_reduction = 1.0 - torch.exp(-6.0 * torch.relu(head_height - tail_height))

        head_aspect = aspect[:, :tail_size].mean(dim=1, keepdim=True)
        tail_aspect = aspect[:, -tail_size:].mean(dim=1, keepdim=True)
        aspect_change = 1.0 - torch.exp(-3.0 * torch.relu(tail_aspect - head_aspect))
        posture_change = torch.maximum(height_reduction, aspect_change)

        low_position = torch.sigmoid((tail_center - 0.55) * 8.0)
        peak_motion_value = motion.amax(dim=1, keepdim=True).clamp_min(1e-6)
        tail_motion = motion[:, -tail_size:].mean(dim=1, keepdim=True)
        post_fall_stillness = torch.exp(-4.0 * tail_motion / peak_motion_value)
        ground_contact = ((center_drop_score + posture_change + low_position) / 3.0).clamp(0.0, 1.0)
        ground_hold = (ground_contact * post_fall_stillness).clamp(0.0, 1.0)
        confirmed_fall = (impact * ground_hold).clamp(0.0, 1.0)

        return torch.cat(
            [impact, peak_motion, center_drop_score, posture_change, ground_hold, confirmed_fall],
            dim=1,
        )

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode != "group_posture_guided":
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.posture_group_gate is None:
            raise RuntimeError("Posture-aware group gate is required.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        learned_gate = self.posture_group_gate(gate_context)

        impact = fall_event_score[:, 0:1]
        peak_motion = fall_event_score[:, 1:2]
        ground_hold = fall_event_score[:, 4:5]
        confirmed_fall = fall_event_score[:, 5:6]
        guided_gate = torch.cat(
            [
                0.35 * impact + 0.65 * ground_hold,
                0.70 * impact + 0.30 * peak_motion,
                0.25 * impact + 0.50 * ground_hold + 0.25 * confirmed_fall,
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        group_gate = 0.5 * learned_gate + 0.5 * guided_gate

        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        return snn_pooled * gate, gate


original_parse_args = base.parse_args


def posture_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = "group_posture_guided"
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
    config["snn_fusion_gate_mode"] = "group_posture_guided"
    config["group_gate_context"] = list(CONTEXT_NAMES)
    config["group_gate_guidance_mix"] = 0.5
    config["snn_adl_gate_penalty"] = 0.0
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def main() -> None:
    base.FallEventDetector = PostureGroupFallEventDetector
    base.parse_args = posture_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

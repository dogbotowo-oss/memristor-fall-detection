from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_ordered_group_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class OrderedGroupFallEventDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = "group_channel" if requested_mode == "group_ordered_event" else requested_mode
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode == "group_ordered_event":
            self.snn_fusion_gate_mode = "group_ordered_event"
            self.snn_channel_gate = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype)
        return (values * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    def extract_fall_event_score(self, features: torch.Tensor) -> torch.Tensor:
        event_series = features[:, :, 6]
        motion = features[:, :, 1]
        center_y = features[:, :, 2]
        soft_height = features[:, :, 4]
        aspect = features[:, :, 5]
        batch_size, time_steps = event_series.shape

        peak_index = torch.argmax(event_series, dim=1, keepdim=True)
        time_index = torch.arange(time_steps, device=features.device).view(1, -1).expand(batch_size, -1)
        pre_mask = time_index < peak_index
        post_mask = time_index > peak_index
        late_mask = time_index >= (peak_index + 2).clamp(max=time_steps - 1)

        peak = event_series.amax(dim=1, keepdim=True)
        baseline = event_series.mean(dim=1, keepdim=True)
        spread = event_series.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        impact_normalized = torch.sigmoid((((peak - baseline) / spread) - 1.0) * 1.5)
        impact_trigger = torch.sigmoid(
            (impact_normalized - self.snn_event_score_threshold) * self.snn_event_score_sharpness
        )
        impact = impact_normalized * impact_trigger

        pre_center = self._masked_mean(center_y, pre_mask)
        post_center = self._masked_mean(center_y, post_mask)
        center_drop = 1.0 - torch.exp(-8.0 * torch.relu(post_center - pre_center))

        pre_height = self._masked_mean(soft_height, pre_mask)
        post_height = self._masked_mean(soft_height, post_mask)
        height_reduction = 1.0 - torch.exp(-6.0 * torch.relu(pre_height - post_height))

        pre_aspect = self._masked_mean(aspect, pre_mask)
        post_aspect = self._masked_mean(aspect, post_mask)
        aspect_change = 1.0 - torch.exp(-3.0 * torch.relu(post_aspect - pre_aspect))
        posture_change = torch.maximum(height_reduction, aspect_change)

        peak_motion = motion.amax(dim=1, keepdim=True).clamp_min(1e-6)
        late_motion = self._masked_mean(motion, late_mask)
        post_stillness = torch.exp(-4.0 * late_motion / peak_motion)
        low_position = torch.sigmoid((post_center - 0.55) * 8.0)
        ground_hold = (
            ((center_drop + posture_change + low_position) / 3.0).clamp(0.0, 1.0)
            * post_stillness
        ).clamp(0.0, 1.0)

        valid_order = ((peak_index >= 2) & (peak_index <= time_steps - 4)).to(features.dtype)
        ordered_hold = torch.sqrt((center_drop * ground_hold).clamp_min(0.0))
        order_factor = (0.25 + 0.75 * ordered_hold) * (0.5 + 0.5 * valid_order)
        return (impact * order_factor).clamp(0.0, 1.0)

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode != "group_ordered_event":
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.snn_group_gate is None:
            raise RuntimeError("Group gate is required for ordered-event fusion.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        return snn_pooled * gate, gate


original_parse_args = base.parse_args


def ordered_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = "group_ordered_event"
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
    config["snn_fusion_gate_mode"] = "group_ordered_event"
    config["ordered_event_sequence"] = [
        "impact",
        "center_drop",
        "posture_change",
        "ground_hold",
        "post_stillness",
    ]
    config["snn_adl_gate_penalty"] = 0.0
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def main() -> None:
    base.FallEventDetector = OrderedGroupFallEventDetector
    base.parse_args = ordered_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
REFINED_MODE = "group"
REFINED_FLOOR = 0.20
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_gate_refine_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class RefinedFallEventDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = "group_channel" if requested_mode in {"group", "group_channel_residual"} else requested_mode
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode == "group":
            self.snn_fusion_gate_mode = "group"
            self.snn_channel_gate = None
        elif requested_mode == "group_channel_residual":
            self.snn_fusion_gate_mode = "group_channel_residual"
        self.snn_fusion_gate_floor = float(np.clip(REFINED_FLOOR, 0.0, 0.95))

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode not in {"group", "group_channel_residual"}:
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.snn_group_gate is None:
            raise RuntimeError("Group gate is required for refined grouped fusion.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        group_gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        if self.snn_fusion_gate_mode == "group":
            gate = group_gate
        else:
            if self.snn_channel_gate is None:
                raise RuntimeError("Channel gate is required for residual group-channel fusion.")
            channel_gate = self.snn_channel_gate(gate_context)
            original_gate = channel_gate * group_gate
            gate = self.snn_fusion_gate_floor + (1.0 - self.snn_fusion_gate_floor) * original_gate
        return snn_pooled * gate, gate


original_parse_args = base.parse_args


def refined_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = REFINED_MODE
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
    config["snn_fusion_gate_mode"] = REFINED_MODE
    config["snn_fusion_gate_floor"] = float(REFINED_FLOOR)
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def parse_runner_args() -> tuple[str, float]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--refined-gate-mode",
        choices=["group", "group_channel_residual"],
        required=True,
    )
    parser.add_argument("--refined-gate-floor", type=float, default=0.20)
    runner_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(runner_args.refined_gate_mode), float(runner_args.refined_gate_floor)


def main() -> None:
    global REFINED_MODE, REFINED_FLOOR
    REFINED_MODE, REFINED_FLOOR = parse_runner_args()
    base.FallEventDetector = RefinedFallEventDetector
    base.parse_args = refined_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

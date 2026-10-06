from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


BASE_CODE = Path(__file__).with_name("run_ordered_group_gate.py")
MODE = "group_only_ordered_residual"
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("ordered_residual_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ordered gate: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ordered = load_base_module()
BaseFallEventDetector = ordered.base.FallEventDetector
OrderedGroupFallEventDetector = ordered.OrderedGroupFallEventDetector


class OrderedResidualFallEventDetector(OrderedGroupFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "group_channel", **kwargs: Any) -> None:
        BaseFallEventDetector.__init__(self, *args, snn_fusion_gate_mode="group_channel", **kwargs)
        self.snn_fusion_gate_mode = str(snn_fusion_gate_mode)
        if self.snn_fusion_gate_mode == "group_only_ordered_residual":
            self.snn_channel_gate = None
        self.ordered_residual = nn.Linear(1, 3)
        nn.init.zeros_(self.ordered_residual.weight)
        nn.init.zeros_(self.ordered_residual.bias)

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_group_gate is None:
            raise RuntimeError("Group gate is required for ordered residual fusion.")
        gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        ordered_residual = 0.25 * torch.tanh(self.ordered_residual(fall_event_score))
        group_gate = torch.clamp(group_gate + ordered_residual, 0.0, 1.0)
        if self.snn_fusion_gate_mode == "group_only_ordered_residual":
            gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        elif self.snn_fusion_gate_mode == "group_channel_ordered_residual":
            if self.snn_channel_gate is None:
                raise RuntimeError("Channel gate is required for group-channel residual fusion.")
            channel_gate = self.snn_channel_gate(gate_context)
            gate = channel_gate * torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        else:
            raise RuntimeError(f"Unsupported ordered residual mode: {self.snn_fusion_gate_mode}")
        return snn_pooled * gate, gate


original_parse_args = ordered.base.parse_args


def parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = MODE
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
    config["snn_fusion_gate_mode"] = MODE
    config["ordered_residual_scale"] = 0.25
    checkpoint["config"] = config
    torch.save(checkpoint, LAST_ARGS.model_path)


def parse_runner_args() -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ordered-residual-mode",
        choices=["group_only_ordered_residual", "group_channel_ordered_residual"],
        required=True,
    )
    parser.add_argument("--ordered-residual-epochs", type=int, default=6)
    runner_args, remaining = parser.parse_known_args()
    global MODE
    MODE = str(runner_args.ordered_residual_mode)
    sys.argv = [sys.argv[0], *remaining]
    return MODE, int(runner_args.ordered_residual_epochs)


def main() -> None:
    global MODE
    MODE, residual_epochs = parse_runner_args()
    ordered.ResidualEpochs = residual_epochs
    ordered.base.FallEventDetector = OrderedResidualFallEventDetector
    ordered.base.parse_args = parse_args
    ordered.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

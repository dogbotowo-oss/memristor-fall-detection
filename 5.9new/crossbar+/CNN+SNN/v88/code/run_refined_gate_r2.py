from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
REFINED_MODE = "group_residual"
REFINED_FLOOR = 0.20
REFINED_GROUP_BIAS: float | None = None
REFINED_CHANNEL_BIAS: float | None = None
REFINED_DROP_FES_CONTEXT = False
REFINED_LATE_FUSION_LAMBDA = 0.2
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_gate_refine_r2_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class RefinedFallEventDetectorR2(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_fusion_gate_mode: str = "scalar", **kwargs: Any) -> None:
        requested_mode = str(snn_fusion_gate_mode)
        base_mode = (
            "group_channel"
            if requested_mode in {"group", "group_residual", "group_channel_residual"}
            else requested_mode
        )
        super().__init__(*args, snn_fusion_gate_mode=base_mode, **kwargs)
        if requested_mode in {"group", "group_residual"}:
            self.snn_fusion_gate_mode = requested_mode
            self.snn_channel_gate = None
        elif requested_mode == "group_channel_residual":
            self.snn_fusion_gate_mode = "group_channel_residual"
        self.snn_fusion_gate_floor = float(np.clip(REFINED_FLOOR, 0.0, 0.95))
        # S1: drop the (near-chance) fall_event_score from the gate context by
        # rebuilding the group gate with one fewer input feature.
        if REFINED_DROP_FES_CONTEXT and self.snn_group_gate is not None:
            old_gate = self.snn_group_gate
            in_dim = int(old_gate[0].in_features) - 1
            hidden = int(old_gate[0].out_features)
            out_dim = int(old_gate[2].out_features)
            self.snn_group_gate = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, out_dim),
                torch.nn.Sigmoid(),
            )
        # Binary ADL/Fall auxiliary head on pre-gate SNN features (option 2 of the revive combo).
        if self.snn_input is not None:
            snn_fusion_dim = int(self.snn_input[0].out_features) * 3
            self.snn_class_head = torch.nn.Linear(snn_fusion_dim, 2)
        else:
            self.snn_class_head = None
        # Optional init-bias overrides applied after base construction.
        if REFINED_GROUP_BIAS is not None and self.snn_group_gate is not None:
            torch.nn.init.constant_(self.snn_group_gate[2].bias, float(REFINED_GROUP_BIAS))
        if REFINED_CHANNEL_BIAS is not None and self.snn_channel_gate is not None:
            torch.nn.init.constant_(self.snn_channel_gate[2].bias, float(REFINED_CHANNEL_BIAS))

    def apply_snn_fusion_gate(
        self,
        pooled: torch.Tensor,
        snn_pooled: torch.Tensor,
        fall_event_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_fusion_gate_mode not in {"group", "group_residual", "group_channel_residual"}:
            return super().apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
        if self.snn_group_gate is None:
            raise RuntimeError("Group gate is required for refined grouped fusion.")
        if REFINED_DROP_FES_CONTEXT:
            gate_context = torch.cat([pooled, snn_pooled], dim=1)
        else:
            gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
        group_gate = self.snn_group_gate(gate_context)
        group_dim = snn_pooled.size(1) // 3
        if group_dim * 3 != snn_pooled.size(1):
            raise RuntimeError("SNN pooled dimension must be divisible by 3.")
        group_gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
        if self.snn_fusion_gate_mode == "group":
            gate = group_gate
        elif self.snn_fusion_gate_mode == "group_residual":
            gate = self.snn_fusion_gate_floor + (1.0 - self.snn_fusion_gate_floor) * group_gate
        else:
            if self.snn_channel_gate is None:
                raise RuntimeError("Channel gate is required for residual group-channel fusion.")
            channel_gate = self.snn_channel_gate(gate_context)
            original_gate = channel_gate * group_gate
            gate = self.snn_fusion_gate_floor + (1.0 - self.snn_fusion_gate_floor) * original_gate
        return snn_pooled * gate, gate

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, gate_info = super().forward_with_gate(x)
        lam = float(REFINED_LATE_FUSION_LAMBDA)
        if lam > 0.0 and self.snn_class_head is not None and "snn_pooled_pregate" in gate_info:
            # v88: late fusion adds the SNN class-head logits directly on the
            # {normal, fall} entries so the LIF pathway cannot be gated away.
            snn_aux = self.snn_class_head(gate_info["snn_pooled_pregate"])
            fused = logits.clone()
            fused[:, base.LABEL_NAME_TO_ID["normal"]] += lam * snn_aux[:, 0]
            fused[:, base.LABEL_NAME_TO_ID["fall"]] += lam * snn_aux[:, 1]
            gate_info["snn_class_logits"] = snn_aux
            return fused, gate_info
        return logits, gate_info


original_parse_args = base.parse_args


def refined_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_fusion_gate_mode = REFINED_MODE
    LAST_ARGS = parsed
    return parsed


def update_checkpoint_metadata() -> None:
    if LAST_ARGS is None:
        return
    paths = [LAST_ARGS.model_path]
    ema_path = LAST_ARGS.model_path.with_name(LAST_ARGS.model_path.stem + "_ema.pth")
    if ema_path.exists():
        paths.append(ema_path)
    for path in paths:
        if not path.exists():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        config = dict(checkpoint.get("config", {}))
        config["snn_fusion_gate_mode"] = REFINED_MODE
        config["snn_fusion_gate_floor"] = float(REFINED_FLOOR)
        config["refined_drop_fes_context"] = bool(REFINED_DROP_FES_CONTEXT)
        config["snn_late_fusion_lambda"] = float(REFINED_LATE_FUSION_LAMBDA)
        if REFINED_GROUP_BIAS is not None:
            config["refined_group_bias"] = float(REFINED_GROUP_BIAS)
        if REFINED_CHANNEL_BIAS is not None:
            config["refined_channel_bias"] = float(REFINED_CHANNEL_BIAS)
        checkpoint["config"] = config
        torch.save(checkpoint, path)


def parse_runner_args() -> tuple[str, float, float | None, float | None, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--refined-gate-mode",
        choices=["group", "group_residual", "group_channel_residual"],
        required=True,
    )
    parser.add_argument("--refined-gate-floor", type=float, default=0.20)
    parser.add_argument("--refined-group-bias", type=float, default=None)
    parser.add_argument("--refined-channel-bias", type=float, default=None)
    parser.add_argument("--refined-drop-fes-context", action="store_true")
    parser.add_argument("--refined-late-fusion-lambda", type=float, default=0.2)
    runner_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    group_bias = None if runner_args.refined_group_bias is None else float(runner_args.refined_group_bias)
    channel_bias = None if runner_args.refined_channel_bias is None else float(runner_args.refined_channel_bias)
    return (
        str(runner_args.refined_gate_mode),
        float(runner_args.refined_gate_floor),
        group_bias,
        channel_bias,
        bool(runner_args.refined_drop_fes_context),
        float(runner_args.refined_late_fusion_lambda),
    )


def main() -> None:
    global REFINED_MODE, REFINED_FLOOR, REFINED_GROUP_BIAS, REFINED_CHANNEL_BIAS, REFINED_DROP_FES_CONTEXT
    global REFINED_LATE_FUSION_LAMBDA
    (
        REFINED_MODE,
        REFINED_FLOOR,
        REFINED_GROUP_BIAS,
        REFINED_CHANNEL_BIAS,
        REFINED_DROP_FES_CONTEXT,
        REFINED_LATE_FUSION_LAMBDA,
    ) = parse_runner_args()
    base.FallEventDetector = RefinedFallEventDetectorR2
    base.parse_args = refined_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

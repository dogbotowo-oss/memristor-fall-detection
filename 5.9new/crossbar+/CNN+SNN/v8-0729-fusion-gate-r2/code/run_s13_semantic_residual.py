"""S13 binary ADL/Fall detector with semantic SNN residual fusion."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


BASE_CODE = Path(__file__).with_name("automatic_fall_detector.py")
SEMANTIC_DIM = 48
RESIDUAL_MAX_SCALE = 0.25
RESIDUAL_INIT_SCALE: float | None = None
DYNAMIC_ENCODER_MODE = "legacy"
DYNAMICS_MODE = "uniform"
PRESERVE_CHECKPOINT_DYNAMICS = False
LAST_ARGS: argparse.Namespace | None = None


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("s13_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import base detector: {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BaseFallEventDetector = base.FallEventDetector


class S13SemanticResidualDetector(BaseFallEventDetector):
    def __init__(self, *args: Any, snn_hidden_dim: int = 32, **kwargs: Any) -> None:
        if snn_hidden_dim % 2 != 0:
            raise ValueError("S13 requires an even snn_hidden_dim")
        if DYNAMIC_ENCODER_MODE == "multistream" and snn_hidden_dim % 4 != 0:
            raise ValueError("S14 multistream encoding requires snn_hidden_dim divisible by four")
        kwargs["num_classes"] = 2
        kwargs["snn_input_mode"] = (
            "scalar7_flow_kinematic" if DYNAMIC_ENCODER_MODE == "multistream" else "scalar7_flow"
        )
        kwargs["snn_pool_mode"] = "seg4x3"
        kwargs["snn_fusion_gate_mode"] = "group_channel"
        super().__init__(*args, snn_hidden_dim=snn_hidden_dim, **kwargs)
        self.snn_input = nn.Identity()
        self.snn_fusion_gate = None
        self.snn_group_gate = None
        self.snn_channel_gate = None
        self.snn_fusion_gate_mode = "s13_semantic_residual"
        self.snn_dynamic_encoder_mode = DYNAMIC_ENCODER_MODE
        if self.snn_dynamic_encoder_mode == "multistream":
            state_hidden = snn_hidden_dim // 4
            dynamic_hidden = snn_hidden_dim - state_hidden
            transient_hidden = dynamic_hidden // 2
            flow_hidden = dynamic_hidden - transient_hidden
            self.snn_transient_hidden = transient_hidden
            self.snn_flow_hidden = flow_hidden
            self.snn_dynamic_input = None
            self.snn_transient_input = nn.Sequential(
                nn.Linear(12, transient_hidden),
                nn.LayerNorm(transient_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(transient_hidden, transient_hidden),
            )
            self.snn_flow_input = nn.Sequential(
                nn.Linear(256, flow_hidden),
                nn.LayerNorm(flow_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(flow_hidden, flow_hidden),
            )
        else:
            dynamic_hidden = snn_hidden_dim // 2
            state_hidden = snn_hidden_dim - dynamic_hidden
            self.snn_dynamic_input = nn.Sequential(
                nn.Linear(131, dynamic_hidden),
                nn.LayerNorm(dynamic_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dynamic_hidden, dynamic_hidden),
            )
            self.snn_transient_input = None
            self.snn_flow_input = None
            self.snn_transient_hidden = 0
            self.snn_flow_hidden = 0
        self.snn_dynamic_hidden = dynamic_hidden
        self.snn_state_hidden = state_hidden
        self.snn_state_input = nn.Sequential(
            nn.Linear(4, state_hidden),
            nn.LayerNorm(state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(state_hidden, state_hidden),
        )
        self.snn_dynamic_compressor = nn.Sequential(
            nn.Linear(dynamic_hidden * 12, SEMANTIC_DIM),
            nn.LayerNorm(SEMANTIC_DIM),
            nn.ReLU(inplace=True),
        )
        self.snn_state_compressor = nn.Sequential(
            nn.Linear(state_hidden * 12, SEMANTIC_DIM),
            nn.LayerNorm(SEMANTIC_DIM),
            nn.ReLU(inplace=True),
        )
        self.snn_fusion_dim = SEMANTIC_DIM * 2
        main_dim = int(kwargs.get("hidden_dim", 64)) * 6
        self.snn_semantic_gate = nn.Sequential(
            nn.Linear(main_dim + self.snn_fusion_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.snn_semantic_gate[2].bias, 0.0)
        self.snn_to_main = nn.Linear(self.snn_fusion_dim, main_dim, bias=False)
        if RESIDUAL_INIT_SCALE is None:
            residual_scale_raw = -6.0
        else:
            initial_fraction = float(np.clip(RESIDUAL_INIT_SCALE / RESIDUAL_MAX_SCALE, 1e-4, 1.0 - 1e-4))
            residual_scale_raw = float(np.log(initial_fraction / (1.0 - initial_fraction)))
        self.snn_residual_scale_raw = nn.Parameter(torch.tensor(residual_scale_raw, dtype=torch.float32))
        self.head = nn.Sequential(
            nn.Linear(main_dim, int(kwargs.get("hidden_dim", 64)) * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(int(kwargs.get("hidden_dim", 64)) * 2, 2),
        )
        self.snn_class_head = nn.Linear(self.snn_fusion_dim, 2)
        self.temporal_aux_head = nn.Sequential(
            nn.Linear(self.snn_fusion_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )
        self.snn_dynamics_mode = DYNAMICS_MODE
        self._initialize_multiscale_dynamics()

    def _initialize_multiscale_dynamics(self) -> None:
        if (
            self.snn_dynamic_encoder_mode == "multistream"
            and self.snn_dynamics_mode == "multiscale"
            and self.snn_decay_logit is not None
            and self.snn_threshold_logit is not None
        ):
            decay_values = [0.55] * self.snn_transient_hidden
            decay_values += [0.70] * self.snn_flow_hidden
            decay_values += [0.92] * self.snn_state_hidden
            threshold_values = [0.28] * self.snn_transient_hidden
            threshold_values += [0.32] * self.snn_flow_hidden
            threshold_values += [0.45] * self.snn_state_hidden
            decay_logits = [
                self._bounded_logit(value, self.snn_decay_min, self.snn_decay_max) for value in decay_values
            ]
            threshold_logits = [
                self._bounded_logit(value, self.snn_threshold_min, self.snn_threshold_max)
                for value in threshold_values
            ]
            with torch.no_grad():
                self.snn_decay_logit.copy_(torch.tensor(decay_logits, dtype=self.snn_decay_logit.dtype))
                self.snn_threshold_logit.copy_(
                    torch.tensor(threshold_logits, dtype=self.snn_threshold_logit.dtype)
                )

    def on_initialization_checkpoint_loaded(self) -> None:
        if not PRESERVE_CHECKPOINT_DYNAMICS:
            self._initialize_multiscale_dynamics()
        if self.snn_dynamics_mode == "multiscale":
            decay, threshold = self._current_snn_dynamics(self.snn_decay_logit.view(1, 1, -1))
            transient_end = self.snn_transient_hidden
            flow_end = transient_end + self.snn_flow_hidden
            print(
                f"{'Preserved' if PRESERVE_CHECKPOINT_DYNAMICS else 'Reinitialized'} multiscale LIF after checkpoint load | "
                f"decay={float(decay[0, :transient_end].mean()):.3f}/"
                f"{float(decay[0, transient_end:flow_end].mean()):.3f}/"
                f"{float(decay[0, flow_end:].mean()):.3f} "
                f"threshold={float(threshold[0, :transient_end].mean()):.3f}/"
                f"{float(threshold[0, transient_end:flow_end].mean()):.3f}/"
                f"{float(threshold[0, flow_end:].mean()):.3f}",
                flush=True,
            )
    @staticmethod
    def _pool_group(spikes: torch.Tensor) -> torch.Tensor:
        segments = 4
        segment_length = spikes.size(1) // segments
        trimmed = spikes[:, : segments * segment_length, :]
        grouped = trimmed.reshape(trimmed.size(0), segments, segment_length, trimmed.size(2))
        return torch.cat(
            [
                grouped.mean(dim=2).reshape(trimmed.size(0), -1),
                grouped.max(dim=2).values.reshape(trimmed.size(0), -1),
                grouped[:, :, -1, :].reshape(trimmed.size(0), -1),
            ],
            dim=1,
        )

    def run_semantic_snn(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow = features[:, :, 7:135]
        flow_scale = flow.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        normalized_flow = (flow / flow_scale).clamp(-3.0, 3.0) / 3.0
        if self.snn_dynamic_encoder_mode == "multistream":
            flow_energy = flow.square().mean(dim=2, keepdim=True).sqrt().clamp(0.0, 1.0)
            transient_features = torch.cat(
                [features[:, :, 1:2], features[:, :, 3:4], features[:, :, 6:7], features[:, :, 135:143], flow_energy],
                dim=2,
            )
            dynamic_current = torch.cat(
                [self.snn_transient_input(transient_features), self.snn_flow_input(torch.cat([flow, normalized_flow], dim=2))],
                dim=2,
            )
        else:
            dynamic_features = torch.cat(
                [features[:, :, 1:2], features[:, :, 3:4], features[:, :, 6:7], normalized_flow],
                dim=2,
            )
            dynamic_current = self.snn_dynamic_input(dynamic_features)
        state_features = features[:, :, [0, 2, 4, 5]]
        current = torch.cat([dynamic_current, self.snn_state_input(state_features)], dim=2)
        decay, threshold = self._current_snn_dynamics(current)
        membrane = torch.zeros(current.size(0), current.size(2), device=current.device, dtype=current.dtype)
        spikes = []
        for time_index in range(current.size(1)):
            membrane = decay * membrane + current[:, time_index, :]
            spike = torch.sigmoid((membrane - threshold) * self.snn_surrogate_scale)
            membrane = membrane * (1.0 - spike.detach())
            spikes.append(spike)
        spike_train = torch.stack(spikes, dim=1)
        dynamic = self.snn_dynamic_compressor(self._pool_group(spike_train[:, :, : self.snn_dynamic_hidden]))
        state = self.snn_state_compressor(self._pool_group(spike_train[:, :, self.snn_dynamic_hidden :]))
        return dynamic, state

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, time_steps, channels, height, width = x.shape
        encoded = self.encoder(x.view(batch_size * time_steps, channels, height, width))
        encoded = self.proj(encoded.view(batch_size * time_steps, -1)).view(batch_size, time_steps, -1)
        temporal_out, _ = self.temporal(encoded)
        main_pooled = torch.cat(
            [temporal_out.mean(dim=1), temporal_out.max(dim=1).values, temporal_out[:, -1, :]],
            dim=1,
        )
        features = self.extract_temporal_event_features(x, mode=self.snn_input_mode)
        dynamic, state = self.run_semantic_snn(features)
        snn_pregate = torch.cat([dynamic, state], dim=1)
        semantic_gate = self.snn_semantic_gate(torch.cat([main_pooled, snn_pregate], dim=1))
        gated_snn = torch.cat(
            [dynamic * semantic_gate[:, 0:1], state * semantic_gate[:, 1:2]],
            dim=1,
        )
        residual_scale = RESIDUAL_MAX_SCALE * torch.sigmoid(self.snn_residual_scale_raw)
        residual_update = residual_scale * self.snn_to_main(gated_snn)
        fused = main_pooled + residual_update
        head_input_weight = self.head[0].weight.detach()
        head_input_bias = self.head[0].bias.detach()
        head_output_weight = self.head[3].weight.detach()
        head_output_bias = self.head[3].bias.detach()
        counterfactual_main = main_pooled.detach()
        counterfactual_gate = self.snn_semantic_gate(
            torch.cat([counterfactual_main, snn_pregate], dim=1)
        )
        counterfactual_gated_snn = torch.cat(
            [dynamic * counterfactual_gate[:, 0:1], state * counterfactual_gate[:, 1:2]],
            dim=1,
        )
        counterfactual_residual_update = residual_scale * self.snn_to_main(counterfactual_gated_snn)
        counterfactual_fused = counterfactual_main + counterfactual_residual_update
        counterfactual_main_hidden = torch.relu(
            torch.nn.functional.linear(counterfactual_main, head_input_weight, head_input_bias)
        )
        counterfactual_fused_hidden = torch.relu(
            torch.nn.functional.linear(counterfactual_fused, head_input_weight, head_input_bias)
        )
        counterfactual_main_logits = torch.nn.functional.linear(
            counterfactual_main_hidden, head_output_weight, head_output_bias
        )
        counterfactual_fused_logits = torch.nn.functional.linear(
            counterfactual_fused_hidden, head_output_weight, head_output_bias
        )
        snn_fall_margin_delta = (
            counterfactual_fused_logits[:, 1] - counterfactual_fused_logits[:, 0]
            - counterfactual_main_logits[:, 1]
            + counterfactual_main_logits[:, 0]
        )
        gate_info = {
            "snn_pooled_pregate": snn_pregate,
            "snn_pooled": gated_snn,
            "snn_gate_mean": semantic_gate.mean(dim=1, keepdim=True),
            "snn_semantic_gate": semantic_gate,
            "snn_residual_scale": residual_scale.reshape(1, 1).expand(batch_size, 1),
            "snn_residual_norm_ratio": (
                residual_update.norm(dim=1, keepdim=True) / main_pooled.norm(dim=1, keepdim=True).clamp_min(1e-6)
            ),
            "snn_fall_margin_delta": snn_fall_margin_delta,
            "snn_class_logits": self.snn_class_head(snn_pregate),
            "temporal_aux_logits": self.temporal_aux_head(snn_pregate),
        }
        return self.head(fused), gate_info


original_parse_args = base.parse_args


def s13_parse_args() -> argparse.Namespace:
    global LAST_ARGS
    parsed = original_parse_args()
    parsed.snn_input_mode = (
        "scalar7_flow_kinematic" if DYNAMIC_ENCODER_MODE == "multistream" else "scalar7_flow"
    )
    parsed.snn_pool_mode = "seg4x3"
    parsed.snn_fusion_gate_mode = "s13_semantic_residual"
    LAST_ARGS = parsed
    initial_scale = RESIDUAL_MAX_SCALE * float(torch.sigmoid(torch.tensor(-6.0))) if RESIDUAL_INIT_SCALE is None else RESIDUAL_INIT_SCALE
    print(
        "S13 semantic residual | classes=2 labels=normal/fall "
        f"semantic_dim={SEMANTIC_DIM} dynamic_encoder={DYNAMIC_ENCODER_MODE} dynamics={DYNAMICS_MODE} "
        f"residual_init={initial_scale:.6f} "
        f"residual_max={RESIDUAL_MAX_SCALE:.3f} "
        f"gate_margin_weight={float(parsed.snn_semantic_gate_margin_weight):.3f} "
        f"gate_margin={float(parsed.snn_semantic_gate_margin):.3f}",
        flush=True,
    )
    return parsed


def parse_s13_args() -> None:
    global SEMANTIC_DIM, RESIDUAL_MAX_SCALE, RESIDUAL_INIT_SCALE
    global DYNAMIC_ENCODER_MODE, DYNAMICS_MODE, PRESERVE_CHECKPOINT_DYNAMICS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--refined-gate-mode", default="group")
    parser.add_argument("--refined-gate-floor", type=float, default=0.0)
    parser.add_argument("--refined-group-bias", type=float, default=0.0)
    parser.add_argument("--refined-channel-bias", type=float, default=None)
    parser.add_argument("--refined-drop-fes-context", action="store_true")
    parser.add_argument("--refined-late-fusion-lambda", type=float, default=0.0)
    parser.add_argument("--s13-semantic-dim", type=int, default=48)
    parser.add_argument("--s13-residual-max-scale", type=float, default=0.25)
    parser.add_argument("--s13-residual-init-scale", type=float, default=None)
    parser.add_argument("--s13-dynamic-encoder-mode", choices=["legacy", "multistream"], default="legacy")
    parser.add_argument("--s13-dynamics-mode", choices=["uniform", "multiscale"], default="uniform")
    parser.add_argument("--s13-preserve-checkpoint-dynamics", action="store_true")
    args, remaining = parser.parse_known_args()
    if float(args.refined_late_fusion_lambda) != 0.0:
        raise ValueError("S13 uses one residual fusion path; late fusion must be disabled")
    SEMANTIC_DIM = max(8, int(args.s13_semantic_dim))
    RESIDUAL_MAX_SCALE = float(np.clip(args.s13_residual_max_scale, 0.01, 1.0))
    RESIDUAL_INIT_SCALE = (
        None
        if args.s13_residual_init_scale is None
        else float(np.clip(args.s13_residual_init_scale, 1e-5, RESIDUAL_MAX_SCALE * 0.99))
    )
    DYNAMIC_ENCODER_MODE = str(args.s13_dynamic_encoder_mode)
    DYNAMICS_MODE = str(args.s13_dynamics_mode)
    PRESERVE_CHECKPOINT_DYNAMICS = bool(args.s13_preserve_checkpoint_dynamics)
    if DYNAMICS_MODE == "multiscale" and DYNAMIC_ENCODER_MODE != "multistream":
        raise ValueError("Multiscale SNN dynamics require the multistream dynamic encoder")
    sys.argv = [sys.argv[0], *remaining]


def update_checkpoint_metadata() -> None:
    if LAST_ARGS is None:
        return
    paths = [LAST_ARGS.model_path, LAST_ARGS.model_path.with_name(LAST_ARGS.model_path.stem + "_ema.pth")]
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(checkpoint.get("config", {}))
        config.update(
            {
                "binary_adl_fall": True,
                "num_classes": 2,
                "snn_fusion_gate_mode": "s13_semantic_residual",
                "snn_semantic_groups": ["dynamic", "state"],
                "snn_semantic_dim_per_group": SEMANTIC_DIM,
                "snn_residual_max_scale": RESIDUAL_MAX_SCALE,
                "snn_residual_init_scale": RESIDUAL_INIT_SCALE,
                "snn_dynamic_encoder_mode": DYNAMIC_ENCODER_MODE,
                "snn_dynamics_mode": DYNAMICS_MODE,
                "snn_preserve_checkpoint_dynamics": PRESERVE_CHECKPOINT_DYNAMICS,
                "snn_multiscale_dynamics_initial": {
                    "transient": {"decay": 0.55, "threshold": 0.28},
                    "flow": {"decay": 0.70, "threshold": 0.32},
                    "state": {"decay": 0.92, "threshold": 0.45},
                } if DYNAMICS_MODE == "multiscale" else None,
                "snn_late_fusion_lambda": 0.0,
                "temporal_aux_attachment": "snn_pregate",
            }
        )
        checkpoint["config"] = config
        checkpoint["class_names"] = {0: "normal", 1: "fall"}
        torch.save(checkpoint, path)


def main() -> None:
    parse_s13_args()
    base.LABEL_NAME_TO_ID = {"normal": 0, "running": 0, "pre_fall": 1, "fall": 1}
    base.LABEL_ID_TO_NAME = {0: "normal", 1: "fall"}
    base.FallEventDetector = S13SemanticResidualDetector
    base.parse_args = s13_parse_args
    base.main()
    update_checkpoint_metadata()


if __name__ == "__main__":
    main()

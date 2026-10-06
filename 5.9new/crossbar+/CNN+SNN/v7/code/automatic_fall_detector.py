from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "fall_video.mp4"
DEFAULT_VIDEO_WINDOWS_NPZ = SCRIPT_DIR / "fall_video_windows.npz"
DEFAULT_VIDEO_LABELS_CSV = SCRIPT_DIR / "fall_video_frame_labels.csv"
DEFAULT_RUNNING_DIR = SCRIPT_DIR / "running"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_PICTURE_DIR = SCRIPT_DIR / "picture"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "image"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "automatic_fall_event_detector.pth"
DEFAULT_REPORT_JSON = SCRIPT_DIR / "automatic_fall_detector_report.json"
DEFAULT_PLOT_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_detector_plot.png"
DEFAULT_TRAINING_PLOT_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_training_curve.png"
DEFAULT_KEYFRAME_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_key_frame.png"
DEFAULT_PROCESS_DIR = DEFAULT_IMAGE_DIR / "automatic_fall_process_frames"
DEFAULT_PROCESS_OVERVIEW_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_process_overview.png"
DEFAULT_TEST_REPORT_JSON = SCRIPT_DIR / "automatic_fall_detector_test_report.json"
DEFAULT_TEST_PLOT_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_detector_test_plot.png"
DEFAULT_TEST_KEYFRAME_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_test_key_frame.png"
DEFAULT_TEST_PROCESS_DIR = DEFAULT_IMAGE_DIR / "automatic_fall_test_process_frames"
DEFAULT_TEST_PROCESS_OVERVIEW_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_test_process_overview.png"
DEFAULT_TEST_OUTPUT_DIR = DEFAULT_IMAGE_DIR / "automatic_fall_test_outputs"
DEFAULT_TEST_BATCH_SUMMARY_JSON = SCRIPT_DIR / "automatic_fall_detector_test_batch_summary.json"
DEFAULT_LTP_FILE = DEFAULT_DATA_DIR / "LTP-0.001-1MS.xls"
DEFAULT_PLASTICITY_PLOT_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_plasticity_analysis.png"
DEFAULT_PLASTICITY_REPORT_JSON = SCRIPT_DIR / "automatic_fall_plasticity_report.json"
DEFAULT_PLASTICITY_OVERVIEW_PATH = DEFAULT_IMAGE_DIR / "automatic_fall_plasticity_tracking_overview.png"
DEFAULT_NEXT_STAGE_PLAN_PATH = SCRIPT_DIR / "NEXT_STAGE_CONTINUOUS_FALL_PLAN.md"

LABEL_ID_TO_NAME = {
    0: "normal",
    1: "running",
    2: "pre_fall",
    3: "fall",
}
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}
LABEL_VISUAL_COLORS = {
    "normal": (205, 205, 205),
    "running": (239, 177, 74),
    "pre_fall": (242, 138, 95),
    "fall": (216, 73, 61),
}

NORMAL_XML_NAMES = {"normal", "no fall", "nofall", "nofall"}
FALL_XML_NAMES = {"fall"}


@dataclass
class AuxiliaryImageRecord:
    image_path: Path
    xml_path: Path
    label_id: int
    bbox: Optional[tuple[int, int, int, int]]


@dataclass
class TrainArtifacts:
    checkpoint_path: Path
    history: list[dict[str, float]]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    aux_summary: dict[str, object]
    running_summary: dict[str, object]
    external_labeled_summary: dict[str, object]
    external_val_labeled_summary: dict[str, object]
    external_no_fall_summary: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run a lightweight automatic fall detector from video windows + self-labeled images."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "train_infer", "test", "batch_test", "plasticity_analysis"],
        default="train_infer",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument(
        "--skip-main-video-training",
        action="store_true",
        help="Exclude the main video from training and validation; keep it only for inference.",
    )
    parser.add_argument("--test-video", type=Path, default=None)
    parser.add_argument("--test-video-dir", type=Path, default=None)
    parser.add_argument(
        "--assume-test-videos-no-fall",
        action="store_true",
        help="Treat external test videos as no-fall ADL videos and report false-alarm statistics.",
    )
    parser.add_argument(
        "--test-label-from-name",
        action="store_true",
        help="Use filename pattern to create video-level test labels: names containing the fall pattern are fall, others normal.",
    )
    parser.add_argument("--video-windows-npz", type=Path, default=DEFAULT_VIDEO_WINDOWS_NPZ)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_VIDEO_LABELS_CSV)
    parser.add_argument("--test-labels-csv", type=Path, default=None)
    parser.add_argument("--running-dir", type=Path, default=DEFAULT_RUNNING_DIR)
    parser.add_argument("--external-labeled-dir", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--external-val-labeled-dir",
        type=Path,
        nargs="+",
        default=None,
        help="Optional labeled video directories reserved for validation. Use this for subject/video-level validation splits.",
    )
    parser.add_argument("--external-fall-name-pattern", type=str, default="fall")
    parser.add_argument("--external-no-fall-dir", type=Path, default=None)
    parser.add_argument("--picture-dir", type=Path, default=DEFAULT_PICTURE_DIR)
    parser.add_argument("--aux-root", type=Path, default=None)
    parser.add_argument("--disable-auxiliary-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--init-model-path",
        type=Path,
        default=None,
        help="Optional checkpoint path used to initialize the detector before training.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--training-plot-path", type=Path, default=DEFAULT_TRAINING_PLOT_PATH)
    parser.add_argument("--keyframe-path", type=Path, default=DEFAULT_KEYFRAME_PATH)
    parser.add_argument("--process-dir", type=Path, default=DEFAULT_PROCESS_DIR)
    parser.add_argument("--process-overview-path", type=Path, default=DEFAULT_PROCESS_OVERVIEW_PATH)
    parser.add_argument("--test-report-json", type=Path, default=DEFAULT_TEST_REPORT_JSON)
    parser.add_argument("--test-plot-path", type=Path, default=DEFAULT_TEST_PLOT_PATH)
    parser.add_argument("--test-keyframe-path", type=Path, default=DEFAULT_TEST_KEYFRAME_PATH)
    parser.add_argument("--test-process-dir", type=Path, default=DEFAULT_TEST_PROCESS_DIR)
    parser.add_argument("--test-process-overview-path", type=Path, default=DEFAULT_TEST_PROCESS_OVERVIEW_PATH)
    parser.add_argument("--test-output-dir", type=Path, default=DEFAULT_TEST_OUTPUT_DIR)
    parser.add_argument("--test-batch-summary-json", type=Path, default=DEFAULT_TEST_BATCH_SUMMARY_JSON)
    parser.add_argument(
        "--plasticity-method",
        choices=["ltp", "ltp_decay", "stdp_ltp", "stdp_ltp_decay", "compare_all", "compare_ltp_decay"],
        default="compare_all",
    )
    parser.add_argument("--ltp-file", type=Path, default=DEFAULT_LTP_FILE)
    parser.add_argument("--plasticity-plot-path", type=Path, default=DEFAULT_PLASTICITY_PLOT_PATH)
    parser.add_argument("--plasticity-report-json", type=Path, default=DEFAULT_PLASTICITY_REPORT_JSON)
    parser.add_argument("--plasticity-overview-path", type=Path, default=DEFAULT_PLASTICITY_OVERVIEW_PATH)
    parser.add_argument("--next-stage-plan-path", type=Path, default=DEFAULT_NEXT_STAGE_PLAN_PATH)
    parser.add_argument("--decay-tau", type=float, default=8.0)
    parser.add_argument("--decay-floor", type=float, default=0.55)
    parser.add_argument("--decay-event-percentile", type=float, default=0.82)
    parser.add_argument("--stdp-a-plus", type=float, default=0.30)
    parser.add_argument("--stdp-a-minus", type=float, default=0.18)
    parser.add_argument("--stdp-tau-plus", type=float, default=5.0)
    parser.add_argument("--stdp-tau-minus", type=float, default=7.0)
    parser.add_argument("--running-suppression", type=float, default=0.85)
    parser.add_argument("--running-gate-margin", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--use-snn-temporal-branch",
        action="store_true",
        help="Enable a LIF-inspired temporal branch using motion/center/posture event features, then fuse it with the CNN-GRU stream.",
    )
    parser.add_argument(
        "--use-snn-fusion-gate",
        action="store_true",
        help="Enable a learnable gate to down-weight the SNN branch before final fusion.",
    )
    parser.add_argument(
        "--snn-fusion-gate-mode",
        choices=["scalar", "channel", "group_channel"],
        default="group_channel",
        help="Gate shape for the SNN branch: scalar uses one weight, channel uses per-channel weights, group_channel uses event-aware group and channel gates.",
    )
    parser.add_argument("--snn-hidden-dim", type=int, default=32)
    parser.add_argument("--snn-decay", type=float, default=0.82)
    parser.add_argument("--snn-threshold", type=float, default=0.55)
    parser.add_argument("--snn-surrogate-scale", type=float, default=8.0)
    parser.add_argument(
        "--learnable-snn-dynamics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learnable per-channel decay/threshold in the SNN temporal branch.",
    )
    parser.add_argument(
        "--freeze-cnn-gru-epochs",
        type=int,
        default=0,
        help="Freeze CNN encoder/projection/GRU for the first N epochs; train only SNN temporal branch and fusion head.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--video-window-stride", type=int, default=1)
    parser.add_argument("--running-window-stride", type=int, default=6)
    parser.add_argument("--max-running-videos", type=int, default=0)
    parser.add_argument("--external-labeled-stride", type=int, default=6)
    parser.add_argument("--max-external-labeled-videos", type=int, default=0)
    parser.add_argument("--external-no-fall-stride", type=int, default=6)
    parser.add_argument("--max-external-no-fall-videos", type=int, default=0)
    parser.add_argument("--use-pixel-human", action="store_true")
    parser.add_argument("--use-silhouette-human", action="store_true")
    parser.add_argument("--pixel-grid-size", type=int, default=20)
    parser.add_argument("--infer-stride", type=int, default=4)
    parser.add_argument("--fall-prob-threshold", type=float, default=0.0)
    parser.add_argument("--fall-margin-threshold", type=float, default=0.0)
    parser.add_argument("--min-fall-duration-sec", type=float, default=0.0)
    parser.add_argument("--use-event-shape-filter", action="store_true")  # 事件形态过滤
    parser.add_argument("--event-shape-pre-sec", type=float, default=1.0)
    parser.add_argument("--event-shape-post-sec", type=float, default=2.0)
    parser.add_argument("--event-shape-min-post-sec", type=float, default=1.0)
    parser.add_argument("--event-shape-short-end-max-duration-sec", type=float, default=1.5)
    parser.add_argument("--event-shape-short-end-min-post-sec", type=float, default=0.25)
    parser.add_argument("--event-shape-max-post-event-ratio", type=float, default=999.0)
    parser.add_argument("--event-shape-reject-insufficient-post-context", action="store_true")
    parser.add_argument("--event-shape-strong-peak-score", type=float, default=1.01)
    parser.add_argument("--event-shape-strong-mean-score", type=float, default=1.01)
    parser.add_argument("--event-shape-start-grace-sec", type=float, default=0.75)
    parser.add_argument("--event-shape-boundary-max-duration-sec", type=float, default=0.0)
    parser.add_argument("--video-source-weight", type=float, default=2.6)
    parser.add_argument("--external-no-fall-weight", type=float, default=0.35)
    parser.add_argument("--max-aux-fall", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--teacher-student-noise",
        action="store_true",
        help="Use EMA teacher supervision with noisy student inputs during training.",
    )
    parser.add_argument("--student-noise-std", type=float, default=0.035)
    parser.add_argument("--student-noise-prob", type=float, default=0.75)
    parser.add_argument("--student-frame-drop-prob", type=float, default=0.08)
    parser.add_argument(
        "--device-noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for device/hybrid perturbations; with hybrid_vp this is the device mix ratio lambda clamped to [0.0, 1.0].",
    )
    parser.add_argument(
        "--crossbar-readout-noise-scale",
        type=float,
        default=0.0,
        help="Measured-device-inspired Crossbar readout nonideality scale. Applies row/column gain drift and read noise after conductance mapping.",
    )
    parser.add_argument(
        "--hard-negative-normal-weight",
        type=float,
        default=0.0,
        help="Extra sampler weight for normal ADL windows that look close to falls, such as low posture or high motion windows.",
    )
    parser.add_argument(
        "--hard-negative-score-mode",
        choices=["raw", "quantile"],
        default="quantile",
        help="Hard-negative score shaping before sampling boost. quantile focuses on the highest-risk ADL windows by rank.",
    )
    parser.add_argument(
        "--hard-negative-min-percentile",
        type=float,
        default=0.70,
        help="Lower percentile cutoff for ADL hard-negative boosting. Windows below this cutoff keep the normal sampling weight.",
    )
    parser.add_argument(
        "--hard-negative-score-gamma",
        type=float,
        default=1.0,
        help="Nonlinear hard-negative score shaping. Higher values concentrate the boost on the very hardest ADL windows.",
    )
    parser.add_argument(
        "--hard-negative-max-multiplier",
        type=float,
        default=4.0,
        help="Maximum total sampler multiplier for a hard-negative ADL window. Values are clamped to [1.0, 4.0].",
    )
    parser.add_argument(
        "--adl-context-path-keyword",
        type=str,
        default="",
        help="Only external ADL videos whose path contains this keyword receive long-segment ADL context duplication.",
    )
    parser.add_argument(
        "--adl-context-min-percentile",
        type=float,
        default=0.80,
        help="Percentile used to select the hardest ADL windows before expanding their temporal context.",
    )
    parser.add_argument(
        "--adl-context-radius-windows",
        type=int,
        default=0,
        help="Neighbor radius around each selected ADL hard-negative window for long-segment context duplication.",
    )
    parser.add_argument(
        "--adl-context-extra-copies",
        type=int,
        default=0,
        help="How many extra times to duplicate selected ADL context windows into the training set.",
    )
    parser.add_argument(
        "--normal-positive-penalty",
        type=float,
        default=0.0,
        help="Additional training penalty that discourages pre_fall/fall probability on normal ADL samples.",
    )
    parser.add_argument("--teacher-ema", type=float, default=0.996)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--consistency-ramp-epochs", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--disable-crossbar", action="store_true")
    parser.add_argument(
        "--use-device-dynamics",
        action="store_true",
        help="Apply measured EPSC/PPF/LTP/LTD device response curves inside the Crossbar input mapping.",
    )
    parser.add_argument(
        "--student-noise-mode",
        choices=["gaussian", "device", "hybrid", "hybrid_vp"],
        default="gaussian",
        help="Noise model used with --teacher-student-noise: gaussian is the legacy noise, device adds measured device-response perturbations, hybrid directly adds gaussian noise and a weak device profile, hybrid_vp uses variance-preserving gaussian/device mixing.",
    )
    parser.add_argument(
        "--skip-visual-outputs",
        action="store_true",
        help="Skip per-video plots, key-frame images and process-frame visualizations during large batch experiments.",
    )
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--save-plot", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_xml_label(label: str) -> Optional[int]:
    cleaned = re.sub(r"\s+", " ", (label or "").strip().lower().replace("_", " "))
    if cleaned in FALL_XML_NAMES:
        return LABEL_NAME_TO_ID["fall"]
    if cleaned in NORMAL_XML_NAMES:
        return LABEL_NAME_TO_ID["normal"]
    return None


def find_conductance_file(data_dir: Path, stem_name: str) -> Path:
    for suffix in (".csv", ".xlsx", ".xls"):
        candidate = data_dir / f"{stem_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing conductance file for {stem_name} in {data_dir}")


def load_conductance_values(file_path: Path) -> np.ndarray:
    values: list[float] = []
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 3 and parts[0] == "DataValue":
                try:
                    value = float(parts[2])
                except ValueError:
                    continue
                if value > 0:
                    values.append(value)
    elif suffix in {".xls", ".xlsx"}:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            if df.empty:
                continue
            picked = False
            for column_idx in range(min(df.shape[1], 3)):
                column = pd.to_numeric(df.iloc[:, column_idx], errors="coerce").dropna()
                column = column[column > 0]
                if not column.empty:
                    values.extend(column.astype(float).tolist())
                    picked = True
                    break
            if not picked:
                numeric_df = df.apply(pd.to_numeric, errors="coerce")
                flat = numeric_df.to_numpy().reshape(-1)
                flat = flat[np.isfinite(flat)]
                flat = flat[flat > 0]
                values.extend(flat.astype(float).tolist())
    else:
        raise ValueError(f"Unsupported conductance format: {file_path}")

    if not values:
        raise ValueError(f"No positive conductance values found in {file_path}")
    return np.asarray(values, dtype=np.float32)


def load_conductance_pair(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        hrs_path = find_conductance_file(data_dir, "HRS")
        lrs_path = find_conductance_file(data_dir, "LRS")
        hrs = load_conductance_values(hrs_path)
        lrs = load_conductance_values(lrs_path)
        scale = max(float(hrs.max()), float(lrs.max()), 1e-12)
        hrs_norm = np.clip(hrs / scale, 0.0, 1.0)
        lrs_norm = np.clip(lrs / scale, 0.0, 1.0)
        log(f"Loaded conductance data: HRS={len(hrs_norm)}, LRS={len(lrs_norm)}")
        return hrs_norm, lrs_norm
    except Exception as exc:
        log(f"Conductance loading failed, using fallback values. Reason: {exc}")
        hrs_fallback = np.array([0.08, 0.10, 0.12, 0.15], dtype=np.float32)
        lrs_fallback = np.array([0.70, 0.78, 0.86, 0.92], dtype=np.float32)
        return hrs_fallback, lrs_fallback


def find_first_existing_data_file(data_dir: Path, preferred_names: list[str], glob_pattern: str) -> Path:
    for file_name in preferred_names:
        candidate = data_dir / file_name
        if candidate.exists():
            return candidate
    matches = sorted(data_dir.glob(glob_pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No data file found for pattern {glob_pattern} in {data_dir}")


def normalize_device_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.ones(1, dtype=np.float32)
    values = np.abs(values)
    if len(values) >= 5:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        values = np.convolve(values, kernel, mode="same").astype(np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max - value_min <= 1e-12:
        return np.ones_like(values, dtype=np.float32)
    return np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0).astype(np.float32)


def load_device_response_curve(file_path: Path, accumulate: bool = False) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        excel_file = pd.ExcelFile(file_path)
    run_sheet = next((name for name in excel_file.sheet_names if str(name).lower().startswith("run")), excel_file.sheet_names[0])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = pd.read_excel(file_path, sheet_name=run_sheet)
    if "DrainI" not in df.columns:
        raise ValueError(f"Device response file {file_path} missing DrainI column.")
    drain_i = pd.to_numeric(df["DrainI"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    curve = normalize_device_curve(drain_i)
    if accumulate:
        curve = np.maximum.accumulate(curve).astype(np.float32)
    return curve


def load_device_dynamics(data_dir: Path) -> Optional[dict[str, object]]:
    try:
        source_files = {
            "epsc": find_first_existing_data_file(
                data_dir,
                ["EPSC-0.001-60-60.xls", "EPSC-0.002-60-60.xls"],
                "EPSC*.xls",
            ),
            "ppf": find_first_existing_data_file(
                data_dir,
                ["PPF-0.001-60-60.xls", "PPF-0.0015-60-60.xls"],
                "PPF*.xls",
            ),
            "ltp": find_first_existing_data_file(
                data_dir,
                ["LTP-0.001-1MS.xls", "LTP-0.001-30-60.xls"],
                "LTP*.xls",
            ),
            "ltd": find_first_existing_data_file(
                data_dir,
                ["LTD-0.02-1MS.xls"],
                "LTD*.xls",
            ),
        }
        dynamics: dict[str, object] = {
            "epsc_curve": load_device_response_curve(source_files["epsc"]),
            "ppf_curve": load_device_response_curve(source_files["ppf"]),
            "ltp_curve": load_device_response_curve(source_files["ltp"], accumulate=True),
            "ltd_curve": load_device_response_curve(source_files["ltd"]),
            "source_files": {key: str(value) for key, value in source_files.items()},
        }
        log(
            "Loaded device dynamics: "
            + ", ".join(f"{key}={Path(value).name}" for key, value in dynamics["source_files"].items())
        )
        return dynamics
    except Exception as exc:
        log(f"Device dynamics loading failed; continuing without measured dynamics. Reason: {exc}")
        return None


def build_device_noise_profile(device_dynamics: Optional[dict[str, object]], length: int = 64) -> Optional[np.ndarray]:
    if device_dynamics is None:
        return None
    epsc = resample_curve_to_frames(np.asarray(device_dynamics["epsc_curve"], dtype=np.float32), length)
    ppf = resample_curve_to_frames(np.asarray(device_dynamics["ppf_curve"], dtype=np.float32), length)
    ltp = resample_curve_to_frames(np.asarray(device_dynamics["ltp_curve"], dtype=np.float32), length)
    ltd = resample_curve_to_frames(np.asarray(device_dynamics["ltd_curve"], dtype=np.float32), length)
    profile = 0.34 * epsc + 0.26 * ppf + 0.24 * ltp - 0.16 * ltd
    return normalize_device_curve(profile)


def apply_consistent_augmentation(window_frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-0.05, 0.05))
    shift_y = int(rng.integers(-4, 5))
    shift_x = int(rng.integers(-4, 5))
    noise_scale = float(rng.uniform(0.0, 0.015))

    augmented = []
    for frame in window_frames:
        updated = np.clip(alpha * frame + beta, 0.0, 1.0)
        updated = np.roll(updated, shift=(shift_y, shift_x), axis=(0, 1))
        if noise_scale > 0:
            updated = np.clip(updated + rng.normal(0.0, noise_scale, size=updated.shape), 0.0, 1.0)
        augmented.append(updated.astype(np.float32))
    return np.stack(augmented, axis=0)


def build_static_temporal_window(
    image: np.ndarray,
    window_size: int,
    rng: np.random.Generator,
    augment: bool,
) -> np.ndarray:
    if not augment:
        return np.repeat(image[np.newaxis, :, :], window_size, axis=0).astype(np.float32)

    x_amplitude = int(rng.integers(-3, 4))
    y_amplitude = int(rng.integers(-3, 4))
    noise_scale = float(rng.uniform(0.0, 0.012))
    brightness = float(rng.uniform(0.96, 1.04))
    frames = []
    for index in range(window_size):
        phase = 0.0 if window_size == 1 else index / (window_size - 1)
        shift_x = int(round(x_amplitude * math.sin(math.pi * phase)))
        shift_y = int(round(y_amplitude * math.cos(math.pi * phase) * 0.5))
        frame = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
        frame = np.clip(frame * brightness, 0.0, 1.0)
        if noise_scale > 0:
            frame = np.clip(frame + rng.normal(0.0, noise_scale, size=frame.shape), 0.0, 1.0)
        frames.append(frame.astype(np.float32))
    return np.stack(frames, axis=0)


class EPSCModel:
    def __init__(self, tau: float = 0.75, baseline: float = 0.25) -> None:
        self.tau = tau
        self.baseline = baseline
        self.state = baseline

    def step(self, signal: float) -> float:
        self.state = self.baseline + self.tau * self.state + (1.0 - self.tau) * signal
        return self.state


class PPFModel:
    def __init__(self, strength: float = 0.25, threshold: float = 0.05) -> None:
        self.strength = strength
        self.threshold = threshold
        self.prev_signal = 0.0

    def step(self, signal: float) -> float:
        delta = abs(signal - self.prev_signal)
        factor = 1.0 + self.strength * np.tanh(delta / max(self.threshold, 1e-6))
        self.prev_signal = signal
        return float(factor)


def image_to_crossbar_array(
    image: np.ndarray,
    hrs_values: np.ndarray,
    lrs_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    foreground = 1.0 - np.asarray(image, dtype=np.float32)
    threshold = max(0.08, float(np.quantile(foreground, 0.75)))

    crossbar = np.empty_like(foreground, dtype=np.float32)
    low_mask = foreground < threshold
    high_mask = ~low_mask

    if np.any(low_mask):
        hrs_idx = rng.integers(0, len(hrs_values), size=int(low_mask.sum()))
        crossbar[low_mask] = hrs_values[hrs_idx]
    if np.any(high_mask):
        lrs_idx = rng.integers(0, len(lrs_values), size=int(high_mask.sum()))
        crossbar[high_mask] = lrs_values[lrs_idx]

    crossbar = np.clip(crossbar + foreground * 0.12, 0.0, 1.0)
    crossbar = np.clip(crossbar + rng.normal(0.0, 0.01, size=crossbar.shape), 0.0, 1.0)
    return crossbar.astype(np.float32)


def build_crossbar_window(
    window_frames: np.ndarray,
    hrs_values: np.ndarray,
    lrs_values: np.ndarray,
    seed: int,
    device_dynamics: Optional[dict[str, object]] = None,
    readout_noise_scale: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    epsc = EPSCModel()
    ppf = PPFModel()
    if device_dynamics is not None:
        num_frames = int(len(window_frames))
        epsc_curve = resample_curve_to_frames(np.asarray(device_dynamics["epsc_curve"], dtype=np.float32), num_frames)
        ppf_curve = resample_curve_to_frames(np.asarray(device_dynamics["ppf_curve"], dtype=np.float32), num_frames)
        ltp_curve = resample_curve_to_frames(np.asarray(device_dynamics["ltp_curve"], dtype=np.float32), num_frames)
        ltd_curve = resample_curve_to_frames(np.asarray(device_dynamics["ltd_curve"], dtype=np.float32), num_frames)
    else:
        epsc_curve = ppf_curve = ltp_curve = ltd_curve = None
    mapped_frames = []
    prev_signal = float(np.mean(window_frames[0])) if len(window_frames) > 0 else 0.0
    ltp_state = 0.0
    for frame_index, frame in enumerate(window_frames):
        crossbar = image_to_crossbar_array(frame, hrs_values, lrs_values, rng)
        signal = float(np.mean(frame))
        if device_dynamics is None:
            epsc_gain = epsc.step(signal)
            ppf_gain = ppf.step(signal)
            weighted = np.clip(crossbar * (0.75 + epsc_gain) * ppf_gain, 0.0, 1.0)
        else:
            motion_delta = abs(signal - prev_signal)
            ltp_state = max(ltp_state * 0.92, signal)
            epsc_gain = 0.18 + 0.55 * float(epsc_curve[frame_index])
            ppf_gain = 1.0 + 0.16 * float(ppf_curve[frame_index]) + 0.28 * math.tanh(motion_delta / 0.04)
            ltp_gain = 0.92 + 0.20 * float(ltp_curve[frame_index]) * float(np.clip(ltp_state, 0.0, 1.0))
            calmness = 1.0 - min(1.0, motion_delta / 0.08)
            ltd_scale = 1.0 - 0.14 * float(ltd_curve[frame_index]) * calmness
            device_jitter = rng.normal(1.0, 0.006, size=crossbar.shape)
            weighted = np.clip(crossbar * (0.75 + epsc_gain) * ppf_gain * ltp_gain * ltd_scale * device_jitter, 0.0, 1.0)
        scale = float(np.clip(readout_noise_scale, 0.0, 2.0))
        if scale > 0:
            if device_dynamics is None:
                dynamic_factor = 1.0
            else:
                dynamic_factor = (
                    0.55
                    + 0.35 * float(epsc_curve[frame_index])
                    + 0.20 * float(ppf_curve[frame_index])
                    + 0.15 * float(ltp_curve[frame_index])
                    + 0.10 * float(ltd_curve[frame_index])
                )
            row_sigma = 0.006 * scale * dynamic_factor
            col_sigma = 0.006 * scale * dynamic_factor
            read_sigma = 0.004 * scale * dynamic_factor
            row_gain = rng.normal(1.0, row_sigma, size=(weighted.shape[0], 1))
            col_gain = rng.normal(1.0, col_sigma, size=(1, weighted.shape[1]))
            read_noise = rng.normal(0.0, read_sigma, size=weighted.shape)
            weighted = np.clip(weighted * row_gain * col_gain + read_noise, 0.0, 1.0)
        prev_signal = signal
        mapped_frames.append(weighted.astype(np.float32))
    return np.stack(mapped_frames, axis=0)


def compute_adl_hard_negative_score(window_frames: np.ndarray) -> float:
    frames = np.asarray(window_frames, dtype=np.float32)
    if frames.ndim != 3 or len(frames) == 0:
        return 0.0

    foreground = np.clip(1.0 - frames, 0.0, 1.0)
    total_foreground = float(foreground.sum())
    if total_foreground <= 1e-6:
        return 0.0

    mass = foreground.sum(axis=(1, 2)) + 1e-6
    height = int(foreground.shape[1])
    width = int(foreground.shape[2])
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(1, height, 1)
    center_y_series = (foreground * y_coords).sum(axis=(1, 2)) / mass
    center_y = float(center_y_series.mean())

    bottom_start = max(0, int(height * 0.62))
    bottom_ratio = float(foreground[:, bottom_start:, :].sum() / max(total_foreground, 1e-6))
    if len(frames) > 1:
        motion = float(np.mean(np.abs(np.diff(frames, axis=0))))
    else:
        motion = 0.0
    motion_score = min(1.0, motion / 0.08)

    bbox_heights: list[float] = []
    bbox_aspects: list[float] = []
    bbox_centers: list[float] = []
    frame_mass_threshold = max(float(np.percentile(mass, 30)) * 0.18, 1e-5)
    for fg, frame_mass in zip(foreground, mass):
        if float(frame_mass) <= frame_mass_threshold:
            continue
        threshold = max(0.06, float(np.quantile(fg, 0.82)) * 0.35)
        mask = fg > threshold
        if int(mask.sum()) < 8:
            continue
        ys, xs = np.where(mask)
        box_h = (float(ys.max() - ys.min() + 1) / max(float(height), 1.0))
        box_w = (float(xs.max() - xs.min() + 1) / max(float(width), 1.0))
        bbox_heights.append(float(np.clip(box_h, 0.0, 1.0)))
        bbox_aspects.append(float(np.clip(box_w / max(box_h, 1e-3), 0.0, 4.0) / 4.0))
        bbox_centers.append(float(np.clip((float(ys.min() + ys.max()) * 0.5) / max(float(height - 1), 1.0), 0.0, 1.0)))

    if len(bbox_heights) >= 2:
        heights = np.asarray(bbox_heights, dtype=np.float32)
        aspects = np.asarray(bbox_aspects, dtype=np.float32)
        centers = np.asarray(bbox_centers, dtype=np.float32)
        height_change = float(np.percentile(heights, 90) - np.percentile(heights, 10))
        aspect_change = float(np.percentile(aspects, 90) - np.percentile(aspects, 10))
        posture_speed = float(np.mean(np.abs(np.diff(aspects))) + 0.65 * np.mean(np.abs(np.diff(centers))))
    else:
        height_change = 0.0
        aspect_change = 0.0
        posture_speed = 0.0

    height_change_score = min(1.0, height_change / 0.36)
    aspect_change_score = min(1.0, aspect_change / 0.22)
    posture_speed_score = min(1.0, posture_speed / 0.055)

    score = (
        0.34 * center_y
        + 0.25 * bottom_ratio
        + 0.13 * motion_score
        + 0.10 * height_change_score
        + 0.10 * aspect_change_score
        + 0.08 * posture_speed_score
    )
    return float(np.clip(score, 0.0, 1.0))


def path_matches_keyword(video_path: Path, keyword: str) -> bool:
    normalized_keyword = str(keyword or "").replace("\\", "/").strip().lower()
    if not normalized_keyword:
        return False
    normalized_path = str(video_path).replace("\\", "/").lower()
    return normalized_keyword in normalized_path


def select_adl_context_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    min_percentile: float,
    context_radius_windows: int,
    min_keep: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(windows) == 0:
        return (
            np.empty((0,) + windows.shape[1:], dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"enabled": False, "original_windows": 0, "kept_windows": 0},
        )
    unique_labels = {int(label) for label in labels.tolist()}
    if unique_labels != {LABEL_NAME_TO_ID["normal"]}:
        return (
            np.empty((0,) + windows.shape[1:], dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                "enabled": False,
                "reason": "labels_not_pure_adl",
                "original_windows": int(len(windows)),
                "kept_windows": 0,
            },
        )

    scores = np.asarray([compute_adl_hard_negative_score(window) for window in windows], dtype=np.float32)
    percentile = float(np.clip(min_percentile, 0.0, 0.95))
    threshold = float(np.quantile(scores, percentile)) if len(scores) else 0.0
    keep_mask = scores >= threshold
    min_keep = max(1, min(int(min_keep), len(windows)))
    if int(keep_mask.sum()) < min_keep:
        top_indices = np.argsort(scores, kind="mergesort")[-min_keep:]
        keep_mask = np.zeros((len(windows),), dtype=bool)
        keep_mask[top_indices] = True

    core_kept_windows = int(keep_mask.sum())
    context_radius = max(0, int(context_radius_windows))
    if context_radius > 0 and core_kept_windows > 0:
        expanded_mask = keep_mask.copy()
        for index in np.flatnonzero(keep_mask):
            start = max(0, int(index) - context_radius)
            end = min(len(windows), int(index) + context_radius + 1)
            expanded_mask[start:end] = True
        keep_mask = expanded_mask

    selected_windows = windows[keep_mask].astype(np.float32)
    selected_labels = labels[keep_mask].astype(np.int64)
    return (
        selected_windows,
        selected_labels,
        {
            "enabled": True,
            "original_windows": int(len(windows)),
            "kept_windows": int(keep_mask.sum()),
            "core_kept_windows": core_kept_windows,
            "context_radius_windows": context_radius,
            "context_added_windows": int(keep_mask.sum()) - core_kept_windows,
            "score_threshold": threshold,
            "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "score_top10_mean": float(np.mean(sorted(scores.tolist())[-min(10, len(scores)):])) if len(scores) else 0.0,
            "min_percentile": percentile,
        },
    )

def discover_aux_dataset_root(explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        if not explicit_root.exists():
            raise FileNotFoundError(f"Auxiliary dataset root does not exist: {explicit_root}")
        return explicit_root

    preferred = SCRIPT_DIR / "鑷爣鏁版嵁闆?fall-nofall)"
    if preferred.exists():
        return preferred

    best_root: Optional[Path] = None
    best_score = 0
    for child in SCRIPT_DIR.iterdir():
        if not child.is_dir():
            continue
        jpg_count = 0
        xml_count = 0
        for _, _, files in os.walk(child):
            jpg_count += sum(
                1 for file_name in files if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            xml_count += sum(1 for file_name in files if file_name.lower().endswith(".xml"))
        score = min(jpg_count, xml_count)
        if score > best_score:
            best_score = score
            best_root = child

    if best_root is None or best_score == 0:
        raise FileNotFoundError("Unable to find a self-labeled image+xml dataset under the project directory.")
    return best_root


def find_best_media_subdirs(root: Path) -> tuple[Path, Path]:
    best_image_dir: Optional[Path] = None
    best_xml_dir: Optional[Path] = None
    best_image_count = 0
    best_xml_count = 0

    for current, _, files in os.walk(root):
        current_path = Path(current)
        image_count = sum(1 for file_name in files if file_name.lower().endswith((".jpg", ".jpeg", ".png")))
        xml_count = sum(1 for file_name in files if file_name.lower().endswith(".xml"))
        if image_count > best_image_count:
            best_image_count = image_count
            best_image_dir = current_path
        if xml_count > best_xml_count:
            best_xml_count = xml_count
            best_xml_dir = current_path

    if best_image_dir is None or best_xml_dir is None:
        raise FileNotFoundError(f"Unable to locate image/xml subdirectories inside {root}")
    return best_image_dir, best_xml_dir


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
    if not boxes:
        return None
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return x1, y1, x2, y2


def parse_bndbox(node: ET.Element) -> Optional[tuple[int, int, int, int]]:
    try:
        xmin = int(float(node.findtext("xmin", "0")))
        ymin = int(float(node.findtext("ymin", "0")))
        xmax = int(float(node.findtext("xmax", "0")))
        ymax = int(float(node.findtext("ymax", "0")))
    except ValueError:
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def load_auxiliary_records(aux_root: Path) -> tuple[list[AuxiliaryImageRecord], dict[str, object]]:
    image_dir, xml_dir = find_best_media_subdirs(aux_root)

    image_index: dict[str, Path] = {}
    for current, _, files in os.walk(image_dir):
        current_path = Path(current)
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_index[Path(file_name).stem] = current_path / file_name

    records: list[AuxiliaryImageRecord] = []
    summary_counter = Counter()

    for current, _, files in os.walk(xml_dir):
        current_path = Path(current)
        for file_name in files:
            if not file_name.lower().endswith(".xml"):
                continue
            xml_path = current_path / file_name
            summary_counter["xml_total"] += 1

            if xml_path.stat().st_size == 0:
                summary_counter["xml_empty"] += 1
                continue

            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                summary_counter["xml_parse_error"] += 1
                continue

            root = tree.getroot()
            filename_text = Path((root.findtext("filename") or "").strip()).stem
            stem = filename_text or xml_path.stem
            image_path = image_index.get(stem)
            if image_path is None:
                summary_counter["missing_image"] += 1
                continue

            fall_boxes: list[tuple[int, int, int, int]] = []
            normal_boxes: list[tuple[int, int, int, int]] = []

            for obj in root.findall(".//object"):
                label_id = normalize_xml_label(obj.findtext("name", ""))
                if label_id is None:
                    summary_counter["unknown_object_label"] += 1
                    continue
                bndbox = obj.find("bndbox")
                parsed_box = parse_bndbox(bndbox) if bndbox is not None else None
                if label_id == LABEL_NAME_TO_ID["fall"] and parsed_box is not None:
                    fall_boxes.append(parsed_box)
                elif label_id == LABEL_NAME_TO_ID["normal"] and parsed_box is not None:
                    normal_boxes.append(parsed_box)

            if fall_boxes:
                chosen_label = LABEL_NAME_TO_ID["fall"]
                chosen_box = union_boxes(fall_boxes)
            elif normal_boxes:
                chosen_label = LABEL_NAME_TO_ID["normal"]
                chosen_box = union_boxes(normal_boxes)
            else:
                summary_counter["no_supported_objects"] += 1
                continue

            records.append(
                AuxiliaryImageRecord(
                    image_path=image_path,
                    xml_path=xml_path,
                    label_id=chosen_label,
                    bbox=chosen_box,
                )
            )
            summary_counter[f"record_{LABEL_ID_TO_NAME[chosen_label]}"] += 1

    summary = {
        "aux_root": str(aux_root),
        "image_dir": str(image_dir),
        "xml_dir": str(xml_dir),
        "num_records": len(records),
        "counters": dict(summary_counter),
    }
    return records, summary


def crop_with_margin(image: np.ndarray, bbox: Optional[tuple[int, int, int, int]], margin_ratio: float = 0.12) -> np.ndarray:
    if bbox is None:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = int(round(box_width * margin_ratio))
    margin_y = int(round(box_height * margin_ratio))

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def estimate_video_background(gray_frames: np.ndarray, sample_count: int = 24) -> np.ndarray:
    if len(gray_frames) == 0:
        raise ValueError("Cannot estimate background from empty frame list.")
    sample_indices = np.linspace(0, len(gray_frames) - 1, num=min(sample_count, len(gray_frames)), dtype=np.int32)
    stack = gray_frames[sample_indices]
    return np.median(stack.astype(np.float32), axis=0).astype(np.uint8)


def build_silhouette_mask(gray: np.ndarray, background: Optional[np.ndarray] = None) -> np.ndarray:
    if background is not None:
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        cleaned = np.zeros_like(mask)
        cleaned[labels == largest_label] = 255
        mask = cleaned
    return mask


def pixel_human_render(
    gray: np.ndarray,
    image_size: int,
    pixel_grid_size: int,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = build_silhouette_mask(gray, background=background)
    ys, xs = np.where(mask > 0)
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    if len(xs) == 0 or len(ys) == 0:
        return (canvas.astype(np.float32) / 255.0).astype(np.float32)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    pad_x = int(width * 0.08)
    pad_y = int(height * 0.08)
    h, w = gray.shape
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    cropped = mask[y0 : y1 + 1, x0 : x1 + 1]

    inner_grid = max(4, pixel_grid_size - 4)
    scale = min(inner_grid / max(cropped.shape[1], 1), inner_grid / max(cropped.shape[0], 1))
    resized_w = max(1, int(round(cropped.shape[1] * scale)))
    resized_h = max(1, int(round(cropped.shape[0] * scale)))
    resized = cv2.resize(cropped, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    binary = (resized > 40).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    grid = np.zeros((pixel_grid_size, pixel_grid_size), dtype=np.uint8)
    x_offset = (pixel_grid_size - resized_w) // 2
    y_offset = (pixel_grid_size - resized_h) // 2
    grid[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = binary

    block_image = np.full((pixel_grid_size, pixel_grid_size), 255, dtype=np.uint8)
    block_image[grid > 0] = 0
    rendered = cv2.resize(block_image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (rendered.astype(np.float32) / 255.0).astype(np.float32)


def silhouette_human_render(
    gray: np.ndarray,
    image_size: int,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = build_silhouette_mask(gray, background=background)
    ys, xs = np.where(mask > 0)
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    if len(xs) == 0 or len(ys) == 0:
        return (canvas.astype(np.float32) / 255.0).astype(np.float32)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    pad_x = int(width * 0.10)
    pad_y = int(height * 0.08)
    h, w = gray.shape
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    cropped = mask[y0 : y1 + 1, x0 : x1 + 1]

    inner_size = max(8, image_size - 16)
    scale = min(inner_size / max(cropped.shape[1], 1), inner_size / max(cropped.shape[0], 1))
    resized_w = max(1, int(round(cropped.shape[1] * scale)))
    resized_h = max(1, int(round(cropped.shape[0] * scale)))
    resized = cv2.resize(cropped, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    binary = (resized > 30).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    x_offset = (image_size - resized_w) // 2
    y_offset = (image_size - resized_h) // 2
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = 255
    canvas[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w][binary > 0] = 0
    return (canvas.astype(np.float32) / 255.0).astype(np.float32)


def list_picture_frames(picture_dir: Path) -> list[Path]:
    if not picture_dir.exists():
        return []
    return sorted([path for path in picture_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def read_color_image(image_path: Path) -> np.ndarray:
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def map_indices_to_sequence(total_count: int, sequence_count: int) -> list[int]:
    if sequence_count <= 0:
        return []
    if total_count <= 1:
        return [0] * sequence_count
    return [
        int(round(index * (total_count - 1) / max(sequence_count - 1, 1)))
        for index in range(sequence_count)
    ]


def crop_foreground_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((max(8, mask.shape[0] // 2), max(8, mask.shape[1] // 2)), dtype=np.uint8)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    pad_x = int(round(width * 0.10))
    pad_y = int(round(height * 0.08))
    h, w = mask.shape
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return mask[y0 : y1 + 1, x0 : x1 + 1]


def render_tracking_panel(
    image_bgr: np.ndarray,
    background_gray: np.ndarray,
    label_name: str,
    tracked: bool,
    sequence_index: int,
    source_frame_index: int,
    fps: float,
    panel_width: int = 96,
    panel_height: int = 96,
    header_height: int = 10,
    footer_height: int = 20,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = build_silhouette_mask(gray, background=background_gray)
    cropped = crop_foreground_mask(mask)

    panel = np.full((panel_height + header_height + footer_height, panel_width, 3), 252, dtype=np.uint8)
    if cropped.size > 0 and np.any(cropped > 0):
        inner_width = panel_width - 18
        inner_height = panel_height - 16
        scale = min(
            inner_width / max(cropped.shape[1], 1),
            inner_height / max(cropped.shape[0], 1),
        )
        resized_w = max(1, int(round(cropped.shape[1] * scale)))
        resized_h = max(1, int(round(cropped.shape[0] * scale)))
        silhouette = cv2.resize(cropped, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        binary = (silhouette > 30).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        x_offset = (panel_width - resized_w) // 2
        y_offset = header_height + (panel_height - resized_h) // 2
        region = np.full((resized_h, resized_w, 3), 252, dtype=np.uint8)
        region[binary > 0] = (0, 0, 0)
        panel[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = region

    header_color = np.array(LABEL_VISUAL_COLORS.get(label_name, LABEL_VISUAL_COLORS["normal"]), dtype=np.uint8)
    panel[:header_height, :, :] = header_color
    footer_y = panel_height + header_height
    if footer_height > 0:
        panel[footer_y:, :, :] = 248
        seq_text = f"{sequence_index + 1:02d}"
        time_text = f"f{source_frame_index}"
        cv2.putText(panel, seq_text, (8, footer_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (95, 95, 95), 1, cv2.LINE_AA)
        cv2.putText(panel, time_text, (panel_width - 40, footer_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (125, 125, 125), 1, cv2.LINE_AA)

    border_color = (36, 36, 255) if tracked else (166, 166, 166)
    border_thickness = 4 if tracked else 1
    cv2.rectangle(panel, (1, 1), (panel_width - 2, panel_height + header_height + footer_height - 2), border_color, border_thickness)
    if tracked:
        cv2.circle(panel, (panel_width - 10, 10), 4, (36, 36, 255), thickness=-1)
    return panel


def build_tracking_display_panels(
    picture_dir: Path,
    frame_label_ids: np.ndarray,
    tracking_mask: np.ndarray,
    fps: float,
    panel_width: int = 96,
    panel_height: int = 96,
    header_height: int = 10,
    footer_height: int = 20,
) -> tuple[list[np.ndarray], list[int], list[bool]]:
    frame_paths = list_picture_frames(picture_dir)
    if not frame_paths:
        return [], [], []

    images = [read_color_image(path) for path in frame_paths]
    gray_stack = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images], axis=0)
    background_gray = estimate_video_background(gray_stack)
    mapped_indices = map_indices_to_sequence(len(tracking_mask), len(images))

    panels: list[np.ndarray] = []
    tracked_flags: list[bool] = []
    for sequence_index, (image, source_frame_index) in enumerate(zip(images, mapped_indices)):
        safe_index = min(max(source_frame_index, 0), max(len(frame_label_ids) - 1, 0))
        label_id = int(frame_label_ids[safe_index]) if len(frame_label_ids) > 0 else LABEL_NAME_TO_ID["normal"]
        label_name = LABEL_ID_TO_NAME.get(label_id, "normal")
        tracked = bool(tracking_mask[min(max(source_frame_index, 0), max(len(tracking_mask) - 1, 0))]) if len(tracking_mask) > 0 else False
        panels.append(
            render_tracking_panel(
                image_bgr=image,
                background_gray=background_gray,
                label_name=label_name,
                tracked=tracked,
                sequence_index=sequence_index,
                source_frame_index=source_frame_index,
                fps=fps,
                panel_width=panel_width,
                panel_height=panel_height,
                header_height=header_height,
                footer_height=footer_height,
            )
        )
        tracked_flags.append(tracked)
    return panels, mapped_indices, tracked_flags


def compose_tracking_strip(panels: list[np.ndarray], gap: int = 3) -> Optional[np.ndarray]:
    if not panels:
        return None
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + gap * max(0, len(panels) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x_cursor = 0
    for panel in panels:
        panel_height, panel_width = panel.shape[:2]
        y_offset = (height - panel_height) // 2
        canvas[y_offset : y_offset + panel_height, x_cursor : x_cursor + panel_width] = panel
        x_cursor += panel_width + gap
    return canvas


def load_auxiliary_frame(
    record: AuxiliaryImageRecord,
    image_size: int,
    use_pixel_human: bool = False,
    use_silhouette_human: bool = False,
    pixel_grid_size: int = 20,
) -> np.ndarray:
    file_bytes = np.fromfile(str(record.image_path), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {record.image_path}")
    cropped = crop_with_margin(image, record.bbox)
    if use_silhouette_human:
        return silhouette_human_render(cropped, image_size=image_size, background=None)
    if use_pixel_human:
        return pixel_human_render(cropped, image_size=image_size, pixel_grid_size=pixel_grid_size, background=None)
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).astype(np.float32)


def load_video_windows(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(npz_path) as loaded:
        frames = loaded["frames"].astype(np.float32)
        frame_labels = loaded["frame_labels"].astype(np.int64)
        windows = loaded["windows"].astype(np.float32)
        window_labels = loaded["window_labels"].astype(np.int64)
    if windows.ndim != 4:
        raise ValueError(f"Expected windows in shape [N, T, H, W], got {windows.shape}")
    return frames, frame_labels, windows, window_labels, int(windows.shape[1])


def choose_window_label(frame_labels: np.ndarray) -> int:
    if np.any(frame_labels == LABEL_NAME_TO_ID["fall"]):
        return LABEL_NAME_TO_ID["fall"]
    if np.any(frame_labels == LABEL_NAME_TO_ID["pre_fall"]):
        return LABEL_NAME_TO_ID["pre_fall"]
    if np.any(frame_labels == LABEL_NAME_TO_ID["running"]):
        return LABEL_NAME_TO_ID["running"]
    return LABEL_NAME_TO_ID["normal"]


def build_video_windows_from_frames(
    frames: np.ndarray,
    frame_labels: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size > len(frames):
        raise ValueError("Window size cannot exceed total frame count.")
    windows = []
    labels = []
    for start in range(0, len(frames) - window_size + 1, stride):
        end = start + window_size
        windows.append(frames[start:end])
        labels.append(choose_window_label(frame_labels[start:end]))
    return np.stack(windows, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_fixed_label_windows(
    frames: np.ndarray,
    window_size: int,
    stride: int,
    label_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < window_size:
        return (
            np.empty((0, window_size, frames.shape[-2], frames.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    windows = []
    labels = []
    for start in range(0, len(frames) - window_size + 1, stride):
        end = start + window_size
        windows.append(frames[start:end])
        labels.append(label_id)
    return np.stack(windows, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def list_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    supported_suffixes = {".avi", ".mp4", ".mov", ".m4v", ".mkv"}
    return sorted(
        path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def infer_video_level_label_id(video_path: Path, fall_name_pattern: str) -> int:
    path_parts = {part.lower().strip() for part in video_path.parts}
    if "fall" in path_parts:
        return LABEL_NAME_TO_ID["fall"]
    if path_parts.intersection({"adl", "normal", "no_fall", "nofall"}):
        return LABEL_NAME_TO_ID["normal"]
    fall_pattern = fall_name_pattern.lower().strip()
    return LABEL_NAME_TO_ID["fall"] if fall_pattern and fall_pattern in video_path.stem.lower() else LABEL_NAME_TO_ID["normal"]


def parse_time_range_seconds(text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for start_text, end_text in re.findall(r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
        start_sec = float(start_text)
        end_sec = float(end_text)
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        ranges.append((start_sec, end_sec))
    return ranges


def load_gmdcsa_video_labels(video_path: Path, frame_count: int, fps: float) -> Optional[np.ndarray]:
    action_dir = video_path.parent
    subject_dir = action_dir.parent
    action_name = action_dir.name.strip().lower()
    if action_name not in {"adl", "fall"}:
        return None
    metadata_csv = subject_dir / f"{action_dir.name}.csv"
    if not metadata_csv.exists():
        return None

    labels = np.full(frame_count, LABEL_NAME_TO_ID["normal"], dtype=np.int64)
    if action_name == "adl":
        return labels

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_row = {str(key).strip(): value for key, value in row.items()}
            if (normalized_row.get("File Name") or "").strip() != video_path.name:
                continue
            classes_text = normalized_row.get("Classes") or ""
            for match in re.finditer(r"Falling[^\[]*\[([^\]]+)\]", classes_text, flags=re.IGNORECASE):
                for start_sec, end_sec in parse_time_range_seconds(match.group(1)):
                    start_frame = max(0, int(math.floor(start_sec * fps)))
                    end_frame = min(frame_count, int(math.ceil(end_sec * fps)))
                    if end_frame > start_frame:
                        labels[start_frame:end_frame] = LABEL_NAME_TO_ID["fall"]
            return labels
    return None


def load_external_no_fall_windows(
    video_dir: Optional[Path],
    image_size: int,
    window_size: int,
    stride: int,
    max_videos: int,
    use_pixel_human: bool,
    use_silhouette_human: bool,
    pixel_grid_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if video_dir is None:
        empty_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        return empty_windows, {"external_no_fall_dir": None, "num_videos": 0, "num_windows": 0, "videos": []}

    video_files = list_video_files(video_dir)
    if max_videos > 0:
        video_files = video_files[:max_videos]

    all_windows = []
    summary_rows = []
    for video_path in video_files:
        frames, fps = read_video_frames(
            video_path,
            image_size=image_size,
            use_pixel_human=use_pixel_human,
            use_silhouette_human=use_silhouette_human,
            pixel_grid_size=pixel_grid_size,
        )
        windows, _ = build_fixed_label_windows(
            frames,
            window_size=window_size,
            stride=max(1, stride),
            label_id=LABEL_NAME_TO_ID["normal"],
        )
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        summary_rows.append(
            {
                "video_name": video_path.name,
                "fps": round(float(fps), 4),
                "num_frames": int(len(frames)),
                "num_windows": int(len(windows)),
            }
        )

    if not all_windows:
        empty_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        return empty_windows, {
            "external_no_fall_dir": str(video_dir),
            "num_videos": 0,
            "num_windows": 0,
            "videos": [],
        }

    windows_array = np.concatenate(all_windows, axis=0).astype(np.float32)
    return windows_array, {
        "external_no_fall_dir": str(video_dir),
        "num_videos": len(summary_rows),
        "num_windows": int(len(windows_array)),
        "videos": summary_rows,
    }


def load_external_labeled_windows(
    video_dir: Optional[Path | list[Path]],
    image_size: int,
    window_size: int,
    stride: int,
    max_videos: int,
    fall_name_pattern: str,
    use_pixel_human: bool,
    use_silhouette_human: bool,
    pixel_grid_size: int,
    adl_context_path_keyword: str = "",
    adl_context_min_percentile: float = 0.80,
    adl_context_radius_windows: int = 0,
    adl_context_extra_copies: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if video_dir is None:
        empty_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, {"external_labeled_dir": None, "num_videos": 0, "num_windows": 0, "videos": []}

    video_dirs = video_dir if isinstance(video_dir, list) else [video_dir]
    video_files = []
    for root_dir in video_dirs:
        video_files.extend(list_video_files(root_dir))
    video_files = sorted(dict.fromkeys(video_files))
    if max_videos > 0:
        video_files = video_files[:max_videos]

    all_windows = []
    all_labels = []
    summary_rows = []
    for video_path in video_files:
        frames, fps = read_video_frames(
            video_path,
            image_size=image_size,
            use_pixel_human=use_pixel_human,
            use_silhouette_human=use_silhouette_human,
            pixel_grid_size=pixel_grid_size,
        )
        frame_labels = load_gmdcsa_video_labels(video_path, frame_count=len(frames), fps=fps)
        if frame_labels is not None:
            windows, labels = build_video_windows_from_frames(
                frames,
                frame_labels,
                window_size=window_size,
                stride=max(1, stride),
            )
            assigned_label = "metadata"
        else:
            label_id = infer_video_level_label_id(video_path, fall_name_pattern)
            windows, labels = build_fixed_label_windows(
                frames,
                window_size=window_size,
                stride=max(1, stride),
                label_id=label_id,
            )
            assigned_label = LABEL_ID_TO_NAME[label_id]
        if len(windows) == 0:
            continue
        adl_context_summary: dict[str, object] = {"enabled": False}
        extra_copies = max(0, int(adl_context_extra_copies))
        if (
            extra_copies > 0
            and path_matches_keyword(video_path, adl_context_path_keyword)
            and set(int(value) for value in labels.tolist()) == {LABEL_NAME_TO_ID["normal"]}
        ):
            boosted_windows, boosted_labels, adl_context_summary = select_adl_context_windows(
                windows,
                labels,
                min_percentile=adl_context_min_percentile,
                context_radius_windows=adl_context_radius_windows,
                min_keep=1,
            )
            if len(boosted_windows) > 0:
                for _ in range(extra_copies):
                    all_windows.append(boosted_windows)
                    all_labels.append(boosted_labels)
                adl_context_summary = {
                    **adl_context_summary,
                    "path_keyword": adl_context_path_keyword,
                    "extra_copies": extra_copies,
                    "added_windows": int(len(boosted_windows) * extra_copies),
                }
        all_windows.append(windows)
        all_labels.append(labels)
        summary_rows.append(
            {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "assigned_label": assigned_label,
                "fps": round(float(fps), 4),
                "num_frames": int(len(frames)),
                "num_windows": int(len(windows)),
                "window_class_counts": class_count_dict(labels),
                "adl_context_boost": adl_context_summary,
            }
        )

    boost_rows = [
        row.get("adl_context_boost", {})
        for row in summary_rows
        if isinstance(row.get("adl_context_boost"), dict) and row.get("adl_context_boost", {}).get("enabled")
    ]

    if not all_windows:
        empty_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, {
            "external_labeled_dir": [str(path) for path in video_dirs],
            "fall_name_pattern": fall_name_pattern,
            "num_videos": 0,
            "num_windows": 0,
            "videos": [],
        }

    windows_array = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels_array = np.concatenate(all_labels, axis=0).astype(np.int64)
    return windows_array, labels_array, {
        "external_labeled_dir": [str(path) for path in video_dirs],
        "fall_name_pattern": fall_name_pattern,
        "num_videos": len(summary_rows),
        "num_windows": int(len(windows_array)),
        "class_counts": class_count_dict(labels_array),
        "adl_context_boost": {
            "enabled": bool(boost_rows),
            "path_keyword": adl_context_path_keyword,
            "min_percentile": float(np.clip(adl_context_min_percentile, 0.0, 0.95)),
            "radius_windows": max(0, int(adl_context_radius_windows)),
            "extra_copies": max(0, int(adl_context_extra_copies)),
            "boosted_videos": len(boost_rows),
            "boosted_core_windows": int(sum(int(row.get("core_kept_windows", 0)) for row in boost_rows)),
            "boosted_context_windows": int(sum(int(row.get("context_added_windows", 0)) for row in boost_rows)),
            "boosted_added_windows": int(sum(int(row.get("added_windows", 0)) for row in boost_rows)),
        },
        "videos": summary_rows,
    }


def load_running_windows(
    running_dir: Path,
    image_size: int,
    window_size: int,
    stride: int,
    max_videos: int,
    use_pixel_human: bool,
    use_silhouette_human: bool,
    pixel_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    video_files = list_video_files(running_dir)
    if max_videos == 0:
        video_files = []
    elif max_videos > 0:
        video_files = video_files[:max_videos]

    all_windows = []
    all_labels = []
    summary_rows = []

    for video_path in video_files:
        frames, fps = read_video_frames(
            video_path,
            image_size=image_size,
            use_pixel_human=use_pixel_human,
            use_silhouette_human=use_silhouette_human,
            pixel_grid_size=pixel_grid_size,
        )
        windows, labels = build_fixed_label_windows(
            frames,
            window_size=window_size,
            stride=max(1, stride),
            label_id=LABEL_NAME_TO_ID["running"],
        )
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        summary_rows.append(
            {
                "video_name": video_path.name,
                "fps": round(float(fps), 4),
                "num_frames": int(len(frames)),
                "num_windows": int(len(windows)),
            }
        )

    if not all_windows:
        empty_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, {
            "running_dir": str(running_dir),
            "num_videos": 0,
            "num_windows": 0,
            "videos": [],
        }

    windows_array = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels_array = np.concatenate(all_labels, axis=0).astype(np.int64)
    summary = {
        "running_dir": str(running_dir),
        "num_videos": len(summary_rows),
        "num_windows": int(len(windows_array)),
        "videos": summary_rows,
    }
    return windows_array, labels_array, summary


def split_video_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label_id in sorted(set(labels.tolist())):
        label_indices = [idx for idx, value in enumerate(labels.tolist()) if value == label_id]
        rng.shuffle(label_indices)
        if len(label_indices) <= 3:
            split_count = 1 if len(label_indices) >= 2 and val_ratio > 0 else 0
        else:
            split_count = max(1, int(round(len(label_indices) * val_ratio)))
            split_count = min(split_count, len(label_indices) - 1)
        val_indices.extend(label_indices[:split_count])
        train_indices.extend(label_indices[split_count:])

    if not train_indices:
        train_indices = [idx for idx in range(len(labels))]
        val_indices = []

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    return (
        windows[train_indices],
        labels[train_indices],
        windows[val_indices] if val_indices else np.empty((0,) + windows.shape[1:], dtype=windows.dtype),
        labels[val_indices] if val_indices else np.empty((0,), dtype=labels.dtype),
    )


def load_frame_labels_csv(labels_csv: Path, expected_video_name: str) -> Optional[np.ndarray]:
    if not labels_csv.exists():
        return None
    rows: list[tuple[int, int]] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("video_name") or "").strip() != expected_video_name:
                continue
            frame_index_text = (row.get("frame_index") or "").strip()
            label_id_text = (row.get("label_id") or "").strip()
            if frame_index_text == "" or label_id_text == "":
                continue
            rows.append((int(frame_index_text), int(label_id_text)))
    if not rows:
        return None
    frame_count = max(frame_index for frame_index, _ in rows) + 1
    labels = np.full(frame_count, fill_value=-1, dtype=np.int64)
    for frame_index, label_id in rows:
        labels[frame_index] = label_id
    if np.any(labels < 0):
        return None
    return labels


def split_auxiliary_records(
    records: list[AuxiliaryImageRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[AuxiliaryImageRecord], list[AuxiliaryImageRecord]]:
    by_label: dict[int, list[AuxiliaryImageRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label_id].append(record)

    rng = random.Random(seed)
    train_records: list[AuxiliaryImageRecord] = []
    val_records: list[AuxiliaryImageRecord] = []

    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        if len(shuffled) <= 8:
            split_count = 1 if len(shuffled) >= 4 and val_ratio > 0 else 0
        else:
            split_count = max(1, int(round(len(shuffled) * val_ratio)))
            split_count = min(split_count, len(shuffled) - 2)
        val_records.extend(shuffled[:split_count])
        train_records.extend(shuffled[split_count:])

    return train_records, val_records


def rebalance_auxiliary_records(
    records: list[AuxiliaryImageRecord],
    max_aux_fall: int,
    seed: int,
) -> list[AuxiliaryImageRecord]:
    fall_records = [record for record in records if record.label_id == LABEL_NAME_TO_ID["fall"]]
    normal_records = [record for record in records if record.label_id == LABEL_NAME_TO_ID["normal"]]
    rng = random.Random(seed)
    rng.shuffle(fall_records)
    if max_aux_fall > 0:
        fall_records = fall_records[:max_aux_fall]
    balanced = normal_records + fall_records
    rng.shuffle(balanced)
    return balanced


class CombinedFallDataset(Dataset):
    def __init__(
        self,
        video_windows: np.ndarray,
        video_labels: np.ndarray,
        auxiliary_records: list[AuxiliaryImageRecord],
        window_size: int,
        image_size: int,
        use_crossbar: bool,
        use_pixel_human: bool,
        use_silhouette_human: bool,
        pixel_grid_size: int,
        hrs_values: np.ndarray,
        lrs_values: np.ndarray,
        device_dynamics: Optional[dict[str, object]],
        crossbar_readout_noise_scale: float,
        augment: bool,
        base_seed: int,
    ) -> None:
        self.video_windows = video_windows
        self.video_labels = video_labels
        self.auxiliary_records = auxiliary_records
        self.window_size = window_size
        self.image_size = image_size
        self.use_crossbar = use_crossbar
        self.use_pixel_human = use_pixel_human
        self.use_silhouette_human = use_silhouette_human
        self.pixel_grid_size = pixel_grid_size
        self.hrs_values = hrs_values
        self.lrs_values = lrs_values
        self.device_dynamics = device_dynamics
        self.crossbar_readout_noise_scale = crossbar_readout_noise_scale
        self.augment = augment
        self.base_seed = base_seed

        self.samples: list[dict[str, object]] = []
        for index, label_id in enumerate(video_labels.tolist()):
            hard_negative_score = 0.0
            if int(label_id) == LABEL_NAME_TO_ID["normal"]:
                hard_negative_score = compute_adl_hard_negative_score(video_windows[index])
            self.samples.append(
                {
                    "source": "video",
                    "index": index,
                    "label_id": int(label_id),
                    "hard_negative_score": hard_negative_score,
                }
            )
        for index, record in enumerate(auxiliary_records):
            self.samples.append(
                {
                    "source": "aux",
                    "index": index,
                    "label_id": int(record.label_id),
                    "hard_negative_score": 0.0,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label_id = int(sample["label_id"])
        sample_seed = self.base_seed + index * 97 + label_id * 1009
        rng = np.random.default_rng(sample_seed)

        if sample["source"] == "video":
            window_frames = self.video_windows[int(sample["index"])].copy()
            if self.augment:
                window_frames = apply_consistent_augmentation(window_frames, rng)
        else:
            record = self.auxiliary_records[int(sample["index"])]
            frame = load_auxiliary_frame(
                record,
                image_size=self.image_size,
                use_pixel_human=self.use_pixel_human,
                use_silhouette_human=self.use_silhouette_human,
                pixel_grid_size=self.pixel_grid_size,
            )
            window_frames = build_static_temporal_window(
                frame,
                window_size=self.window_size,
                rng=rng,
                augment=self.augment,
            )

        if self.use_crossbar:
            window_frames = build_crossbar_window(
                window_frames,
                self.hrs_values,
                self.lrs_values,
                sample_seed,
                device_dynamics=self.device_dynamics,
                readout_noise_scale=self.crossbar_readout_noise_scale,
            )

        tensor = torch.from_numpy(window_frames[:, np.newaxis, :, :]).float()
        target = torch.tensor(label_id, dtype=torch.long)
        return tensor, target


class FallEventDetector(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 4,
        use_snn_temporal_branch: bool = False,
        snn_hidden_dim: int = 32,
        snn_decay: float = 0.82,
        snn_threshold: float = 0.55,
        snn_surrogate_scale: float = 8.0,
        learnable_snn_dynamics: bool = True,
        use_snn_fusion_gate: bool = False,
        snn_fusion_gate_mode: str = "scalar",
    ) -> None:
        super().__init__()
        self.use_snn_temporal_branch = bool(use_snn_temporal_branch)
        self.learnable_snn_dynamics = bool(learnable_snn_dynamics)
        self.use_snn_fusion_gate = bool(use_snn_fusion_gate and use_snn_temporal_branch)
        snn_fusion_gate_mode = str(snn_fusion_gate_mode).lower()
        if snn_fusion_gate_mode not in {"scalar", "channel", "group_channel"}:
            snn_fusion_gate_mode = "scalar"
        self.snn_fusion_gate_mode = snn_fusion_gate_mode
        self.snn_gate_context_dim = 1
        self.snn_decay_min = 0.05
        self.snn_decay_max = 0.99
        self.snn_threshold_min = 0.05
        self.snn_threshold_max = 1.25
        self.snn_decay = float(np.clip(snn_decay, 0.0, 0.99))
        self.snn_threshold = float(np.clip(snn_threshold, self.snn_threshold_min, self.snn_threshold_max))
        self.snn_surrogate_scale = float(max(1e-3, snn_surrogate_scale))
        if self.use_snn_temporal_branch and self.learnable_snn_dynamics:
            decay_init = self._bounded_logit(self.snn_decay, self.snn_decay_min, self.snn_decay_max)
            threshold_init = self._bounded_logit(
                self.snn_threshold, self.snn_threshold_min, self.snn_threshold_max
            )
            self.snn_decay_logit = nn.Parameter(torch.full((snn_hidden_dim,), decay_init, dtype=torch.float32))
            self.snn_threshold_logit = nn.Parameter(torch.full((snn_hidden_dim,), threshold_init, dtype=torch.float32))
        else:
            self.register_parameter("snn_decay_logit", None)
            self.register_parameter("snn_threshold_logit", None)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(96 * 2 * 2, hidden_dim)
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )
        if self.use_snn_temporal_branch:
            self.snn_input = nn.Sequential(
                nn.Linear(7, snn_hidden_dim),
                nn.LayerNorm(snn_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(snn_hidden_dim, snn_hidden_dim),
            )
            snn_fusion_dim = snn_hidden_dim * 3
            if self.use_snn_fusion_gate:
                gate_input_dim = hidden_dim * 6 + snn_fusion_dim + self.snn_gate_context_dim
                if self.snn_fusion_gate_mode == "group_channel":
                    self.snn_group_gate = nn.Sequential(
                        nn.Linear(gate_input_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, 3),
                        nn.Sigmoid(),
                    )
                    self.snn_channel_gate = nn.Sequential(
                        nn.Linear(gate_input_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, snn_fusion_dim),
                        nn.Sigmoid(),
                    )
                    nn.init.constant_(self.snn_group_gate[2].bias, -0.5)
                    nn.init.constant_(self.snn_channel_gate[2].bias, -1.0)
                    self.snn_fusion_gate = None
                else:
                    gate_out_dim = snn_fusion_dim if self.snn_fusion_gate_mode == "channel" else 1
                    self.snn_fusion_gate = nn.Sequential(
                        nn.Linear(gate_input_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, gate_out_dim),
                        nn.Sigmoid(),
                    )
                    nn.init.constant_(self.snn_fusion_gate[2].bias, -1.0)
                    self.snn_group_gate = None
                    self.snn_channel_gate = None
            else:
                self.snn_fusion_gate = None
                self.snn_group_gate = None
                self.snn_channel_gate = None
        else:
            self.snn_input = None
            self.snn_fusion_gate = None
            self.snn_group_gate = None
            self.snn_channel_gate = None
            snn_fusion_dim = 0
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 6 + snn_fusion_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    @staticmethod
    def _bounded_logit(value: float, lower: float, upper: float) -> float:
        normalized = (float(value) - float(lower)) / max(float(upper) - float(lower), 1e-6)
        normalized = float(np.clip(normalized, 1e-4, 1.0 - 1e-4))
        return math.log(normalized / (1.0 - normalized))

    def _current_snn_dynamics(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.snn_decay_logit is not None and self.snn_threshold_logit is not None:
            decay = self.snn_decay_min + (self.snn_decay_max - self.snn_decay_min) * torch.sigmoid(self.snn_decay_logit)
            threshold = self.snn_threshold_min + (
                self.snn_threshold_max - self.snn_threshold_min
            ) * torch.sigmoid(self.snn_threshold_logit)
            decay = decay.to(device=current.device, dtype=current.dtype).view(1, -1)
            threshold = threshold.to(device=current.device, dtype=current.dtype).view(1, -1)
        else:
            decay = current.new_full((1, current.size(2)), self.snn_decay)
            threshold = current.new_full((1, current.size(2)), self.snn_threshold)
        return decay, threshold

    @staticmethod
    def extract_temporal_event_features(x: torch.Tensor) -> torch.Tensor:
        frames = x[:, :, 0].float()
        batch_size, time_steps, height, width = frames.shape
        frame_min = frames.amin(dim=(2, 3), keepdim=True)
        frame_max = frames.amax(dim=(2, 3), keepdim=True)
        norm = (frames - frame_min) / (frame_max - frame_min).clamp_min(1e-6)
        mass = norm.mean(dim=(2, 3))
        motion = torch.zeros_like(mass)
        if time_steps > 1:
            motion[:, 1:] = torch.mean(torch.abs(norm[:, 1:] - norm[:, :-1]), dim=(2, 3))

        row_mass = norm.sum(dim=3)
        col_mass = norm.sum(dim=2)
        row_total = row_mass.sum(dim=2).clamp_min(1e-6)
        col_total = col_mass.sum(dim=2).clamp_min(1e-6)
        y_coords = torch.linspace(0.0, 1.0, height, device=x.device, dtype=norm.dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=x.device, dtype=norm.dtype)
        center_y = (row_mass * y_coords.view(1, 1, height)).sum(dim=2) / row_total
        center_x = (col_mass * x_coords.view(1, 1, width)).sum(dim=2) / col_total
        var_y = (row_mass * (y_coords.view(1, 1, height) - center_y.unsqueeze(-1)).pow(2)).sum(dim=2) / row_total
        var_x = (col_mass * (x_coords.view(1, 1, width) - center_x.unsqueeze(-1)).pow(2)).sum(dim=2) / col_total
        soft_height = torch.sqrt(var_y.clamp_min(1e-6)) * 3.46
        soft_width = torch.sqrt(var_x.clamp_min(1e-6)) * 3.46
        aspect = soft_width / soft_height.clamp_min(1e-4)
        center_drop = torch.zeros_like(center_y)
        if time_steps > 1:
            center_drop[:, 1:] = torch.relu(center_y[:, 1:] - center_y[:, :-1])
        height_change = torch.zeros_like(soft_height)
        aspect_change = torch.zeros_like(aspect)
        if time_steps > 1:
            height_change[:, 1:] = torch.relu(soft_height[:, :-1] - soft_height[:, 1:])
            aspect_change[:, 1:] = torch.relu(aspect[:, 1:] - aspect[:, :-1])
        center_drop_accel = torch.zeros_like(center_drop)
        if time_steps > 1:
            center_drop_accel[:, 1:] = torch.relu(center_drop[:, 1:] - center_drop[:, :-1])
        running_min_center_y = torch.cummin(center_y, dim=1).values
        relative_low_position = torch.relu(center_y - running_min_center_y)
        peak_motion = torch.cummax(motion, dim=1).values.clamp_min(1e-6)
        post_peak_hold = relative_low_position * torch.relu(1.0 - motion / peak_motion)
        fall_event_like = motion * (
            center_drop
            + 0.5 * center_drop_accel
            + height_change
            + 0.5 * aspect_change
            + 0.35 * post_peak_hold
        )
        return torch.stack(
            [mass, motion, center_y, center_drop, soft_height, aspect, fall_event_like],
            dim=2,
        )

    @staticmethod
    def extract_fall_event_score(features: torch.Tensor) -> torch.Tensor:
        fall_event_score = features[:, :, 6].amax(dim=1, keepdim=True)
        return fall_event_score

    def run_lif_temporal_branch(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        if self.snn_input is None:
            raise RuntimeError("SNN temporal branch is disabled.")
        if features is None:
            features = self.extract_temporal_event_features(x)
        current = self.snn_input(features)
        decay, threshold = self._current_snn_dynamics(current)
        membrane = torch.zeros(current.size(0), current.size(2), device=current.device, dtype=current.dtype)
        spikes = []
        for time_index in range(current.size(1)):
            membrane = decay * membrane + current[:, time_index, :]
            spike = torch.sigmoid((membrane - threshold) * self.snn_surrogate_scale)
            membrane = membrane * (1.0 - spike.detach())
            spikes.append(spike)
        spike_train = torch.stack(spikes, dim=1)
        return torch.cat(
            [
                spike_train.mean(dim=1),
                spike_train.max(dim=1).values,
                spike_train[:, -1, :],
            ],
            dim=1,
        )

    def apply_snn_fusion_gate(self, pooled: torch.Tensor, snn_pooled: torch.Tensor, fall_event_score: torch.Tensor) -> torch.Tensor:
        if self.snn_fusion_gate_mode == "group_channel" and self.snn_group_gate is not None and self.snn_channel_gate is not None:
            gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
            group_gate = self.snn_group_gate(gate_context)
            channel_gate = self.snn_channel_gate(gate_context)
            group_dim = snn_pooled.size(1) // 3
            if group_dim * 3 != snn_pooled.size(1):
                raise RuntimeError("SNN pooled dimension must be divisible by 3 for group_channel gate.")
            group_gate = torch.repeat_interleave(group_gate, repeats=group_dim, dim=1)
            gate = channel_gate * group_gate
            return snn_pooled * gate
        if self.snn_fusion_gate is not None:
            gate_context = torch.cat([pooled, snn_pooled, fall_event_score], dim=1)
            fusion_gate = self.snn_fusion_gate(gate_context)
            return snn_pooled * fusion_gate
        return snn_pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        encoded = x.view(batch_size * time_steps, channels, height, width)
        encoded = self.encoder(encoded).view(batch_size * time_steps, -1)
        encoded = self.proj(encoded)
        encoded = encoded.view(batch_size, time_steps, -1)
        temporal_out, _ = self.temporal(encoded)
        pooled_mean = temporal_out.mean(dim=1)
        pooled_max = temporal_out.max(dim=1).values
        pooled_last = temporal_out[:, -1, :]
        pooled = torch.cat([pooled_mean, pooled_max, pooled_last], dim=1)
        if self.use_snn_temporal_branch:
            features = self.extract_temporal_event_features(x)
            snn_pooled = self.run_lif_temporal_branch(x, features)
            fall_event_score = self.extract_fall_event_score(features)
            snn_pooled = self.apply_snn_fusion_gate(pooled, snn_pooled, fall_event_score)
            pooled = torch.cat([pooled, snn_pooled], dim=1)
        return self.head(pooled)


def load_compatible_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    current_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    unexpected_keys = []
    for key, value in state_dict.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value
    missing_keys = [key for key in current_state if key not in compatible_state]
    current_state.update(compatible_state)
    model.load_state_dict(current_state)
    return missing_keys, unexpected_keys, skipped_keys


def set_cnn_gru_frozen(model: FallEventDetector, freeze: bool) -> dict[str, int]:
    freeze_prefixes = ("encoder.", "proj.", "temporal.")
    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        should_freeze = bool(freeze and name.startswith(freeze_prefixes))
        parameter.requires_grad_(not should_freeze)
        count = int(parameter.numel())
        if should_freeze:
            frozen += count
        else:
            trainable += count
    return {"frozen_parameters": frozen, "trainable_parameters": trainable}


def class_count_dict(values: list[int] | np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {LABEL_ID_TO_NAME[label_id]: int(counts.get(label_id, 0)) for label_id in LABEL_ID_TO_NAME}


def clamp_hard_negative_max_multiplier(value: float) -> float:
    return float(np.clip(float(value), 1.0, 4.0))


def shape_hard_negative_scores(
    scores: list[float] | np.ndarray,
    mode: str,
    min_percentile: float,
    gamma: float,
) -> np.ndarray:
    raw_scores = np.asarray(scores, dtype=np.float32)
    if len(raw_scores) == 0:
        return raw_scores

    threshold = float(np.clip(min_percentile, 0.0, 0.95))
    gamma = max(0.05, float(gamma))
    if mode == "quantile":
        order = np.argsort(raw_scores, kind="mergesort")
        ranks = np.zeros_like(raw_scores, dtype=np.float32)
        if len(order) > 1:
            ranks[order] = np.linspace(0.0, 1.0, num=len(order), dtype=np.float32)
        else:
            ranks[order] = 1.0
        shaped = np.clip((ranks - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    else:
        clipped = np.clip(raw_scores, 0.0, 1.0)
        shaped = np.clip((clipped - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return np.power(shaped, gamma).astype(np.float32)


def make_weighted_sampler(
    dataset: CombinedFallDataset,
    video_source_weight: float,
    hard_negative_normal_weight: float = 0.0,
    hard_negative_score_mode: str = "quantile",
    hard_negative_min_percentile: float = 0.70,
    hard_negative_score_gamma: float = 1.0,
    hard_negative_max_multiplier: float = 4.0,
) -> WeightedRandomSampler:
    labels = [int(sample["label_id"]) for sample in dataset.samples]
    label_counts = Counter(labels)
    normal_sample_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if int(sample["label_id"]) == LABEL_NAME_TO_ID["normal"]
    ]
    normal_scores = np.asarray(
        [float(dataset.samples[index].get("hard_negative_score", 0.0)) for index in normal_sample_indices],
        dtype=np.float32,
    )
    shaped_normal_scores: dict[int, float] = {}
    if len(normal_sample_indices) > 0 and hard_negative_normal_weight > 0:
        shaped = shape_hard_negative_scores(
            normal_scores,
            mode=hard_negative_score_mode,
            min_percentile=hard_negative_min_percentile,
            gamma=hard_negative_score_gamma,
        )
        shaped_normal_scores = {
            normal_sample_indices[index]: float(score)
            for index, score in enumerate(shaped.tolist())
        }
    max_multiplier = clamp_hard_negative_max_multiplier(hard_negative_max_multiplier)
    sample_weights = []
    for index, sample in enumerate(dataset.samples):
        label_id = int(sample["label_id"])
        source = str(sample["source"])
        label_weight = 1.0 / math.sqrt(max(label_counts[label_id], 1))
        source_weight = video_source_weight if source == "video" else 1.0
        hard_weight = 1.0
        if label_id == LABEL_NAME_TO_ID["normal"] and hard_negative_normal_weight > 0:
            hard_score = shaped_normal_scores.get(index, float(sample.get("hard_negative_score", 0.0)))
            hard_weight += float(hard_negative_normal_weight) * hard_score
            hard_weight = min(hard_weight, max_multiplier)
        sample_weights.append(label_weight * source_weight * hard_weight)
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(sample_weights), replacement=True)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler],
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def apply_student_noise(
    batch_x: torch.Tensor,
    noise_std: float,
    noise_prob: float,
    frame_drop_prob: float,
    noise_mode: str = "gaussian",
    device_noise_profile: Optional[np.ndarray] = None,
    device_noise_scale: float = 1.0,
) -> torch.Tensor:
    if noise_std <= 0 and frame_drop_prob <= 0:
        return batch_x
    batch_size = int(batch_x.size(0))
    time_steps = int(batch_x.size(1))
    apply_mask = (
        torch.rand((batch_size, 1, 1, 1, 1), device=batch_x.device, dtype=batch_x.dtype)
        < float(np.clip(noise_prob, 0.0, 1.0))
    ).to(batch_x.dtype)
    noisy = batch_x
    if noise_mode == "hybrid_vp" and device_noise_profile is not None and len(device_noise_profile) > 0:
        mix = float(np.clip(device_noise_scale, 0.0, 1.0))
        profile = resample_curve_to_frames(np.asarray(device_noise_profile, dtype=np.float32), time_steps)
        profile = profile - float(profile.mean())
        profile_std = float(profile.std())
        if profile_std <= 1e-6:
            profile = np.ones_like(profile, dtype=np.float32)
        else:
            profile = profile / profile_std
        profile_tensor = torch.tensor(profile, device=batch_x.device, dtype=batch_x.dtype).view(1, time_steps, 1, 1, 1)
        gaussian_unit = torch.randn_like(batch_x)
        device_pattern = torch.randn(
            (batch_size, 1, 1, batch_x.size(-2), batch_x.size(-1)),
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        device_unit = profile_tensor * device_pattern
        device_std = device_unit.flatten(1).std(dim=1, unbiased=False).clamp_min(1e-6).view(batch_size, 1, 1, 1, 1)
        device_unit = device_unit / device_std
        mixed_unit = math.sqrt(max(0.0, 1.0 - mix)) * gaussian_unit + math.sqrt(mix) * device_unit
        noisy = noisy + mixed_unit * float(noise_std) * apply_mask
    elif noise_mode in {"device", "hybrid"} and device_noise_profile is not None and len(device_noise_profile) > 0:
        scale = float(np.clip(device_noise_scale, 0.0, 2.0))
        profile = resample_curve_to_frames(np.asarray(device_noise_profile, dtype=np.float32), time_steps)
        profile = profile - float(profile.mean())
        profile_tensor = torch.tensor(profile, device=batch_x.device, dtype=batch_x.dtype).view(1, time_steps, 1, 1, 1)
        temporal_gain = 1.0 + apply_mask * profile_tensor * float(noise_std) * 2.5 * scale
        spatial_jitter = 1.0 + apply_mask * torch.randn(
            (batch_size, 1, 1, batch_x.size(-2), batch_x.size(-1)),
            device=batch_x.device,
            dtype=batch_x.dtype,
        ) * float(noise_std) * 0.35 * scale
        noisy = noisy * temporal_gain * spatial_jitter
        gaussian_scale = 1.0 if noise_mode == "hybrid" else 0.45 * scale
        gaussian = torch.randn_like(batch_x) * float(noise_std) * gaussian_scale
        noisy = noisy + gaussian * apply_mask
    elif noise_std > 0:
        gaussian = torch.randn_like(batch_x) * float(noise_std)
        noisy = noisy + gaussian * apply_mask
    if frame_drop_prob > 0 and time_steps > 1:
        drop_mask = (
            torch.rand((batch_size, time_steps, 1, 1, 1), device=batch_x.device, dtype=batch_x.dtype)
            < float(np.clip(frame_drop_prob, 0.0, 0.5))
        )
        replacement = noisy.mean(dim=1, keepdim=True).expand_as(noisy)
        noisy = torch.where(drop_mask, replacement, noisy)
    return torch.clamp(noisy, 0.0, 1.0)


def update_ema_teacher(student: nn.Module, teacher: nn.Module, ema_decay: float) -> None:
    decay = float(np.clip(ema_decay, 0.0, 0.9999))
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)


def consistency_ramp_weight(base_weight: float, epoch: int, ramp_epochs: int) -> float:
    if base_weight <= 0:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    phase = min(1.0, max(0.0, float(epoch) / float(ramp_epochs)))
    return float(base_weight) * phase * phase


def compute_class_weights(labels: list[int] | np.ndarray, device: torch.device) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    weights = []
    for label_id in sorted(LABEL_ID_TO_NAME):
        count = max(int(counts.get(label_id, 0)), 1)
        weights.append(1.0 / math.sqrt(count))
    weights = np.asarray(weights, dtype=np.float32)
    min_weight = float(weights.min())
    if min_weight > 0:
        weights = np.minimum(weights, min_weight * 2.5)
    weights = weights / max(weights.sum(), 1e-6) * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    preds_all: list[int] = []
    targets_all: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            preds = torch.argmax(logits, dim=1)

            batch_size = int(batch_x.size(0))
            loss_sum += float(loss.item()) * batch_size
            total += batch_size
            correct += int((preds == batch_y).sum().item())
            preds_all.extend(preds.detach().cpu().tolist())
            targets_all.extend(batch_y.detach().cpu().tolist())

    metrics = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
    }
    for label_id, label_name in LABEL_ID_TO_NAME.items():
        class_total = sum(1 for target in targets_all if target == label_id)
        class_correct = sum(
            1 for pred, target in zip(preds_all, targets_all) if target == label_id and pred == target
        )
        metrics[f"{label_name}_accuracy"] = class_correct / class_total if class_total else float("nan")
    class_accuracies = [
        float(metrics[f"{label_name}_accuracy"])
        for label_name in LABEL_NAME_TO_ID
        if not math.isnan(float(metrics[f"{label_name}_accuracy"]))
    ]
    metrics["balanced_accuracy"] = float(np.mean(class_accuracies)) if class_accuracies else float("nan")
    return metrics


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_training_plot(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_accuracy"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(epochs, train_loss, marker="o", linewidth=2.0, label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", linewidth=2.0, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2.0, label="train_acc")
    axes[1].plot(epochs, val_acc, marker="o", linewidth=2.0, label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend()

    figure.tight_layout()
    ensure_parent_dir(output_path)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_detector(
    args: argparse.Namespace,
    device: torch.device,
) -> TrainArtifacts:
    window_size = args.window_size
    image_size = args.image_size
    hard_negative_max_multiplier = clamp_hard_negative_max_multiplier(args.hard_negative_max_multiplier)
    if abs(float(args.hard_negative_max_multiplier) - hard_negative_max_multiplier) > 1e-6:
        log(
            "Hard-negative max multiplier was clamped | "
            f"requested={float(args.hard_negative_max_multiplier):.2f} "
            f"used={hard_negative_max_multiplier:.2f}"
        )

    if args.skip_main_video_training:
        frames = np.empty((0, image_size, image_size), dtype=np.float32)
        frame_labels = np.empty((0,), dtype=np.int64)
        fall_video_windows = np.empty((0, window_size, image_size, image_size), dtype=np.float32)
        fall_video_window_labels = np.empty((0,), dtype=np.int64)
        main_video_summary = {"skipped": True, "video_path": str(args.video)}
    else:
        frames, _ = read_video_frames(
            args.video,
            image_size=args.image_size,
            use_pixel_human=args.use_pixel_human,
            use_silhouette_human=args.use_silhouette_human,
            pixel_grid_size=args.pixel_grid_size,
        )
        frame_labels = load_frame_labels_csv(args.labels_csv, expected_video_name=args.video.name)
        if frame_labels is None:
            raise FileNotFoundError(
                f"Missing or incomplete label CSV for training: {args.labels_csv}. "
                "Please rerun the annotation script first."
            )
        if len(frame_labels) != len(frames):
            raise ValueError(
                f"Frame label count {len(frame_labels)} does not match decoded video frames {len(frames)}."
            )
        image_size = int(frames.shape[-1])
        if args.image_size != image_size:
            args.image_size = image_size
        fall_video_windows, fall_video_window_labels = build_video_windows_from_frames(
            frames,
            frame_labels,
            window_size=args.window_size,
            stride=max(1, args.video_window_stride),
        )
        main_video_summary = {
            "skipped": False,
            "video_path": str(args.video),
            "num_frames": int(len(frames)),
            "num_windows": int(len(fall_video_windows)),
            "class_counts": class_count_dict(fall_video_window_labels),
        }

    if args.disable_auxiliary_data:
        aux_records = []
        aux_summary = {"disabled": True, "num_records": 0}
    else:
        aux_root = discover_aux_dataset_root(args.aux_root)
        aux_records, aux_summary = load_auxiliary_records(aux_root)
        aux_records = rebalance_auxiliary_records(aux_records, max_aux_fall=args.max_aux_fall, seed=args.seed)
    train_aux_records, val_aux_records = split_auxiliary_records(aux_records, args.val_ratio, args.seed)
    running_windows, running_labels, running_summary = load_running_windows(
        args.running_dir,
        image_size=args.image_size,
        window_size=args.window_size,
        stride=max(1, args.running_window_stride),
        max_videos=max(0, args.max_running_videos),
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
    )
    external_labeled_windows, external_labeled_labels, external_labeled_summary = load_external_labeled_windows(
        args.external_labeled_dir,
        image_size=args.image_size,
        window_size=args.window_size,
        stride=max(1, args.external_labeled_stride),
        max_videos=max(0, args.max_external_labeled_videos),
        fall_name_pattern=args.external_fall_name_pattern,
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
        adl_context_path_keyword=args.adl_context_path_keyword,
        adl_context_min_percentile=args.adl_context_min_percentile,
        adl_context_radius_windows=args.adl_context_radius_windows,
        adl_context_extra_copies=args.adl_context_extra_copies,
    )
    external_val_labeled_windows, external_val_labeled_labels, external_val_labeled_summary = load_external_labeled_windows(
        args.external_val_labeled_dir,
        image_size=args.image_size,
        window_size=args.window_size,
        stride=max(1, args.external_labeled_stride),
        max_videos=0,
        fall_name_pattern=args.external_fall_name_pattern,
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
    )
    external_no_fall_windows, external_no_fall_summary = load_external_no_fall_windows(
        args.external_no_fall_dir,
        image_size=args.image_size,
        window_size=args.window_size,
        stride=max(1, args.external_no_fall_stride),
        max_videos=max(0, args.max_external_no_fall_videos),
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
    )
    combined_video_windows = fall_video_windows
    combined_video_labels = fall_video_window_labels
    if len(running_windows) > 0:
        combined_video_windows = np.concatenate([combined_video_windows, running_windows], axis=0).astype(np.float32)
        combined_video_labels = np.concatenate([combined_video_labels, running_labels], axis=0).astype(np.int64)
    if len(external_labeled_windows) > 0:
        combined_video_windows = np.concatenate([combined_video_windows, external_labeled_windows], axis=0).astype(np.float32)
        combined_video_labels = np.concatenate([combined_video_labels, external_labeled_labels], axis=0).astype(np.int64)

    explicit_video_validation = len(external_val_labeled_windows) > 0
    if explicit_video_validation:
        train_video_windows = combined_video_windows
        train_video_labels = combined_video_labels
        val_video_windows = external_val_labeled_windows
        val_video_labels = external_val_labeled_labels
        train_aux_records = aux_records
        val_aux_records = []
        validation_split_summary = {
            "mode": "external_val_labeled_dir",
            "note": "Validation is split by held-out videos/subjects instead of random windows.",
        }
    else:
        train_video_windows, train_video_labels, val_video_windows, val_video_labels = split_video_windows(
            combined_video_windows,
            combined_video_labels,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        validation_split_summary = {
            "mode": "random_window_split",
            "val_ratio": float(args.val_ratio),
            "note": "Legacy fallback: windows are stratified by class and randomly split.",
        }
    adl_context_boost_summary = external_labeled_summary.get("adl_context_boost", {})
    if isinstance(adl_context_boost_summary, dict) and adl_context_boost_summary.get("enabled"):
        log(
            "ADL context boost | "
            f"keyword={adl_context_boost_summary.get('path_keyword')} "
            f"videos={int(adl_context_boost_summary.get('boosted_videos', 0))} "
            f"added_windows={int(adl_context_boost_summary.get('boosted_added_windows', 0))} "
            f"core_windows={int(adl_context_boost_summary.get('boosted_core_windows', 0))} "
            f"context_windows={int(adl_context_boost_summary.get('boosted_context_windows', 0))} "
            f"radius={int(adl_context_boost_summary.get('radius_windows', 0))} "
            f"copies={int(adl_context_boost_summary.get('extra_copies', 0))}"
        )
    log(
        "Dataset summary | "
        f"main_video={main_video_summary} | "
        f"fall_video_windows={len(fall_video_windows)} {class_count_dict(fall_video_window_labels)} | "
        f"running_windows={len(running_windows)} | "
        f"external_labeled_windows={len(external_labeled_windows)} {class_count_dict(external_labeled_labels)} | "
        f"external_val_labeled_windows={len(external_val_labeled_windows)} {class_count_dict(external_val_labeled_labels)} | "
        f"external_no_fall_windows={len(external_no_fall_windows)} | "
        f"combined_windows={len(combined_video_windows)} {class_count_dict(combined_video_labels)} | "
        f"aux_records={len(aux_records)} train_aux={len(train_aux_records)} val_aux={len(val_aux_records)} | "
        f"validation_split={validation_split_summary['mode']}"
    )

    use_crossbar = not args.disable_crossbar
    hrs_values, lrs_values = load_conductance_pair(args.data_dir) if use_crossbar else (
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    use_crossbar_readout_noise = use_crossbar and args.crossbar_readout_noise_scale > 0
    device_dynamics = load_device_dynamics(args.data_dir) if (
        args.use_device_dynamics
        or use_crossbar_readout_noise
        or args.student_noise_mode in {"device", "hybrid", "hybrid_vp"}
    ) else None
    crossbar_device_dynamics = device_dynamics if (
        use_crossbar and (args.use_device_dynamics or use_crossbar_readout_noise)
    ) else None
    device_noise_profile = build_device_noise_profile(device_dynamics) if args.student_noise_mode in {"device", "hybrid", "hybrid_vp"} else None

    train_dataset = CombinedFallDataset(
        video_windows=train_video_windows,
        video_labels=train_video_labels,
        auxiliary_records=train_aux_records,
        window_size=args.window_size,
        image_size=args.image_size,
        use_crossbar=use_crossbar,
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=crossbar_device_dynamics,
        crossbar_readout_noise_scale=args.crossbar_readout_noise_scale,
        augment=True,
        base_seed=args.seed,
    )
    val_dataset = CombinedFallDataset(
        video_windows=val_video_windows,
        video_labels=val_video_labels,
        auxiliary_records=val_aux_records,
        window_size=args.window_size,
        image_size=args.image_size,
        use_crossbar=use_crossbar,
        use_pixel_human=args.use_pixel_human,
        use_silhouette_human=args.use_silhouette_human,
        pixel_grid_size=args.pixel_grid_size,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=crossbar_device_dynamics,
        crossbar_readout_noise_scale=args.crossbar_readout_noise_scale,
        augment=False,
        base_seed=args.seed + 9999,
    )

    if args.hard_negative_normal_weight > 0:
        hard_scores = [
            float(sample.get("hard_negative_score", 0.0))
            for sample in train_dataset.samples
            if int(sample["label_id"]) == LABEL_NAME_TO_ID["normal"]
        ]
        if hard_scores:
            shaped_scores = shape_hard_negative_scores(
                hard_scores,
                mode=args.hard_negative_score_mode,
                min_percentile=args.hard_negative_min_percentile,
                gamma=args.hard_negative_score_gamma,
            )
            multipliers = 1.0 + float(args.hard_negative_normal_weight) * shaped_scores
            multipliers = np.minimum(multipliers, hard_negative_max_multiplier)
            boosted_windows = int(np.sum(shaped_scores > 1e-6))
            log(
                "Hard-negative ADL sampler | "
                f"normal_windows={len(hard_scores)} "
                f"mean_score={float(np.mean(hard_scores)):.3f} "
                f"top10_mean={float(np.mean(sorted(hard_scores)[-min(10, len(hard_scores)):])):.3f} "
                f"mode={args.hard_negative_score_mode} "
                f"min_percentile={float(np.clip(args.hard_negative_min_percentile, 0.0, 0.95)):.2f} "
                f"gamma={max(0.05, float(args.hard_negative_score_gamma)):.2f} "
                f"boosted_windows={boosted_windows} "
                f"weight={args.hard_negative_normal_weight:.2f} "
                f"multiplier_mean={float(np.mean(multipliers)):.2f} "
                f"multiplier_max={float(np.max(multipliers)):.2f} "
                f"multiplier_cap={hard_negative_max_multiplier:.2f}"
            )

    sampler = make_weighted_sampler(
        train_dataset,
        video_source_weight=args.video_source_weight,
        hard_negative_normal_weight=args.hard_negative_normal_weight,
        hard_negative_score_mode=args.hard_negative_score_mode,
        hard_negative_min_percentile=args.hard_negative_min_percentile,
        hard_negative_score_gamma=args.hard_negative_score_gamma,
        hard_negative_max_multiplier=hard_negative_max_multiplier,
    )
    train_loader = make_loader(train_dataset, args.batch_size, False, sampler, device)
    val_loader = make_loader(val_dataset, max(1, args.batch_size), False, None, device)
    no_fall_loader: Optional[DataLoader] = None
    if len(external_no_fall_windows) > 0:
        no_fall_tensor = torch.from_numpy(external_no_fall_windows[:, :, np.newaxis, :, :]).float()
        no_fall_dataset = TensorDataset(no_fall_tensor)
        no_fall_loader = DataLoader(
            no_fall_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    model = FallEventDetector(
        hidden_dim=args.hidden_dim,
        num_classes=len(LABEL_ID_TO_NAME),
        use_snn_temporal_branch=args.use_snn_temporal_branch,
        snn_hidden_dim=args.snn_hidden_dim,
        snn_decay=args.snn_decay,
        snn_threshold=args.snn_threshold,
        snn_surrogate_scale=args.snn_surrogate_scale,
        learnable_snn_dynamics=args.learnable_snn_dynamics,
        use_snn_fusion_gate=args.use_snn_fusion_gate,
        snn_fusion_gate_mode=args.snn_fusion_gate_mode,
    ).to(device)
    if args.init_model_path is not None:
        if not args.init_model_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_model_path}")
        try:
            init_checkpoint = torch.load(args.init_model_path, map_location=device, weights_only=False)
        except TypeError:
            init_checkpoint = torch.load(args.init_model_path, map_location=device)
        missing_keys, unexpected_keys, skipped_keys = load_compatible_model_state(model, init_checkpoint["model_state"])
        log(
            f"Initialized model from {args.init_model_path} | "
            f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)} skipped_shape_keys={len(skipped_keys)}"
        )
    teacher_model: Optional[FallEventDetector] = None
    if args.teacher_student_noise:
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(args.lr * 0.05, 1e-6),
    )
    class_weights = compute_class_weights(
        [int(sample["label_id"]) for sample in train_dataset.samples],
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_score = float("-inf")
    patience_count = 0
    history: list[dict[str, float]] = []
    last_freeze_state: Optional[bool] = None

    for epoch in range(1, args.epochs + 1):
        freeze_cnn_gru = bool(args.use_snn_temporal_branch and epoch <= max(0, int(args.freeze_cnn_gru_epochs)))
        if last_freeze_state is None or freeze_cnn_gru != last_freeze_state:
            freeze_summary = set_cnn_gru_frozen(model, freeze_cnn_gru)
            state_text = "frozen" if freeze_cnn_gru else "unfrozen"
            log(
                f"CNN/GRU freeze schedule | epoch={epoch} state={state_text} "
                f"trainable={freeze_summary['trainable_parameters']} frozen={freeze_summary['frozen_parameters']}"
            )
            last_freeze_state = freeze_cnn_gru
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        external_no_fall_loss_sum = 0.0
        no_fall_iterator = iter(no_fall_loader) if no_fall_loader is not None else None

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            student_x = batch_x
            if args.teacher_student_noise:
                student_x = apply_student_noise(
                    batch_x,
                    noise_std=args.student_noise_std,
                    noise_prob=args.student_noise_prob,
                    frame_drop_prob=args.student_frame_drop_prob,
                    noise_mode=args.student_noise_mode,
                    device_noise_profile=device_noise_profile,
                    device_noise_scale=args.device_noise_scale,
                )
            logits = model(student_x)
            supervised_loss = criterion(logits, batch_y)
            loss = supervised_loss
            if args.normal_positive_penalty > 0:
                normal_mask = batch_y == LABEL_NAME_TO_ID["normal"]
                if bool(normal_mask.any()):
                    normal_probs = torch.softmax(logits[normal_mask], dim=1)
                    normal_positive_mass = (
                        normal_probs[:, LABEL_NAME_TO_ID["pre_fall"]]
                        + normal_probs[:, LABEL_NAME_TO_ID["fall"]]
                    )
                    loss = loss + float(args.normal_positive_penalty) * normal_positive_mass.mean()
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_x)
                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                consistency_weight = consistency_ramp_weight(
                    args.consistency_weight,
                    epoch=epoch,
                    ramp_epochs=args.consistency_ramp_epochs,
                )
                consistency_loss = F.mse_loss(torch.softmax(logits, dim=1), teacher_probs)
                # Keep any previously added penalties, then add teacher-student consistency.
                loss = loss + consistency_weight * consistency_loss
            if no_fall_iterator is not None and args.external_no_fall_weight > 0:
                try:
                    (no_fall_batch,) = next(no_fall_iterator)
                except StopIteration:
                    no_fall_iterator = iter(no_fall_loader)
                    (no_fall_batch,) = next(no_fall_iterator)
                no_fall_batch = no_fall_batch.to(device, non_blocking=True)
                no_fall_student = no_fall_batch
                if args.teacher_student_noise:
                    no_fall_student = apply_student_noise(
                        no_fall_batch,
                        noise_std=args.student_noise_std,
                        noise_prob=args.student_noise_prob,
                        frame_drop_prob=args.student_frame_drop_prob,
                        noise_mode=args.student_noise_mode,
                        device_noise_profile=device_noise_profile,
                        device_noise_scale=args.device_noise_scale,
                    )
                no_fall_logits = model(no_fall_student)
                no_fall_probs = torch.softmax(no_fall_logits, dim=1)
                positive_mass = (
                    no_fall_probs[:, LABEL_NAME_TO_ID["pre_fall"]]
                    + no_fall_probs[:, LABEL_NAME_TO_ID["fall"]]
                )
                no_fall_loss = positive_mass.mean() + 0.5 * no_fall_probs[:, LABEL_NAME_TO_ID["fall"]].mean()
                loss = loss + float(args.external_no_fall_weight) * no_fall_loss
                external_no_fall_loss_sum += float(no_fall_loss.item()) * int(no_fall_batch.size(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            if teacher_model is not None:
                update_ema_teacher(model, teacher_model, ema_decay=args.teacher_ema)

            preds = torch.argmax(logits, dim=1)
            batch_size = int(batch_x.size(0))
            train_loss_sum += float(loss.item()) * batch_size
            train_correct += int((preds == batch_y).sum().item())
            train_total += batch_size

        scheduler.step()

        train_metrics = {
            "train_loss": train_loss_sum / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
        }
        if len(external_no_fall_windows) > 0:
            train_metrics["external_no_fall_loss"] = external_no_fall_loss_sum / max(len(external_no_fall_windows), 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device) if len(val_dataset) > 0 else {
            "loss": float("nan"),
            "accuracy": float("nan"),
        }

        score = val_metrics.get("balanced_accuracy", val_metrics["accuracy"])
        if math.isnan(score):
            score = -train_metrics["train_loss"]

        history_row = {
            "epoch": float(epoch),
            **train_metrics,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", float("nan"))),
            "val_normal_accuracy": float(val_metrics.get("normal_accuracy", float("nan"))),
            "val_fall_accuracy": float(val_metrics.get("fall_accuracy", float("nan"))),
        }
        history.append(history_row)
        log(
            f"Epoch {epoch:02d} | train_loss={history_row['train_loss']:.4f} "
            f"train_acc={history_row['train_accuracy']:.3f} "
            f"val_loss={history_row['val_loss']:.4f} val_acc={history_row['val_accuracy']:.3f} "
            f"val_bal={history_row['val_balanced_accuracy']:.3f} "
            f"val_normal={history_row['val_normal_accuracy']:.3f} val_fall={history_row['val_fall_accuracy']:.3f}"
        )

        if score > best_score:
            best_score = score
            patience_count = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": {
                    "window_size": args.window_size,
                    "image_size": args.image_size,
                    "hidden_dim": args.hidden_dim,
                    "use_snn_temporal_branch": bool(args.use_snn_temporal_branch),
                    "snn_hidden_dim": int(args.snn_hidden_dim),
                    "snn_decay": float(args.snn_decay),
                    "snn_threshold": float(args.snn_threshold),
                    "snn_surrogate_scale": float(args.snn_surrogate_scale),
                    "learnable_snn_dynamics": bool(args.learnable_snn_dynamics),
                    "use_snn_fusion_gate": bool(args.use_snn_fusion_gate),
                    "snn_fusion_gate_mode": args.snn_fusion_gate_mode,
                    "freeze_cnn_gru_epochs": max(0, int(args.freeze_cnn_gru_epochs)),
                    "use_crossbar": use_crossbar,
                    "use_device_dynamics": bool(crossbar_device_dynamics is not None),
                    "device_dynamics_sources": (
                        crossbar_device_dynamics.get("source_files", {}) if crossbar_device_dynamics is not None else {}
                    ),
                    "crossbar_readout_noise_scale": float(np.clip(args.crossbar_readout_noise_scale, 0.0, 2.0)),
                    "use_pixel_human": args.use_pixel_human,
                    "use_silhouette_human": args.use_silhouette_human,
                    "pixel_grid_size": args.pixel_grid_size,
                    "infer_stride": args.infer_stride,
                    "video_window_stride": args.video_window_stride,
                    "max_aux_fall": args.max_aux_fall,
                    "skip_main_video_training": args.skip_main_video_training,
                    "teacher_student_noise": args.teacher_student_noise,
                    "student_noise_mode": args.student_noise_mode,
                    "student_noise_std": args.student_noise_std,
                    "student_noise_prob": args.student_noise_prob,
                    "student_frame_drop_prob": args.student_frame_drop_prob,
                    "device_noise_scale": float(np.clip(args.device_noise_scale, 0.0, 2.0)),
                    "teacher_ema": args.teacher_ema,
                    "consistency_weight": args.consistency_weight,
                    "external_no_fall_weight": args.external_no_fall_weight,
                    "hard_negative_normal_weight": float(args.hard_negative_normal_weight),
                    "hard_negative_score_mode": args.hard_negative_score_mode,
                    "hard_negative_min_percentile": float(np.clip(args.hard_negative_min_percentile, 0.0, 0.95)),
                    "hard_negative_score_gamma": max(0.05, float(args.hard_negative_score_gamma)),
                    "hard_negative_max_multiplier": hard_negative_max_multiplier,
                    "adl_context_path_keyword": args.adl_context_path_keyword,
                    "adl_context_min_percentile": float(np.clip(args.adl_context_min_percentile, 0.0, 0.95)),
                    "adl_context_radius_windows": max(0, int(args.adl_context_radius_windows)),
                    "adl_context_extra_copies": max(0, int(args.adl_context_extra_copies)),
                    "normal_positive_penalty": float(args.normal_positive_penalty),
                    "validation_split": validation_split_summary,
                },
                "class_names": LABEL_ID_TO_NAME,
            }
            torch.save(checkpoint, args.model_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log(f"Early stopping triggered at epoch {epoch}.")
                break

    return TrainArtifacts(
        checkpoint_path=args.model_path,
        history=history,
        train_counts=class_count_dict([int(sample["label_id"]) for sample in train_dataset.samples]),
        val_counts=class_count_dict([int(sample["label_id"]) for sample in val_dataset.samples]),
        aux_summary=aux_summary,
        running_summary=running_summary,
        external_labeled_summary=external_labeled_summary,
        external_val_labeled_summary={
            **external_val_labeled_summary,
            "validation_split": validation_split_summary,
        },
        external_no_fall_summary=external_no_fall_summary,
    )


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FallEventDetector, dict[str, object]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    model = FallEventDetector(
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_classes=len(LABEL_ID_TO_NAME),
        use_snn_temporal_branch=bool(config.get("use_snn_temporal_branch", False)),
        snn_hidden_dim=int(config.get("snn_hidden_dim", 32)),
        snn_decay=float(config.get("snn_decay", 0.82)),
        snn_threshold=float(config.get("snn_threshold", 0.55)),
        snn_surrogate_scale=float(config.get("snn_surrogate_scale", 8.0)),
        learnable_snn_dynamics=bool(config.get("learnable_snn_dynamics", False)),
        use_snn_fusion_gate=bool(config.get("use_snn_fusion_gate", False)),
        snn_fusion_gate_mode=str(config.get("snn_fusion_gate_mode", "scalar")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def read_video_frames(
    video_path: Path,
    image_size: int,
    use_pixel_human: bool = False,
    use_silhouette_human: bool = False,
    pixel_grid_size: int = 20,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    raw_frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_frames.append(gray)
    finally:
        cap.release()

    if not raw_frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if fps <= 0:
        raise ValueError(f"Unable to read FPS from {video_path}")
    if use_silhouette_human:
        resized_gray_frames = np.stack(
            [cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA) for gray in raw_frames],
            axis=0,
        ).astype(np.uint8)
        background = estimate_video_background(resized_gray_frames)
        frames = [
            silhouette_human_render(
                gray,
                image_size=image_size,
                background=background,
            )
            for gray in resized_gray_frames
        ]
    elif use_pixel_human:
        resized_gray_frames = np.stack(
            [cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA) for gray in raw_frames],
            axis=0,
        ).astype(np.uint8)
        background = estimate_video_background(resized_gray_frames)
        frames = [
            pixel_human_render(
                gray,
                image_size=image_size,
                pixel_grid_size=pixel_grid_size,
                background=background,
            )
            for gray in resized_gray_frames
        ]
    else:
        frames = [
            (cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).astype(np.float32)
            for gray in raw_frames
        ]
    return np.stack(frames, axis=0).astype(np.float32), fps


def smooth_probabilities(frame_probs: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if len(frame_probs) < 3 or kernel_size <= 1:
        return frame_probs
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.zeros_like(frame_probs)
    for class_index in range(frame_probs.shape[1]):
        padded = np.pad(frame_probs[:, class_index], (kernel_size // 2, kernel_size // 2), mode="edge")
        smoothed[:, class_index] = np.convolve(padded, kernel, mode="valid")[: len(frame_probs)]
    return smoothed


def calibrate_frame_predictions(
    frame_probs: np.ndarray,
    fps: float,
    fall_prob_threshold: float = 0.0,
    fall_margin_threshold: float = 0.0,
    min_fall_duration_sec: float = 0.0,
) -> np.ndarray:
    frame_preds = np.argmax(frame_probs, axis=1).astype(np.int64)
    if len(frame_preds) == 0:
        return frame_preds

    positive_labels = {LABEL_NAME_TO_ID["pre_fall"], LABEL_NAME_TO_ID["fall"]}
    fallback_labels = [LABEL_NAME_TO_ID["normal"], LABEL_NAME_TO_ID["running"]]
    prob_threshold = float(np.clip(fall_prob_threshold, 0.0, 1.0))
    margin_threshold = float(max(0.0, fall_margin_threshold))

    for frame_index, label_id in enumerate(frame_preds.tolist()):
        if label_id not in positive_labels:
            continue
        positive_score = float(frame_probs[frame_index, label_id])
        fallback_scores = frame_probs[frame_index, fallback_labels]
        fallback_label = fallback_labels[int(np.argmax(fallback_scores))]
        fallback_score = float(np.max(fallback_scores))
        if positive_score < prob_threshold or positive_score < fallback_score + margin_threshold:
            frame_preds[frame_index] = fallback_label

    min_frames = int(math.ceil(max(0.0, min_fall_duration_sec) * max(fps, 1e-6)))
    if min_frames <= 1:
        return frame_preds

    start = 0
    current_label = int(frame_preds[0])
    for index in range(1, len(frame_preds) + 1):
        next_label = int(frame_preds[index]) if index < len(frame_preds) else -1
        if next_label == current_label:
            continue
        if current_label in positive_labels and index - start < min_frames:
            fallback_slice = frame_probs[start:index][:, fallback_labels]
            fallback_indices = np.argmax(fallback_slice, axis=1)
            frame_preds[start:index] = np.asarray([fallback_labels[int(item)] for item in fallback_indices], dtype=np.int64)
        start = index
        current_label = next_label
    return frame_preds


def predict_video_frames(
    model: nn.Module,
    frames: np.ndarray,
    fps: float,
    window_size: int,
    stride: int,
    use_crossbar: bool,
    hrs_values: np.ndarray,
    lrs_values: np.ndarray,
    device_dynamics: Optional[dict[str, object]],
    crossbar_readout_noise_scale: float,
    device: torch.device,
    batch_size: int,
    fall_prob_threshold: float = 0.0,
    fall_margin_threshold: float = 0.0,
    min_fall_duration_sec: float = 0.0,
) -> dict[str, object]:
    if len(frames) < window_size:
        raise ValueError("Video is shorter than the model window size.")

    starts = list(range(0, len(frames) - window_size + 1, stride))
    frame_probs = np.zeros((len(frames), len(LABEL_ID_TO_NAME)), dtype=np.float32)
    frame_counts = np.zeros(len(frames), dtype=np.float32)
    window_predictions = []

    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch_windows = []
            for start in batch_starts:
                window_frames = frames[start : start + window_size].copy()
                if use_crossbar:
                    window_frames = build_crossbar_window(
                        window_frames,
                        hrs_values,
                        lrs_values,
                        seed=10000 + start,
                        device_dynamics=device_dynamics,
                        readout_noise_scale=crossbar_readout_noise_scale,
                    )
                batch_windows.append(window_frames[:, np.newaxis, :, :])
            batch_tensor = torch.from_numpy(np.stack(batch_windows, axis=0)).float().to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for start, prob in zip(batch_starts, probs):
                end = start + window_size
                frame_probs[start:end] += prob[np.newaxis, :]
                frame_counts[start:end] += 1.0
                window_predictions.append(
                    {
                        "start_frame": int(start),
                        "end_frame": int(end - 1),
                        "start_time_sec": round(start / fps, 4),
                        "end_time_sec": round((end - 1) / fps, 4),
                        "probabilities": {
                            LABEL_ID_TO_NAME[class_id]: round(float(prob[class_id]), 6)
                            for class_id in LABEL_ID_TO_NAME
                        },
                        "predicted_label": LABEL_ID_TO_NAME[int(np.argmax(prob))],
                    }
                )

    frame_counts = np.maximum(frame_counts, 1.0)
    frame_probs /= frame_counts[:, np.newaxis]
    frame_probs = smooth_probabilities(frame_probs, kernel_size=5)
    frame_preds = calibrate_frame_predictions(
        frame_probs,
        fps=fps,
        fall_prob_threshold=fall_prob_threshold,
        fall_margin_threshold=fall_margin_threshold,
        min_fall_duration_sec=min_fall_duration_sec,
    )

    segments = []
    if len(frame_preds) > 0:
        start = 0
        current_label = int(frame_preds[0])
        for index in range(1, len(frame_preds) + 1):
            next_label = int(frame_preds[index]) if index < len(frame_preds) else -1
            if next_label != current_label:
                segments.append(
                    {
                        "label_id": current_label,
                        "label_name": LABEL_ID_TO_NAME[current_label],
                        "start_frame": start,
                        "end_frame": index - 1,
                        "start_time_sec": round(start / fps, 4),
                        "end_time_sec": round((index - 1) / fps, 4),
                        "num_frames": int(index - start),
                    }
                )
                start = index
                current_label = next_label

    return {
        "fps": fps,
        "num_frames": int(len(frames)),
        "frame_probabilities": frame_probs,
        "frame_predictions": frame_preds,
        "window_predictions": window_predictions,
        "segments": segments,
    }


def compute_frame_metrics(predicted: np.ndarray, ground_truth: Optional[np.ndarray]) -> Optional[dict[str, object]]:
    if ground_truth is None or len(predicted) != len(ground_truth):
        return None

    accuracy = float(np.mean(predicted == ground_truth))
    confusion = np.zeros((len(LABEL_ID_TO_NAME), len(LABEL_ID_TO_NAME)), dtype=np.int64)
    for truth, pred in zip(ground_truth.tolist(), predicted.tolist()):
        confusion[int(truth), int(pred)] += 1

    class_accuracy = {}
    for label_id, label_name in LABEL_ID_TO_NAME.items():
        mask = ground_truth == label_id
        class_accuracy[label_name] = float(np.mean(predicted[mask] == ground_truth[mask])) if np.any(mask) else None

    return {
        "frame_accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def load_ltp_pulse_curve(ltp_file: Path) -> dict[str, np.ndarray]:
    if not ltp_file.exists():
        raise FileNotFoundError(f"LTP file not found: {ltp_file}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        excel_file = pd.ExcelFile(ltp_file)
    run_sheet = next((name for name in excel_file.sheet_names if str(name).lower().startswith("run")), excel_file.sheet_names[0])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = pd.read_excel(ltp_file, sheet_name=run_sheet)

    required_columns = {"Time", "DrainI", "DrainV"}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"LTP file {ltp_file} missing required columns: {required_columns}")

    run_df = df.loc[:, ["Time", "DrainI", "DrainV"]].copy()
    run_df["Time"] = pd.to_numeric(run_df["Time"], errors="coerce")
    run_df["DrainI"] = pd.to_numeric(run_df["DrainI"], errors="coerce")
    run_df["DrainV"] = pd.to_numeric(run_df["DrainV"], errors="coerce").fillna(0.0)
    run_df = run_df.dropna(subset=["Time", "DrainI"]).reset_index(drop=True)

    pulse_mask = np.abs(run_df["DrainV"].to_numpy(dtype=np.float32)) > 1e-12
    pulse_indices = np.flatnonzero(pulse_mask)
    if len(pulse_indices) == 0:
        raise ValueError(f"No non-zero DrainV pulse rows found in {ltp_file}")

    groups: list[np.ndarray] = []
    current_group = [int(pulse_indices[0])]
    for index in pulse_indices[1:]:
        if int(index) == current_group[-1] + 1:
            current_group.append(int(index))
        else:
            groups.append(np.asarray(current_group, dtype=np.int64))
            current_group = [int(index)]
    groups.append(np.asarray(current_group, dtype=np.int64))

    if len(groups) <= 1:
        pulse_currents = run_df.loc[pulse_mask, "DrainI"].to_numpy(dtype=np.float32)
        pulse_times = run_df.loc[pulse_mask, "Time"].to_numpy(dtype=np.float32)
    else:
        pulse_currents = np.asarray(
            [float(run_df.iloc[group]["DrainI"].max()) for group in groups],
            dtype=np.float32,
        )
        pulse_times = np.asarray(
            [float(run_df.iloc[group]["Time"].mean()) for group in groups],
            dtype=np.float32,
        )
    pulse_currents = np.maximum.accumulate(pulse_currents)
    if len(pulse_currents) > 1:
        normalized = (pulse_currents - pulse_currents.min()) / max(float(pulse_currents.max() - pulse_currents.min()), 1e-6)
    else:
        normalized = np.ones_like(pulse_currents, dtype=np.float32)
    normalized = np.maximum.accumulate(normalized.astype(np.float32))
    return {
        "pulse_times": pulse_times.astype(np.float32),
        "pulse_currents": pulse_currents.astype(np.float32),
        "normalized_gain": normalized.astype(np.float32),
        "num_pulses": np.asarray([len(pulse_currents)], dtype=np.int64),
    }


def resample_curve_to_frames(curve: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(curve) == 0:
        return np.ones(num_frames, dtype=np.float32)
    if len(curve) == 1:
        return np.full(num_frames, float(curve[0]), dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=len(curve), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=num_frames, dtype=np.float32)
    return np.interp(target_x, source_x, curve.astype(np.float32)).astype(np.float32)


def build_ltp_threshold_curve(danger_score: np.ndarray, ltp_gain: np.ndarray) -> np.ndarray:
    base_threshold = float(np.quantile(danger_score, 0.78))
    base_threshold = float(np.clip(base_threshold, 0.18, 0.72))
    dynamic_scale = 1.05 - 0.34 * ltp_gain
    threshold = np.clip(base_threshold * dynamic_scale, 0.12, 0.82)
    return threshold.astype(np.float32)


def build_fall_specific_score(
    pre_fall_score: np.ndarray,
    fall_score: np.ndarray,
    running_score: np.ndarray,
    suppression: float,
) -> np.ndarray:
    danger_score = np.clip(pre_fall_score + fall_score, 0.0, 1.0)
    fall_specific = np.clip(danger_score - suppression * running_score, 0.0, 1.0)
    return fall_specific.astype(np.float32)


def detect_local_peaks(signal: np.ndarray, percentile: float, min_distance: int = 4) -> np.ndarray:
    if len(signal) == 0:
        return np.zeros(0, dtype=np.int64)
    threshold = float(np.quantile(signal, percentile))
    threshold = max(threshold, float(signal.mean()))
    peaks: list[int] = []
    last_kept = -min_distance
    for index in range(1, len(signal) - 1):
        if signal[index] < threshold:
            continue
        if signal[index] < signal[index - 1] or signal[index] < signal[index + 1]:
            continue
        if index - last_kept < min_distance:
            if peaks and signal[index] > signal[peaks[-1]]:
                peaks[-1] = index
                last_kept = index
            continue
        peaks.append(index)
        last_kept = index
    if not peaks:
        return np.asarray([int(np.argmax(signal))], dtype=np.int64)
    return np.asarray(peaks, dtype=np.int64)


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)
    diff = np.abs(np.diff(frames, axis=0))
    energy = np.concatenate(
        [
            np.zeros(1, dtype=np.float32),
            diff.reshape(diff.shape[0], -1).mean(axis=1).astype(np.float32),
        ]
    )
    if len(energy) > 1:
        energy = (energy - energy.min()) / max(float(energy.max() - energy.min()), 1e-6)
    return energy.astype(np.float32)


def build_stdp_gain(
    motion_energy: np.ndarray,
    danger_score: np.ndarray,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
) -> dict[str, np.ndarray]:
    num_frames = len(danger_score)
    pre_events = detect_local_peaks(motion_energy, percentile=0.72, min_distance=4)
    danger_gradient = np.gradient(danger_score.astype(np.float32))
    post_signal = np.clip(0.65 * danger_score + 0.35 * np.maximum(danger_gradient, 0.0), 0.0, None)
    post_events = detect_local_peaks(post_signal, percentile=0.76, min_distance=5)

    event_drive = np.zeros(num_frames, dtype=np.float32)
    window_spread = max(2.0, 0.5 * (tau_plus + tau_minus))
    for post_index in post_events.tolist():
        if len(pre_events) == 0:
            break
        nearest_idx = int(np.argmin(np.abs(pre_events - post_index)))
        pre_index = int(pre_events[nearest_idx])
        delta_t = float(post_index - pre_index)
        if delta_t >= 0:
            amplitude = a_plus * math.exp(-delta_t / max(tau_plus, 1e-6))
        else:
            amplitude = -a_minus * math.exp(delta_t / max(tau_minus, 1e-6))
        if abs(amplitude) < 1e-6:
            continue
        support = np.arange(num_frames, dtype=np.float32)
        spread = np.exp(-np.abs(support - float(post_index)) / max(window_spread, 1e-6))
        event_drive += float(amplitude) * spread.astype(np.float32)

    event_drive = np.clip(event_drive, -0.45, 0.75)
    stdp_gain = np.clip(1.0 + event_drive, 0.72, 1.65).astype(np.float32)
    return {
        "stdp_gain": stdp_gain,
        "pre_events": pre_events.astype(np.int64),
        "post_events": post_events.astype(np.int64),
        "event_drive": event_drive.astype(np.float32),
    }


def apply_exponential_decay(
    combined_score: np.ndarray,
    motion_energy: np.ndarray,
    danger_score: np.ndarray,
    tau_decay: float,
    decay_floor: float = 0.55,
    event_percentile: float = 0.82,
) -> np.ndarray:
    if len(combined_score) == 0:
        return combined_score.astype(np.float32)
    activation_signal = np.clip(0.55 * motion_energy + 0.45 * danger_score, 0.0, 1.0)
    activation_events = detect_local_peaks(
        activation_signal,
        percentile=float(np.clip(event_percentile, 0.55, 0.95)),
        min_distance=5,
    )
    activation_set = {int(index) for index in activation_events.tolist()}
    last_event_index = -10_000
    decayed = np.zeros_like(combined_score, dtype=np.float32)
    floor = float(np.clip(decay_floor, 0.0, 0.95))
    for index, (score, activation) in enumerate(zip(combined_score.tolist(), activation_signal.tolist())):
        if index in activation_set:
            last_event_index = index
        elapsed = max(0, index - last_event_index)
        decay_scale = math.exp(-elapsed / max(tau_decay, 1e-6))
        attenuation = floor + (1.0 - floor) * decay_scale
        attenuated_score = float(score) * float(np.clip(attenuation, floor, 1.0))
        decayed[index] = max(attenuated_score, 0.10 * float(activation))
    return np.clip(decayed, 0.0, 1.0).astype(np.float32)


def summarize_tracking_mask(tracking_mask: np.ndarray, positive_mask: np.ndarray) -> dict[str, object]:
    tracked_indices = np.flatnonzero(tracking_mask)
    overlap = tracking_mask & positive_mask
    overlap_count = int(overlap.sum())
    tracked_count = int(tracking_mask.sum())
    positive_count = int(positive_mask.sum())
    union = int((tracking_mask | positive_mask).sum())
    return {
        "first_tracked_frame": int(tracked_indices[0]) if len(tracked_indices) > 0 else None,
        "tracked_frames": tracked_count,
        "overlap_frames": overlap_count,
        "overlap_ratio_vs_positive": float(overlap_count / max(positive_count, 1)),
        "precision_vs_positive": float(overlap_count / max(tracked_count, 1)),
        "iou_vs_positive": float(overlap_count / max(union, 1)),
    }


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def analyze_plasticity_method(
    method_name: str,
    danger_score: np.ndarray,
    running_score: np.ndarray,
    frames: np.ndarray,
    ltp_gain: np.ndarray,
    threshold_curve: np.ndarray,
    positive_mask: np.ndarray,
    stdp_payload: Optional[dict[str, np.ndarray]] = None,
    decay_tau: float = 8.0,
    decay_floor: float = 0.55,
    decay_event_percentile: float = 0.82,
    running_gate_margin: float = 0.04,
) -> dict[str, object]:
    adjusted_score = danger_score.astype(np.float32).copy()
    method_threshold = threshold_curve.astype(np.float32).copy()
    stdp_gain = np.ones_like(danger_score, dtype=np.float32)
    if stdp_payload is not None:
        stdp_gain = stdp_payload["stdp_gain"].astype(np.float32)
        adjusted_score = np.clip(adjusted_score * ltp_gain * stdp_gain, 0.0, 1.0)
    else:
        adjusted_score = np.clip(adjusted_score * ltp_gain, 0.0, 1.0)

    motion_energy = compute_motion_energy(frames)
    if method_name in {"stdp_ltp_decay", "ltp_decay"}:
        adjusted_score = apply_exponential_decay(
            adjusted_score,
            motion_energy,
            danger_score,
            tau_decay=decay_tau,
            decay_floor=decay_floor,
            event_percentile=decay_event_percentile,
        )

    running_gate = adjusted_score >= (running_score.astype(np.float32) + float(running_gate_margin))
    tracking_mask = (adjusted_score >= method_threshold) & running_gate
    tracking_summary = summarize_tracking_mask(tracking_mask, positive_mask)
    return {
        "method_name": method_name,
        "danger_score": danger_score.astype(np.float32),
        "running_score": running_score.astype(np.float32),
        "adjusted_score": adjusted_score.astype(np.float32),
        "threshold_curve": method_threshold.astype(np.float32),
        "tracking_mask": tracking_mask.astype(bool),
        "tracking_summary": tracking_summary,
        "max_adjusted_score": float(adjusted_score.max()) if len(adjusted_score) > 0 else 0.0,
        "stdp_gain": stdp_gain.astype(np.float32),
        "motion_energy": motion_energy.astype(np.float32),
    }


def build_method_output_path(base_path: Path, method_name: str) -> Path:
    simple_name = {
        "ltp": "LTP",
        "ltp_decay": "LTP_DECAY",
        "stdp_ltp": "STDP_LTP",
        "stdp_ltp_decay": "STDP_LTP_DECAY",
    }[method_name]
    stem_lower = base_path.stem.lower()
    if "tracking" in stem_lower:
        return base_path.with_name(f"{simple_name}{base_path.suffix}")
    if "analysis" in stem_lower:
        return base_path.with_name(f"{simple_name}_curve{base_path.suffix}")
    return base_path.with_name(f"{simple_name}{base_path.suffix}")


def select_representative_tracking_frames(result: dict[str, object], count: int = 3) -> list[int]:
    mask = np.flatnonzero(result["tracking_mask"])
    if len(mask) == 0:
        return [0] * count
    if len(mask) == 1:
        return [int(mask[0])] * count
    selected = np.linspace(0, len(mask) - 1, num=count)
    return [int(mask[int(round(value))]) for value in selected.tolist()]


def save_individual_plasticity_plot(
    plot_path: Path,
    video_path: Path,
    fps: float,
    ltp_payload: dict[str, np.ndarray],
    ltp_gain: np.ndarray,
    positive_mask: np.ndarray,
    result: dict[str, object],
) -> None:
    method_name = str(result["method_name"])
    panel_title = {
        "ltp": "(a) LTP",
        "ltp_decay": "(b) LTP + decay",
        "stdp_ltp": "(b) STDP + LTP",
        "stdp_ltp_decay": "(c) STDP + LTP + decay",
    }[method_name]
    times = np.arange(len(positive_mask), dtype=np.float32) / max(fps, 1e-6)
    pulse_index = np.arange(1, len(ltp_payload["pulse_currents"]) + 1, dtype=np.int32)
    color = {
        "ltp": "#d62728",
        "ltp_decay": "#e07a5f",
        "stdp_ltp": "#1f77b4",
        "stdp_ltp_decay": "#2ca02c",
    }[method_name]

    figure = plt.figure(figsize=(11.2, 7.8))
    grid = figure.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.95, 1.55, 1.15],
        hspace=0.34,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[2, 0]),
    ]

    axes[0].plot(pulse_index, ltp_payload["pulse_currents"], color="#cc2f2f", linewidth=2.2, label="LTP current")
    axes[0].plot(pulse_index, ltp_payload["normalized_gain"], color="#f08c00", linewidth=1.8, label="LTP gain")
    axes[0].set_title(panel_title, loc="left", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Pulse index")
    axes[0].set_ylabel("Response")
    axes[0].grid(alpha=0.16, linestyle="--", linewidth=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, ncol=2)

    axes[1].plot(times, result["danger_score"], color="#8a8a8a", linewidth=1.4, alpha=0.78, label="Fall-specific score")
    axes[1].plot(times, result["running_score"], color="#42a5f5", linewidth=1.2, alpha=0.82, label="Running score")
    axes[1].plot(times, result["adjusted_score"], color=color, linewidth=2.3, label="Tracked score")
    axes[1].plot(times, result["threshold_curve"], color=color, linewidth=1.5, linestyle="--", label="Dynamic threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.16, linestyle="--", linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper left", ncol=2, frameon=False)

    axes[2].plot(times, ltp_gain, color="#f08c00", linewidth=1.9, label="LTP gain")
    if method_name != "ltp":
        axes[2].plot(times, result["stdp_gain"], color="#6a4c93", linewidth=1.7, label="STDP gain")
    axes[2].plot(times, result["motion_energy"], color="#2a9d8f", linewidth=1.3, alpha=0.88, label="Motion energy")
    axes[2].fill_between(times, 0.0, result["tracking_mask"].astype(np.float32), step="post", color="#d62728", alpha=0.12, label="Tracked interval")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gain")
    axes[2].set_ylim(0.0, max(1.7, float(np.max(result["stdp_gain"])) + 0.1))
    axes[2].grid(alpha=0.16, linestyle="--", linewidth=0.8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].legend(loc="upper left", ncol=4, frameon=False, fontsize=9)

    summary = result["tracking_summary"]
    summary_text = (
        f"first tracked frame: {summary['first_tracked_frame']}   "
        f"tracked frames: {summary['tracked_frames']}   "
        f"IoU: {summary['iou_vs_positive']:.4f}"
    )
    figure.text(0.02, 0.015, summary_text, fontsize=10)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    ensure_parent_dir(plot_path)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def save_plasticity_method_frames(
    plot_path: Path,
    video_path: Path,
    picture_dir: Path,
    fps: float,
    frame_label_ids: np.ndarray,
    result: dict[str, object],
) -> None:
    panels, mapped_indices, tracked_flags = build_tracking_display_panels(
        picture_dir=picture_dir,
        frame_label_ids=frame_label_ids,
        tracking_mask=np.asarray(result["tracking_mask"], dtype=bool),
        fps=fps,
        panel_width=94,
        panel_height=94,
        header_height=10,
        footer_height=20,
    )
    if panels:
        columns = 6
        rows = int(math.ceil(len(panels) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.3 * rows))
        axes_array = np.atleast_1d(axes).reshape(rows, columns)
        for axis in axes_array.flatten():
            axis.axis("off")
        for axis, panel in zip(axes_array.flatten(), panels):
            axis.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            axis.axis("off")
        summary = result["tracking_summary"]
        mapped_count = int(sum(tracked_flags))
        figure.suptitle(
            f"{str(result['method_name']).upper()} | red border = tracked | mapped tracked panels = {mapped_count}",
            x=0.02,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
        figure.text(
            0.02,
            0.02,
            (
                f"top bar = class label | first tracked frame = {summary['first_tracked_frame']} | "
                f"tracked frames = {summary['tracked_frames']} | IoU = {summary['iou_vs_positive']:.4f}"
            ),
            fontsize=10,
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        ensure_parent_dir(plot_path)
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
        return

    tracked_frames = np.flatnonzero(result["tracking_mask"])
    if len(tracked_frames) == 0:
        tracked_frames = np.asarray([0], dtype=np.int64)
    columns = 5
    rows = int(math.ceil(len(tracked_frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.1 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flatten():
        axis.axis("off")
    for axis, frame_index in zip(axes_array.flatten(), tracked_frames.tolist()):
        frame = read_original_video_frame(video_path, frame_index)
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (14, 14), (width - 14, height - 14), (36, 36, 255), thickness=5)
        label_text = f"f{frame_index} | {frame_index / max(fps, 1e-6):.2f}s"
        cv2.putText(frame, label_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.axis("off")
    figure.suptitle(
        f"{result['method_name']} tracked frames ({len(tracked_frames)})",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    ensure_parent_dir(plot_path)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def save_plasticity_analysis_plot(
    plot_path: Path,
    fps: float,
    ltp_payload: dict[str, np.ndarray],
    method_results: list[dict[str, object]],
    positive_mask: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(12, 9.4), sharex=False, gridspec_kw={"height_ratios": [1.0, 2.0, 1.05, 0.72]})

    pulse_index = np.arange(1, len(ltp_payload["pulse_currents"]) + 1, dtype=np.int32)
    axes[0].plot(pulse_index, ltp_payload["pulse_currents"], color="#cc2f2f", linewidth=2.2, marker="o", label="LTP current")
    axes[0].plot(pulse_index, ltp_payload["normalized_gain"], color="#f08c00", linewidth=1.8, marker="s", label="LTP gain")
    axes[0].set_title("(d) Plasticity comparison overview", loc="left", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Pulse index")
    axes[0].set_ylabel("LTP")
    axes[0].grid(alpha=0.18, linestyle="--", linewidth=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False)

    times = np.arange(len(positive_mask), dtype=np.float32) / max(fps, 1e-6)
    colors = {
        "ltp": "#cc2f2f",
        "ltp_decay": "#e07a5f",
        "stdp_ltp": "#1f77b4",
        "stdp_ltp_decay": "#2ca25f",
    }
    for result in method_results:
        method_name = str(result["method_name"])
        label_text = {
            "ltp": "LTP",
            "ltp_decay": "LTP + decay",
            "stdp_ltp": "STDP + LTP",
            "stdp_ltp_decay": "STDP + LTP + decay",
        }[method_name]
        axes[1].plot(times, result["adjusted_score"], linewidth=2.2, color=colors[method_name], label=label_text)
        axes[1].plot(times, result["threshold_curve"], linewidth=1.2, linestyle="--", color=colors[method_name], alpha=0.85)
    axes[1].set_ylabel("Score / Threshold")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.18, linestyle="--", linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper left", ncol=2, frameon=False)

    row_height = 0.68
    for row_index, result in enumerate(method_results):
        method_name = str(result["method_name"])
        label_text = {
            "ltp": "LTP",
            "ltp_decay": "LTP + decay",
            "stdp_ltp": "STDP + LTP",
            "stdp_ltp_decay": "STDP + LTP + decay",
        }[method_name]
        y_base = float(len(method_results) - 1 - row_index)
        axes[2].barh(y_base + row_height / 2, times[-1] if len(times) > 0 else 0.0, left=0.0, height=row_height, color="#f5f5f5", edgecolor="none")
        for start_frame, end_frame in mask_to_segments(result["tracking_mask"]):
            start_time = start_frame / max(fps, 1e-6)
            width = (end_frame - start_frame + 1) / max(fps, 1e-6)
            axes[2].broken_barh([(start_time, width)], (y_base, row_height), facecolors=colors[method_name], alpha=0.92)
        axes[2].text(-0.18, y_base + row_height / 2, label_text, fontsize=9.5, color=colors[method_name], va="center", ha="right")
    axes[2].set_ylim(-0.1, len(method_results) + 0.15)
    axes[2].set_ylabel("Tracked")
    axes[2].set_yticks([])
    axes[2].set_xlim(0.0, times[-1] if len(times) > 0 else 1.0)
    axes[2].grid(alpha=0.10, linestyle="--", axis="x", linewidth=0.8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    axes[2].spines["left"].set_visible(False)

    axes[3].barh(0.5, times[-1] if len(times) > 0 else 0.0, left=0.0, height=0.48, color="#f3f4f6", edgecolor="none")
    if true_labels is not None and len(true_labels) == len(positive_mask):
        prefall_mask = np.asarray(true_labels == LABEL_NAME_TO_ID["pre_fall"], dtype=bool)
        fall_mask = np.asarray(true_labels == LABEL_NAME_TO_ID["fall"], dtype=bool)
        for start_frame, end_frame in mask_to_segments(prefall_mask):
            start_time = start_frame / max(fps, 1e-6)
            width = (end_frame - start_frame + 1) / max(fps, 1e-6)
            axes[3].broken_barh([(start_time, width)], (0.26, 0.48), facecolors="#f4a261", alpha=0.96)
        for start_frame, end_frame in mask_to_segments(fall_mask):
            start_time = start_frame / max(fps, 1e-6)
            width = (end_frame - start_frame + 1) / max(fps, 1e-6)
            axes[3].broken_barh([(start_time, width)], (0.26, 0.48), facecolors="#d62828", alpha=0.96)
        axes[3].text(-0.18, 0.5, "Ground truth", fontsize=9.5, color="#444444", va="center", ha="right")
    else:
        for start_frame, end_frame in mask_to_segments(positive_mask):
            start_time = start_frame / max(fps, 1e-6)
            width = (end_frame - start_frame + 1) / max(fps, 1e-6)
            axes[3].broken_barh([(start_time, width)], (0.26, 0.48), facecolors="#d62828", alpha=0.96)
        axes[3].text(-0.18, 0.5, "Positive labels", fontsize=9.5, color="#444444", va="center", ha="right")
    axes[3].set_ylabel("Reference")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yticks([])
    axes[3].set_xlim(0.0, times[-1] if len(times) > 0 else 1.0)
    axes[3].grid(alpha=0.10, linestyle="--", axis="x", linewidth=0.8)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)
    axes[3].spines["left"].set_visible(False)
    figure.tight_layout(rect=(0.06, 0.0, 1.0, 1.0))
    ensure_parent_dir(plot_path)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def save_plasticity_tracking_overview(
    video_path: Path,
    overview_path: Path,
    picture_dir: Path,
    fps: float,
    frame_label_ids: np.ndarray,
    method_results: list[dict[str, object]],
) -> None:
    if not method_results:
        return
    rows = len(method_results)
    figure, axes = plt.subplots(rows, 1, figsize=(15.5, 2.8 * rows))
    axes_array = np.atleast_1d(axes)
    figure.patch.set_facecolor("white")

    for axis, result in zip(axes_array, method_results):
        axis.axis("off")
        panels, mapped_indices, tracked_flags = build_tracking_display_panels(
            picture_dir=picture_dir,
            frame_label_ids=frame_label_ids,
            tracking_mask=np.asarray(result["tracking_mask"], dtype=bool),
            fps=fps,
            panel_width=28,
            panel_height=28,
            header_height=5,
            footer_height=0,
        )
        if panels:
            strip = compose_tracking_strip(panels, gap=2)
            axis.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
            summary = result["tracking_summary"]
            axis.set_title(
                (
                    f"{str(result['method_name']).upper()} | "
                    f"first = {summary['first_tracked_frame']} | "
                    f"tracked = {summary['tracked_frames']} | "
                    f"IoU = {summary['iou_vs_positive']:.4f}"
                ),
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )
            axis.text(
                0.995,
                -0.10,
                f"mapped panels tracked: {int(sum(tracked_flags))}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#555555",
            )
            continue

        mask = np.flatnonzero(result["tracking_mask"])
        chosen = [0] if len(mask) == 0 else [int(mask[0]), int(mask[min(len(mask) - 1, len(mask) // 2)]), int(mask[-1])]
        while len(chosen) < 3:
            chosen.append(chosen[-1])
        preview = []
        for frame_index in chosen[:3]:
            frame = read_original_video_frame(video_path, frame_index)
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (14, 14), (width - 14, height - 14), (36, 36, 255), thickness=5)
            preview.append(frame)
        axis.imshow(cv2.cvtColor(np.concatenate(preview, axis=1), cv2.COLOR_BGR2RGB))

    figure.suptitle("Plasticity tracking overview | top bar = class, red border = tracked", x=0.01, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    ensure_parent_dir(overview_path)
    figure.savefig(overview_path, dpi=180)
    plt.close(figure)


def write_plasticity_report_json(
    report_json: Path,
    args: argparse.Namespace,
    fps: float,
    ltp_payload: dict[str, np.ndarray],
    stdp_payload: dict[str, np.ndarray],
    method_results: list[dict[str, object]],
) -> None:
    report = {
        "video_path": str(args.video),
        "model_path": str(args.model_path),
        "ltp_file": str(args.ltp_file),
        "fps": float(fps),
        "ltp": {
            "num_pulses": int(ltp_payload["num_pulses"][0]),
            "pulse_currents": [float(value) for value in ltp_payload["pulse_currents"].tolist()],
            "normalized_gain": [float(value) for value in ltp_payload["normalized_gain"].tolist()],
        },
        "stdp": {
            "pre_events": [int(value) for value in stdp_payload["pre_events"].tolist()],
            "post_events": [int(value) for value in stdp_payload["post_events"].tolist()],
        },
        "methods": [],
    }
    for result in method_results:
        method_name = str(result["method_name"])
        report["methods"].append(
            {
                "method_name": method_name,
                "tracking_summary": result["tracking_summary"],
                "max_adjusted_score": float(result["max_adjusted_score"]),
                "max_running_score": float(np.max(result["running_score"])) if len(result["running_score"]) > 0 else 0.0,
                "plot_path": str(build_method_output_path(args.plasticity_plot_path, method_name)),
                "frames_path": str(build_method_output_path(args.plasticity_overview_path, method_name)),
            }
        )
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def print_plasticity_summary(method_results: list[dict[str, object]]) -> None:
    for result in method_results:
        summary = result["tracking_summary"]
        log(
            f"[{result['method_name']}] first_tracked_frame={summary['first_tracked_frame']} "
            f"tracked_frames={summary['tracked_frames']} "
            f"max_score={result['max_adjusted_score']:.4f} "
            f"overlap_ratio={summary['overlap_ratio_vs_positive']:.4f} "
            f"iou={summary['iou_vs_positive']:.4f}"
        )


def write_next_stage_plan_document(output_path: Path) -> None:
    content = """# Next Stage Roadmap: Consecutive Fall Event Detection

## Goal
Extend the current single-video fall detector to recognize two consecutive fall events after concatenating video A and video B, while keeping person identity separation as a second-stage objective.

## Stage 1: Consecutive event boundary detection
1. Concatenate two fall clips at frame level and keep both original annotation timelines.
2. Generate event-level labels with `event_id=1` and `event_id=2`.
3. Evaluate whether the detector outputs two independent process segments rather than one merged segment.
4. Report boundary deviation in frames for each event start and end.

## Stage 2: Same-person versus different-person discrimination
1. Freeze the event detector or reuse its process proposals.
2. Add lightweight appearance or pose descriptors on top of each detected event segment.
3. Compare whether two detected falls belong to the same person or different persons.
4. Use event-level metrics instead of frame-only metrics.

## Stage 3: Paper-style generalization study
1. Evaluate single-event, dual-event, cross-person and cross-scene settings separately.
2. Compare plain detector, LTP-enhanced detector, STDP+LTP and STDP+LTP+decay tracking.
3. Measure segment overlap, event recall, false merge rate and latency to first detection.

## Expected deliverables
- concatenated-video test set
- event-boundary evaluation script
- dual-event visualization report
- cross-person discrimination benchmark
"""
    output_path.write_text(content, encoding="utf-8")


def save_detection_plot(
    plot_path: Path,
    fps: float,
    frame_probs: np.ndarray,
    frame_preds: np.ndarray,
    true_labels: Optional[np.ndarray],
    show_plot: bool,
    save_plot: bool,
) -> None:
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    if frame_probs.ndim != 2 or frame_probs.shape[1] != len(LABEL_ID_TO_NAME):
        raise ValueError(f"Expected frame_probs in shape [N, {len(LABEL_ID_TO_NAME)}], got {frame_probs.shape}")

    frame_count = int(frame_probs.shape[0])
    fps = float(max(fps, 1e-6))

    ordered_ids = sorted(LABEL_ID_TO_NAME)
    class_names = [LABEL_ID_TO_NAME[idx] for idx in ordered_ids]
    heatmap = frame_probs[:, ordered_ids].T  # [C, N]

    # Slightly wide (paper-friendly) heatmap only (no pred/true stripe).
    figure, axis = plt.subplots(1, 1, figsize=(5.2, 3.6))
    prob_im = axis.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_ylabel("Class")
    axis.tick_params(axis="y", labelsize=9)

    tick_count = 5 if frame_count >= 5 else max(2, frame_count)
    tick_positions = np.linspace(0, max(frame_count - 1, 1), num=tick_count, dtype=int)
    tick_labels = [f"{pos / fps:.1f}" for pos in tick_positions.tolist()]
    axis.set_xticks(tick_positions, labels=tick_labels)
    axis.set_xlabel("Time (s)")
    axis.tick_params(axis="x", labelsize=9)

    cbar = figure.colorbar(prob_im, ax=axis, fraction=0.045, pad=0.02)
    cbar.set_label("P", rotation=0, labelpad=10)

    # Manual margins so long ytick labels (e.g. "pre_fall") are not clipped.
    figure.subplots_adjust(left=0.26, right=0.93, top=0.96, bottom=0.16)

    if save_plot:
        ensure_parent_dir(plot_path)
        figure.savefig(plot_path, dpi=180)
    if show_plot:
        plt.show()
    plt.close(figure)


def read_original_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return frame


def select_key_frame(frame_probs: np.ndarray) -> dict[str, object]:
    fall_index = LABEL_NAME_TO_ID["fall"]
    prefall_index = LABEL_NAME_TO_ID["pre_fall"]
    running_index = LABEL_NAME_TO_ID["running"]
    normal_index = LABEL_NAME_TO_ID["normal"]

    fall_peak = int(np.argmax(frame_probs[:, fall_index]))
    prefall_peak = int(np.argmax(frame_probs[:, prefall_index]))
    running_peak = int(np.argmax(frame_probs[:, running_index]))
    normal_peak = int(np.argmax(frame_probs[:, normal_index]))

    fall_score = float(frame_probs[fall_peak, fall_index])
    prefall_score = float(frame_probs[prefall_peak, prefall_index])
    running_score = float(frame_probs[running_peak, running_index])
    normal_score = float(frame_probs[normal_peak, normal_index])

    if fall_score >= max(0.35, prefall_score):
        chosen_class = "fall"
        chosen_index = fall_peak
        chosen_score = fall_score
    elif prefall_score >= max(0.35, normal_score):
        chosen_class = "pre_fall"
        chosen_index = prefall_peak
        chosen_score = prefall_score
    elif running_score >= max(0.35, normal_score):
        chosen_class = "running"
        chosen_index = running_peak
        chosen_score = running_score
    else:
        chosen_class = "normal"
        chosen_index = normal_peak
        chosen_score = normal_score

    return {
        "frame_index": int(chosen_index),
        "label_name": chosen_class,
        "score": float(chosen_score),
        "fall_peak_frame": int(fall_peak),
        "pre_fall_peak_frame": int(prefall_peak),
        "running_peak_frame": int(running_peak),
    }


def save_key_frame_visual(
    video_path: Path,
    keyframe_path: Path,
    key_frame: dict[str, object],
    fps: float,
) -> None:
    frame_index = int(key_frame["frame_index"])
    label_name = str(key_frame["label_name"])
    score = float(key_frame["score"])
    frame = read_original_video_frame(video_path, frame_index)

    color = {
        "normal": (80, 220, 80),
        "running": (255, 180, 0),
        "pre_fall": (0, 190, 255),
        "fall": (40, 80, 255),
    }[label_name]

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (18, 18), (width - 18, height - 18), color, thickness=6)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (min(width - 18, 560), 150), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0)

    lines = [
        f"Detected key frame: {label_name}",
        f"Frame index: {frame_index}",
        f"Time: {frame_index / max(fps, 1e-6):.3f} s",
        f"Confidence: {score:.4f}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (34, 52 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    ensure_parent_dir(keyframe_path)
    cv2.imwrite(str(keyframe_path), frame)


def select_process_segment(
    frame_probs: np.ndarray,
    frame_preds: np.ndarray,
    fps: float,
    fall_prob_threshold: float = 0.0,
    fall_margin_threshold: float = 0.0,
    min_fall_duration_sec: float = 0.0,
) -> Optional[dict[str, object]]:
    preferred_labels = [LABEL_NAME_TO_ID["fall"], LABEL_NAME_TO_ID["pre_fall"]]
    candidate_segments: list[dict[str, object]] = []
    min_frames = max(1, int(math.ceil(max(0.0, min_fall_duration_sec) * max(fps, 1e-6))))
    min_peak_score = max(0.45, float(np.clip(fall_prob_threshold, 0.0, 1.0)))

    if len(frame_preds) > 0:
        start = 0
        current_label = int(frame_preds[0])
        for index in range(1, len(frame_preds) + 1):
            next_label = int(frame_preds[index]) if index < len(frame_preds) else -1
            if next_label != current_label:
                if current_label in preferred_labels:
                    mean_score = float(frame_probs[start:index, current_label].mean())
                    peak_score = float(frame_probs[start:index, current_label].max())
                    segment_len = int(index - start)
                    if segment_len >= min_frames and peak_score >= min_peak_score:
                        candidate_segments.append(
                            {
                                "label_id": current_label,
                                "label_name": LABEL_ID_TO_NAME[current_label],
                                "start_frame": int(start),
                                "end_frame": int(index - 1),
                                "num_frames": segment_len,
                                "mean_score": mean_score,
                                "peak_score": peak_score,
                                "start_time_sec": round(start / max(fps, 1e-6), 4),
                                "end_time_sec": round((index - 1) / max(fps, 1e-6), 4),
                            }
                        )
                start = index
                current_label = next_label

    if candidate_segments:
        candidate_segments.sort(
            key=lambda item: (
                1 if item["label_id"] == LABEL_NAME_TO_ID["fall"] else 0,
                item["num_frames"],
                item["peak_score"],
            ),
            reverse=True,
        )
        best_segment = candidate_segments[0]
        if best_segment["num_frames"] <= max(18, int(len(frame_preds) * 0.7)):
            return best_segment

    peak_fall = int(np.argmax(frame_probs[:, LABEL_NAME_TO_ID["fall"]]))
    peak_prefall = int(np.argmax(frame_probs[:, LABEL_NAME_TO_ID["pre_fall"]]))
    fall_score = float(frame_probs[peak_fall, LABEL_NAME_TO_ID["fall"]])
    prefall_score = float(frame_probs[peak_prefall, LABEL_NAME_TO_ID["pre_fall"]])
    fallback_scores = frame_probs[:, [LABEL_NAME_TO_ID["normal"], LABEL_NAME_TO_ID["running"]]].max(axis=1)
    if max(fall_score, prefall_score) < min_peak_score:
        return None
    if fall_score >= prefall_score:
        center = peak_fall
        label_id = LABEL_NAME_TO_ID["fall"]
    else:
        center = peak_prefall
        label_id = LABEL_NAME_TO_ID["pre_fall"]
    if float(frame_probs[center, label_id]) < float(fallback_scores[center]) + max(0.0, fall_margin_threshold):
        return None

    chosen_scores = frame_probs[:, label_id]
    peak_score = float(chosen_scores[center])
    high_threshold = max(0.42, peak_score * 0.72)
    low_threshold = max(0.28, peak_score * 0.48)

    start_frame = center
    while start_frame > 0 and chosen_scores[start_frame - 1] >= high_threshold:
        start_frame -= 1
    while start_frame > 0 and chosen_scores[start_frame - 1] >= low_threshold and center - start_frame < 18:
        start_frame -= 1

    end_frame = center
    while end_frame < len(frame_preds) - 1 and chosen_scores[end_frame + 1] >= high_threshold:
        end_frame += 1
    while end_frame < len(frame_preds) - 1 and chosen_scores[end_frame + 1] >= low_threshold and end_frame - center < 18:
        end_frame += 1

    if end_frame - start_frame + 1 < max(8, min_frames):
        half_width = 6
        start_frame = max(0, center - half_width)
        end_frame = min(len(frame_preds) - 1, center + half_width)
    if end_frame - start_frame + 1 < min_frames:
        return None
    if end_frame - start_frame + 1 > 24:
        half_width = 12
        start_frame = max(0, center - half_width)
        end_frame = min(len(frame_preds) - 1, center + half_width)

    return {
        "label_id": label_id,
        "label_name": LABEL_ID_TO_NAME[label_id],
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "num_frames": int(end_frame - start_frame + 1),
        "mean_score": float(frame_probs[start_frame : end_frame + 1, label_id].mean()),
        "peak_score": float(frame_probs[center, label_id]),
        "start_time_sec": round(start_frame / max(fps, 1e-6), 4),
        "end_time_sec": round(end_frame / max(fps, 1e-6), 4),
    }


def compute_event_motion_series(frames: np.ndarray) -> np.ndarray:
    normalized = np.asarray(frames, dtype=np.float32)
    if normalized.ndim != 3 or len(normalized) == 0:
        return np.zeros((0,), dtype=np.float32)
    motion = np.zeros((len(normalized),), dtype=np.float32)
    if len(normalized) > 1:
        motion[1:] = np.mean(np.abs(np.diff(normalized, axis=0)), axis=(1, 2)).astype(np.float32)
    return motion


def mean_motion_window(motion: np.ndarray, start: int, end: int) -> float:
    start = max(0, min(int(start), len(motion)))
    end = max(start, min(int(end), len(motion)))
    if end <= start:
        return 0.0
    return float(np.mean(motion[start:end]))


def apply_event_shape_filter(
    frames: np.ndarray,
    process_segment: Optional[dict[str, object]],
    fps: float,
    pre_sec: float,
    post_sec: float,
    min_post_sec: float,
    short_end_max_duration_sec: float,
    short_end_min_post_sec: float,
    max_post_event_ratio: float,
    reject_insufficient_post_context: bool,
    strong_peak_score: float,
    strong_mean_score: float,
    start_grace_sec: float,
    boundary_max_duration_sec: float,
) -> tuple[Optional[dict[str, object]], Optional[dict[str, object]]]:
    if process_segment is None:
        return None, None

    motion = compute_event_motion_series(frames)
    if len(motion) == 0:
        return process_segment, {
            "enabled": True,
            "accepted": True,
            "reject_reason": None,
            "note": "No motion series was available; original event was kept.",
        }

    fps_safe = max(float(fps), 1e-6)
    start_frame = int(process_segment["start_frame"])
    end_frame = int(process_segment["end_frame"])
    start_sec = float(process_segment["start_time_sec"])
    end_sec = float(process_segment["end_time_sec"])
    duration_sec = max(0.0, end_sec - start_sec)
    pre_frames = max(1, int(round(max(0.0, pre_sec) * fps_safe)))
    post_frames = max(1, int(round(max(0.0, post_sec) * fps_safe)))
    pre_start = max(0, start_frame - pre_frames)
    post_start = min(len(motion), end_frame + 1)
    post_end = min(len(motion), end_frame + 1 + post_frames)

    event_motion = mean_motion_window(motion, start_frame + 1, end_frame + 1)
    pre_motion = mean_motion_window(motion, pre_start, start_frame)
    post_motion = mean_motion_window(motion, post_start, post_end)
    post_available_sec = max(0.0, (post_end - post_start) / fps_safe)
    post_event_ratio = post_motion / max(event_motion, 1e-6)
    still_threshold = max(float(np.percentile(motion, 25)) * 1.20, 0.003)
    post_values = motion[post_start:post_end]
    post_still_ratio = float(np.mean(post_values <= still_threshold)) if len(post_values) else 0.0
    near_start = start_sec <= max(0.0, float(start_grace_sec))
    near_end = post_available_sec < max(0.0, float(min_post_sec))
    peak_score = float(process_segment.get("peak_score", 0.0) or 0.0)
    mean_score = float(process_segment.get("mean_score", 0.0) or 0.0)
    strong_candidate = (
        peak_score >= max(0.0, float(strong_peak_score))
        and mean_score >= max(0.0, float(strong_mean_score))
    )

    reject_reason = None
    filter_note = None
    short_end_no_post_context = (
        post_available_sec < max(0.0, float(short_end_min_post_sec))
        and duration_sec <= max(0.0, float(short_end_max_duration_sec))
    )
    if short_end_no_post_context and not strong_candidate:
        reject_reason = "short_end_no_post_context"
    elif (
        post_available_sec < max(0.0, float(min_post_sec))
        and bool(reject_insufficient_post_context)
        and not strong_candidate
    ):
        reject_reason = "insufficient_post_fall_context"
    elif post_available_sec < max(0.0, float(min_post_sec)):
        filter_note = "insufficient_post_context_kept"
    elif post_event_ratio > max(0.0, float(max_post_event_ratio)) and not strong_candidate:
        reject_reason = "no_post_fall_motion_drop"
    elif near_start and duration_sec <= max(0.0, float(boundary_max_duration_sec)) and not strong_candidate:
        reject_reason = "boundary_start_short_event"
    elif post_event_ratio > max(0.0, float(max_post_event_ratio)) and strong_candidate:
        filter_note = "high_confidence_event_kept"

    result = {
        "enabled": True,
        "accepted": reject_reason is None,
        "reject_reason": reject_reason,
        "note": filter_note,
        "pre_motion": pre_motion,
        "event_motion": event_motion,
        "post_motion": post_motion,
        "post_available_sec": post_available_sec,
        "post_event_ratio": post_event_ratio,
        "post_still_ratio": post_still_ratio,
        "near_start": near_start,
        "near_end": near_end,
        "short_end_no_post_context": short_end_no_post_context,
        "peak_score": peak_score,
        "mean_score": mean_score,
        "strong_candidate": strong_candidate,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_sec,
        "end_time_sec": end_sec,
        "duration_sec": duration_sec,
        "params": {
            "pre_sec": float(pre_sec),
            "post_sec": float(post_sec),
            "min_post_sec": float(min_post_sec),
            "short_end_max_duration_sec": float(short_end_max_duration_sec),
            "short_end_min_post_sec": float(short_end_min_post_sec),
            "max_post_event_ratio": float(max_post_event_ratio),
            "reject_insufficient_post_context": bool(reject_insufficient_post_context),
            "strong_peak_score": float(strong_peak_score),
            "strong_mean_score": float(strong_mean_score),
            "start_grace_sec": float(start_grace_sec),
            "boundary_max_duration_sec": float(boundary_max_duration_sec),
        },
    }
    if reject_reason is None:
        filtered_segment = dict(process_segment)
        filtered_segment["event_shape_filter"] = result
        return filtered_segment, result

    result["rejected_process_segment"] = dict(process_segment)
    return None, result


def save_process_frames(
    video_path: Path,
    process_dir: Path,
    overview_path: Path,
    process_segment: Optional[dict[str, object]],
    frame_probs: np.ndarray,
    fps: float,
    max_overview_frames: int = 6,
) -> list[dict[str, object]]:
    process_dir.mkdir(parents=True, exist_ok=True)
    for old_file in process_dir.glob("*.png"):
        old_file.unlink()
    if process_segment is None:
        if overview_path.exists():
            overview_path.unlink()
        return []

    start_frame = int(process_segment["start_frame"])
    end_frame = int(process_segment["end_frame"])
    label_name = str(process_segment["label_name"])
    label_id = int(process_segment["label_id"])
    border_color = {
        "normal": (80, 220, 80),
        "running": (255, 180, 0),
        "pre_fall": (0, 190, 255),
        "fall": (40, 80, 255),
    }[label_name]

    saved_rows: list[dict[str, object]] = []
    overview_images = []
    selected_indices = np.linspace(start_frame, end_frame, num=min(max_overview_frames, end_frame - start_frame + 1))
    selected_indices = {int(round(value)) for value in selected_indices.tolist()}

    for frame_index in range(start_frame, end_frame + 1):
        frame = read_original_video_frame(video_path, frame_index)
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (16, 16), (width - 16, height - 16), border_color, thickness=5)
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (min(width - 16, 600), 140), (0, 0, 0), thickness=-1)
        frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

        label_score = float(frame_probs[frame_index, label_id])
        lines = [
            f"Detected process: {label_name}",
            f"Frame {frame_index} | Time {frame_index / max(fps, 1e-6):.3f}s",
            f"{label_name} probability: {label_score:.4f}",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (30, 48 + idx * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out_path = process_dir / f"process_frame_{frame_index:04d}.png"
        ensure_parent_dir(out_path)
        cv2.imwrite(str(out_path), frame)
        saved_rows.append(
            {
                "frame_index": int(frame_index),
                "time_sec": round(frame_index / max(fps, 1e-6), 4),
                "label_name": label_name,
                "score": round(label_score, 6),
                "output_path": str(out_path),
            }
        )
        if frame_index in selected_indices:
            overview_images.append((frame_index, frame.copy()))

    if overview_images:
        # Paper-style overview: prefer a compact 2-row grid instead of a long strip.
        if len(overview_images) <= 4:
            cols = 2
        else:
            cols = 3
        rows = int(math.ceil(len(overview_images) / cols))
        figure, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows))
        axes_array = np.atleast_1d(axes).reshape(rows, cols)
        for axis in axes_array.flatten():
            axis.axis("off")
        for axis, (frame_index, image_bgr) in zip(axes_array.flatten(), overview_images):
            axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            axis.set_title(f"frame {frame_index} | {frame_index / max(fps, 1e-6):.2f}s", fontsize=10)
            axis.axis("off")
        figure.tight_layout()
        ensure_parent_dir(overview_path)
        figure.savefig(overview_path, dpi=180)
        plt.close(figure)

    return saved_rows


def write_report_json(
    report_json: Path,
    args: argparse.Namespace,
    config: dict[str, object],
    train_artifacts: Optional[TrainArtifacts],
    video_path: Path,
    labels_csv: Optional[Path],
    inference: dict[str, object],
    frame_metrics: Optional[dict[str, object]],
    key_frame: Optional[dict[str, object]],
    process_segment: Optional[dict[str, object]],
    process_frames: Optional[list[dict[str, object]]],
    event_shape_filter: Optional[dict[str, object]] = None,
    evaluation_split: str = "inference",
    no_fall_metrics: Optional[dict[str, object]] = None,
) -> None:
    frame_preds = inference["frame_predictions"]
    frame_probs = inference["frame_probabilities"]
    report = {
        "evaluation_split": evaluation_split,
        "video_path": str(video_path),
        "labels_csv": str(labels_csv) if labels_csv is not None else None,
        "model_path": str(args.model_path),
        "window_size": int(config.get("window_size", args.window_size)),
        "image_size": int(config.get("image_size", args.image_size)),
        "infer_stride": int(config.get("infer_stride", args.infer_stride)),
        "use_crossbar": bool(config.get("use_crossbar", not args.disable_crossbar)),
        "use_device_dynamics": bool(config.get("use_device_dynamics", args.use_device_dynamics)),
        "device_dynamics_sources": config.get("device_dynamics_sources", {}),
        "crossbar_readout_noise_scale": float(config.get("crossbar_readout_noise_scale", args.crossbar_readout_noise_scale)),
        "use_pixel_human": bool(config.get("use_pixel_human", args.use_pixel_human)),
        "use_silhouette_human": bool(config.get("use_silhouette_human", args.use_silhouette_human)),
        "pixel_grid_size": int(config.get("pixel_grid_size", args.pixel_grid_size)),
        "prediction_counts": class_count_dict(frame_preds),
        "max_probability_frames": {
            LABEL_ID_TO_NAME[class_id]: int(np.argmax(frame_probs[:, class_id]))
            for class_id in LABEL_ID_TO_NAME
        },
        "key_frame": key_frame,
        "process_segment": process_segment,
        "event_shape_filter": event_shape_filter,
        "process_frames": process_frames,
        "segments": inference["segments"],
        "frame_metrics": frame_metrics,
        "no_fall_metrics": no_fall_metrics,
    }
    if train_artifacts is not None:
        report["training"] = {
            "history": train_artifacts.history,
            "train_counts": train_artifacts.train_counts,
            "val_counts": train_artifacts.val_counts,
            "aux_summary": train_artifacts.aux_summary,
            "running_summary": train_artifacts.running_summary,
            "external_labeled_summary": train_artifacts.external_labeled_summary,
            "external_val_labeled_summary": train_artifacts.external_val_labeled_summary,
            "external_no_fall_summary": train_artifacts.external_no_fall_summary,
        }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def print_inference_summary(
    inference: dict[str, object],
    frame_metrics: Optional[dict[str, object]],
    key_frame: Optional[dict[str, object]],
    process_segment: Optional[dict[str, object]],
    event_shape_filter: Optional[dict[str, object]] = None,
) -> None:
    counts = class_count_dict(inference["frame_predictions"])
    log(f"Prediction counts: {counts}")
    if frame_metrics is not None:
        log(f"Frame accuracy against current labels: {frame_metrics['frame_accuracy']:.4f}")
        class_accuracy = frame_metrics.get("class_accuracy", {})
        log(f"Class accuracy: {class_accuracy}")

    segments = inference.get("segments", [])
    if segments:
        top_segments = ", ".join(
            f"{segment['label_name']}[{segment['start_time_sec']:.2f}s-{segment['end_time_sec']:.2f}s]"
            for segment in segments[:6]
        )
        log(f"Predicted segments: {top_segments}")
    if key_frame is not None:
        log(
            f"Key frame: index={key_frame['frame_index']} "
            f"label={key_frame['label_name']} score={key_frame['score']:.4f}"
        )
    if process_segment is not None:
        log(
            f"Detected process: {process_segment['label_name']} "
            f"[{process_segment['start_time_sec']:.2f}s-{process_segment['end_time_sec']:.2f}s] "
            f"frames={process_segment['start_frame']}-{process_segment['end_frame']}"
        )
    else:
        log("Detected process: none (danger confidence below threshold)")
        if event_shape_filter is not None and event_shape_filter.get("accepted") is False:
            log(f"Event shape filter rejected process: {event_shape_filter.get('reject_reason')}")


def summarize_no_fall_assumption(inference: dict[str, object]) -> dict[str, object]:
    frame_preds = np.asarray(inference["frame_predictions"], dtype=np.int64)
    positive_mask = np.isin(frame_preds, [LABEL_NAME_TO_ID["pre_fall"], LABEL_NAME_TO_ID["fall"]])
    positive_segments = [
        segment for segment in inference.get("segments", [])
        if int(segment.get("label_id", -1)) in {LABEL_NAME_TO_ID["pre_fall"], LABEL_NAME_TO_ID["fall"]}
    ]
    fall_probs = np.asarray(inference["frame_probabilities"], dtype=np.float32)[:, LABEL_NAME_TO_ID["fall"]]
    return {
        "assumption": "no_fall_adl",
        "total_frames": int(len(frame_preds)),
        "predicted_positive_frames": int(positive_mask.sum()),
        "predicted_positive_ratio": float(positive_mask.mean()) if len(positive_mask) > 0 else 0.0,
        "predicted_positive_segments": int(len(positive_segments)),
        "max_fall_probability": float(fall_probs.max()) if len(fall_probs) > 0 else 0.0,
        "mean_fall_probability": float(fall_probs.mean()) if len(fall_probs) > 0 else 0.0,
    }


def build_video_level_labels_from_name(video_path: Path, frame_count: int, fall_name_pattern: str) -> np.ndarray:
    label_id = infer_video_level_label_id(video_path, fall_name_pattern)
    return np.full(frame_count, label_id, dtype=np.int64)


def list_test_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Test video directory not found: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for --test-video-dir, got: {video_dir}")
    suffixes = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}
    videos = [path for path in sorted(video_dir.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]
    if not videos:
        raise FileNotFoundError(f"No supported video files found in {video_dir}")
    return videos


def run_detection_inference(
    args: argparse.Namespace,
    device: torch.device,
    train_artifacts: Optional[TrainArtifacts],
    video_path: Path,
    labels_csv: Optional[Path],
    report_json: Path,
    plot_path: Path,
    keyframe_path: Path,
    process_dir: Path,
    process_overview_path: Path,
    evaluation_split: str,
) -> dict[str, object]:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    model, config = load_model_checkpoint(args.model_path, device)
    window_size = int(config.get("window_size", args.window_size))
    image_size = int(config.get("image_size", args.image_size))
    infer_stride = int(config.get("infer_stride", args.infer_stride))
    use_crossbar = bool(config.get("use_crossbar", not args.disable_crossbar))
    use_device_dynamics = bool(config.get("use_device_dynamics", args.use_device_dynamics))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", args.crossbar_readout_noise_scale))
    use_pixel_human = bool(config.get("use_pixel_human", args.use_pixel_human))
    use_silhouette_human = bool(config.get("use_silhouette_human", args.use_silhouette_human))
    pixel_grid_size = int(config.get("pixel_grid_size", args.pixel_grid_size))
    hrs_values, lrs_values = load_conductance_pair(args.data_dir) if use_crossbar else (
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    device_dynamics = load_device_dynamics(args.data_dir) if (
        use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
    ) else None

    frames, fps = read_video_frames(
        video_path,
        image_size=image_size,
        use_pixel_human=use_pixel_human,
        use_silhouette_human=use_silhouette_human,
        pixel_grid_size=pixel_grid_size,
    )
    inference = predict_video_frames(
        model=model,
        frames=frames,
        fps=fps,
        window_size=window_size,
        stride=infer_stride,
        use_crossbar=use_crossbar,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=crossbar_readout_noise_scale,
        device=device,
        batch_size=max(1, args.batch_size),
        fall_prob_threshold=args.fall_prob_threshold,
        fall_margin_threshold=args.fall_margin_threshold,
        min_fall_duration_sec=args.min_fall_duration_sec,
    )

    true_labels = None
    if labels_csv is not None:
        true_labels = load_frame_labels_csv(labels_csv, expected_video_name=video_path.name)
    if true_labels is None:
        true_labels = load_gmdcsa_video_labels(video_path, frame_count=int(inference["num_frames"]), fps=fps)
    if true_labels is None and args.test_label_from_name:
        true_labels = build_video_level_labels_from_name(
            video_path,
            frame_count=int(inference["num_frames"]),
            fall_name_pattern=args.external_fall_name_pattern,
        )
    frame_metrics = compute_frame_metrics(inference["frame_predictions"], true_labels)
    no_fall_metrics = summarize_no_fall_assumption(inference) if args.assume_test_videos_no_fall else None
    key_frame = select_key_frame(inference["frame_probabilities"])
    key_frame["time_sec"] = round(float(key_frame["frame_index"]) / fps, 4)
    if not args.skip_visual_outputs:
        save_key_frame_visual(video_path, keyframe_path, key_frame, fps)
    process_segment = select_process_segment(
        inference["frame_probabilities"],
        inference["frame_predictions"],
        fps,
        fall_prob_threshold=args.fall_prob_threshold,
        fall_margin_threshold=args.fall_margin_threshold,
        min_fall_duration_sec=args.min_fall_duration_sec,
    )
    event_shape_filter = None
    if args.use_event_shape_filter:
        process_segment, event_shape_filter = apply_event_shape_filter(
            frames,
            process_segment,
            fps,
            pre_sec=args.event_shape_pre_sec,
            post_sec=args.event_shape_post_sec,
            min_post_sec=args.event_shape_min_post_sec,
            short_end_max_duration_sec=args.event_shape_short_end_max_duration_sec,
            short_end_min_post_sec=args.event_shape_short_end_min_post_sec,
            max_post_event_ratio=args.event_shape_max_post_event_ratio,
            reject_insufficient_post_context=args.event_shape_reject_insufficient_post_context,
            strong_peak_score=args.event_shape_strong_peak_score,
            strong_mean_score=args.event_shape_strong_mean_score,
            start_grace_sec=args.event_shape_start_grace_sec,
            boundary_max_duration_sec=args.event_shape_boundary_max_duration_sec,
        )
    if args.skip_visual_outputs:
        process_frames = []
    else:
        process_frames = save_process_frames(
            video_path,
            process_dir,
            process_overview_path,
            process_segment,
            inference["frame_probabilities"],
            fps,
        )
        save_detection_plot(
            plot_path=plot_path,
            fps=fps,
            frame_probs=inference["frame_probabilities"],
            frame_preds=inference["frame_predictions"],
            true_labels=true_labels,
            show_plot=args.show_plot,
            save_plot=args.save_plot or not args.show_plot,
        )
    write_report_json(
        report_json,
        args,
        config,
        train_artifacts,
        video_path,
        labels_csv,
        inference,
        frame_metrics,
        key_frame,
        process_segment,
        process_frames,
        event_shape_filter=event_shape_filter,
        evaluation_split=evaluation_split,
        no_fall_metrics=no_fall_metrics,
    )
    print_inference_summary(inference, frame_metrics, key_frame, process_segment, event_shape_filter)
    log(f"Saved detection report to {report_json}")
    if not args.skip_visual_outputs:
        log(f"Saved detection plot to {plot_path}")
        log(f"Saved key frame visualization to {keyframe_path}")
    if process_segment is not None and not args.skip_visual_outputs:
        log(f"Saved process frames to {process_dir}")
        log(f"Saved process overview to {process_overview_path}")
    return {
        "evaluation_split": evaluation_split,
        "video_path": str(video_path),
        "labels_csv": str(labels_csv) if labels_csv is not None else None,
        "report_json": str(report_json),
        "plot_path": str(plot_path),
        "keyframe_path": str(keyframe_path),
        "process_overview_path": str(process_overview_path),
        "prediction_counts": class_count_dict(inference["frame_predictions"]),
        "frame_metrics": frame_metrics,
        "no_fall_metrics": no_fall_metrics,
        "key_frame": key_frame,
        "process_segment": process_segment,
        "event_shape_filter": event_shape_filter,
        "num_frames": int(inference["num_frames"]),
        "fps": float(inference["fps"]),
    }


def run_batch_test_directory(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.test_video_dir is None:
        raise ValueError("Mode 'batch_test' requires --test-video-dir.")
    videos = list_test_videos(args.test_video_dir)
    args.test_output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for video_path in videos:
        try:
            relative_stem = video_path.relative_to(args.test_video_dir).with_suffix("")
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(relative_stem.parts)).strip("_")
        except ValueError:
            stem = video_path.stem
        video_output_dir = args.test_output_dir / stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Running batch test for {video_path.name}")
        row = run_detection_inference(
            args=args,
            device=device,
            train_artifacts=None,
            video_path=video_path,
            labels_csv=None,
            report_json=video_output_dir / f"{stem}_report.json",
            plot_path=video_output_dir / f"{stem}_plot.png",
            keyframe_path=video_output_dir / f"{stem}_key_frame.png",
            process_dir=video_output_dir / "process_frames",
            process_overview_path=video_output_dir / f"{stem}_process_overview.png",
            evaluation_split="independent_test_batch",
        )
        summary_rows.append(row)

    videos_with_detected_process = sum(1 for row in summary_rows if row.get("process_segment") is not None)
    mean_frame_accuracy = None
    metric_rows = [row["frame_metrics"] for row in summary_rows if row.get("frame_metrics")]
    if metric_rows:
        mean_frame_accuracy = float(sum(float(row["frame_accuracy"]) for row in metric_rows) / len(metric_rows))
    mean_positive_ratio = None
    if args.assume_test_videos_no_fall and summary_rows:
        ratios = [float(row["no_fall_metrics"]["predicted_positive_ratio"]) for row in summary_rows if row.get("no_fall_metrics")]
        if ratios:
            mean_positive_ratio = float(sum(ratios) / len(ratios))
    summary = {
        "test_video_dir": str(args.test_video_dir),
        "num_videos": len(summary_rows),
        "videos_with_detected_process": int(videos_with_detected_process),
        "videos_without_detected_process": int(len(summary_rows) - videos_with_detected_process),
        "assume_test_videos_no_fall": bool(args.assume_test_videos_no_fall),
        "test_label_from_name": bool(args.test_label_from_name),
        "mean_frame_accuracy": mean_frame_accuracy,
        "mean_predicted_positive_ratio": mean_positive_ratio,
        "results": summary_rows,
    }
    args.test_batch_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved batch test summary to {args.test_batch_summary_json}")
    log(
        f"Batch test finished: total={len(summary_rows)} "
        f"with_detected_process={videos_with_detected_process}"
    )


def run_plasticity_analysis(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    model, config = load_model_checkpoint(args.model_path, device)
    window_size = int(config.get("window_size", args.window_size))
    image_size = int(config.get("image_size", args.image_size))
    infer_stride = int(config.get("infer_stride", args.infer_stride))
    use_crossbar = bool(config.get("use_crossbar", not args.disable_crossbar))
    use_device_dynamics = bool(config.get("use_device_dynamics", args.use_device_dynamics))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", args.crossbar_readout_noise_scale))
    use_pixel_human = bool(config.get("use_pixel_human", args.use_pixel_human))
    use_silhouette_human = bool(config.get("use_silhouette_human", args.use_silhouette_human))
    pixel_grid_size = int(config.get("pixel_grid_size", args.pixel_grid_size))
    hrs_values, lrs_values = load_conductance_pair(args.data_dir) if use_crossbar else (
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    device_dynamics = load_device_dynamics(args.data_dir) if (
        use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
    ) else None

    frames, fps = read_video_frames(
        args.video,
        image_size=image_size,
        use_pixel_human=use_pixel_human,
        use_silhouette_human=use_silhouette_human,
        pixel_grid_size=pixel_grid_size,
    )
    inference = predict_video_frames(
        model=model,
        frames=frames,
        fps=fps,
        window_size=window_size,
        stride=infer_stride,
        use_crossbar=use_crossbar,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=crossbar_readout_noise_scale,
        device=device,
        batch_size=max(1, args.batch_size),
    )

    true_labels = load_frame_labels_csv(args.labels_csv, expected_video_name=args.video.name)
    if true_labels is None or len(true_labels) != len(frames):
        positive_mask = np.isin(inference["frame_predictions"], [LABEL_NAME_TO_ID["pre_fall"], LABEL_NAME_TO_ID["fall"]])
        display_label_ids = inference["frame_predictions"].astype(np.int64)
    else:
        positive_mask = np.isin(true_labels, [LABEL_NAME_TO_ID["pre_fall"], LABEL_NAME_TO_ID["fall"]])
        display_label_ids = true_labels.astype(np.int64)

    frame_probs = inference["frame_probabilities"]
    running_score = frame_probs[:, LABEL_NAME_TO_ID["running"]].astype(np.float32)
    danger_score = build_fall_specific_score(
        pre_fall_score=frame_probs[:, LABEL_NAME_TO_ID["pre_fall"]].astype(np.float32),
        fall_score=frame_probs[:, LABEL_NAME_TO_ID["fall"]].astype(np.float32),
        running_score=running_score,
        suppression=args.running_suppression,
    )

    ltp_payload = load_ltp_pulse_curve(args.ltp_file)
    ltp_gain = resample_curve_to_frames(ltp_payload["normalized_gain"], len(frames))
    threshold_curve = build_ltp_threshold_curve(danger_score, ltp_gain)
    stdp_payload = build_stdp_gain(
        motion_energy=compute_motion_energy(frames),
        danger_score=danger_score,
        a_plus=args.stdp_a_plus,
        a_minus=args.stdp_a_minus,
        tau_plus=args.stdp_tau_plus,
        tau_minus=args.stdp_tau_minus,
    )

    if args.plasticity_method == "compare_all":
        chosen_methods = ["ltp", "stdp_ltp", "stdp_ltp_decay"]
    elif args.plasticity_method == "compare_ltp_decay":
        chosen_methods = ["ltp", "ltp_decay"]
    else:
        chosen_methods = [args.plasticity_method]

    method_results: list[dict[str, object]] = []
    for method_name in chosen_methods:
        stdp_input = stdp_payload if method_name in {"stdp_ltp", "stdp_ltp_decay"} else None
        result = analyze_plasticity_method(
                method_name=method_name,
                danger_score=danger_score,
                running_score=running_score,
                frames=frames,
                ltp_gain=ltp_gain,
                threshold_curve=threshold_curve,
                positive_mask=positive_mask.astype(bool),
                stdp_payload=stdp_input,
                decay_tau=args.decay_tau,
                decay_floor=args.decay_floor,
                decay_event_percentile=args.decay_event_percentile,
                running_gate_margin=args.running_gate_margin,
            )
        method_results.append(result)
        save_individual_plasticity_plot(
            plot_path=build_method_output_path(args.plasticity_plot_path, method_name),
            video_path=args.video,
            fps=fps,
            ltp_payload=ltp_payload,
            ltp_gain=ltp_gain,
            positive_mask=positive_mask.astype(bool),
            result=result,
        )
        save_plasticity_method_frames(
            plot_path=build_method_output_path(args.plasticity_overview_path, method_name),
            video_path=args.video,
            picture_dir=args.picture_dir,
            fps=fps,
            frame_label_ids=display_label_ids,
            result=result,
        )

    save_plasticity_analysis_plot(
        plot_path=args.plasticity_plot_path,
        fps=fps,
        ltp_payload=ltp_payload,
        method_results=method_results,
        positive_mask=positive_mask.astype(bool),
        true_labels=true_labels,
    )
    save_plasticity_tracking_overview(
        video_path=args.video,
        overview_path=args.plasticity_overview_path,
        picture_dir=args.picture_dir,
        fps=fps,
        frame_label_ids=display_label_ids,
        method_results=method_results,
    )
    write_plasticity_report_json(
        report_json=args.plasticity_report_json,
        args=args,
        fps=fps,
        ltp_payload=ltp_payload,
        stdp_payload=stdp_payload,
        method_results=method_results,
    )
    write_next_stage_plan_document(args.next_stage_plan_path)

    log(f"Saved plasticity analysis plot to {args.plasticity_plot_path}")
    log(f"Saved plasticity tracking overview to {args.plasticity_overview_path}")
    log(f"Saved plasticity analysis report to {args.plasticity_report_json}")
    log(f"Saved next-stage roadmap to {args.next_stage_plan_path}")
    for result in method_results:
        method_name = str(result["method_name"])
        log(f"Saved {method_name} plot to {build_method_output_path(args.plasticity_plot_path, method_name)}")
        log(f"Saved {method_name} frames to {build_method_output_path(args.plasticity_overview_path, method_name)}")
    print_plasticity_summary(method_results)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    log(f"Using device: {device}")

    if args.mode == "plasticity_analysis":
        run_plasticity_analysis(args, device)
        return
    if args.mode == "batch_test":
        run_batch_test_directory(args, device)
        return

    train_artifacts: Optional[TrainArtifacts] = None
    if args.mode in {"train", "train_infer"}:
        train_artifacts = train_detector(args, device)
        save_training_plot(train_artifacts.history, args.training_plot_path)
        log(f"Saved detector checkpoint to {train_artifacts.checkpoint_path}")
        log(f"Saved training curve to {args.training_plot_path}")
        log(f"Train counts: {train_artifacts.train_counts}")
        log(f"Val counts: {train_artifacts.val_counts}")

    if args.mode in {"infer", "train_infer"}:
        inference_video = args.video
        inference_labels = args.labels_csv
        inference_report = args.report_json
        inference_plot = args.plot_path
        inference_keyframe = args.keyframe_path
        inference_process_dir = args.process_dir
        inference_process_overview = args.process_overview_path
        evaluation_split = "validation_video"

        if args.mode == "train_infer" and args.test_video is not None:
            inference_video = args.test_video
            inference_labels = args.test_labels_csv
            inference_report = args.test_report_json
            inference_plot = args.test_plot_path
            inference_keyframe = args.test_keyframe_path
            inference_process_dir = args.test_process_dir
            inference_process_overview = args.test_process_overview_path
            evaluation_split = "independent_test"

        run_detection_inference(
            args=args,
            device=device,
            train_artifacts=train_artifacts,
            video_path=inference_video,
            labels_csv=inference_labels,
            report_json=inference_report,
            plot_path=inference_plot,
            keyframe_path=inference_keyframe,
            process_dir=inference_process_dir,
            process_overview_path=inference_process_overview,
            evaluation_split=evaluation_split,
        )

    if args.mode == "test":
        if args.test_video is None:
            raise ValueError("Mode 'test' requires --test-video.")
        run_detection_inference(
            args=args,
            device=device,
            train_artifacts=None,
            video_path=args.test_video,
            labels_csv=args.test_labels_csv,
            report_json=args.test_report_json,
            plot_path=args.test_plot_path,
            keyframe_path=args.test_keyframe_path,
            process_dir=args.test_process_dir,
            process_overview_path=args.test_process_overview_path,
            evaluation_split="independent_test",
        )


if __name__ == "__main__":
    main()
